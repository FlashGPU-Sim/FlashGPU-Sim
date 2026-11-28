#include "../test_support.h"
namespace {
frontend bulk_image(bool load, const char *descriptor) {
  kernel image;
  image.name = "bulk_descriptor";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(
      0, load ? "UBLKCP.S.G" : "UBLKCP.G.S",
      std::string(load ? "[UR4][UR10],UR6" : "[UR10][UR4],UR6") + descriptor)};
  frontend f;
  f.add_kernel(std::move(image));
  register_sm100_functional_semantics(f);
  return f;
}
TEST(SassBulkDescriptor, QualifiedGlobalPayloadAndCompletion) {
  for (bool load : {false, true})
    for (uint64_t qualifier : {0x1000000000000000ull, 0x12f0000000000000ull,
                               0x14f0000000000000ull, 0ull}) {
      auto f = bulk_image(load, qualifier ? ",desc[UR8]" : ",desc[URZ]");
      mapped_functional_memory memory;
      memory.map(memory_space::kGlobal, 0x100000100ull, 64, true);
      memory.map(memory_space::kShared, 0x100, 0x200, true);
      for (unsigned i = 0; i < 16; ++i)
        memory.initialize<uint32_t>(
            load ? memory_space::kGlobal : memory_space::kShared,
            (load ? 0x100000100ull : 0x200ull) + 4 * i, 100 + i);
      cta_execution_state cta(1);
      ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
      ASSERT_TRUE(cta.arrive_mbarrier(0x100, 1, 64));
      warp_state state;
      state.active_mask = 1u << 7;
      state.write_uniform_register(4, 0x200);
      state.write_uniform_register(5, 0x100);
      state.write_uniform_register(6, 4);
      state.write_uniform_register(8, qualifier);
      state.write_uniform_register(9, qualifier >> 32);
      state.write_uniform_register(10, 0x100);
      state.write_uniform_register(11, 1);
      execution_context context{&memory};
      context.cta = &cta;
      auto result = f.step("bulk_descriptor", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      for (unsigned i = 0; i < 16; ++i)
        EXPECT_EQ(memory.inspect<uint32_t>(
                      load ? memory_space::kShared : memory_space::kGlobal,
                      (load ? 0x200ull : 0x100000100ull) + 4 * i),
                  100 + i);
      EXPECT_EQ(cta.mbarrier_phase(0x100), load ? 1u : 0u);
      EXPECT_TRUE(context.tma_effects[7].valid);
      EXPECT_EQ(context.tma_effects[7].size_bytes, 64u);
    }
}
TEST(SassBulkDescriptor, InvalidQualifierDoesNotCopyOrComplete) {
  for (const char *descriptor : {",desc[UR8]", ",UR8"}) {
    auto f = bulk_image(true, descriptor);
    mapped_functional_memory memory;
    memory.map(memory_space::kShared, 0x100, 0x200, true);
    memory.initialize<uint32_t>(memory_space::kShared, 0x200, 99);
    cta_execution_state cta(1);
    ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
    ASSERT_TRUE(cta.arrive_mbarrier(0x100, 1, 64));
    warp_state state;
    state.active_mask = 1;
    state.write_uniform_register(4, 0x200);
    state.write_uniform_register(5, 0x100);
    state.write_uniform_register(6, 4);
    state.write_uniform_register(8, 1);
    execution_context context{&memory};
    context.cta = &cta;
    EXPECT_EQ(f.step("bulk_descriptor", state, context).status,
              step_status::kUnsupported);
    EXPECT_EQ(state.pc, 0u);
    EXPECT_EQ(cta.mbarrier_phase(0x100), 0u);
    EXPECT_FALSE(context.tma_effects[0].valid);
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared, 0x200), 99u);
  }
}
TEST(SassBulkDescriptor, TimingRetainsImplicitPairsAndSize) {
  for (bool load : {false, true})
    for (bool hint : {false, true}) {
      auto f = bulk_image(load, hint ? ",desc[UR8]" : "");
      const timing_instruction timing(*f.fetch("bulk_descriptor", 0));
      EXPECT_EQ(timing.incount, (load ? 5u : 4u) + (hint ? 2u : 0u));
      EXPECT_EQ(timing.outcount, 0u);
      EXPECT_EQ(timing.op, TENSOR_MEMORY_ACCELERATOR_OP);
    }
}
TEST(SassBulkDescriptor, ClusterPayloadAndBarrierResolveIndependentOwners) {
  for (unsigned issuer : {0u, 1u})
    for (unsigned destination : {0u, 1u})
      for (unsigned owner : {0u, 1u}) {
        SCOPED_TRACE(issuer * 4 + destination * 2 + owner);
        auto f = bulk_image(true, ",desc[UR8]");
        mapped_functional_memory memories[2];
        cta_execution_state ctas[2] = {cta_execution_state(1), cta_execution_state(1)};
        flash_gpgpu_sim::sass::functional_cluster_state cluster;
        for (unsigned rank = 0; rank < 2; ++rank) {
          memories[rank].map(memory_space::kGlobal, 0x100000100ull, 64, true);
          memories[rank].map(memory_space::kShared, 0x100, 0x200, true);
          for (unsigned i = 0; i < 16; ++i) {
            memories[rank].initialize<uint32_t>(memory_space::kGlobal,
                                                0x100000100ull + 4 * i, 100 + i);
            memories[rank].initialize<uint32_t>(memory_space::kShared,
                                                0x200 + 4 * i, 999);
          }
          ASSERT_TRUE(ctas[rank].initialize_mbarrier(0x100, 1));
          ASSERT_TRUE(ctas[rank].arrive_mbarrier(0x100, 1, 64));
          cluster.ctas.push_back(&ctas[rank]);
          cluster.memories.push_back(&memories[rank]);
        }
        warp_state state;
        state.active_mask = 1u << 7;
        state.write_uniform_register(4, (destination << 24) | 0x200);
        state.write_uniform_register(5, (owner << 24) | 0x100);
        state.write_uniform_register(6, 4);
        state.write_uniform_register(8, 0);
        state.write_uniform_register(9, 0x10000000);
        state.write_uniform_register(10, 0x100);
        state.write_uniform_register(11, 1);
        execution_context context{&memories[issuer]};
        context.cta = &ctas[issuer];
        context.cluster = &cluster;
        context.cga_cta_id = issuer;
        const auto result = f.step("bulk_descriptor", state, context);
        ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
        for (unsigned rank = 0; rank < 2; ++rank) {
          EXPECT_EQ(ctas[rank].mbarrier_phase(0x100), rank == owner ? 1u : 0u);
          EXPECT_FALSE(ctas[rank].mbarrier_initialized(0x1000100));
          for (unsigned i = 0; i < 16; ++i)
            EXPECT_EQ(memories[rank].inspect<uint32_t>(memory_space::kShared,
                                                       0x200 + 4 * i),
                      rank == destination ? 100 + i : 999);
        }
        const auto &effect = context.tma_effects[7];
        EXPECT_TRUE(effect.valid);
        EXPECT_TRUE(effect.explicit_cluster_targets);
        EXPECT_EQ(effect.destination_address, (destination << 24) | 0x200u);
        EXPECT_EQ(effect.mbarrier_address, 0x100u);
        EXPECT_EQ(effect.destination_mask, 1u << destination);
        EXPECT_EQ(effect.completion_mask, 1u << owner);
      }
}
}  // namespace
