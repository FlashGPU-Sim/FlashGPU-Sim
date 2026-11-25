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
}  // namespace
