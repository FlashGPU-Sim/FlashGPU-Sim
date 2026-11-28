#include "../test_support.h"
#include "gpgpu-sim/flash/sass/functional/tmem.h"
namespace {
using flash_gpgpu_sim::sass::functional_cluster_state;
using flash_gpgpu_sim::sass::functional_tcgen_state;

struct PairFixture {
  mapped_functional_memory memory[4];
  cta_execution_state cta[4] = {cta_execution_state(4), cta_execution_state(4),
                                cta_execution_state(4), cta_execution_state(4)};
  functional_cluster_state cluster;
  execution_context context;
  frontend sass;
  warp_state state;
  PairFixture(const char *op, const char *operands, unsigned rank = 0) {
    kernel image;
    image.name = "pair";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(0, op, operands)};
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    for (unsigned i = 0; i < 4; ++i) {
      memory[i].map(memory_space::kShared, 0, 8192, true);
      cta[i].tcgen = std::make_shared<functional_tcgen_state>();
      cluster.ctas.push_back(&cta[i]);
      cluster.memories.push_back(&memory[i]);
    }
    context.cta = &cta[rank];
    context.memory = &memory[rank];
    context.cluster = &cluster;
    context.cga_cta_id = rank;
    state.active_mask = 1;
  }
  step_result run() { return sass.step("pair", state, context); }
};
TEST(Tcgen05PairAdapter, AllocationUsesIntersectionOfPeerFreeBlocks) {
  PairFixture f("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", "UP0, UR0, UR1", 2);
  f.cta[2].tcgen->allocated = 1;
  f.cta[3].tcgen->allocated = 2;
  f.state.write_uniform_register(1, 1);
  ASSERT_EQ(f.run().status, step_status::kAdvanced);
  EXPECT_TRUE(f.state.read_uniform_predicate(0));
  EXPECT_EQ(f.state.read_uniform_register(0), 2u);
  EXPECT_EQ(f.cta[2].tcgen->allocated, 5u);
  EXPECT_EQ(f.cta[3].tcgen->allocated, 6u);
  EXPECT_EQ(f.cta[0].tcgen->allocated, 0u);
}
TEST(Tcgen05PairAdapter, OddRankCannotPhysicallyAllocatePair) {
  PairFixture f("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", "UP0, UR0, UR1", 1);
  f.state.write_uniform_register(1, 1);
  EXPECT_EQ(f.run().status, step_status::kUnsupported);
  EXPECT_EQ(f.cta[0].tcgen->allocated, 0u);
  EXPECT_EQ(f.cta[1].tcgen->allocated, 0u);
}
TEST(Tcgen05PairAdapter, MulticastCommitTargetsOnlySelectedRanks) {
  PairFixture f("UTCBAR.2CTA.MULTICAST", "[UR0], URZ, UR1", 2);
  for (auto &cta : f.cta) ASSERT_TRUE(cta.initialize_mbarrier(128, 1));
  f.state.write_uniform_register(0, (2u << 24) | 128);
  f.state.write_uniform_register(1, 0xa);
  ASSERT_EQ(f.run().status, step_status::kAdvanced);
  for (unsigned rank = 0; rank < 4; ++rank)
    EXPECT_EQ(f.cta[rank].mbarrier_phase(128), rank & 1u);
  EXPECT_EQ(f.context.mbarrier_effects[0].multicast_mask, 0xau);
  EXPECT_EQ(f.context.mbarrier_effects[0].address, (2u << 24) | 128);
}
TEST(Tcgen05PairAdapter, ClusterStoreCompletesPeerAndPreservesPayloadWidth) {
  for (unsigned words : {1u, 2u, 4u}) {
    PairFixture f(words == 1   ? "STAS"
                  : words == 2 ? "STAS.64"
                               : "STAS.128",
                  "[R0.64], R8");
    ASSERT_TRUE(f.cta[1].initialize_mbarrier(128, 1));
    ASSERT_TRUE(f.cta[1].arrive_mbarrier(128, 1, words * 4));
    f.state.write_register(0, 0, (1u << 24) | 256);
    f.state.write_register(0, 1, (1u << 24) | 128);
    for (unsigned i = 0; i < words; ++i)
      f.state.write_register(0, 8 + i, 31 + i);
    ASSERT_EQ(f.run().status, step_status::kAdvanced);
    for (unsigned i = 0; i < words; ++i)
      EXPECT_EQ(
          f.memory[1].inspect<uint32_t>(memory_space::kShared, 256 + i * 4),
          31 + i);
    EXPECT_EQ(f.cta[1].mbarrier_phase(128), 1u);
    EXPECT_EQ(f.context.mbarrier_effects[0].address, (1u << 24) | 128);
    EXPECT_EQ(f.context.mbarrier_effects[0].count, words * 4);
  }
}
TEST(Tcgen05PairAdapter, RemoteArrivalUsesLocalStorageAndTaggedTimingTarget) {
  PairFixture f("SYNCS.ARRIVE.TRANS64.RED", "RZ[R0], R1");
  ASSERT_TRUE(f.cta[1].initialize_mbarrier(128, 1));
  f.state.write_register(0, 0, (1u << 24) | 128);
  f.state.write_register(0, 1, 16);
  ASSERT_EQ(f.run().status, step_status::kAdvanced);
  EXPECT_EQ(f.cta[1].mbarrier_pending_transaction_bytes(128), 16u);
  EXPECT_EQ(f.cta[1].mbarrier_pending_arrivals(128), 0u);
  EXPECT_FALSE(f.cta[0].mbarrier_initialized(128));
  EXPECT_EQ(f.context.cta, &f.cta[0]);
  EXPECT_EQ(f.context.cga_cta_id, 0u);
  EXPECT_EQ(f.context.mbarrier_effects[0].address, (1u << 24) | 128);
  EXPECT_EQ(f.context.mbarrier_effects[0].count, 16u);
}
TEST(Tcgen05PairAdapter, InvalidCommitOwnerDoesNotArrive) {
  PairFixture f("UTCBAR", "[UR0], URZ");
  ASSERT_TRUE(f.cta[0].initialize_mbarrier(128, 1));
  f.state.write_uniform_register(0, (255u << 24) | 128);
  EXPECT_EQ(f.run().status, step_status::kUnsupported);
  EXPECT_EQ(f.cta[0].mbarrier_pending_arrivals(128), 1u);
}
TEST(Tcgen05PairAdapter, OverCompletingWarpDoesNotPartiallyStore) {
  PairFixture f("STAS", "[R0.64], R8");
  ASSERT_TRUE(f.cta[1].initialize_mbarrier(128, 1));
  ASSERT_TRUE(f.cta[1].arrive_mbarrier(128, 1, 4));
  f.state.active_mask = 3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    f.state.write_register(lane, 0, (1u << 24) | (256 + lane * 4));
    f.state.write_register(lane, 1, (1u << 24) | 128);
    f.state.write_register(lane, 8, 42);
  }
  EXPECT_EQ(f.run().status, step_status::kUnsupported);
  EXPECT_EQ(f.memory[1].inspect<uint32_t>(memory_space::kShared, 256), 0u);
  EXPECT_EQ(f.cta[1].mbarrier_pending_transaction_bytes(128), 4u);
  EXPECT_FALSE(f.context.mbarrier_effects[0].valid);
}
TEST(Tcgen05PairAdapter, ClusterStoreProjectsAllPayloadDependencies) {
  for (unsigned words : {1u, 2u, 4u}) {
    const timing_instruction inst(official_instruction(0,
                                                       words == 1 ? "STAS"
                                                       : words == 2
                                                           ? "STAS.64"
                                                           : "STAS.128",
                                                       "[R0.64], R8"));
    EXPECT_EQ(inst.outcount, 0u);
    std::vector<int> expected{1, 2};
    for (unsigned i = 0; i < words; ++i) expected.push_back(9 + i);
    EXPECT_EQ(
        std::vector<int>(inst.arch_reg.src, inst.arch_reg.src + inst.incount),
        expected);
    EXPECT_TRUE(inst.get_mbarrier_static_info().per_lane_complete_tx);
  }
}
}  // namespace

TEST(Tcgen05PairAdapter, MmaRejectsTruncatedMaskAndReservedWsDescriptor) {
  for (const char *op : {"UTCHMMA", "UTCHMMA.2CTA", "UTCHMMA.WS"}) {
    const auto mask =
        std::to_string(flash_gpgpu_sim::sass::kZeroUniformRegister - 1);
    const auto operands =
        "gdesc[UR0],gdesc[UR2],tmem[UR4],tmem[UR5],idesc[UR6],UR" + mask +
        ",!UP0";
    PairFixture f(op, operands.c_str());
    auto result = f.run();
    EXPECT_EQ(result.status, step_status::kUnsupported);
    EXPECT_NE(result.detail.find("invalid MMA mask register group"),
              std::string::npos);
  }
  PairFixture f(
      "UTCHMMA.WS",
      "gdesc[UR0],gdesc[UR2],tmem[UR4],tmem[UR5],idesc[UR6],UR12,!UP0");
  f.state.write_uniform_register(6, (4u << 24) | (8u << 17) | 0x10);
  f.state.write_uniform_register(
      13, 1u << 4);  // Descriptor bit 36 is reserved.
  auto result = f.run();
  EXPECT_EQ(result.status, step_status::kUnsupported);
  EXPECT_NE(result.detail.find("invalid WS B zero-column descriptor"),
            std::string::npos);
}

TEST(Tcgen05PairAdapter, MmaRejectsPatternBaseBeforeWritingAccumulator) {
  for (unsigned kind = 0; kind < 4; ++kind) {
    PairFixture f(
        kind < 2    ? "UTCHMMA"
        : kind == 2 ? "UTCQMMA"
                    : "UTCOMMA",
        kind < 2 ? "gdesc[UR0],gdesc[UR2],tmem[UR4],tmem[UR5],idesc[UR6],UP0"
                 : "gdesc[UR0],gdesc[UR2],tmem[UR4],tmem[UR5],idesc[UR6],tmem["
                   "UR8],UP0");
    f.cta[0].tcgen->allocated = 0xffff;
    const uint64_t ad = (6ull << 61) | (1ull << 49) | (1ull << 46) |
                        (16ull << 32) | (1ull << 16) | 8;
    const uint64_t bd = (1ull << 46) | (16ull << 32) | (8ull << 16) | 256;
    f.state.write_uniform_register(0, static_cast<uint32_t>(ad));
    f.state.write_uniform_register(1, ad >> 32);
    f.state.write_uniform_register(2, static_cast<uint32_t>(bd));
    f.state.write_uniform_register(3, bd >> 32);
    uint32_t idesc = (8u << 24) | (2u << 17) | 0x10;
    if (kind == 1) idesc |= (2u << 7) | (2u << 10);
    if (kind >= 2) idesc = (1u << 27) | (1u << 23) | (2u << 17);
    if (kind == 3) idesc |= (1u << 7) | (1u << 10);
    f.state.write_uniform_register(6, idesc);
    f.state.write_uniform_register(8, 64);
    f.state.write_uniform_register(9, 68);
    f.cta[0].tcgen->memory.write_words({0, 0}, 0, {0x12345678});
    EXPECT_EQ(f.run().status, step_status::kUnsupported) << kind;
    EXPECT_EQ(f.cta[0].tcgen->memory.read_words({0, 0}, 0, 1)[0], 0x12345678u);
    EXPECT_EQ(f.state.pc, 0u);
    EXPECT_FALSE(f.context.tcgen_mma_effect.valid);
  }
}

TEST(Tcgen05PairAdapter, Fp8RejectsTruncatedMaskBeforeSideEffects) {
  const auto operands =
      "gdesc[UR0],gdesc[UR2],tmem[UR4],tmem[UR5],idesc[UR6],UR" +
      std::to_string(flash_gpgpu_sim::sass::kZeroUniformRegister - 2) + ",UP0";
  PairFixture f("UTCQMMA", operands.c_str());
  const auto result = f.run();
  EXPECT_EQ(result.status, step_status::kUnsupported);
  EXPECT_NE(result.detail.find("invalid dense FP8 output-lane mask"),
            std::string::npos);
  EXPECT_EQ(f.state.pc, 0u);
  EXPECT_FALSE(f.context.tcgen_mma_effect.valid);
}
