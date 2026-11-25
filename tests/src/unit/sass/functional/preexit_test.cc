#include "../test_support.h"

namespace {
TEST(SassPreexit, PreservesLanesUntilActualGuardedExit) {
  for (const char *guard : {"", "@P0", "@!P0"}) {
    kernel image;
    image.name = "preexit";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "PREEXIT", "", guard),
        official_instruction(16, "MOV", "R2,0x2a"),
        official_instruction(32, "EXIT", "", "@P0"),
        official_instruction(48, "MOV", "R2,0x35"),
        official_instruction(64, "EXIT")};
    frontend f;
    f.add_kernel(std::move(image));
    register_sm100_functional_semantics(f);
    warp_state s;
    s.active_mask = 3;
    s.write_predicate(0, 0, true);
    s.write_predicate(1, 0, false);
    execution_context c;
    ASSERT_EQ(f.step("preexit", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(s.pc, 16u);
    EXPECT_EQ(s.active_mask, 3u);
    ASSERT_EQ(f.step("preexit", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(s.read_register(0, 2), 42u);
    EXPECT_EQ(s.read_register(1, 2), 42u);
    ASSERT_EQ(f.step("preexit", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(s.active_mask, 2u);
    ASSERT_EQ(f.step("preexit", s, c).status, step_status::kAdvanced);
    EXPECT_EQ(s.read_register(0, 2), 42u);
    EXPECT_EQ(s.read_register(1, 2), 53u);
    EXPECT_EQ(f.step("preexit", s, c).status, step_status::kExited);
  }
}

TEST(SassPreexit, TimingIssuesWithoutWarpRetirementOrDataDependencies) {
  const timing_instruction projected(official_instruction(0, "PREEXIT"));
  EXPECT_EQ(projected.op, ALU_OP);
  EXPECT_NE(projected.op, EXIT_OPS);
  EXPECT_EQ(projected.incount, 0u);
  EXPECT_EQ(projected.outcount, 0u);
  const timing_instruction guarded(official_instruction(0, "PREEXIT", "", "@P0"));
  EXPECT_NE(guarded.op, EXIT_OPS);
  EXPECT_EQ(guarded.outcount, 0u);
}
}  // namespace
