#include "../test_support.h"

namespace {
using flash_gpgpu_sim::cta_reduction_op;
frontend reduction_image(std::vector<instruction> instructions) {
  kernel image;
  image.name = "reduction";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = std::move(instructions);
  frontend f;
  f.add_kernel(std::move(image));
  register_sm100_functional_semantics(f);
  return f;
}
TEST(SassCtaReduction, FastNextGenerationPreservesSlowWarpResult) {
  auto f = reduction_image(
      {official_instruction(0, "BAR.RED.OR.DEFER_BLOCKING", "0,P0"),
       official_instruction(16, "B2R.RESULT", "RZ,P1"),
       official_instruction(32, "BAR.RED.AND.DEFER_BLOCKING", "0,!P0"),
       official_instruction(48, "B2R.RESULT", "RZ,P2")});
  cta_execution_state cta(2);
  execution_context c;
  c.cta = &cta;
  warp_state a, b;
  b.write_predicate(31, 0, true);
  c.warp_id = 0;
  ASSERT_EQ(f.step("reduction", a, c).status, step_status::kAdvanced);
  EXPECT_EQ(f.step("reduction", a, c).status, step_status::kBlocked);
  EXPECT_EQ(a.pc, 16u);
  c.warp_id = 1;
  ASSERT_EQ(f.step("reduction", b, c).status, step_status::kAdvanced);
  ASSERT_EQ(f.step("reduction", b, c).status, step_status::kAdvanced);
  ASSERT_EQ(f.step("reduction", b, c).status, step_status::kAdvanced);
  c.warp_id = 0;
  ASSERT_EQ(f.step("reduction", a, c).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_TRUE(a.read_predicate(lane, 1));
    EXPECT_TRUE(b.read_predicate(lane, 1));
  }
  ASSERT_EQ(f.step("reduction", a, c).status, step_status::kAdvanced);
  ASSERT_EQ(f.step("reduction", a, c).status, step_status::kAdvanced);
  c.warp_id = 1;
  ASSERT_EQ(f.step("reduction", b, c).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_FALSE(a.read_predicate(lane, 2));
    EXPECT_FALSE(b.read_predicate(lane, 2));
  }
}
TEST(SassCtaReduction, PopcountExplicitCountAndCtaIsolation) {
  auto f = reduction_image(
      {official_instruction(0, "BAR.RED.POPC.DEFER_BLOCKING", "3,64,!P0"),
       official_instruction(16, "B2R.RESULT", "R4")});
  cta_execution_state ctas[2] = {cta_execution_state(4),
                                 cta_execution_state(4)};
  warp_state states[2][2];
  execution_context c;
  for (unsigned cta = 0; cta < 2; ++cta) {
    c.cta = &ctas[cta];
    for (unsigned warp = 0; warp < 2; ++warp) {
      c.warp_id = warp;
      for (unsigned lane = 0; lane < 32; ++lane)
        states[cta][warp].write_predicate(lane, 0, lane >= 3 + cta);
      ASSERT_EQ(f.step("reduction", states[cta][warp], c).status,
                step_status::kAdvanced);
    }
  }
  for (unsigned cta = 0; cta < 2; ++cta) {
    c.cta = &ctas[cta];
    for (unsigned warp = 0; warp < 2; ++warp) {
      c.warp_id = warp;
      ASSERT_EQ(f.step("reduction", states[cta][warp], c).status,
                step_status::kAdvanced);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(states[cta][warp].read_register(lane, 4), 6u + 2 * cta);
    }
    uint32_t ignored;
    cta_reduction_op op;
    EXPECT_FALSE(ctas[cta].read_barrier_reduction(2, ignored, op));
  }
}
TEST(SassCtaReduction, NonparticipantExitPublishesOnlyOnce) {
  auto f = reduction_image(
      {official_instruction(0, "BAR.RED.OR.DEFER_BLOCKING", "0,P0"),
       official_instruction(16, "B2R.RESULT", "RZ,P1")});
  cta_execution_state cta(3);
  execution_context c;
  c.cta = &cta;
  c.warp_id = 0;
  warp_state s;
  s.write_predicate(0, 0, true);
  ASSERT_EQ(f.step("reduction", s, c).status, step_status::kAdvanced);
  cta.retire_warp(1);
  EXPECT_EQ(f.step("reduction", s, c).status, step_status::kBlocked);
  cta.retire_warp(2);
  ASSERT_EQ(f.step("reduction", s, c).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_TRUE(s.read_predicate(lane, 1));
}
TEST(SassCtaReduction, MismatchedOperationsDoNotCompleteBarrier) {
  auto f = reduction_image(
      {official_instruction(0, "BAR.RED.OR.DEFER_BLOCKING", "0,P0"),
       official_instruction(16, "BAR.RED.AND.DEFER_BLOCKING", "0,P0")});
  cta_execution_state cta(2);
  execution_context c;
  c.cta = &cta;
  warp_state a, b;
  c.warp_id = 0;
  ASSERT_EQ(f.step("reduction", a, c).status, step_status::kAdvanced);
  c.warp_id = 1;
  b.pc = 16;
  EXPECT_EQ(f.step("reduction", b, c).status, step_status::kUnsupported);
  EXPECT_EQ(b.pc, 16u);
  EXPECT_EQ(cta.generation(0), 0u);
  uint32_t value;
  cta_reduction_op op;
  EXPECT_FALSE(cta.read_barrier_reduction(0, value, op));
}
TEST(SassCtaReduction, ResultShapeMismatchPreservesDestinations) {
  auto f = reduction_image(
      {official_instruction(0, "BAR.RED.OR.DEFER_BLOCKING", "0,PT"),
       official_instruction(16, "B2R.RESULT", "R4"),
       official_instruction(32, "B2R.RESULT", "RZ,P1")});
  cta_execution_state cta(1);
  execution_context c;
  c.cta = &cta;
  c.warp_id = 0;
  warp_state s;
  s.write_register(0, 4, 99);
  ASSERT_EQ(f.step("reduction", s, c).status, step_status::kAdvanced);
  EXPECT_EQ(f.step("reduction", s, c).status, step_status::kUnsupported);
  EXPECT_EQ(s.pc, 16u);
  EXPECT_EQ(s.read_register(0, 4), 99u);
  s.pc = 32;
  ASSERT_EQ(f.step("reduction", s, c).status, step_status::kAdvanced);
  EXPECT_TRUE(s.read_predicate(0, 1));
}
TEST(SassCtaReduction, PredicatedOffResultDoesNotRequireAnArrival) {
  auto f =
      reduction_image({official_instruction(0, "B2R.RESULT", "R4", "@P0")});
  cta_execution_state cta(1);
  execution_context c;
  c.cta = &cta;
  warp_state s;
  s.write_register(0, 4, 99);
  ASSERT_EQ(f.step("reduction", s, c).status, step_status::kAdvanced);
  EXPECT_EQ(s.pc, 16u);
  EXPECT_EQ(s.read_register(0, 4), 99u);
}
TEST(SassCtaReduction, TimingResultIsAnOrderedDestination) {
  const auto result = official_instruction(0, "B2R.RESULT", "RZ,P1");
  const timing_instruction pred(result);
  EXPECT_EQ(pred.outcount, 1u);
  EXPECT_EQ(pred.incount, 0u);
  EXPECT_TRUE(orders_deferred_cta_barrier(result, architecture::kSm100));
  const timing_instruction integer(official_instruction(0, "B2R.RESULT", "R4"));
  EXPECT_EQ(integer.outcount, 1u);
  EXPECT_EQ(integer.incount, 0u);
  const timing_instruction arrive(
      official_instruction(0, "BAR.RED.OR.DEFER_BLOCKING", "2,64,P0"));
  EXPECT_EQ(arrive.op, BARRIER_OP);
  EXPECT_EQ(arrive.bar_type, ARRIVE);
  EXPECT_EQ(arrive.bar_id, 2u);
  EXPECT_EQ(arrive.bar_count, 64u);
}
}  // namespace
