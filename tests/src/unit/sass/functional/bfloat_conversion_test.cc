#include "../test_support.h"
namespace {
TEST(SassBfloatConversion, MergePreservesLowHalfAndAliasingAndPredicates) {
  kernel image;
  image.name = "bf16_merge";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(
      0, "F2FP.SATFINITE.RELU.BF16.F32.MERGE_C.RZ", "R3,R1,R3", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 1, 0x3f818000);
    state.write_register(lane, 3, 0xa5f01234u + lane);
    state.write_predicate(lane, 0, lane % 2 == 0);
  }
  auto result = sass.step("bf16_merge", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(
        state.read_register(lane, 3),
        lane % 2 ? 0xa5f01234u + lane : ((0x1234u + lane) << 16) | 0x3f81u);
}
TEST(SassBfloatConversion, RejectsMalformedMergeRegister) {
  for (const auto operand : {"R3,R1,-R2", "R3,R1,R2.H0", "R3,R1,R256"}) {
    kernel image;
    image.name = "invalid_merge";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "F2FP.BF16.F32.MERGE_C", operand)};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    warp_state state;
    execution_context context;
    const auto result = sass.step("invalid_merge", state, context);
    EXPECT_NE(result.status, step_status::kAdvanced) << operand;
    EXPECT_EQ(state.pc, 0u);
  }
}
}  // namespace
