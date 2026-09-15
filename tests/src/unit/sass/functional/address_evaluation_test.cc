#include <gtest/gtest.h>

#include "gpgpu-sim/flash/sass/decode/operand_parser.h"
#include "gpgpu-sim/flash/sass/functional/internal.h"

namespace {
using namespace flash_gpgpu_sim::sass;

TEST(SassAddressEvaluationTest, SharesConstantAndMemoryTermArithmetic) {
  warp_state state;
  state.write_register(3, 2, 0xfffffff0u);
  state.write_register(3, 3, 1);
  state.write_uniform_register(4, 0x80);
  const auto memory = parse_operand("[R2.64+UR4+-0x10]");
  const auto constant = parse_operand("c[0][R2.64+UR4+-0x10]");
  ASSERT_EQ(memory.kind, operand_kind::kMemoryAddress);
  ASSERT_EQ(constant.kind, operand_kind::kConstantMemory);
  uint64_t value = 0;
  ASSERT_TRUE(functional_detail::evaluate_address(memory, state, 3, value));
  EXPECT_EQ(value, 0x200000060ull);
  EXPECT_EQ(functional_detail::evaluate_address_terms(constant.constant_address,
                                                      state, 3),
            value);
  ASSERT_TRUE(
      functional_detail::evaluate_shared_address(memory, state, 3, value));
  EXPECT_EQ(value, 0x60u);
}

TEST(SassAddressEvaluationTest, PreservesModuloArithmeticAndShapeChecks) {
  warp_state state;
  state.write_register(0, 0, 1);
  uint64_t value = 0;
  ASSERT_TRUE(functional_detail::evaluate_address(parse_operand("[-R0]"), state,
                                                  0, value));
  EXPECT_EQ(value, ~uint64_t{0});
  EXPECT_FALSE(functional_detail::evaluate_address(parse_operand("[R0][R2]"),
                                                   state, 0, value));
  EXPECT_FALSE(functional_detail::evaluate_address(parse_operand("R0"), state,
                                                   0, value));
  EXPECT_EQ(functional_detail::evaluate_address_terms({}, state, 0), 0u);
}

}  // namespace
