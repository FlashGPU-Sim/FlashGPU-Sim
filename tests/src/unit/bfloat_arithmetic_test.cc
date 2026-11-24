#include <gtest/gtest.h>
#include "gpgpu-sim/flash/bfloat_arithmetic.h"
using namespace flash_gpgpu_sim;
TEST(BfloatArithmetic, MidpointResidualAndZeroSign) {
  // 1.5*1.0078125 is an exact BF16 midpoint (1.51171875).
  const float a = bf16_to_f32(0x3fc0), b = bf16_to_f32(0x3f81);
  EXPECT_EQ(bf16_fma_rn(a, b, 0), 0x3fc2);
  EXPECT_EQ(bf16_fma_rn(a, b, -bf16_to_f32(1)), 0x3fc1);
  EXPECT_EQ(bf16_fma_rn(a, b, bf16_to_f32(1)), 0x3fc2);
  EXPECT_EQ(bf16_mul_rn(-0.0f, 2.0f), 0x8000);
  EXPECT_EQ(bf16_add_rn(-0.0f, -0.0f), 0x8000);
}
TEST(BfloatArithmetic, PreservesHostRounding) {
  const int saved = std::fegetround();
  for (int mode : {FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD}) {
    std::fesetround(mode);
    EXPECT_EQ(bf16_add_rn(1.0f, bf16_to_f32(0x3b80)), 0x3f80);
    EXPECT_EQ(bf16_mul_rn(bf16_to_f32(0x3fc0), bf16_to_f32(0x3f81)), 0x3fc2);
    EXPECT_EQ(bf16_fma_rn(bf16_to_f32(0x3fc0), bf16_to_f32(0x3f81), 0), 0x3fc2);
    EXPECT_EQ(std::fegetround(), mode);
  }
  std::fesetround(saved);
}
