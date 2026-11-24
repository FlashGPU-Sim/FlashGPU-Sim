#include <gtest/gtest.h>
#include "gpgpu-sim/flash/bfloat_conversion.h"
using flash_gpgpu_sim::f32_bits_to_bf16;
TEST(BfloatBits, ExactValuesAndCanonicalNaNs) {
  for (uint32_t value = 0; value < 65536; ++value) {
    const uint16_t expected = (value & 0x7fff) > 0x7f80 ? 0x7fff : value;
    EXPECT_EQ(f32_bits_to_bf16(value << 16), expected) << value;
    EXPECT_EQ(f32_bits_to_bf16(value << 16, true), expected) << value;
  }
}
TEST(BfloatBits, SpecialValuesAndRoundingBoundaries) {
  EXPECT_EQ(f32_bits_to_bf16(0xff812345, false, true, true), 0x7fff);
  EXPECT_EQ(f32_bits_to_bf16(0xff800000, false, true, true), 0);
  EXPECT_EQ(f32_bits_to_bf16(0x80000000, false, true, true), 0x8000);
  EXPECT_EQ(f32_bits_to_bf16(0x7f7fffff), 0x7f80);
  EXPECT_EQ(f32_bits_to_bf16(0x7f7fffff, true), 0x7f7f);
  EXPECT_EQ(f32_bits_to_bf16(0x7f7fffff, false, false, true), 0x7f7f);
  EXPECT_EQ(f32_bits_to_bf16(0x3f808000), 0x3f80);
  EXPECT_EQ(f32_bits_to_bf16(0x3f818000), 0x3f82);
  EXPECT_EQ(f32_bits_to_bf16(0x3f818000, true), 0x3f81);
}
