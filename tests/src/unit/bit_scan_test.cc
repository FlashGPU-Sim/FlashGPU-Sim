#include <gtest/gtest.h>
#include "gpgpu-sim/flash/bit_scan.h"

TEST(BitScan, EveryBitAndItsSignedComplement) {
  using flash_gpgpu_sim::find_non_sign_bit;
  for (unsigned bits : {32u, 64u}) {
    EXPECT_EQ(find_non_sign_bit(0, bits, false, false), UINT32_MAX);
    EXPECT_EQ(find_non_sign_bit(0, bits, true, true), UINT32_MAX);
    EXPECT_EQ(find_non_sign_bit(UINT64_MAX, bits, true, false), UINT32_MAX);
    EXPECT_EQ(find_non_sign_bit(UINT64_MAX, bits, true, true), UINT32_MAX);
    for (unsigned i = 0; i < bits; ++i) {
      const uint64_t value = uint64_t(1) << i;
      EXPECT_EQ(find_non_sign_bit(value, bits, false, false), i);
      EXPECT_EQ(find_non_sign_bit(value, bits, false, true), bits - 1 - i);
      if (i < bits - 1) {
        EXPECT_EQ(find_non_sign_bit(value, bits, true, false), i);
        EXPECT_EQ(find_non_sign_bit(~value, bits, true, false), i);
        EXPECT_EQ(find_non_sign_bit(~value, bits, true, true), bits - 1 - i);
      } else {
        EXPECT_EQ(find_non_sign_bit(value, bits, true, false), bits - 2);
      }
    }
  }
}

TEST(BitScan, IgnoresBitsBeyondOperandWidth) {
  using flash_gpgpu_sim::find_non_sign_bit;
  EXPECT_EQ(find_non_sign_bit(0xffffffff00000000ull, 32, false, false), UINT32_MAX);
  EXPECT_EQ(find_non_sign_bit(0x1234567880000000ull, 32, true, true), 1u);
}
