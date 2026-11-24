#include <gtest/gtest.h>
#include <set>
#include "gpgpu-sim/flash/tcgen05/copy.h"
using namespace flash_gpgpu_sim;
TEST(Tcgen05CopyLayout, BroadcastSelectorsCoverAllDatapaths) {
  for (auto shape :
       {tcgen05_copy_shape::warpx4, tcgen05_copy_shape::warpx2_02_13,
        tcgen05_copy_shape::warpx2_01_23}) {
    std::vector<uint32_t> addresses;
    ASSERT_TRUE(tcgen05_copy_addresses((1ull << 46) | 0x40, shape, addresses));
    ASSERT_EQ(addresses.size(), 512u);
    for (unsigned row = 0; row < 128; ++row)
      for (unsigned col = 0; col < 4; ++col) {
        const unsigned source = shape == tcgen05_copy_shape::warpx4 ? row % 32
                                : shape == tcgen05_copy_shape::warpx2_02_13
                                    ? row % 64
                                    : (row / 64) * 32 + row % 32;
        EXPECT_EQ(addresses[row * 4 + col], 0x400 + source * 16 + col * 4);
      }
  }
}
TEST(Tcgen05CopyLayout, MalformedDescriptorsFailBeforeProducingAddresses) {
  constexpr uint64_t valid = (1ull << 46) | 0x40;
  for (unsigned bit :
       {14u, 15u, 16u, 30u, 31u, 32u, 47u, 48u, 49u, 52u, 60u, 61u, 62u, 63u}) {
    std::vector<uint32_t> addresses{0xdeadbeef};
    EXPECT_FALSE(tcgen05_copy_addresses(valid ^ (1ull << bit),
                                        tcgen05_copy_shape::warpx4, addresses));
    EXPECT_TRUE(addresses.empty());
  }
  std::vector<uint32_t> addresses;
  EXPECT_FALSE(tcgen05_copy_addresses(
      valid, tcgen05_copy_shape::rows128_bits256, addresses));
}
TEST(Tcgen05CopyLayout, SwizzledCopyIsPermutationAndKeepsRowWidths) {
  for (unsigned mode : {2u, 4u, 6u}) {
    const unsigned width = mode == 2 ? 128 : mode == 4 ? 64 : 32;
    const uint64_t descriptor = (uint64_t(mode) << 61) | (1ull << 46) |
                                (uint64_t(width / 2) << 32) | (1ull << 16) |
                                0x80;
    std::vector<uint32_t> addresses;
    ASSERT_TRUE(tcgen05_copy_addresses(
        descriptor, tcgen05_copy_shape::rows128_bits256, addresses));
    std::set<uint32_t> unique(addresses.begin(), addresses.end());
    EXPECT_EQ(unique.size(), 1024u);
    EXPECT_GE(*unique.begin(), 0x800u);
    EXPECT_LT(*unique.rbegin(), 0x800u + 128 * width);
  }
}
