#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include "gpgpu-sim/flash/tma_opaque_tensormap.h"

namespace {
using flash_gpgpu_sim::decode_sm100_driver_tensormap;
using flash_gpgpu_sim::decode_sm100_opaque_tensormap;
tensormap_descriptor_t map(std::initializer_list<uint64_t> words) {
  tensormap_descriptor_t value{};
  std::copy(words.begin(), words.end(), value.raw_u64);
  return value;
}
// Retained B200 DeepInfer by-value descriptors, addresses anonymized only.
const auto byte2 = map({0x100000, 0x1c000266010, 0, 0, 0x1ff00001bff, 0,
                        0x7f00000000000000, 0x7f, 0x4000, 0x400});
const auto bf16_3 = map({0x200000, 0xe0000266520, 0x380, 0, 0x1ff00001bff, 3,
                         0x3f00000000000000, 0x33f, 0x8000, 0x400});
const auto byte4 =
    map({0x300000, 0x1c000266030, 0x2a000000150000, 0, 0xbff00001bff, 1,
         0x7f00000000000000, 0x7f, 0x4000, 0x400});
TEST(DriverTensorMap, RetainedDenseTwoDimensionalBytes) {
  tensormap_descriptor_t d{};
  ASSERT_TRUE(decode_sm100_driver_tensormap(byte2, 2, d));
  EXPECT_EQ(d.fields.globalAddress, 0x100000u);
  EXPECT_EQ(d.fields.tensorDataType, TMA_DTYPE_U8);
  EXPECT_EQ(d.fields.tensorRank, 1u);
  EXPECT_EQ(d.fields.globalDim[0], 7168u);
  EXPECT_EQ(d.fields.globalDim[1], 512u);
  EXPECT_EQ(d.fields.boxDim[0], 128u);
  EXPECT_EQ(d.fields.boxDim[1], 128u);
  EXPECT_EQ(d.fields.globalStrides[0], 7168u);
  EXPECT_EQ(d.fields.swizzle, TMA_SWIZZLE_128B);
}
TEST(DriverTensorMap, RetainedBfloatThreeDimensionalInterleavedPlanes) {
  tensormap_descriptor_t d{};
  ASSERT_TRUE(decode_sm100_driver_tensormap(bf16_3, 3, d));
  EXPECT_EQ(d.fields.tensorDataType, TMA_DTYPE_BF16);
  EXPECT_EQ(d.fields.globalDim[0], 7168u);
  EXPECT_EQ(d.fields.globalDim[1], 512u);
  EXPECT_EQ(d.fields.globalDim[2], 4u);
  EXPECT_EQ(d.fields.boxDim[0], 64u);
  EXPECT_EQ(d.fields.boxDim[1], 64u);
  EXPECT_EQ(d.fields.boxDim[2], 4u);
  EXPECT_EQ(d.fields.globalStrides[0], 57344u);
  EXPECT_EQ(d.fields.globalStrides[1], 14336u);
}
TEST(DriverTensorMap, RetainedFourDimensionalWeightGeometry) {
  tensormap_descriptor_t d{};
  ASSERT_TRUE(decode_sm100_driver_tensormap(byte4, 4, d));
  EXPECT_EQ(d.fields.globalDim[0], 7168u);
  EXPECT_EQ(d.fields.globalDim[1], 3072u);
  EXPECT_EQ(d.fields.globalDim[2], 2u);
  EXPECT_EQ(d.fields.globalDim[3], 1u);
  EXPECT_EQ(d.fields.boxDim[2], 1u);
  EXPECT_EQ(d.fields.boxDim[3], 1u);
  EXPECT_EQ(d.fields.globalStrides[0], 7168u);
  EXPECT_EQ(d.fields.globalStrides[1], 22020096u);
  EXPECT_EQ(d.fields.globalStrides[2], 44040192u);
}
TEST(DriverTensorMap, RejectionPreservesDestinationAndDoesNotAliasRank) {
  tensormap_descriptor_t d{}, saved{};
  d.fields.globalAddress = 0xdeadbeef;
  saved = d;
  for (unsigned rank : {0, 1, 3, 4, 5, 6}) {
    EXPECT_FALSE(decode_sm100_driver_tensormap(byte2, rank, d));
    EXPECT_EQ(std::memcmp(&d, &saved, sizeof(d)), 0);
  }
  for (unsigned index : {2, 3, 5, 6, 7, 8, 9, 10, 15}) {
    auto bad = byte2;
    bad.raw_u64[index] ^= 1;
    EXPECT_FALSE(decode_sm100_driver_tensormap(bad, 2, d)) << index;
    EXPECT_EQ(std::memcmp(&d, &saved, sizeof(d)), 0);
  }
  auto overflow = byte2;
  overflow.raw_u64[4] |= UINT32_MAX;
  EXPECT_FALSE(decode_sm100_driver_tensormap(overflow, 2, d));
  auto stride = byte2;
  stride.raw_u64[1] &= UINT32_MAX;
  EXPECT_FALSE(decode_sm100_driver_tensormap(stride, 2, d));
}
TEST(DriverTensorMap, PackedAndTf32RequireTheirOwnPayloadSemantics) {
  for (uint32_t control : {0x264590u, 0x276390u, 0x266630u, 0x2763a0u}) {
    auto encoded = byte2;
    encoded.raw_u64[1] = (encoded.raw_u64[1] & 0xffffffff00000000ull) | control;
    tensormap_descriptor_t d{};
    for (unsigned rank = 2; rank <= 5; ++rank)
      EXPECT_FALSE(decode_sm100_driver_tensormap(encoded, rank, d));
  }
}
TEST(DriverTensorMap, PtxOpaqueEntryUsesSameCanonicalFields) {
  for (auto encoded : {byte2, bf16_3, byte4}) {
    tensormap_descriptor_t a{}, b{};
    ASSERT_TRUE(decode_sm100_opaque_tensormap(encoded, a));
    ASSERT_TRUE(decode_sm100_driver_tensormap(encoded, a.num_dims(), b));
    EXPECT_EQ(std::memcmp(&a, &b, sizeof(a)), 0);
  }
}
}  // namespace
