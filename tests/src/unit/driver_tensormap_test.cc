#include <gtest/gtest.h>
#include "sass/test_support.h"
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
TEST(DriverTensorMap, PackedRequiresItsOwnPayloadSemantics) {
  for (uint32_t control : {0x264590u, 0x266630u}) {
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

TEST(DriverTensorMap, CapturedTf32FormatsAndDestinationPreservation) {
  for (unsigned rank : {2u, 3u}) {
    // MHC's retained 3D TF32 descriptor; only the address is anonymized.
    auto encoded = map({0x400000, 0x1c00002763a0, 0x700, 0, 0x1700001bff, 3,
                        0x1f00000000000000, 0x317, 0x3000, 0x400});
    if (rank == 2) {
      encoded = byte2;
      encoded.raw_u64[1] = ((byte2.raw_u64[1] >> 32) * 4 << 32) | 0x276390;
      encoded.raw_u64[8] *= 4;
    }
    tensormap_descriptor_t d{};
    ASSERT_TRUE(decode_sm100_driver_tensormap(encoded, rank, d));
    EXPECT_EQ(d.fields.tensorDataType, TMA_DTYPE_TF32);
    EXPECT_EQ(d.get_element_size(), 4u);
    const auto saved = d;
    encoded.raw_u64[8] ^= 1;
    EXPECT_FALSE(decode_sm100_driver_tensormap(encoded, rank, d));
    EXPECT_EQ(std::memcmp(&d, &saved, sizeof(d)), 0);
  }
}

TEST(DriverTensorMap, Native3DStoreAddressFormsAndQualifierGuards) {
  for (const char *operands : {"[UR4][UR8]", "[UR4],[UR8]",
                              "[UR4][UR8],desc[UR12]",
                              "[UR4],[UR8],desc[UR12]"}) {
    for (const uint64_t qualifier : {0x1000000000000000ull,
                                     0x12f0000000000000ull,
                                     0x14f0000000000000ull, 1ull}) {
      const bool qualified = std::strstr(operands, "desc") != nullptr;
      const bool valid = !qualified || qualifier != 1;
      const auto inst = official_instruction(0, "UTMASTG.3D", operands);
      kernel image;
      image.name = "store3d";
      image.arch = architecture::kSm100;
      image.instruction_bytes = 16;
      image.instructions = {inst};
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm100_functional_semantics(sass);
      mapped_functional_memory memory;
      memory.map(flash_gpgpu_sim::sass::memory_space::kGlobal, 0x1000, 128, true);
      memory.map(flash_gpgpu_sim::sass::memory_space::kGlobal, 0x2000, 128, true);
      memory.map(flash_gpgpu_sim::sass::memory_space::kShared, 0, 32, true);
      const uint64_t words[16] = {0x2000, 0x100240020ull, 4, 0,
          0x30000000full, 1, 0x0f00000000000000ull, 1, 32, 0x10};
      ASSERT_TRUE(memory.write(flash_gpgpu_sim::sass::memory_space::kGlobal, 0x1000, words, sizeof(words)));
      for (unsigned i = 0; i < 128; ++i)
        memory.initialize<uint8_t>(flash_gpgpu_sim::sass::memory_space::kGlobal, 0x2000+i, 0xa5);
      for (unsigned i = 0; i < 32; ++i)
        memory.initialize<uint8_t>(flash_gpgpu_sim::sass::memory_space::kShared, i, i+1);
      cta_execution_state cta(32);
      execution_context context{&memory};
      context.cta = &cta;
      warp_state state;
      state.active_mask = 1u << 7;
      state.write_uniform_register(4, 0);
      state.write_uniform_register(5, 0);
      state.write_uniform_register(6, 1);
      state.write_uniform_register(7, 1);
      state.write_uniform_register(8, 0x1000);
      state.write_uniform_register(9, 0);
      state.write_uniform_register(12, qualifier);
      state.write_uniform_register(13, qualifier >> 32);
      const auto result = sass.step("store3d", state, context);
      if (valid) ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      else EXPECT_NE(result.status, step_status::kAdvanced);
      for (unsigned i = 0; i < 128; ++i)
        EXPECT_EQ(memory.inspect<uint8_t>(flash_gpgpu_sim::sass::memory_space::kGlobal, 0x2000+i),
                  valid && i >= 80 && i < 112 ? i-79 : 0xa5) << operands << ":" << i;
    }
  }
}
