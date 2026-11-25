#include <gtest/gtest.h>

#include "gpgpu-sim/flash/tcgen05.h"

#include <cmath>
#include <cstring>

using namespace flash_gpgpu_sim;

namespace {

constexpr tcgen05_tmem_scope_t kScope0{0, 0};
constexpr tcgen05_tmem_scope_t kScope1{0, 1};
// The peer CTA of kScope0's pair, on the other SM of the same cluster.
constexpr tcgen05_tmem_scope_t kScopePeer{1, 0};

std::vector<uint32_t> sequence(uint32_t first, uint32_t count) {
  std::vector<uint32_t> values;
  values.reserve(count);
  for (uint32_t i = 0; i < count; ++i) values.push_back(first + i);
  return values;
}

uint64_t make_shared_desc(uint32_t start, uint32_t lbo, uint32_t sbo,
                          uint8_t swizzle = 0) {
  return ((static_cast<uint64_t>((start >> 4) & 0x3fff)) << 0) |
         ((static_cast<uint64_t>((lbo >> 4) & 0x3fff)) << 16) |
         ((static_cast<uint64_t>((sbo >> 4) & 0x3fff)) << 32) |
         (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(swizzle) << 61);
}

uint32_t make_f16_idesc(uint32_t m, uint32_t n, bool transpose_b = false) {
  return (TCGEN05_MMA_TYPE_FIELD_ONE << 4) | ((n >> 3) << 17) |
         (static_cast<uint32_t>(transpose_b) << 16) | ((m >> 4) << 24);
}

uint32_t make_mxf4_idesc(uint32_t n, uint8_t a_scale_factor_id = 0,
                         uint8_t b_scale_factor_id = 0, uint32_t m = 128,
                         uint32_t k = 64, bool negate_a = false,
                         bool negate_b = false) {
  return (static_cast<uint32_t>(b_scale_factor_id) << 4) |
         (TCGEN05_MXF4_FORMAT_E2M1 << 7) | (TCGEN05_MXF4_FORMAT_E2M1 << 10) |
         (static_cast<uint32_t>(negate_a) << 13) |
         (static_cast<uint32_t>(negate_b) << 14) | ((n >> 3) << 17) |
         (TCGEN05_SCALE_FORMAT_UE8M0 << 23) | ((m >> 7) << 27) |
         (static_cast<uint32_t>(a_scale_factor_id) << 29) |
         (static_cast<uint32_t>(k == 96) << 31);
}

uint32_t make_mxf8f6f4_idesc(uint32_t n, uint8_t a_type, uint8_t b_type,
                             uint8_t a_scale_factor_id = 0,
                             uint8_t b_scale_factor_id = 0, uint32_t m = 128,
                             bool negate_a = false, bool negate_b = false,
                             bool transpose_a = false,
                             bool transpose_b = false) {
  return (static_cast<uint32_t>(b_scale_factor_id) << 4) |
         (static_cast<uint32_t>(a_type) << 7) |
         (static_cast<uint32_t>(b_type) << 10) |
         (static_cast<uint32_t>(negate_a) << 13) |
         (static_cast<uint32_t>(negate_b) << 14) |
         (static_cast<uint32_t>(transpose_a) << 15) |
         (static_cast<uint32_t>(transpose_b) << 16) | ((n >> 3) << 17) |
         (TCGEN05_SCALE_FORMAT_UE8M0 << 23) | ((m >> 7) << 27) |
         (static_cast<uint32_t>(a_scale_factor_id) << 29);
}

}  // namespace

TEST(Tcgen05TmemTest, Mxf4ScaleReadSelectsSubpartitionAndByteId) {
  tcgen05_tmem_manager_t manager;
  uint32_t allocation = manager.alloc(kScope0, 512);
  uint32_t scale_base = allocation + 64;
  std::vector<uint32_t> words(128 * 4, 0);
  for (uint32_t data_path = 0; data_path < 128; ++data_path) {
    for (uint32_t column = 0; column < 4; ++column) {
      uint32_t byte0 = 1 + data_path % 32;
      uint32_t byte1 = 40 + column;
      uint32_t byte2 = 80 + data_path % 32;
      uint32_t byte3 = 120 + column;
      words[data_path * 4 + column] =
          byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24);
    }
  }
  manager.write_matrix_words(kScope0, scale_base, words, 128, 4);

  std::vector<uint8_t> low = manager.read_mxf4_scale_matrix(
      kScope0, scale_base, 128, 2, /*scale_factor_id=*/0);
  // The instruction descriptor, not the TMEM address, selects the byte ID.
  // DeepGEMM therefore reuses the same address for consecutive K slices.
  std::vector<uint8_t> high = manager.read_mxf4_scale_matrix(
      kScope0, scale_base, 128, 2, /*scale_factor_id=*/2);

  EXPECT_EQ(low[0], 1u);
  EXPECT_EQ(low[1], 40u);
  EXPECT_EQ(low[32 * 2], 1u);
  EXPECT_EQ(low[32 * 2 + 1], 41u);
  EXPECT_EQ(low[127 * 2], 32u);
  EXPECT_EQ(low[127 * 2 + 1], 43u);
  EXPECT_EQ(high[0], 80u);
  EXPECT_EQ(high[1], 120u);
  EXPECT_EQ(high[32 * 2], 80u);
  EXPECT_EQ(high[32 * 2 + 1], 121u);
  EXPECT_EQ(high[127 * 2], 111u);
  EXPECT_EQ(high[127 * 2 + 1], 123u);
}

TEST(Tcgen05TmemTest, ClearCtaLeavesOtherCtasAndThePeerSmIntact) {
  tcgen05_tmem_manager_t manager;
  uint32_t scope0_base = manager.alloc(kScope0, 32);
  uint32_t peer_base = manager.alloc(kScopePeer, 32);
  uint32_t scope1_base = manager.alloc(kScope1, 32);

  manager.clear_cta(/*sm_id=*/0, /*cta_id=*/0);

  EXPECT_FALSE(manager.has_allocation(kScope0, scope0_base));
  EXPECT_TRUE(manager.has_allocation(kScope1, scope1_base));
  EXPECT_TRUE(manager.has_allocation(kScopePeer, peer_base));
}

TEST(Tcgen05TmemTest, DecodeInstructionDescriptorMxf4Block32) {
  tcgen05_mma_descriptor_t desc =
      tcgen05_decode_mxf4_mma_descriptor(make_mxf4_idesc(64, 2, 2), 1);

  EXPECT_FALSE(desc.sparse);
  EXPECT_EQ(desc.a_type, TCGEN05_MXF4_FORMAT_E2M1);
  EXPECT_EQ(desc.b_type, TCGEN05_MXF4_FORMAT_E2M1);
  EXPECT_EQ(desc.scale_format, TCGEN05_SCALE_FORMAT_UE8M0);
  EXPECT_EQ(desc.a_scale_factor_id, 2u);
  EXPECT_EQ(desc.b_scale_factor_id, 2u);
  EXPECT_EQ(desc.m, 128u);
  EXPECT_EQ(desc.n, 64u);
  EXPECT_EQ(desc.k, 64u);
}

TEST(Tcgen05TmemTest, Mxf4DenseShapesMatchPtxIsaTable) {
  for (uint32_t n = 8; n <= 256; n += 8) {
    EXPECT_TRUE(tcgen05_mxf4_dense_shape_supported(128, n, 64, 1));
    tcgen05_mma_descriptor_t desc =
        tcgen05_decode_mxf4_mma_descriptor(make_mxf4_idesc(n), /*cta_group=*/1);
    EXPECT_EQ(desc.m, 128u);
    EXPECT_EQ(desc.n, n);
    EXPECT_EQ(desc.k, 64u);
  }

  for (uint32_t n = 16; n <= 256; n += 16) {
    for (uint32_t m : {128u, 256u}) {
      EXPECT_TRUE(tcgen05_mxf4_dense_shape_supported(m, n, 64, 2));
      tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf4_mma_descriptor(
          make_mxf4_idesc(n, 0, 0, m, 64), /*cta_group=*/2);
      EXPECT_EQ(desc.m, m);
      EXPECT_EQ(desc.n, n);
      EXPECT_EQ(desc.k, 64u);
    }

    EXPECT_TRUE(tcgen05_mxf4_dense_shape_supported(256, n, 96, 2));
    for (uint8_t scale_factor_id = 0; scale_factor_id < 4; ++scale_factor_id) {
      tcgen05_mma_descriptor_t k96 = tcgen05_decode_mxf4_mma_descriptor(
          make_mxf4_idesc(n, scale_factor_id, scale_factor_id, 256, 96),
          /*cta_group=*/2);
      EXPECT_EQ(k96.m, 256u);
      EXPECT_EQ(k96.n, n);
      EXPECT_EQ(k96.k, 96u);
      EXPECT_EQ(k96.a_scale_factor_id, scale_factor_id);
      EXPECT_EQ(k96.b_scale_factor_id, scale_factor_id);
    }
  }

  EXPECT_FALSE(tcgen05_mxf4_dense_shape_supported(256, 8, 64, 1));
  EXPECT_FALSE(tcgen05_mxf4_dense_shape_supported(128, 8, 64, 2));
  EXPECT_FALSE(tcgen05_mxf4_dense_shape_supported(128, 16, 96, 2));
}

TEST(Tcgen05TmemTest, Mxf4DescriptorPreservesNegateBits) {
  tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf4_mma_descriptor(
      make_mxf4_idesc(8, 0, 0, 128, 64, true, false), 1);
  EXPECT_TRUE(desc.negate_a);
  EXPECT_FALSE(desc.negate_b);
}

TEST(Tcgen05TmemTest, DecodeInstructionDescriptorDeepGemmW4A8) {
  tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
      make_mxf8f6f4_idesc(64, TCGEN05_MXF8F6F4_FORMAT_E4M3,
                          TCGEN05_MXF8F6F4_FORMAT_E2M1,
                          /*a_scale_factor_id=*/3, /*b_scale_factor_id=*/2),
      /*cta_group=*/1);

  EXPECT_EQ(desc.a_type, TCGEN05_MXF8F6F4_FORMAT_E4M3);
  EXPECT_EQ(desc.b_type, TCGEN05_MXF8F6F4_FORMAT_E2M1);
  EXPECT_EQ(desc.scale_format, TCGEN05_SCALE_FORMAT_UE8M0);
  EXPECT_EQ(desc.a_scale_factor_id, 3u);
  EXPECT_EQ(desc.b_scale_factor_id, 2u);
  EXPECT_EQ(desc.m, 128u);
  EXPECT_EQ(desc.n, 64u);
  EXPECT_EQ(desc.k, 32u);
}

TEST(Tcgen05TmemTest, Mxf8f6f4SupportsAllOrderedFormatPairs) {
  const uint8_t formats[] = {
      TCGEN05_MXF8F6F4_FORMAT_E4M3, TCGEN05_MXF8F6F4_FORMAT_E5M2,
      TCGEN05_MXF8F6F4_FORMAT_E2M3, TCGEN05_MXF8F6F4_FORMAT_E3M2,
      TCGEN05_MXF8F6F4_FORMAT_E2M1};

  for (uint8_t a_type : formats) {
    for (uint8_t b_type : formats) {
      tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
          make_mxf8f6f4_idesc(8, a_type, b_type), /*cta_group=*/1);
      EXPECT_EQ(desc.a_type, a_type);
      EXPECT_EQ(desc.b_type, b_type);
    }
  }
}

TEST(Tcgen05TmemTest, Mxf8f6f4DenseShapesMatchPtxIsaTable) {
  for (uint32_t n = 8; n <= 256; n += 8) {
    EXPECT_TRUE(tcgen05_mxf8f6f4_dense_shape_supported(128, n, 32, 1));
  }
  for (uint32_t n = 16; n <= 256; n += 16) {
    for (uint32_t m : {128u, 256u}) {
      EXPECT_TRUE(tcgen05_mxf8f6f4_dense_shape_supported(m, n, 32, 2));
      tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
          make_mxf8f6f4_idesc(n, TCGEN05_MXF8F6F4_FORMAT_E4M3,
                              TCGEN05_MXF8F6F4_FORMAT_E2M1, 0, 0, m),
          /*cta_group=*/2);
      EXPECT_EQ(desc.m, m);
      EXPECT_EQ(desc.n, n);
      EXPECT_EQ(desc.k, 32u);
    }
  }

  EXPECT_FALSE(tcgen05_mxf8f6f4_dense_shape_supported(256, 8, 32, 1));
  EXPECT_FALSE(tcgen05_mxf8f6f4_dense_shape_supported(128, 8, 32, 2));
  EXPECT_FALSE(tcgen05_mxf8f6f4_dense_shape_supported(128, 16, 64, 2));
}

TEST(Tcgen05TmemTest, Mxf8f6f4SharedLocationsMatchPaddedTmaContainers) {
  auto fp4_0 =
      tcgen05_mxf8f6f4_shared_location(0, TCGEN05_MXF8F6F4_FORMAT_E2M1);
  auto fp4_1 =
      tcgen05_mxf8f6f4_shared_location(1, TCGEN05_MXF8F6F4_FORMAT_E2M1);
  auto fp4_15 =
      tcgen05_mxf8f6f4_shared_location(15, TCGEN05_MXF8F6F4_FORMAT_E2M1);
  auto fp4_16 =
      tcgen05_mxf8f6f4_shared_location(16, TCGEN05_MXF8F6F4_FORMAT_E2M1);
  EXPECT_EQ(fp4_0.byte_offset, 0u);
  EXPECT_EQ(fp4_0.bit_offset, 0u);
  EXPECT_EQ(fp4_1.byte_offset, 0u);
  EXPECT_EQ(fp4_1.bit_offset, 4u);
  EXPECT_EQ(fp4_15.byte_offset, 7u);
  EXPECT_EQ(fp4_15.bit_offset, 4u);
  EXPECT_EQ(fp4_16.byte_offset, 16u);
  EXPECT_EQ(fp4_16.bit_offset, 0u);

  auto fp6_3 =
      tcgen05_mxf8f6f4_shared_location(3, TCGEN05_MXF8F6F4_FORMAT_E2M3);
  auto fp6_15 =
      tcgen05_mxf8f6f4_shared_location(15, TCGEN05_MXF8F6F4_FORMAT_E3M2);
  auto fp6_16 =
      tcgen05_mxf8f6f4_shared_location(16, TCGEN05_MXF8F6F4_FORMAT_E2M3);
  EXPECT_EQ(fp6_3.byte_offset, 2u);
  EXPECT_EQ(fp6_3.bit_offset, 2u);
  EXPECT_EQ(fp6_15.byte_offset, 11u);
  EXPECT_EQ(fp6_15.bit_offset, 2u);
  EXPECT_EQ(fp6_16.byte_offset, 16u);
  EXPECT_EQ(fp6_16.bit_offset, 0u);

  auto fp8_15 =
      tcgen05_mxf8f6f4_shared_location(15, TCGEN05_MXF8F6F4_FORMAT_E4M3);
  auto fp8_16 =
      tcgen05_mxf8f6f4_shared_location(16, TCGEN05_MXF8F6F4_FORMAT_E5M2);
  EXPECT_EQ(fp8_15.byte_offset, 15u);
  EXPECT_EQ(fp8_15.bit_offset, 0u);
  EXPECT_EQ(fp8_16.byte_offset, 16u);
  EXPECT_EQ(fp8_16.bit_offset, 0u);
}

TEST(Tcgen05TmemTest, Mxf4KMajorPackedAddressLinearFallback) {
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x100, /*lbo=*/0, /*sbo=*/0));

  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32), 0x100u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 2, 32), 0x122u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 127, 31, 32),
            0x10ffu);
}

TEST(Tcgen05TmemTest, Mxf4KMajorPackedAddressApplies128ByteSwizzle) {
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x400, /*lbo=*/0x10, /*sbo=*/0x400,
                       /*swizzle=*/2));

  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32), 0x400u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 0, 32), 0x490u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 7, 0, 32), 0x7f0u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 8, 0, 32), 0x800u);
}

TEST(Tcgen05TmemTest, Bf16KMajorAddressUsesElementBytesAndKSliceBase) {
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x8800, /*lbo=*/0x10, /*sbo=*/0x400,
                       /*swizzle=*/2));

  // A kind::f16 MMA consumes K=16 BF16 values (32 bytes) per logical row.
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32),
            0x8800u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 16, 32),
            0x8810u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 0, 32),
            0x8890u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 7, 30, 32),
            0x8beeu);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 8, 0, 32),
            0x8c00u);

  // CUTLASS advances the matrix descriptor by 32 bytes for the next K=16
  // slice. Keep that non-period-aligned base in the swizzle coordinate.
  desc.start_address += 32;
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32),
            0x8820u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 0, 32),
            0x88b0u);
}

TEST(Tcgen05TmemTest, SwizzledKMajorIgnoresRelativeLeadingField) {
  // The captured cuBLAS descriptor has LBO=8192. Applying it to the second
  // 16-byte K segment reads the other operand instead of adjacent K values.
  for (unsigned mode : {2u, 4u, 6u}) {
    auto desc = tcgen05_decode_shared_descriptor(
        make_shared_desc(0x400, 0x2000, 0x400, mode));
    EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 16, 32),
              0x410u);
    EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 30, 32),
              0x41eu);
    EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 8, 16, 32),
              0x810u);
    desc.leading_dimension_byte_offset = 0;
    EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 16, 32),
              0x410u);
  }
}

TEST(Tcgen05TmemTest, Bf16MnMajorAddressMatchesPtxCanonicalNoSwizzleLayout) {
  // PTX 9.3 Figure 190 uses T=8 for BF16.  In normalized coordinates the
  // first eight MN elements are contiguous, K advances in 128-bit columns,
  // SBO advances to the next MN group, and LBO advances to the next K group.
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x200, /*lbo=*/0x100, /*sbo=*/0x80));

  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 0, 2, 16), 0x200u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 7, 0, 2, 16), 0x20eu);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 8, 0, 2, 16), 0x280u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 1, 2, 16), 0x210u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 8, 2, 16), 0x300u);
}

TEST(Tcgen05TmemTest, Bf16MnMajorAddressApplies128ByteSwizzle) {
  // This is the descriptor shape emitted by the FlashInfer MLA QK mainloop:
  // 64 BF16 MN elements per swizzle group, eight K elements per group, then
  // LBO/SBO advance the outer MN/K coordinates.
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x1000, /*lbo=*/0x1000, /*sbo=*/0x400,
                       /*swizzle=*/2));

  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 0, 2, 128), 0x1000u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 1, 0, 2, 128), 0x1002u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 63, 0, 2, 128), 0x107eu);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 64, 0, 2, 128), 0x2000u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 1, 2, 128), 0x1090u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 7, 2, 128), 0x13f0u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 0, 8, 2, 128), 0x1400u);
  EXPECT_EQ(tcgen05_shared_mn_major_byte_address(desc, 63, 15, 2, 128),
            0x178eu);
}

TEST(Tcgen05TmemTest, Mxf4NumericFormats) {
  const float expected_e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                  2.0f, 3.0f, 4.0f, 6.0f};
  for (uint8_t value = 0; value < 8; ++value) {
    EXPECT_FLOAT_EQ(tcgen05_e2m1_to_f32(value), expected_e2m1[value]);
    EXPECT_FLOAT_EQ(tcgen05_e2m1_to_f32(value | 0x8), -expected_e2m1[value]);
  }
  EXPECT_FLOAT_EQ(tcgen05_ue8m0_to_f32(127), 1.0f);
  EXPECT_FLOAT_EQ(tcgen05_ue8m0_to_f32(128), 2.0f);
  EXPECT_FLOAT_EQ(tcgen05_ue8m0_to_f32(126), 0.5f);
  EXPECT_TRUE(std::isnan(tcgen05_ue8m0_to_f32(0xff)));
}

TEST(Tcgen05TmemTest, Mxf8f6f4NumericFormats) {
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x38, TCGEN05_MXF8F6F4_FORMAT_E4M3),
                  1.0f);
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x7e, TCGEN05_MXF8F6F4_FORMAT_E4M3),
                  448.0f);
  EXPECT_TRUE(
      std::isnan(tcgen05_mxf8f6f4_to_f32(0x7f, TCGEN05_MXF8F6F4_FORMAT_E4M3)));

  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x3c, TCGEN05_MXF8F6F4_FORMAT_E5M2),
                  1.0f);
  EXPECT_TRUE(
      std::isinf(tcgen05_mxf8f6f4_to_f32(0x7c, TCGEN05_MXF8F6F4_FORMAT_E5M2)));
  EXPECT_TRUE(
      std::isnan(tcgen05_mxf8f6f4_to_f32(0x7d, TCGEN05_MXF8F6F4_FORMAT_E5M2)));

  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x08, TCGEN05_MXF8F6F4_FORMAT_E2M3),
                  1.0f);
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x1f, TCGEN05_MXF8F6F4_FORMAT_E2M3),
                  7.5f);
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x0c, TCGEN05_MXF8F6F4_FORMAT_E3M2),
                  1.0f);
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x1f, TCGEN05_MXF8F6F4_FORMAT_E3M2),
                  28.0f);
  EXPECT_FLOAT_EQ(tcgen05_mxf8f6f4_to_f32(0x02, TCGEN05_MXF8F6F4_FORMAT_E2M1),
                  1.0f);
}

TEST(Tcgen05TmemTest, MmaMxf4Block32KnownScalesAndAccum) {
  tcgen05_mma_descriptor_t desc =
      tcgen05_decode_mxf4_mma_descriptor(make_mxf4_idesc(8), 1);
  std::vector<uint8_t> a(desc.m * desc.k, 0x2);  // E2M1 1.0
  // B is K-major for each output column.
  std::vector<uint8_t> b(desc.n * desc.k, 0x2);
  std::vector<uint8_t> scale_a(desc.m * 2);
  std::vector<uint8_t> scale_b(desc.n * 2);
  for (uint32_t row = 0; row < desc.m; ++row) {
    scale_a[row * 2] = 127;      // 1.0 for k=[0,32)
    scale_a[row * 2 + 1] = 128;  // 2.0 for k=[32,64)
  }
  for (uint32_t col = 0; col < desc.n; ++col) {
    scale_b[col * 2] = 127;      // 1.0 for k=[0,32)
    scale_b[col * 2 + 1] = 126;  // 0.5 for k=[32,64)
  }
  std::vector<uint32_t> input_d(desc.m * desc.n, tcgen05_f32_to_bits(2.5f));

  std::vector<uint32_t> result = tcgen05_mma_mxf4_compute_words(
      desc, a, b, scale_a, scale_b, 32, input_d, true);

  ASSERT_EQ(result.size(), desc.m * desc.n);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.front()), 66.5f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.back()), 66.5f);
}

TEST(Tcgen05TmemTest, MmaMxf4AppliesDescriptorNegation) {
  tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf4_mma_descriptor(
      make_mxf4_idesc(8, 0, 0, 128, 64, true, false), 1);
  std::vector<uint8_t> a(desc.m * desc.k, 0x2);
  std::vector<uint8_t> b(desc.n * desc.k, 0x2);
  std::vector<uint8_t> scale_a(desc.m * 2, 127);
  std::vector<uint8_t> scale_b(desc.n * 2, 127);

  std::vector<uint32_t> result = tcgen05_mma_mxf4_compute_words(
      desc, a, b, scale_a, scale_b, 32, {}, false);

  ASSERT_EQ(result.size(), desc.m * desc.n);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.front()), -64.0f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.back()), -64.0f);
}

TEST(Tcgen05TmemTest, MmaMxf8f6f4DeepGemmW4A8KnownScalesAndAccum) {
  tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
      make_mxf8f6f4_idesc(8, TCGEN05_MXF8F6F4_FORMAT_E4M3,
                          TCGEN05_MXF8F6F4_FORMAT_E2M1),
      /*cta_group=*/1);
  std::vector<uint8_t> a(desc.m * desc.k, 0x38);  // E4M3 1.0
  std::vector<uint8_t> b(desc.n * desc.k, 0x02);  // E2M1 1.0
  std::vector<uint8_t> scale_a(desc.m, 128);      // 2.0
  std::vector<uint8_t> scale_b(desc.n, 126);      // 0.5
  std::vector<uint32_t> input_d(desc.m * desc.n, tcgen05_f32_to_bits(2.5f));

  std::vector<uint32_t> result = tcgen05_mma_mxf8f6f4_compute_words(
      desc, a, b, scale_a, scale_b, input_d, true);

  ASSERT_EQ(result.size(), desc.m * desc.n);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.front()), 34.5f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.back()), 34.5f);
}

TEST(Tcgen05TmemTest, MmaMxf8f6f4ComputesAllOrderedFormatPairs) {
  const uint8_t formats[] = {
      TCGEN05_MXF8F6F4_FORMAT_E4M3, TCGEN05_MXF8F6F4_FORMAT_E5M2,
      TCGEN05_MXF8F6F4_FORMAT_E2M3, TCGEN05_MXF8F6F4_FORMAT_E3M2,
      TCGEN05_MXF8F6F4_FORMAT_E2M1};
  const uint8_t one[] = {0x38, 0x3c, 0x08, 0x0c, 0x02};

  for (unsigned a_format = 0; a_format < 5; ++a_format) {
    for (unsigned b_format = 0; b_format < 5; ++b_format) {
      tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
          make_mxf8f6f4_idesc(8, formats[a_format], formats[b_format]),
          /*cta_group=*/1);
      std::vector<uint8_t> a(desc.m * desc.k, one[a_format]);
      std::vector<uint8_t> b(desc.n * desc.k, one[b_format]);
      std::vector<uint8_t> scale_a(desc.m, 127);
      std::vector<uint8_t> scale_b(desc.n, 127);

      std::vector<uint32_t> result = tcgen05_mma_mxf8f6f4_compute_words(
          desc, a, b, scale_a, scale_b, {}, false);
      ASSERT_EQ(result.size(), desc.m * desc.n);
      EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.front()), 32.0f);
      EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.back()), 32.0f);
    }
  }
}

TEST(Tcgen05F16MmaTest, F16DestinationUsesHalfEncodingAndRoundsTiesToEven) {
  const auto desc =
      tcgen05_decode_f16_mma_descriptor((4u << 24) | (1u << 17), 1);
  std::vector<uint16_t> a(desc.m * desc.k, 0x3c00);  // 1
  std::vector<uint16_t> b(desc.n * desc.k, 0x0200);  // 2^-15; K16 sum = 2^-11
  std::vector<uint32_t> d(desc.m * desc.n);
  for (unsigned i = 0; i < d.size(); ++i) d[i] = (i & 1) ? 0x3c01 : 0x3c00;
  const auto out = tcgen05_mma_f16_compute_words(desc, a, b, d, true);
  for (unsigned i = 0; i < out.size(); ++i)
    EXPECT_EQ(out[i], (i & 1) ? 0x3c02u : 0x3c00u);
  const auto zero = tcgen05_mma_f16_compute_words(desc, a, b, d, false);
  for (auto v : zero)
    EXPECT_EQ(v, 0x1000u);  // no stale D when input-D is disabled
}

TEST(Tcgen05F16MmaTest, FloatToHalfPreservesSubnormalsAndSpecialValues) {
  EXPECT_EQ(tcgen05_f32_to_f16(1.00048828125f), 0x3c00);
  EXPECT_EQ(tcgen05_f32_to_f16(1.00146484375f), 0x3c02);
  EXPECT_EQ(tcgen05_f32_to_f16(std::ldexp(1.0f, -24)), 1);
  EXPECT_EQ(tcgen05_f32_to_f16(std::ldexp(1.0f, -25)), 0);
  EXPECT_EQ(tcgen05_f32_to_f16(-0.0f), 0x8000);
  EXPECT_EQ(tcgen05_f32_to_f16(65520.0f), 0x7c00);
  EXPECT_EQ(tcgen05_f32_to_f16(INFINITY), 0x7c00);
  EXPECT_EQ(tcgen05_f32_to_f16(-INFINITY), 0xfc00);
  EXPECT_EQ(tcgen05_f32_to_f16(NAN) & 0x7e00, 0x7e00);
}

TEST(Tcgen05TmemTest, LogicalAccumulatorsKeepUnselectedDatapaths) {
  for (bool folded : {false, true}) {
    tcgen05_tmem_manager_t manager;
    const tcgen05_tmem_scope_t scope{3, 7};
    const unsigned base = manager.alloc(scope, 32);
    manager.write_matrix_words(
        scope, base, std::vector<uint32_t>(128 * 32, 0xfeed), 128, 32);
    std::vector<uint32_t> values(64 * 16);
    for (unsigned i = 0; i < values.size(); ++i) values[i] = i + 1;
    manager.write_accumulator(scope, base, values, 64, 16, folded);
    EXPECT_EQ(manager.read_accumulator(scope, base, 64, 16, folded), values);
    // Check selected physical positions, not just inverse mapping round trips.
    EXPECT_EQ(manager.read_words(scope, base + (32u << 16), 1)[0],
              folded ? 513u : 257u);
    if (folded) {
      EXPECT_EQ(manager.read_words(scope, base + (64u << 16), 1)[0], 9u);
      EXPECT_EQ(manager.read_words(scope, base + 8, 1)[0], 0xfeedu);
    } else {
      EXPECT_EQ(manager.read_words(scope, base + (16u << 16), 1)[0], 0xfeedu);
      EXPECT_EQ(manager.read_words(scope, base + 16, 1)[0], 0xfeedu);
    }
  }
}

TEST(Tcgen05Tmem, DensePhysicalAImagesShareNumericsAcrossLayouts) {
  for (unsigned group : {1u, 2u})
    for (bool folded : {false, true})
      for (bool bf16 : {false, true})
        for (bool transpose : {false, true}) {
          const unsigned m = group * (folded ? 64 : 128), n = 64;
          auto desc = tcgen05_decode_f16_mma_descriptor(
              make_f16_idesc(m, n, transpose) |
                  (bf16 ? (1u << 7) | (1u << 10) : 0),
              group);
          auto encode = [&](float value) -> uint16_t {
            return bf16 ? tcgen05_f32_to_bits(value) >> 16
                        : tcgen05_f32_to_f16(value);
          };
          std::vector<std::vector<uint16_t>> a(group,
                                               std::vector<uint16_t>(128 * 16));
          for (unsigned rank = 0; rank < group; ++rank)
            for (unsigned row = 0; row < 128; ++row)
              for (unsigned k = 0; k < 16; ++k)
                a[rank][row * 16 + k] = encode(float(
                    (rank + 1) * (row / 64 + 1) * (row % 3 + 1) * (k % 2 + 1)));
          std::vector<uint16_t> b(n * 16);
          for (unsigned col = 0; col < n; ++col)
            for (unsigned k = 0; k < 16; ++k)
              b[transpose ? k * n + col : col * 16 + k] =
                  encode(float((col % 5 + 1) * (k % 3 + 1)));
          std::vector<uint32_t> input(m * n);
          for (unsigned i = 0; i < input.size(); ++i)
            input[i] = tcgen05_f32_to_bits(float(i % 7));
          for (bool accumulate : {false, true}) {
            const auto out = tcgen05_mma_f16_tmem_a_compute_words(
                desc, a, b, input, accumulate, folded);
            for (unsigned row = 0; row < m; ++row)
              for (unsigned col = 0; col < n; ++col) {
                const unsigned local_m = m / group, rank = row / local_m;
                const unsigned physical =
                    row % local_m + (folded && col >= n / 2 ? 64 : 0);
                float expected = accumulate ? float((row * n + col) % 7) : 0;
                for (unsigned k = 0; k < 16; ++k)
                  expected += float((rank + 1) * (physical / 64 + 1) *
                                    (physical % 3 + 1) * (k % 2 + 1) *
                                    (col % 5 + 1) * (k % 3 + 1));
                ASSERT_EQ(tcgen05_bits_to_f32(out[row * n + col]), expected)
                    << group << "," << folded << "," << bf16 << "," << transpose
                    << "," << row << "," << col;
              }
          }
        }
}

TEST(Tcgen05Mma, DenseDescriptorValidationIsSharedAcrossFrontends) {
  for (unsigned n : {64u, 128u, 256u}) {
    const auto raw = make_f16_idesc(64, n);
    EXPECT_TRUE(tcgen05_f16_descriptor_supported(raw, 1, true));
    EXPECT_FALSE(tcgen05_f16_descriptor_supported(raw, 2, true));
    for (unsigned bit : {0u, 1u, 2u, 3u, 6u, 23u, 29u, 30u, 31u})
      EXPECT_FALSE(
          tcgen05_f16_descriptor_supported(raw | (1u << bit), 1, true));
    for (unsigned modifiers = 0; modifiers < 16; ++modifiers)
      EXPECT_TRUE(
          tcgen05_f16_descriptor_supported(raw | (modifiers << 13), 1, true));
  }
  EXPECT_FALSE(
      tcgen05_f16_descriptor_supported(make_f16_idesc(64, 32), 1, true));
  EXPECT_FALSE(
      tcgen05_f16_descriptor_supported(make_f16_idesc(128, 64), 1, true));
  EXPECT_TRUE(tcgen05_f16_descriptor_supported(make_f16_idesc(128, 16), 1));
  EXPECT_FALSE(tcgen05_f16_descriptor_supported(make_f16_idesc(128, 16), 0));
  EXPECT_FALSE(
      tcgen05_f16_descriptor_supported(make_f16_idesc(128, 16) | (2u << 7), 1));
}

TEST(Tcgen05Mma, DenseTmemDatapathAlignment) {
  for (unsigned lane : {0u, 16u}) {
    const auto base = lane << 16;
    EXPECT_TRUE(tcgen05_mma_tmem_alignment_supported(64, false, base, true,
                                                     base + 128));
    EXPECT_TRUE(
        tcgen05_mma_tmem_alignment_supported(64, false, base, false, 0));
    EXPECT_FALSE(tcgen05_mma_tmem_alignment_supported(
        64, false, base, true, ((16 - lane) << 16) + 128));
    EXPECT_FALSE(tcgen05_mma_tmem_alignment_supported(64, false, base, true,
                                                      base + 129));
  }
  EXPECT_FALSE(
      tcgen05_mma_tmem_alignment_supported(64, false, 1u << 16, false, 0));
  EXPECT_FALSE(
      tcgen05_mma_tmem_alignment_supported(128, false, 16u << 16, false, 0));
  EXPECT_FALSE(
      tcgen05_mma_tmem_alignment_supported(64, true, 16u << 16, false, 0));
}

TEST(Tcgen05Mma, SharedDescriptorGateIsCommonAndLayoutAware) {
  for (unsigned mode : {0u, 2u, 4u, 6u}) {
    const auto raw = make_shared_desc(0x400, 16, 1024, mode);
    EXPECT_TRUE(tcgen05_mma_shared_descriptor_supported(raw));
    EXPECT_TRUE(tcgen05_mma_shared_descriptor_supported(raw, true));
    for (unsigned bit : {14u, 15u, 30u, 31u, 47u, 48u, 52u, 53u, 54u, 60u})
      EXPECT_FALSE(
          tcgen05_mma_shared_descriptor_supported(raw | (1ull << bit), true));
    EXPECT_FALSE(
        tcgen05_mma_shared_descriptor_supported(raw & ~(1ull << 46), true));
  }
  const auto atom32 = make_shared_desc(0x400, 2048, 512, 1);
  EXPECT_FALSE(tcgen05_mma_shared_descriptor_supported(atom32));
  EXPECT_TRUE(tcgen05_mma_shared_descriptor_supported(atom32, true));
  for (unsigned mode : {3u, 5u, 7u})
    EXPECT_FALSE(tcgen05_mma_shared_descriptor_supported(
        make_shared_desc(0x400, 16, 1024, mode), true));
}
TEST(Tcgen05Mma, MnMajor32ByteAtomsKeepAdjacentHalvesTogether) {
  auto d = tcgen05_decode_mma_shared_descriptor(
      make_shared_desc(0x400, 2048, 512, 1), true);
  auto address = [&](unsigned row, unsigned k) {
    return tcgen05_shared_mn_major_byte_address(d, row, k, 2, 128);
  };
  EXPECT_EQ(address(0, 0), 0x400u);
  EXPECT_EQ(address(0, 1), 0x4a0u);
  EXPECT_EQ(address(8, 1), 0x4b0u);
  EXPECT_EQ(address(16, 0), 0x420u);
  EXPECT_EQ(address(32, 1), 0x4e0u);
  EXPECT_EQ(address(0, 4), 0x600u);
  EXPECT_EQ(address(64, 0), 0xc00u);
  d.start_address += 128;
  EXPECT_EQ(address(0, 0), 0x4a0u);
}

TEST(Tcgen05Mma, Tf32DescriptorAcceptsDenseKMajorAndRejectsOtherForms) {
  for (unsigned m : {64u, 128u})
    for (unsigned n : {8u, 64u, 256u}) {
      const uint32_t raw =
          ((m / 16) << 24) | ((n / 8) << 17) | 0x10 | (2u << 7) | (2u << 10);
      EXPECT_TRUE(tcgen05_tf32_descriptor_supported(raw));
      EXPECT_TRUE(tcgen05_tf32_descriptor_supported(raw | (3u << 13)));
      for (unsigned bit : {0u, 2u, 3u, 6u, 15u, 16u, 23u, 29u, 30u, 31u})
        EXPECT_FALSE(tcgen05_tf32_descriptor_supported(raw | (1u << bit)))
            << bit;
      EXPECT_FALSE(tcgen05_tf32_descriptor_supported(raw & ~(3u << 4)));
      EXPECT_FALSE(tcgen05_tf32_descriptor_supported(raw & ~(7u << 7)));
    }
  EXPECT_FALSE(
      tcgen05_tf32_descriptor_supported((16u << 24) | (8u << 17) | 0x910));
}
