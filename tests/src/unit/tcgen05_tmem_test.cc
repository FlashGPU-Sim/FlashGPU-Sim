#include <gtest/gtest.h>

#include "gpgpu-sim/flash/tcgen05.h"

#include <cmath>
#include <cstring>

using namespace flash_gpgpu_sim;

namespace {

constexpr tcgen05_tmem_scope_t kScope0{0, 0, 1};
constexpr tcgen05_tmem_scope_t kScope1{0, 1, 1};
constexpr tcgen05_tmem_scope_t kScopeGroup2{0, 0, 2};

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
         (TCGEN05_MXF4_FORMAT_E2M1 << 7) |
         (TCGEN05_MXF4_FORMAT_E2M1 << 10) |
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

TEST(Tcgen05TmemTest, AllocSingleCTA) {
  tcgen05_tmem_manager_t manager;

  uint32_t base = manager.alloc(kScope0, 64);

  EXPECT_TRUE(manager.has_allocation(kScope0, base));
  EXPECT_EQ(manager.allocation_count(kScope0), 1u);
  EXPECT_TRUE(manager.contains_range(kScope0, base, 64));
  EXPECT_FALSE(manager.contains_range(kScope0, base + 63, 2));
}

TEST(Tcgen05TmemTest, AllocNoOverlap) {
  tcgen05_tmem_manager_t manager;

  uint32_t first = manager.alloc(kScope0, 64);
  uint32_t second = manager.alloc(kScope0, 32);

  EXPECT_GE(second, first + 64);
  EXPECT_TRUE(manager.contains_range(kScope0, first, 64));
  EXPECT_TRUE(manager.contains_range(kScope0, second, 32));
  EXPECT_FALSE(manager.contains_range(kScope0, first + 60, 8));
}

TEST(Tcgen05TmemTest, DeallocReleasesRange) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);

  manager.write_words(kScope0, base, sequence(0x10, 8));
  manager.dealloc(kScope0, base, 32);

  EXPECT_FALSE(manager.has_allocation(kScope0, base));
  EXPECT_FALSE(manager.contains_range(kScope0, base, 1));
  EXPECT_EQ(manager.allocation_count(kScope0), 0u);
}

TEST(Tcgen05TmemTest, RetiredAllocationRemainsAccessibleUntilCtaClear) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);

  manager.write_words(kScope0, base, sequence(0x20, 8));
  manager.dealloc(kScope0, base, 32);
  manager.write_words(kScope0, base + 2, {0xaa, 0xbb});

  EXPECT_FALSE(manager.has_allocation(kScope0, base));
  EXPECT_FALSE(manager.contains_range(kScope0, base, 1));
  EXPECT_EQ(manager.read_words(kScope0, base, 4),
            std::vector<uint32_t>({0x20, 0x21, 0xaa, 0xbb}));
}

TEST(Tcgen05TmemTest, RelinquishPermitIsIdempotentForNow) {
  tcgen05_tmem_manager_t manager;

  EXPECT_FALSE(manager.permit_relinquished(kScope0));
  manager.relinquish_alloc_permit(kScope0);
  manager.relinquish_alloc_permit(kScope0);

  EXPECT_TRUE(manager.permit_relinquished(kScope0));
}

TEST(Tcgen05TmemTest, AllocRejectsColumnsBelowMinimum) {
  tcgen05_tmem_manager_t manager;

  EXPECT_DEATH(manager.alloc(kScope0, 16), "ncols must be in");
}

TEST(Tcgen05TmemTest, AllocRejectsColumnsAboveMaximum) {
  tcgen05_tmem_manager_t manager;

  EXPECT_DEATH(manager.alloc(kScope0, 1024), "ncols must be in");
}

TEST(Tcgen05TmemTest, AllocRejectsNonPowerOfTwoColumns) {
  tcgen05_tmem_manager_t manager;

  EXPECT_DEATH(manager.alloc(kScope0, 48), "power of two");
}

TEST(Tcgen05TmemTest, AllocRejectsIncreasingSizeWithinCta) {
  tcgen05_tmem_manager_t manager;

  uint32_t base = manager.alloc(kScope0, 64);
  EXPECT_TRUE(manager.has_allocation(kScope0, base));
  EXPECT_DEATH(manager.alloc(kScope0, 128), "must not increase");
}

TEST(Tcgen05TmemTest, AllocAllowsEqualAndDecreasingSizesWithinCta) {
  tcgen05_tmem_manager_t manager;

  uint32_t first = manager.alloc(kScope0, 64);
  uint32_t second = manager.alloc(kScope0, 64);
  uint32_t third = manager.alloc(kScope0, 32);

  EXPECT_TRUE(manager.contains_range(kScope0, first, 64));
  EXPECT_TRUE(manager.contains_range(kScope0, second, 64));
  EXPECT_TRUE(manager.contains_range(kScope0, third, 32));
}

TEST(Tcgen05TmemTest, AllocRejectsAfterRelinquishPermit) {
  tcgen05_tmem_manager_t manager;

  manager.relinquish_alloc_permit(kScope0);

  EXPECT_DEATH(manager.alloc(kScope0, 32), "relinquish_alloc_permit");
}

TEST(Tcgen05TmemTest, DeallocRequiresMatchingRange) {
  tcgen05_tmem_manager_t manager;

  uint32_t base = manager.alloc(kScope0, 64);

  EXPECT_DEATH(manager.dealloc(kScope0, base, 32), "size does not match");
}

TEST(Tcgen05TmemTest, AddressDecodeRegisterAndSymbolicBase) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);
  uint32_t register_offset = 4;
  uint32_t symbolic_address = base + register_offset;

  manager.write_words(kScope0, symbolic_address, {0xdeadbeef});

  EXPECT_EQ(manager.read_words(kScope0, symbolic_address, 1)[0], 0xdeadbeefu);
}

TEST(Tcgen05TmemTest, BoundsRejectOverflow) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);

  EXPECT_TRUE(manager.contains_range(kScope0, base + 31, 1));
  EXPECT_FALSE(manager.contains_range(kScope0, base + 31, 2));
  EXPECT_FALSE(manager.contains_range(kScope0, base + 32, 1));
}

TEST(Tcgen05TmemTest, EncodedLaneBitsDoNotAffectColumnBounds) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 64);
  uint32_t lane32_base = tcgen05_encode_tmem_address(32, base);
  std::vector<uint32_t> values = sequence(0x6000, 16);

  EXPECT_TRUE(manager.contains_range(kScope0, lane32_base, 16));

  manager.write_words(kScope0, lane32_base, values);

  EXPECT_EQ(manager.read_words(kScope0, lane32_base, 16), values);
  EXPECT_EQ(manager.read_words(kScope0, base, 16), std::vector<uint32_t>(16, 0));
}

TEST(Tcgen05TmemTest, MatrixStoreUsesRowsAsTmemLanes) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);
  std::vector<uint32_t> values = {1, 2, 3, 4, 5, 6};

  manager.write_matrix_words(kScope0, base, values, /*rows=*/2, /*columns=*/3);

  EXPECT_EQ(manager.read_words(kScope0, tcgen05_encode_tmem_address(0, base),
                               3),
            std::vector<uint32_t>({1, 2, 3}));
  EXPECT_EQ(manager.read_words(kScope0, tcgen05_encode_tmem_address(1, base),
                               3),
            std::vector<uint32_t>({4, 5, 6}));
}

TEST(Tcgen05TmemTest, PackedU16MatrixReadSplitsWordsByRow) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);
  std::vector<uint32_t> values = {0x00020001u, 0x00040003u, 0x00060005u,
                                  0x00080007u};

  manager.write_matrix_words(kScope0, base, values, /*rows=*/2,
                             /*columns=*/2);

  EXPECT_EQ(manager.read_matrix_packed_u16(kScope0, base, /*rows=*/2,
                                           /*columns=*/4),
            std::vector<uint16_t>({1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(Tcgen05TmemTest, Warpx4CopyBroadcasts32x128BitRows) {
  std::vector<uint32_t> source(128, 0);
  for (uint32_t data_path = 0; data_path < 32; ++data_path) {
    for (uint32_t word = 0; word < 4; ++word) {
      source[data_path * 4 + word] =
          0x01020408u * (data_path + 1) ^ (0x11111111u * word);
    }
  }

  std::vector<uint32_t> result = tcgen05_warpx4_32x128b_words(source);

  ASSERT_EQ(result.size(), 128u * 4u);
  for (uint32_t data_path = 0; data_path < 128; ++data_path) {
    for (uint32_t word = 0; word < 4; ++word) {
      EXPECT_EQ(result[data_path * 4 + word],
                source[(data_path % 32) * 4 + word]);
    }
  }
}

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
  std::vector<uint8_t> high = manager.read_mxf4_scale_matrix(
      kScope0, scale_base | 0x80000000u, 128, 2,
      /*scale_factor_id=*/2);

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

TEST(Tcgen05TmemTest, StoreLoadRoundTripX16) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);
  std::vector<uint32_t> values = sequence(100, 16);

  manager.write_words(kScope0, base, values);

  EXPECT_EQ(manager.read_words(kScope0, base, 16), values);
}

TEST(Tcgen05TmemTest, StoreLoadRoundTripX32) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);
  std::vector<uint32_t> values = sequence(200, 32);

  manager.write_words(kScope0, base, values);

  EXPECT_EQ(manager.read_words(kScope0, base, 32), values);
}

TEST(Tcgen05TmemTest, UnwrittenWordsReadAsZero) {
  tcgen05_tmem_manager_t manager;
  uint32_t base = manager.alloc(kScope0, 32);

  EXPECT_EQ(manager.read_words(kScope0, base, 4),
            std::vector<uint32_t>({0, 0, 0, 0}));
}

TEST(Tcgen05TmemTest, ScopeIsolation) {
  tcgen05_tmem_manager_t manager;
  uint32_t scope0_base = manager.alloc(kScope0, 32);
  uint32_t scope1_base = manager.alloc(kScope1, 32);
  uint32_t group2_base = manager.alloc(kScopeGroup2, 32);

  manager.write_words(kScope0, scope0_base, {1, 2, 3, 4});
  manager.write_words(kScope1, scope1_base, {5, 6, 7, 8});
  manager.write_words(kScopeGroup2, group2_base, {9, 10, 11, 12});

  EXPECT_EQ(manager.read_words(kScope0, scope0_base, 4),
            std::vector<uint32_t>({1, 2, 3, 4}));
  EXPECT_EQ(manager.read_words(kScope1, scope1_base, 4),
            std::vector<uint32_t>({5, 6, 7, 8}));
  EXPECT_EQ(manager.read_words(kScopeGroup2, group2_base, 4),
            std::vector<uint32_t>({9, 10, 11, 12}));
}

TEST(Tcgen05TmemTest, ClearCtaRemovesAllCtaGroups) {
  tcgen05_tmem_manager_t manager;
  uint32_t scope0_base = manager.alloc(kScope0, 32);
  uint32_t group2_base = manager.alloc(kScopeGroup2, 32);
  uint32_t scope1_base = manager.alloc(kScope1, 32);

  manager.clear_cta(/*sm_id=*/0, /*cta_id=*/0);

  EXPECT_FALSE(manager.has_allocation(kScope0, scope0_base));
  EXPECT_FALSE(manager.has_allocation(kScopeGroup2, group2_base));
  EXPECT_TRUE(manager.has_allocation(kScope1, scope1_base));
}

TEST(Tcgen05TmemTest, ClearCtaRemovesPermitState) {
  tcgen05_tmem_manager_t manager;

  manager.relinquish_alloc_permit(kScope0);
  manager.clear_cta(/*sm_id=*/0, /*cta_id=*/0);

  uint32_t base = manager.alloc(kScope0, 32);

  EXPECT_TRUE(manager.has_allocation(kScope0, base));
  EXPECT_FALSE(manager.permit_relinquished(kScope0));
}

TEST(Tcgen05TmemTest, DecodeSharedMemoryDescriptorF16) {
  uint64_t encoded = make_shared_desc(/*start=*/0x120, /*lbo=*/0x40,
                                      /*sbo=*/0x80, /*swizzle=*/2);

  tcgen05_shared_descriptor_t desc =
      tcgen05_decode_shared_descriptor(encoded);

  EXPECT_EQ(desc.start_address, 0x120u);
  EXPECT_EQ(desc.leading_dimension_byte_offset, 0x40u);
  EXPECT_EQ(desc.stride_dimension_byte_offset, 0x80u);
  EXPECT_EQ(desc.fixed_constant, 1u);
  EXPECT_FALSE(desc.leading_dimension_absolute);
  EXPECT_EQ(desc.swizzle_mode, 2u);
}

TEST(Tcgen05TmemTest, DecodeInstructionDescriptorF16) {
  tcgen05_mma_descriptor_t desc =
      tcgen05_decode_f16_mma_descriptor(make_f16_idesc(64, 8), 1);

  EXPECT_FALSE(desc.sparse);
  EXPECT_EQ(desc.d_type, TCGEN05_MMA_TYPE_FIELD_ONE);
  EXPECT_EQ(desc.a_type, TCGEN05_MMA_TYPE_FIELD_F16);
  EXPECT_EQ(desc.b_type, TCGEN05_MMA_TYPE_FIELD_F16);
  EXPECT_FALSE(desc.transpose_a);
  EXPECT_FALSE(desc.transpose_b);
  EXPECT_EQ(desc.m, 64u);
  EXPECT_EQ(desc.n, 8u);
  EXPECT_EQ(desc.k, 16u);
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
    tcgen05_mma_descriptor_t desc = tcgen05_decode_mxf4_mma_descriptor(
        make_mxf4_idesc(n), /*cta_group=*/1);
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
    for (uint8_t scale_factor_id = 0; scale_factor_id < 4;
         ++scale_factor_id) {
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

TEST(Tcgen05TmemTest, Mxf4KMajorPackedAddressLinearFallback) {
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x100, /*lbo=*/0, /*sbo=*/0));

  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32),
            0x100u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 2, 32),
            0x122u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 127, 31, 32),
            0x10ffu);
}

TEST(Tcgen05TmemTest, Mxf4KMajorPackedAddressApplies128ByteSwizzle) {
  tcgen05_shared_descriptor_t desc = tcgen05_decode_shared_descriptor(
      make_shared_desc(/*start=*/0x400, /*lbo=*/0x10, /*sbo=*/0x400,
                       /*swizzle=*/2));

  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 0, 0, 32),
            0x400u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 1, 0, 32),
            0x490u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 7, 0, 32),
            0x7f0u);
  EXPECT_EQ(tcgen05_shared_k_major_packed_byte_address(desc, 8, 0, 32),
            0x800u);
}

TEST(Tcgen05TmemTest, Mxf4TheoreticalTimingScalesWithDescriptorWork) {
  EXPECT_EQ(tcgen05_mxf4_dense_work(128, 8, 64), 131072u);
  EXPECT_EQ(tcgen05_mxf4_compute_cycles(128, 8, 64, 30947), 5u);
  EXPECT_EQ(tcgen05_mxf4_compute_cycles(128, 256, 64, 30947), 136u);
  EXPECT_EQ(tcgen05_mxf4_compute_cycles(256, 256, 96, 30947), 407u);
}

TEST(Tcgen05TmemTest, Mxf8f6f4TheoreticalTimingScalesWithDescriptorWork) {
  EXPECT_EQ(tcgen05_mxf8f6f4_dense_work(128, 8, 32), 65536u);
  EXPECT_EQ(tcgen05_mxf8f6f4_compute_cycles(128, 8, 32, 15474), 5u);
  EXPECT_EQ(tcgen05_mxf8f6f4_compute_cycles(128, 256, 32, 15474), 136u);
  EXPECT_EQ(tcgen05_mxf8f6f4_compute_cycles(256, 256, 32, 15474), 272u);
}

TEST(Tcgen05TmemTest, Mxf4TimingBackendSerializesFrontendPipes) {
  tcgen05_timing_model_t timing;
  EXPECT_TRUE(timing.can_issue(100));
  timing.reserve(100, 75);
  EXPECT_FALSE(timing.can_issue(174));
  EXPECT_TRUE(timing.can_issue(175));
  EXPECT_EQ(timing.backend_ready_cycle(), 175u);
  timing.reset();
  EXPECT_TRUE(timing.can_issue(0));
}

TEST(Tcgen05TmemTest, Mxf4NumericFormats) {
  const float expected_e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  for (uint8_t value = 0; value < 8; ++value) {
    EXPECT_FLOAT_EQ(tcgen05_e2m1_to_f32(value), expected_e2m1[value]);
    EXPECT_FLOAT_EQ(tcgen05_e2m1_to_f32(value | 0x8),
                    -expected_e2m1[value]);
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

TEST(Tcgen05TmemTest, MmaF16KnownPatternNoAccum) {
  tcgen05_mma_descriptor_t desc =
      tcgen05_decode_f16_mma_descriptor(make_f16_idesc(64, 8), 1);
  std::vector<uint16_t> a(desc.m * desc.k, tcgen05_f32_to_f16(1.0f));
  std::vector<uint16_t> b(desc.k * desc.n);
  for (uint32_t k = 0; k < desc.k; ++k) {
    for (uint32_t n = 0; n < desc.n; ++n) {
      b[k * desc.n + n] = tcgen05_f32_to_f16(static_cast<float>(n + 1));
    }
  }

  std::vector<uint32_t> result =
      tcgen05_mma_f16_compute_words(desc, a, b, {}, false);

  ASSERT_EQ(result.size(), desc.m * desc.n);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result[0]), 16.0f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result[7]), 128.0f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result[(desc.m - 1) * desc.n + 3]),
                  64.0f);
}

TEST(Tcgen05TmemTest, MmaF16KnownPatternWithAccum) {
  tcgen05_mma_descriptor_t desc =
      tcgen05_decode_f16_mma_descriptor(make_f16_idesc(64, 8), 1);
  std::vector<uint16_t> a(desc.m * desc.k, tcgen05_f32_to_f16(1.0f));
  std::vector<uint16_t> b(desc.k * desc.n, tcgen05_f32_to_f16(1.0f));
  std::vector<uint32_t> input_d(desc.m * desc.n,
                                tcgen05_f32_to_bits(2.5f));

  std::vector<uint32_t> result =
      tcgen05_mma_f16_compute_words(desc, a, b, input_d, true);

  ASSERT_EQ(result.size(), desc.m * desc.n);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result[0]), 18.5f);
  EXPECT_FLOAT_EQ(tcgen05_bits_to_f32(result.back()), 18.5f);
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
  std::vector<uint32_t> input_d(desc.m * desc.n,
                                tcgen05_f32_to_bits(2.5f));

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
