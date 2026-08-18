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
                         uint8_t b_scale_factor_id = 0) {
  return (static_cast<uint32_t>(b_scale_factor_id) << 4) |
         (TCGEN05_MXF4_FORMAT_E2M1 << 7) |
         (TCGEN05_MXF4_FORMAT_E2M1 << 10) | ((n >> 3) << 17) |
         (TCGEN05_SCALE_FORMAT_UE8M0 << 23) | ((128u >> 4) << 24) |
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

TEST(Tcgen05TmemTest, Warpx4CopyTransposesAndBroadcasts32x128Bits) {
  std::vector<uint32_t> source(128, 0);
  uint32_t expected[32][4] = {};
  for (uint32_t data_path = 0; data_path < 32; ++data_path) {
    for (uint32_t word = 0; word < 4; ++word) {
      expected[data_path][word] =
          0x01020408u * (data_path + 1) ^ (0x11111111u * word);
      for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t value = (expected[data_path][word] >> bit) & 0x1;
        uint32_t source_bit = data_path + (word * 32 + bit) * 32;
        source[source_bit / 32] |= value << (source_bit % 32);
      }
    }
  }

  std::vector<uint32_t> result = tcgen05_warpx4_32x128b_words(source);

  ASSERT_EQ(result.size(), 128u * 4u);
  for (uint32_t data_path = 0; data_path < 128; ++data_path) {
    for (uint32_t word = 0; word < 4; ++word) {
      EXPECT_EQ(result[data_path * 4 + word], expected[data_path % 32][word]);
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
      tcgen05_decode_mxf4_mma_descriptor(make_mxf4_idesc(64, 2, 3), 1);

  EXPECT_FALSE(desc.sparse);
  EXPECT_EQ(desc.a_type, TCGEN05_MXF4_FORMAT_E2M1);
  EXPECT_EQ(desc.b_type, TCGEN05_MXF4_FORMAT_E2M1);
  EXPECT_EQ(desc.scale_format, TCGEN05_SCALE_FORMAT_UE8M0);
  EXPECT_EQ(desc.a_scale_factor_id, 2u);
  EXPECT_EQ(desc.b_scale_factor_id, 3u);
  EXPECT_EQ(desc.m, 128u);
  EXPECT_EQ(desc.n, 64u);
  EXPECT_EQ(desc.k, 64u);
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
