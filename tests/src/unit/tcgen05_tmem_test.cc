#include <gtest/gtest.h>

#include "gpgpu-sim/flash/tcgen05.h"

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
