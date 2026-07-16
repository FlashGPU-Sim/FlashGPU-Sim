#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "../../../src/gpgpu-sim/flash/tma_helpers.h"

using namespace flash_gpgpu_sim;

//=============================================================================
// Helper: create a minimal valid tensormap descriptor for testing
//=============================================================================

static tensormap_descriptor_t make_1d_tensormap(uint64_t base_addr,
                                                 uint32_t elem_size,
                                                 uint32_t box_dim0,
                                                 uint32_t global_dim0) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = base_addr;
  tm.fields.tensorRank = 0; // 1D = rank 0 (num_dims = rank + 1 = 1)
  tm.fields.boxDim[0] = box_dim0;
  tm.fields.globalDim[0] = global_dim0;
  // Set data type based on elem_size
  if (elem_size == 1)
    tm.fields.tensorDataType = TMA_DTYPE_U8;
  else if (elem_size == 2)
    tm.fields.tensorDataType = TMA_DTYPE_F16;
  else if (elem_size == 4)
    tm.fields.tensorDataType = TMA_DTYPE_F32;
  else
    tm.fields.tensorDataType = TMA_DTYPE_U8; // fallback
  return tm;
}

static tensormap_descriptor_t make_2d_tensormap(uint64_t base_addr,
                                                 uint32_t elem_size,
                                                 uint32_t box_dim0,
                                                 uint32_t box_dim1,
                                                 uint32_t global_dim0,
                                                 uint32_t global_dim1,
                                                 uint64_t stride1) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = base_addr;
  tm.fields.tensorRank = 1; // 2D = rank 1
  tm.fields.boxDim[0] = box_dim0;
  tm.fields.boxDim[1] = box_dim1;
  tm.fields.globalDim[0] = global_dim0;
  tm.fields.globalDim[1] = global_dim1;
  tm.fields.globalStrides[0] = stride1;
  if (elem_size == 1)
    tm.fields.tensorDataType = TMA_DTYPE_U8;
  else if (elem_size == 2)
    tm.fields.tensorDataType = TMA_DTYPE_F16;
  else if (elem_size == 4)
    tm.fields.tensorDataType = TMA_DTYPE_F32;
  return tm;
}

//=============================================================================
// gen_aligned_req Tests
//=============================================================================

TEST(TMAHelpersTest, GenAlignedReq_Empty) {
  std::vector<std::pair<uint64_t, uint32_t>> reqs;
  gen_aligned_req(0x1000, 0, reqs);
  EXPECT_TRUE(reqs.empty());
}

TEST(TMAHelpersTest, GenAlignedReq_ExactCacheLine) {
  std::vector<std::pair<uint64_t, uint32_t>> reqs;
  gen_aligned_req(0x1000, 128, reqs);
  ASSERT_EQ(reqs.size(), 1u);
  EXPECT_EQ(reqs[0].first, 0x1000u);
  EXPECT_EQ(reqs[0].second, 128u);
}

TEST(TMAHelpersTest, GenAlignedReq_SmallUnaligned) {
  std::vector<std::pair<uint64_t, uint32_t>> reqs;
  // 64 bytes starting at 0x1040 (aligned to 64, not 128 boundary)
  gen_aligned_req(0x1040, 64, reqs);
  ASSERT_EQ(reqs.size(), 1u);
  EXPECT_EQ(reqs[0].first, 0x1040u);
  EXPECT_EQ(reqs[0].second, 64u);
}

TEST(TMAHelpersTest, GenAlignedReq_CrossesCacheLine) {
  std::vector<std::pair<uint64_t, uint32_t>> reqs;
  // 200 bytes starting at 0x10C0 (192 from 128B boundary)
  // Should split: 64 bytes to boundary, then 128, then 8
  gen_aligned_req(0x10C0, 200, reqs);
  ASSERT_EQ(reqs.size(), 3u);
  EXPECT_EQ(reqs[0].first, 0x10C0u);
  EXPECT_EQ(reqs[0].second, 64u); // to 0x1100 boundary
  EXPECT_EQ(reqs[1].first, 0x1100u);
  EXPECT_EQ(reqs[1].second, 128u); // full cache line
  EXPECT_EQ(reqs[2].first, 0x1180u);
  EXPECT_EQ(reqs[2].second, 8u); // remaining
}

TEST(TMAHelpersTest, GenAlignedReq_LargeMultipleCacheLines) {
  std::vector<std::pair<uint64_t, uint32_t>> reqs;
  gen_aligned_req(0x1000, 512, reqs);
  ASSERT_EQ(reqs.size(), 4u);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(reqs[i].first, 0x1000u + i * 128);
    EXPECT_EQ(reqs[i].second, 128u);
  }
}

//=============================================================================
// global_to_tile_offset Tests
//=============================================================================

TEST(TMAHelpersTest, GlobalToTileOffset_1D) {
  auto tm = make_1d_tensormap(0x1000, 4, 16, 100);
  // 1D: tile offset equals global offset
  EXPECT_EQ(global_to_tile_offset(0x1000, 0x1000, tm), 0u);
  EXPECT_EQ(global_to_tile_offset(0x1008, 0x1000, tm), 8u);
  EXPECT_EQ(global_to_tile_offset(0x1030, 0x1000, tm), 0x30u);
}

TEST(TMAHelpersTest, GlobalToTileOffset_2D_SameStride) {
  // 2D: 4-byte elements, 16 cols (box), 100 cols (global), row stride = 64
  auto tm = make_2d_tensormap(0x1000, 4, 16, 8, 100, 50, 64);

  // First row, first element
  EXPECT_EQ(global_to_tile_offset(0x1000, 0x1000, tm), 0u);
  // First row, element 4 (byte 16)
  EXPECT_EQ(global_to_tile_offset(0x1010, 0x1000, tm), 16u);
  // Second row (byte 64 in global)
  EXPECT_EQ(global_to_tile_offset(0x1040, 0x1000, tm), 64u);
}

//=============================================================================
// apply_tma_swizzle Tests
//=============================================================================

TEST(TMAHelpersTest, SwizzleNone) {
  // No swizzle: address unchanged
  EXPECT_EQ(apply_tma_swizzle(0x100, 0, TMA_SWIZZLE_NONE, 128), 0x100u);
  EXPECT_EQ(apply_tma_swizzle(0x200, 0, TMA_SWIZZLE_NONE, 256), 0x200u);
}

TEST(TMAHelpersTest, Swizzle32B_Pattern) {
  // 32B swizzle: XOR with ((row_bits & 0x1) << 4)
  // addr=0x000: row_bits=0, XOR 0 -> 0x000
  EXPECT_EQ(apply_tma_swizzle(0x000, 0, TMA_SWIZZLE_32B, 128), 0x000u);
  // addr=0x010: row_bits=0, XOR 0 -> 0x010
  EXPECT_EQ(apply_tma_swizzle(0x010, 0, TMA_SWIZZLE_32B, 128), 0x010u);
  // addr=0x080: row_bits=1, XOR 0x10 -> 0x090
  EXPECT_EQ(apply_tma_swizzle(0x080, 0, TMA_SWIZZLE_32B, 128), 0x090u);
  // addr=0x090: row_bits=1, XOR 0x10 -> 0x080
  EXPECT_EQ(apply_tma_swizzle(0x090, 0, TMA_SWIZZLE_32B, 128), 0x080u);
}

TEST(TMAHelpersTest, Swizzle64B_Pattern) {
  // 64B swizzle: XOR with ((row_bits & 0x3) << 4)
  // addr=0x000: row_bits=0, XOR 0 -> 0x000
  EXPECT_EQ(apply_tma_swizzle(0x000, 0, TMA_SWIZZLE_64B, 128), 0x000u);
  // addr=0x080: row_bits=1, XOR 0x10 -> 0x090
  EXPECT_EQ(apply_tma_swizzle(0x080, 0, TMA_SWIZZLE_64B, 128), 0x090u);
  // addr=0x100: row_bits=2, XOR 0x20 -> 0x120
  EXPECT_EQ(apply_tma_swizzle(0x100, 0, TMA_SWIZZLE_64B, 128), 0x120u);
  // addr=0x180: row_bits=3, XOR 0x30 -> 0x1B0
  EXPECT_EQ(apply_tma_swizzle(0x180, 0, TMA_SWIZZLE_64B, 128), 0x1B0u);
}

TEST(TMAHelpersTest, Swizzle128B_Pattern) {
  // 128B swizzle: XOR with ((row_bits & 0x7) << 4)
  // addr=0x000: row_bits=0, XOR 0 -> 0x000
  EXPECT_EQ(apply_tma_swizzle(0x000, 0, TMA_SWIZZLE_128B, 128), 0x000u);
  // addr=0x080: row_bits=1, XOR 0x10 -> 0x090
  EXPECT_EQ(apply_tma_swizzle(0x080, 0, TMA_SWIZZLE_128B, 128), 0x090u);
  // addr=0x200: row_bits=4, 4&7=4, XOR 0x40 -> 0x240
  EXPECT_EQ(apply_tma_swizzle(0x200, 0, TMA_SWIZZLE_128B, 128), 0x240u);
  // addr=0x380: row_bits=7, 7&7=7, XOR 0x70 -> 0x3F0
  EXPECT_EQ(apply_tma_swizzle(0x380, 0, TMA_SWIZZLE_128B, 128), 0x3F0u);
}

TEST(TMAHelpersTest, SwizzleRoundTrip) {
  // Applying swizzle twice should return to original
  for (uint64_t addr : {0x000ULL, 0x010ULL, 0x080ULL, 0x100ULL, 0x180ULL,
                         0x200ULL, 0x300ULL}) {
    uint64_t swizzled = apply_tma_swizzle(addr, 0, TMA_SWIZZLE_128B, 128);
    uint64_t back = apply_tma_swizzle(swizzled, 0, TMA_SWIZZLE_128B, 128);
    EXPECT_EQ(back, addr) << "Round-trip failed for addr=0x" << std::hex << addr;
  }
}

//=============================================================================
// generate_tma_requests Tests
//=============================================================================

TEST(TMAHelpersTest, GenerateTMARequests_InvalidTensorMap) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = 0; // invalid (address is 0)

  uint32_t coords[5] = {0, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);
  EXPECT_TRUE(reqs.empty());
}

TEST(TMAHelpersTest, GenerateTMARequests_1D_SingleLine) {
  auto tm = make_1d_tensormap(0x1000, 4, 8, 100);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);

  // 8 elements * 4 bytes = 32 bytes, fits in one 128B-aligned request
  ASSERT_EQ(reqs.size(), 1u);
  EXPECT_EQ(reqs[0].first, 0x1000u);
  EXPECT_EQ(reqs[0].second, 32u);
}

TEST(TMAHelpersTest, GenerateTMARequests_1D_CrossesLine) {
  auto tm = make_1d_tensormap(0x10E0, 4, 16, 100);
  // 16 elements * 4 bytes = 64 bytes starting at 0x10E0
  // 0x10E0 is 32 bytes from 0x1100 boundary
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);

  ASSERT_EQ(reqs.size(), 2u);
  EXPECT_EQ(reqs[0].first, 0x10E0u);
  EXPECT_EQ(reqs[0].second, 32u);
  EXPECT_EQ(reqs[1].first, 0x1100u);
  EXPECT_EQ(reqs[1].second, 32u);
}

TEST(TMAHelpersTest, GenerateTMARequests_2D) {
  // 2D: 4-byte elements, 4 cols x 2 rows tile, 100x50 global, row_stride=400
  auto tm = make_2d_tensormap(0x1000, 4, 4, 2, 100, 50, 400);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);

  // 4 cols * 4 bytes = 16 bytes per row
  // Row 0: 16 bytes at 0x1000
  // Row 1: 16 bytes at 0x1000 + 400 = 0x1190
  // Each row fits in a single 128B-aligned request
  ASSERT_EQ(reqs.size(), 2u);
  EXPECT_EQ(reqs[0].first, 0x1000u);
  EXPECT_EQ(reqs[0].second, 16u);
  EXPECT_EQ(reqs[1].first, 0x1190u);
  EXPECT_EQ(reqs[1].second, 16u);
}

TEST(TMAHelpersTest, GenerateTMARequests_OOB_Dim0_Clamped) {
  // 1D: box_dim=16, global_dim=8, so only first 8 elements are valid
  auto tm = make_1d_tensormap(0x1000, 4, 16, 8);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);

  // Only 8 valid elements * 4 bytes = 32 bytes
  ASSERT_EQ(reqs.size(), 1u);
  EXPECT_EQ(reqs[0].first, 0x1000u);
  EXPECT_EQ(reqs[0].second, 32u);
}

TEST(TMAHelpersTest, GenerateTMARequests_StartOffset) {
  // 1D: box_dim=16, start at coord 4, global_dim=100
  auto tm = make_1d_tensormap(0x1000, 4, 16, 100);
  uint32_t coords[5] = {4, 0, 0, 0, 0};
  auto reqs = generate_tma_requests(tm, coords);

  // 16 elements from offset 4, so 16*4=64 bytes starting at 0x1000+4*4=0x1010
  ASSERT_EQ(reqs.size(), 1u);
  EXPECT_EQ(reqs[0].first, 0x1010u);
  EXPECT_EQ(reqs[0].second, 64u);
}

//=============================================================================
// tma_agu_unit_t Tests
//=============================================================================

class TMAAGUTest : public ::testing::Test {
protected:
  tma_agu_unit_t agu;
};

TEST_F(TMAAGUTest, LinearInit_Basic) {
  tma_agu_state_t state;
  agu.init_linear(state, 0x1000, 256);

  EXPECT_FALSE(state.is_tensor);
  EXPECT_FALSE(state.done);
  EXPECT_EQ(state.linear_addr, 0x1000u);
  EXPECT_EQ(state.linear_remaining, 256u);
}

TEST_F(TMAAGUTest, LinearInit_ZeroBytes) {
  tma_agu_state_t state;
  agu.init_linear(state, 0x1000, 0);

  EXPECT_TRUE(state.done);
}

TEST_F(TMAAGUTest, LinearGenNextReq_SingleAligned) {
  tma_agu_state_t state;
  agu.init_linear(state, 0x1000, 32); // 32 bytes at aligned address

  uint64_t addr;
  uint32_t size;
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 32u);

  // Should be done after one request
  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
  EXPECT_TRUE(state.done);
}

TEST_F(TMAAGUTest, LinearGenNextReq_MultipleSectors) {
  tma_agu_state_t state;
  agu.init_linear(state, 0x1000, 96); // 3 sectors worth

  uint64_t addr;
  uint32_t size;

  // First sector
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 32u);

  // Second sector
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1020u);
  EXPECT_EQ(size, 32u);

  // Third sector
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1040u);
  EXPECT_EQ(size, 32u);

  // Done
  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
}

TEST_F(TMAAGUTest, LinearGenNextReq_UnalignedStart) {
  tma_agu_state_t state;
  agu.init_linear(state, 0x1010, 80); // start at +16 within sector

  uint64_t addr;
  uint32_t size;

  // First: from 0x1010 to 0x1020 boundary = 16 bytes
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1010u);
  EXPECT_EQ(size, 16u);

  // Second: 0x1020 to 0x1040 = 32 bytes
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1020u);
  EXPECT_EQ(size, 32u);

  // Third: remaining 32 bytes
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1040u);
  EXPECT_EQ(size, 32u);

  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(state.linear_remaining, 0u);
}

TEST_F(TMAAGUTest, TensorInit_1D) {
  auto tm = make_1d_tensormap(0x1000, 4, 16, 100);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  EXPECT_TRUE(state.is_tensor);
  EXPECT_FALSE(state.done);
  EXPECT_EQ(state.num_dims, 1u);
  EXPECT_EQ(state.elem_size, 4u);
  EXPECT_EQ(state.box_dim[0], 16u);
  EXPECT_EQ(state.global_dim[0], 100u);
  EXPECT_EQ(state.row_bytes, 64u); // 16 * 4
  EXPECT_EQ(state.curr_row_addr, 0x1000u);
}

TEST_F(TMAAGUTest, TensorInit_2D) {
  auto tm = make_2d_tensormap(0x1000, 4, 8, 4, 100, 50, 400);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  EXPECT_TRUE(state.is_tensor);
  EXPECT_EQ(state.num_dims, 2u);
  EXPECT_EQ(state.box_dim[0], 8u);
  EXPECT_EQ(state.box_dim[1], 4u);
  EXPECT_EQ(state.global_dim[0], 100u);
  EXPECT_EQ(state.global_dim[1], 50u);
  EXPECT_EQ(state.global_strides[0], 4u);  // elem_size
  EXPECT_EQ(state.global_strides[1], 400u); // globalStrides[0]
}

TEST_F(TMAAGUTest, TensorGenNextReq_1D_SingleRow) {
  auto tm = make_1d_tensormap(0x1000, 4, 2, 100); // 2 elements = 8 bytes
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  uint64_t addr;
  uint32_t size;
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 8u);
  EXPECT_FALSE(state.is_fill_request);

  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
  EXPECT_TRUE(state.done);
}

TEST_F(TMAAGUTest, TensorGenNextReq_1D_MultiSector) {
  auto tm = make_1d_tensormap(0x1000, 4, 24, 100); // 24 elements = 96 bytes
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  uint64_t addr;
  uint32_t size;

  // 3 sectors of 32 bytes each
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 32u);

  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1020u);
  EXPECT_EQ(size, 32u);

  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1040u);
  EXPECT_EQ(size, 32u);

  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
}

TEST_F(TMAAGUTest, TensorGenNextReq_2D_TwoRows) {
  // 2D: 8 cols * 2 rows, 4 bytes/elem, row_stride=400
  // Row 1 starts at 0x1190 which is misaligned (16 bytes from 32B boundary),
  // so it gets split into two 16-byte requests.
  auto tm = make_2d_tensormap(0x1000, 4, 8, 2, 100, 50, 400);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  uint64_t addr;
  uint32_t size;

  // Row 0: 8 * 4 = 32 bytes at 0x1000 (aligned, one sector)
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 32u);

  // Row 1: 16 bytes at 0x1190 (to 32B boundary)
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1190u);
  EXPECT_EQ(size, 16u);

  // Row 1 continued: remaining 16 bytes at 0x11A0
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x11A0u);
  EXPECT_EQ(size, 16u);

  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
  EXPECT_TRUE(state.done);
}

TEST_F(TMAAGUTest, TensorGenNextReq_OOB_HigherDim) {
  // 2D: start_coords[1] = 60, but global_dim[1] = 50 -> higher dim OOB
  auto tm = make_2d_tensormap(0x1000, 4, 8, 2, 100, 50, 400);
  uint32_t coords[5] = {0, 60, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  // Row should be marked as OOB
  EXPECT_TRUE(state.row_is_oob);

  uint64_t addr;
  uint32_t size;
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_TRUE(state.is_fill_request);
}

TEST_F(TMAAGUTest, TensorGenNextReq_OOB_Dim0) {
  // 1D: box_dim=16, global_dim=4 -> only first 4 elements (16 bytes) valid
  // SECTOR_SIZE=32, so first request returns a full 32-byte sector
  // but only the first 16 bytes are valid; the rest is fill.
  auto tm = make_1d_tensormap(0x1000, 4, 16, 4);
  uint32_t coords[5] = {0, 0, 0, 0, 0};
  tma_agu_state_t state;

  agu.init_tensor(state, tm, coords);

  // valid_row_bytes = min(16, 4) * 4 = 16 bytes
  EXPECT_EQ(state.valid_row_bytes, 16u);

  uint64_t addr;
  uint32_t size;

  // First sector: 32 bytes at 0x1000, first 16 valid
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1000u);
  EXPECT_EQ(size, 32u);
  EXPECT_FALSE(state.is_fill_request);

  // Second sector: remaining 32 bytes (offset 32 >= valid 16) -> fill
  ASSERT_TRUE(agu.gen_next_req(state, addr, size));
  EXPECT_EQ(addr, 0x1020u);
  EXPECT_EQ(size, 32u);
  EXPECT_TRUE(state.is_fill_request);

  EXPECT_FALSE(agu.gen_next_req(state, addr, size));
}

//=============================================================================
// tma_oob_fill_table_t Tests
//=============================================================================

TEST(TMAOOBFillTableTest, ZeroPatternAllZeros) {
  const auto &table = tma_oob_fill_table_t::instance();

  // Zero pattern for any dtype should be all zeros
  for (uint32_t dtype = 0; dtype < 8; dtype++) {
    const unsigned char *pat = table.get_pattern(TMA_OOB_ZERO, dtype);
    for (uint32_t i = 0; i < 128; i++) {
      EXPECT_EQ(pat[i], 0u) << "Zero pattern dtype=" << dtype << " byte=" << i;
    }
  }
}

TEST(TMAOOBFillTableTest, NaN_PatternF32) {
  const auto &table = tma_oob_fill_table_t::instance();

  EXPECT_TRUE(table.nan_supported[TMA_DTYPE_F32]);

  const unsigned char *pat = table.get_pattern(TMA_OOB_NAN, TMA_DTYPE_F32);
  // Should be 0x7FFFFFFF for each u32
  for (uint32_t i = 0; i < 32; i++) {
    uint32_t val;
    memcpy(&val, pat + i * 4, 4);
    EXPECT_EQ(val, 0x7FFFFFFFu) << "F32 NaN at element " << i;
  }
}

TEST(TMAOOBFillTableTest, NaN_PatternF16) {
  const auto &table = tma_oob_fill_table_t::instance();

  EXPECT_TRUE(table.nan_supported[TMA_DTYPE_F16]);

  const unsigned char *pat = table.get_pattern(TMA_OOB_NAN, TMA_DTYPE_F16);
  for (uint32_t i = 0; i < 64; i++) {
    uint16_t val;
    memcpy(&val, pat + i * 2, 2);
    EXPECT_EQ(val, 0x7FFFu) << "F16 NaN at element " << i;
  }
}

TEST(TMAOOBFillTableTest, NaN_PatternU8_FP8) {
  const auto &table = tma_oob_fill_table_t::instance();

  // U8 treated as FP8: NaN = 0x7F
  EXPECT_TRUE(table.nan_supported[TMA_DTYPE_U8]);

  const unsigned char *pat = table.get_pattern(TMA_OOB_NAN, TMA_DTYPE_U8);
  for (uint32_t i = 0; i < 128; i++) {
    EXPECT_EQ(pat[i], 0x7Fu) << "FP8 NaN at byte " << i;
  }
}

TEST(TMAOOBFillTableTest, NaN_FallbackForIntegerTypes) {
  const auto &table = tma_oob_fill_table_t::instance();

  // U32 (integer): NaN not supported, should fall back to zero
  EXPECT_FALSE(table.nan_supported[TMA_DTYPE_U32]);

  const unsigned char *pat = table.get_pattern(TMA_OOB_NAN, TMA_DTYPE_U32);
  for (uint32_t i = 0; i < 128; i++) {
    EXPECT_EQ(pat[i], 0u) << "U32 NaN fallback should be zero at byte " << i;
  }
}

TEST(TMAOOBFillTableTest, OutOfRangeDtype) {
  const auto &table = tma_oob_fill_table_t::instance();

  // dtype >= NUM_DTYPES should be clamped to 0
  const unsigned char *pat = table.get_pattern(TMA_OOB_ZERO, 100);
  // Should not crash, return zero pattern
  for (uint32_t i = 0; i < 128; i++) {
    EXPECT_EQ(pat[i], 0u);
  }
}
