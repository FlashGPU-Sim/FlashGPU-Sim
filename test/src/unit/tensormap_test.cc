#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "../../../src/gpgpu-sim/flash/tensormap.h"

// ============================================================================
// Standalone unit tests for tensormap_descriptor_t
//
// Tests the pure-C++ methods of tensormap_descriptor_t without requiring
// the full simulator (no memory_space, no ptx_thread_info).
// ============================================================================

namespace {

// Helper: create a minimal valid 1D tensormap descriptor
tensormap_descriptor_t make_1d(uint64_t base_addr, uint32_t data_type,
                                uint32_t box_dim0, uint32_t global_dim0) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = base_addr;
  tm.fields.tensorDataType = data_type;
  tm.fields.tensorRank = 0;  // 1D
  tm.fields.boxDim[0] = box_dim0;
  tm.fields.globalDim[0] = global_dim0;
  return tm;
}

// Helper: create a 2D tensormap
tensormap_descriptor_t make_2d(uint64_t base_addr, uint32_t data_type,
                                uint32_t box0, uint32_t box1,
                                uint32_t glob0, uint32_t glob1,
                                uint64_t stride1) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = base_addr;
  tm.fields.tensorDataType = data_type;
  tm.fields.tensorRank = 1;  // 2D
  tm.fields.boxDim[0] = box0;
  tm.fields.boxDim[1] = box1;
  tm.fields.globalDim[0] = glob0;
  tm.fields.globalDim[1] = glob1;
  tm.fields.globalStrides[0] = stride1;
  return tm;
}

// Helper: create a 3D tensormap
tensormap_descriptor_t make_3d(uint64_t base_addr, uint32_t data_type,
                                uint32_t box0, uint32_t box1, uint32_t box2,
                                uint32_t glob0, uint32_t glob1, uint32_t glob2,
                                uint64_t stride1, uint64_t stride2) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = base_addr;
  tm.fields.tensorDataType = data_type;
  tm.fields.tensorRank = 2;  // 3D
  tm.fields.boxDim[0] = box0;
  tm.fields.boxDim[1] = box1;
  tm.fields.boxDim[2] = box2;
  tm.fields.globalDim[0] = glob0;
  tm.fields.globalDim[1] = glob1;
  tm.fields.globalDim[2] = glob2;
  tm.fields.globalStrides[0] = stride1;
  tm.fields.globalStrides[1] = stride2;
  return tm;
}

}  // namespace

// ============================================================================
// Test Suite 1: get_element_size
// ============================================================================

class TensorMapElementSizeTest : public ::testing::Test {};

TEST_F(TensorMapElementSizeTest, U8) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U8, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 1u);
}

TEST_F(TensorMapElementSizeTest, U16) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U16, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 2u);
}

TEST_F(TensorMapElementSizeTest, U32) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U32, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 4u);
}

TEST_F(TensorMapElementSizeTest, U64) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U64, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 8u);
}

TEST_F(TensorMapElementSizeTest, F16) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F16, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 2u);
}

TEST_F(TensorMapElementSizeTest, F32) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F32, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 4u);
}

TEST_F(TensorMapElementSizeTest, F64) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F64, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 8u);
}

TEST_F(TensorMapElementSizeTest, BF16) {
  auto tm = make_1d(0x1000, TMA_DTYPE_BF16, 16, 64);
  EXPECT_EQ(tm.get_element_size(), 2u);
}

TEST_F(TensorMapElementSizeTest, UnknownDefaultsTo4) {
  auto tm = make_1d(0x1000, 99, 16, 64);  // invalid type
  EXPECT_EQ(tm.get_element_size(), 4u);
}

// ============================================================================
// Test Suite 2: get_tile_size_bytes
// ============================================================================

class TensorMapTileSizeTest : public ::testing::Test {};

TEST_F(TensorMapTileSizeTest, OneDimF32) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F32, 16, 64);
  EXPECT_EQ(tm.get_tile_size_bytes(), 16u * 4u);  // 64 bytes
}

TEST_F(TensorMapTileSizeTest, OneDimU8) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U8, 32, 128);
  EXPECT_EQ(tm.get_tile_size_bytes(), 32u * 1u);  // 32 bytes
}

TEST_F(TensorMapTileSizeTest, TwoDimF32) {
  auto tm = make_2d(0x1000, TMA_DTYPE_F32, 4, 4, 16, 16, 16 * 4);
  EXPECT_EQ(tm.get_tile_size_bytes(), 4u * 4u * 4u);  // 64 bytes
}

TEST_F(TensorMapTileSizeTest, TwoDimF16) {
  auto tm = make_2d(0x1000, TMA_DTYPE_F16, 8, 8, 32, 32, 32 * 2);
  EXPECT_EQ(tm.get_tile_size_bytes(), 8u * 8u * 2u);  // 128 bytes
}

TEST_F(TensorMapTileSizeTest, ThreeDimF32) {
  auto tm = make_3d(0x1000, TMA_DTYPE_F32, 2, 2, 2, 4, 4, 4, 4 * 4, 4 * 4 * 4);
  EXPECT_EQ(tm.get_tile_size_bytes(), 2u * 2u * 2u * 4u);  // 32 bytes
}

TEST_F(TensorMapTileSizeTest, RankAbove4ReturnsZero) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.tensorRank = 5;  // 6D (invalid)
  tm.fields.boxDim[0] = 2;
  tm.fields.boxDim[1] = 2;
  EXPECT_EQ(tm.get_tile_size_bytes(), 0u);
}

// ============================================================================
// Test Suite 3: calculate_src_addr
// ============================================================================

class TensorMapSrcAddrTest : public ::testing::Test {};

TEST_F(TensorMapSrcAddrTest, OneDimOrigin) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F32, 16, 64);
  int32_t coords[5] = {0, 0, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x1000u);
}

TEST_F(TensorMapSrcAddrTest, OneDimOffset) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F32, 16, 64);
  int32_t coords[5] = {5, 0, 0, 0, 0};
  // base + 5 * 4 = 0x1000 + 20 = 0x1014
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x1000u + 5 * 4);
}

TEST_F(TensorMapSrcAddrTest, OneDimU8Offset) {
  auto tm = make_1d(0x1000, TMA_DTYPE_U8, 16, 64);
  int32_t coords[5] = {10, 0, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x1000u + 10 * 1);
}

TEST_F(TensorMapSrcAddrTest, TwoDimOrigin) {
  // 16 columns (dim0), 8 rows (dim1), stride1 = 16 * 4 = 64
  auto tm = make_2d(0x2000, TMA_DTYPE_F32, 4, 4, 16, 8, 16 * 4);
  int32_t coords[5] = {0, 0, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x2000u);
}

TEST_F(TensorMapSrcAddrTest, TwoDimColOffset) {
  // coord0=col=3 -> base + 3*4 = 0x200c
  auto tm = make_2d(0x2000, TMA_DTYPE_F32, 4, 4, 16, 8, 16 * 4);
  int32_t coords[5] = {3, 0, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x2000u + 3 * 4);
}

TEST_F(TensorMapSrcAddrTest, TwoDimRowOffset) {
  // coord0=0, coord1=2 -> base + 2 * (16*4) = 0x2000 + 128 = 0x2080
  auto tm = make_2d(0x2000, TMA_DTYPE_F32, 4, 4, 16, 8, 16 * 4);
  int32_t coords[5] = {0, 2, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x2000u + 2 * 16 * 4);
}

TEST_F(TensorMapSrcAddrTest, TwoDimColAndRowOffset) {
  // coord0=5, coord1=3 -> base + 5*4 + 3*(16*4) = 0x2000 + 20 + 192 = 0x20D4
  auto tm = make_2d(0x2000, TMA_DTYPE_F32, 4, 4, 16, 8, 16 * 4);
  int32_t coords[5] = {5, 3, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords),
            0x2000u + 5 * 4 + 3 * 16 * 4);
}

TEST_F(TensorMapSrcAddrTest, ThreeDimOrigin) {
  auto tm = make_3d(0x3000, TMA_DTYPE_F32,
                    2, 2, 2,    // box dims
                    4, 4, 4,    // global dims
                    4 * 4,          // stride1 = D0 * elem_size = 4*4 = 16
                    4 * 4 * 4);     // stride2 = D0 * D1 * elem_size = 4*4*4 = 64
  int32_t coords[5] = {0, 0, 0, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x3000u);
}

TEST_F(TensorMapSrcAddrTest, ThreeDimD2Offset) {
  // coord0=0, coord1=0, coord2=1 -> base + 1 * stride2 = 0x3000 + 64 = 0x3040
  auto tm = make_3d(0x3000, TMA_DTYPE_F32,
                    2, 2, 2,       // box dims
                    4, 4, 4,       // global dims
                    4 * 4,         // stride1
                    4 * 4 * 4);    // stride2
  int32_t coords[5] = {0, 0, 1, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x3000u + 64);
}

TEST_F(TensorMapSrcAddrTest, ThreeDimAllOffsets) {
  // coord0=3, coord1=2, coord2=1
  // addr = 0x3000 + 3*4 + 2*16 + 1*64 = 0x3000 + 12 + 32 + 64 = 0x306C
  auto tm = make_3d(0x3000, TMA_DTYPE_F32,
                    2, 2, 2,
                    4, 4, 4,
                    4 * 4,
                    4 * 4 * 4);
  int32_t coords[5] = {3, 2, 1, 0, 0};
  EXPECT_EQ(tm.calculate_src_addr(coords), 0x3000u + 12 + 32 + 64);
}

// ============================================================================
// Test Suite 4: num_dims
// ============================================================================

class TensorMapNumDimsTest : public ::testing::Test {};

TEST_F(TensorMapNumDimsTest, Rank0Is1D) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.tensorRank = 0;
  EXPECT_EQ(tm.num_dims(), 1u);
}

TEST_F(TensorMapNumDimsTest, Rank1Is2D) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.tensorRank = 1;
  EXPECT_EQ(tm.num_dims(), 2u);
}

TEST_F(TensorMapNumDimsTest, Rank4Is5D) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.tensorRank = 4;
  EXPECT_EQ(tm.num_dims(), 5u);
}

// ============================================================================
// Test Suite 5: is_valid
// ============================================================================

class TensorMapValidTest : public ::testing::Test {};

TEST_F(TensorMapValidTest, ValidTensormap) {
  auto tm = make_1d(0x1000, TMA_DTYPE_F32, 16, 64);
  EXPECT_TRUE(tm.is_valid());
}

TEST_F(TensorMapValidTest, ZeroAddressInvalid) {
  auto tm = make_1d(0x0, TMA_DTYPE_F32, 16, 64);
  EXPECT_FALSE(tm.is_valid());
}

TEST_F(TensorMapValidTest, RankAbove4Invalid) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = 0x1000;
  tm.fields.tensorRank = 5;
  EXPECT_FALSE(tm.is_valid());
}

TEST_F(TensorMapValidTest, Rank4MaxValid) {
  tensormap_descriptor_t tm;
  memset(&tm, 0, sizeof(tm));
  tm.fields.globalAddress = 0x1000;
  tm.fields.tensorRank = 4;  // 5D is max
  tm.fields.tensorDataType = TMA_DTYPE_F32;
  EXPECT_TRUE(tm.is_valid());
}

// ============================================================================
// Test Suite 6: sizeof / alignment
// ============================================================================

class TensorMapLayoutTest : public ::testing::Test {};

TEST_F(TensorMapLayoutTest, SizeIs128) {
  EXPECT_EQ(sizeof(tensormap_descriptor_t), 128u);
}

TEST_F(TensorMapLayoutTest, AlignmentIs128) {
  EXPECT_EQ(alignof(tensormap_descriptor_t), 128u);
}
