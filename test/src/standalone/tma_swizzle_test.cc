#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// Re-define constants to avoid dependency issues if needed, or include tma.h
#define TEST_TMA_SWIZZLE_NONE      0u
#define TEST_TMA_SWIZZLE_32B       1u
#define TEST_TMA_SWIZZLE_64B       2u
#define TEST_TMA_SWIZZLE_128B      3u
#define TEST_TMA_SWIZZLE_96B       4u

// Local implementation of the proposed swizzle function for verification
// Mask-based implementation matching hardware behavior
static uint64_t test_apply_tma_swizzle(uint64_t linear_offset, uint32_t swizzle_mode, uint32_t row_bytes) {
  if (swizzle_mode == TEST_TMA_SWIZZLE_NONE) return linear_offset;
  
  uint32_t mask = 0;
  constexpr uint32_t shift = 4;  // 16B granularity
  
  switch (swizzle_mode) {
    case TEST_TMA_SWIZZLE_128B: mask = 0x7; break;  // 3 bits, cycle of 8
    case TEST_TMA_SWIZZLE_64B:  mask = 0x3; break;  // 2 bits, cycle of 4
    case TEST_TMA_SWIZZLE_32B:  mask = 0x1; break;  // 1 bit, cycle of 2
    // TODO: 96B swizzle not yet implemented - mask and row stride unknown
    case TEST_TMA_SWIZZLE_96B:  mask = 0x1; break;  // placeholder, likely incorrect
    default: return linear_offset;
  }
  
  // Extract row information assuming 128B stride
  uint32_t row_bits = (uint32_t)(linear_offset >> 7);
  
  // Apply XOR permutation
  return linear_offset ^ ((uint64_t)(row_bits & mask) << shift);
}

class TMASwizzleTest : public ::testing::Test {
protected:
    void PrintLayout(uint32_t mode, const std::string& name, uint32_t num_rows) {
        std::cout << "\n=== " << name << " Layout (16B blocks) ===" << std::endl;
        uint32_t row_bytes = 128; // Standard 128B row for swizzle visualization
        
        for (uint32_t r = 0; r < num_rows; ++r) {
            std::cout << "Row " << r << ": ";
            for (uint32_t b = 0; b < 8; ++b) {
                uint64_t linear_offset = r * row_bytes + b * 16;
                uint64_t swizzled = test_apply_tma_swizzle(linear_offset, mode, row_bytes);
                uint32_t phys_block = (uint32_t)((swizzled % row_bytes) / 16);
                std::cout << phys_block << " ";
            }
            std::cout << std::endl;
        }
    }
    
    void VerifyPattern(uint32_t mode, const std::vector<std::vector<int>>& expected) {
        uint32_t row_bytes = 128;
        for (size_t r = 0; r < expected.size(); ++r) {
            for (size_t b = 0; b < 8; ++b) {
                uint64_t linear_offset = r * row_bytes + b * 16;
                uint64_t swizzled = test_apply_tma_swizzle(linear_offset, mode, row_bytes);
                uint32_t phys_block = (uint32_t)((swizzled % row_bytes) / 16);
                EXPECT_EQ(phys_block, (uint32_t)expected[r][b]) 
                    << "Mismatch at Row " << r << ", Block " << b;
            }
        }
    }
};

TEST_F(TMASwizzleTest, Swizzle32B) {
    std::vector<std::vector<int>> expected = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 3, 2, 5, 4, 7, 6}
    };
    PrintLayout(TEST_TMA_SWIZZLE_32B, "32B Swizzle", 2);
    VerifyPattern(TEST_TMA_SWIZZLE_32B, expected);
}

TEST_F(TMASwizzleTest, Swizzle64B) {
    std::vector<std::vector<int>> expected = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 3, 2, 5, 4, 7, 6},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 2, 1, 0, 7, 6, 5, 4}
    };
    PrintLayout(TEST_TMA_SWIZZLE_64B, "64B Swizzle", 4);
    VerifyPattern(TEST_TMA_SWIZZLE_64B, expected);
}

TEST_F(TMASwizzleTest, Swizzle128B) {
    std::vector<std::vector<int>> expected = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 3, 2, 5, 4, 7, 6},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 2, 1, 0, 7, 6, 5, 4},
        {4, 5, 6, 7, 0, 1, 2, 3},
        {5, 4, 7, 6, 1, 0, 3, 2},
        {6, 7, 4, 5, 2, 3, 0, 1},
        {7, 6, 5, 4, 3, 2, 1, 0}
    };
    PrintLayout(TEST_TMA_SWIZZLE_128B, "128B Swizzle", 8);
    VerifyPattern(TEST_TMA_SWIZZLE_128B, expected);
}

// TODO: 96B swizzle not yet implemented - expected pattern unknown
// Re-enable this test once 96B swizzle behavior is verified on hardware
TEST_F(TMASwizzleTest, DISABLED_Swizzle96B) {
    std::vector<std::vector<int>> expected = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 3, 2, 5, 4, 7, 6}
    };
    PrintLayout(TEST_TMA_SWIZZLE_96B, "96B Swizzle", 2);
    VerifyPattern(TEST_TMA_SWIZZLE_96B, expected);
}
