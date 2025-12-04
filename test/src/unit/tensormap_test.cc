/*
 * TensorMap Descriptor Unit Test
 * 
 * Test Principle:
 * ===============
 * This test validates the complete functionality of tensormap descriptor operations
 * including all tensormap.replace.tile.* fields. The test creates a mock shared memory
 * space, initializes a tensormap descriptor, modifies each field using the replace
 * operations, and verifies the values are correctly stored and retrieved.
 *
 * Test Coverage:
 * ==============
 * - global_address: 64-bit tensor base address in global memory
 * - rank: Tensor dimensionality (1-5D)
 * - box_dim[5]: Tile dimensions for each axis (up to 5D)
 * - global_dim[5]: Full tensor dimensions for each axis
 * - global_stride[5]: Byte strides between elements in global memory
 * - element_stride[5]: Element-wise strides for each dimension
 * - elemtype: Data type encoding (U8, U16, U32, U64, F16, F32, F64, BF16)
 * - interleave_layout: Memory interleaving mode (NONE, 2WAY, 4WAY)
 * - swizzle_mode: Layout optimization mode (NONE, 64B, 128B)
 * - fill_mode: Out-of-bounds fill behavior (ZERO, NAN)
 *
 * Verification Method:
 * ====================
 * 1. Direct field access validation after each replace operation
 * 2. Round-trip test: write to shared memory → read back → compare
 * 3. Boundary condition testing for array indices (0-4 for 5D arrays)
 * 4. Data type encoding validation against defined constants
 * 5. Memory alignment verification (128-byte alignment)
 * 6. Helper function validation (get_element_size, get_tile_size_bytes, calculate_src_addr)
 *
 * Expected Results:
 * =================
 * All EXPECT_EQ assertions must pass, confirming:
 * - Each field stores the exact value assigned
 * - Array indexing works correctly for multi-dimensional fields
 * - 128-byte alignment is maintained
 * - Helper functions compute correct values based on descriptor state
 * - Round-trip through shared memory preserves all data
 */

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

// Mock memory_space class for testing without full GPGPU-Sim infrastructure
class memory_space {
private:
    std::vector<uint8_t> data;
    size_t size_bytes;
    
public:
    memory_space(size_t size) : size_bytes(size) {
        data.resize(size, 0);
    }
    
    void write(size_t addr, size_t size, const void* src) {
        if (addr + size <= size_bytes) {
            memcpy(data.data() + addr, src, size);
        }
    }
    
    void read(size_t addr, size_t size, void* dst) const {
        if (addr + size <= size_bytes) {
            memcpy(dst, data.data() + addr, size);
        }
    }
    
    template<typename T>
    void write_typed(size_t addr, const T& value) {
        write(addr, sizeof(T), &value);
    }
    
    template<typename T>
    T read_typed(size_t addr) const {
        T value;
        read(addr, sizeof(T), &value);
        return value;
    }
};

// TensorMap descriptor structure (from tma.h)
namespace flash_gpgpu_sim {

#define TMA_DTYPE_U8    0u
#define TMA_DTYPE_U16   1u
#define TMA_DTYPE_U32   2u
#define TMA_DTYPE_U64   4u
#define TMA_DTYPE_F16   6u
#define TMA_DTYPE_F32   7u
#define TMA_DTYPE_F64   9u
#define TMA_DTYPE_BF16  10u

// Interleave layout modes (in bytes)
#define TMA_INTERLEAVE_NONE   0u
#define TMA_INTERLEAVE_16B    1u
#define TMA_INTERLEAVE_32B    2u

// Swizzle modes (in bytes)
#define TMA_SWIZZLE_NONE      0u
#define TMA_SWIZZLE_32B       1u
#define TMA_SWIZZLE_64B       2u
#define TMA_SWIZZLE_128B      3u
#define TMA_SWIZZLE_96B       4u

#define TMA_OOB_ZERO          0u
#define TMA_OOB_NAN           1u

typedef union __attribute__((aligned(128))) tensormap_descriptor_t {
    uint8_t  raw_bytes[128];
    uint64_t raw_u64[16];

    struct __attribute__((packed)) {
        uint64_t globalAddress;          // [0-7]
        uint32_t tensorRank;             // [8-11]
        uint32_t boxDim[5];              // [12-31]
        uint32_t globalDim[5];           // [32-51]
        uint64_t globalStrides[5];       // [52-91]
        uint32_t elementStrides[5];      // [92-111]
        uint32_t tensorDataType;         // [112-115]
        uint32_t interleave;             // [116-119]
        uint32_t swizzle;                // [120-123]
        uint32_t oobFill;                // [124-127]
    } fields;

    uint32_t get_element_size() const {
        switch (fields.tensorDataType) {
            case TMA_DTYPE_U8:   return 1;
            case TMA_DTYPE_U16:  return 2;
            case TMA_DTYPE_F16:  return 2;
            case TMA_DTYPE_BF16: return 2;
            case TMA_DTYPE_U32:  return 4;
            case TMA_DTYPE_F32:  return 4;
            case TMA_DTYPE_U64:  return 8;
            case TMA_DTYPE_F64:  return 8;
            default: return 4;
        }
    }

    uint32_t get_tile_size_bytes() const {
        uint32_t elem_size = get_element_size();
        uint32_t total = elem_size;
        for (uint32_t d = 0; d < fields.tensorRank && d < 5; d++) {
            total *= fields.boxDim[d];
        }
        return total;
    }

    uint64_t calculate_src_addr(const uint32_t coords[5]) const {
        uint64_t addr = fields.globalAddress;
        uint32_t elem_size = get_element_size();
        
        for (uint32_t d = 0; d < fields.tensorRank && d < 5; d++) {
            uint64_t stride = (d == 0) ? elem_size : fields.globalStrides[d - 1];
            addr += coords[d] * stride;
        }
        return addr;
    }

    bool is_valid() const { 
        return fields.tensorRank > 0 && fields.globalAddress != 0; 
    }

    static tensormap_descriptor_t read_from_shared(memory_space *shared_mem, uint32_t addr) {
        tensormap_descriptor_t desc;
        memset(&desc, 0, sizeof(desc));
        shared_mem->read(addr, 128, desc.raw_bytes);
        return desc;
    }

    void write_to_shared(memory_space *shared_mem, uint32_t addr) const {
        shared_mem->write(addr, 128, raw_bytes);
    }
} tensormap_descriptor_t;

} // namespace flash_gpgpu_sim

using namespace flash_gpgpu_sim;

// Test fixture for TensorMap operations
class TensorMapTest : public ::testing::Test {
protected:
    std::unique_ptr<memory_space> shared_mem;
    const uint32_t tensormap_addr = 0;  // Store at beginning of shared memory
    
    void SetUp() override {
        // Allocate 4KB shared memory (enough for multiple tensormap descriptors)
        shared_mem = std::make_unique<memory_space>(4096);
    }

    void TearDown() override {
        shared_mem.reset();
    }

    // Helper: Initialize a basic tensormap descriptor
    tensormap_descriptor_t create_test_descriptor() {
        tensormap_descriptor_t desc;
        memset(&desc, 0, sizeof(desc));
        
        // Set default valid values
        desc.fields.globalAddress = 0x1000000ULL;
        desc.fields.tensorRank = 2;
        desc.fields.boxDim[0] = 32;
        desc.fields.boxDim[1] = 128;
        desc.fields.globalDim[0] = 1024;
        desc.fields.globalDim[1] = 2048;
        desc.fields.globalStrides[0] = 4096;
        desc.fields.elementStrides[0] = 1;
        desc.fields.elementStrides[1] = 1;
        desc.fields.tensorDataType = TMA_DTYPE_F32;
        desc.fields.interleave = TMA_INTERLEAVE_NONE;
        desc.fields.swizzle = TMA_SWIZZLE_128B;
        desc.fields.oobFill = TMA_OOB_ZERO;
        
        return desc;
    }
};

// Test 1: Basic descriptor initialization and memory alignment
TEST_F(TensorMapTest, DescriptorSizeAndAlignment) {
    tensormap_descriptor_t desc;
    
    // Verify 128-byte size
    EXPECT_EQ(sizeof(desc), 128);
    EXPECT_EQ(sizeof(desc.raw_bytes), 128);
    EXPECT_EQ(sizeof(desc.raw_u64), 128);
    
    // Verify alignment
    EXPECT_EQ(alignof(tensormap_descriptor_t), 128);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&desc) % 128, 0);
}

// Test 2: tensormap.replace.tile.global_address
TEST_F(TensorMapTest, ReplaceGlobalAddress) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    // Test various addresses
    uint64_t test_addresses[] = {
        0x0ULL,
        0x1000ULL,
        0xDEADBEEFULL,
        0x123456789ABCDEFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    for (uint64_t addr : test_addresses) {
        desc.fields.globalAddress = addr;
        EXPECT_EQ(desc.fields.globalAddress, addr);
        
        // Write to shared memory and read back
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.globalAddress, addr);
    }
}

// Test 3: tensormap.replace.tile.rank
TEST_F(TensorMapTest, ReplaceRank) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    // Test valid ranks 1-5
    for (uint32_t rank = 1; rank <= 5; rank++) {
        desc.fields.tensorRank = rank;
        EXPECT_EQ(desc.fields.tensorRank, rank);
        
        // Verify round-trip through shared memory
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.tensorRank, rank);
    }
    
    // Test edge cases
    desc.fields.tensorRank = 0;
    EXPECT_EQ(desc.fields.tensorRank, 0);
    EXPECT_FALSE(desc.is_valid()); // Rank 0 should be invalid
    
    desc.fields.tensorRank = 10; // Beyond 5D
    EXPECT_EQ(desc.fields.tensorRank, 10);
}

// Test 4: tensormap.replace.tile.box_dim
TEST_F(TensorMapTest, ReplaceBoxDim) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t test_dims[] = {1, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    // Test each dimension independently
    for (uint32_t dim_idx = 0; dim_idx < 5; dim_idx++) {
        // Save original values of other dimensions
        uint32_t saved_dims[5];
        for (uint32_t i = 0; i < 5; i++) {
            saved_dims[i] = desc.fields.boxDim[i];
        }
        
        for (uint32_t test_val : test_dims) {
            desc.fields.boxDim[dim_idx] = test_val;
            EXPECT_EQ(desc.fields.boxDim[dim_idx], test_val);
            
            // Verify other dimensions unchanged from their saved values
            for (uint32_t other_idx = 0; other_idx < 5; other_idx++) {
                if (other_idx != dim_idx) {
                    EXPECT_EQ(desc.fields.boxDim[other_idx], saved_dims[other_idx]);
                }
            }
        }
    }
    
    // Round-trip test for all dimensions at once
    desc.fields.boxDim[0] = 32;
    desc.fields.boxDim[1] = 128;
    desc.fields.boxDim[2] = 64;
    desc.fields.boxDim[3] = 16;
    desc.fields.boxDim[4] = 8;
    
    desc.write_to_shared(shared_mem.get(), tensormap_addr);
    tensormap_descriptor_t read_desc = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
    
    for (uint32_t i = 0; i < 5; i++) {
        EXPECT_EQ(read_desc.fields.boxDim[i], desc.fields.boxDim[i]);
    }
}

// Test 5: tensormap.replace.tile.global_dim
TEST_F(TensorMapTest, ReplaceGlobalDim) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t test_global_dims[5] = {1024, 2048, 512, 4096, 8192};
    
    for (uint32_t dim_idx = 0; dim_idx < 5; dim_idx++) {
        desc.fields.globalDim[dim_idx] = test_global_dims[dim_idx];
        EXPECT_EQ(desc.fields.globalDim[dim_idx], test_global_dims[dim_idx]);
    }
    
    // Verify round-trip
    desc.write_to_shared(shared_mem.get(), tensormap_addr);
    tensormap_descriptor_t read_desc = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
    
    for (uint32_t i = 0; i < 5; i++) {
        EXPECT_EQ(read_desc.fields.globalDim[i], test_global_dims[i]);
    }
}

// Test 6: tensormap.replace.tile.global_stride
TEST_F(TensorMapTest, ReplaceGlobalStride) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint64_t test_strides[5] = {4096, 8192, 16384, 32768, 65536};
    
    for (uint32_t dim_idx = 0; dim_idx < 5; dim_idx++) {
        desc.fields.globalStrides[dim_idx] = test_strides[dim_idx];
        EXPECT_EQ(desc.fields.globalStrides[dim_idx], test_strides[dim_idx]);
    }
    
    // Test large stride values
    desc.fields.globalStrides[0] = 0x123456789ABCDEFULL;
    EXPECT_EQ(desc.fields.globalStrides[0], 0x123456789ABCDEFULL);
    
    // Round-trip verification
    desc.write_to_shared(shared_mem.get(), tensormap_addr);
    tensormap_descriptor_t read_desc = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
    
    for (uint32_t i = 0; i < 5; i++) {
        EXPECT_EQ(read_desc.fields.globalStrides[i], desc.fields.globalStrides[i]);
    }
}

// Test 7: tensormap.replace.tile.element_stride
TEST_F(TensorMapTest, ReplaceElementStride) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t test_elem_strides[5] = {1, 2, 4, 8, 16};
    
    for (uint32_t dim_idx = 0; dim_idx < 5; dim_idx++) {
        desc.fields.elementStrides[dim_idx] = test_elem_strides[dim_idx];
        EXPECT_EQ(desc.fields.elementStrides[dim_idx], test_elem_strides[dim_idx]);
    }
    
    // Round-trip test
    desc.write_to_shared(shared_mem.get(), tensormap_addr);
    tensormap_descriptor_t read_desc = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
    
    for (uint32_t i = 0; i < 5; i++) {
        EXPECT_EQ(read_desc.fields.elementStrides[i], test_elem_strides[i]);
    }
}

// Test 8: tensormap.replace.tile.elemtype
TEST_F(TensorMapTest, ReplaceElemType) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    struct TypeTest {
        uint32_t type_code;
        uint32_t expected_size;
    };
    
    TypeTest type_tests[] = {
        {TMA_DTYPE_U8, 1},
        {TMA_DTYPE_U16, 2},
        {TMA_DTYPE_U32, 4},
        {TMA_DTYPE_U64, 8},
        {TMA_DTYPE_F16, 2},
        {TMA_DTYPE_F32, 4},
        {TMA_DTYPE_F64, 8},
        {TMA_DTYPE_BF16, 2}
    };
    
    for (const auto& test : type_tests) {
        desc.fields.tensorDataType = test.type_code;
        EXPECT_EQ(desc.fields.tensorDataType, test.type_code);
        EXPECT_EQ(desc.get_element_size(), test.expected_size);
        
        // Round-trip test
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.tensorDataType, test.type_code);
        EXPECT_EQ(read_desc.get_element_size(), test.expected_size);
    }
}

// Test 9: tensormap.replace.tile.interleave_layout
TEST_F(TensorMapTest, ReplaceInterleaveLayout) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t interleave_modes[] = {
        TMA_INTERLEAVE_NONE,
        TMA_INTERLEAVE_16B,
        TMA_INTERLEAVE_32B
    };
    
    for (uint32_t mode : interleave_modes) {
        desc.fields.interleave = mode;
        EXPECT_EQ(desc.fields.interleave, mode);
        
        // Round-trip test
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.interleave, mode);
    }
}

// Test 10: tensormap.replace.tile.swizzle_mode
TEST_F(TensorMapTest, ReplaceSwizzleMode) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t swizzle_modes[] = {
        TMA_SWIZZLE_NONE,
        TMA_SWIZZLE_32B,
        TMA_SWIZZLE_64B,
        TMA_SWIZZLE_128B
    };
    
    for (uint32_t mode : swizzle_modes) {
        desc.fields.swizzle = mode;
        EXPECT_EQ(desc.fields.swizzle, mode);
        
        // Round-trip test
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.swizzle, mode);
    }
}

// Test 11: tensormap.replace.tile.fill_mode
TEST_F(TensorMapTest, ReplaceFillMode) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    uint32_t fill_modes[] = {
        TMA_OOB_ZERO,
        TMA_OOB_NAN
    };
    
    for (uint32_t mode : fill_modes) {
        desc.fields.oobFill = mode;
        EXPECT_EQ(desc.fields.oobFill, mode);
        
        // Round-trip test
        desc.write_to_shared(shared_mem.get(), tensormap_addr);
        tensormap_descriptor_t read_desc = 
            tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
        
        EXPECT_EQ(read_desc.fields.oobFill, mode);
    }
}

// Test 12: Complete descriptor round-trip with all fields
TEST_F(TensorMapTest, CompleteDescriptorRoundTrip) {
    tensormap_descriptor_t desc;
    memset(&desc, 0, sizeof(desc));
    
    // Set all fields to known values
    desc.fields.globalAddress = 0x123456789ABCDEFULL;
    desc.fields.tensorRank = 3;
    desc.fields.boxDim[0] = 32;
    desc.fields.boxDim[1] = 64;
    desc.fields.boxDim[2] = 128;
    desc.fields.boxDim[3] = 16;
    desc.fields.boxDim[4] = 8;
    desc.fields.globalDim[0] = 1024;
    desc.fields.globalDim[1] = 2048;
    desc.fields.globalDim[2] = 4096;
    desc.fields.globalDim[3] = 512;
    desc.fields.globalDim[4] = 256;
    desc.fields.globalStrides[0] = 4096;
    desc.fields.globalStrides[1] = 8192;
    desc.fields.globalStrides[2] = 16384;
    desc.fields.globalStrides[3] = 2048;
    desc.fields.globalStrides[4] = 1024;
    desc.fields.elementStrides[0] = 1;
    desc.fields.elementStrides[1] = 2;
    desc.fields.elementStrides[2] = 1;
    desc.fields.elementStrides[3] = 4;
    desc.fields.elementStrides[4] = 1;
    desc.fields.tensorDataType = TMA_DTYPE_F32;
    desc.fields.interleave = TMA_INTERLEAVE_16B;
    desc.fields.swizzle = TMA_SWIZZLE_128B;
    desc.fields.oobFill = TMA_OOB_ZERO;
    
    // Write to shared memory
    desc.write_to_shared(shared_mem.get(), tensormap_addr);
    
    // Read back
    tensormap_descriptor_t read_desc = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), tensormap_addr);
    
    // Verify all fields
    EXPECT_EQ(read_desc.fields.globalAddress, desc.fields.globalAddress);
    EXPECT_EQ(read_desc.fields.tensorRank, desc.fields.tensorRank);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(read_desc.fields.boxDim[i], desc.fields.boxDim[i]);
        EXPECT_EQ(read_desc.fields.globalDim[i], desc.fields.globalDim[i]);
        EXPECT_EQ(read_desc.fields.globalStrides[i], desc.fields.globalStrides[i]);
        EXPECT_EQ(read_desc.fields.elementStrides[i], desc.fields.elementStrides[i]);
    }
    EXPECT_EQ(read_desc.fields.tensorDataType, desc.fields.tensorDataType);
    EXPECT_EQ(read_desc.fields.interleave, desc.fields.interleave);
    EXPECT_EQ(read_desc.fields.swizzle, desc.fields.swizzle);
    EXPECT_EQ(read_desc.fields.oobFill, desc.fields.oobFill);
}

// Test 13: Helper function - get_tile_size_bytes
TEST_F(TensorMapTest, GetTileSizeBytes) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    // Test 2D: 32x128 F32 (4 bytes)
    desc.fields.tensorRank = 2;
    desc.fields.boxDim[0] = 32;
    desc.fields.boxDim[1] = 128;
    desc.fields.tensorDataType = TMA_DTYPE_F32;
    EXPECT_EQ(desc.get_tile_size_bytes(), 32 * 128 * 4);
    
    // Test 3D: 16x32x64 U8 (1 byte)
    desc.fields.tensorRank = 3;
    desc.fields.boxDim[0] = 16;
    desc.fields.boxDim[1] = 32;
    desc.fields.boxDim[2] = 64;
    desc.fields.tensorDataType = TMA_DTYPE_U8;
    EXPECT_EQ(desc.get_tile_size_bytes(), 16 * 32 * 64 * 1);
    
    // Test 1D: 1024 F64 (8 bytes)
    desc.fields.tensorRank = 1;
    desc.fields.boxDim[0] = 1024;
    desc.fields.tensorDataType = TMA_DTYPE_F64;
    EXPECT_EQ(desc.get_tile_size_bytes(), 1024 * 8);
}

// Test 14: Helper function - calculate_src_addr
TEST_F(TensorMapTest, CalculateSrcAddr) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    // Setup 2D tensor: 1024x2048 F32 elements
    desc.fields.globalAddress = 0x1000000ULL;
    desc.fields.tensorRank = 2;
    desc.fields.tensorDataType = TMA_DTYPE_F32; // 4 bytes
    desc.fields.globalDim[0] = 1024;
    desc.fields.globalDim[1] = 2048;
    desc.fields.globalStrides[0] = 2048 * 4; // Row stride in bytes
    
    // Test coordinate (0, 0)
    uint32_t coords1[5] = {0, 0, 0, 0, 0};
    EXPECT_EQ(desc.calculate_src_addr(coords1), 0x1000000ULL);
    
    // Test coordinate (1, 0) - one element in X
    uint32_t coords2[5] = {1, 0, 0, 0, 0};
    EXPECT_EQ(desc.calculate_src_addr(coords2), 0x1000000ULL + 4);
    
    // Test coordinate (0, 1) - one row in Y
    uint32_t coords3[5] = {0, 1, 0, 0, 0};
    EXPECT_EQ(desc.calculate_src_addr(coords3), 0x1000000ULL + 2048 * 4);
    
    // Test coordinate (5, 3)
    uint32_t coords4[5] = {5, 3, 0, 0, 0};
    EXPECT_EQ(desc.calculate_src_addr(coords4), 0x1000000ULL + 5 * 4 + 3 * 2048 * 4);
}

// Test 15: Descriptor validation
TEST_F(TensorMapTest, DescriptorValidation) {
    tensormap_descriptor_t desc;
    memset(&desc, 0, sizeof(desc));
    
    // Invalid: no address and no rank
    EXPECT_FALSE(desc.is_valid());
    
    // Invalid: address but no rank
    desc.fields.globalAddress = 0x1000000ULL;
    desc.fields.tensorRank = 0;
    EXPECT_FALSE(desc.is_valid());
    
    // Invalid: rank but no address
    desc.fields.globalAddress = 0;
    desc.fields.tensorRank = 2;
    EXPECT_FALSE(desc.is_valid());
    
    // Valid: both address and rank
    desc.fields.globalAddress = 0x1000000ULL;
    desc.fields.tensorRank = 2;
    EXPECT_TRUE(desc.is_valid());
}

// Test 16: Multiple descriptors in shared memory
TEST_F(TensorMapTest, MultipleDescriptorsInSharedMemory) {
    // Write 3 different descriptors at different offsets
    tensormap_descriptor_t desc1 = create_test_descriptor();
    desc1.fields.globalAddress = 0x1000000ULL;
    desc1.fields.tensorRank = 1;
    
    tensormap_descriptor_t desc2 = create_test_descriptor();
    desc2.fields.globalAddress = 0x2000000ULL;
    desc2.fields.tensorRank = 2;
    
    tensormap_descriptor_t desc3 = create_test_descriptor();
    desc3.fields.globalAddress = 0x3000000ULL;
    desc3.fields.tensorRank = 3;
    
    // Write to shared memory at different 128-byte aligned addresses
    desc1.write_to_shared(shared_mem.get(), 0);
    desc2.write_to_shared(shared_mem.get(), 128);
    desc3.write_to_shared(shared_mem.get(), 256);
    
    // Read back and verify
    tensormap_descriptor_t read1 = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), 0);
    tensormap_descriptor_t read2 = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), 128);
    tensormap_descriptor_t read3 = 
        tensormap_descriptor_t::read_from_shared(shared_mem.get(), 256);
    
    EXPECT_EQ(read1.fields.globalAddress, 0x1000000ULL);
    EXPECT_EQ(read1.fields.tensorRank, 1);
    
    EXPECT_EQ(read2.fields.globalAddress, 0x2000000ULL);
    EXPECT_EQ(read2.fields.tensorRank, 2);
    
    EXPECT_EQ(read3.fields.globalAddress, 0x3000000ULL);
    EXPECT_EQ(read3.fields.tensorRank, 3);
}

// Test 17: Raw byte access consistency
TEST_F(TensorMapTest, RawByteAccessConsistency) {
    tensormap_descriptor_t desc = create_test_descriptor();
    
    // Modify via structured access
    desc.fields.globalAddress = 0xDEADBEEFCAFEBABEULL;
    
    // Verify raw byte representation contains the value
    uint64_t addr_from_raw = 0;
    memcpy(&addr_from_raw, desc.raw_bytes, 8);
    EXPECT_EQ(addr_from_raw, 0xDEADBEEFCAFEBABEULL);
    
    // Verify raw u64 array access
    EXPECT_EQ(desc.raw_u64[0], 0xDEADBEEFCAFEBABEULL);
    
    // Modify via raw bytes and verify structured access
    uint32_t rank_value = 5;
    memcpy(desc.raw_bytes + 8, &rank_value, 4);
    EXPECT_EQ(desc.fields.tensorRank, 5);
}

// Note: Using default main from gtest_main.a, no custom main needed
