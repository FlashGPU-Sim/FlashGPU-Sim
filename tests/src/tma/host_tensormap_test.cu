// Host-side TensorMap Test using cuTensorMapEncodeTiled
// This test demonstrates:
// 1. Creating tensormap on host using cuTensorMapEncodeTiled
// 2. Using TMA to load data from global to shared memory
// 3. Performing simple computation (+1)
// 4. Using TMA to store data back to global memory

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

#include "ptx/mbarrier.cuh"

using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;

// CUtensorMap is an opaque 128-byte structure
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap must be 128 bytes");

// Fence for TMA operations
__device__ inline void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;");
}

// TMA Load: 1D tensor from global to shared memory
__device__ inline void tma_load_1d(
    uint32_t smem_addr, const CUtensorMap *tmap_ptr, int32_t coord0, uint32_t mbar_addr) {
    uint64_t tmap_addr = reinterpret_cast<uint64_t>(tmap_ptr);
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2}], [%3];"
                 :: "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(mbar_addr));
}

// TMA Store: 1D tensor from shared to global memory
__device__ inline void tma_store_1d(
    const CUtensorMap *tmap_ptr, int32_t coord0, uint32_t smem_addr) {
    uint64_t tmap_addr = reinterpret_cast<uint64_t>(tmap_ptr);
    asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group "
                 "[%0, {%1}], [%2];"
                 :: "l"(tmap_addr), "r"(coord0), "r"(smem_addr));
}

// Bulk group operations for TMA store
__device__ inline void bulk_group_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

template<int N>
__device__ inline void bulk_group_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N));
}

// Kernel: Load data using TMA, add 1, store back using TMA
template <int TILE_SIZE>
__global__ void tma_add_one_kernel(
    const CUtensorMap *load_tmap,
    const CUtensorMap *store_tmap,
    int num_tiles) {

    constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);
    extern __shared__ uint8_t smem[];

    // Layout: [data TILE_BYTES][mbarrier 8B]
    float *data = reinterpret_cast<float*>(smem);
    uint64_t *bar = reinterpret_cast<uint64_t*>(smem + TILE_BYTES);

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_tiles) return;

    int coord0 = bid * TILE_SIZE;  // Starting element index

    // Initialize mbarrier (once per block)
    if (tid == 0) {
        mbarrier_init(bar, 1);
    }
    __syncthreads();

    // TMA Load
    if (tid == 0) {
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        tma_load_1d(smem_addr, load_tmap, coord0, mbar_addr);
    }

    // Wait for TMA load to complete
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
    }
    fence_proxy_async();
    __syncthreads();

    // Compute: Add 1 to each element
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        data[i] = data[i] + 1.0f;
    }
    __syncthreads();

    // TMA Store
    if (tid == 0) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        tma_store_1d(store_tmap, coord0, smem_addr);
        bulk_group_commit();
    }

    // Wait for TMA store to complete
    if (tid == 0) {
        bulk_group_wait<0>();
    }
    __syncthreads();

    // Invalidate mbarrier
    if (tid == 0) {
        mbarrier_inval(bar);
    }
}

// Test fixture
class HostTensorMapTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA Driver API
        CUresult result = cuInit(0);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to initialize CUDA Driver API";
    }

    void TearDown() override {
    }

    // Helper to check CUDA errors
    void checkCudaError(cudaError_t error, const char *message) {
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        }
        ASSERT_EQ(error, cudaSuccess);
    }

    void checkCuError(CUresult result, const char *message) {
        if (result != CUDA_SUCCESS) {
            const char *error_string;
            cuGetErrorString(result, &error_string);
            fprintf(stderr, "CUDA Driver Error: %s - %s\n", message, error_string);
        }
        ASSERT_EQ(result, CUDA_SUCCESS);
    }
};

// Test: Basic 1D TMA with host-side tensormap creation
TEST_F(HostTensorMapTest, Basic1DTMAAddOne) {
    constexpr int TILE_SIZE = 256;       // Elements per tile
    constexpr int NUM_TILES = 4;
    constexpr int NUM_ELEMENTS = TILE_SIZE * NUM_TILES;
    constexpr int DATA_BYTES = NUM_ELEMENTS * sizeof(float);

    // Allocate host memory
    std::vector<float> h_input(NUM_ELEMENTS);
    std::vector<float> h_output(NUM_ELEMENTS);

    // Initialize input data
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    checkCudaError(cudaMalloc(&d_input, DATA_BYTES), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, DATA_BYTES), "cudaMalloc d_output");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input.data(), DATA_BYTES, cudaMemcpyHostToDevice),
                   "cudaMemcpy input to device");
    checkCudaError(cudaMemset(d_output, 0, DATA_BYTES), "cudaMemset d_output");

    // Create tensormaps on host using cuTensorMapEncodeTiled
    CUtensorMap load_tmap, store_tmap;

    // 1D tensormap parameters
    const uint64_t globalDim[1] = {NUM_ELEMENTS};
    const uint64_t globalStrides[1] = {1 * sizeof(float)};  // Stride in bytes
    const uint32_t boxDim[1] = {TILE_SIZE};                 // Tile size
    const uint32_t elementStrides[1] = {1};                 // Element stride

    // Create load tensormap
    CUresult result = cuTensorMapEncodeTiled(
        &load_tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        1,  // tensorRank (1D)
        d_input,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    checkCuError(result, "cuTensorMapEncodeTiled for load");

    // Create store tensormap
    result = cuTensorMapEncodeTiled(
        &store_tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        1,  // tensorRank (1D)
        d_output,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    checkCuError(result, "cuTensorMapEncodeTiled for store");

    // Copy tensormaps to device
    CUtensorMap *d_load_tmap = nullptr;
    CUtensorMap *d_store_tmap = nullptr;
    checkCudaError(cudaMalloc(&d_load_tmap, sizeof(CUtensorMap)), "cudaMalloc d_load_tmap");
    checkCudaError(cudaMalloc(&d_store_tmap, sizeof(CUtensorMap)), "cudaMalloc d_store_tmap");
    checkCudaError(cudaMemcpy(d_load_tmap, &load_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy load_tmap to device");
    checkCudaError(cudaMemcpy(d_store_tmap, &store_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy store_tmap to device");

    // Launch kernel
    constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);
    constexpr int SHARED_MEM_SIZE = TILE_BYTES + 8;  // data + mbarrier
    const int threads_per_block = 256;

    tma_add_one_kernel<TILE_SIZE>
        <<<NUM_TILES, threads_per_block, SHARED_MEM_SIZE>>>(
            d_load_tmap, d_store_tmap, NUM_TILES);

    cudaError_t kernel_error = cudaGetLastError();
    checkCudaError(kernel_error, "Kernel launch");

    cudaError_t sync_error = cudaDeviceSynchronize();
    checkCudaError(sync_error, "Kernel execution");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output.data(), d_output, DATA_BYTES, cudaMemcpyDeviceToHost),
                   "cudaMemcpy output to host");

    // Verify results: each element should be input + 1
    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        float expected = h_input[i] + 1.0f;
        if (h_output[i] != expected) {
            if (errors < 10) {
                printf("Mismatch at %d: expected %.1f, got %.1f\n", i, expected, h_output[i]);
            }
            errors++;
        }
    }

    EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_load_tmap);
    cudaFree(d_store_tmap);
}

// Test: Multiple blocks processing different tiles
TEST_F(HostTensorMapTest, MultiBlockTMA) {
    constexpr int TILE_SIZE = 128;
    constexpr int NUM_TILES = 8;
    constexpr int NUM_ELEMENTS = TILE_SIZE * NUM_TILES;
    constexpr int DATA_BYTES = NUM_ELEMENTS * sizeof(float);

    std::vector<float> h_input(NUM_ELEMENTS);
    std::vector<float> h_output(NUM_ELEMENTS);

    // Initialize with sequential values
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_input[i] = static_cast<float>(i * 2);
    }

    float *d_input = nullptr;
    float *d_output = nullptr;
    checkCudaError(cudaMalloc(&d_input, DATA_BYTES), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, DATA_BYTES), "cudaMalloc d_output");
    checkCudaError(cudaMemcpy(d_input, h_input.data(), DATA_BYTES, cudaMemcpyHostToDevice),
                   "cudaMemcpy input");
    checkCudaError(cudaMemset(d_output, 0, DATA_BYTES), "cudaMemset output");

    // Create tensormaps
    CUtensorMap load_tmap, store_tmap;
    const uint64_t globalDim[1] = {NUM_ELEMENTS};
    const uint64_t globalStrides[1] = {1 * sizeof(float)};
    const uint32_t boxDim[1] = {TILE_SIZE};
    const uint32_t elementStrides[1] = {1};

    checkCuError(cuTensorMapEncodeTiled(
        &load_tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, d_input,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled load");

    checkCuError(cuTensorMapEncodeTiled(
        &store_tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, d_output,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled store");

    CUtensorMap *d_load_tmap = nullptr;
    CUtensorMap *d_store_tmap = nullptr;
    checkCudaError(cudaMalloc(&d_load_tmap, sizeof(CUtensorMap)), "cudaMalloc load_tmap");
    checkCudaError(cudaMalloc(&d_store_tmap, sizeof(CUtensorMap)), "cudaMalloc store_tmap");
    checkCudaError(cudaMemcpy(d_load_tmap, &load_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy load_tmap");
    checkCudaError(cudaMemcpy(d_store_tmap, &store_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy store_tmap");

    constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);
    constexpr int SHARED_MEM_SIZE = TILE_BYTES + 8;
    const int threads_per_block = 128;

    tma_add_one_kernel<TILE_SIZE>
        <<<NUM_TILES, threads_per_block, SHARED_MEM_SIZE>>>(
            d_load_tmap, d_store_tmap, NUM_TILES);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");
    checkCudaError(cudaMemcpy(h_output.data(), d_output, DATA_BYTES, cudaMemcpyDeviceToHost),
                   "cudaMemcpy output");

    // Verify
    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        float expected = h_input[i] + 1.0f;
        if (h_output[i] != expected) {
            if (errors < 10) {
                printf("Block %d, elem %d: expected %.1f, got %.1f\n",
                       i / TILE_SIZE, i % TILE_SIZE, expected, h_output[i]);
            }
            errors++;
        }
    }

    EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_load_tmap);
    cudaFree(d_store_tmap);
}

// Test: Small tile size
TEST_F(HostTensorMapTest, SmallTile) {
    constexpr int TILE_SIZE = 64;
    constexpr int NUM_TILES = 2;
    constexpr int NUM_ELEMENTS = TILE_SIZE * NUM_TILES;
    constexpr int DATA_BYTES = NUM_ELEMENTS * sizeof(float);

    std::vector<float> h_input(NUM_ELEMENTS, 42.0f);
    std::vector<float> h_output(NUM_ELEMENTS);

    float *d_input = nullptr;
    float *d_output = nullptr;
    checkCudaError(cudaMalloc(&d_input, DATA_BYTES), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, DATA_BYTES), "cudaMalloc d_output");
    checkCudaError(cudaMemcpy(d_input, h_input.data(), DATA_BYTES, cudaMemcpyHostToDevice),
                   "cudaMemcpy input");
    checkCudaError(cudaMemset(d_output, 0, DATA_BYTES), "cudaMemset output");

    CUtensorMap load_tmap, store_tmap;
    const uint64_t globalDim[1] = {NUM_ELEMENTS};
    const uint64_t globalStrides[1] = {1 * sizeof(float)};
    const uint32_t boxDim[1] = {TILE_SIZE};
    const uint32_t elementStrides[1] = {1};

    checkCuError(cuTensorMapEncodeTiled(
        &load_tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, d_input,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled load");

    checkCuError(cuTensorMapEncodeTiled(
        &store_tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, d_output,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled store");

    CUtensorMap *d_load_tmap = nullptr;
    CUtensorMap *d_store_tmap = nullptr;
    checkCudaError(cudaMalloc(&d_load_tmap, sizeof(CUtensorMap)), "cudaMalloc load_tmap");
    checkCudaError(cudaMalloc(&d_store_tmap, sizeof(CUtensorMap)), "cudaMalloc store_tmap");
    checkCudaError(cudaMemcpy(d_load_tmap, &load_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy load_tmap");
    checkCudaError(cudaMemcpy(d_store_tmap, &store_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                   "cudaMemcpy store_tmap");

    constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);
    constexpr int SHARED_MEM_SIZE = TILE_BYTES + 8;
    const int threads_per_block = 64;

    tma_add_one_kernel<TILE_SIZE>
        <<<NUM_TILES, threads_per_block, SHARED_MEM_SIZE>>>(
            d_load_tmap, d_store_tmap, NUM_TILES);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");
    checkCudaError(cudaMemcpy(h_output.data(), d_output, DATA_BYTES, cudaMemcpyDeviceToHost),
                   "cudaMemcpy output");

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        EXPECT_EQ(h_output[i], 43.0f) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_load_tmap);
    cudaFree(d_store_tmap);
}
