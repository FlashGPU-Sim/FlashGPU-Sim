// Multi-dimensional TMA tests using cp.async.bulk.tensor with inline PTX tensormap
// Coverage: 1D, 2D, 3D, 4D, 5D tensors
// Uses tensormap.replace.* to dynamically construct tensormap (Triton style)

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cmath>
#include <vector>

#include "ptx/mbarrier.cuh"
#include "ptx/tma.cuh"

using flashgpu::test::ptx::cp_async_bulk_tensor_1d_load;
using flashgpu::test::ptx::cp_async_bulk_tensor_2d_load;
using flashgpu::test::ptx::cp_async_bulk_tensor_3d_load;
using flashgpu::test::ptx::cp_async_bulk_tensor_4d_load;
using flashgpu::test::ptx::cp_async_bulk_tensor_5d_load;
using flashgpu::test::ptx::fence_proxy_tensormap_acquire;
using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;
using flashgpu::test::ptx::tensormap_cp_fenceproxy;
using flashgpu::test::ptx::tensormap_set_global_address;

// ============================================================================
// Tensormap size: 128 bytes (1024 bits)
// ============================================================================
constexpr int TENSORMAP_SIZE = 128;

// ============================================================================
// Tensormap helper - initialize to zeros in shared memory
// ============================================================================

__device__ inline void tensormap_init_smem(void* tmap_smem, int tid) {
    // First 32 threads each write 4 bytes of zeros
    uint32_t* p = reinterpret_cast<uint32_t*>(tmap_smem);
    if (tid < 32) {
        p[tid] = 0;
    }
    asm volatile("bar.warp.sync -1;");
}

// Note: Other tensormap fields (rank, box_dim, global_dim, etc.) must use 
// immediate values in inline asm since PTX requires compile-time constants
// for the second operand. See kernel implementations below for usage.

// ============================================================================
// 1D TMA Kernel using cp.async.bulk.tensor.1d
// ============================================================================

template <int BOX_DIM>
__global__ void tma_tensor_1d_kernel(const float *in, float *out, int N, uint8_t *global_scratch) {
    constexpr int TILE_BYTES = BOX_DIM * sizeof(float);
    extern __shared__ uint8_t smem[];
    
    // Manual 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);
    
    // Layout: [tensormap 128B][data TILE_BYTES][mbarrier 8B]
    float* data = reinterpret_cast<float*>(tmap_smem + TENSORMAP_SIZE);
    uint64_t* bar = reinterpret_cast<uint64_t*>(tmap_smem + TENSORMAP_SIZE + TILE_BYTES);
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;



    int coord0 = bid * BOX_DIM;  // Starting element index
    
    // Per-block global tensormap location
    uint8_t* global_tmap = global_scratch + bid * 256;
    
    // Initialize tensormap in shared memory
    tensormap_init_smem(tmap_smem, tid);
    __syncthreads();
    
    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        
        // Set tensormap fields (following Triton 1D pattern)
        tensormap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        
        uint32_t box = BOX_DIM;
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box));
        
        uint32_t gdim = static_cast<uint32_t>(N);
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gdim));
        
        uint32_t estride = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(estride));
        
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    
    // Copy tensormap to global and fence
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tensormap_cp_fenceproxy(tmap_g, tmap_s);
        fence_proxy_tensormap_acquire(tmap_g);
    }
    __syncthreads();
    
    // Initialize mbarrier
    if (tid == 0) {
        mbarrier_init(bar, 1);
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    
    // Issue TMA load - elect one thread
    uint32_t elected;
    uint32_t pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);
    
    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        cp_async_bulk_tensor_1d_load(smem_data, tmap_g, coord0, mbar_addr);
    }
    __syncthreads();
    
    // Wait for TMA
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
        mbarrier_inval(bar);
    }
    __syncthreads();
    
    // Add 1.0f
    if (tid < BOX_DIM && coord0 + tid < N) {
        data[tid] += 1.0f;
    }
    __syncthreads();
    
    // Write back
    if (tid < BOX_DIM && coord0 + tid < N) {
        out[coord0 + tid] = data[tid];
    }
}

// ============================================================================
// 2D TMA Kernel using cp.async.bulk.tensor.2d
// ============================================================================

template <int BOX0, int BOX1>
__global__ void tma_tensor_2d_kernel(const float *in, float *out, int D0, int D1, uint8_t *global_scratch) {
    constexpr int TILE_ELEMS = BOX0 * BOX1;
    constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);
    extern __shared__ uint8_t smem[];
    
    // Manual 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);
    
    float* data = reinterpret_cast<float*>(tmap_smem + TENSORMAP_SIZE);
    uint64_t* bar = reinterpret_cast<uint64_t*>(tmap_smem + TENSORMAP_SIZE + TILE_BYTES);
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;



    int coord0 = blockIdx.x * BOX0;
    int coord1 = blockIdx.y * BOX1;
    
    uint8_t* global_tmap = global_scratch + (blockIdx.y * gridDim.x + blockIdx.x) * 256;
    
    tensormap_init_smem(tmap_smem, tid);
    __syncthreads();
    
    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        
        tensormap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x1;" :: "l"(tmap_s));
        
        uint32_t box0 = BOX0, box1 = BOX1;
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box0));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box1));
        
        uint32_t gd0 = D0, gd1 = D1;
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));
        
        uint64_t stride0 = D0 * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(stride0));
        
        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));
        // Only dim 0 needs element stride if contiguous? Removing dim 1 stride just in case.
        
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tensormap_cp_fenceproxy(tmap_g, tmap_s);
        fence_proxy_tensormap_acquire(tmap_g);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_init(bar, 1);
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    
    uint32_t elected;
    uint32_t pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);
    
    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        cp_async_bulk_tensor_2d_load(smem_data, tmap_g, coord0, coord1, mbar_addr);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
        mbarrier_inval(bar);
    }
    __syncthreads();
    
    int lx = threadIdx.x, ly = threadIdx.y;
    int gx = coord0 + lx, gy = coord1 + ly;
    if (lx < BOX0 && ly < BOX1 && gx < D0 && gy < D1) {
        data[ly * BOX0 + lx] += 1.0f;
    }
    __syncthreads();
    
    if (lx < BOX0 && ly < BOX1 && gx < D0 && gy < D1) {
        out[gy * D0 + gx] = data[ly * BOX0 + lx];
    }
}

// ============================================================================
// 3D TMA Kernel using cp.async.bulk.tensor.3d  
// ============================================================================

template <int BOX0, int BOX1, int BOX2>
__global__ void tma_tensor_3d_kernel(const float *in, float *out, 
                                      int D0, int D1, int D2, uint8_t *global_scratch) {
    constexpr int TILE_ELEMS = BOX0 * BOX1 * BOX2;
    constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);
    extern __shared__ uint8_t smem[];
    
    // Manual 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);
    
    float* data = reinterpret_cast<float*>(tmap_smem + TENSORMAP_SIZE);
    uint64_t* bar = reinterpret_cast<uint64_t*>(tmap_smem + TENSORMAP_SIZE + TILE_BYTES);
    
    int tid = threadIdx.x;

    int coord0 = blockIdx.x * BOX0;
    int coord1 = blockIdx.y * BOX1;
    int coord2 = blockIdx.z * BOX2;
    
    uint8_t* global_tmap = global_scratch + 
        ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 256;
    
    tensormap_init_smem(tmap_smem, tid);
    __syncthreads();
    
    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        
        tensormap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x2;" :: "l"(tmap_s));
        
        uint32_t box0 = BOX0, box1 = BOX1, box2 = BOX2;
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box0));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box1));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(box2));
        
        uint32_t gd0 = D0, gd1 = D1, gd2 = D2;
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(gd2));
        
        uint64_t stride0 = D0 * sizeof(float);
        uint64_t stride1 = D0 * D1 * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(stride0));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x1, %1;" :: "l"(tmap_s), "l"(stride1));
        
        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));

        
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tensormap_cp_fenceproxy(tmap_g, tmap_s);
        fence_proxy_tensormap_acquire(tmap_g);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_init(bar, 1);
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    
    uint32_t elected;
    uint32_t pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);
    
    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        cp_async_bulk_tensor_3d_load(smem_data, tmap_g, coord0, coord1, coord2, mbar_addr);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
        mbarrier_inval(bar);
    }
    __syncthreads();
    
    // Process all elements with single thread loop (simple approach)
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int lx = i % BOX0;
        int ly = (i / BOX0) % BOX1;
        int lz = i / (BOX0 * BOX1);
        int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz;
        if (gx < D0 && gy < D1 && gz < D2) {
            data[i] += 1.0f;
        }
    }
    __syncthreads();
    
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int lx = i % BOX0;
        int ly = (i / BOX0) % BOX1;
        int lz = i / (BOX0 * BOX1);
        int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz;
        if (gx < D0 && gy < D1 && gz < D2) {
            out[gz * D1 * D0 + gy * D0 + gx] = data[i];
        }
    }
}

// ============================================================================
// Test Fixtures
// ============================================================================

class TMA1DTest : public ::testing::Test {
protected:
    static constexpr int BOX = 128;
    void Run(int N, int seed = 42) {
        std::vector<float> h_in(N), h_out(N);
        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;
        
        ASSERT_EQ(cudaMalloc(&d_in, N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, N * sizeof(float)), cudaSuccess);
        
        int grid = (N + BOX - 1) / BOX;
        ASSERT_EQ(cudaMalloc(&d_scratch, grid * 256), cudaSuccess);
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (int i = 0; i < N; i++) h_in[i] = dist(gen);
        ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), N * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_out, 0, N * sizeof(float)), cudaSuccess);
        
        size_t smem = TENSORMAP_SIZE + BOX * sizeof(float) + 16 + 128; // +128 for alignment padding
        tma_tensor_1d_kernel<BOX><<<grid, BOX, smem>>>(d_in, d_out, N, d_scratch);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, N * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (int i = 0; i < N; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f) << "at " << i;
        
        EXPECT_EQ(cudaFree(d_in), cudaSuccess);
        EXPECT_EQ(cudaFree(d_out), cudaSuccess);
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }
};

TEST_F(TMA1DTest, Size_1024) { Run(1024); }
TEST_F(TMA1DTest, Size_8192) { Run(8192, 123); }
TEST_F(TMA1DTest, Size_256) { Run(256, 456); }

class TMA2DTest : public ::testing::Test {
protected:
    static constexpr int BOX0 = 16, BOX1 = 16;
    void Run(int D0, int D1, int seed = 42) {
        size_t tot = (size_t)D0 * D1;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;
        
        ASSERT_EQ(cudaMalloc(&d_in, tot * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, tot * sizeof(float)), cudaSuccess);
        
        dim3 grid((D0 + BOX0 - 1) / BOX0, (D1 + BOX1 - 1) / BOX1);
        ASSERT_EQ(cudaMalloc(&d_scratch, grid.x * grid.y * 256), cudaSuccess);
        
        for (int y = 0; y < D1; ++y) {
            for (int x = 0; x < D0; ++x) {
                h_in[y * D0 + x] =
                    static_cast<float>(seed + y * 4096 + x);
            }
        }
        ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), tot * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_out, 0, tot * sizeof(float)), cudaSuccess);
        
        dim3 block(BOX0, BOX1);
        size_t smem = TENSORMAP_SIZE + BOX0 * BOX1 * sizeof(float) + 16 + 128; // +128 for alignment padding
        tma_tensor_2d_kernel<BOX0, BOX1><<<grid, block, smem>>>(d_in, d_out, D0, D1, d_scratch);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, tot * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f)
                << " Mismatch at index [" << i / D0 << ", " << i % D0 << "]";
        
        EXPECT_EQ(cudaFree(d_in), cudaSuccess);
        EXPECT_EQ(cudaFree(d_out), cudaSuccess);
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }
};

TEST_F(TMA2DTest, Square_64x64) { Run(64, 64); }
TEST_F(TMA2DTest, Rect_128x32) { Run(128, 32, 123); }
TEST_F(TMA2DTest, Large_256x256) { Run(256, 256, 456); }

class TMA3DTest : public ::testing::Test {
protected:
    static constexpr int BOX0 = 16, BOX1 = 16, BOX2 = 16;
    void Run(int D0, int D1, int D2, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;
        
        ASSERT_EQ(cudaMalloc(&d_in, tot * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, tot * sizeof(float)), cudaSuccess);
        
        dim3 grid((D0 + BOX0 - 1) / BOX0, (D1 + BOX1 - 1) / BOX1, (D2 + BOX2 - 1) / BOX2);
        ASSERT_EQ(cudaMalloc(&d_scratch, grid.x * grid.y * grid.z * 256),
                  cudaSuccess);
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), tot * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_out, 0, tot * sizeof(float)), cudaSuccess);
        
        size_t smem = TENSORMAP_SIZE + BOX0 * BOX1 * BOX2 * sizeof(float) + 16 + 128; // +128 for alignment padding
        tma_tensor_3d_kernel<BOX0, BOX1, BOX2><<<grid, 128, smem>>>(d_in, d_out, D0, D1, D2, d_scratch);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, tot * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        
        EXPECT_EQ(cudaFree(d_in), cudaSuccess);
        EXPECT_EQ(cudaFree(d_out), cudaSuccess);
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }
};

TEST_F(TMA3DTest, Cube_16x16x16) { Run(16, 16, 16); }
TEST_F(TMA3DTest, Cube_32x32x32) { Run(32, 32, 32); }
TEST_F(TMA3DTest, Mixed_64x32x16) { Run(64, 32, 16); }
TEST_F(TMA3DTest, Flat_16x16x1) { Run(16, 16, 1); }

// ============================================================================
// 4D TMA Kernel using cp.async.bulk.tensor.4d
// ============================================================================

template <int BOX0, int BOX1, int BOX2, int BOX3>
__global__ void tma_tensor_4d_kernel(const float *in, float *out,
                                      int D0, int D1, int D2, int D3, uint8_t *global_scratch) {
    constexpr int TILE_ELEMS = BOX0 * BOX1 * BOX2 * BOX3;
    constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);
    extern __shared__ uint8_t smem[];
    
    // Manual 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);
    
    float* data = reinterpret_cast<float*>(tmap_smem + TENSORMAP_SIZE);
    uint64_t* bar = reinterpret_cast<uint64_t*>(tmap_smem + TENSORMAP_SIZE + TILE_BYTES);
    
    int tid = threadIdx.x;




    
    // Linearized block index
    int linear_bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    
    const int tiles2 = (D2 + BOX2 - 1) / BOX2;
    const int tile2 = blockIdx.z % tiles2;
    const int tile3 = blockIdx.z / tiles2;
    int coord0 = blockIdx.x * BOX0;
    int coord1 = blockIdx.y * BOX1;
    int coord2 = tile2 * BOX2;
    int coord3 = tile3 * BOX3;
    
    uint8_t* global_tmap = global_scratch + linear_bid * 256;
    
    tensormap_init_smem(tmap_smem, tid);
    __syncthreads();
    
    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        
        tensormap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x3;" :: "l"(tmap_s));
        
        uint32_t box0 = BOX0, box1 = BOX1, box2 = BOX2, box3 = BOX3;
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box0));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box1));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(box2));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x3, %1;" :: "l"(tmap_s), "r"(box3));
        
        uint32_t gd0 = D0, gd1 = D1, gd2 = D2, gd3 = D3;
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(gd2));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x3, %1;" :: "l"(tmap_s), "r"(gd3));
        
        uint64_t stride0 = D0 * sizeof(float);
        uint64_t stride1 = (uint64_t)D0 * D1 * sizeof(float);
        uint64_t stride2 = (uint64_t)D0 * D1 * D2 * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(stride0));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x1, %1;" :: "l"(tmap_s), "l"(stride1));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x2, %1;" :: "l"(tmap_s), "l"(stride2));
        
        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));

        
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tensormap_cp_fenceproxy(tmap_g, tmap_s);
        fence_proxy_tensormap_acquire(tmap_g);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_init(bar, 1);
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    
    uint32_t elected;
    uint32_t pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);
    
    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        cp_async_bulk_tensor_4d_load(smem_data, tmap_g, coord0, coord1, coord2, coord3, mbar_addr);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
        mbarrier_inval(bar);
    }
    __syncthreads();
    
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        data[i] += 1.0f;
    }
    __syncthreads();
    
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int lx = i % BOX0;
        int ly = (i / BOX0) % BOX1;
        int lz = (i / (BOX0 * BOX1)) % BOX2;
        int lw = i / (BOX0 * BOX1 * BOX2);
        int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz, gw = coord3 + lw;
        if (gx < D0 && gy < D1 && gz < D2 && gw < D3) {
            out[((gw * D2 + gz) * D1 + gy) * D0 + gx] = data[i];
        }
    }
}

class TMA4DTest : public ::testing::Test {
protected:
    static constexpr int BOX0 = 8, BOX1 = 8, BOX2 = 8, BOX3 = 8;
    void Run(int D0, int D1, int D2, int D3, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2 * D3;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;
        
        ASSERT_EQ(cudaMalloc(&d_in, tot * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, tot * sizeof(float)), cudaSuccess);
        
        const int tiles2 = (D2 + BOX2 - 1) / BOX2;
        const int tiles3 = (D3 + BOX3 - 1) / BOX3;
        dim3 grid((D0 + BOX0 - 1) / BOX0,
                  (D1 + BOX1 - 1) / BOX1, tiles2 * tiles3);
        ASSERT_EQ(cudaMalloc(&d_scratch, grid.x * grid.y * grid.z * 256),
                  cudaSuccess);
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), tot * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_out, 0, tot * sizeof(float)), cudaSuccess);
        
        constexpr int TILE_ELEMS = BOX0 * BOX1 * BOX2 * BOX3;
        size_t smem = TENSORMAP_SIZE + TILE_ELEMS * sizeof(float) + 16 + 128; // +128 for alignment padding
        tma_tensor_4d_kernel<BOX0, BOX1, BOX2, BOX3><<<grid, 128, smem>>>(d_in, d_out, D0, D1, D2, D3, d_scratch);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, tot * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        
        EXPECT_EQ(cudaFree(d_in), cudaSuccess);
        EXPECT_EQ(cudaFree(d_out), cudaSuccess);
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }
};

TEST_F(TMA4DTest, Small_8x8x8x8) { Run(8, 8, 8, 8); }
TEST_F(TMA4DTest, Mixed_16x8x8x4) { Run(16, 8, 8, 4, 123); }
TEST_F(TMA4DTest, Large_16x16x8x8) { Run(16, 16, 8, 8, 456); }
TEST_F(TMA4DTest, CrossDim3Tiles_8x8x8x16) { Run(8, 8, 8, 16, 789); }

// ============================================================================  
// 5D TMA Kernel using cp.async.bulk.tensor.5d
// ============================================================================

template <int BOX0, int BOX1, int BOX2, int BOX3, int BOX4>
__global__ void tma_tensor_5d_kernel(const float *in, float *out,
                                      int D0, int D1, int D2, int D3, int D4, uint8_t *global_scratch) {
    constexpr int TILE_ELEMS = BOX0 * BOX1 * BOX2 * BOX3 * BOX4;
    constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);
    extern __shared__ uint8_t smem[];
    
    // Manual 128-byte alignment for tensormap
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
    uint8_t* tmap_smem = reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~127);
    
    float* data = reinterpret_cast<float*>(tmap_smem + TENSORMAP_SIZE);
    uint64_t* bar = reinterpret_cast<uint64_t*>(tmap_smem + TENSORMAP_SIZE + TILE_BYTES);
    
    int tid = threadIdx.x;



    int linear_bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    
    const int tiles2 = (D2 + BOX2 - 1) / BOX2;
    const int tiles3 = (D3 + BOX3 - 1) / BOX3;
    int high_tile = blockIdx.z;
    const int tile2 = high_tile % tiles2;
    high_tile /= tiles2;
    const int tile3 = high_tile % tiles3;
    const int tile4 = high_tile / tiles3;

    int coord0 = blockIdx.x * BOX0;
    int coord1 = blockIdx.y * BOX1;
    int coord2 = tile2 * BOX2;
    int coord3 = tile3 * BOX3;
    int coord4 = tile4 * BOX4;
    
    uint8_t* global_tmap = global_scratch + linear_bid * 256;
    
    tensormap_init_smem(tmap_smem, tid);
    __syncthreads();
    
    if (tid == 0) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        
        tensormap_set_global_address(tmap_s, reinterpret_cast<uint64_t>(in));
        asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x4;" :: "l"(tmap_s));
        
        uint32_t box0 = BOX0, box1 = BOX1, box2 = BOX2, box3 = BOX3, box4 = BOX4;
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(box0));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(box1));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(box2));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x3, %1;" :: "l"(tmap_s), "r"(box3));
        asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x4, %1;" :: "l"(tmap_s), "r"(box4));
        
        uint32_t gd0 = D0, gd1 = D1, gd2 = D2, gd3 = D3, gd4 = D4;
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(gd0));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, %1;" :: "l"(tmap_s), "r"(gd1));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x2, %1;" :: "l"(tmap_s), "r"(gd2));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x3, %1;" :: "l"(tmap_s), "r"(gd3));
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x4, %1;" :: "l"(tmap_s), "r"(gd4));
        
        uint64_t stride0 = D0 * sizeof(float);
        uint64_t stride1 = (uint64_t)D0 * D1 * sizeof(float);
        uint64_t stride2 = (uint64_t)D0 * D1 * D2 * sizeof(float);
        uint64_t stride3 = (uint64_t)D0 * D1 * D2 * D3 * sizeof(float);
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, %1;" :: "l"(tmap_s), "l"(stride0));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x1, %1;" :: "l"(tmap_s), "l"(stride1));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x2, %1;" :: "l"(tmap_s), "l"(stride2));
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x3, %1;" :: "l"(tmap_s), "l"(stride3));
        
        uint32_t es = 1;
        asm volatile("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, %1;" :: "l"(tmap_s), "r"(es));
        
        asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
        asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;" :: "l"(tmap_s));
    }
    
    if (tid < 32) {
        uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        tensormap_cp_fenceproxy(tmap_g, tmap_s);
        fence_proxy_tensormap_acquire(tmap_g);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_init(bar, 1);
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    
    uint32_t elected;
    uint32_t pred;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync %0|p, -1;\n"
                 "selp.u32 %1, 1, 0, p;\n"
                 "}" : "=r"(elected), "=r"(pred));
    bool is_elected = (pred != 0) && (tid < 32);
    
    if (is_elected) {
        uint32_t smem_data = static_cast<uint32_t>(__cvta_generic_to_shared(data));
        uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
        cp_async_bulk_tensor_5d_load(smem_data, tmap_g, coord0, coord1, coord2, coord3, coord4, mbar_addr);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_wait_parity(bar, 0);
        mbarrier_inval(bar);
    }
    __syncthreads();
    
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        data[i] += 1.0f;
    }
    __syncthreads();
    
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int idx = i;
        int lx = idx % BOX0; idx /= BOX0;
        int ly = idx % BOX1; idx /= BOX1;
        int lz = idx % BOX2; idx /= BOX2;
        int lw = idx % BOX3; idx /= BOX3;
        int lv = idx;
        int gx = coord0 + lx, gy = coord1 + ly, gz = coord2 + lz, gw = coord3 + lw, gv = coord4 + lv;
        if (gx < D0 && gy < D1 && gz < D2 && gw < D3 && gv < D4) {
            out[(((gv * D3 + gw) * D2 + gz) * D1 + gy) * D0 + gx] = data[i];
        }
    }
}

class TMA5DTest : public ::testing::Test {
protected:
    static constexpr int BOX0 = 4, BOX1 = 4, BOX2 = 4, BOX3 = 4, BOX4 = 4;
    void Run(int D0, int D1, int D2, int D3, int D4, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2 * D3 * D4;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in = nullptr, *d_out = nullptr;
        uint8_t *d_scratch = nullptr;
        
        ASSERT_EQ(cudaMalloc(&d_in, tot * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, tot * sizeof(float)), cudaSuccess);
        
        const int tiles2 = (D2 + BOX2 - 1) / BOX2;
        const int tiles3 = (D3 + BOX3 - 1) / BOX3;
        const int tiles4 = (D4 + BOX4 - 1) / BOX4;
        dim3 grid((D0 + BOX0 - 1) / BOX0,
                  (D1 + BOX1 - 1) / BOX1,
                  tiles2 * tiles3 * tiles4);
        ASSERT_EQ(cudaMalloc(&d_scratch, grid.x * grid.y * grid.z * 256),
                  cudaSuccess);
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), tot * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_out, 0, tot * sizeof(float)), cudaSuccess);
        
        constexpr int TILE_ELEMS = BOX0 * BOX1 * BOX2 * BOX3 * BOX4;
        size_t smem = TENSORMAP_SIZE + TILE_ELEMS * sizeof(float) + 16 + 128; // +128 for alignment padding
        tma_tensor_5d_kernel<BOX0, BOX1, BOX2, BOX3, BOX4><<<grid, 128, smem>>>(d_in, d_out, D0, D1, D2, D3, D4, d_scratch);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, tot * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        
        EXPECT_EQ(cudaFree(d_in), cudaSuccess);
        EXPECT_EQ(cudaFree(d_out), cudaSuccess);
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }
};

TEST_F(TMA5DTest, Small_4x4x4x4x4) { Run(4, 4, 4, 4, 4); }
TEST_F(TMA5DTest, Mixed_8x4x4x4x2) { Run(8, 4, 4, 4, 2, 123); }
TEST_F(TMA5DTest, Large_8x8x4x4x4) { Run(8, 8, 4, 4, 4, 456); }
TEST_F(TMA5DTest, CrossDim3AndDim4Tiles_4x4x4x8x8) {
    Run(4, 4, 4, 8, 8, 789);
}
