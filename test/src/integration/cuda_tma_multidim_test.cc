// Multi-dimensional TMA tests using cp.async.bulk with inline PTX
// Coverage: 1D, 2D, 3D, 4D, 5D tensors matching example_tensor_add.py
// Uses cp.async.bulk.shared::cta.global.mbarrier for TMA operations

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cmath>
#include <vector>

// ============================================================================
// PTX Helper Functions - mbarrier operations
// ============================================================================

__device__ inline void mbarrier_init(uint64_t *bar, uint32_t count) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(count));
}

__device__ inline void mbarrier_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(p), "r"(bytes));
}

__device__ inline void mbarrier_wait_parity(uint64_t *bar, uint32_t parity) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra.uni DONE;\n"
                 "bra.uni LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" :: "r"(p), "r"(parity));
}

// ============================================================================
// TMA bulk copy - cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
// ============================================================================

template <int BYTES>
__device__ inline void cp_async_bulk(void *smem_dst, const void *global_src,
                                     uint64_t *bar_addr) {
    uint64_t dst_s, src_g, bar_s;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %3, [%2];" :: "l"(dst_s), "l"(src_g), "l"(bar_s), "n"(BYTES));
}

// ============================================================================
// 1D TMA Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void tma_1d_add_kernel(const float *in, float *out, int N) {
    constexpr int BYTES = BLOCK_SIZE * sizeof(float);
    extern __shared__ uint8_t smem[];
    float *data = reinterpret_cast<float*>(smem);
    uint64_t *bar = reinterpret_cast<uint64_t*>(smem + BYTES);

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int off = bid * BLOCK_SIZE;

    if (tid == 0) {
        mbarrier_init(bar, 1);
        // TMA always transfers BYTES amount, so expect_tx must match
        mbarrier_arrive_expect_tx(bar, BYTES);
        cp_async_bulk<BYTES>(data, in + off, bar);
        mbarrier_wait_parity(bar, 0);
    }
    __syncthreads();

    if (off + tid < N && tid < BLOCK_SIZE) data[tid] += 1.0f;
    __syncthreads();
    if (off + tid < N && tid < BLOCK_SIZE) out[off + tid] = data[tid];
}

// ============================================================================
// 2D TMA Kernel - copies row by row using cp.async.bulk
// ============================================================================

template <int BM, int BN>
__global__ void tma_2d_add_kernel(const float *in, float *out, int M, int N) {
    constexpr int TILE_BYTES = BM * BN * sizeof(float);
    constexpr int ROW_BYTES = BN * sizeof(float);
    extern __shared__ uint8_t smem[];
    float *data = reinterpret_cast<float*>(smem);
    uint64_t *bar = reinterpret_cast<uint64_t*>(smem + TILE_BYTES);

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int m_off = blockIdx.y * BM;
    int n_off = blockIdx.x * BN;

    if (tid == 0) {
        // Each row copy is BN*4 bytes, and we copy BM rows
        // Total expected tx = BM * ROW_BYTES
        mbarrier_init(bar, 1);  // 1 arrival
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);  // expect full tile bytes
        
        // Copy each row using TMA bulk copy (each adds ROW_BYTES to tx_complete)
        for (int r = 0; r < BM; r++) {
            const float *src = in + (m_off + r) * N + n_off;
            float *dst = data + r * BN;
            cp_async_bulk<ROW_BYTES>(dst, src, bar);
        }
        mbarrier_wait_parity(bar, 0);
    }
    __syncthreads();

    int lm = threadIdx.y, ln = threadIdx.x;
    if (m_off + lm < M && n_off + ln < N) data[lm * BN + ln] += 1.0f;
    __syncthreads();
    if (m_off + lm < M && n_off + ln < N)
        out[(m_off + lm) * N + (n_off + ln)] = data[lm * BN + ln];
}

// ============================================================================
// N-D TMA Kernel - linearized bulk copy for 3D, 4D, 5D tensors
// ============================================================================

template <int BLOCK_SIZE>
__global__ void tma_nd_add_kernel(const float *in, float *out, int total) {
    constexpr int BYTES = BLOCK_SIZE * sizeof(float);
    extern __shared__ uint8_t smem[];
    float *data = reinterpret_cast<float*>(smem);
    uint64_t *bar = reinterpret_cast<uint64_t*>(smem + BYTES);

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int off = bid * BLOCK_SIZE;

    if (tid == 0) {
        mbarrier_init(bar, 1);
        // TMA always transfers BYTES amount, so expect_tx must match
        mbarrier_arrive_expect_tx(bar, BYTES);
        cp_async_bulk<BYTES>(data, in + off, bar);
        mbarrier_wait_parity(bar, 0);
    }
    __syncthreads();

    if (off + tid < total && tid < BLOCK_SIZE) data[tid] += 1.0f;
    __syncthreads();
    if (off + tid < total && tid < BLOCK_SIZE) out[off + tid] = data[tid];
}

// ============================================================================
// Test Fixtures
// ============================================================================

class TMA1DTest : public ::testing::Test {
protected:
    static constexpr int BLOCK = 128;
    void Run(int N, int seed = 42) {
        std::vector<float> h_in(N), h_out(N);
        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, N * sizeof(float));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (int i = 0; i < N; i++) h_in[i] = dist(gen);
        cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        
        int grid = (N + BLOCK - 1) / BLOCK;
        size_t smem = BLOCK * sizeof(float) + 16;
        tma_1d_add_kernel<BLOCK><<<grid, BLOCK, smem>>>(d_in, d_out, N);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f) << "at " << i;
        cudaFree(d_in); cudaFree(d_out);
    }
};

TEST_F(TMA1DTest, RegularSize_8192) { Run(8192); }
TEST_F(TMA1DTest, RemainderCase_8195) { Run(8195, 123); }
TEST_F(TMA1DTest, SmallSize_256) { Run(256, 456); }

class TMA2DTest : public ::testing::Test {
protected:
    static constexpr int BM = 32, BN = 32;
    void Run(int M, int N, int seed = 42) {
        size_t tot = (size_t)M * N;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in, *d_out;
        cudaMalloc(&d_in, tot * sizeof(float));
        cudaMalloc(&d_out, tot * sizeof(float));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (size_t i = 0; i < tot; i++) h_in[i] = dist(gen);
        cudaMemcpy(d_in, h_in.data(), tot * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(BN, BM);
        size_t smem = BM * BN * sizeof(float) + 16;
        tma_2d_add_kernel<BM, BN><<<grid, block, smem>>>(d_in, d_out, M, N);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        cudaMemcpy(h_out.data(), d_out, tot * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        cudaFree(d_in); cudaFree(d_out);
    }
};

TEST_F(TMA2DTest, Square_128x128) { Run(128, 128); }
TEST_F(TMA2DTest, Rectangle_256x64) { Run(256, 64, 123); }
TEST_F(TMA2DTest, Large_512x512) { Run(512, 512, 456); }

class TMA3DTest : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    void Run(int D0, int D1, int D2, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in, *d_out;
        cudaMalloc(&d_in, tot * sizeof(float));
        cudaMalloc(&d_out, tot * sizeof(float));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        cudaMemcpy(d_in, h_in.data(), tot * sizeof(float), cudaMemcpyHostToDevice);
        
        int grid = (tot + BLOCK - 1) / BLOCK;
        size_t smem = BLOCK * sizeof(float) + 16;
        tma_nd_add_kernel<BLOCK><<<grid, BLOCK, smem>>>(d_in, d_out, tot);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        cudaMemcpy(h_out.data(), d_out, tot * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        cudaFree(d_in); cudaFree(d_out);
    }
};

TEST_F(TMA3DTest, Regular_64x64x64) { Run(64, 64, 64); }
TEST_F(TMA3DTest, MixedSizes_100x50x80) { Run(100, 50, 80, 123); }
TEST_F(TMA3DTest, Small_16x16x16) { Run(16, 16, 16, 456); }

class TMA4DTest : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    void Run(int D0, int D1, int D2, int D3, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2 * D3;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in, *d_out;
        cudaMalloc(&d_in, tot * sizeof(float));
        cudaMalloc(&d_out, tot * sizeof(float));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        cudaMemcpy(d_in, h_in.data(), tot * sizeof(float), cudaMemcpyHostToDevice);
        
        int grid = (tot + BLOCK - 1) / BLOCK;
        size_t smem = BLOCK * sizeof(float) + 16;
        tma_nd_add_kernel<BLOCK><<<grid, BLOCK, smem>>>(d_in, d_out, tot);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        cudaMemcpy(h_out.data(), d_out, tot * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        cudaFree(d_in); cudaFree(d_out);
    }
};

TEST_F(TMA4DTest, Regular_32x32x32x32) { Run(32, 32, 32, 32); }
TEST_F(TMA4DTest, Degenerate_1x64x64x64) { Run(1, 64, 64, 64, 123); }
TEST_F(TMA4DTest, Small_8x8x8x8) { Run(8, 8, 8, 8, 456); }

class TMA5DTest : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    void Run(int D0, int D1, int D2, int D3, int D4, int seed = 42) {
        size_t tot = (size_t)D0 * D1 * D2 * D3 * D4;
        std::vector<float> h_in(tot), h_out(tot);
        float *d_in, *d_out;
        cudaMalloc(&d_in, tot * sizeof(float));
        cudaMalloc(&d_out, tot * sizeof(float));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-10, 10);
        for (auto &v : h_in) v = dist(gen);
        cudaMemcpy(d_in, h_in.data(), tot * sizeof(float), cudaMemcpyHostToDevice);
        
        int grid = (tot + BLOCK - 1) / BLOCK;
        size_t smem = BLOCK * sizeof(float) + 16;
        tma_nd_add_kernel<BLOCK><<<grid, BLOCK, smem>>>(d_in, d_out, tot);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        cudaMemcpy(h_out.data(), d_out, tot * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < tot; i++)
            EXPECT_NEAR(h_out[i], h_in[i] + 1.0f, 1e-4f);
        cudaFree(d_in); cudaFree(d_out);
    }
};

TEST_F(TMA5DTest, Regular_16x16x16x16x16) { Run(16, 16, 16, 16, 16); }
TEST_F(TMA5DTest, NonUniform_8x12x10x14x16) { Run(8, 12, 10, 14, 16, 123); }
TEST_F(TMA5DTest, Small_4x4x4x4x4) { Run(4, 4, 4, 4, 4, 456); }
