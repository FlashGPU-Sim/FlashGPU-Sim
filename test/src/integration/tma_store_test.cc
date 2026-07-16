#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// =============================================================================
// Inline TMA Store PTX Helpers and Kernels
//
// These are self-contained (NOT in cp_kernels.cuh) to avoid template
// instantiation collisions with cuda_tma_test.cc which uses the same
// kernel templates. Everything is defined inline with unique names.
// =============================================================================

// --- PTX helpers ---

__device__ inline void tma_store_fence_proxy_async() {
  asm volatile("fence.proxy.async;");
}

template <int bytes>
__device__ inline void tma_store_cp_async_bulk(void *global_dst,
                                                const void *smem_src) {
  unsigned long long dst_g, src_s;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(dst_g) : "l"(global_dst));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(src_s) : "l"(smem_src));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(dst_g), "l"(src_s), "n"(bytes));
}

__device__ inline void tma_store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
__device__ inline void tma_store_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}

// --- Kernels ---

// Basic TMA store kernel: each warp processes NUM_GROUPS chunks
// Each chunk is loaded from global → modified in shared → TMA-stored to global
template <int NUM_GROUPS, int CHUNK_BYTES, int PIPELINE_DEPTH = 0>
__global__ void tmaStoreKernel(const uint32_t *__restrict__ src,
                                uint32_t *__restrict__ dst, size_t n) {
  extern __shared__ __align__(16) uint8_t smem[];
  int wid = threadIdx.x / 32;
  int lid = threadIdx.x % 32;
  int gid = blockIdx.x * (blockDim.x / 32) + wid;
  int cpe = CHUNK_BYTES / sizeof(uint32_t);
  int wp_start = gid * NUM_GROUPS * cpe;
  uint8_t *ws = smem + wid * CHUNK_BYTES;

  if (lid == 0) {
    for (int g = 0; g < NUM_GROUPS; g++) {
      int cs = wp_start + g * cpe;
      if (cs >= n) break;
      auto *u = reinterpret_cast<uint32_t *>(ws);
      for (int i = 0; i < cpe && (cs + i) < n; i++) u[i] = src[cs + i];
      for (int i = 0; i < cpe && (cs + i) < n; i++) u[i] += 1;
      tma_store_fence_proxy_async();
      tma_store_cp_async_bulk<CHUNK_BYTES>(dst + cs, ws);
      tma_store_commit_group();
      if constexpr (PIPELINE_DEPTH > 0) {
        if (g >= PIPELINE_DEPTH)
          tma_store_wait_group<PIPELINE_DEPTH>();
      }
    }
    tma_store_wait_group<0>();
  }
  __syncthreads();
}

// Multi-group TMA store kernel: each chunk gets its own bulk group
// Single shared memory buffer, so must wait for each store before reusing
template <int CHUNK_BYTES>
__global__ void tmaStoreMultiGroupKernel(const uint32_t *__restrict__ src,
                                          uint32_t *__restrict__ dst,
                                          int num_chunks) {
  extern __shared__ __align__(16) uint8_t smem[];
  int lid = threadIdx.x % 32;
  int cpe = CHUNK_BYTES / sizeof(uint32_t);

  if (threadIdx.x < 32 && lid == 0) {
    for (int c = 0; c < num_chunks; c++) {
      int cs = c * cpe;
      auto *u = reinterpret_cast<uint32_t *>(smem);
      for (int i = 0; i < cpe; i++) u[i] = src[cs + i];
      for (int i = 0; i < cpe; i++) u[i] += (c + 1);
      tma_store_fence_proxy_async();
      tma_store_cp_async_bulk<CHUNK_BYTES>(dst + cs, smem);
      tma_store_commit_group();
      tma_store_wait_group<0>();  // Must wait before reusing smem
    }
  }
}

// Pipelined TMA store kernel: MAX_IN_FLIGHT groups can be in-flight
template <int CHUNK_BYTES, int MAX_IN_FLIGHT>
__global__ void tmaStorePipelinedKernel(const uint32_t *__restrict__ src,
                                         uint32_t *__restrict__ dst,
                                         int num_chunks) {
  extern __shared__ __align__(16) uint8_t smem[];
  int lid = threadIdx.x % 32;
  int cpe = CHUNK_BYTES / sizeof(uint32_t);
  constexpr int NS = MAX_IN_FLIGHT + 1;

  if (threadIdx.x < 32 && lid == 0) {
    for (int c = 0; c < num_chunks; c++) {
      int slot = c % NS;
      uint8_t *ss = smem + slot * CHUNK_BYTES;
      int cs = c * cpe;

      if (c >= NS) tma_store_wait_group<NS - 1>();

      auto *u = reinterpret_cast<uint32_t *>(ss);
      for (int i = 0; i < cpe; i++) u[i] = src[cs + i] * 2;
      tma_store_fence_proxy_async();
      tma_store_cp_async_bulk<CHUNK_BYTES>(dst + cs, ss);
      tma_store_commit_group();
    }
    tma_store_wait_group<0>();
  }
}

// =============================================================================
// Test Fixture
// =============================================================================

class TMAStoreTest : public ::testing::Test {
protected:
  static constexpr int DEFAULT_ELEMENTS = 4096;

  void SetUp() override {
    num_elements = DEFAULT_ELEMENTS;
    data_size_bytes = num_elements * sizeof(uint32_t);
    h_input.resize(num_elements);
    h_output.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i)
      h_input[i] = static_cast<uint32_t>(i + 1);
  }

  size_t num_elements;
  size_t data_size_bytes;
  std::vector<uint32_t> h_input;
  std::vector<uint32_t> h_output;
};

// =============================================================================
// Test 1: Basic TMA store — 256B chunks, 4 groups, single warp
// =============================================================================
TEST_F(TMAStoreTest, Basic256B) {
  constexpr int CB = 256, NG = 4;
  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, h_input.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_output.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (h_output[i] != h_input[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 2: 128B chunks
// =============================================================================
TEST_F(TMAStoreTest, Basic128B) {
  constexpr int CB = 128, NG = 8;
  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, h_input.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_output.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (h_output[i] != h_input[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 3: 64B chunks (minimum supported size)
// =============================================================================
TEST_F(TMAStoreTest, Basic64B) {
  constexpr int CB = 64, NG = 16;
  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, h_input.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_output.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (h_output[i] != h_input[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 4: 512B chunks
// =============================================================================
TEST_F(TMAStoreTest, Basic512B) {
  constexpr int CB = 512, NG = 2;
  const size_t te = 8192;
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (te + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, te);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < te; i++)
    if (out[i] != in[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 5: 1024B chunks
// =============================================================================
TEST_F(TMAStoreTest, Basic1024B) {
  constexpr int CB = 1024, NG = 1;
  const size_t te = 2048;
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (te + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, te);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < te; i++)
    if (out[i] != in[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 6: 16B chunks (4x uint32)
// =============================================================================
TEST_F(TMAStoreTest, Basic16B) {
  constexpr int CB = 16, NG = 64;
  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, h_input.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h_output.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (h_output[i] != h_input[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 7: Multiple blocks, multiple warps per block
// =============================================================================
TEST_F(TMAStoreTest, MultiBlockMultiWarp) {
  constexpr int CB = 256, NG = 2, WB = 4, TB = WB * 32, BL = 2;
  const size_t te = BL * WB * NG * (CB / sizeof(uint32_t));
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  tmaStoreKernel<NG, CB, 0><<<BL, TB, WB * CB>>>(di, dd, te);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < te; i++)
    if (out[i] != in[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 8: Pipelined TMA store — MAX_IN_FLIGHT=2
// =============================================================================
TEST_F(TMAStoreTest, Pipelined2InFlight) {
  constexpr int CB = 128, MIF = 2;
  const int nc = 16;
  const int cpe = CB / sizeof(uint32_t);
  const size_t te = nc * cpe;
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  tmaStorePipelinedKernel<CB, MIF>
      <<<1, 32, (MIF + 1) * CB>>>(di, dd, nc);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < te; i++)
    if (out[i] != in[i] * 2) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 9: Pipelined TMA store — MAX_IN_FLIGHT=7 (deep pipeline)
// =============================================================================
TEST_F(TMAStoreTest, Pipelined7InFlight) {
  constexpr int CB = 64, MIF = 7;
  const int nc = 32;
  const int cpe = CB / sizeof(uint32_t);
  const size_t te = nc * cpe;
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i * 3);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  tmaStorePipelinedKernel<CB, MIF>
      <<<1, 32, (MIF + 1) * CB>>>(di, dd, nc);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < te; i++)
    if (out[i] != in[i] * 2) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 10: Multi-group (sequential groups, one buffer)
// =============================================================================
TEST_F(TMAStoreTest, MultiGroupSequential) {
  constexpr int CB = 128;
  const int nc = 8;
  const int cpe = CB / sizeof(uint32_t);
  const size_t te = nc * cpe;
  const size_t tb = te * sizeof(uint32_t);
  std::vector<uint32_t> in(te), out(te);
  for (size_t i = 0; i < te; i++) in[i] = static_cast<uint32_t>(i * 10);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, tb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, tb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), tb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, tb), cudaSuccess);

  tmaStoreMultiGroupKernel<CB><<<1, 32, CB>>>(di, dd, nc);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, tb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (int c = 0; c < nc; c++)
    for (int i = 0; i < cpe; i++) {
      size_t idx = c * cpe + i;
      if (out[idx] != in[idx] + (c + 1)) errs++;
    }
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 11: Large-scale TMA store — 256KB, 16 groups, 4 warps per block
// =============================================================================
TEST_F(TMAStoreTest, LargeScale) {
  constexpr int CB = 256, NG = 16, TB = 128;
  const size_t le = 65536;
  const size_t lb = le * sizeof(uint32_t);
  std::vector<uint32_t> in(le), out(le);
  for (size_t i = 0; i < le; i++) in[i] = static_cast<uint32_t>(i % 1000);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, lb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, lb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), lb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, lb), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int wb = TB / 32;
  int epb = wb * epw;
  int blk = (le + epb - 1) / epb;
  tmaStoreKernel<NG, CB, 0><<<blk, TB, wb * CB>>>(di, dd, le);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, lb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < le; i++)
    if (out[i] != in[i] + 1) errs++;
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 12: Zero data pattern
// =============================================================================
TEST_F(TMAStoreTest, ZeroPattern) {
  constexpr int CB = 256, NG = 4;
  std::vector<uint32_t> in(num_elements, 0), out(num_elements, 0xFF);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0xFF, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (out[i] != 1) errs++;  // 0+1 = 1
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 13: Max uint32 pattern (wrap-around test)
// =============================================================================
TEST_F(TMAStoreTest, MaxU32Pattern) {
  constexpr int CB = 256, NG = 4;
  std::vector<uint32_t> in(num_elements, 0xFFFFFFFF), out(num_elements, 0);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < num_elements; i++)
    if (out[i] != 0) errs++;  // 0xFFFFFFFF + 1 wraps to 0
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 14: Multiple iterations (stress test) — repeat the same pattern 3 times
// =============================================================================
TEST_F(TMAStoreTest, MultipleIterations) {
  constexpr int CB = 256, NG = 4;
  const int iterations = 3;

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, data_size_bytes), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (num_elements + epw - 1) / epw;

  for (int iter = 0; iter < iterations; iter++) {
    ASSERT_EQ(cudaMemcpy(di, h_input.data(), data_size_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dd, 0, data_size_bytes), cudaSuccess);

    tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, num_elements);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess)
        << "Iteration " << iter << " launch failed";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
        << "Iteration " << iter << " exec failed";
    ASSERT_EQ(cudaMemcpy(h_output.data(), dd, data_size_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);

    int errs = 0;
    for (size_t i = 0; i < num_elements; i++)
      if (h_output[i] != h_input[i] + 1) errs++;
    EXPECT_EQ(errs, 0) << "Iteration " << iter;
  }

  cudaFree(di); cudaFree(dd);
}

// =============================================================================
// Test 15: Non-power-of-two element count with partial last chunk
// =============================================================================
TEST_F(TMAStoreTest, PartialLastChunk) {
  constexpr int CB = 256, NG = 4;
  // Choose a chunk-count that is not a multiple of NUM_GROUPS.
  // Each warp processes NG*(CB/4) = 4*64 = 256 elements.
  // 2 warps = 512. Use 448 = 7 full chunks: warp 0 gets 4 chunks, warp 1 gets 3.
  // The kernel's bounds check (cs >= n) skips the 4th group of warp 1.
  const size_t partial = 448;
  const size_t pb = partial * sizeof(uint32_t);
  std::vector<uint32_t> in(partial), out(partial);
  for (size_t i = 0; i < partial; i++) in[i] = static_cast<uint32_t>(i * 7);

  uint32_t *di = nullptr, *dd = nullptr;
  ASSERT_EQ(cudaMalloc(&di, pb), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dd, pb), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di, in.data(), pb, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dd, 0, pb), cudaSuccess);

  int epw = NG * (CB / sizeof(uint32_t));
  int blk = (partial + epw - 1) / epw;
  tmaStoreKernel<NG, CB, 0><<<blk, 32, CB>>>(di, dd, partial);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dd, pb, cudaMemcpyDeviceToHost), cudaSuccess);

  int errs = 0;
  for (size_t i = 0; i < partial; i++)
    if (out[i] != in[i] + 1) {
      if (errs < 5) printf("idx %zu: exp %u got %u\n", i, in[i]+1, out[i]);
      errs++;
    }
  EXPECT_EQ(errs, 0);

  cudaFree(di); cudaFree(dd);
}
