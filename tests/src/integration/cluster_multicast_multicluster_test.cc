#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "common/gpgpusim_config_topology.h"

// ============================================================================
// Multi-cluster TMA multicast tests
//
// Verifies that cluster multicast works correctly when multiple clusters are
// active simultaneously. Each physical simt_core_cluster should independently
// replicate TMA load data across its SMs without data leaking across cluster
// boundaries (issue-order cluster_group matching).
//
// Requires a multi-cluster config (e.g., SM120_RTX5090_REDUCED_CLUSTER2x2).
//
// Note: plain grid launches only — validates multi-cluster topology + the
// simulator's TMA multicast model, not CUDA cooperative cluster launch APIs.
// ============================================================================

// --- Inline mbarrier helpers ---
__device__ inline void mbarrier_init_impl(unsigned long long *bar_addr,
                                           unsigned expected_arrivals) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_addr));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(expected_arrivals));
}

__device__ inline void mbarrier_arrive_expect_tx_impl(
    unsigned long long *bar_addr, unsigned tx_bytes) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"l"(bar_s), "r"(tx_bytes));
}

__device__ inline void mbarrier_try_wait_impl(unsigned long long *bar_addr,
                                               int parity) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT_IMPL:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra.uni LAB_WAIT_IMPL;\n"
      "}\n" ::"l"(bar_s), "r"(parity));
}

// --- TMA load helpers ---
template <int bytes>
__device__ inline void cp_async_bulk_cta(void *smem_dst,
                                          const void *global_src,
                                          unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %3, [%2];"
      ::"l"(dst_s), "l"(src_g), "l"(bar_s), "n"(bytes));
}

template <int bytes>
__device__ inline void cp_async_bulk_cluster(void *smem_dst,
                                              const void *global_src,
                                              unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %3, [%2];"
      ::"l"(dst_s), "l"(src_g), "l"(bar_s), "n"(bytes));
}

// ============================================================================
// Test 1: Multi-cluster basic execution
//
// Launches many blocks on a config with 2+ clusters. Each block writes its
// blockIdx to a unique output slot. Verifies all blocks run correctly.
// ============================================================================

__global__ void multiClusterBasicKernel(uint8_t *output, int num_blocks) {
  int bid = blockIdx.x;
  if (bid < num_blocks) {
    int tid = threadIdx.x;
    if (tid == 0) {
      output[bid] = static_cast<uint8_t>(bid);
    }
  }
}

class MultiClusterBasicTest : public ::testing::Test {
protected:
  static constexpr int NUM_BLOCKS = 8;  // Enough to span multiple clusters
  static constexpr int THREADS_PER_BLOCK = 32;
};

TEST_F(MultiClusterBasicTest, AllBlocksExecuteCorrectly) {
  std::vector<uint8_t> h_output(NUM_BLOCKS, 0xFF);
  uint8_t *d_output = nullptr;

  ASSERT_EQ(cudaMalloc(&d_output, NUM_BLOCKS), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0xFF, NUM_BLOCKS), cudaSuccess);

  multiClusterBasicKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_output, NUM_BLOCKS);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, NUM_BLOCKS,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  for (int b = 0; b < NUM_BLOCKS; b++) {
    EXPECT_EQ(h_output[b], static_cast<uint8_t>(b))
        << "Block " << b << " did not execute";
  }

  cudaFree(d_output);
}

// ============================================================================
// Test 2: Multi-cluster CTA-scope TMA — no cross-cluster data leak
//
// Launches 2 blocks, each loads its own distinct input via CTA-scope TMA.
// With 2 clusters, blocks should land on different clusters and be isolated.
// CTA-scope means each block only reads into its own SM.
// ============================================================================

template <int CHUNK_BYTES>
__global__ void multiClusterCTALoadKernel(const uint8_t *global_src,
                                           uint8_t *global_dst) {
  __shared__ uint8_t smem[CHUNK_BYTES + 64];
  __shared__ unsigned long long bar;

  uint8_t *data_buf = smem;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid == 0) {
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
    cp_async_bulk_cta<CHUNK_BYTES>(data_buf, global_src + bid * CHUNK_BYTES, &bar);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_try_wait_impl(&bar, 0);
  }
  __syncthreads();

  int offset = bid * CHUNK_BYTES;
  for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
    global_dst[offset + i] = data_buf[i];
  }
}

class MultiClusterCTATest : public ::testing::Test {
protected:
  static constexpr int CHUNK_BYTES = 256;
  static constexpr int NUM_BLOCKS = 2;  // 1 per cluster
  static constexpr int THREADS_PER_BLOCK = 32;

  void SetUp() override {
    h_input.resize(CHUNK_BYTES * NUM_BLOCKS);
    for (int b = 0; b < NUM_BLOCKS; b++) {
      for (int i = 0; i < CHUNK_BYTES; i++) {
        h_input[b * CHUNK_BYTES + i] = static_cast<uint8_t>(b * 41 + i);
      }
    }
  }

  std::vector<uint8_t> h_input;
};

TEST_F(MultiClusterCTATest, CtaScopeLoadWithMultipleClusters) {
  const int total_bytes = NUM_BLOCKS * CHUNK_BYTES;
  uint8_t *d_src = nullptr;
  uint8_t *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), total_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_dst, 0xFF, total_bytes), cudaSuccess);

  multiClusterCTALoadKernel<CHUNK_BYTES>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_output(total_bytes);
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_dst, total_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errors = 0;
  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int i = 0; i < CHUNK_BYTES; i++) {
      uint8_t expected = h_input[b * CHUNK_BYTES + i];
      uint8_t got = h_output[b * CHUNK_BYTES + i];
      if (got != expected) {
        if (errors < 10) {
          printf("Block %d offset %d: expected %u got %u\n", b, i, expected, got);
        }
        errors++;
      }
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_src);
  cudaFree(d_dst);
}

// ============================================================================
// Test 3: Multi-cluster cluster-scope TMA — one block per cluster
//
// Uses cluster multicast with exactly 1 block per cluster. This verifies that
// cluster multicast works when multiple clusters are active, without
// within-cluster interference (which is expected since multicast replicates
// across all SMs in a cluster and multiple blocks in the same cluster would
// overwrite each other).
// ============================================================================

template <int CHUNK_BYTES>
__global__ void multiClusterClusterLoadKernel(const uint8_t *global_src,
                                               uint8_t *global_dst) {
  __shared__ uint8_t smem[CHUNK_BYTES + 64];
  __shared__ unsigned long long bar;
  __shared__ volatile int done;

  uint8_t *data_buf = smem;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid == 0) {
    done = 0;
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
    cp_async_bulk_cluster<CHUNK_BYTES>(data_buf, global_src + bid * CHUNK_BYTES, &bar);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_try_wait_impl(&bar, 0);
    done = 1;
  }
  __syncthreads();

  int offset = bid * CHUNK_BYTES;
  for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
    global_dst[offset + i] = data_buf[i];
  }
}

class MultiClusterClusterTest : public ::testing::Test {
protected:
  static constexpr int CHUNK_BYTES = 256;
  static constexpr int NUM_BLOCKS = 2;  // 1 per cluster (2 clusters x 2 SMs)
  static constexpr int THREADS_PER_BLOCK = 32;

  void SetUp() override {
    h_input.resize(CHUNK_BYTES * NUM_BLOCKS);
    for (int b = 0; b < NUM_BLOCKS; b++) {
      for (int i = 0; i < CHUNK_BYTES; i++) {
        h_input[b * CHUNK_BYTES + i] = static_cast<uint8_t>(b * 67 + i);
      }
    }
  }

  std::vector<uint8_t> h_input;
};

TEST_F(MultiClusterClusterTest, ClusterMulticastWithMultipleClusters) {
  // Distinct per-block tiles + .shared::cluster multicast only isolate when
  // blocks land in different physical clusters. On a single-cluster config
  // both blocks share one cluster_group domain and peer-multicast each other.
  SKIP_IF_N_CLUSTERS_LT(2);
  // Cluster multicast peer path also requires multi-SM clusters.
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  const int total_bytes = NUM_BLOCKS * CHUNK_BYTES;
  uint8_t *d_src = nullptr;
  uint8_t *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, total_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), total_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_dst, 0xFF, total_bytes), cudaSuccess);

  multiClusterClusterLoadKernel<CHUNK_BYTES>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_output(total_bytes);
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_dst, total_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  int errors = 0;
  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int i = 0; i < CHUNK_BYTES; i++) {
      uint8_t expected = h_input[b * CHUNK_BYTES + i];
      uint8_t got = h_output[b * CHUNK_BYTES + i];
      if (got != expected) {
        if (errors < 10) {
          printf("Block %d offset %d: expected %u got %u\n", b, i, expected, got);
        }
        errors++;
      }
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_src);
  cudaFree(d_dst);
}

// ============================================================================
// Test 4: Multi-cluster with many blocks (interleaved scheduling)
//
// Launches more blocks than SMs to exercise interleaved scheduling across
// clusters with a simple (non-TMA) kernel.
// ============================================================================

__global__ void multiClusterInterleaveKernel(uint8_t *output, int num_blocks) {
  int bid = blockIdx.x;
  if (bid < num_blocks) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      output[bid * 256 + i] = static_cast<uint8_t>((bid + i) & 0xFF);
    }
  }
}

class MultiClusterInterleaveTest : public ::testing::Test {
protected:
  static constexpr int NUM_BLOCKS = 16;
  static constexpr int THREADS_PER_BLOCK = 32;
};

TEST_F(MultiClusterInterleaveTest, CorrectOutputWithManyBlocks) {
  const int total_elements = NUM_BLOCKS * 256;
  std::vector<uint8_t> h_output(total_elements, 0xFF);
  uint8_t *d_output = nullptr;

  ASSERT_EQ(cudaMalloc(&d_output, total_elements), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0xFF, total_elements), cudaSuccess);

  multiClusterInterleaveKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
      d_output, NUM_BLOCKS);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, total_elements,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  for (int b = 0; b < NUM_BLOCKS; b++) {
    for (int i = 0; i < 256; i++) {
      uint8_t expected = static_cast<uint8_t>((b + i) & 0xFF);
      EXPECT_EQ(h_output[b * 256 + i], expected)
          << "Block " << b << " elem " << i;
    }
  }

  cudaFree(d_output);
}
