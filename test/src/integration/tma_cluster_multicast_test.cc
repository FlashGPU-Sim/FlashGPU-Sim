#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// Inline mbarrier helpers matching cp_kernels.cuh patterns.
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

// TMA bulk load: shared::cta <- global (regular CTA scope).
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

// TMA cluster multicast: shared::cluster <- global.
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

// TMA load kernel: each block loads the same input chunk via TMA, then
// every thread copies from shared memory to a unique per-thread slot in
// global memory (no race conditions).
template <int CHUNK_BYTES, bool USE_CLUSTER>
__global__ void tmaLoadKernel(const uint8_t *global_src, uint8_t *global_dst,
                               int total_bytes) {
  __shared__ uint8_t smem[CHUNK_BYTES + 64];
  __shared__ unsigned long long bar;
  __shared__ volatile int done;

  uint8_t *data_buf = smem;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // Thread 0 initializes the mbarrier.
  if (tid == 0) {
    done = 0;
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();

  // Thread 0 issues the TMA load.
  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, CHUNK_BYTES);
    if (USE_CLUSTER) {
      cp_async_bulk_cluster<CHUNK_BYTES>(data_buf, global_src, &bar);
    } else {
      cp_async_bulk_cta<CHUNK_BYTES>(data_buf, global_src, &bar);
    }
  }
  __syncthreads();

  // Thread 0 polls the mbarrier until TMA completes.
  if (tid == 0) {
    mbarrier_try_wait_impl(&bar, 0);
    done = 1;
  }
  __syncthreads();

  // All threads copy their portion from shared memory to global memory.
  // Each thread writes to a UNIQUE index to avoid race conditions.
  int offset = bid * CHUNK_BYTES;
  if (offset + CHUNK_BYTES <= total_bytes) {
    for (int i = tid; i < CHUNK_BYTES; i += blockDim.x) {
      global_dst[offset + i] = data_buf[i];
    }
  }
}

class TMAClusterMulticastTest : public ::testing::Test {
protected:
  static constexpr int CHUNK_BYTES = 256;
  static constexpr int NUM_BLOCKS = 2;
  static constexpr int THREADS_PER_BLOCK = 32;

  void SetUp() override {
    h_input.resize(CHUNK_BYTES);
    for (int i = 0; i < CHUNK_BYTES; i++) {
      h_input[i] = static_cast<uint8_t>(i);
    }
  }

  void runTest(bool use_cluster, const char *name) {
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, CHUNK_BYTES), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src, h_input.data(), CHUNK_BYTES,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dst, 0xff, CHUNK_BYTES * NUM_BLOCKS), cudaSuccess);

    
    if (use_cluster) {
      tmaLoadKernel<CHUNK_BYTES, true>
          <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst,
                                                          CHUNK_BYTES * NUM_BLOCKS);
    } else {
      tmaLoadKernel<CHUNK_BYTES, false>
          <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_src, d_dst,
                                                          CHUNK_BYTES * NUM_BLOCKS);
    }

    ASSERT_EQ(cudaGetLastError(), cudaSuccess)
        << name << ": Kernel launch failed";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
        << name << ": Kernel execution failed";

    std::vector<uint8_t> h_output(CHUNK_BYTES * NUM_BLOCKS);
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_dst, CHUNK_BYTES * NUM_BLOCKS,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    int errors = 0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
      for (int i = 0; i < CHUNK_BYTES; i++) {
        int idx = b * CHUNK_BYTES + i;
        uint8_t expected = h_input[i];
        uint8_t got = h_output[idx];
        if (got != expected) {
          if (errors < 10) {
            printf("%s Block %d, offset %d: expected %u, got %u\n",
                   name, b, i, expected, got);
          }
          errors++;
        }
      }
    }
    EXPECT_EQ(errors, 0) << name << ": Total mismatches: " << errors;

    cudaFree(d_src);
    cudaFree(d_dst);
  }

  std::vector<uint8_t> h_input;
};

TEST_F(TMAClusterMulticastTest, CtaScopeTMALoad) {
  runTest(false, "CTA");
}

TEST_F(TMAClusterMulticastTest, ClusterMulticastTMALoad) {
  runTest(true, "Cluster");
}
