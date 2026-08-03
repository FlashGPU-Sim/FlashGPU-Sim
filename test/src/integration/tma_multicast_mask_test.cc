// Selective TMA multicast destinations via PTX .multicast::cluster + ctaMask.
//
// Real CUDA: selective fan-out is cluster-level only (not shared::cta).
// Mask bit i selects TB-cluster rank i for both data and mbarrier complete_tx.
//
// Prefer configs with m >= 4: SM120_RTX5090_REDUCED_CLUSTER4x4
// Smoke: SM120_RTX5090_CLUSTER16x11
//
//   ./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 run test \
//       --target sm120 --group integration "*MulticastMask*"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

namespace {

constexpr int kChunk = 256;
constexpr int kThreads = 32;

__device__ inline void mbarrier_init_impl(unsigned long long *bar,
                                          unsigned expected) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(p),
               "r"(expected));
}

__device__ inline void mbarrier_arrive_expect_tx_impl(unsigned long long *bar,
                                                      unsigned tx_bytes) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"l"(
                   bar_s),
               "r"(tx_bytes));
}

__device__ inline void mbarrier_try_wait_impl(unsigned long long *bar,
                                               int parity) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar));
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT_MASK:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@!P1 bra.uni LAB_WAIT_MASK;\n"
               "}\n" ::"l"(bar_s),
               "r"(parity));
}

// One-producer: rank 0 issues .multicast::cluster load with ctaMask.
// Selected ranks wait on mbarrier and copy smem → out[rank * kChunk].
// Non-selected ranks leave out[] as prefill (0xAB).
template <int BYTES>
__global__ void multicast_mask_one_producer_kernel(const uint8_t *g_src,
                                                    uint8_t *g_out,
                                                    uint16_t cta_mask) {
  __shared__ uint8_t smem[BYTES + 64];
  __shared__ unsigned long long bar;

  int tid = threadIdx.x;
  int rank = (int)blockIdx.x;  // 1D clusterDim=(N,1,1) ⇒ rank == blockIdx.x

  // Prefill smem so non-destinations do not look like a successful load.
  for (int i = tid; i < BYTES; i += blockDim.x) {
    smem[i] = 0x5A;
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, BYTES);
  }
  __syncthreads();

  if (rank == 0 && tid == 0) {
    unsigned long long dst_s, src_g, bar_s;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem));
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(g_src));
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(&bar));
    // PTX: .shared::cluster + .multicast::cluster + ctaMask
    // ctaMask is a 16-bit operand in PTX (ptxas rejects .u32 here).
    asm volatile(
        "{\n"
        "  .reg .b16 mymask;\n"
        "  mov.b16 mymask, %4;\n"
        "  cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
        "multicast::cluster [%0], [%1], %3, [%2], mymask;\n"
        "}\n" ::"l"(dst_s),
        "l"(src_g), "l"(bar_s), "n"(BYTES), "h"(cta_mask));
  }
  __syncthreads();

  const bool selected = ((cta_mask >> rank) & 1) != 0;
  if (selected) {
    if (tid == 0) {
      mbarrier_try_wait_impl(&bar, 0);
    }
    __syncthreads();
    for (int i = tid; i < BYTES; i += blockDim.x) {
      g_out[rank * BYTES + i] = smem[i];
    }
  }
  // Non-selected ranks do not wait (would hang) and do not write g_out.
}

// Legacy bare .shared::cluster (no .multicast::cluster): all co-resident peers
// still receive the tile (compat with existing multicast model).
template <int BYTES>
__global__ void legacy_shared_cluster_all_peers_kernel(const uint8_t *g_src,
                                                        uint8_t *g_out) {
  __shared__ uint8_t smem[BYTES + 64];
  __shared__ unsigned long long bar;
  int tid = threadIdx.x;
  int rank = (int)blockIdx.x;

  if (tid == 0) {
    mbarrier_init_impl(&bar, 1);
  }
  __syncthreads();
  if (tid == 0) {
    mbarrier_arrive_expect_tx_impl(&bar, BYTES);
  }
  __syncthreads();

  if (rank == 0 && tid == 0) {
    unsigned long long dst_s, src_g, bar_s;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem));
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(g_src));
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(&bar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %3, [%2];" ::"l"(dst_s),
        "l"(src_g), "l"(bar_s), "n"(BYTES));
  }
  __syncthreads();
  if (tid == 0) {
    mbarrier_try_wait_impl(&bar, 0);
  }
  __syncthreads();
  for (int i = tid; i < BYTES; i += blockDim.x) {
    g_out[rank * BYTES + i] = smem[i];
  }
}

}  // namespace

class TmaMulticastMaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    h_in.resize(kChunk);
    for (int i = 0; i < kChunk; i++) {
      h_in[i] = static_cast<uint8_t>(i & 0xFF);
    }
  }

  std::vector<uint8_t> h_in;
};

TEST_F(TmaMulticastMaskTest, Mask_AllOnes_FourCtas) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);
  constexpr int N = 4;
  constexpr uint16_t mask = 0xF;

  uint8_t *d_src = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_src, kChunk), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, N * kChunk), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), kChunk, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xAB, N * kChunk), cudaSuccess);

  uint16_t mask_host = mask;
  void *args[] = {&d_src, &d_out, &mask_host};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)multicast_mask_one_producer_kernel<kChunk>,
                dim3(N), dim3(kThreads), dim3(N, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_out(N * kChunk, 0);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, N * kChunk, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int r = 0; r < N; r++) {
    for (int i = 0; i < kChunk; i++) {
      EXPECT_EQ(h_out[r * kChunk + i], h_in[i]) << "rank " << r << " i " << i;
    }
  }
  cudaFree(d_src);
  cudaFree(d_out);
}

TEST_F(TmaMulticastMaskTest, Mask_Subset_Ranks0And1) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);
  constexpr int N = 4;
  constexpr uint16_t mask = 0x3;  // ranks 0 and 1 only

  uint8_t *d_src = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_src, kChunk), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, N * kChunk), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), kChunk, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xAB, N * kChunk), cudaSuccess);

  uint16_t mask_host = mask;
  void *args[] = {&d_src, &d_out, &mask_host};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)multicast_mask_one_producer_kernel<kChunk>,
                dim3(N), dim3(kThreads), dim3(N, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_out(N * kChunk, 0);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, N * kChunk, cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int r = 0; r < N; r++) {
    for (int i = 0; i < kChunk; i++) {
      uint8_t got = h_out[r * kChunk + i];
      if (r == 0 || r == 1) {
        EXPECT_EQ(got, h_in[i]) << "selected rank " << r << " i " << i;
      } else {
        EXPECT_EQ(got, 0xAB) << "non-selected rank " << r << " i " << i;
      }
    }
  }
  cudaFree(d_src);
  cudaFree(d_out);
}

TEST_F(TmaMulticastMaskTest, Mask_SingleRank0) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  constexpr int N = 2;
  constexpr uint16_t mask = 0x1;  // only rank 0

  uint8_t *d_src = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_src, kChunk), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, N * kChunk), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), kChunk, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xAB, N * kChunk), cudaSuccess);

  uint16_t mask_host = mask;
  void *args[] = {&d_src, &d_out, &mask_host};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)multicast_mask_one_producer_kernel<kChunk>,
                dim3(N), dim3(kThreads), dim3(N, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_out(N * kChunk, 0);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, N * kChunk, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < kChunk; i++) {
    EXPECT_EQ(h_out[i], h_in[i]) << "rank0 i " << i;
    EXPECT_EQ(h_out[kChunk + i], 0xAB) << "rank1 i " << i;
  }
  cudaFree(d_src);
  cudaFree(d_out);
}

TEST_F(TmaMulticastMaskTest, NoMask_SharedCluster_LegacyAllPeers) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  constexpr int N = 2;

  uint8_t *d_src = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_src, kChunk), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, N * kChunk), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), kChunk, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xAB, N * kChunk), cudaSuccess);

  void *args[] = {&d_src, &d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)legacy_shared_cluster_all_peers_kernel<kChunk>,
                dim3(N), dim3(kThreads), dim3(N, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_out(N * kChunk, 0);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, N * kChunk, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int r = 0; r < N; r++) {
    for (int i = 0; i < kChunk; i++) {
      EXPECT_EQ(h_out[r * kChunk + i], h_in[i]) << "rank " << r << " i " << i;
    }
  }
  cudaFree(d_src);
  cudaFree(d_out);
}
