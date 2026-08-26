// F8-7: real-shaped cluster path (not a 3-line micro-gtest).
//
// One producer CTA issues a TMA .shared::cluster multicast of a tile; every
// rank waits on mbarrier, then does a small elementwise accumulate and stores
// the result. This is the in-tree stand-in for a Triton/FA cluster kernel:
// no cluster-enabled Triton launcher exists under test/triton_trace/.
//
// Run with NoC on (H200 reduced, or SM120 reduced + overlay):
//   FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh \
//     -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 \
//     --group integration "ClusterRealShaped*"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

namespace {

constexpr int kTile = 256;
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
               "LAB_WAIT_RS:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@!P1 bra.uni LAB_WAIT_RS;\n"
               "}\n" ::"l"(bar_s),
               "r"(parity));
}

// Producer rank 0: TMA cluster multicast of kTile bytes.
// Every rank: wait, acc[i] = tile[i] + rank, store to out[rank * kTile + i].
template <int BYTES>
__global__ void cluster_tma_accumulate_kernel(const uint8_t *g_src,
                                              uint8_t *g_out, int n_ranks) {
  __shared__ uint8_t tile[BYTES];
  __shared__ unsigned long long bar;
  const int tid = threadIdx.x;
  const int rank = blockIdx.x;

  if (tid == 0)
    mbarrier_init_impl(&bar, 1);
  __syncthreads();
  if (tid == 0)
    mbarrier_arrive_expect_tx_impl(&bar, BYTES);
  __syncthreads();

  if (rank == 0 && tid == 0) {
    unsigned long long dst_s, src_g, bar_s;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(tile));
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(g_src));
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(&bar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %3, [%2];" ::"l"(dst_s),
        "l"(src_g), "l"(bar_s), "n"(BYTES));
  }

  if (tid == 0)
    mbarrier_try_wait_impl(&bar, 0);
  __syncthreads();

  for (int i = tid; i < BYTES; i += blockDim.x) {
    uint8_t v = static_cast<uint8_t>(tile[i] + static_cast<uint8_t>(rank));
    g_out[rank * BYTES + i] = v;
  }
  (void)n_ranks;
}

class ClusterRealShapedTest : public ::testing::Test {
 protected:
  static constexpr int kRanks = 2;
  void SetUp() override {
    h_in.resize(kTile);
    for (int i = 0; i < kTile; i++)
      h_in[i] = static_cast<uint8_t>(i * 5 + 3);
  }
  std::vector<uint8_t> h_in;
};

TEST_F(ClusterRealShapedTest, TmaProducerAccumulateTwoCta) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  uint8_t *d_src = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_src, kTile), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, kRanks * kTile), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, h_in.data(), kTile, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xFF, kRanks * kTile), cudaSuccess);

  int n = kRanks;
  void *args[] = {&d_src, &d_out, &n};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)cluster_tma_accumulate_kernel<kTile>,
                dim3(kRanks), dim3(kThreads), dim3(kRanks, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint8_t> h_out(kRanks * kTile);
  ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, kRanks * kTile,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int r = 0; r < kRanks; r++) {
    for (int i = 0; i < kTile; i++) {
      uint8_t exp = static_cast<uint8_t>(h_in[i] + static_cast<uint8_t>(r));
      EXPECT_EQ(h_out[r * kTile + i], exp) << "rank " << r << " i " << i;
    }
  }
  cudaFree(d_src);
  cudaFree(d_out);
}

}  // namespace
