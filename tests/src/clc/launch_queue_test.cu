#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// This is work stealing, not replay of a synthetic fixed response: every grid
// item must be processed exactly once by an admitted CTA or its canceling peer.
template <bool Pair, bool Generic = false, bool ExpectBefore = false>
__global__ __launch_bounds__(32, 1, 2) void clc_work_queue(unsigned *counts,
                                                           unsigned *summary) {
  __shared__ __align__(16) unsigned response[4];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned next[4];
  const unsigned lane = threadIdx.x;
  unsigned rank = 0;
  if constexpr (Pair)
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned dest = (unsigned)__cvta_generic_to_shared(response);
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (lane == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    next[0] = blockIdx.x;
    next[1] = blockIdx.y;
    next[2] = blockIdx.z;
  }
  __syncthreads();
  unsigned phase = 0;
  for (;;) {
    if constexpr (Pair)
      asm volatile(
          "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
              : "memory");
    if (lane == 0) {
      atomicAdd(counts + next[0] + gridDim.x * (next[1] + gridDim.y * next[2]),
                1);
      if constexpr (ExpectBefore)
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16;" ::"r"(bar)
            : "memory");
      // The implementation must retain early completion credit too: enqueue
      // first and publish expect_tx afterwards, as in the retained kernels.
      if constexpr (Pair) {
        if (rank == 0)
          asm volatile(
              "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
              "complete_tx::bytes.multicast::cluster::all.b128 [%0],[%1];" ::
                  "r"(dest),
              "r"(bar)
              : "memory");
      } else if constexpr (Generic) {
        asm volatile(
            "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::"
            "bytes.b128 [%0],[%1];" ::"l"(response),
            "l"(&barrier)
            : "memory");
      } else {
        asm volatile(
            "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
            "complete_tx::bytes.b128 [%0],[%1];" ::"r"(dest),
            "r"(bar)
            : "memory");
      }
      if constexpr (!ExpectBefore)
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16;" ::"r"(bar)
            : "memory");
    }
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],%1; @!p bra wait; }" ::"r"(bar),
        "r"(phase)
        : "memory");
    if (lane == 0) {
      unsigned success, x, y, z, sx, sy, sz;
      asm volatile(
          "{ .reg .b128 r; .reg .pred p; .reg .b32 unused; "
          "ld.shared.b128 r,[%7]; "
          "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p,r; "
          "selp.u32 %0,1,0,p; "
          "mov.u32 %1,0; mov.u32 %2,0; mov.u32 %3,0; "
          "mov.u32 %4,0; mov.u32 %5,0; mov.u32 %6,0; "
          "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
          "{%1,%2,%3,unused},r; "
          "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
          "%4,r; "
          "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
          "%5,r; "
          "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 "
          "%6,r; }"
          : "=r"(success), "=r"(x), "=r"(y), "=r"(z), "=r"(sx), "=r"(sy),
            "=r"(sz)
          : "r"(dest)
          : "memory");
      if (success && (x != sx || y != sy || z != sz)) atomicAdd(summary + 1, 1);
      if (success && rank == 0) atomicAdd(summary, 1);
      next[0] = x + rank;
      next[1] = y;
      next[2] = z;
      next[3] = success;
    }
    __syncthreads();
    if (!next[3]) break;
    phase ^= 1;
  }
}

template <bool Pair, bool Generic = false, bool ExpectBefore = false>
void check_clc(dim3 grid, bool require_cancellation) {
  const size_t total = size_t(grid.x) * grid.y * grid.z;
  unsigned *counts = nullptr, *summary = nullptr;
  ASSERT_EQ(cudaMalloc(&counts, total * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&summary, 8), cudaSuccess);
  ASSERT_EQ(cudaFuncSetAttribute(clc_work_queue<Pair, Generic, ExpectBefore>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 144 * 1024),
            cudaSuccess);
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    ASSERT_EQ(cudaMemset(counts, 0, total * 4), cudaSuccess);
    ASSERT_EQ(cudaMemset(summary, 0, 8), cudaSuccess);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(32);
    config.dynamicSmemBytes = 144 * 1024;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {Pair ? 2u : 1u, 1u, 1u};
    if constexpr (Pair) {
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    ASSERT_EQ(
        cudaLaunchKernelEx(&config, clc_work_queue<Pair, Generic, ExpectBefore>,
                           counts, summary),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<unsigned> result(total);
    unsigned stats[2];
    ASSERT_EQ(
        cudaMemcpy(result.data(), counts, total * 4, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(stats, summary, 8, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (size_t i = 0; i < total; ++i)
      ASSERT_EQ(result[i], 1u) << i << ':' << repeat;
    ASSERT_EQ(stats[1], 0u);
    if (require_cancellation)
      EXPECT_GT(stats[0], 0u);
    else
      EXPECT_EQ(stats[0], 0u);
  }
  ASSERT_EQ(cudaFree(counts), cudaSuccess);
  ASSERT_EQ(cudaFree(summary), cudaSuccess);
}
TEST(ClcLaunch, SingleCtaCancellationAndSlotReuse) {
  check_clc<false>({64, 2, 2}, true);
}
TEST(ClcLaunch, PairedBroadcastCancellationAndSlotReuse) {
  check_clc<true>({64, 2, 2}, true);
}
TEST(ClcLaunch, ExhaustedSingleCtaReportsFailure) {
  check_clc<false>({1, 1, 1}, false);
}
TEST(ClcLaunch, ExhaustedPairReportsFailureToBothPeers) {
  check_clc<true>({2, 1, 1}, false);
}
TEST(ClcLaunch, GenericAddressesAndExpectBeforeCompletion) {
  check_clc<false, true, true>({64, 2, 2}, true);
}
TEST(ClcLaunch, PairExpectBeforeCompletion) {
  check_clc<true, false, true>({64, 2, 2}, true);
}
