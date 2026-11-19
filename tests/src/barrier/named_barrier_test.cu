#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void named_barrier_arrive(uint32_t barrier_id,
                                                     uint32_t num_threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(barrier_id), "r"(num_threads)
               : "memory");
}

__device__ __forceinline__ void named_barrier_sync(uint32_t barrier_id,
                                                   uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(barrier_id), "r"(num_threads)
               : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier,
                                              uint32_t expected_arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(
                   smem_addr(barrier)),
               "r"(expected_arrivals)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t *barrier) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(
                   smem_addr(barrier))
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t *barrier,
                                              uint32_t phase) {
  uint32_t ticks = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "WAIT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
               "@p bra DONE;\n\t"
               "bra WAIT;\n\t"
               "DONE:\n\t"
               "}\n" ::"r"(smem_addr(barrier)),
               "r"(phase), "r"(ticks)
               : "memory");
}

template <int Bytes>
__device__ __forceinline__ void cp_async_bulk_store(void *global_dst,
                                                    const void *smem_src) {
  uint64_t dst_g = 0;
  uint64_t src_s = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(dst_g) : "l"(global_dst));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(src_s) : "l"(smem_src));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :
               : "l"(dst_g), "l"(src_s), "n"(Bytes)
               : "memory");
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async;" ::: "memory");
}

__global__ void named_barrier_does_not_release_mbarrier_kernel(
    int *early_release) {
  __shared__ __align__(8) uint64_t barrier;
  __shared__ volatile int arrived;
  __shared__ volatile int released;

  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;

  if (threadIdx.x == 0) {
    arrived = 0;
    released = 0;
    mbarrier_init(&barrier, 1);
  }
  __syncthreads();

  if (warp_id == 0) {
    named_barrier_arrive(1, 64);
    if (lane_id == 0) {
      arrived = 1;
    }
    mbarrier_wait(&barrier, 0);
    if (lane_id == 0 && released == 0) {
      atomicExch(early_release, 1);
    }
  } else if (warp_id == 1) {
    while (arrived == 0) {
    }
    named_barrier_sync(1, 64);

    uint32_t delay = static_cast<uint32_t>(lane_id);
    for (int i = 0; i < 4096; ++i) {
      asm volatile("add.u32 %0, %0, 1;" : "+r"(delay)::"memory");
    }

    if (lane_id == 0) {
      if (delay == 0xffffffffu) {
        atomicExch(early_release, 2);
      }
      released = 1;
    }
    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive(&barrier);
    }
  }
}

__global__ void bulk_completion_does_not_release_mbarrier_kernel(
    int *early_release, uint8_t *global_store) {
  __shared__ __align__(16) uint8_t store_smem[256];
  __shared__ __align__(8) uint64_t barrier;
  __shared__ volatile int tma_issued;
  __shared__ volatile int release_allowed;

  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;

  if (threadIdx.x == 0) {
    tma_issued = 0;
    release_allowed = 0;
    mbarrier_init(&barrier, 1);
  }
  if (threadIdx.x < 64) {
    reinterpret_cast<uint32_t *>(store_smem)[threadIdx.x] = threadIdx.x;
  }
  __syncthreads();

  if (warp_id == 0) {
    if (lane_id == 0) {
      fence_proxy_async();
      cp_async_bulk_store<256>(global_store, store_smem);
      cp_async_bulk_commit_group();
      tma_issued = 1;
    }
    __syncwarp();

    mbarrier_wait(&barrier, 0);
    if (lane_id == 0 && release_allowed == 0) {
      atomicExch(early_release, 1);
    }
  } else if (warp_id == 1) {
    while (tma_issued == 0) {
    }

    uint32_t delay = static_cast<uint32_t>(lane_id);
    for (int i = 0; i < 8192; ++i) {
      asm volatile("add.u32 %0, %0, 1;" : "+r"(delay)::"memory");
    }

    if (lane_id == 0) {
      if (delay == 0xffffffffu) {
        atomicExch(early_release, 2);
      }
      release_allowed = 1;
    }
    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive(&barrier);
    }
  }
}

template <bool ExitAfterArrive>
__global__ void named_barrier_publish_kernel(int *errors) {
  __shared__ volatile unsigned values[32];
  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  const unsigned address = smem_addr(const_cast<unsigned *>(values) + lane);
  // A predicated-off shared store must not leave an outstanding fence.
  asm volatile("{ .reg .pred p; setp.eq.u32 p, 0, 1; "
               "@p st.shared.u32 [%0], 0; }" :: "r"(address) : "memory");
  __syncthreads();
  for (unsigned round = 1; round <= (ExitAfterArrive ? 1u : 8u); ++round) {
    if (warp == 0) {
      values[lane] = blockIdx.x * 256 + round * 32 + lane;
      named_barrier_arrive(1, 64);
      if (ExitAfterArrive) return;
      named_barrier_sync(2, 64);
    } else {
      named_barrier_sync(1, 64);
      if (values[lane] != blockIdx.x * 256 + round * 32 + lane)
        atomicAdd(errors, 1);
      if (!ExitAfterArrive) named_barrier_arrive(2, 64);
    }
  }
}

template <bool ExitAfterArrive>
void check_named_barrier_publish() {
  int *errors = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&errors, sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMemset(errors, 0, sizeof(int)));
  // Repeated launches also exercise reuse of warp and CTA state.
  for (unsigned launch = 0; launch < 2; ++launch) {
    named_barrier_publish_kernel<ExitAfterArrive><<<64, 64>>>(errors);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  }
  int result = -1;
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, errors, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(result, 0);
  ASSERT_EQ(cudaSuccess, cudaFree(errors));
}

enum class CtaSync { Block, Predicated, Counted, Unaligned, DeviceCall };

__device__ __noinline__ void sync_in_device_function() { __syncthreads(); }

// The exit edge prevents the compiler's static control-flow graph from
// making the barrier the branch's postdominator. Each warp's lane 0 must
// publish before either warp can consume, even if lane 1 exits.
template <CtaSync Form>
__global__ void divergent_cta_barrier_kernel(unsigned *output, bool exit_lane) {
  __shared__ unsigned published[2];
  unsigned lane = threadIdx.x % warpSize;
  unsigned warp = threadIdx.x / warpSize;
  if (lane == 0) published[warp] = 0;
  __syncthreads();
  unsigned sum = 0;
  for (unsigned round = 1; round <= 3; ++round) {
    asm volatile("{ .reg .pred skip, leave, enabled;\n\t"
                 "setp.ge.u32 enabled, %0, 0;\n\t"
                 "setp.gt.u32 skip, %0, 1;\n\t"
                 "@skip bra JOIN%=;\n\t"
                 "setp.eq.u32 leave, %0, %1;\n\t"
                 "@leave exit;\n\t"
                 "setp.ne.u32 skip, %0, 0;\n\t"
                 "@skip bra JOIN%=;\n\t"
                 "@!enabled bar.sync 1;\n\t"  // Always skipped.
                 "st.shared.u32 [%2], %3;\n\t"
                 "JOIN%=: }"
                 :: "r"(lane), "r"(exit_lane ? 1u : ~0u),
                    "r"(smem_addr(&published[warp])), "r"(round)
                 : "memory");
    if constexpr (Form == CtaSync::Predicated) {
      asm volatile("{ .reg .pred enabled; setp.ge.u32 enabled, %0, 0; "
                   "@enabled bar.sync 0; }" :: "r"(lane) : "memory");
    } else if constexpr (Form == CtaSync::Counted) {
      asm volatile("bar.sync 0, 64;" ::: "memory");
    } else if constexpr (Form == CtaSync::Unaligned) {
      asm volatile("barrier.sync 0;" ::: "memory");
    } else if constexpr (Form == CtaSync::DeviceCall) {
      sync_in_device_function();
    } else {
      __syncthreads();
    }
    sum += published[1 - warp];
    __syncthreads();
  }
  output[threadIdx.x] = sum;
}

__global__ void independent_named_barriers(unsigned *output) {
  __shared__ volatile unsigned values[32];
  const unsigned warp = threadIdx.x / warpSize;
  const unsigned lane = threadIdx.x % warpSize;
  if (warp == 0) values[lane] = 0;
  __syncthreads();
  if (warp == 0) {
    asm volatile("barrier.sync 1, 64;" ::: "memory");
    output[lane] = values[lane];
  } else {
    asm volatile("barrier.sync 2, 64;" ::: "memory");
    if (warp == 2) {
      values[lane] = lane + 123;
      asm volatile("barrier.sync 1, 64;" ::: "memory");
    }
  }
}

__device__ __forceinline__ void predicated_barrier_body(unsigned *output) {
  __shared__ unsigned data[64];
  unsigned tid = threadIdx.x;
  data[tid] = 0;
  __syncthreads();
  unsigned result = 0;
  unsigned address = __cvta_generic_to_shared(data + tid);
  unsigned peer = __cvta_generic_to_shared(data + (tid ^ 33));
  asm volatile(
      "{ .reg .pred odd; .reg .b32 parity; and.b32 parity, %1, 1; setp.ne.u32 "
      "odd, parity, 0;\n"
      "@odd barrier.sync 1, 64;\n"
      "bra.uni AFTER%=;\n AFTER%=: \n"
      "@odd ld.shared.u32 %0, [%4];\n"
      "@!odd st.shared.u32 [%2], %3;\n"
      "@!odd barrier.sync 1, 64; }"
      : "+r"(result)
      : "r"(tid), "r"(address), "r"(tid + 123), "r"(peer)
      : "memory");
  output[tid] = __ballot_sync(0xffffffff, tid & 1) == 0xaaaaaaaa
                    ? result : ~0u;
}

__global__ void predicated_barrier_kernel(unsigned *output) {
  predicated_barrier_body(output);
}

__global__ void divergent_arrival_kernel(unsigned *output) {
  __shared__ volatile unsigned payload[32];
  unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  if (warp == 0) payload[lane] = 0;
  __syncthreads();
  if (warp == 0) {
    if (lane & 1) {
      asm volatile("barrier.arrive 1, 96; mov.u32 %0, 34;"
                   : "=r"(tid)::"memory");
    } else {
      asm volatile("barrier.arrive 1, 96; mov.u32 %0, 35;"
                   : "=r"(tid)::"memory");
    }
    asm volatile("barrier.sync 2, 64;" ::: "memory");
    output[32 + lane] = tid;
  } else if (warp == 1) {
    asm volatile("barrier.sync 1, 96;" ::: "memory");
    output[lane] = payload[lane];
  } else {
    asm volatile("barrier.sync 2, 64;" ::: "memory");
    payload[lane] = 123 + lane;
    asm volatile("barrier.arrive 1, 96;" ::: "memory");
  }
}

template <CtaSync Form>
void check_divergent_cta_barrier(unsigned threads, bool exit_lane) {
  SCOPED_TRACE(::testing::Message() << "threads=" << threads
                                    << " exit_lane=" << exit_lane);
  unsigned *output = nullptr;
  std::vector<unsigned> result(threads);
  const size_t bytes = threads * sizeof(unsigned);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output, bytes));
  ASSERT_EQ(cudaSuccess, cudaMemset(output, 0, bytes));
  divergent_cta_barrier_kernel<Form><<<1, threads>>>(output, exit_lane);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result.data(), output, bytes,
                                    cudaMemcpyDeviceToHost));
  EXPECT_EQ(cudaSuccess, cudaFree(output));
  for (unsigned tid = 0; tid < threads; ++tid)
    EXPECT_EQ(result[tid], exit_lane && tid % 32 == 1 ? 0u : 6u) << tid;
}

}  // namespace

TEST(NamedBarrierIntegrationTest, ArrivalSurvivesProducerExit) {
  check_named_barrier_publish<true>();
}

TEST(NamedBarrierIntegrationTest, PublishesSharedStoresAcrossRepeatedGenerations) {
  check_named_barrier_publish<false>();
}

TEST(NamedBarrierIntegrationTest, ArriveDoesNotReleaseMbarrierWaiter) {
  int *d_early_release = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_early_release, sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_early_release, 0, sizeof(int)));

  named_barrier_does_not_release_mbarrier_kernel<<<1, 64>>>(d_early_release);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  int early_release = 0;
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&early_release, d_early_release,
                                    sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(early_release, 0);

  ASSERT_EQ(cudaSuccess, cudaFree(d_early_release));
}

TEST(MixedBarrierIntegrationTest, BulkCompletionDoesNotReleaseMbarrierWaiter) {
  int *d_early_release = nullptr;
  uint8_t *d_store = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_early_release, sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_store, 256));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_early_release, 0, sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_store, 0, 256));

  bulk_completion_does_not_release_mbarrier_kernel<<<1, 64>>>(d_early_release,
                                                              d_store);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  int early_release = 0;
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&early_release, d_early_release,
                                    sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(early_release, 0);

  ASSERT_EQ(cudaSuccess, cudaFree(d_store));
  ASSERT_EQ(cudaSuccess, cudaFree(d_early_release));
}

TEST(NamedBarrierIntegrationTest, CtaSyncWaitsForDivergentLiveLanes) {
  for (bool exit_lane : {false, true}) {
    check_divergent_cta_barrier<CtaSync::Block>(48, exit_lane);
    check_divergent_cta_barrier<CtaSync::Predicated>(48, exit_lane);
  }
}

TEST(NamedBarrierIntegrationTest, CountedSyncWaitsForDivergentLiveLanes) {
  check_divergent_cta_barrier<CtaSync::Counted>(64, false);
}

TEST(NamedBarrierIntegrationTest, UnalignedSyncWaitsForDivergentLiveLanes) {
  check_divergent_cta_barrier<CtaSync::Unaligned>(64, false);
}

TEST(NamedBarrierIntegrationTest, DeviceCallSyncWaitsForDivergentLiveLanes) {
  for (bool exit_lane : {false, true})
    check_divergent_cta_barrier<CtaSync::DeviceCall>(64, exit_lane);
}

TEST(NamedBarrierIntegrationTest, CountedBarrierReleasesOnlyItsWaiters) {
  unsigned result[32];
  unsigned *output = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output, sizeof(result)));
  independent_named_barriers<<<1, 96>>>(output);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result, output, sizeof(result),
                                    cudaMemcpyDeviceToHost));
  EXPECT_EQ(cudaSuccess, cudaFree(output));
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(result[lane], lane + 123) << lane;
}

TEST(NamedBarrierIntegrationTest, PredicatedLanesWaitForTheirPeers) {
  unsigned *output = nullptr;
  unsigned result[64];
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output, sizeof(result)));
  predicated_barrier_kernel<<<1, 64>>>(output);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result, output, sizeof(result),
                                    cudaMemcpyDeviceToHost));
  EXPECT_EQ(cudaSuccess, cudaFree(output));
  for (unsigned tid = 0; tid < 64; ++tid)
    EXPECT_EQ(result[tid], tid & 1 ? (tid ^ 33) + 123 : 0) << tid;
}

TEST(NamedBarrierIntegrationTest, DivergentArrivalsCountWarpOnce) {
  unsigned *output = nullptr;
  unsigned result[64];
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output, sizeof(result)));
  divergent_arrival_kernel<<<1, 96>>>(output);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result, output, sizeof(result),
                                    cudaMemcpyDeviceToHost));
  EXPECT_EQ(cudaSuccess, cudaFree(output));
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(result[lane], 123 + lane) << lane;
    EXPECT_EQ(result[lane + 32], lane & 1 ? 34 : 35) << lane;
  }
}
