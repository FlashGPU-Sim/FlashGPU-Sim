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
    mbarrier_arrive(&barrier);
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
    mbarrier_arrive(&barrier);
  }
}

}  // namespace

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
