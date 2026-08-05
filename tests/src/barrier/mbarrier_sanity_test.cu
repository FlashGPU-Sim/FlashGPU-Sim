#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "execution_mode.h"
#include "ptx/mbarrier.cuh"
#include "ptx/tma.cuh"

namespace {

using flashgpu::test::ptx::cp_async_bulk_tensor_1d_load;
using flashgpu::test::ptx::fence_proxy_tensormap_acquire;
using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_try_wait_parity;
using flashgpu::test::ptx::smem_u32_addr;
using flashgpu::test::ptx::tensormap_cp_fenceproxy;
using flashgpu::test::ptx::tensormap_set_global_address;

constexpr int kOutputSize = 64;
constexpr int kPollLimit = 4096;
constexpr int kTmaTensorMapBytes = 128;
constexpr int kTmaTileElements = 16;
constexpr int kTmaTileBytes = kTmaTileElements * static_cast<int>(sizeof(float));
constexpr int kTmaScratchBytes = 256;

__device__ inline void initialize_tensormap_shared(void* tmap_smem, int tid) {
  uint32_t* words = reinterpret_cast<uint32_t*>(tmap_smem);
  if (tid < 32) {
    words[tid] = 0;
  }
  __syncthreads();
}

__global__ void mbarrier_basic_phase_semantics_kernel(uint32_t* output) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    output[0] = mbarrier_try_wait_parity(&barrier, 0) ? 1u : 0u;
    mbarrier_arrive_expect_tx(&barrier, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    bool observed = false;
    int poll_count = 0;
    for (; poll_count < kPollLimit && !observed; ++poll_count) {
      observed = mbarrier_try_wait_parity(&barrier, 0);
    }

    output[1] = observed ? 1u : 0u;
    output[2] = mbarrier_try_wait_parity(&barrier, 0) ? 1u : 0u;
    output[3] = mbarrier_try_wait_parity(&barrier, 1) ? 1u : 0u;
    output[4] = static_cast<uint32_t>(poll_count);
  }
}

__global__ void mbarrier_true_false_distinguishability_kernel(uint32_t* output) {
  __shared__ uint64_t false_barrier;
  __shared__ uint64_t true_barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&false_barrier, 1);
    mbarrier_init(&true_barrier, 1);
    mbarrier_arrive_expect_tx(&true_barrier, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    bool ready = false;
    for (int i = 0; i < kPollLimit && !ready; ++i) {
      ready = mbarrier_try_wait_parity(&true_barrier, 0);
    }

    uint32_t false_count = 0;
    uint32_t true_count = 0;
    for (int i = 0; i < 32; ++i) {
      false_count += mbarrier_try_wait_parity(&false_barrier, 0) ? 1u : 0u;
      true_count += mbarrier_try_wait_parity(&true_barrier, 0) ? 1u : 0u;
    }

    output[0] = ready ? 1u : 0u;
    output[1] = false_count;
    output[2] = true_count;
  }
}

__global__ void mbarrier_arrive_visibility_kernel(uint32_t* output) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
    mbarrier_arrive_expect_tx(&barrier, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    bool observed = false;
    int poll_count = 0;
    for (; poll_count < kPollLimit && !observed; ++poll_count) {
      observed = mbarrier_try_wait_parity(&barrier, 0);
    }

    output[0] = observed ? 1u : 0u;
    output[1] = static_cast<uint32_t>(poll_count);
    output[2] = observed && mbarrier_try_wait_parity(&barrier, 0) ? 1u : 0u;
  }
}

__global__ void mbarrier_tma_completion_sanity_kernel(const float* source,
                                                      uint8_t* global_scratch,
                                                      uint32_t* output) {
  extern __shared__ uint8_t smem[];

  uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_smem =
      reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~static_cast<uintptr_t>(127));
  float* tile = reinterpret_cast<float*>(tmap_smem + kTmaTensorMapBytes);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(tmap_smem + kTmaTensorMapBytes + kTmaTileBytes);

  const int tid = static_cast<int>(threadIdx.x);
  initialize_tensormap_shared(tmap_smem, tid);

  if (tid == 0) {
    const uint64_t tmap_addr = __cvta_generic_to_shared(tmap_smem);

    tensormap_set_global_address(tmap_addr,
                                 reinterpret_cast<uint64_t>(source));
    asm volatile(
        "tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x0;"
        :
        : "l"(tmap_addr));

    const uint32_t box_dim = kTmaTileElements;
    asm volatile(
        "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;"
        :
        : "l"(tmap_addr), "r"(box_dim));

    const uint32_t global_dim = kTmaTileElements;
    asm volatile(
        "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;"
        :
        : "l"(tmap_addr), "r"(global_dim));

    const uint32_t element_stride = 1;
    asm volatile(
        "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 "
        "[%0], 0x0, %1;"
        :
        : "l"(tmap_addr), "r"(element_stride));

    asm volatile(
        "tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x7;"
        :
        : "l"(tmap_addr));
    asm volatile(
        "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 "
        "[%0], 0x0;"
        :
        : "l"(tmap_addr));
    asm volatile(
        "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], "
        "0x0;"
        :
        : "l"(tmap_addr));
    asm volatile(
        "tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;"
        :
        : "l"(tmap_addr));
  }
  __syncthreads();

  if (tid < 32) {
    const uint64_t tmap_addr = __cvta_generic_to_shared(tmap_smem);
    const uint64_t global_tmap = reinterpret_cast<uint64_t>(global_scratch);
    tensormap_cp_fenceproxy(global_tmap, tmap_addr);
    fence_proxy_tensormap_acquire(global_tmap);
  }
  __syncthreads();

  if (tid == 0) {
    output[0] = 0;
    output[1] = 0;
    output[2] = 0;

    mbarrier_init(barrier, 1);
    mbarrier_arrive_expect_tx(barrier, kTmaTileBytes);
  }
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  uint32_t elected = 0;
  uint32_t pred = 0;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "elect.sync %0|p, -1;\n"
               "selp.u32 %1, 1, 0, p;\n"
               "}\n"
               : "=r"(elected), "=r"(pred));

  if (pred != 0 && elected < 32 && tid < 32) {
    const uint32_t smem_addr = smem_u32_addr(tile);
    const uint32_t mbar_addr = smem_u32_addr(barrier);
    cp_async_bulk_tensor_1d_load(
        smem_addr, reinterpret_cast<uint64_t>(global_scratch), 0, mbar_addr);
  }
  __syncthreads();

  if (tid == 0) {
    bool observed = false;
    int poll_count = 0;
    for (; poll_count < kPollLimit && !observed; ++poll_count) {
      observed = mbarrier_try_wait_parity(barrier, 0);
    }
    output[0] = observed ? 1u : 0u;
    output[1] = static_cast<uint32_t>(poll_count);
  }
  __syncthreads();

  if (tid < kTmaTileElements && output[0] == 1u) {
    const uint32_t got = __float_as_uint(tile[tid]);
    const uint32_t expected = __float_as_uint(source[tid]);
    if (got != expected) {
      atomicAdd(&output[2], 1u);
    }
  }
}

}  // namespace

class MBarrierSanityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaError_t err = cudaMalloc(&d_output_, sizeof(uint32_t) * kOutputSize);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate output buffer";

    err = cudaMemset(d_output_, 0, sizeof(uint32_t) * kOutputSize);
    ASSERT_EQ(err, cudaSuccess) << "Failed to clear output buffer";

    err = cudaMalloc(&d_source_, sizeof(float) * kTmaTileElements);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate TMA source buffer";

    std::vector<float> source(kTmaTileElements);
    for (int i = 0; i < kTmaTileElements; ++i) {
      source[i] = static_cast<float>(100 + i);
    }
    err = cudaMemcpy(d_source_, source.data(),
                     sizeof(float) * static_cast<size_t>(kTmaTileElements),
                     cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to upload TMA source buffer";

    err = cudaMalloc(&d_global_scratch_, kTmaScratchBytes);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate TMA scratch buffer";

    err = cudaMemset(d_global_scratch_, 0, kTmaScratchBytes);
    ASSERT_EQ(err, cudaSuccess) << "Failed to clear TMA scratch buffer";
  }

  void TearDown() override {
    if (d_output_ != nullptr) {
      cudaFree(d_output_);
      d_output_ = nullptr;
    }
    if (d_source_ != nullptr) {
      cudaFree(d_source_);
      d_source_ = nullptr;
    }
    if (d_global_scratch_ != nullptr) {
      cudaFree(d_global_scratch_);
      d_global_scratch_ = nullptr;
    }
  }

  void ResetOutput() {
    const cudaError_t err =
        cudaMemset(d_output_, 0, sizeof(uint32_t) * kOutputSize);
    ASSERT_EQ(err, cudaSuccess) << "Failed to reset output buffer";
  }

  void SynchronizeAndAssert() {
    const cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  }

  std::vector<uint32_t> CopyOutput(int count) {
    std::vector<uint32_t> host(count);
    const cudaError_t err =
        cudaMemcpy(host.data(), d_output_, sizeof(uint32_t) * count,
                   cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy output buffer";
    return host;
  }

  uint32_t* d_output_ = nullptr;
  float* d_source_ = nullptr;
  uint8_t* d_global_scratch_ = nullptr;
};

TEST_F(MBarrierSanityTest, Phase) {
  mbarrier_basic_phase_semantics_kernel<<<1, 32>>>(d_output_);
  SynchronizeAndAssert();

  const auto output = CopyOutput(5);
  EXPECT_EQ(output[0], 0u) << "Initial try_wait on current parity should be false";
  EXPECT_EQ(output[1], 1u) << "Completed phase should become observable";
  EXPECT_EQ(output[2], 1u) << "Old parity should remain true after completion";
  EXPECT_EQ(output[3], 0u) << "New parity should start unsatisfied";
  EXPECT_LT(output[4], static_cast<uint32_t>(kPollLimit))
      << "Barrier completion should be observed before the poll limit";
}

TEST_F(MBarrierSanityTest, TryWait) {
  mbarrier_true_false_distinguishability_kernel<<<1, 32>>>(d_output_);
  SynchronizeAndAssert();

  const auto output = CopyOutput(3);
  EXPECT_EQ(output[0], 1u)
      << "Prepared completed barrier should become observable";
  EXPECT_EQ(output[1], 0u)
      << "Unsatisfied barrier should not spuriously return true";
  EXPECT_EQ(output[2], 32u)
      << "Completed barrier should consistently return true";
}

TEST_F(MBarrierSanityTest, Arrive) {
  mbarrier_arrive_visibility_kernel<<<1, 32>>>(d_output_);
  SynchronizeAndAssert();

  const auto output = CopyOutput(3);
  EXPECT_EQ(output[0], 1u)
      << "arrive_expect_tx(0) should become visible to try_wait";
  EXPECT_LT(output[1], static_cast<uint32_t>(kPollLimit))
      << "Visibility poll should converge before the safety bound";
  EXPECT_EQ(output[2], 1u)
      << "Once visible, try_wait on the old parity should stay true";
}

TEST_F(MBarrierSanityTest, TMA) {
  if (!flashgpu::test::running_on_native_gpu()) {
    GTEST_SKIP() << "TMA mbarrier sanity requires native GPU mode.";
  }

  ResetOutput();

  const size_t shared_memory_bytes =
      static_cast<size_t>(kTmaTensorMapBytes + kTmaTileBytes + sizeof(uint64_t) +
                          128);
  mbarrier_tma_completion_sanity_kernel<<<1, 32, shared_memory_bytes>>>(
      d_source_, d_global_scratch_, d_output_);
  SynchronizeAndAssert();

  const auto output = CopyOutput(3);
  EXPECT_EQ(output[0], 1u)
      << "TMA completion should flip the observer barrier";
  EXPECT_LT(output[1], static_cast<uint32_t>(kPollLimit))
      << "TMA barrier notification should be observed before the safety bound";
  EXPECT_EQ(output[2], 0u) << "Shared-memory tile should match source data";
}
