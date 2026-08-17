#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <numeric>

#include "ptx/mbarrier.cuh"

namespace {

using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;
using flashgpu::test::ptx::smem_u64_addr;

constexpr unsigned kChunkBytes = 4096;
constexpr unsigned kCopiesPerPhase = 4;
constexpr unsigned kPhaseBytes = kChunkBytes * kCopiesPerPhase;
constexpr unsigned kPhases = 2;
constexpr unsigned kWordsPerPhase = kPhaseBytes / sizeof(uint32_t);
constexpr unsigned kTotalWords = kPhases * kWordsPerPhase;

__device__ __forceinline__ uint64_t global_address(const void* pointer) {
  uint64_t address = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(address) : "l"(pointer));
  return address;
}

__device__ __forceinline__ void linear_bulk_g2s(void* destination,
                                                const void* source,
                                                uint64_t* barrier) {
  const uint64_t destination_s = smem_u64_addr(destination);
  const uint64_t source_g = global_address(source);
  const uint64_t barrier_s = smem_u64_addr(barrier);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "l"(destination_s), "l"(source_g), "n"(kChunkBytes),
        "l"(barrier_s)
      : "memory");
}

__global__ void linear_bulk_two_phase_kernel(const uint32_t* input,
                                              uint64_t* phase_sums) {
  extern __shared__ __align__(16) uint8_t shared[];
  if (threadIdx.x != 0) return;

  auto* barrier = reinterpret_cast<uint64_t*>(shared + kPhaseBytes);
  mbarrier_init(barrier, 1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

  for (unsigned phase = 0; phase < kPhases; ++phase) {
    mbarrier_arrive_expect_tx(barrier, kPhaseBytes);
    for (unsigned copy = 0; copy < kCopiesPerPhase; ++copy) {
      linear_bulk_g2s(shared + copy * kChunkBytes,
                      input + phase * kWordsPerPhase +
                          copy * (kChunkBytes / sizeof(uint32_t)),
                      barrier);
    }

    mbarrier_wait_parity(barrier, phase & 1);
    asm volatile("" ::: "memory");

    const auto* words = reinterpret_cast<const uint32_t*>(shared);
    uint64_t sum = 0;
    for (unsigned word = 0; word < kWordsPerPhase; ++word) {
      sum += words[word];
    }
    phase_sums[phase] = sum;
  }

  mbarrier_inval(barrier);
}

TEST(TmaLinearBulkLifecycleIntegrationTest,
     ReusesBarrierAfterFourCopiesComplete) {
  std::array<uint32_t, kTotalWords> input{};
  for (unsigned i = 0; i < input.size(); ++i) input[i] = i + 1;

  std::array<uint64_t, kPhases> expected{};
  for (unsigned phase = 0; phase < kPhases; ++phase) {
    expected[phase] = std::accumulate(
        input.begin() + phase * kWordsPerPhase,
        input.begin() + (phase + 1) * kWordsPerPhase, uint64_t{0});
  }

  uint32_t* device_input = nullptr;
  uint64_t* device_sums = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_sums, sizeof(expected)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), sizeof(input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  constexpr unsigned kSharedBytes = kPhaseBytes + sizeof(uint64_t);
  linear_bulk_two_phase_kernel<<<1, 1, kSharedBytes>>>(device_input,
                                                       device_sums);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::array<uint64_t, kPhases> actual{};
  ASSERT_EQ(cudaMemcpy(actual.data(), device_sums, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, expected);

  EXPECT_EQ(cudaFree(device_sums), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

}  // namespace
