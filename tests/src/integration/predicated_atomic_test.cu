#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

__global__ void predicated_atomic_kernel(uint32_t* counter,
                                         uint32_t* lane_zero_old) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t old = 0;
  // Keep all lanes on one PTX instruction and predicate the atomic itself.
  // A CUDA `if (lane == 0)` may instead compile into divergent control flow,
  // which does not exercise predicated-off atomic bookkeeping.
  asm volatile("{\n\t"
               ".reg .pred selected;\n\t"
               "setp.eq.u32 selected, %2, 0;\n\t"
               "@selected atom.global.add.u32 %0, [%1], 1;\n\t"
               "}"
               : "+r"(old)
               : "l"(reinterpret_cast<uint64_t>(counter)), "r"(lane)
               : "memory");
  if (lane == 0) *lane_zero_old = old;
}

TEST(PredicatedAtomicIntegrationTest,
     PredicatedOffLanesDoNotCreateOutstandingAtomics) {
  uint32_t* device_counter = nullptr;
  uint32_t* device_lane_zero_old = nullptr;
  ASSERT_EQ(cudaMalloc(&device_counter, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_lane_zero_old, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_counter, 0, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_lane_zero_old, 0xff, sizeof(uint32_t)),
            cudaSuccess);

  predicated_atomic_kernel<<<1, 32>>>(device_counter, device_lane_zero_old);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t counter = 0;
  uint32_t lane_zero_old = ~uint32_t{0};
  ASSERT_EQ(cudaMemcpy(&counter, device_counter, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&lane_zero_old, device_lane_zero_old, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(counter, 1U);
  EXPECT_EQ(lane_zero_old, 0U);

  EXPECT_EQ(cudaFree(device_lane_zero_old), cudaSuccess);
  EXPECT_EQ(cudaFree(device_counter), cudaSuccess);
}

}  // namespace
