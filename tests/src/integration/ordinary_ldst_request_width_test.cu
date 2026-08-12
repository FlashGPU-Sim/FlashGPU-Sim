#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ uint32_t load_global_cg(const uint32_t *address) {
  uint32_t value;
  asm volatile("ld.global.cg.u32 %0, [%1];"
               : "=r"(value)
               : "l"(reinterpret_cast<uint64_t>(address)));
  return value;
}

__device__ __forceinline__ void store_global_cg(uint32_t *address,
                                                uint32_t value) {
  asm volatile("st.global.cg.u32 [%0], %1;"
               :
               : "l"(reinterpret_cast<uint64_t>(address)), "r"(value)
               : "memory");
}

__global__ void ordinary_ldst_request_width_kernel(const uint32_t *input,
                                                   uint32_t *output,
                                                   uint32_t *atomic_count) {
  const unsigned lane = threadIdx.x & 31;
  const uint32_t value = load_global_cg(input + lane) ^ 0x5a5a5a5aU;
  store_global_cg(output + lane, value);
  if (lane == 0) atomicAdd(atomic_count, 1U);
}

TEST(OrdinaryLdstRequestWidthIntegrationTest,
     MultiSectorLoadStoreAndAtomicCompleteExactlyOnce) {
  constexpr unsigned kThreads = 32;
  std::vector<uint32_t> input(kThreads);
  std::vector<uint32_t> output(kThreads, 0);
  for (unsigned lane = 0; lane < kThreads; ++lane)
    input[lane] = 0x10203040U + 17U * lane;

  uint32_t *device_input = nullptr;
  uint32_t *device_output = nullptr;
  uint32_t *device_atomic_count = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, kThreads * sizeof(uint32_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, kThreads * sizeof(uint32_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_atomic_count, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(),
                       kThreads * sizeof(uint32_t), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output, 0, kThreads * sizeof(uint32_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_atomic_count, 0, sizeof(uint32_t)), cudaSuccess);

  ordinary_ldst_request_width_kernel<<<1, kThreads>>>(
      device_input, device_output, device_atomic_count);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t atomic_count = 0;
  ASSERT_EQ(cudaMemcpy(output.data(), device_output,
                       kThreads * sizeof(uint32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&atomic_count, device_atomic_count, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  uint64_t checksum = atomic_count;
  for (unsigned lane = 0; lane < kThreads; ++lane) {
    EXPECT_EQ(output[lane], input[lane] ^ 0x5a5a5a5aU) << "lane " << lane;
    checksum += output[lane];
  }
  EXPECT_EQ(atomic_count, 1U);
  std::printf("ordinary_ldst_request_width_checksum = %llu\n",
              static_cast<unsigned long long>(checksum));

  EXPECT_EQ(cudaFree(device_atomic_count), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

}  // namespace
