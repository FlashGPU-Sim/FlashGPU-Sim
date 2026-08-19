#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

struct Words8 {
  uint32_t value[8];
};

__device__ __forceinline__ Words8 load_global_v8(const uint32_t *address) {
  Words8 words;
  asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(words.value[0]), "=r"(words.value[1]),
                 "=r"(words.value[2]), "=r"(words.value[3]),
                 "=r"(words.value[4]), "=r"(words.value[5]),
                 "=r"(words.value[6]), "=r"(words.value[7])
               : "l"(reinterpret_cast<uint64_t>(address)));
  return words;
}

__device__ __forceinline__ void store_global_v8(uint32_t *address,
                                                const Words8 &words) {
  asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
               :
               : "l"(reinterpret_cast<uint64_t>(address)), "r"(words.value[0]),
                 "r"(words.value[1]), "r"(words.value[2]), "r"(words.value[3]),
                 "r"(words.value[4]), "r"(words.value[5]), "r"(words.value[6]),
                 "r"(words.value[7])
               : "memory");
}

__global__ void vector_ldst_v8_kernel(const uint32_t *input, uint32_t *output) {
  const unsigned lane = threadIdx.x;
  Words8 words = load_global_v8(input + lane);
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) words.value[i] ^= 0xa5a50000U + 17U * i;
  store_global_v8(output + 8 * lane, words);
}

TEST(BlackwellVectorLdstIntegrationTest,
     GlobalV8B32MovesAllEightWordsPerThread) {
  constexpr unsigned kThreads = 32;
  constexpr unsigned kInputWords = kThreads + 7;
  constexpr unsigned kOutputWords = 8 * kThreads;
  std::vector<uint32_t> input(kInputWords);
  std::vector<uint32_t> output(kOutputWords, 0);
  for (unsigned i = 0; i < kInputWords; ++i) input[i] = 0x10203040U + 31U * i;

  uint32_t *device_input = nullptr;
  uint32_t *device_output = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, kInputWords * sizeof(uint32_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, kOutputWords * sizeof(uint32_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(),
                       kInputWords * sizeof(uint32_t), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output, 0, kOutputWords * sizeof(uint32_t)),
            cudaSuccess);

  vector_ldst_v8_kernel<<<1, kThreads>>>(device_input, device_output);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output,
                       kOutputWords * sizeof(uint32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (unsigned lane = 0; lane < kThreads; ++lane) {
    for (unsigned i = 0; i < 8; ++i) {
      const unsigned index = 8 * lane + i;
      EXPECT_EQ(output[index], input[lane + i] ^ (0xa5a50000U + 17U * i))
          << "lane " << lane << ", element " << i;
    }
  }

  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

}  // namespace
