#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// CUDA 12.8 only supports PTX ISA 8.7, while .v8.b32 memory operands were
// added in PTX ISA 8.8. The test launcher replaces this placeholder with the
// adjacent handwritten PTX fixture so CI can exercise the simulator without
// requiring a newer ptxas. Keep the signature synchronized with that fixture.
extern "C" __global__ void vector_ldst_v8_kernel(const uint32_t *input,
                                                  uint32_t *output) {
  const unsigned lane = threadIdx.x;
  output[lane] = input[lane];
}

namespace {

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
