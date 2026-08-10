#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

__global__ void unpack_with_discarded_components(uint16_t *output,
                                                 uint64_t packed) {
  uint16_t word0 = 0;
  uint16_t word2 = 0;
  asm volatile("mov.b64 {%0, _, %1, _}, %2;"
               : "=h"(word0), "=h"(word2)
               : "l"(packed));
  output[0] = word0;
  output[1] = word2;
}

TEST(VectorOperandIntegrationTest, DiscardsPlaceholderComponents) {
  constexpr uint64_t kPacked = 0x1122334455667788ULL;
  constexpr uint16_t kExpected[] = {0x7788, 0x3344};

  uint16_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  unpack_with_discarded_components<<<1, 1>>>(output, kPacked);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint16_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], kExpected[0]);
  EXPECT_EQ(actual[1], kExpected[1]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

}  // namespace
