#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

__global__ void load_with_large_address_offset(uint32_t *output,
                                               uint64_t biased_base) {
  uint32_t value = 0;
  asm volatile("ld.global.u32 %0, [%1+4294967296];"
               : "=r"(value)
               : "l"(biased_base));
  output[0] = value;
}

TEST(AddressOperandIntegrationTest, PreservesOffsetAbove32Bits) {
  constexpr uint32_t kExpected = 0x5A17C0DEu;
  constexpr uint64_t kLargeOffset = 1ULL << 32;

  uint32_t *input = nullptr;
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&input, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(input, &kExpected, sizeof(kExpected), cudaMemcpyHostToDevice),
      cudaSuccess);

  const uint64_t biased_base = reinterpret_cast<uint64_t>(input) - kLargeOffset;
  load_with_large_address_offset<<<1, 1>>>(output, biased_base);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpected);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
}

}  // namespace
