#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

__global__ void multiply_halves(const uint64_t *inputs, uint64_t *output) {
  const unsigned index = threadIdx.x;
  const uint64_t a = inputs[index / 8], b = inputs[index % 8];
  uint64_t high_unsigned, high_signed, low_unsigned, low_signed;
  asm volatile("mul.hi.u64 %0, %1, %2;"
               : "=l"(high_unsigned) : "l"(a), "l"(b));
  asm volatile("mul.hi.s64 %0, %1, %2;"
               : "=l"(high_signed) : "l"(a), "l"(b));
  asm volatile("mul.lo.u64 %0, %1, %2;"
               : "=l"(low_unsigned) : "l"(a), "l"(b));
  asm volatile("mul.lo.s64 %0, %1, %2;"
               : "=l"(low_signed) : "l"(a), "l"(b));
  output[4 * index] = high_unsigned;
  output[4 * index + 1] = high_signed;
  output[4 * index + 2] = low_unsigned;
  output[4 * index + 3] = low_signed;
}

TEST(IntegerMultiply64, HighAndLowHalves) {
  const uint64_t inputs[] = {
      0, 1, 2, UINT64_MAX, 0x8000000000000000ULL,
      0x7fffffffffffffffULL, 5270498306774157605ULL,
      0xfedcba9876543210ULL};
  uint64_t *device_inputs = nullptr, *device_output = nullptr;
  uint64_t output[64 * 4] = {};
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_inputs, sizeof(inputs)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_output, sizeof(output)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_inputs, inputs, sizeof(inputs),
                                    cudaMemcpyHostToDevice));
  multiply_halves<<<1, 64>>>(device_inputs, device_output);
  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output, device_output, sizeof(output),
                                    cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaFree(device_output));
  ASSERT_EQ(cudaSuccess, cudaFree(device_inputs));
  for (unsigned i = 0; i < 64; ++i) {
    const uint64_t a = inputs[i / 8], b = inputs[i % 8];
    int64_t signed_a, signed_b;
    std::memcpy(&signed_a, &a, sizeof(a));
    std::memcpy(&signed_b, &b, sizeof(b));
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    const __int128 signed_product = static_cast<__int128>(signed_a) * signed_b;
    EXPECT_EQ(static_cast<uint64_t>(product >> 64), output[4 * i]) << i;
    EXPECT_EQ(static_cast<uint64_t>(
                  static_cast<unsigned __int128>(signed_product) >> 64),
              output[4 * i + 1]) << i;
    EXPECT_EQ(static_cast<uint64_t>(product), output[4 * i + 2]) << i;
    EXPECT_EQ(static_cast<uint64_t>(product), output[4 * i + 3]) << i;
  }
}

}  // namespace
