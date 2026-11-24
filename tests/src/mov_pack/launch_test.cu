#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ void pack_literals(unsigned long long *out) {
  unsigned i = threadIdx.x, b, c;
  unsigned short a;
  unsigned long long d;
  asm volatile(
      "{.reg .b8 lo; .reg .b16 half; cvt.u8.u32 lo,%4; cvt.u16.u32 half,%4;"
      "mov.b16 %0,{lo,0x12};mov.b32 %1,{lo,1,2,3};"
      "mov.b32 %2,{half,0xabcd};mov.b64 %3,{%4,0xef123456};}"
      : "=h"(a), "=r"(b), "=r"(c), "=l"(d)
      : "r"(i + 0x1234));
  out[i * 4] = a;
  out[i * 4 + 1] = b;
  out[i * 4 + 2] = c;
  out[i * 4 + 3] = d;
}
TEST(MovLiteralPack, MixedRegistersAndIntegersAtAllWidths) {
  std::array<unsigned long long, 128> actual{};
  unsigned long long *out;
  ASSERT_EQ(cudaMalloc(&out, sizeof(actual)), cudaSuccess);
  pack_literals<<<1, 32>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, sizeof(actual), cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned i = 0; i < 32; ++i) {
    EXPECT_EQ(actual[i * 4], 0x1200u | ((i + 0x1234) & 255));
    EXPECT_EQ(actual[i * 4 + 1], 0x03020100u | ((i + 0x1234) & 255));
    EXPECT_EQ(actual[i * 4 + 2], 0xabcd0000u | (i + 0x1234));
    EXPECT_EQ(actual[i * 4 + 3], 0xef12345600000000ULL | (i + 0x1234));
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
