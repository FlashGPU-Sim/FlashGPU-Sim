#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ void vector_orders(const unsigned *in, unsigned *out) {
  const unsigned i = threadIdx.x;
  unsigned a, b, c, d;
  asm volatile("ld.global.u32.v4 {%0,%1,%2,%3},[%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "l"(in + 4 * i)
               : "memory");
  asm volatile("st.global.v4.u32 [%0],{%1,%2,%3,%4};" ::"l"(out + 4 * i),
               "r"(d), "r"(c), "r"(b), "r"(a)
               : "memory");
  unsigned long long x, y;
  asm volatile("ld.global.v2.u64 {%0,%1},[%2];"
               : "=l"(x), "=l"(y)
               : "l"(in + 4 * i)
               : "memory");
  asm volatile("st.global.u64.v2 [%0],{%1,%2};" ::"l"(out + 128 + 4 * i),
               "l"(y), "l"(x)
               : "memory");
}
TEST(PtxVectorOrder, ScalarBeforeOrAfterVector) {
  std::array<unsigned, 128> input;
  for (unsigned i = 0; i < 128; ++i) input[i] = 0xabc00000u + i;
  unsigned *in, *out;
  ASSERT_EQ(cudaMalloc(&in, 512), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 1024), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), 512, cudaMemcpyHostToDevice),
            cudaSuccess);
  vector_orders<<<1, 32>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::array<unsigned, 256> actual;
  ASSERT_EQ(cudaMemcpy(actual.data(), out, 1024, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 128; ++i) {
    EXPECT_EQ(actual[i], input[(i & ~3u) + (3 - (i & 3))]);
    EXPECT_EQ(actual[128 + i], input[i ^ 2]);
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
