#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ void wide_copy(const unsigned *in, unsigned *out) {
  unsigned a, b, c, d, e, f, g, h;
  in += 8 * threadIdx.x;
  out += 8 * threadIdx.x;
  asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f), "=r"(g),
                 "=r"(h)
               : "l"(in));
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(out),
               "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(g), "r"(h)
               : "memory");
}
__global__ void signed_bytes(const signed char *in, int *out) {
  int value;
  asm volatile("ld.global.s8 %0,[%1];" : "=r"(value) : "l"(in + threadIdx.x));
  out[threadIdx.x] = value;
}
TEST(NativeDataMovement, WideGlobalCopyPreservesGuards) {
  std::array<unsigned, 272> in, out;
  for (unsigned i = 0; i < in.size(); ++i) in[i] = 0x80000001u + 17 * i;
  unsigned *a, *b;
  ASSERT_EQ(cudaMalloc(&a, sizeof(in)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&b, sizeof(out)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(a, in.data(), sizeof(in), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(b, 0xa5, sizeof(out)), cudaSuccess);
  wide_copy<<<1, 32>>>(a + 8, b + 8);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), b, sizeof(out), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < out.size(); ++i)
    EXPECT_EQ(out[i], i >= 8 && i < 264 ? in[i] : 0xa5a5a5a5u) << i;
  EXPECT_EQ(cudaFree(a), cudaSuccess);
  EXPECT_EQ(cudaFree(b), cudaSuccess);
}
TEST(NativeDataMovement, SignedByteValues) {
  std::array<signed char, 32> in;
  std::array<int, 32> out;
  for (unsigned i = 0; i < 32; ++i) in[i] = static_cast<signed char>(i * 8);
  signed char *a;
  int *b;
  ASSERT_EQ(cudaMalloc(&a, sizeof(in)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&b, sizeof(out)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(a, in.data(), sizeof(in), cudaMemcpyHostToDevice),
            cudaSuccess);
  signed_bytes<<<1, 32>>>(a, b);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), b, sizeof(out), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(out[i], int(in[i])) << i;
  EXPECT_EQ(cudaFree(a), cudaSuccess);
  EXPECT_EQ(cudaFree(b), cudaSuccess);
}
}  // namespace
