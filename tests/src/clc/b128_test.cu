#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Exercise all four words, including the high 64 bits which a truncated CLC
// handle can accidentally leave zero. Also covers legacy four-register packing.
__global__ void scalar_b128_roundtrip(const unsigned *input, unsigned *output) {
  __shared__ __align__(16) unsigned scratch[4];
  if (threadIdx.x) return;
  unsigned a, b, c, d;
  unsigned address = (unsigned)__cvta_generic_to_shared(scratch);
  asm volatile(
      "{ .reg .b128 r,s; "
      "ld.global.b128 r,[%4]; st.shared.b128 [%5],r; "
      "ld.shared.b128 s,[%5]; mov.b128 {%0,%1,%2,%3},s; "
      "mov.b128 r,{%3,%2,%1,%0}; st.global.b128 [%6],r; }"
      : "=&r"(a), "=&r"(b), "=&r"(c), "=&r"(d)
      : "l"(input), "r"(address), "l"(output)
      : "memory");
  asm volatile(
      "{ .reg .b128 r; .reg .b64 lo,hi; ld.global.b128 r,[%0]; "
      "mov.b128 {lo,hi},r; mov.b128 r,{hi,lo}; st.global.b128 [%1],r; }" ::"l"(
          input),
      "l"(output + 8)
      : "memory");
  output[4] = a;
  output[5] = b;
  output[6] = c;
  output[7] = d;
}

TEST(PtxB128, ScalarMemoryAndRegisterRoundtrip) {
  const unsigned input[4] = {0x12345678, 0xabcdef01, 0x87654321, 0xfedcba98};
  unsigned *in, *out, result[12];
  ASSERT_EQ(cudaMalloc(&in, sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, sizeof(result)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input, sizeof(input), cudaMemcpyHostToDevice),
            cudaSuccess);
  scalar_b128_roundtrip<<<1, 32>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], input[3 - i]);
    EXPECT_EQ(result[i + 4], input[i]);
    EXPECT_EQ(result[i + 8], input[(i + 2) % 4]);
  }
  ASSERT_EQ(cudaFree(in), cudaSuccess);
  ASSERT_EQ(cudaFree(out), cudaSuccess);
}
