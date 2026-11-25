#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
namespace {
struct Payload {
  uint64_t a, b, c, d;
};
__device__ __noinline__ Payload mutate(Payload value, unsigned n) {
  value.a += n;
  value.b ^= value.a;
  value.c += value.b;
  value.d = value.a + value.b + value.c;
  asm volatile("" : "+l"(value.a), "+l"(value.b), "+l"(value.c), "+l"(value.d));
  return value;
}
__global__ void nested(Payload *out) {
  const unsigned lane = threadIdx.x;
  Payload value{lane + 1ull, 0x100000000ull + lane, 11, 19};
  for (unsigned n = 1; n <= 3; ++n) value = mutate(value, n);
  out[lane] = value;
}
TEST(PtxCallOwnership, ParameterCopiesAndRepeatedReturns) {
  Payload *out;
  ASSERT_EQ(cudaMalloc(&out, 32 * sizeof(Payload)), cudaSuccess);
  nested<<<1, 32>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  Payload values[32];
  ASSERT_EQ(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned lane = 0; lane < 32; ++lane) {
    Payload v{lane + 1ull, 0x100000000ull + lane, 11, 19};
    for (unsigned n = 1; n <= 3; ++n) {
      v.a += n;
      v.b ^= v.a;
      v.c += v.b;
      v.d = v.a + v.b + v.c;
    }
    EXPECT_EQ(values[lane].a, v.a);
    EXPECT_EQ(values[lane].b, v.b);
    EXPECT_EQ(values[lane].c, v.c);
    EXPECT_EQ(values[lane].d, v.d);
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
