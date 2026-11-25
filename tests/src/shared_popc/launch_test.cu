#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ void increments(unsigned *out) {
  __shared__ unsigned counters[8];
  const unsigned tid = threadIdx.x;
  if (tid < 8) counters[tid] = 0xfffffff0u + tid;
  __syncthreads();
  const unsigned address = __cvta_generic_to_shared(counters + tid % 4);
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    if (tid % 3 != 0) {
      asm volatile("{ .reg .u32 ignored; atom.shared.add.u32 ignored, [%0], 1; }"
                   :: "r"(address) : "memory");
    }
  }
  __syncthreads();
  if (tid < 8) out[blockIdx.x * 8 + tid] = counters[tid];
}
TEST(SharedPopc, PredicatedMultiwarpAddressesAndCtaIsolation) {
  unsigned *out;
  std::array<unsigned, 32> actual{};
  ASSERT_EQ(cudaMalloc(&out, sizeof(actual)), cudaSuccess);
  increments<<<4, 128>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(), out, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned block = 0; block < 4; ++block) {
    for (unsigned index = 0; index < 8; ++index) {
      unsigned expected = 0xfffffff0u + index;
      for (unsigned tid = 0; tid < 128; ++tid)
        if (tid % 3 && tid % 4 == index) expected += 3;
      EXPECT_EQ(actual[block * 8 + index], expected) << block << ':' << index;
    }
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
