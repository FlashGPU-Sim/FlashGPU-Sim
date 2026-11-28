#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
template <bool Wide>
__global__ void scan(const uint64_t *input, unsigned *output) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned result;
  if constexpr (Wide)
    asm volatile("bfind.u64 %0,%1;" : "=r"(result) : "l"(input[i]));
  else
    asm volatile("bfind.u32 %0,%1;" : "=r"(result) : "r"(unsigned(input[i])));
  output[i] = result;
}

template <bool Wide>
void check() {
  // Powers, complements, zero and word boundaries without device-side
  // constant folding. Reference uses repeated division, not the simulator's CLZ.
  std::vector<uint64_t> input(256);
  for (unsigned i = 0; i < input.size(); ++i) {
    const auto bit = uint64_t(1) << (i % 64);
    input[i] = i < 64 ? bit : i < 128 ? ~bit : i < 192 ? bit - 1 : 0;
  }
  uint64_t *device_input;
  unsigned *device_output;
  ASSERT_EQ(cudaMalloc(&device_input, input.size() * sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, input.size() * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), input.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice), cudaSuccess);
  scan<Wide><<<2, 128>>>(device_input, device_output);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> output(input.size());
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, output.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i) {
    uint64_t value = Wide ? input[i] : uint32_t(input[i]);
    unsigned expected = UINT32_MAX;
    while (value) { ++expected; value /= 2; }
    EXPECT_EQ(output[i], expected) << "input=" << input[i];
  }
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}
TEST(LeadingBitBase, U32Boundaries) { check<false>(); }
TEST(LeadingBitBase, U64Boundaries) { check<true>(); }
}  // namespace
