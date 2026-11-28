#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
template <bool Wide, bool Signed>
__global__ void scan(const uint64_t *input, unsigned *output) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned result;
  if constexpr (Wide && Signed)
    asm volatile("bfind.s64 %0,%1;" : "=r"(result) : "l"(input[i]));
  else if constexpr (Wide)
    asm volatile("bfind.u64 %0,%1;" : "=r"(result) : "l"(input[i]));
  else if constexpr (Signed)
    asm volatile("bfind.s32 %0,%1;" : "=r"(result) : "r"(unsigned(input[i])));
  else
    asm volatile("bfind.u32 %0,%1;" : "=r"(result) : "r"(unsigned(input[i])));
  output[i] = result;
}

template <bool Wide, bool Signed>
void check(bool include_zero) {
  constexpr unsigned bits = Wide ? 64 : 32;
  constexpr uint64_t mask = Wide ? UINT64_MAX : UINT32_MAX;
  std::vector<uint64_t> input(256);
  for (unsigned i = 0; i < input.size(); ++i) {
    const uint64_t bit = uint64_t(1) << (i % bits);
    // Every bit, its complement, carry boundaries, zero and all-sign values.
    input[i] = (i < 64 ? bit : i < 128 ? ~bit : i < 192 ? bit - 1
                                                             : UINT64_MAX) & mask;
    if (!include_zero) input[i] |= 1;
  }
  uint64_t *device_input;
  unsigned *device_output;
  ASSERT_EQ(cudaMalloc(&device_input, input.size() * sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, input.size() * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), input.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice), cudaSuccess);
  scan<Wide, Signed><<<2, 128>>>(device_input, device_output);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> output(input.size());
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, output.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i) {
    uint64_t value = input[i];
    if (Signed && value / (uint64_t(1) << (bits - 1))) value = mask - value;
    // Repeated division is independent of the implementation's CLZ primitive.
    unsigned expected = UINT32_MAX;
    while (value) { ++expected; value /= 2; }
    EXPECT_EQ(output[i], expected) << "input=" << input[i];
  }
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}
TEST(LeadingBit, U64Nonzero) { check<true, false>(false); }
TEST(LeadingBit, U32Boundaries) { check<false, false>(true); }
TEST(LeadingBit, U64Boundaries) { check<true, false>(true); }
TEST(LeadingBit, S32Boundaries) { check<false, true>(true); }
TEST(LeadingBit, S64Boundaries) { check<true, true>(true); }
}  // namespace
