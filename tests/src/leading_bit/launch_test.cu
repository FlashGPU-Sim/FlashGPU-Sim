#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
template <unsigned Form>
__global__ void scan(const uint64_t *input, unsigned *output) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned index, shift;
#define SCAN(TYPE, CONSTRAINT, VALUE) \
  asm volatile("bfind." TYPE " %0,%2; bfind.shiftamt." TYPE " %1,%2;" \
               : "=&r"(index), "=r"(shift) : CONSTRAINT(VALUE))
  if constexpr (Form == 0) { SCAN("u32", "r", uint32_t(input[i])); }
  if constexpr (Form == 1) { SCAN("s32", "r", uint32_t(input[i])); }
  if constexpr (Form == 2) { SCAN("u64", "l", input[i]); }
  if constexpr (Form == 3) { SCAN("s64", "l", input[i]); }
#undef SCAN
  output[2*i] = index;
  output[2*i+1] = shift;
}

template <unsigned Form>
void check() {
  constexpr unsigned bits = Form < 2 ? 32 : 64;
  constexpr bool sign = Form & 1;
  std::vector<uint64_t> input(256);
  for (unsigned i = 0; i < input.size(); ++i) {
    const uint64_t bit = uint64_t(1) << (i % 64);
    input[i] = i < 64 ? bit : i < 128 ? ~bit : i < 192 ? bit - 1 : 0;
  }
  input[255] = UINT64_MAX;
  uint64_t *device_input;
  unsigned *device_output;
  ASSERT_EQ(cudaMalloc(&device_input, input.size() * sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, input.size() * 2 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), input.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice), cudaSuccess);
  scan<Form><<<2, 128>>>(device_input, device_output);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> output(input.size() * 2);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, output.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i) {
    uint64_t value = bits == 32 ? uint32_t(input[i]) : input[i];
    const bool negative = sign && ((value >> (bits-1)) & 1);
    // Locate the highest bit differing from the sign bit without the
    // simulator's inversion/CLZ implementation.
    unsigned expected = UINT32_MAX;
    for (unsigned bit = 0; bit < bits; ++bit)
      if (((value >> bit) & 1) != negative) expected = bit;
    EXPECT_EQ(output[2*i], expected) << "input=" << input[i];
    EXPECT_EQ(output[2*i+1], expected == UINT32_MAX ? expected : bits-1-expected)
        << "input=" << input[i];
  }
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}
TEST(LeadingBitLaunch, U32) { check<0>(); }
TEST(LeadingBitLaunch, S32) { check<1>(); }
TEST(LeadingBitLaunch, U64) { check<2>(); }
TEST(LeadingBitLaunch, S64) { check<3>(); }

__global__ void predicated(unsigned *out) {
  unsigned value = 0x80000000u, immediate;
  const unsigned lane = threadIdx.x;
  asm volatile("{ .reg .pred p; setp.ne.u32 p,%1,0; "
               "@p bfind.shiftamt.s32 %0,%0; }"
               : "+r"(value) : "r"(lane));
  asm volatile("bfind.shiftamt.u64 %0,0x100000000;" : "=r"(immediate));
  out[2*lane] = value;
  out[2*lane+1] = immediate;
}
TEST(LeadingBitLaunch, ImmediatePredicationAndAlias) {
  unsigned *device;
  ASSERT_EQ(cudaMalloc(&device, 64 * sizeof(unsigned)), cudaSuccess);
  predicated<<<1,32>>>(device);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned actual[64];
  ASSERT_EQ(cudaMemcpy(actual, device, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < 32; ++i) {
    EXPECT_EQ(actual[2*i], i ? 1u : 0x80000000u);
    EXPECT_EQ(actual[2*i+1], 31u);
  }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
} // namespace
