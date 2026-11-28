#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>
namespace {
constexpr unsigned outputs = 12;
__global__ void signs(const unsigned *input, unsigned *out) {
  const unsigned i = threadIdx.x;
  const unsigned a = input[i];
  unsigned packed;
  unsigned short scalar;
#define SIGN(OP, TYPE, MOD, J)                              \
  packed = a;                                               \
  asm volatile(OP MOD "." TYPE "x2 %0,%0;" : "+r"(packed)); \
  scalar = a;                                               \
  asm volatile(OP MOD "." TYPE " %0,%0;" : "+h"(scalar));   \
  out[i * outputs + J] = packed;                            \
  out[i * outputs + J + 1] = scalar;
  SIGN("abs", "bf16", "", 0)
  SIGN("neg", "bf16", "", 2)
  SIGN("abs", "f16", "", 4)
  SIGN("neg", "f16", "", 6)
  SIGN("abs", "f16", ".ftz", 8)
  SIGN("neg", "f16", ".ftz", 10)
#undef SIGN
}
void check(const std::vector<unsigned> &input) {
  unsigned *in, *out;
  ASSERT_EQ(cudaMalloc(&in, input.size() * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, input.size() * outputs * 4), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(in, input.data(), input.size() * 4, cudaMemcpyHostToDevice),
      cudaSuccess);
  signs<<<1, input.size()>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> result(input.size() * outputs);
  ASSERT_EQ(
      cudaMemcpy(result.data(), out, result.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i) {
    for (unsigned op = 0; op < outputs; ++op) {
      const bool bf16 = op < 4, negate = op % 4 >= 2, ftz = op >= 8,
                 packed = op % 2 == 0;
      const unsigned exponent = bf16 ? 0x7f80 : 0x7c00;
      const unsigned mantissa = bf16 ? 0x7f : 0x3ff;
      for (unsigned h = 0; h < (packed ? 2u : 1u); ++h) {
        const uint16_t bits = input[i] >> (16 * h),
                       actual = result[i * outputs + op] >> (16 * h);
        if ((bits & exponent) == exponent && (bits & mantissa)) {
          EXPECT_EQ(actual & exponent, exponent);
          EXPECT_NE(actual & mantissa,
                    0u);  // NaN payload/sign unspecified by ISA.
        } else {
          unsigned magnitude = bits & 0x7fff;
          if (ftz && !(bits & exponent)) magnitude = 0;
          const unsigned sign = negate ? ((bits & 0x8000) ? 0 : 0x8000) : 0;
          EXPECT_EQ(actual, magnitude | sign) << i << ":" << op << ":" << h;
        }
      }
      if (!packed) EXPECT_EQ(result[i * outputs + op] >> 16, 0u);
    }
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(HalfSign, FinitePatternsAndAliasedDestination) {
  std::vector<unsigned> input(128);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = uint16_t(i * 701 + 0x8011) |
               (uint32_t(uint16_t(i * 1009 + 0x155)) << 16);
  check(input);
}
TEST(HalfSign, ZerosSubnormalsInfinitiesAndNaNs) {
  const std::array<uint16_t, 16> patterns{
      0,     0x8000, 1,      0x8001, 0x7f,   0x807f, 0x3ff,  0x83ff,
      0x400, 0x8400, 0x7c00, 0xfc00, 0x7f80, 0xff80, 0x7fff, 0xffff};
  std::vector<unsigned> input(128);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = patterns[i % patterns.size()] |
               (uint32_t(patterns[(i / 8 + 3) % patterns.size()]) << 16);
  check(input);
}
}  // namespace
