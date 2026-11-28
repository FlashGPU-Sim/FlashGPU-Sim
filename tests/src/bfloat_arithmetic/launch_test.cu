#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
struct Input {
  unsigned a, b, c;
};
constexpr unsigned stride = 8;
// Host oracle rounds the mathematical long-double value directly to BF16.
// It does not include or call simulator arithmetic/conversion helpers.
long double decode(uint16_t bits) {
  uint32_t wide = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}
uint16_t encode(long double value) {
  if (std::isnan(value)) return 0x7fff;
  const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  value = std::fabs(value);
  if (std::isinf(value)) return sign | 0x7f80;
  if (value == 0) return sign;
  int exponent;
  std::frexp(value, &exponent);
  const int quantum = std::max(exponent - 8, -133);
  const long double units = std::ldexp(value, -quantum);
  unsigned rounded = std::floor(units);
  const long double fraction = units - rounded;
  if (fraction > 0.5L || (fraction == 0.5L && (rounded & 1))) ++rounded;
  if (!rounded) return sign;
  const long double exact = std::ldexp((long double)rounded, quantum);
  if (exact >= std::ldexp(1.0L, 128)) return sign | 0x7f80;
  const float result = exact;
  uint32_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  return sign | (bits >> 16);
}
unsigned reference(const Input &in, unsigned op, bool packed) {
  unsigned result = 0;
  for (unsigned half = 0; half < (packed ? 2u : 1u); ++half) {
    const auto a = decode(in.a >> (16 * half)), b = decode(in.b >> (16 * half)),
               c = decode(in.c >> (16 * half));
    const auto value = op == 0   ? a + b
                       : op == 1 ? a - b
                       : op == 2 ? a * b
                                 : std::fma(a, b, c);
    result |= unsigned(encode(value)) << (16 * half);
  }
  return result;
}
__global__ void arithmetic(const Input *in, unsigned *out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned a = in[i].a, b = in[i].b, c = in[i].c;
  unsigned packed;
  unsigned short scalar;
#define BINARY(OP, J)                                                      \
  asm volatile(OP ".rn.bf16x2 %0,%1,%2;" : "=r"(packed) : "r"(a), "r"(b)); \
  asm volatile(OP ".rn.bf16 %0,%1,%2;"                                     \
               : "=h"(scalar)                                              \
               : "h"((unsigned short)a), "h"((unsigned short)b));          \
  out[i * stride + J] = packed;                                            \
  out[i * stride + 4 + J] = scalar;
  BINARY("add", 0)
  BINARY("sub", 1)
  BINARY("mul", 2)
#undef BINARY
  packed = a;
  asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(packed) : "r"(b), "r"(c));
  scalar = a;
  asm volatile("fma.rn.bf16 %0,%0,%1,%2;"
               : "+h"(scalar)
               : "h"((unsigned short)b), "h"((unsigned short)c));
  out[i * stride + 3] = packed;
  out[i * stride + 7] = scalar;
}
void check(const std::vector<Input> &in) {
  ASSERT_EQ(in.size() % 128, 0u);
  Input *input;
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&input, in.size() * sizeof(Input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, in.size() * stride * 4), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(input, in.data(), in.size() * sizeof(Input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  arithmetic<<<in.size() / 128, 128>>>(input, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> actual(in.size() * stride);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned i = 0; i < in.size(); ++i)
    for (unsigned j = 0; j < stride; ++j)
      EXPECT_EQ(actual[i * stride + j], reference(in[i], j % 4, j < 4))
          << i << ":" << j;
  EXPECT_EQ(cudaFree(input), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(BfloatArithmeticLaunch, FinitePackedAndScalar) {
  std::vector<Input> in;
  uint32_t seed = 0x12345678;
  auto next = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 16) & 0x807fu) | (((seed % 41) + 107) << 7);
  };
  for (unsigned i = 0; i < 1024; ++i) {
    unsigned values[3];
    for (auto &v : values) {
      v = next();
      v |= next() << 16;
    }
    in.push_back({values[0], values[1], values[2]});
  }
  check(in);
}
TEST(BfloatArithmeticLaunch, SpecialValuesAndSubnormalBoundaries) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x7f,   0x80,
                             0x8080, 0x3f80, 0xbf80, 0x3f81, 0x7f7f, 0xff7f,
                             0x7f80, 0xff80, 0x7fc1, 0xff81};
  std::vector<Input> in;
  for (unsigned a = 0; a < 16; ++a)
    for (unsigned b = 0; b < 16; ++b)
      in.push_back({unsigned(values[a]) | (unsigned(values[15 - a]) << 16),
                    unsigned(values[b]) | (unsigned(values[15 - b]) << 16),
                    unsigned(values[(a + b) % 16]) |
                        (unsigned(values[(a + 2 * b) % 16]) << 16)});
  check(in);
}
__global__ void literal_midpoint(const unsigned *in, unsigned *out) {
  const unsigned i = threadIdx.x;
  unsigned a = in[i], c = (i & 1) ? 0x00018001u : 0x80010001u;
  unsigned result;
  asm volatile(
      "{ .reg .b32 scale; mov.b32 scale,0x3fc03fc0; "
      "fma.rn.bf16x2 %0,%1,scale,%2; }"
      : "=r"(result)
      : "r"(a), "r"(c));
  out[i * 2] = result;
  asm volatile("mul.bf16x2 %0,%1,%1;" : "=r"(result) : "r"(a));
  out[i * 2 + 1] = result;
}
TEST(BfloatArithmeticLaunch, LiteralFmaMidpointAndDefaultRounding) {
  unsigned *in, *out;
  ASSERT_EQ(cudaMalloc(&in, 128 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 256 * sizeof(unsigned)), cudaSuccess);
  std::vector<unsigned> input(128, 0x3f813f81u), actual(256);
  ASSERT_EQ(cudaMemcpy(in, input.data(), 512, cudaMemcpyHostToDevice),
            cudaSuccess);
  literal_midpoint<<<1, 128>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(), out, 1024, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 128; ++i) {
    EXPECT_EQ(actual[2 * i], (i & 1) ? 0x3fc23fc1u : 0x3fc13fc2u);
    EXPECT_EQ(actual[2 * i + 1], 0x3f823f82u);
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
