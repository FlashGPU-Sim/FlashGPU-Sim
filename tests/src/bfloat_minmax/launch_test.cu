#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
namespace {
struct Pair {
  unsigned a, b;
};
__global__ void compare(const Pair *in, unsigned *out) {
  const unsigned i = threadIdx.x;
  unsigned a = in[i].a, b = in[i].b, result;
  unsigned short scalar;
  result = a;
  asm volatile("min.bf16x2 %0,%0,%1;" : "+r"(result) : "r"(b));
  out[i * 4] = result;
  result = b;
  asm volatile("max.bf16x2 %0,%1,%0;" : "+r"(result) : "r"(a));
  out[i * 4 + 1] = result;
  asm volatile("min.bf16 %0,%1,%2;"
               : "=h"(scalar)
               : "h"((unsigned short)a), "h"((unsigned short)b));
  out[i * 4 + 2] = scalar;
  asm volatile("max.bf16 %0,%1,%2;"
               : "=h"(scalar)
               : "h"((unsigned short)b), "h"((unsigned short)a));
  out[i * 4 + 3] = scalar;
}
float decode(uint16_t x) {
  unsigned u = unsigned(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
void check(const std::vector<Pair> &input) {
  Pair *in;
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&in, input.size() * sizeof(Pair)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, input.size() * 16), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), input.size() * sizeof(Pair),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  compare<<<1, input.size()>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> actual(input.size() * 4);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i)
    for (unsigned op = 0; op < 4; ++op) {
      for (unsigned h = 0; h < (op < 2 ? 2u : 1u); ++h) {
        float a = decode(input[i].a >> (16 * h)),
              b = decode(input[i].b >> (16 * h));
        float result = decode(actual[i * 4 + op] >> (16 * h));
        bool minimum = op % 2 == 0;
        if (std::isnan(a) && std::isnan(b))
          EXPECT_TRUE(std::isnan(result));
        else {
          float expected = std::isnan(a)   ? b
                           : std::isnan(b) ? a
                           : minimum       ? std::min(a, b)
                                           : std::max(a, b);
          if (a == 0 && b == 0)
            expected = std::copysign(
                0.0f,
                minimum ? (std::signbit(a) || std::signbit(b) ? -1.0f : 1.0f)
                        : (std::signbit(a) && std::signbit(b) ? -1.0f : 1.0f));
          EXPECT_EQ(result, expected) << i << ":" << op << ":" << h;
          if (expected == 0)
            EXPECT_EQ(std::signbit(result), std::signbit(expected));
        }
      }
      if (op >= 2) EXPECT_EQ(actual[i * 4 + op] >> 16, 0u);
    }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(BfloatMinMax, FiniteAndIndependentPackedHalves) {
  std::vector<Pair> v(128);
  for (unsigned i = 0; i < v.size(); ++i)
    v[i] = {unsigned(i * 17039 + 0x80118011), unsigned(0x7e500007 - i * 32347)};
  check(v);
}
TEST(BfloatMinMax, BothOrdersOfSignedZeroSubnormalInfinityAndNaN) {
  constexpr std::array<uint16_t, 16> p{
      0,      0x8000, 1,      0x8001, 0x7f,   0x807f, 0x0080, 0x8080,
      0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffc0};
  std::vector<Pair> v;
  for (unsigned i = 0; i < p.size(); ++i)
    for (unsigned j = 0; j < p.size(); ++j)
      v.push_back(
          {p[i] | (unsigned(p[j]) << 16), p[j] | (unsigned(p[i]) << 16)});
  check(v);
}
}  // namespace
