#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
namespace {
// Independent bit-pattern oracle: ties, signed zero, subnormals, overflow,
// infinities and NaNs. The simulator conversion helper is not included.
constexpr std::array<uint32_t, 16> bits = {
    0,          0x80000000, 0x3f800000, 0xbf800000, 0x3f808000, 0x3f818000,
    0xbf808000, 0xbf818000, 0x00008000, 0x00018000, 0x80018000, 0x7f7fffff,
    0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345};
constexpr std::array<uint16_t, 16> rn = {
    0, 0x8000, 0x3f80, 0xbf80, 0x3f80, 0x3f82, 0xbf80, 0xbf82,
    0, 2,      0x8002, 0x7f80, 0xff80, 0x7f80, 0xff80, 0x7fff};
constexpr std::array<uint16_t, 16> rz = {
    0, 0x8000, 0x3f80, 0xbf80, 0x3f80, 0x3f81, 0xbf80, 0xbf81,
    0, 1,      0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fff};
template <int Mode>
__global__ void bfloat_convert(const unsigned *in, unsigned *out) {
  const unsigned t = threadIdx.x;
  float a = __uint_as_float(in[t % 16]), b = __uint_as_float(in[(31 - t) % 16]);
  unsigned packed;
  unsigned short scalar;
#define CONVERT(MODS)                              \
  asm volatile("cvt." MODS ".bf16x2.f32 %0,%1,%2;" \
               : "=r"(packed)                      \
               : "f"(a), "f"(b));                  \
  asm volatile("cvt." MODS ".bf16.f32 %0,%1;" : "=h"(scalar) : "f"(a))
  if constexpr (Mode == 0) {
    CONVERT("rn");
  }
  if constexpr (Mode == 1) {
    CONVERT("rz");
  }
  if constexpr (Mode == 2) {
    CONVERT("rn.relu");
  }
  if constexpr (Mode == 3) {
    CONVERT("rz.relu");
  }
  if constexpr (Mode == 4) {
    CONVERT("rn.satfinite");
  }
  if constexpr (Mode == 5) {
    CONVERT("rz.satfinite");
  }
  if constexpr (Mode == 6) {
    CONVERT("rn.relu.satfinite");
  }
  if constexpr (Mode == 7) {
    CONVERT("rz.relu.satfinite");
  }
#undef CONVERT
  float up;
  asm volatile("cvt.f32.bf16 %0,%1;" : "=f"(up) : "h"(scalar));
  out[t * 3] = packed;
  out[t * 3 + 1] = scalar;
  out[t * 3 + 2] = __float_as_uint(up);
}
template <int Mode>
void check() {
  unsigned *in, *out;
  std::array<unsigned, 96> result{};
  ASSERT_EQ(cudaMalloc(&in, 64), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 384), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, bits.data(), 64, cudaMemcpyHostToDevice),
            cudaSuccess);
  bfloat_convert<Mode><<<1, 32>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), out, 384, cudaMemcpyDeviceToHost),
            cudaSuccess);
  auto expected = [](unsigned i) -> unsigned {
    unsigned value = (Mode & 1 ? rz : rn)[i];
    if ((Mode & 2) && (bits[i] & 0x80000000u) && (bits[i] & 0x7fffffffu))
      value = 0;
    if ((Mode & 4) && (value & 0x7fff) == 0x7f80)
      value = (value & 0x8000) | 0x7f7f;
    return value;
  };
  for (unsigned t = 0; t < 32; ++t) {
    const auto a = expected(t % 16), b = expected((31 - t) % 16);
    EXPECT_EQ(result[t * 3], (a << 16) | b) << Mode << ":" << t;
    EXPECT_EQ(result[t * 3 + 1], a) << Mode << ":" << t;
    EXPECT_EQ(result[t * 3 + 2], a << 16) << Mode << ":" << t;
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(BfloatConversion, Rounding) {
  check<0>();
  check<1>();
}
TEST(BfloatConversion, Relu) {
  check<2>();
  check<3>();
}
TEST(BfloatConversion, Satfinite) {
  check<4>();
  check<5>();
}
TEST(BfloatConversion, ReluSatfinite) {
  check<6>();
  check<7>();
}
}  // namespace
