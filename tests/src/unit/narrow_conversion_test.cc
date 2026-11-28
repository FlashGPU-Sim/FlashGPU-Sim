#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "gpgpu-sim/flash/narrow_conversion.h"
using namespace flash_gpgpu_sim;
namespace {
float from_bits(unsigned bits) {
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}
void check_pack(float x) {
  EXPECT_EQ(f32_to_e4m3_satfinite_rn(x),
            __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
  EXPECT_EQ(f32_to_e2m1_satfinite_rn(x),
            __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest));
  EXPECT_EQ(f32_to_ue8m0_satfinite_rp(x),
            __nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundPosInf));
}
TEST(NarrowConversion, AllBfloatEncodingsAndRandomFloatBits) {
  for (unsigned bits = 0; bits < 65536; ++bits) {
    SCOPED_TRACE(bits);
    check_pack(from_bits(bits << 16));
  }
  unsigned bits = 1234567;
  for (unsigned i = 0; i < 65536; ++i) {
    bits = 1664525u * bits + 1013904223u;
    SCOPED_TRACE(bits);
    check_pack(from_bits(bits));
  }
}
TEST(NarrowConversion, AllFp8AndScaleEncodingsToBfloat) {
  for (unsigned code = 0; code < 256; ++code) {
    const float fp8 =
        __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const unsigned expected =
        (code & 127) == 127 ? 0x7fff
                            : __bfloat16_as_ushort(__float2bfloat16_rn(fp8));
    EXPECT_EQ(e4m3_scaled_to_bf16(code), expected) << code;
    EXPECT_EQ(e4m3_to_f16(code), (code & 127) == 127
                                     ? 0x7fff
                                     : __half_as_ushort(__float2half_rn(fp8)))
        << code;
    const unsigned scale = code == 255
                               ? 0x7fff
                               : __bfloat16_as_ushort(__float2bfloat16_rn(
                                     std::ldexp(1.f, int(code) - 127)));
    EXPECT_EQ(ue8m0_to_bf16(code), scale) << code;
  }
}
TEST(NarrowConversion, MidpointsAndNeighboringFloatEncodings) {
  for (unsigned code = 0; code < 126; ++code) {
    const float a =
        __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float b =
        __half2float(__half(__nv_cvt_fp8_to_halfraw(code + 1, __NV_E4M3)));
    for (float sign : {-1.f, 1.f}) {
      const float midpoint = sign * (a + b) * .5f;
      check_pack(std::nextafter(midpoint, -INFINITY));
      check_pack(midpoint);
      check_pack(std::nextafter(midpoint, INFINITY));
    }
  }
  for (float midpoint : {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f})
    for (float sign : {-1.f, 1.f}) {
      const float x = sign * midpoint;
      check_pack(std::nextafter(x, -INFINITY));
      check_pack(x);
      check_pack(std::nextafter(x, INFINITY));
    }
}
}  // namespace

TEST(NarrowConversion, AllFp8CodesAndScaleFactors) {
  for (unsigned code = 0; code < 256; ++code)
    for (unsigned scale = 0; scale < 256; ++scale) {
      const float f =
          __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
      const unsigned expected = (code & 127) == 127 || scale == 255
                                    ? 0x7fff
                                    : __bfloat16_as_ushort(__float2bfloat16_rn(
                                          std::ldexp(f, int(scale) - 127)));
      ASSERT_EQ(e4m3_scaled_to_bf16(code, scale), expected)
          << code << '/' << scale;
    }
}
