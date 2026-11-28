#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
__global__ void unpack_bytes(unsigned *out) {
  unsigned code = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned short packed = code | ((255 - code) << 8);
  unsigned a, b, c;
  asm volatile("cvt.rn.bf16x2.ue8m0x2 %0,%1;" : "=r"(a) : "h"(packed));
  asm volatile("cvt.rn.bf16x2.e4m3x2 %0,%1;" : "=r"(b) : "h"(packed));
  asm volatile("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(c) : "h"(packed));
  out[3 * code] = a;
  out[3 * code + 1] = b;
  out[3 * code + 2] = c;
}
TEST(NarrowLaunch, UnpackAllFp8AndScaleCodes) {
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&out, 768 * 4), cudaSuccess);
  unpack_bytes<<<8, 32>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> actual(768);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned code = 0; code < 256; ++code)
    for (unsigned lane = 0; lane < 2; ++lane) {
      unsigned c = lane ? 255 - code : code;
      const unsigned ue = c == 255 ? 0x7fff
                                   : __bfloat16_as_ushort(__float2bfloat16_rn(
                                         std::ldexp(1.f, int(c) - 127)));
      const float f =
          __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
      const unsigned fp = (c & 127) == 127
                              ? 0x7fff
                              : __bfloat16_as_ushort(__float2bfloat16_rn(f));
      ASSERT_EQ((actual[3 * code] >> (16 * lane)) & 65535, ue)
          << code << '/' << lane;
      ASSERT_EQ((actual[3 * code + 1] >> (16 * lane)) & 65535, fp)
          << code << '/' << lane;
      const unsigned hf =
          (c & 127) == 127 ? 0x7fff : __half_as_ushort(__float2half_rn(f));
      ASSERT_EQ((actual[3 * code + 2] >> (16 * lane)) & 65535, hf)
          << code << '/' << lane;
    }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}

template <int Format>
__global__ void pack_float(const float *input, unsigned short *out,
                           unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  unsigned short result;
  if constexpr (Format == 0)
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;"
                 : "=h"(result)
                 : "f"(input[count - 1 - i]), "f"(input[i]));
  if constexpr (Format == 1)
    asm volatile(
        "{.reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v,%1,%2;mov.b16 %0,{v,0};}"
        : "=h"(result)
        : "f"(input[count - 1 - i]), "f"(input[i]));
  if constexpr (Format == 2)
    asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0,%1,%2;"
                 : "=h"(result)
                 : "f"(input[count - 1 - i]), "f"(input[i]));
  out[i] = result;
}
unsigned convert(float x, int format) {
  return format == 0 ? __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)
         : format == 1
             ? __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest)
             : __nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundPosInf);
}
std::vector<float> boundary_values() {
  std::vector<float> values;
  for (unsigned c = 0; c < 126; ++c) {
    const float a = __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
    const float b =
        __half2float(__half(__nv_cvt_fp8_to_halfraw(c + 1, __NV_E4M3)));
    for (float sign : {-1.f, 1.f}) {
      const float midpoint = sign * (a + b) * .5f;
      values.push_back(std::nextafter(midpoint, -INFINITY));
      values.push_back(midpoint);
      values.push_back(std::nextafter(midpoint, INFINITY));
    }
  }
  for (unsigned bits :
       {0u, 0x80000000u, 1u, 0x80000001u, 0x7fffffffu, 0xffffffffu, 0x7f800000u,
        0xff800000u, 0x7f7fffffu, 0x00400000u, 0x00400001u}) {
    float x;
    std::memcpy(&x, &bits, 4);
    values.push_back(x);
  }
  return values;
}
template <int Format>
void check_float() {
  const auto input = boundary_values();
  float *in;
  unsigned short *out;
  ASSERT_EQ(cudaMalloc(&in, input.size() * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, input.size() * 2), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(in, input.data(), input.size() * 4, cudaMemcpyHostToDevice),
      cudaSuccess);
  pack_float<Format><<<(input.size() + 31) / 32, 32>>>(in, out, input.size());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned short> actual(input.size());
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 2, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned i = 0; i < input.size(); ++i)
    ASSERT_EQ(actual[i], convert(input[i], Format) |
                             (convert(input[input.size() - 1 - i], Format)
                              << (Format == 1 ? 4 : 8)))
        << i;
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(NarrowLaunch, F32ToFp8MidpointsAndSpecials) { check_float<0>(); }
TEST(NarrowLaunch, F32ToFp4MidpointsAndSpecials) { check_float<1>(); }
TEST(NarrowLaunch, F32ToScaleMidpointsAndSpecials) { check_float<2>(); }

template <bool Fp4>
__global__ void pack_bfloat(unsigned short *out) {
  const unsigned code = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned pair = code | ((65535 - code) << 16);
  unsigned short result;
  if constexpr (Fp4)
    asm volatile(
        "{.reg .b8 v; cvt.rn.satfinite.e2m1x2.bf16x2 v,%1; mov.b16 %0,{v,0};}"
        : "=h"(result)
        : "r"(pair));
  else
    asm volatile("cvt.rn.satfinite.e4m3x2.bf16x2 %0,%1;"
                 : "=h"(result)
                 : "r"(pair));
  out[code] = result;
}
template <bool Fp4>
void check_bfloat() {
  unsigned short *out;
  ASSERT_EQ(cudaMalloc(&out, 65536 * 2), cudaSuccess);
  pack_bfloat<Fp4><<<256, 256>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned short> actual(65536);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 2, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned code = 0; code < 65536; ++code) {
    const float lo = __bfloat162float(__ushort_as_bfloat16(code));
    const float hi = __bfloat162float(__ushort_as_bfloat16(65535 - code));
    ASSERT_EQ(actual[code], convert(lo, Fp4 ? 1 : 0) |
                                (convert(hi, Fp4 ? 1 : 0) << (Fp4 ? 4 : 8)))
        << code;
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(NarrowLaunch, AllBfloatEncodingsToFp8) { check_bfloat<false>(); }
TEST(NarrowLaunch, AllBfloatEncodingsToFp4) { check_bfloat<true>(); }
}  // namespace
