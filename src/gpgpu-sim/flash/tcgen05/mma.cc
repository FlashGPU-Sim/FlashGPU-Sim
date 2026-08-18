#include "mma.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace flash_gpgpu_sim {

namespace {

float input_to_f32(uint16_t value, uint8_t type) {
  switch (type) {
  case TCGEN05_MMA_TYPE_FIELD_F16:
    return tcgen05_f16_to_f32(value);
  case TCGEN05_MMA_TYPE_FIELD_ONE:
    return tcgen05_bf16_to_f32(value);
  default:
    assert(false && "Unsupported TCGen05 f16 MMA input data type");
    return 0.0f;
  }
}

} // namespace

float tcgen05_f16_to_f32(uint16_t f16) {
  uint32_t sign = (f16 >> 15) & 0x1;
  uint32_t exp = (f16 >> 10) & 0x1f;
  uint32_t frac = f16 & 0x3ff;

  if (exp == 0) {
    if (frac == 0)
      return sign ? -0.0f : 0.0f;
    float result = frac / 1024.0f / 16384.0f;
    return sign ? -result : result;
  }
  if (exp == 31)
    return frac ? NAN : (sign ? -INFINITY : INFINITY);

  uint32_t f32_exp = exp - 15 + 127;
  uint32_t f32_frac = frac << 13;
  uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;
  return tcgen05_bits_to_f32(f32_bits);
}

float tcgen05_bf16_to_f32(uint16_t bf16) {
  return tcgen05_bits_to_f32(static_cast<uint32_t>(bf16) << 16);
}

float tcgen05_e2m1_to_f32(uint8_t e2m1) {
  static const float kPositiveValues[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                           2.0f, 3.0f, 4.0f, 6.0f};
  e2m1 &= 0xf;
  float value = kPositiveValues[e2m1 & 0x7];
  return (e2m1 & 0x8) ? -value : value;
}

float tcgen05_ue8m0_to_f32(uint8_t ue8m0) {
  if (ue8m0 == 0xff)
    return NAN;
  return std::ldexp(1.0f, static_cast<int>(ue8m0) - 127);
}

uint16_t tcgen05_f32_to_f16(float f32) {
  uint32_t f32_bits = tcgen05_f32_to_bits(f32);
  uint32_t sign = (f32_bits >> 31) & 0x1;
  int32_t exp = ((f32_bits >> 23) & 0xff) - 127 + 15;
  uint32_t frac = (f32_bits >> 13) & 0x3ff;

  if (exp <= 0)
    return sign << 15;
  if (exp >= 31)
    return (sign << 15) | 0x7c00;

  return static_cast<uint16_t>((sign << 15) | (exp << 10) | frac);
}

uint32_t tcgen05_f32_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float tcgen05_bits_to_f32(uint32_t value) {
  float bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<uint32_t> tcgen05_mma_f16_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint16_t> &a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input_d,
    bool enable_input_d) {
  assert(desc.d_type == TCGEN05_MMA_TYPE_FIELD_ONE &&
         "Only f32 TCGen05 MMA outputs are supported");
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MMA A matrix size does not match descriptor");
  assert(b.size() == desc.k * desc.n &&
         "TCGen05 MMA B matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MMA D matrix size does not match descriptor");

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum = enable_input_d
                      ? tcgen05_bits_to_f32(input_d[row * desc.n + col])
                      : 0.0f;
      for (uint32_t k = 0; k < desc.k; ++k) {
        uint32_t a_index =
            desc.transpose_a ? (k * desc.m + row) : (row * desc.k + k);
        uint32_t b_index =
            desc.transpose_b ? (col * desc.k + k) : (k * desc.n + col);
        float a_value = input_to_f32(a[a_index], desc.a_type);
        float b_value = input_to_f32(b[b_index], desc.b_type);
        if (desc.negate_a)
          a_value = -a_value;
        if (desc.negate_b)
          b_value = -b_value;
        sum += a_value * b_value;
      }
      output[row * desc.n + col] = tcgen05_f32_to_bits(sum);
    }
  }
  return output;
}

std::vector<uint32_t> tcgen05_mma_mxf4_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint8_t> &a,
    const std::vector<uint8_t> &b, const std::vector<uint8_t> &scale_a,
    const std::vector<uint8_t> &scale_b, uint32_t scale_vector_size,
    const std::vector<uint32_t> &input_d, bool enable_input_d) {
  assert(desc.a_type == TCGEN05_MXF4_FORMAT_E2M1 &&
         desc.b_type == TCGEN05_MXF4_FORMAT_E2M1 &&
         "Only E2M1 TCGen05 MXFP4 inputs are supported");
  assert(desc.scale_format == TCGEN05_SCALE_FORMAT_UE8M0 &&
         "Only UE8M0 TCGen05 MXFP4 scale factors are supported");
  assert(scale_vector_size == 32 &&
         "TCGen05 MXFP4 block32 requires 32-element scale vectors");
  assert(desc.k % scale_vector_size == 0 &&
         "TCGen05 MXFP4 K must be divisible by the scale vector size");
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MXFP4 A matrix size does not match descriptor");
  assert(b.size() == desc.n * desc.k &&
         "TCGen05 MXFP4 B matrix size does not match descriptor");
  uint32_t scales_per_row = desc.k / scale_vector_size;
  assert(scale_a.size() == desc.m * scales_per_row &&
         "TCGen05 MXFP4 A scale matrix size does not match descriptor");
  assert(scale_b.size() == desc.n * scales_per_row &&
         "TCGen05 MXFP4 B scale matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MXFP4 D matrix size does not match descriptor");

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum = enable_input_d
                      ? tcgen05_bits_to_f32(input_d[row * desc.n + col])
                      : 0.0f;
      for (uint32_t k = 0; k < desc.k; ++k) {
        uint32_t scale_k = k / scale_vector_size;
        float a_value = tcgen05_e2m1_to_f32(a[row * desc.k + k]);
        float b_value = tcgen05_e2m1_to_f32(b[col * desc.k + k]);
        a_value *=
            tcgen05_ue8m0_to_f32(scale_a[row * scales_per_row + scale_k]);
        b_value *=
            tcgen05_ue8m0_to_f32(scale_b[col * scales_per_row + scale_k]);
        sum += a_value * b_value;
      }
      output[row * desc.n + col] = tcgen05_f32_to_bits(sum);
    }
  }
  return output;
}

} // namespace flash_gpgpu_sim
