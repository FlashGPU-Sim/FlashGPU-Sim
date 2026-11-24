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

float narrow_float_to_f32(uint8_t raw, uint32_t exponent_bits,
                          uint32_t mantissa_bits, bool ieee_nan_and_infinity,
                          bool canonical_nan_only) {
  const uint32_t total_bits = 1 + exponent_bits + mantissa_bits;
  const uint32_t value_mask = (1u << total_bits) - 1;
  const uint32_t mantissa_mask = (1u << mantissa_bits) - 1;
  const uint32_t exponent_mask = (1u << exponent_bits) - 1;
  const uint32_t value = raw & value_mask;
  const bool negative = ((value >> (total_bits - 1)) & 1) != 0;
  const uint32_t exponent = (value >> mantissa_bits) & exponent_mask;
  const uint32_t mantissa = value & mantissa_mask;

  if (ieee_nan_and_infinity && exponent == exponent_mask) {
    if (mantissa == 0)
      return negative ? -INFINITY : INFINITY;
    return NAN;
  }
  if (canonical_nan_only && exponent == exponent_mask &&
      mantissa == mantissa_mask)
    return NAN;

  const int bias = (1 << (exponent_bits - 1)) - 1;
  float magnitude = 0.0f;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa),
                           1 - bias - static_cast<int>(mantissa_bits));
  } else {
    const uint32_t significand = (1u << mantissa_bits) | mantissa;
    magnitude = std::ldexp(static_cast<float>(significand),
                           static_cast<int>(exponent) - bias -
                               static_cast<int>(mantissa_bits));
  }
  return negative ? -magnitude : magnitude;
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

float tcgen05_mxf8f6f4_to_f32(uint8_t value, uint8_t format) {
  switch (format) {
  case TCGEN05_MXF8F6F4_FORMAT_E4M3:
    return narrow_float_to_f32(value, 4, 3, false, true);
  case TCGEN05_MXF8F6F4_FORMAT_E5M2:
    return narrow_float_to_f32(value, 5, 2, true, false);
  case TCGEN05_MXF8F6F4_FORMAT_E2M3:
    return narrow_float_to_f32(value, 2, 3, false, false);
  case TCGEN05_MXF8F6F4_FORMAT_E3M2:
    return narrow_float_to_f32(value, 3, 2, false, false);
  case TCGEN05_MXF8F6F4_FORMAT_E2M1:
    return tcgen05_e2m1_to_f32(value);
  default:
    assert(false && "Unsupported TCGen05 MXF8F6F4 input data type");
    return 0.0f;
  }
}

float tcgen05_ue8m0_to_f32(uint8_t ue8m0) {
  if (ue8m0 == 0xff)
    return NAN;
  return std::ldexp(1.0f, static_cast<int>(ue8m0) - 127);
}

// Round to nearest, ties to even, including subnormals, NaNs and overflow.
uint16_t tcgen05_f32_to_f16(float f32) {
  const uint32_t bits = tcgen05_f32_to_bits(f32);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    if (mantissa == 0)
      return sign | 0x7c00u;
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
    payload |= 0x0200u;
    return sign | 0x7c00u | payload;
  }

  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31)
    return sign | 0x7c00u;
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    mantissa |= 0x800000u;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
      ++rounded;
    return sign | static_cast<uint16_t>(rounded);
  }

  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      ++half_exponent;
      if (half_exponent >= 31)
        return sign | 0x7c00u;
    }
  }
  return sign | static_cast<uint16_t>(half_exponent << 10) |
         static_cast<uint16_t>(rounded);
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
  assert((desc.d_type == TCGEN05_MMA_TYPE_FIELD_ONE ||
          desc.d_type == TCGEN05_MMA_TYPE_FIELD_F16) &&
         "Only f16/f32 TCGen05 MMA outputs are supported");
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MMA A matrix size does not match descriptor");
  assert(b.size() == desc.k * desc.n &&
         "TCGen05 MMA B matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MMA D matrix size does not match descriptor");

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum =
          enable_input_d
              ? (desc.d_type == TCGEN05_MMA_TYPE_FIELD_F16
                     ? tcgen05_f16_to_f32(uint16_t(input_d[row * desc.n + col]))
                     : tcgen05_bits_to_f32(input_d[row * desc.n + col]))
              : 0.0f;
      for (uint32_t k = 0; k < desc.k; ++k) {
        uint32_t a_index =
            desc.transpose_a ? (k * desc.m + row) : (row * desc.k + k);
        // A non-transposed B descriptor is K-major: N logical rows, each
        // containing K contiguous values.  Transpose B selects N-major, where
        // the N dimension is contiguous instead.
        uint32_t b_index =
            desc.transpose_b ? (k * desc.n + col) : (col * desc.k + k);
        float a_value = input_to_f32(a[a_index], desc.a_type);
        float b_value = input_to_f32(b[b_index], desc.b_type);
        if (desc.negate_a)
          a_value = -a_value;
        if (desc.negate_b)
          b_value = -b_value;
        sum += a_value * b_value;
      }
      // PTX leaves accumulation order/precision unspecified for F16 D.
      // Use FP32 internal accumulation and RN at each native K16 MMA.
      // F16 D occupies the low half of each 32-bit TMEM column.
      output[row * desc.n + col] = desc.d_type == TCGEN05_MMA_TYPE_FIELD_F16
                                       ? tcgen05_f32_to_f16(sum)
                                       : tcgen05_f32_to_bits(sum);
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
        if (desc.negate_a)
          a_value = -a_value;
        if (desc.negate_b)
          b_value = -b_value;
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

// Keep non-finite inputs and overflow on the original scalar evaluation path.
// NaN sign/payload selection is compiler-dependent; finite results and signed
// zeros retain their original bit patterns.
static std::vector<uint32_t>
mixed_mma_scalar(const tcgen05_mma_descriptor_t &desc,
                 const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                 const std::vector<uint8_t> &scale_a,
                 const std::vector<uint8_t> &scale_b,
                 const std::vector<uint32_t> &input_d, bool enable_input_d) {
  assert(tcgen05_mxf8f6f4_format_supported(desc.a_type) &&
         tcgen05_mxf8f6f4_format_supported(desc.b_type) &&
         "Unsupported TCGen05 MXF8F6F4 input format");
  assert(desc.scale_format == TCGEN05_SCALE_FORMAT_UE8M0 &&
         "TCGen05 MXF8F6F4 requires UE8M0 scale factors");
  assert((desc.k == 32 || desc.k == 64) &&
         "Dense TCGen05 MXF8F6F4 requires K32 or K64");
  const uint32_t scale_blocks = desc.k / 32;
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MXF8F6F4 A matrix size does not match descriptor");
  assert(b.size() == desc.n * desc.k &&
         "TCGen05 MXF8F6F4 B matrix size does not match descriptor");
  assert(scale_a.size() == desc.m * scale_blocks &&
         "TCGen05 MXF8F6F4 A scale matrix size does not match descriptor");
  assert(scale_b.size() == desc.n * scale_blocks &&
         "TCGen05 MXF8F6F4 B scale matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MXF8F6F4 D matrix size does not match descriptor");

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum = enable_input_d
                      ? tcgen05_bits_to_f32(input_d[row * desc.n + col])
                      : 0.0f;
      for (uint32_t block = 0; block < scale_blocks; ++block) {
        const float a_scale =
            tcgen05_ue8m0_to_f32(scale_a[row * scale_blocks + block]);
        const float b_scale =
            tcgen05_ue8m0_to_f32(scale_b[col * scale_blocks + block]);
        for (uint32_t k = block * 32; k < (block + 1) * 32; ++k) {
          // Extracted logical values occupy one byte each, independently of
          // native padded K32 or compact K64 shared-memory storage.
          uint32_t a_index =
              desc.transpose_a ? k * desc.m + row : row * desc.k + k;
          uint32_t b_index =
              desc.transpose_b ? k * desc.n + col : col * desc.k + k;
          float a_value = tcgen05_mxf8f6f4_to_f32(a[a_index], desc.a_type);
          float b_value = tcgen05_mxf8f6f4_to_f32(b[b_index], desc.b_type);
          if (desc.negate_a)
            a_value = -a_value;
          if (desc.negate_b)
            b_value = -b_value;
          sum += (a_value * a_scale) * (b_value * b_scale);
        }
      }
      output[row * desc.n + col] = tcgen05_f32_to_bits(sum);
    }
  }
  return output;
}

std::vector<uint32_t> tcgen05_mma_mxf8f6f4_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint8_t> &a,
    const std::vector<uint8_t> &b, const std::vector<uint8_t> &scale_a,
    const std::vector<uint8_t> &scale_b, const std::vector<uint32_t> &input_d,
    bool enable_input_d) {
  assert(tcgen05_mxf8f6f4_format_supported(desc.a_type) &&
         tcgen05_mxf8f6f4_format_supported(desc.b_type) &&
         "Unsupported TCGen05 MXF8F6F4 input format");
  assert(desc.scale_format == TCGEN05_SCALE_FORMAT_UE8M0 &&
         "TCGen05 MXF8F6F4 requires UE8M0 scale factors");
  assert((desc.k == 32 || desc.k == 64) &&
         "Dense TCGen05 MXF8F6F4 requires K32 or K64");
  const uint32_t scale_blocks = desc.k / 32;
  assert(a.size() == desc.m * desc.k &&
         "TCGen05 MXF8F6F4 A matrix size does not match descriptor");
  assert(b.size() == desc.n * desc.k &&
         "TCGen05 MXF8F6F4 B matrix size does not match descriptor");
  assert(scale_a.size() == desc.m * scale_blocks &&
         "TCGen05 MXF8F6F4 A scale matrix size does not match descriptor");
  assert(scale_b.size() == desc.n * scale_blocks &&
         "TCGen05 MXF8F6F4 B scale matrix size does not match descriptor");
  assert((!enable_input_d || input_d.size() == desc.m * desc.n) &&
         "TCGen05 MXF8F6F4 D matrix size does not match descriptor");

  // Decode each operand once per MMA, not once per output element. Keep the
  // original float conversion, negation and scale multiplication order. The
  // reduction below still visits K in order, with the same multiply/add
  // rounding; this is a host optimization, not a tensor-core timing change.
  bool finite_operands = true;
  auto decode_scaled = [&](const std::vector<uint8_t> &values,
                           const std::vector<uint8_t> &scales, uint32_t rows,
                           uint8_t type, bool transpose, bool negate) {
    std::vector<float> decoded(rows * desc.k);
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t block = 0; block < scale_blocks; ++block) {
        const float scale =
            tcgen05_ue8m0_to_f32(scales[row * scale_blocks + block]);
        for (uint32_t k = block * 32; k < (block + 1) * 32; ++k) {
          const uint32_t index = transpose ? k * rows + row : row * desc.k + k;
          float value = tcgen05_mxf8f6f4_to_f32(values[index], type);
          if (negate)
            value = -value;
          const float scaled = value * scale;
          finite_operands &= std::isfinite(scaled);
          decoded[row * desc.k + k] = scaled;
        }
      }
    }
    return decoded;
  };
  const auto decoded_a = decode_scaled(a, scale_a, desc.m, desc.a_type,
                                       desc.transpose_a, desc.negate_a);
  const auto decoded_b = decode_scaled(b, scale_b, desc.n, desc.b_type,
                                       desc.transpose_b, desc.negate_b);
  if (enable_input_d)
    for (const auto word : input_d)
      finite_operands &= std::isfinite(tcgen05_bits_to_f32(word));
  if (!finite_operands)
    return mixed_mma_scalar(desc, a, b, scale_a, scale_b, input_d,
                            enable_input_d);

  std::vector<uint32_t> output(desc.m * desc.n, 0);
  for (uint32_t row = 0; row < desc.m; ++row) {
    for (uint32_t col = 0; col < desc.n; ++col) {
      float sum = enable_input_d
                      ? tcgen05_bits_to_f32(input_d[row * desc.n + col])
                      : 0.0f;
      for (uint32_t k = 0; k < desc.k; ++k)
        sum += decoded_a[row * desc.k + k] * decoded_b[col * desc.k + k];
      if (!std::isfinite(sum))
        return mixed_mma_scalar(desc, a, b, scale_a, scale_b, input_d,
                                enable_input_d);
      output[row * desc.n + col] = tcgen05_f32_to_bits(sum);
    }
  }
  return output;
}

std::vector<uint32_t> tcgen05_mma_tf32_compute_words(
    unsigned m, unsigned n, const std::vector<uint32_t> &a,
    const std::vector<uint32_t> &b, const std::vector<uint32_t> &input_d,
    bool accumulate, bool negate_a, bool negate_b) {
  assert((m == 64 || m == 128) && n >= 8 && n <= 256 && n % 8 == 0);
  assert(a.size() == m * 8 && b.size() == n * 8);
  assert(!accumulate || input_d.size() == m * n);
  std::vector<uint32_t> output(m * n);
  for (unsigned row = 0; row < m; ++row)
    for (unsigned col = 0; col < n; ++col) {
      float sum =
          accumulate ? tcgen05_bits_to_f32(input_d[row * n + col]) : 0.0f;
      for (unsigned k = 0; k < 8; ++k) {
        float av = tcgen05_bits_to_f32(a[row * 8 + k] & ~0x1fffu);
        float bv = tcgen05_bits_to_f32(b[col * 8 + k] & ~0x1fffu);
        if (negate_a)
          av = -av;
        if (negate_b)
          bv = -bv;
        sum = std::fma(av, bv, sum);
      }
      output[row * n + col] = tcgen05_f32_to_bits(sum);
    }
  return output;
}

std::vector<uint32_t> tcgen05_mma_f16_tmem_a_compute_words(
    const tcgen05_mma_descriptor_t &desc,
    const std::vector<std::vector<uint16_t>> &physical_a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input,
    bool accumulate, bool folded, unsigned lane_origin) {
  const unsigned group = physical_a.size();
  const unsigned m = desc.m, n = desc.n;
  assert((group == 1 || group == 2) && desc.k == 16 && !desc.transpose_a);
  const unsigned local_m = m / group;
  assert(folded ? local_m == 64 : (local_m == 64 || local_m == 128));
  assert(lane_origin == 0 || (!folded && local_m == 64 && lane_origin == 16));
  assert(n && n % (folded ? 2u : 1u) == 0);
  assert(b.size() == n * 16 && (!accumulate || input.size() == m * n));
  for (const auto &image : physical_a)
    assert(image.size() == 128 * 16);
  // PTX ISA Layout B (M128/CTA2) and Layout E (M64/WS) map the
  // N halves to lower/upper datapaths. Do not assume the A images equal:
  // retained FlashMLA uses different physical images for its dual GEMM.
  const unsigned parts = folded ? 2 : 1, part_n = n / parts;
  std::vector<uint16_t> a(m * 16);
  std::vector<uint32_t> output(m * n);
  for (unsigned part = 0; part < parts; ++part) {
    for (unsigned rank = 0; rank < group; ++rank)
      for (unsigned row = 0; row < local_m; ++row)
        for (unsigned k = 0; k < 16; ++k)
          a[(rank * local_m + row) * 16 + k] =
              physical_a[rank][(folded ? row + part * 64
                                : local_m == 64
                                    ? (row / 16) * 32 + row % 16 + lane_origin
                                    : row) *
                                   16 +
                               k];
    std::vector<uint16_t> part_b(part_n * 16);
    for (unsigned col = 0; col < part_n; ++col)
      for (unsigned k = 0; k < 16; ++k)
        part_b[desc.transpose_b ? k * part_n + col : col * 16 + k] =
            b[desc.transpose_b ? k * n + part * part_n + col
                               : (part * part_n + col) * 16 + k];
    std::vector<uint32_t> part_input;
    if (accumulate) {
      part_input.resize(m * part_n);
      for (unsigned row = 0; row < m; ++row)
        for (unsigned col = 0; col < part_n; ++col)
          part_input[row * part_n + col] = input[row * n + part * part_n + col];
    }
    auto part_desc = desc;
    part_desc.n = part_n;
    const auto result = tcgen05_mma_f16_compute_words(part_desc, a, part_b,
                                                      part_input, accumulate);
    for (unsigned row = 0; row < m; ++row)
      for (unsigned col = 0; col < part_n; ++col)
        output[row * n + part * part_n + col] = result[row * part_n + col];
  }
  return output;
}

} // namespace flash_gpgpu_sim
