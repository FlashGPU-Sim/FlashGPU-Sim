#pragma once
#include "bfloat_conversion.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
namespace flash_gpgpu_sim {
inline float bf16_to_f32(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
inline uint16_t ue8m0_to_bf16(uint8_t code) {
  return code == 255 ? 0x7fff : code == 0 ? 0x40 : uint16_t(code) << 7;
}
inline uint16_t e4m3_to_f16(uint8_t code) {
  const unsigned magnitude = code & 127, exponent = magnitude >> 3;
  if (magnitude == 127)
    return 0x7fff;
  const uint16_t sign = (code & 128) << 8;
  if (exponent)
    return sign | ((exponent + 8) << 10) | ((code & 7) << 7);
  if (!magnitude)
    return sign;
  const unsigned top = magnitude >= 4 ? 2 : magnitude >= 2 ? 1 : 0;
  return sign | ((top + 6) << 10) | ((magnitude << (10 - top)) & 1023);
}
inline uint16_t e4m3_scaled_to_bf16(uint8_t code, uint8_t scale = 127) {
  const unsigned magnitude = code & 127, exponent = magnitude >> 3;
  if (magnitude == 127 || scale == 255)
    return 0x7fff;
  float value = exponent
                    ? std::ldexp(float(8 + (magnitude & 7)), int(exponent) - 10)
                    : std::ldexp(float(magnitude & 7), -9);
  if (code & 128)
    value = -value;
  return f32_to_bf16_rn(std::ldexp(value, int(scale) - 127));
}
inline uint8_t f32_to_e4m3_satfinite_rn(float value) {
  if (std::isnan(value))
    return 0x7f;
  const uint8_t sign = std::signbit(value) ? 0x80 : 0;
  const float magnitude = std::fabs(value);
  if (magnitude >= 448.0f)
    return sign | 0x7e;
  if (magnitude == 0)
    return sign;
  // Quantize in units of the current exponent's ULP, including subnormals.
  int exponent = 0;
  std::frexp(magnitude, &exponent);
  const int encoded_exponent = std::max(0, exponent + 6);
  const float scaled =
      std::ldexp(magnitude, 10 - std::max(1, encoded_exponent));
  unsigned rounded = static_cast<unsigned>(scaled);
  const float remainder = scaled - rounded;
  if (remainder > 0.5f || (remainder == 0.5f && (rounded & 1)))
    ++rounded;
  const unsigned encoded =
      encoded_exponent ? (encoded_exponent - 1) * 8 + rounded : rounded;
  return sign | static_cast<uint8_t>(encoded);
}

inline uint8_t f32_to_e2m1_satfinite_rn(float value) {
  // CUDA's FP4 reference conversion: NaN saturates to positive MAXNORM.
  if (std::isnan(value))
    return 7;
  const uint8_t sign = std::signbit(value) ? 8 : 0;
  const float magnitude = std::fabs(value);
  constexpr float levels[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  unsigned code = 7;
  for (unsigned i = 0; i < 7; ++i) {
    const float midpoint = (levels[i] + levels[i + 1]) * 0.5f;
    if (magnitude < midpoint || (magnitude == midpoint && !(i & 1))) {
      code = i;
      break;
    }
  }
  return sign | code;
}

inline uint8_t f32_to_ue8m0_satfinite_rp(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits &= 0x7fffffff;
  const unsigned exponent = bits >> 23, fraction = bits & 0x7fffff;
  if (exponent == 255)
    return fraction ? 255 : 254;
  if (exponent == 0)
    return fraction > 0x400000 ? 1 : 0;
  return std::min(254u, exponent + (fraction != 0));
}

} // namespace flash_gpgpu_sim
