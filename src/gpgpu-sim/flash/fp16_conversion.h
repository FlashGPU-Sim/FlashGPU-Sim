#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
namespace flash_gpgpu_sim {
namespace fp16_detail {
inline float from_bits(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
} // namespace fp16_detail
inline float fp16_to_f32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x3ffu;
  if (exponent == 0) {
    if (mantissa == 0)
      return fp16_detail::from_bits(sign);
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return sign == 0 ? magnitude : -magnitude;
  }
  if (exponent == 0x1f)
    return fp16_detail::from_bits(sign | 0x7f800000u | (mantissa << 13));
  return fp16_detail::from_bits(sign | ((exponent + 112u) << 23) |
                                (mantissa << 13));
}

} // namespace flash_gpgpu_sim
