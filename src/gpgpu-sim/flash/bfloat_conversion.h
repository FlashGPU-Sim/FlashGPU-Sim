#pragma once
#include <cstdint>
#include <cstring>

namespace flash_gpgpu_sim {
// ISA-neutral BF16 conversion. No host rounding-mode or floating-point
// arithmetic is used, so ties, subnormals and overflow are deterministic.
inline uint16_t f32_bits_to_bf16(uint32_t bits, bool toward_zero = false,
                                 bool relu = false, bool satfinite = false) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return 0x7fff;
  if (relu && (bits >> 31) && magnitude)
    return 0;
  uint16_t result =
      toward_zero ? bits >> 16 : (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  if (satfinite && (result & 0x7fff) == 0x7f80)
    result = (result & 0x8000) | 0x7f7f;
  return result;
}
inline uint16_t f32_to_bf16_rn(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return f32_bits_to_bf16(bits);
}
} // namespace flash_gpgpu_sim
