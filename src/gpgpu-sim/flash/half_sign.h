#pragma once
#include <cstdint>
namespace flash_gpgpu_sim {
// IEEE binary16 and BF16 share their sign-bit location. Preserve payloads;
// abs/neg allow an unspecified NaN, so no arithmetic conversion is needed.
inline uint16_t half_sign_bits(uint16_t bits, bool absolute, bool negate) {
  if (absolute)
    bits &= 0x7fff;
  if (negate)
    bits ^= 0x8000;
  return bits;
}
inline uint16_t half_flush_subnormal(uint16_t bits) {
  return (bits & 0x7c00) == 0 ? bits & 0x8000 : bits;
}
} // namespace flash_gpgpu_sim
