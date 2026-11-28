#ifndef FLASH_GPGPU_SIM_BIT_SCAN_H
#define FLASH_GPGPU_SIM_BIT_SCAN_H
#include <cassert>
#include <cstdint>
namespace flash_gpgpu_sim {
// PTX bfind and native FLO: locate the most significant non-sign bit.
inline uint32_t find_non_sign_bit(uint64_t value, unsigned bits, bool is_signed,
                                  bool shift_amount) {
  assert(bits == 32 || bits == 64);
  const uint64_t mask = bits == 32 ? UINT32_MAX : UINT64_MAX;
  value &= mask;
  if (is_signed && (value & (uint64_t(1) << (bits - 1))))
    value = (~value) & mask;
  if (!value)
    return UINT32_MAX;
  const unsigned bit = 63u - static_cast<unsigned>(__builtin_clzll(value));
  return shift_amount ? bits - 1 - bit : bit;
}
} // namespace flash_gpgpu_sim
#endif
