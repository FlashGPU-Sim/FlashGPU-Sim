#pragma once
#include "bfloat_arithmetic.h"
#include <cstdint>
namespace flash_gpgpu_sim {
// Non-propagating min/max: numeric values win over either kind of NaN;
// -0 is ordered below +0. Both frontends use the same pair operation.
inline uint16_t bf16_minmax(uint16_t a, uint16_t b, bool minimum) {
  const bool a_nan = (a & 0x7fff) > 0x7f80;
  const bool b_nan = (b & 0x7fff) > 0x7f80;
  if (a_nan)
    return b_nan ? 0x7fff : b;
  if (b_nan)
    return a;
  if (!(a & 0x7fff) && !(b & 0x7fff))
    return minimum ? (a | b) : (a & b);
  const float x = bf16_to_f32(a), y = bf16_to_f32(b);
  return minimum ? (x < y ? a : b) : (x > y ? a : b);
}
} // namespace flash_gpgpu_sim
