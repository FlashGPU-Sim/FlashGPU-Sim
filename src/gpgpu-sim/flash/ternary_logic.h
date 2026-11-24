#pragma once
#include <cstdint>
namespace flash_gpgpu_sim {
// LUT bit index = (a_bit << 2) | (b_bit << 1) | c_bit. Expand its eight
// minterms over the whole word instead of evaluating 32 scalar bit lanes.
inline uint32_t ternary_logic(uint32_t a, uint32_t b, uint32_t c, uint8_t lut) {
  uint32_t result = 0;
  for (unsigned term = 0; term < 8; ++term)
    if (lut & (1u << term))
      result |= (term & 4 ? a : ~a) & (term & 2 ? b : ~b) & (term & 1 ? c : ~c);
  return result;
}
inline bool ternary_logic_predicate(uint32_t result, bool q, bool use_and) {
  return use_and ? result != 0 && q : result != 0 || q;
}
} // namespace flash_gpgpu_sim
