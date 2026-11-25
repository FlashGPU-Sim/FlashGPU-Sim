#pragma once

#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {
// PTX zero-column descriptor examples and CUTLASS MaskAndShiftB agree:
// skip_span counts zeroed columns; use_span counts original B columns.
// The PTX field table's prose has these two descriptions reversed.
inline unsigned tcgen05_ws_max_shift(uint32_t idesc) {
  const unsigned code = idesc >> 30;
  return code ? 4u << code : 0;
}
inline bool tcgen05_ws_mask_supported(uint64_t raw, unsigned m, unsigned n,
                                      unsigned max_shift) {
  if ((m != 32 && m != 64 && m != 128) || !n || n % (128 / m) ||
      (max_shift != 0 && max_shift != 8 && max_shift != 16 &&
       max_shift != 32) ||
      (raw & ((7ull << 36) | (3ull << 62))))
    return false;
  const unsigned shift = (raw >> 56) & 63;
  return max_shift <= (m == 32 ? 16u : 32u) && shift <= max_shift;
}
inline bool tcgen05_ws_zero_column(uint64_t raw, unsigned m, unsigned n,
                                   unsigned col) {
  if (!(raw & (1ull << 39)))
    return false;
  const unsigned width = n / (128 / m), segment = col / width;
  const unsigned start = (raw >> (8 * segment)) & 255;
  const bool first_zero = (raw >> (32 + segment)) & 1;
  const unsigned zeros = ((raw >> 40) & 255) + 1;
  const unsigned uses = ((raw >> 48) & 255) + 1;
  // B200 clamps an oversized start count within the first span: at least
  // one column of that span remains. It does not wrap the start through
  // the whole pattern (native WS mask/collector boundary probes).
  const unsigned last_in_first = (first_zero ? zeros : uses) - 1;
  const unsigned offset = start < last_in_first ? start : last_in_first;
  const unsigned position = (col % width + offset) % (zeros + uses);
  return first_zero ? position < zeros : position >= uses;
}
// Collector storage retains the unmodified N+max_shift source columns.
// Mask and shift are per instruction and must not mutate the saved B bank.
template <typename T>
bool tcgen05_ws_select_b(uint64_t raw, unsigned m, unsigned n, unsigned k,
                         unsigned max_shift, bool transpose,
                         std::vector<T> &values) {
  if (!tcgen05_ws_mask_supported(raw, m, n, max_shift) ||
      values.size() != std::size_t(n + max_shift) * k)
    return false;
  const unsigned shift = (raw >> 56) & 63;
  if (!max_shift && !(raw & (1ull << 39)))
    return true;
  std::vector<T> selected(n * k);
  for (unsigned col = 0; col < n; ++col) {
    if (tcgen05_ws_zero_column(raw, m, n, col))
      continue;
    for (unsigned inner = 0; inner < k; ++inner)
      selected[transpose ? inner * n + col : col * k + inner] =
          values[transpose ? inner * (n + max_shift) + col + shift
                           : (col + shift) * k + inner];
  }
  values.swap(selected);
  return true;
}
} // namespace flash_gpgpu_sim
