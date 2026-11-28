#pragma once
#include <algorithm>
#include <cstdint>
namespace flash_gpgpu_sim {
// PTX shf.l/r.{wrap,clamp}.b32. Use a wide intermediate so boundary counts
// 0 and 32 never evaluate a 32-bit C++ shift by its full word width.
inline uint32_t funnel_shift32(uint32_t low, uint32_t high, uint32_t count,
                               bool left, bool wrap) {
  const unsigned n = wrap ? count & 31u : std::min(count, 32u);
  const uint64_t pair = (uint64_t(high) << 32) | low;
  return left ? uint32_t((pair << n) >> 32) : uint32_t(pair >> n);
}
} // namespace flash_gpgpu_sim
