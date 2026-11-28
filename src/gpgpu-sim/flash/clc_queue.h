#pragma once
#include <array>
#include <cstdint>

namespace flash_gpgpu_sim {
// Queue cursor in the simulator's x-major admission order. Only a whole,
// unstarted (1|2,1,1) cluster may be consumed. No state changes on failure.
inline bool cancel_pending_clc_cluster(const std::array<uint32_t, 3> &grid,
                                       const std::array<uint32_t, 3> &cluster,
                                       std::array<uint32_t, 3> &next,
                                       std::array<uint32_t, 3> &canceled) {
  if ((cluster[0] != 1 && cluster[0] != 2) || cluster[1] != 1 ||
      cluster[2] != 1 || !grid[0] || !grid[1] || !grid[2] ||
      grid[0] % cluster[0] || next[0] >= grid[0] || next[1] >= grid[1] ||
      next[2] >= grid[2] || next[0] % cluster[0] || next[1] > UINT16_MAX ||
      next[2] > UINT16_MAX)
    return false;
  canceled = next;
  next[0] += cluster[0];
  if (next[0] == grid[0]) {
    next[0] = 0;
    if (++next[1] == grid[1]) {
      next[1] = 0;
      ++next[2];
    }
  }
  return true;
}
} // namespace flash_gpgpu_sim
