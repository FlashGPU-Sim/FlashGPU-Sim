#ifndef FLASH_GPGPU_SIM_SASS_LAUNCH_GEOMETRY_H_
#define FLASH_GPGPU_SIM_SASS_LAUNCH_GEOMETRY_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace runtime_detail {

// Simulator-local address layout shared by functional and timing launches.
constexpr uint64_t kLocalThreadStride = 0x10000;

inline uint32_t active_mask_for_warp(uint64_t threads, unsigned warp_id) {
  const uint64_t first_thread = uint64_t{warp_id} * kWarpLanes;
  if (first_thread >= threads)
    return 0;
  const unsigned remaining = static_cast<unsigned>(
      std::min<uint64_t>(kWarpLanes, threads - first_thread));
  return remaining == kWarpLanes ? std::numeric_limits<uint32_t>::max()
                                 : (uint32_t{1} << remaining) - 1;
}

} // namespace runtime_detail
} // namespace sass
} // namespace flash_gpgpu_sim

#endif
