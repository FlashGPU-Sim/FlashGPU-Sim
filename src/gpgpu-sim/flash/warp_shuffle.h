#pragma once
#include <cstdint>

namespace flash_gpgpu_sim {
enum class shuffle_mode { up, down, butterfly, index };
struct shuffle_selection {
  unsigned lane;
  bool in_bounds;
};

// PTX shfl/shfl.sync lane selection. UP interprets the packed clamp as a lower
// bound; the other modes use it as an upper bound. Participation and source
// snapshots belong to the frontend's register/warp adapter.
inline shuffle_selection select_shuffle_lane(shuffle_mode mode, unsigned lane,
                                             uint32_t selector,
                                             uint32_t configuration) {
  const unsigned mask = (configuration >> 8) & 31u;
  const unsigned bound = (lane & mask) | ((configuration & 31u) & ~mask);
  const int delta = selector & 31u;
  int source = 0;
  switch (mode) {
  case shuffle_mode::up:
    source = int(lane) - delta;
    break;
  case shuffle_mode::down:
    source = int(lane) + delta;
    break;
  case shuffle_mode::butterfly:
    source = lane ^ delta;
    break;
  case shuffle_mode::index:
    source = (lane & mask) | (delta & ~mask);
    break;
  }
  const bool valid =
      mode == shuffle_mode::up ? source >= int(bound) : source <= int(bound);
  return {valid ? unsigned(source) : lane, valid};
}
} // namespace flash_gpgpu_sim
