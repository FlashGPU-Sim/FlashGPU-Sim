#pragma once
#include <array>
#include <cstdint>
class kernel_info_t;

namespace flash_gpgpu_sim {
// The same opaque representation is decoded by PTX query_cancel and the
// compiler-generated native response loads/bitfields (CUDA 13.3 SM100).
inline std::array<uint32_t, 4> clc_response(bool canceled, uint32_t x,
                                            uint32_t y, uint32_t z) {
  return canceled ? std::array<uint32_t, 4>{{x, y | (z << 16), 1, 0}}
                  : std::array<uint32_t, 4>{{0, 0, 0, 0}};
}
struct clc_request {
  kernel_info_t *kernel = nullptr;
  unsigned cta = 0, warp = 0, generation = 0;
  uint64_t logical_cta = 0, ready_cycle = 0;
  uint32_t response_address = 0, barrier_address = 0;
  bool broadcast = false, arbitrated = false;
  std::array<uint32_t, 4> response{};
};
} // namespace flash_gpgpu_sim
