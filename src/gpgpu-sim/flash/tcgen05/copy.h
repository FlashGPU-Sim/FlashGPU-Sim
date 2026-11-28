#pragma once
#include <cstdint>
#include <vector>
namespace flash_gpgpu_sim {
enum class tcgen05_copy_shape {
  rows128_bits256,
  rows128_bits128,
  warpx2_02_13,
  warpx2_01_23,
  warpx4
};
// One address for each destination word, in datapath-major order. Both
// frontends use this layout; memory ownership and error transport stay outside.
bool tcgen05_copy_addresses(uint64_t descriptor, tcgen05_copy_shape shape,
                            std::vector<uint32_t> &addresses);
} // namespace flash_gpgpu_sim
