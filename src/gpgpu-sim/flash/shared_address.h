#pragma once
#include <cstdint>

namespace flash_gpgpu_sim {
// Shared-state encoding compatible with captured SM100 rank-bit operations.
// Generic pointers use a
// separate owner-SM window; never apply this mask to a generic pointer.
constexpr uint32_t kSharedClusterOffsetMask = 0x00ffffffu;
inline uint32_t shared_cluster_offset(uint64_t address) {
  return static_cast<uint32_t>(address) & kSharedClusterOffsetMask;
}
inline unsigned shared_cluster_rank(uint32_t address) { return address >> 24; }
inline uint32_t shared_cluster_address(unsigned rank, uint32_t offset) {
  return (rank << 24) | shared_cluster_offset(offset);
}
} // namespace flash_gpgpu_sim
