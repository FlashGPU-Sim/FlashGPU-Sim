#pragma once
#include <cstdint>

namespace flash_gpgpu_sim {
// Each source destination contributes one completion. Paired multicast may
// credit the same owner twice; the upper half retains that second credit.
inline bool tma_completion_targets(unsigned cluster_size, uint16_t destinations,
                                   unsigned barrier_rank, bool paired,
                                   bool multicast, uint32_t &completions) {
  completions = 0;
  if (!cluster_size || cluster_size > 16 || barrier_rank >= cluster_size ||
      (uint32_t(destinations) >> cluster_size))
    return false;
  for (unsigned rank = 0; rank < cluster_size; ++rank)
    if (destinations & (1u << rank)) {
      const unsigned owner =
          paired
              ? (multicast ? (rank & ~1u) | (barrier_rank & 1u) : barrier_rank)
              : rank;
      if (owner >= cluster_size || (paired && owner / 2 != rank / 2) ||
          (!paired && !multicast && owner != barrier_rank))
        return false;
      const uint32_t bit = 1u << owner;
      completions |= completions & bit ? bit << 16 : bit;
    }
  return true;
}

} // namespace flash_gpgpu_sim
