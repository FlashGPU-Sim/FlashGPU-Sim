#pragma once
#include "shared_address.h"
namespace flash_gpgpu_sim {
struct cluster_store_request {
  unsigned owner = 0;
  uint32_t destination = 0, barrier = 0, bytes = 0;
};
// Architectural shared-state addresses have rank bits in the high byte. Both
// frontends validate the same extent/owner contract before touching memory.
inline bool decode_cluster_store(uint64_t destination, uint64_t barrier,
                                 unsigned bytes, unsigned cluster_size,
                                 cluster_store_request &request) {
  if (cluster_size <= 1 || (bytes != 4 && bytes != 8 && bytes != 16) ||
      destination > UINT32_MAX || barrier > UINT32_MAX)
    return false;
  const unsigned owner = shared_cluster_rank(destination);
  if (owner >= cluster_size || owner != shared_cluster_rank(barrier) ||
      destination % bytes || barrier % 8)
    return false;
  request = {owner, shared_cluster_offset(destination),
             shared_cluster_offset(barrier), bytes};
  return true;
}
} // namespace flash_gpgpu_sim
