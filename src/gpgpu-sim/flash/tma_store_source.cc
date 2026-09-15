#include "tma_store_source.h"

#include <algorithm>
#include <limits>

namespace flash_gpgpu_sim {

uint64_t tma_store_source_tracker_t::schedule(uint64_t issue_cycle,
                                              uint32_t bytes) {
  if (!enabled())
    return 0;

  const uint64_t ready_cycle =
      issue_cycle > std::numeric_limits<uint64_t>::max() - m_fixed_latency
          ? std::numeric_limits<uint64_t>::max()
          : issue_cycle + m_fixed_latency;
  const uint64_t service_cycles =
      (static_cast<uint64_t>(bytes) + m_bytes_per_cycle - 1) /
      m_bytes_per_cycle;
  const uint64_t start_cycle = std::max(ready_cycle, m_available_cycle);
  m_available_cycle =
      start_cycle > std::numeric_limits<uint64_t>::max() - service_cycles
          ? std::numeric_limits<uint64_t>::max()
          : start_cycle + service_cycles;
  return m_available_cycle;
}

} // namespace flash_gpgpu_sim
