#ifndef FLASH_GPGPU_SIM_TMA_STORE_SOURCE_H
#define FLASH_GPGPU_SIM_TMA_STORE_SOURCE_H

#include <cstdint>

namespace flash_gpgpu_sim {

// Models when a TMA global store has consumed its shared-memory source.  This
// is intentionally separate from destination memory acknowledgements: a bulk
// wait may release the producer once the source is reusable while the global
// writes continue through the memory hierarchy.
class tma_store_source_tracker_t {
public:
  tma_store_source_tracker_t() = default;
  tma_store_source_tracker_t(uint32_t bytes_per_cycle, uint32_t fixed_latency)
      : m_bytes_per_cycle(bytes_per_cycle), m_fixed_latency(fixed_latency) {}

  bool enabled() const { return m_bytes_per_cycle != 0; }

  // Returns the absolute source-release cycle, or zero when the calibrated
  // source model is disabled. Transactions share one ordered source engine.
  uint64_t schedule(uint64_t issue_cycle, uint32_t bytes);

private:
  uint32_t m_bytes_per_cycle = 0;
  uint32_t m_fixed_latency = 0;
  uint64_t m_available_cycle = 0;
};

} // namespace flash_gpgpu_sim

#endif
