#ifndef FLASH_GPGPU_SIM_INSTRUCTION_FETCH_TIMING_H_
#define FLASH_GPGPU_SIM_INSTRUCTION_FETCH_TIMING_H_

#include <algorithm>
#include <cstdint>

namespace flash_gpgpu_sim {

struct instruction_loop_buffer_config {
  unsigned backedge_redirect_latency = 0;
  uint64_t capacity_bytes = 0;
  uint64_t refill_granularity_bytes = 0;
  unsigned refill_latency = 0;
  unsigned max_refill_latency = 0;
};

// Return the fetch bubble after a taken physical-ISA backedge.  Source ISA
// frontends leave this disabled because their PCs do not describe a native
// instruction working set.
inline unsigned
instruction_loop_refill_delay(uint64_t branch_pc, uint64_t target_pc,
                              unsigned instruction_bytes,
                              const instruction_loop_buffer_config &config) {
  if (target_pc >= branch_pc)
    return 0;

  const unsigned redirect = config.backedge_redirect_latency;
  if (config.capacity_bytes == 0 || config.refill_granularity_bytes == 0 ||
      config.refill_latency == 0 || config.max_refill_latency == 0)
    return redirect;

  const uint64_t span = branch_pc - target_pc + instruction_bytes;
  if (span <= config.capacity_bytes)
    return redirect;
  const uint64_t excess = span - config.capacity_bytes;
  const uint64_t refills = 1 + (excess - 1) / config.refill_granularity_bytes;
  return redirect +
         static_cast<unsigned>(std::min<uint64_t>(
             config.max_refill_latency, refills * config.refill_latency));
}

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_INSTRUCTION_FETCH_TIMING_H_
