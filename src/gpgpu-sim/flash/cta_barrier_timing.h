#ifndef FLASH_GPGPU_SIM_CTA_BARRIER_TIMING_H_
#define FLASH_GPGPU_SIM_CTA_BARRIER_TIMING_H_

#include <cassert>
#include <cstdint>

namespace flash_gpgpu_sim {

// Models the SM-wide admission path used by CTA barrier arrivals. Unlike the
// per-scheduler math pipelines, all schedulers contend for this single path.
class cta_barrier_timing {
public:
  bool can_issue(uint64_t cycle, unsigned interval) const {
    return interval == 0 || cycle >= next_issue_cycle_;
  }

  void issue(uint64_t cycle, unsigned interval) {
    assert(can_issue(cycle, interval));
    if (interval != 0)
      next_issue_cycle_ = cycle + interval;
  }

  void reset() { next_issue_cycle_ = 0; }

  uint64_t next_issue_cycle() const { return next_issue_cycle_; }

private:
  uint64_t next_issue_cycle_ = 0;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_CTA_BARRIER_TIMING_H_
