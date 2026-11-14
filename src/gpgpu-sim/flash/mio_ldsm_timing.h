#ifndef FLASH_GPGPU_SIM_MIO_LDSM_TIMING_H_
#define FLASH_GPGPU_SIM_MIO_LDSM_TIMING_H_

#include "panic.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {

// Per-subcore admission cadence for matrix shared-memory loads. The shared
// data path remains SM-wide and is modeled independently by the bank and data
// wavefront machinery.
class mio_ldsm_timing {
public:
  explicit mio_ldsm_timing(unsigned schedulers)
      : next_issue_cycles_(schedulers, 0) {}

  bool can_issue(unsigned scheduler, uint64_t cycle, unsigned interval) const {
    assert(scheduler < next_issue_cycles_.size());
    return interval == 0 || cycle >= next_issue_cycles_[scheduler];
  }

  void issue(unsigned scheduler, uint64_t cycle, unsigned interval) {
    assert(can_issue(scheduler, cycle, interval));
    if (interval != 0)
      next_issue_cycles_[scheduler] = cycle + interval;
  }

  void reset() {
    std::fill(next_issue_cycles_.begin(), next_issue_cycles_.end(), 0);
  }

  uint64_t next_issue_cycle(unsigned scheduler) const {
    if (scheduler >= next_issue_cycles_.size())
      panic("MIO LDSM scheduler is out of range");
    return next_issue_cycles_[scheduler];
  }

private:
  std::vector<uint64_t> next_issue_cycles_;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_MIO_LDSM_TIMING_H_
