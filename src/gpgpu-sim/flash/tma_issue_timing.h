#pragma once

#include <limits>

namespace flash_gpgpu_sim {

// TMA data completion is asynchronous, but native SM120 SASS keeps the
// issuing warp in an elected-lane dispatch/helper path before it can execute
// the next PTX instruction. Track that issue-side occupancy independently of
// the TMA transaction and memory-service timing.
class tma_issue_timing_t {
public:
  void reset() { m_ready_cycle = 0; }

  void begin(unsigned long long issue_cycle, unsigned latency) {
    if (latency == 0)
      return;
    m_ready_cycle = saturating_add(issue_cycle, latency);
  }

  bool ready(unsigned long long cycle) const { return cycle >= m_ready_cycle; }

private:
  static unsigned long long saturating_add(unsigned long long value,
                                           unsigned long long delta) {
    const unsigned long long maximum =
        std::numeric_limits<unsigned long long>::max();
    return value > maximum - delta ? maximum : value + delta;
  }

  unsigned long long m_ready_cycle = 0;
};

// Optional PTX expansion: back-to-back transfers may issue before a following
// non-TMA instruction. Native SASS does not use this synthetic helper delay.
class tma_instruction_issue_hold_t {
public:
  void reset() {
    next_tma_.reset();
    next_other_.reset();
  }
  void note_issue(unsigned long long cycle, unsigned latency,
                  unsigned interval) {
    reset();
    next_tma_.begin(cycle, interval);
    next_other_.begin(cycle, latency);
  }
  bool stalls(unsigned long long cycle, bool next_is_tma) const {
    return !(next_is_tma ? next_tma_ : next_other_).ready(cycle);
  }

private:
  tma_issue_timing_t next_tma_, next_other_;
};
} // namespace flash_gpgpu_sim
