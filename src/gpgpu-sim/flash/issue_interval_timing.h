#pragma once

#include <deque>
#include <limits>

namespace flash_gpgpu_sim {

// Track a minimum issue-to-issue recurrence for a modeled resource. The
// owner chooses the scope by keeping one instance per warp, per SM, or per
// device.
class issue_interval_timing_t {
public:
  void reset() { m_next_issue_cycle = 0; }

  void begin(unsigned long long issue_cycle, unsigned issue_interval) {
    if (issue_interval == 0)
      return;
    m_next_issue_cycle = saturating_add(issue_cycle, issue_interval);
  }

  bool ready(unsigned long long cycle) const {
    return cycle >= m_next_issue_cycle;
  }

private:
  static unsigned long long saturating_add(unsigned long long value,
                                           unsigned long long delta) {
    const unsigned long long maximum =
        std::numeric_limits<unsigned long long>::max();
    return value > maximum - delta ? maximum : value + delta;
  }

  unsigned long long m_next_issue_cycle = 0;
};

// Model a finite frontend queue feeding a recurrent backend service. A zero
// queue depth preserves the strict issue-to-issue gate above. With a nonzero
// depth, frontend issues may run ahead until the scheduled backend dispatches
// fill the queue; sustained throughput still converges to one instruction per
// issue_interval cycles.
class issue_service_queue_timing_t {
public:
  void reset() {
    m_strict.reset();
    m_next_service_cycle = 0;
    m_pending_service_cycles.clear();
  }

  bool ready(unsigned long long cycle, unsigned issue_interval,
             unsigned queue_depth) {
    if (issue_interval == 0)
      return true;
    if (queue_depth == 0)
      return m_strict.ready(cycle);
    retire_dispatched(cycle);
    return m_pending_service_cycles.size() < queue_depth;
  }

  unsigned long long begin(unsigned long long issue_cycle,
                           unsigned issue_interval, unsigned queue_depth) {
    if (issue_interval == 0)
      return issue_cycle;
    if (queue_depth == 0) {
      m_strict.begin(issue_cycle, issue_interval);
      return issue_cycle;
    }

    retire_dispatched(issue_cycle);
    const unsigned long long service_cycle =
        m_next_service_cycle > issue_cycle ? m_next_service_cycle : issue_cycle;
    m_next_service_cycle = saturating_add(service_cycle, issue_interval);
    if (service_cycle > issue_cycle)
      m_pending_service_cycles.push_back(service_cycle);
    return service_cycle;
  }

  std::size_t pending() const { return m_pending_service_cycles.size(); }

private:
  void retire_dispatched(unsigned long long cycle) {
    while (!m_pending_service_cycles.empty() &&
           m_pending_service_cycles.front() <= cycle)
      m_pending_service_cycles.pop_front();
  }

  static unsigned long long saturating_add(unsigned long long value,
                                           unsigned long long delta) {
    const unsigned long long maximum =
        std::numeric_limits<unsigned long long>::max();
    return value > maximum - delta ? maximum : value + delta;
  }

  issue_interval_timing_t m_strict;
  unsigned long long m_next_service_cycle = 0;
  std::deque<unsigned long long> m_pending_service_cycles;
};

} // namespace flash_gpgpu_sim
