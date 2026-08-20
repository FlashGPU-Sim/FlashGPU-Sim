#include "timing.h"

#include <algorithm>
#include <cassert>
#include <tuple>

namespace flash_gpgpu_sim {

bool tcgen05_thread_stream_key_t::operator<(
    const tcgen05_thread_stream_key_t &other) const {
  return std::tie(hw_cta_id, warp_id, lane_id, cta_group) <
         std::tie(other.hw_cta_id, other.warp_id, other.lane_id,
                  other.cta_group);
}

bool tcgen05_warp_stream_key_t::operator<(
    const tcgen05_warp_stream_key_t &other) const {
  return std::tie(hw_cta_id, warp_id) <
         std::tie(other.hw_cta_id, other.warp_id);
}

tcgen05_unit_t::tcgen05_unit_t(const tcgen05_timing_config_t &config)
    : m_config(config) {
  assert(m_config.mma_issue_interval > 0);
  assert(m_config.mma_f16_flops_per_cycle > 0);
}

bool tcgen05_unit_t::can_enqueue(tcgen05_op_kind_t kind, uint64_t cycle) {
  assert(kind < TCGEN05_TIMING_OP_COUNT);
  if (m_config.async_queue_depth != 0 &&
      m_pending_ops.size() >= m_config.async_queue_depth) {
    if (cycle != m_last_queue_full_stall_cycle) {
      ++m_stats.queue_full_stall_cycles;
      m_last_queue_full_stall_cycle = cycle;
    }
    return false;
  }
  if (cycle < m_next_issue_cycle[kind]) {
    if (cycle != m_last_issue_interval_stall_cycle) {
      ++m_stats.issue_interval_stall_cycles;
      m_last_issue_interval_stall_cycle = cycle;
    }
    return false;
  }
  return true;
}

uint64_t
tcgen05_unit_t::enqueue_thread_op(const tcgen05_thread_stream_key_t &stream,
                                  const tcgen05_op_t &op, uint64_t cycle) {
  assert(op.kind == TCGEN05_TIMING_MMA || op.kind == TCGEN05_TIMING_CP ||
         op.kind == TCGEN05_TIMING_SHIFT);
  assert(can_enqueue(op.kind, cycle));

  pending_op_t pending;
  pending.thread_scoped = true;
  pending.thread_stream = stream;
  pending.kind = op.kind;
  pending.sequence = ++m_thread_next_sequence[stream];

  if (op.kind == TCGEN05_TIMING_MMA) {
    assert(op.work > 0);
    if (cycle >= m_mma_backend_available_cycle) {
      m_mma_busy_period_start_cycle = cycle;
      m_mma_busy_period_work = 0;
      // cycle() runs before scheduler issue, so account for the first busy
      // cycle of a newly started backend interval here.
      ++m_stats.backend_busy_cycles;
    }
    m_mma_busy_period_work += op.work;
    const uint64_t compute_cycles = std::max<uint64_t>(
        1, (m_mma_busy_period_work + m_config.mma_f16_flops_per_cycle - 1) /
               m_config.mma_f16_flops_per_cycle);
    pending.backend_done_cycle = m_mma_busy_period_start_cycle + compute_cycles;
    pending.completion_cycle =
        pending.backend_done_cycle + m_config.mma_completion_base;
    m_mma_backend_available_cycle = pending.backend_done_cycle;
    m_next_issue_cycle[op.kind] = cycle + m_config.mma_issue_interval;
  } else {
    assert(op.completion_latency > 0 && op.initiation_interval > 0);
    pending.backend_done_cycle = cycle + op.completion_latency;
    pending.completion_cycle = pending.backend_done_cycle;
    m_next_issue_cycle[op.kind] = cycle + op.initiation_interval;
  }

  m_pending_ops.push_back(pending);
  ++m_stats.issued[op.kind];
  m_stats.max_queue_occupancy =
      std::max<unsigned>(m_stats.max_queue_occupancy, m_pending_ops.size());
  return pending.sequence;
}

uint64_t
tcgen05_unit_t::enqueue_warp_mem_op(const tcgen05_warp_stream_key_t &stream,
                                    const tcgen05_op_t &op, uint64_t cycle) {
  assert(op.kind == TCGEN05_TIMING_LD || op.kind == TCGEN05_TIMING_ST);
  assert(op.completion_latency > 0 && op.initiation_interval > 0);
  assert(can_enqueue(op.kind, cycle));

  std::map<tcgen05_warp_stream_key_t, uint64_t> &sequences =
      op.kind == TCGEN05_TIMING_LD ? m_ld_next_sequence : m_st_next_sequence;
  pending_op_t pending;
  pending.thread_scoped = false;
  pending.warp_stream = stream;
  pending.kind = op.kind;
  pending.sequence = ++sequences[stream];
  pending.backend_done_cycle = cycle + op.completion_latency;
  pending.completion_cycle = pending.backend_done_cycle;
  m_next_issue_cycle[op.kind] = cycle + op.initiation_interval;
  m_pending_ops.push_back(pending);
  ++m_stats.issued[op.kind];
  m_stats.max_queue_occupancy =
      std::max<unsigned>(m_stats.max_queue_occupancy, m_pending_ops.size());
  return pending.sequence;
}

void tcgen05_unit_t::commit(const tcgen05_thread_stream_key_t &stream,
                            uint32_t mbarrier_addr) {
  pending_commit_t commit;
  commit.stream = stream;
  commit.cutoff = m_thread_next_sequence[stream];
  commit.mbarrier_addr = mbarrier_addr;
  m_pending_commits.push_back(commit);
  resolve_controls();
}

uint64_t
tcgen05_unit_t::last_warp_sequence(const tcgen05_warp_stream_key_t &stream,
                                   tcgen05_op_kind_t kind) const {
  const std::map<tcgen05_warp_stream_key_t, uint64_t> &sequences =
      kind == TCGEN05_TIMING_LD ? m_ld_next_sequence : m_st_next_sequence;
  std::map<tcgen05_warp_stream_key_t, uint64_t>::const_iterator it =
      sequences.find(stream);
  return it == sequences.end() ? 0 : it->second;
}

bool tcgen05_unit_t::wait_ld(const tcgen05_warp_stream_key_t &stream) {
  pending_wait_t wait;
  wait.stream = stream;
  wait.kind = TCGEN05_TIMING_LD;
  wait.cutoff = last_warp_sequence(stream, wait.kind);
  if (warp_cutoff_satisfied(stream, wait.kind, wait.cutoff))
    return true;
  m_pending_waits.push_back(wait);
  return false;
}

bool tcgen05_unit_t::wait_st(const tcgen05_warp_stream_key_t &stream) {
  pending_wait_t wait;
  wait.stream = stream;
  wait.kind = TCGEN05_TIMING_ST;
  wait.cutoff = last_warp_sequence(stream, wait.kind);
  if (warp_cutoff_satisfied(stream, wait.kind, wait.cutoff))
    return true;
  m_pending_waits.push_back(wait);
  return false;
}

bool tcgen05_unit_t::thread_cutoff_satisfied(
    const tcgen05_thread_stream_key_t &stream, uint64_t cutoff) const {
  for (std::vector<pending_op_t>::const_iterator it = m_pending_ops.begin();
       it != m_pending_ops.end(); ++it) {
    if (it->thread_scoped && !(it->thread_stream < stream) &&
        !(stream < it->thread_stream) && it->sequence <= cutoff)
      return false;
  }
  return true;
}

bool tcgen05_unit_t::warp_cutoff_satisfied(
    const tcgen05_warp_stream_key_t &stream, tcgen05_op_kind_t kind,
    uint64_t cutoff) const {
  for (std::vector<pending_op_t>::const_iterator it = m_pending_ops.begin();
       it != m_pending_ops.end(); ++it) {
    if (!it->thread_scoped && it->kind == kind && !(it->warp_stream < stream) &&
        !(stream < it->warp_stream) && it->sequence <= cutoff)
      return false;
  }
  return true;
}

void tcgen05_unit_t::resolve_controls() {
  for (std::vector<pending_commit_t>::iterator it = m_pending_commits.begin();
       it != m_pending_commits.end();) {
    if (!thread_cutoff_satisfied(it->stream, it->cutoff)) {
      ++it;
      continue;
    }
    tcgen05_completion_event_t event;
    event.kind = tcgen05_completion_event_t::MBARRIER_ARRIVAL;
    event.hw_cta_id = it->stream.hw_cta_id;
    event.warp_id = it->stream.warp_id;
    event.mbarrier_addr = it->mbarrier_addr;
    m_completion_events.push_back(event);
    it = m_pending_commits.erase(it);
  }

  for (std::vector<pending_wait_t>::iterator it = m_pending_waits.begin();
       it != m_pending_waits.end();) {
    if (!warp_cutoff_satisfied(it->stream, it->kind, it->cutoff)) {
      ++it;
      continue;
    }
    tcgen05_completion_event_t event;
    event.kind = it->kind == TCGEN05_TIMING_LD
                     ? tcgen05_completion_event_t::RELEASE_LD_WAIT
                     : tcgen05_completion_event_t::RELEASE_ST_WAIT;
    event.hw_cta_id = it->stream.hw_cta_id;
    event.warp_id = it->stream.warp_id;
    m_completion_events.push_back(event);
    it = m_pending_waits.erase(it);
  }
}

void tcgen05_unit_t::cycle(uint64_t cycle) {
  if (cycle < m_mma_backend_available_cycle)
    ++m_stats.backend_busy_cycles;
  m_stats.commit_wait_cycles += m_pending_commits.size();
  for (std::vector<pending_wait_t>::const_iterator it = m_pending_waits.begin();
       it != m_pending_waits.end(); ++it) {
    if (it->kind == TCGEN05_TIMING_LD)
      ++m_stats.ld_wait_cycles;
    else
      ++m_stats.st_wait_cycles;
  }

  for (std::vector<pending_op_t>::iterator it = m_pending_ops.begin();
       it != m_pending_ops.end();) {
    if (cycle < it->completion_cycle) {
      ++it;
      continue;
    }
    ++m_stats.completed[it->kind];
    it = m_pending_ops.erase(it);
  }
  resolve_controls();
}

std::vector<tcgen05_completion_event_t>
tcgen05_unit_t::take_completion_events() {
  std::vector<tcgen05_completion_event_t> result;
  result.swap(m_completion_events);
  return result;
}

void tcgen05_unit_t::cleanup_cta(unsigned hw_cta_id) {
  m_pending_ops.erase(
      std::remove_if(m_pending_ops.begin(), m_pending_ops.end(),
                     [hw_cta_id](const pending_op_t &op) {
                       return op.thread_scoped
                                  ? op.thread_stream.hw_cta_id == hw_cta_id
                                  : op.warp_stream.hw_cta_id == hw_cta_id;
                     }),
      m_pending_ops.end());
  m_pending_commits.erase(
      std::remove_if(m_pending_commits.begin(), m_pending_commits.end(),
                     [hw_cta_id](const pending_commit_t &commit) {
                       return commit.stream.hw_cta_id == hw_cta_id;
                     }),
      m_pending_commits.end());
  m_pending_waits.erase(
      std::remove_if(m_pending_waits.begin(), m_pending_waits.end(),
                     [hw_cta_id](const pending_wait_t &wait) {
                       return wait.stream.hw_cta_id == hw_cta_id;
                     }),
      m_pending_waits.end());
  for (std::map<tcgen05_thread_stream_key_t, uint64_t>::iterator it =
           m_thread_next_sequence.begin();
       it != m_thread_next_sequence.end();) {
    if (it->first.hw_cta_id == hw_cta_id)
      it = m_thread_next_sequence.erase(it);
    else
      ++it;
  }
  for (std::map<tcgen05_warp_stream_key_t, uint64_t>::iterator it =
           m_ld_next_sequence.begin();
       it != m_ld_next_sequence.end();) {
    if (it->first.hw_cta_id == hw_cta_id)
      it = m_ld_next_sequence.erase(it);
    else
      ++it;
  }
  for (std::map<tcgen05_warp_stream_key_t, uint64_t>::iterator it =
           m_st_next_sequence.begin();
       it != m_st_next_sequence.end();) {
    if (it->first.hw_cta_id == hw_cta_id)
      it = m_st_next_sequence.erase(it);
    else
      ++it;
  }
}

unsigned tcgen05_unit_t::queue_occupancy() const {
  return static_cast<unsigned>(m_pending_ops.size());
}

} // namespace flash_gpgpu_sim
