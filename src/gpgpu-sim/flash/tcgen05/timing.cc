#include "timing.h"

#include <algorithm>
#include <cassert>
#include <tuple>

namespace flash_gpgpu_sim {

namespace {
#include "native_recurrence_table.h"
}
unsigned tcgen05_b200_recurrence(tcgen05_native_profile profile, unsigned m,
                                 unsigned n, unsigned k, unsigned group,
                                 bool tmem_a, bool ws) {
  if (profile == tcgen05_native_profile::unknown || unsigned(profile) > 5 ||
      !m || m >= 512 || !n || n >= 512 || !k || k >= 256 ||
      (group != 1 && group != 2))
    return 0;
  const uint64_t key = uint64_t(profile) << 30 | uint64_t(group) << 28 |
                       uint64_t(tmem_a) << 27 | uint64_t(ws) << 26 |
                       uint64_t(m) << 17 | uint64_t(n) << 8 | k;
  const auto end = std::end(kB200Recurrence);
  const auto row = std::lower_bound(
      std::begin(kB200Recurrence), end, key,
      [](const auto &r, uint64_t value) { return r.key < value; });
  return row != end && row->key == key ? row->cycles : 0;
}

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

bool tcgen05_unit_t::can_enqueue(tcgen05_op_kind_t kind, uint64_t cycle,
                                 unsigned partition,
                                 const tcgen05_thread_stream_key_t *owner) {
  if (owner && kind == TCGEN05_TIMING_MMA && m_config.mma_precision_timing) {
    if (cycle < m_owner_issue[*owner])
      return false;
    const unsigned limit = m_config.owner_queue_depth[owner->cta_group - 1];
    if (limit) {
      unsigned count = 0;
      for (const auto &p : m_pending_ops)
        if (p.kind == kind && !(p.thread_stream < *owner) &&
            !(*owner < p.thread_stream))
          ++count;
      if (count >= limit)
        return false;
    }
  }
  assert(kind < TCGEN05_TIMING_OP_COUNT);
  if (m_config.async_queue_depth != 0 &&
      m_pending_ops.size() >= m_config.async_queue_depth) {
    if (cycle != m_last_queue_full_stall_cycle) {
      ++m_stats.queue_full_stall_cycles;
      m_last_queue_full_stall_cycle = cycle;
    }
    return false;
  }
  const uint64_t next = (kind == TCGEN05_TIMING_LD && partition != ~0u)
                            ? m_mem_next_issue[partition]
                            : m_next_issue_cycle[kind];
  if (cycle < next) {
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
                                  const tcgen05_op_t &op, uint64_t cycle,
                                  tcgen05_unit_t *peer,
                                  std::function<uint64_t(uint64_t)> feed) {
  assert(op.kind == TCGEN05_TIMING_MMA || op.kind == TCGEN05_TIMING_CP ||
         op.kind == TCGEN05_TIMING_SHIFT);
  // Admission is checked once before the warp executes. Several PTX lanes
  // may contribute distinct single-thread operations in that same issue.
  // Serialize their front-end service without dropping a lane's operation.
  cycle = std::max(cycle, m_next_issue_cycle[op.kind]);

  pending_op_t pending;
  pending.thread_scoped = true;
  pending.thread_stream = stream;
  pending.kind = op.kind;
  pending.sequence = ++m_thread_next_sequence[stream];

  if (op.kind == TCGEN05_TIMING_MMA && m_config.mma_precision_timing) {
    pending.backend_done_cycle = schedule_mma(stream, op, cycle, peer, feed);
    pending.completion_cycle =
        pending.backend_done_cycle + m_config.mma_completion_tail_latency;
    m_next_issue_cycle[op.kind] = cycle + m_config.mma_issue_interval;
    m_owner_issue[stream] = cycle + m_config.owner_issue_interval;
  } else if (op.kind == TCGEN05_TIMING_MMA) {
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
        pending.backend_done_cycle + m_config.mma_completion_tail_latency;
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

void tcgen05_unit_t::reserve_compute(uint64_t start, uint64_t compute_done,
                                     uint64_t service_done,
                                     uint64_t issue_cycle) {
  m_mma_backend_available_cycle = service_done;
  m_compute_windows.emplace_back(start, compute_done);
  if (start == issue_cycle && m_last_cycle_tick == issue_cycle)
    ++m_stats.backend_busy_cycles;
}

uint64_t
tcgen05_unit_t::schedule_mma(const tcgen05_thread_stream_key_t &stream,
                             const tcgen05_op_t &op, uint64_t cycle,
                             tcgen05_unit_t *peer,
                             const std::function<uint64_t(uint64_t)> &feed) {
  assert(op.work && (stream.cta_group == 1 || peer));
  unsigned rate = m_config.mma_rates[unsigned(op.mma.kind)];
  if (!rate)
    rate = m_config.mma_f16_flops_per_cycle;
  const uint64_t capacity = uint64_t(rate) * stream.cta_group;
  const uint64_t compute =
      std::max<uint64_t>(1, (op.work + capacity - 1) / capacity);
  uint64_t start =
      std::max(cycle + (feed ? 1 : 0), m_mma_backend_available_cycle);
  if (peer)
    start = std::max(start, peer->m_mma_backend_available_cycle);
  const uint64_t compute_done = start + compute;
  uint64_t service_done = compute_done;
  if (feed)
    service_done = std::max(service_done, feed(start));
  reserve_compute(start, compute_done, service_done, cycle);
  if (peer)
    peer->reserve_compute(start, compute_done, service_done, cycle);
  unsigned recurrence = 0;
  if (m_config.b200_recurrence) {
    recurrence =
        tcgen05_b200_recurrence(op.mma.profile, op.mma.m, op.mma.n, op.mma.k,
                                stream.cta_group, op.mma.tmem_a, op.mma.ws);
    if (recurrence)
      ++m_stats.recurrence_table_ops;
    else
      ++m_stats.recurrence_fallback_ops;
  }
  if (!recurrence)
    recurrence = m_config.owner_recurrence[stream.cta_group - 1];
  if (recurrence) {
    const auto found = m_owner_done.find(stream);
    const bool first = found == m_owner_done.end();
    const auto previous = first ? 0 : found->second;
    const auto done =
        std::max(cycle, previous) + recurrence +
        (first ? m_config.owner_startup[stream.cta_group - 1] : 0);
    m_owner_done[stream] = done;
    service_done = std::max(service_done, done);
  }
  return service_done;
}

uint64_t
tcgen05_unit_t::enqueue_warp_mem_op(const tcgen05_warp_stream_key_t &stream,
                                    const tcgen05_op_t &op, uint64_t cycle,
                                    uint64_t pipeline_completion_tag) {
  assert(op.kind == TCGEN05_TIMING_LD || op.kind == TCGEN05_TIMING_ST);
  assert(op.completion_latency > 0 && op.initiation_interval > 0);
  assert(
      can_enqueue(op.kind, cycle, op.partitioned_mem ? stream.partition : ~0u));

  std::map<tcgen05_warp_stream_key_t, uint64_t> &sequences =
      op.kind == TCGEN05_TIMING_LD ? m_ld_next_sequence : m_st_next_sequence;
  pending_op_t pending;
  pending.thread_scoped = false;
  pending.warp_stream = stream;
  pending.kind = op.kind;
  pending.sequence = ++sequences[stream];
  pending.pipeline_completion_tag = pipeline_completion_tag;
  pending.backend_done_cycle = pipeline_completion_tag
                                   ? std::numeric_limits<uint64_t>::max()
                                   : cycle + op.completion_latency;
  pending.completion_cycle = pending.backend_done_cycle;
  if (op.partitioned_mem)
    m_mem_next_issue[stream.partition] = cycle + op.initiation_interval;
  else
    m_next_issue_cycle[op.kind] = cycle + op.initiation_interval;
  m_pending_ops.push_back(pending);
  ++m_stats.issued[op.kind];
  m_stats.max_queue_occupancy =
      std::max<unsigned>(m_stats.max_queue_occupancy, m_pending_ops.size());
  return pending.sequence;
}

void tcgen05_unit_t::complete_warp_mem_op(uint64_t tag, uint64_t cycle) {
  assert(tag != 0);
  for (auto &pending : m_pending_ops) {
    if (!pending.thread_scoped && pending.pipeline_completion_tag == tag) {
      assert(pending.completion_cycle == std::numeric_limits<uint64_t>::max());
      pending.backend_done_cycle = pending.completion_cycle = cycle;
      return;
    }
  }
  assert(false && "TMEM pipeline completion must match a pending operation");
}

void tcgen05_unit_t::commit(const tcgen05_thread_stream_key_t &stream,
                            uint32_t mbarrier_addr, uint16_t multicast_mask) {
  pending_commit_t commit;
  commit.stream = stream;
  commit.cutoff = m_thread_next_sequence[stream];
  commit.mbarrier_addr = mbarrier_addr;
  commit.multicast_mask = multicast_mask;
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
    event.multicast_mask = it->multicast_mask;
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
  m_last_cycle_tick = cycle;
  if (m_config.mma_precision_timing) {
    while (!m_compute_windows.empty() &&
           m_compute_windows.front().second <= cycle)
      m_compute_windows.pop_front();
    if (!m_compute_windows.empty() && cycle >= m_compute_windows.front().first)
      ++m_stats.backend_busy_cycles;
  } else if (cycle < m_mma_backend_available_cycle)
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
  for (auto *map : {&m_owner_done, &m_owner_issue})
    for (auto it = map->begin(); it != map->end();)
      if (it->first.hw_cta_id == hw_cta_id)
        it = map->erase(it);
      else
        ++it;
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
