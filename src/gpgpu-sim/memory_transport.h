// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MEMORY_TRANSPORT_H
#define MEMORY_TRANSPORT_H

#include <assert.h>
#include <limits.h>
#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <queue>
#include <vector>

#include "mem_fetch.h"

// Return the data-sector charge for a memory-transport packet.  WRITE_ACK is
// control-only, but still consumes one service slot (see
// memory_transport_service_slots()).
inline unsigned memory_transport_data_sectors(enum mf_type type,
                                              unsigned data_bytes) {
  if (type == WRITE_ACK) return 0;
  return data_bytes / SECTOR_SIZE + (data_bytes % SECTOR_SIZE != 0);
}

inline unsigned memory_transport_data_sectors(const mem_fetch *mf) {
  return memory_transport_data_sectors(mf->get_type(), mf->get_data_size());
}

inline unsigned memory_transport_service_slots(unsigned data_sectors) {
  return std::max(1u, data_sectors);
}

struct memory_transport_service_stats {
  memory_transport_service_stats()
      : accepted_data_sectors(0),
        accepted_control_packets(0),
        width_limited_ticks(0),
        downstream_full_ticks(0),
        service_ticks(0),
        max_service_slots_per_tick(0) {}

  void record_accept(unsigned data_sectors) {
    accepted_data_sectors += data_sectors;
    if (data_sectors == 0) ++accepted_control_packets;
  }

  void add(const memory_transport_service_stats &other) {
    accepted_data_sectors += other.accepted_data_sectors;
    accepted_control_packets += other.accepted_control_packets;
    width_limited_ticks += other.width_limited_ticks;
    downstream_full_ticks += other.downstream_full_ticks;
    service_ticks += other.service_ticks;
    max_service_slots_per_tick =
        std::max(max_service_slots_per_tick,
                 other.max_service_slots_per_tick);
  }

  void record_tick_service(unsigned service_slots) {
    if (service_slots != 0) ++service_ticks;
    max_service_slots_per_tick =
        std::max<unsigned long long>(max_service_slots_per_tick, service_slots);
  }

  void print(FILE *fout, const char *name) const {
    fprintf(fout, "%s_accepted_data_sectors = %llu\n", name,
            accepted_data_sectors);
    fprintf(fout, "%s_accepted_control_packets = %llu\n", name,
            accepted_control_packets);
    fprintf(fout, "%s_width_limited_ticks = %llu\n", name, width_limited_ticks);
    fprintf(fout, "%s_downstream_full_ticks = %llu\n", name,
            downstream_full_ticks);
    fprintf(fout, "%s_service_ticks = %llu\n", name, service_ticks);
    fprintf(fout, "%s_max_service_slots_per_tick = %llu\n", name,
            max_service_slots_per_tick);
  }

  unsigned long long accepted_data_sectors;
  unsigned long long accepted_control_packets;
  unsigned long long width_limited_ticks;
  unsigned long long downstream_full_ticks;
  unsigned long long service_ticks;
  unsigned long long max_service_slots_per_tick;
};

// A per-tick sector budget with bounded oversized-packet credit.  Idle or
// downstream-blocked ticks do not accumulate credit.  Credit is retained only
// when a head packet is larger than the configured single-tick budget, which
// guarantees forward progress for, e.g., a 128-byte packet at one sector/tick
// without splitting the packet.
class memory_transport_service_budget {
 public:
  memory_transport_service_budget()
      : m_width(0),
        m_available(0),
        m_credit(0),
        m_reserved_credit(0),
        m_credit_cap(0),
        m_consumed(0),
        m_width_limited(false),
        m_downstream_full(false),
        m_active(false) {}

  void begin_tick(unsigned width) {
    m_width = width;
    m_reserved_credit = m_credit;
    if (width != 0) {
      const unsigned long long available =
          static_cast<unsigned long long>(width) + m_credit;
      m_available = static_cast<unsigned>(
          std::min<unsigned long long>(available, UINT_MAX));
    } else {
      m_available = 0;
    }
    m_credit = 0;
    m_credit_cap = 0;
    m_consumed = 0;
    m_width_limited = false;
    m_downstream_full = false;
    m_active = true;
  }

  bool explicit_width() const { return m_width != 0; }

  bool can_accept(unsigned data_sectors) const {
    return explicit_width() &&
           memory_transport_service_slots(data_sectors) <= m_available;
  }

  void consume(unsigned data_sectors) {
    const unsigned slots = memory_transport_service_slots(data_sectors);
    assert(explicit_width());
    assert(slots <= m_available);
    m_available -= slots;
    m_consumed += slots;
  }

  void note_width_limited(unsigned data_sectors) {
    m_width_limited = true;
    const unsigned slots = memory_transport_service_slots(data_sectors);
    if (slots > m_width) m_credit_cap = std::max(m_credit_cap, slots - m_width);
  }
  void note_downstream_full() { m_downstream_full = true; }

  void end_tick(memory_transport_service_stats *stats) {
    assert(m_active);
    if (m_width_limited) ++stats->width_limited_ticks;
    if (m_downstream_full) ++stats->downstream_full_ticks;
    stats->record_tick_service(m_consumed);
    m_credit = m_width_limited && !m_downstream_full
                   ? std::min(m_available, m_credit_cap)
                   : 0;
    m_active = false;
    m_reserved_credit = 0;
  }

  unsigned remaining_slots() const { return m_available; }
  unsigned carried_credit() const { return m_credit; }
  bool has_reserved_credit() const { return m_reserved_credit != 0; }
  bool active() const { return m_active; }

 private:
  unsigned m_width;
  unsigned m_available;
  unsigned m_credit;
  unsigned m_reserved_credit;
  unsigned m_credit_cap;
  unsigned m_consumed;
  bool m_width_limited;
  bool m_downstream_full;
  bool m_active;
};

enum ldst_request_issue_stop_reason {
  LDST_REQUEST_DRAINED,
  LDST_REQUEST_WIDTH_LIMITED,
  LDST_REQUEST_DOWNSTREAM_FULL
};

struct ldst_request_issue_result {
  ldst_request_issue_result(ldst_request_issue_stop_reason stop_reason,
                            unsigned issued_children)
      : reason(stop_reason), issued(issued_children) {}

  ldst_request_issue_stop_reason reason;
  unsigned issued;
};

// Drain ordinary global/local coalescer children into the request injection
// path.  On sector-coalescing architectures each queue element is one internal
// 32-byte sector child; this helper deliberately does not claim that each child
// is a distinct physical L2 request.  The callbacks keep the helper independent
// of warp, ICNT, and cache implementation details while the before/after count
// assertion guarantees exactly one queue element is consumed per successful
// injection.
template <typename PendingCount, typename DownstreamFull, typename IssueOne>
ldst_request_issue_result memory_transport_issue_ldst_sector_children(
    memory_transport_service_budget *budget,
    memory_transport_service_stats *stats, PendingCount pending_count,
    DownstreamFull downstream_full, IssueOne issue_one) {
  assert(budget);
  assert(stats);
  assert(budget->active());

  unsigned issued = 0;
  while (pending_count() != 0) {
    if (!budget->can_accept(/*data_sectors=*/1)) {
      budget->note_width_limited(/*data_sectors=*/1);
      return ldst_request_issue_result(LDST_REQUEST_WIDTH_LIMITED, issued);
    }
    if (downstream_full()) {
      budget->note_downstream_full();
      return ldst_request_issue_result(LDST_REQUEST_DOWNSTREAM_FULL, issued);
    }

    const unsigned before = pending_count();
    const unsigned data_sectors = issue_one();
    assert(pending_count() + 1 == before);
    budget->consume(/*data_sectors=*/1);
    stats->record_accept(data_sectors);
    ++issued;
  }

  return ldst_request_issue_result(LDST_REQUEST_DRAINED, issued);
}

// Track the response packets belonging to each dynamic instruction separately
// from instruction/RF writeback.  Every response can retire at the configured
// transport width; only the final response produces one instruction-level
// completion for the ordinary writeback arbiter.
template <typename Key, typename T>
class memory_transport_response_retirement_queue {
 public:
  memory_transport_response_retirement_queue() : m_retired_responses(0) {}

  void expect_responses(const Key &key, unsigned count, const T &completion) {
    assert(count != 0);
    assert(m_pending.find(key) == m_pending.end());
    m_pending.insert(
        std::make_pair(key, pending_instruction(count, completion)));
  }

  bool has_pending_responses(const Key &key) const {
    return m_pending.find(key) != m_pending.end();
  }

  unsigned pending_responses(const Key &key) const {
    typename std::map<Key, pending_instruction>::const_iterator found =
        m_pending.find(key);
    assert(found != m_pending.end());
    return found->second.remaining;
  }

  bool retire_response(const Key &key) {
    typename std::map<Key, pending_instruction>::iterator found =
        m_pending.find(key);
    assert(found != m_pending.end());
    assert(found->second.remaining != 0);
    --found->second.remaining;
    ++m_retired_responses;
    if (found->second.remaining != 0) return false;

    m_completions.push_back(found->second.completion);
    m_pending.erase(found);
    return true;
  }

  bool completion_ready() const { return !m_completions.empty(); }
  size_t completion_count() const { return m_completions.size(); }
  size_t pending_instruction_count() const { return m_pending.size(); }
  unsigned long long retired_response_count() const {
    return m_retired_responses;
  }

  const T &next_completion() const {
    assert(completion_ready());
    return m_completions.front();
  }

  void pop_completion() {
    assert(completion_ready());
    m_completions.pop_front();
  }

 private:
  struct pending_instruction {
    pending_instruction(unsigned response_count, const T &value)
        : remaining(response_count), completion(value) {}

    unsigned remaining;
    T completion;
  };

  std::map<Key, pending_instruction> m_pending;
  std::deque<T> m_completions;
  unsigned long long m_retired_responses;
};

enum rop_delay_output_stop_reason {
  ROP_DELAY_OUTPUT_NO_READY_WORK,
  ROP_DELAY_OUTPUT_WIDTH_LIMITED,
  ROP_DELAY_OUTPUT_DOWNSTREAM_FULL,
};

struct rop_delay_output_service_result {
  rop_delay_output_service_result()
      : reason(ROP_DELAY_OUTPUT_NO_READY_WORK),
        accepted_items(0),
        accepted_sectors(0) {}

  rop_delay_output_stop_reason reason;
  unsigned accepted_items;
  unsigned accepted_sectors;
};

struct rop_delay_output_service_stats {
  rop_delay_output_service_stats()
      : accepted_items(0),
        accepted_sectors(0),
        service_ticks(0),
        max_sectors_per_tick(0),
        width_limited_ticks(0),
        downstream_full_ticks(0),
        queue_full_ticks(0) {}

  void record(const rop_delay_output_service_result &result) {
    accepted_items += result.accepted_items;
    accepted_sectors += result.accepted_sectors;
    if (result.accepted_items != 0) ++service_ticks;
    max_sectors_per_tick =
        std::max<unsigned long long>(max_sectors_per_tick,
                                     result.accepted_sectors);
    if (result.reason == ROP_DELAY_OUTPUT_WIDTH_LIMITED)
      ++width_limited_ticks;
    if (result.reason == ROP_DELAY_OUTPUT_DOWNSTREAM_FULL) {
      ++downstream_full_ticks;
      // The production ROP-delay consumer has exactly one downstream: the
      // bounded icnt-to-L2 FIFO. Keep the exact queue-full count separate so a
      // future downstream can be distinguished without changing this ABI.
      ++queue_full_ticks;
    }
  }

  rop_delay_output_service_stats &operator+=(
      const rop_delay_output_service_stats &rhs) {
    accepted_items += rhs.accepted_items;
    accepted_sectors += rhs.accepted_sectors;
    service_ticks += rhs.service_ticks;
    max_sectors_per_tick =
        std::max(max_sectors_per_tick, rhs.max_sectors_per_tick);
    width_limited_ticks += rhs.width_limited_ticks;
    downstream_full_ticks += rhs.downstream_full_ticks;
    queue_full_ticks += rhs.queue_full_ticks;
    return *this;
  }

  void print(FILE *fout, const char *name) const {
    fprintf(fout, "%s_accepted_items = %llu\n", name, accepted_items);
    fprintf(fout, "%s_accepted_sectors = %llu\n", name, accepted_sectors);
    fprintf(fout, "%s_service_ticks = %llu\n", name, service_ticks);
    fprintf(fout, "%s_max_sectors_per_tick = %llu\n", name,
            max_sectors_per_tick);
    fprintf(fout, "%s_width_limited_ticks = %llu\n", name,
            width_limited_ticks);
    fprintf(fout, "%s_downstream_full_ticks = %llu\n", name,
            downstream_full_ticks);
    fprintf(fout, "%s_queue_full_ticks = %llu\n", name, queue_full_ticks);
  }

  unsigned long long accepted_items;
  unsigned long long accepted_sectors;
  unsigned long long service_ticks;
  unsigned long long max_sectors_per_tick;
  unsigned long long width_limited_ticks;
  unsigned long long downstream_full_ticks;
  unsigned long long queue_full_ticks;
};

// The local and remote priority queues retain the legacy arbitration rule:
// the earlier ready cycle wins across queues, and a tie selects local. Within
// each queue, insertion order is deterministic FIFO for equal ready cycles.
template <typename T>
class rop_delay_output_queue {
 public:
  rop_delay_output_queue() : m_next_sequence(0) {}

  void push(T item, unsigned long long ready_cycle, bool remote) {
    entry value = {ready_cycle, m_next_sequence++, item};
    if (remote)
      m_remote.push(value);
    else
      m_local.push(value);
  }

  bool empty() const { return m_local.empty() && m_remote.empty(); }
  std::size_t size() const { return m_local.size() + m_remote.size(); }

  bool has_ready(unsigned long long cycle) const {
    return next_ready_queue(cycle) != NULL;
  }

  // Width one is deliberately the legacy compatibility mode: it accepts one
  // ready queue item per tick even if a non-sector cache represented that item
  // with more than one 32-byte sector. Widths above one require every item to
  // be exactly one sector, making the configured unit unambiguous.
  template <typename SectorCount, typename DownstreamFull, typename Accept>
  rop_delay_output_service_result service(unsigned long long cycle,
                                           unsigned width,
                                           SectorCount sector_count,
                                           DownstreamFull downstream_full,
                                           Accept accept) {
    assert(width > 0);
    rop_delay_output_service_result result;

    while (true) {
      queue_type *queue = next_ready_queue(cycle);
      if (queue == NULL) {
        result.reason = ROP_DELAY_OUTPUT_NO_READY_WORK;
        return result;
      }

      const unsigned sectors = sector_count(queue->top().item);
      assert(sectors > 0);
      if (width == 1) {
        if (result.accepted_items == 1) {
          result.reason = ROP_DELAY_OUTPUT_WIDTH_LIMITED;
          return result;
        }
      } else {
        assert(sectors == 1 &&
               "multi-issue ROP output requires 32-byte sector children");
        if (result.accepted_sectors + sectors > width) {
          result.reason = ROP_DELAY_OUTPUT_WIDTH_LIMITED;
          return result;
        }
      }

      if (downstream_full()) {
        result.reason = ROP_DELAY_OUTPUT_DOWNSTREAM_FULL;
        return result;
      }

      const T item = queue->top().item;
      queue->pop();
      accept(item);
      ++result.accepted_items;
      result.accepted_sectors += sectors;
    }
  }

 private:
  struct entry {
    unsigned long long ready_cycle;
    unsigned long long sequence;
    T item;
  };

  static bool earlier(const entry &lhs, const entry &rhs) {
    if (lhs.ready_cycle != rhs.ready_cycle)
      return lhs.ready_cycle < rhs.ready_cycle;
    return lhs.sequence < rhs.sequence;
  }

  struct later_first {
    bool operator()(const entry &lhs, const entry &rhs) const {
      return earlier(rhs, lhs);
    }
  };

  typedef std::priority_queue<entry, std::vector<entry>, later_first>
      queue_type;

  static bool ready(const queue_type &queue, unsigned long long cycle) {
    return !queue.empty() && cycle >= queue.top().ready_cycle;
  }

  queue_type *next_ready_queue(unsigned long long cycle) {
    const bool local_ready = ready(m_local, cycle);
    const bool remote_ready = ready(m_remote, cycle);
    if (!local_ready && !remote_ready) return NULL;
    if (local_ready && remote_ready)
      return m_local.top().ready_cycle <= m_remote.top().ready_cycle
                 ? &m_local
                 : &m_remote;
    return local_ready ? &m_local : &m_remote;
  }

  const queue_type *next_ready_queue(unsigned long long cycle) const {
    const bool local_ready = ready(m_local, cycle);
    const bool remote_ready = ready(m_remote, cycle);
    if (!local_ready && !remote_ready) return NULL;
    if (local_ready && remote_ready)
      return m_local.top().ready_cycle <= m_remote.top().ready_cycle
                 ? &m_local
                 : &m_remote;
    return local_ready ? &m_local : &m_remote;
  }

  queue_type m_local;
  queue_type m_remote;
  unsigned long long m_next_sequence;
};

#endif  // MEMORY_TRANSPORT_H
