// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MEM_TRANSPORT_BUDGET_H
#define MEM_TRANSPORT_BUDGET_H

#include <assert.h>
#include <limits.h>
#include <stdio.h>

#include <algorithm>

#include "mem_fetch.h"

// Keep this header limited to layer-independent sector accounting, per-tick
// service budgets, and their shared statistics. Pipeline-owned issue,
// retirement, and delay queues belong with the owning shader/L2 subsystem.

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

#endif  // MEM_TRANSPORT_BUDGET_H
