#ifndef GPGPU_SIM_TRANSPORT_H
#define GPGPU_SIM_TRANSPORT_H

#include <assert.h>

#include <algorithm>
#include <deque>
#include <vector>

// Generic flit-level interconnect primitives. Occupancy is payload flits.
// No request/reply subnets and no shader/memory endpoint roles.

struct transport_packet_metadata_t {
  unsigned packet_id = 0;
  unsigned src = 0;
  unsigned dst = 0;
  unsigned payload_bytes = 0;
  unsigned total_flits = 0;
  unsigned remaining_flits = 0;
  unsigned long long created_cycle = 0;
};

struct interconnect_stats_t {
  unsigned long long flits_moved = 0;
  unsigned long long packets_in = 0;
  unsigned long long packets_out = 0;
  unsigned long long stall_events = 0;
  unsigned occupancy_high_water = 0;
  unsigned long long latency_sum = 0;
  unsigned long long latency_samples = 0;

  void note_occupancy(unsigned flits) {
    if (flits > occupancy_high_water) occupancy_high_water = flits;
  }
};

// One bounded queue per destination. Occupancy is remaining payload flits.
// Packet type must provide remaining_flits and created_cycle.
template <typename Pkt = transport_packet_metadata_t>
struct bounded_voq_t {
  void init(unsigned n_dst, unsigned flit_limit_per_dst) {
    m_limit = flit_limit_per_dst;
    m_q.assign(n_dst, {});
    m_occ.assign(n_dst, 0);
  }
  unsigned n_dst() const { return (unsigned)m_q.size(); }
  unsigned flit_limit() const { return m_limit; }
  unsigned occupancy_flits(unsigned dst) const { return m_occ[dst]; }
  unsigned occupancy_flits() const {
    unsigned sum = 0;
    for (unsigned o : m_occ) sum += o;
    return sum;
  }
  bool empty(unsigned dst) const { return m_q[dst].empty(); }
  template <typename Predicate>
  bool contains(unsigned dst, Predicate predicate) const {
    return std::find_if(m_q[dst].begin(), m_q[dst].end(), predicate) !=
           m_q[dst].end();
  }
  bool can_push(unsigned dst, unsigned flits) const {
    if (flits > m_limit) return m_q[dst].empty() && m_occ[dst] == 0;
    return m_occ[dst] + flits <= m_limit;
  }
  bool push(unsigned dst, Pkt pkt) {
    assert(dst < m_q.size());
    const unsigned flits = pkt.remaining_flits;
    if (!can_push(dst, flits)) {
      m_stats.stall_events++;
      return false;
    }
    m_q[dst].push_back(pkt);
    m_occ[dst] += std::min(flits, m_limit);
    m_stats.packets_in++;
    m_stats.note_occupancy(occupancy_flits());
    return true;
  }
  template <typename Predicate>
  bool push_before(unsigned dst, Pkt pkt, Predicate before) {
    assert(dst < m_q.size());
    const unsigned flits = pkt.remaining_flits;
    if (!can_push(dst, flits)) {
      m_stats.stall_events++;
      return false;
    }
    auto pos = std::find_if(m_q[dst].begin(), m_q[dst].end(), before);
    m_q[dst].insert(pos, pkt);
    m_occ[dst] += std::min(flits, m_limit);
    m_stats.packets_in++;
    m_stats.note_occupancy(occupancy_flits());
    return true;
  }
  bool push_front(unsigned dst, Pkt pkt) {
    return push_before(dst, pkt, [](const Pkt &) { return true; });
  }
  const Pkt *front(unsigned dst) const {
    if (m_q[dst].empty()) return nullptr;
    return &m_q[dst].front();
  }
  Pkt *front_mutable(unsigned dst) {
    if (m_q[dst].empty()) return nullptr;
    return &m_q[dst].front();
  }
  // Consume one payload flit from dest's head. Returns true if that packet ends.
  bool grant_flit(unsigned dst, unsigned long long now, Pkt *completed) {
    assert(dst < m_q.size());
    assert(!m_q[dst].empty());
    assert(m_occ[dst] > 0);
    Pkt &p = m_q[dst].front();
    assert(p.remaining_flits > 0);
    const unsigned before = p.remaining_flits;
    p.remaining_flits--;
    if (before <= m_limit) m_occ[dst]--;
    m_stats.flits_moved++;
    if (p.remaining_flits > 0) return false;
    if (completed) *completed = p;
    m_stats.packets_out++;
    m_stats.latency_sum += now - p.created_cycle;
    m_stats.latency_samples++;
    m_q[dst].pop_front();
    return true;
  }
  const interconnect_stats_t &stats() const { return m_stats; }

 private:
  unsigned m_limit = 0;
  std::vector<std::deque<Pkt>> m_q;
  std::vector<unsigned> m_occ;
  interconnect_stats_t m_stats;
};

// Independent flit-credit pool per queue. take() never touches another queue.
struct flit_credit_counters_t {
  void init(unsigned n_queues, unsigned depth) {
    m_depth = depth;
    m_free.assign(n_queues, depth);
  }
  unsigned remaining(unsigned q) const { return m_free[q]; }
  unsigned depth() const { return m_depth; }
  bool has(unsigned q, unsigned flits) const { return m_free[q] >= flits; }
  bool take(unsigned q, unsigned flits) {
    if (m_free[q] < flits) return false;
    m_free[q] -= flits;
    return true;
  }
  void give(unsigned q, unsigned flits) {
    m_free[q] += flits;
    assert(m_free[q] <= m_depth);
  }

 private:
  unsigned m_depth = 0;
  std::vector<unsigned> m_free;
};

struct round_robin_arbiter_t {
  void init(unsigned n_inputs) {
    m_n = n_inputs;
    m_next = 0;
  }
  // Returns the granted input, or ~0u if none is ready.
  unsigned grant(const bool *ready, unsigned n) {
    assert(n == m_n);
    for (unsigned k = 0; k < m_n; k++) {
      const unsigned i = (m_next + k) % m_n;
      if (ready[i]) {
        m_next = (i + 1) % m_n;
        return i;
      }
    }
    return ~0u;
  }

 private:
  unsigned m_n = 0;
  unsigned m_next = 0;
};

// Downstream ejection buffer. Occupancy is payload flits.
struct interconnect_sink_t {
  void init(unsigned flit_limit) {
    m_limit = flit_limit;
    m_occ = 0;
  }
  unsigned occupancy_flits() const { return m_occ; }
  unsigned free_flits() const { return m_limit - m_occ; }
  bool can_accept(unsigned flits) const { return m_occ + flits <= m_limit; }
  bool push(unsigned flits) {
    if (!can_accept(flits)) return false;
    m_occ += flits;
    return true;
  }
  void pop(unsigned flits) {
    assert(m_occ >= flits);
    m_occ -= flits;
  }

 private:
  unsigned m_limit = 0;
  unsigned m_occ = 0;
};

#endif
