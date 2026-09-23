#include "dsm_fabric.h"

#include <assert.h>
#include <string.h>

static bool streq(const char *a, const char *b) {
  if (!a) return false;
  return strcmp(a, b) == 0;
}

dsm_fabric_t::dsm_fabric_t(const gpu_topology_t &topo, unsigned gpc,
                           const dsm_fabric_config_t &cfg)
    : m_cfg(cfg), m_gpc(gpc) {
  if (!m_cfg.cpcs) m_cfg.cpcs = topo.cpcs_per_gpc();
  if (!m_cfg.lanes_per_cpc) m_cfg.lanes_per_cpc = 4;
  if (!m_cfg.gx_planes) m_cfg.gx_planes = 2;
  if (!m_cfg.flit_payload_bytes) m_cfg.flit_payload_bytes = 32;
  if (!m_cfg.shaper_period) m_cfg.shaper_period = 3;
  if (!m_cfg.request_vc_flits) m_cfg.request_vc_flits = 64;
  if (!m_cfg.response_vc_flits) m_cfg.response_vc_flits = 64;
  if (!m_cfg.ejection_vc_flits) m_cfg.ejection_vc_flits = 64;
  if (!m_cfg.response_priority_bound) m_cfg.response_priority_bound = 4;

  if (streq(m_cfg.shaper, "fixed_tdm"))
    m_shaper = SHAPER_FIXED_TDM;
  else if (streq(m_cfg.shaper, "hard_rate_cap"))
    m_shaper = SHAPER_HARD_RATE_CAP;
  else
    m_shaper = SHAPER_SKIP_MOD;

  m_index_is_slot = streq(m_cfg.shaper_index, "cpc_slot");
  m_rsp_priority = !m_cfg.vc_arbiter ||
                   streq(m_cfg.vc_arbiter, "bounded_response_priority");

  m_n = topo.num_sms_in_gpc(gpc);
  assert(m_n > 0);
  assert(m_n <= 64);
  m_cpc.resize(m_n);
  m_slot.resize(m_n);
  for (unsigned local = 0; local < m_n; local++) {
    const sm_location_t loc = topo.locate_sm(topo.sm_id_at(gpc, local));
    m_cpc[local] = loc.cpc_id;
    m_slot[local] = loc.cpc_slot;
  }
  m_hrc_acc.assign(m_n, 0);
  m_consec_rsp.assign(m_n, 0);
  m_last_eligible.assign(m_n, 0);
  for (unsigned v = 0; v < 2; v++) {
    m_dst_next[v].assign(m_n, 0);
    m_in[v].resize(m_n);
    const unsigned depth =
        v == 0 ? m_cfg.request_vc_flits : m_cfg.response_vc_flits;
    for (unsigned s = 0; s < m_n; s++) m_in[v][s].init(m_n, depth);
    m_ej[v].assign(m_n, {});
    m_credit[v].init(m_n, m_cfg.ejection_vc_flits);
  }

  const unsigned n_routes = num_routes();
  assert(n_routes > 0);
  assert(n_routes <= 64);
  m_route_rr.resize((size_t)m_cfg.cpcs * n_routes);
  for (auto &rr : m_route_rr) rr.init(m_n);
  m_cpc_rr.resize(m_cfg.cpcs);
  for (auto &rr : m_cpc_rr) rr.init(n_routes);
}

bool dsm_fabric_t::can_inject(unsigned src, dsm_vc_t vc,
                              unsigned flits) const {
  if (src >= m_n) return false;
  for (unsigned d = 0; d < m_n; d++) {
    if (can_inject(src, vc, d, flits)) return true;
  }
  return false;
}

bool dsm_fabric_t::can_inject(unsigned src, dsm_vc_t vc, unsigned dest,
                              unsigned flits) const {
  if (src >= m_n || dest >= m_n) return false;
  return m_in[vci(vc)][src].can_push(dest, flits);
}

void dsm_fabric_t::inject(std::unique_ptr<dsm_packet_t> packet) {
  if (!packet) return;
  dsm_packet_t pkt = *packet;
  const unsigned src = pkt.network_src_sm_id;
  const unsigned dst = pkt.network_dst_sm_id;
  if (!pkt.total_flits) {
    pkt.total_flits = dsm_payload_flits(
        pkt.packet_class, pkt.payload_bytes, m_cfg.flit_payload_bytes);
    pkt.remaining_flits = pkt.total_flits;
  }
  if (!pkt.remaining_flits) pkt.remaining_flits = pkt.total_flits;
  if (!can_inject(src, pkt.vc, dst, pkt.remaining_flits)) {
    m_stats.stall_inject++;
    return;
  }
  if (!pkt.packet_id) pkt.packet_id = ++m_next_pkt_id;
  if (!pkt.created_cycle) pkt.created_cycle = m_now;
  pkt.injected_cycle = m_now;
  const dsm_route_t rt =
      hash_route(pkt.payload_address, src, dst, pkt.packet_id);
  pkt.gx_plane = rt.gx_plane;
  pkt.route_lane = rt.lane;
  const unsigned v = vci(pkt.vc);
  const bool data_queued = m_in[v][src].contains(
      dst, [](const dsm_packet_t &queued) {
        return queued.packet_class != dsm_packet_class_t::read_command;
      });
  const bool pushed =
      pkt.packet_class == dsm_packet_class_t::read_command && data_queued
          ? m_in[v][src].push_front(dst, pkt)
          : m_in[v][src].push(dst, pkt);
  assert(pushed);
  m_stats.packets_injected++;
  note_hw(v);
}

bool dsm_fabric_t::ejection_visible(const dsm_packet_t &p) const {
  unsigned long long vis = p.tail_arrival_cycle;
  unsigned floor = m_cfg.base_latency_cycles;
  if (p.packet_class == dsm_packet_class_t::write_data &&
      m_cfg.store_visibility_latency_cycles)
    floor = m_cfg.store_visibility_latency_cycles;
  if (floor) {
    const unsigned long long floor_at =
        p.injected_cycle + floor;
    if (floor_at > vis) vis = floor_at;
  }
  return vis <= m_now;
}

const dsm_packet_t *dsm_fabric_t::top(unsigned dst, dsm_vc_t vc) const {
  if (dst >= m_n) return nullptr;
  const unsigned v = vci(vc);
  if (m_ej[v][dst].empty()) return nullptr;
  const dsm_packet_t &p = m_ej[v][dst].front();
  if (!ejection_visible(p)) return nullptr;
  return &p;
}

std::unique_ptr<dsm_packet_t> dsm_fabric_t::pop(unsigned dst, dsm_vc_t vc) {
  if (!top(dst, vc)) return nullptr;
  const unsigned v = vci(vc);
  dsm_packet_t pkt = m_ej[v][dst].front();
  m_ej[v][dst].pop_front();
  m_credit[v].give(dst, pkt.ejection_reserved_flits);
  m_stats.packets_ejected++;
  return std::make_unique<dsm_packet_t>(pkt);
}

bool dsm_fabric_t::busy() const {
  for (unsigned v = 0; v < 2; v++) {
    for (unsigned s = 0; s < m_n; s++) {
      if (m_in[v][s].occupancy_flits()) return true;
      if (!m_ej[v][s].empty()) return true;
    }
  }
  return false;
}

unsigned dsm_fabric_t::occupancy_flits(unsigned src, dsm_vc_t vc,
                                       unsigned dst) const {
  if (src >= m_n || dst >= m_n) return 0;
  return m_in[vci(vc)][src].occupancy_flits(dst);
}

unsigned dsm_fabric_t::credit_remaining(unsigned dst, dsm_vc_t vc) const {
  if (dst >= m_n) return 0;
  return m_credit[vci(vc)].remaining(dst);
}

bool dsm_fabric_t::sm_eligible(unsigned sm, unsigned long long cycle) const {
  if (sm >= m_n) return false;
  if (m_shaper == SHAPER_HARD_RATE_CAP) {
    if (cycle == m_now && sm < m_last_eligible.size())
      return m_last_eligible[sm] != 0;
  }
  return shaper_allows(sm, cycle);
}

dsm_route_t dsm_fabric_t::hash_route(uint64_t addr, unsigned src, unsigned dst,
                                     unsigned uid) const {
  const unsigned n_routes = num_routes();
  // Cache-line / 32 B interleave across gx_planes * lanes_per_cpc, mixed
  // with src/dst. uid is coarse so a stream of packets with the same
  // address stays on one route (stride camping).
  uint64_t idx = addr / m_cfg.flit_payload_bytes;
  idx += (uint64_t)src * 2;
  idx += (uint64_t)dst * 3;
  idx += uid / 16;
  idx += m_cfg.route_seed;
  const unsigned route = n_routes ? (unsigned)(idx % n_routes) : 0;
  dsm_route_t r;
  r.gx_plane = m_cfg.lanes_per_cpc ? route / m_cfg.lanes_per_cpc : 0;
  r.lane = m_cfg.lanes_per_cpc ? route % m_cfg.lanes_per_cpc : 0;
  return r;
}

bool dsm_fabric_t::shaper_allows(unsigned sm, unsigned long long cycle) const {
  const unsigned period = m_cfg.shaper_period ? m_cfg.shaper_period : 3;
  const unsigned slot = m_slot[sm];
  const unsigned idx = m_index_is_slot ? slot : sm;
  if (m_shaper == SHAPER_SKIP_MOD)
    return (cycle % period) != (idx % period);
  if (m_shaper == SHAPER_FIXED_TDM) {
    static const unsigned k_phase[3] = {
        (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3),
        (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5),
        (1u << 0) | (1u << 1) | (1u << 4) | (1u << 5),
    };
    return (k_phase[cycle % 3] >> slot) & 1u;
  }
  return false;
}

unsigned dsm_fabric_t::peek_dst(unsigned sm, unsigned vc) const {
  const unsigned start = m_dst_next[vc][sm];
  for (unsigned k = 0; k < m_n; k++) {
    const unsigned d = (start + k) % m_n;
    if (!m_in[vc][sm].empty(d)) return d;
  }
  return ~0u;
}

bool dsm_fabric_t::pick_head(unsigned sm, unsigned *vc, unsigned *dst) const {
  const unsigned d_req = peek_dst(sm, 0);
  const unsigned d_rsp = peek_dst(sm, 1);
  const bool req = d_req != ~0u;
  const bool rsp = d_rsp != ~0u;
  if (!req && !rsp) return false;
  unsigned pick;
  if (m_rsp_priority) {
    if (rsp && m_consec_rsp[sm] < m_cfg.response_priority_bound)
      pick = 1;
    else if (req)
      pick = 0;
    else
      pick = 1;
  } else {
    pick = req ? 0 : 1;
  }
  *vc = pick;
  *dst = pick ? d_rsp : d_req;
  return true;
}

void dsm_fabric_t::note_hw(unsigned vc) {
  unsigned occ = 0;
  for (unsigned s = 0; s < m_n; s++) occ += m_in[vc][s].occupancy_flits();
  unsigned &hw = vc ? m_stats.occupancy_high_water_response
                    : m_stats.occupancy_high_water_request;
  if (occ > hw) hw = occ;
}

void dsm_fabric_t::cycle(unsigned long long cycle) {
  m_now = cycle;
  const unsigned n_routes = num_routes();
  std::vector<unsigned char> eligible(m_n, 0);
  for (unsigned sm = 0; sm < m_n; sm++) {
    bool allow = false;
    if (m_shaper == SHAPER_HARD_RATE_CAP) {
      m_hrc_acc[sm] += m_cfg.lanes_per_cpc;
      if (m_hrc_acc[sm] >= (unsigned)k_sm_slots_per_cpc) {
        allow = true;
        m_hrc_acc[sm] -= (unsigned)k_sm_slots_per_cpc;
      }
    } else {
      allow = shaper_allows(sm, cycle);
    }
    eligible[sm] = allow ? 1 : 0;
    m_last_eligible[sm] = eligible[sm];
    if (allow) m_stats.eligibility_slots++;
  }

  struct req_t {
    unsigned sm, vc, dst, cpc, route;
  };
  std::vector<req_t> reqs;
  std::vector<int> sm_req(m_n, -1);
  for (unsigned sm = 0; sm < m_n; sm++) {
    if (!eligible[sm]) continue;
    unsigned vc = 0, dst = 0;
    if (!pick_head(sm, &vc, &dst)) continue;
    const dsm_packet_t *p = m_in[vc][sm].front(dst);
    req_t r;
    r.sm = sm;
    r.vc = vc;
    r.dst = dst;
    r.cpc = m_cpc[sm];
    const unsigned granted_flits = p->total_flits - p->remaining_flits;
    const uint64_t flit_addr =
        p->payload_address +
        (uint64_t)granted_flits * m_cfg.flit_payload_bytes;
    const dsm_route_t route =
        hash_route(flit_addr, sm, dst, p->packet_id);
    r.route = route.gx_plane * m_cfg.lanes_per_cpc + route.lane;
    sm_req[sm] = (int)reqs.size();
    reqs.push_back(r);
  }

  std::vector<unsigned char> granted(m_n, 0);
  bool ready[64];
  for (unsigned cpc = 0; cpc < m_cfg.cpcs; cpc++) {
    std::vector<int> route_winner(n_routes, -1);
    unsigned n_winners = 0;
    for (unsigned route = 0; route < n_routes; route++) {
      memset(ready, 0, sizeof(ready));
      unsigned n_ready = 0;
      for (const req_t &r : reqs) {
        if (r.cpc == cpc && r.route == route) {
          ready[r.sm] = true;
          n_ready++;
        }
      }
      if (!n_ready) continue;
      if (n_ready > 1) m_stats.lane_conflicts++;
      const unsigned win =
          m_route_rr[(size_t)cpc * n_routes + route].grant(ready, m_n);
      if (win != ~0u) {
        route_winner[route] = (int)win;
        n_winners++;
      }
    }
    memset(ready, 0, sizeof(ready));
    for (unsigned route = 0; route < n_routes; route++) {
      if (route_winner[route] >= 0) ready[route] = true;
    }
    unsigned taken = 0;
    const unsigned cap = m_cfg.lanes_per_cpc;
    for (unsigned g = 0; g < cap; g++) {
      if (taken >= cap) break;
      const unsigned route = m_cpc_rr[cpc].grant(ready, n_routes);
      if (route == ~0u) break;
      ready[route] = false;
      const int sm = route_winner[route];
      if (sm < 0) continue;
      const req_t &r = reqs[sm_req[sm]];
      const dsm_packet_t *selected = m_in[r.vc][r.sm].front(r.dst);
      std::vector<unsigned> multicast_dsts;
      multicast_dsts.push_back(r.dst);
      if (selected->multicast_group) {
        for (unsigned dst = 0; dst < m_n; dst++) {
          if (dst == r.dst) continue;
          const dsm_packet_t *candidate = m_in[r.vc][r.sm].front(dst);
          if (candidate &&
              candidate->multicast_group == selected->multicast_group)
            multicast_dsts.push_back(dst);
        }
      }
      bool has_ejection_space = true;
      for (unsigned dst : multicast_dsts) {
        const dsm_packet_t *head = m_in[r.vc][r.sm].front(dst);
        const unsigned reserve_limit =
            std::min(head->total_flits, m_credit[r.vc].depth());
        if (head->ejection_reserved_flits < reserve_limit &&
            !m_credit[r.vc].has(dst, 1)) {
          has_ejection_space = false;
          break;
        }
      }
      if (!has_ejection_space) {
        m_stats.stall_ejection++;
        continue;
      }
      for (unsigned dst : multicast_dsts) {
        dsm_packet_t *head = m_in[r.vc][r.sm].front_mutable(dst);
        const unsigned reserve_limit =
            std::min(head->total_flits, m_credit[r.vc].depth());
        if (head->ejection_reserved_flits < reserve_limit) {
          const bool reserved = m_credit[r.vc].take(dst, 1);
          assert(reserved);
          head->ejection_reserved_flits++;
        }
      }
      const bool control = dsm_packet_is_control(selected->packet_class);
      const dsm_packet_class_t selected_class = selected->packet_class;
      const unsigned selected_payload = selected->payload_bytes;
      unsigned packet_cap = 1;
      if (multicast_dsts.size() == 1 && selected->total_flits == 1 &&
          selected_payload > 0) {
        if (selected_class == dsm_packet_class_t::read_command)
          packet_cap = std::max(1u, 128u / selected_payload);
        else if (!control && selected_payload < m_cfg.flit_payload_bytes)
          packet_cap = m_cfg.flit_payload_bytes / selected_payload;
      }
      unsigned packets = 0;
      while (packets < packet_cap) {
        bool any_tail = false;
        for (unsigned dst : multicast_dsts) {
          if (m_in[r.vc][r.sm].empty(dst)) continue;
          dsm_packet_t done{};
          const bool tail = m_in[r.vc][r.sm].grant_flit(dst, cycle, &done);
          if (tail) {
            done.tail_arrival_cycle = cycle;
            m_ej[r.vc][dst].push_back(done);
            any_tail = true;
          }
        }
        packets++;
        if (!any_tail || packets >= packet_cap) break;
        dsm_packet_t *next = m_in[r.vc][r.sm].front_mutable(r.dst);
        if (!next || next->packet_class != selected_class ||
            next->payload_bytes != selected_payload || next->total_flits != 1)
          break;
        const unsigned reserve_limit =
            std::min(next->total_flits, m_credit[r.vc].depth());
        if (next->ejection_reserved_flits < reserve_limit) {
          if (!m_credit[r.vc].has(r.dst, 1)) break;
          const bool reserved = m_credit[r.vc].take(r.dst, 1);
          assert(reserved);
          next->ejection_reserved_flits++;
        }
      }
      m_stats.flits_granted++;
      if (r.vc)
        m_stats.flits_response++;
      else
        m_stats.flits_request++;
      if (!control)
        m_stats.payload_bytes_granted += m_cfg.flit_payload_bytes;
      m_dst_next[r.vc][r.sm] = (r.dst + 1) % m_n;
      if (r.vc)
        m_consec_rsp[r.sm]++;
      else
        m_consec_rsp[r.sm] = 0;
      granted[r.sm] = 1;
      taken++;
    }
    if (n_winners > taken) m_stats.stall_lane += n_winners - taken;
  }

  for (unsigned sm = 0; sm < m_n; sm++) {
    if (!eligible[sm]) continue;
    if (granted[sm])
      m_stats.eligibility_used++;
    else
      m_stats.eligibility_wasted++;
  }
  note_hw(0);
  note_hw(1);
}

void dsm_fabric_t::display_state(FILE *fp) const {
  if (!fp) return;
  fprintf(fp,
          "dsm_fabric gpc=%u sms=%u flit_payload_bytes=%u gx_planes=%u "
          "lanes_per_cpc=%u routes=%u\n",
          m_gpc, m_n, m_cfg.flit_payload_bytes, m_cfg.gx_planes,
          m_cfg.lanes_per_cpc, num_routes());
  fprintf(fp, "eligibility used=%llu wasted=%llu slots=%llu\n",
          m_stats.eligibility_used, m_stats.eligibility_wasted,
          m_stats.eligibility_slots);
  fprintf(fp,
          "flits granted=%llu request=%llu response=%llu payload_bytes=%llu\n",
          m_stats.flits_granted, m_stats.flits_request, m_stats.flits_response,
          m_stats.payload_bytes_granted);
  fprintf(fp,
          "stall inject=%llu lane=%llu ejection=%llu lane_conflicts=%llu "
          "packets in=%llu out=%llu\n",
          m_stats.stall_inject, m_stats.stall_lane, m_stats.stall_ejection,
          m_stats.lane_conflicts, m_stats.packets_injected,
          m_stats.packets_ejected);
}
