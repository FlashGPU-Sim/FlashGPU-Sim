#ifndef GPGPU_SIM_DSM_FABRIC_H
#define GPGPU_SIM_DSM_FABRIC_H

#include <stdio.h>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "gpu_topology.h"
#include "transport.h"

// Intra-GPC DSM fabric: 32 B payload grants, request/response VCs sharing
// one physical-lane scheduler, per-SM shaper, GPCMMU hash, GX planes,
// CPC 6 slots → 4 lanes. PG'd slots never inject and never take eligibility.

enum class dsm_vc_t { request = 0, response = 1 };

enum class dsm_packet_class_t {
  read_command,
  read_data,
  write_data,
  write_ack,
  atomic_request,
  atomic_response,
  tma_data,
  mbarrier_request,
  mbarrier_completion
};

struct dsm_packet_t {
  unsigned packet_id = 0;
  unsigned transaction_id = 0;
  unsigned network_src_sm_id = 0;
  unsigned network_dst_sm_id = 0;
  unsigned transaction_requester_sm_id = 0;
  unsigned transaction_target_sm_id = 0;
  dsm_vc_t vc = dsm_vc_t::request;
  dsm_packet_class_t packet_class = dsm_packet_class_t::write_data;
  unsigned payload_bytes = 0;
  unsigned total_flits = 0;
  unsigned remaining_flits = 0;
  unsigned route_lane = 0;
  unsigned gx_plane = 0;
  uint64_t payload_address = 0;
  unsigned long long created_cycle = 0;
  unsigned long long injected_cycle = 0;
  unsigned long long tail_arrival_cycle = 0;
};

struct dsm_route_t {
  unsigned gx_plane = 0;
  unsigned lane = 0;
};

struct dsm_fabric_config_t {
  unsigned cpcs = 0;  // 0 → topology
  unsigned lanes_per_cpc = 4;
  unsigned gx_planes = 2;
  unsigned flit_payload_bytes = 32;
  unsigned shaper_period = 3;
  unsigned request_vc_flits = 64;
  unsigned response_vc_flits = 64;
  unsigned ejection_vc_flits = 64;
  unsigned route_seed = 0;
  unsigned response_priority_bound = 4;
  const char *shaper = "skip_mod";
  const char *shaper_index = "sm_id";
  const char *vc_arbiter = "bounded_response_priority";
};

struct dsm_fabric_stats_t {
  unsigned long long flits_granted = 0;
  unsigned long long payload_bytes_granted = 0;
  unsigned long long packets_injected = 0;
  unsigned long long packets_ejected = 0;
  unsigned long long eligibility_slots = 0;
  unsigned long long eligibility_used = 0;
  unsigned long long eligibility_wasted = 0;
  unsigned long long stall_inject = 0;
  unsigned long long stall_lane = 0;
  unsigned long long stall_ejection = 0;
  unsigned long long lane_conflicts = 0;
  unsigned long long flits_request = 0;
  unsigned long long flits_response = 0;
  unsigned occupancy_high_water_request = 0;
  unsigned occupancy_high_water_response = 0;
};

inline bool dsm_packet_is_control(dsm_packet_class_t c) {
  return c == dsm_packet_class_t::read_command ||
         c == dsm_packet_class_t::write_ack ||
         c == dsm_packet_class_t::mbarrier_request ||
         c == dsm_packet_class_t::mbarrier_completion;
}

inline dsm_vc_t dsm_vc_of(dsm_packet_class_t c) {
  switch (c) {
    case dsm_packet_class_t::read_data:
    case dsm_packet_class_t::write_ack:
    case dsm_packet_class_t::atomic_response:
    case dsm_packet_class_t::mbarrier_completion:
      return dsm_vc_t::response;
    default:
      return dsm_vc_t::request;
  }
}

inline unsigned dsm_payload_flits(dsm_packet_class_t c, unsigned bytes,
                                  unsigned flit_payload_bytes) {
  if (dsm_packet_is_control(c)) return 1;
  const unsigned f = flit_payload_bytes ? flit_payload_bytes : 32;
  return (bytes + f - 1) / f;
}

class dsm_fabric_t {
 public:
  dsm_fabric_t(const gpu_topology_t &topo, unsigned gpc,
               const dsm_fabric_config_t &cfg);

  bool can_inject(unsigned physical_source, dsm_vc_t vc,
                  unsigned flits) const;
  bool can_inject(unsigned physical_source, dsm_vc_t vc, unsigned dest,
                  unsigned flits) const;
  void inject(std::unique_ptr<dsm_packet_t> packet);
  const dsm_packet_t *top(unsigned physical_destination, dsm_vc_t vc) const;
  std::unique_ptr<dsm_packet_t> pop(unsigned physical_destination,
                                    dsm_vc_t vc);
  void cycle(unsigned long long cycle);
  bool busy() const;
  void display_state(FILE *fp) const;

  unsigned num_sms() const { return m_n; }
  unsigned gpc_id() const { return m_gpc; }
  unsigned flit_payload_bytes() const { return m_cfg.flit_payload_bytes; }
  unsigned gx_planes() const { return m_cfg.gx_planes; }
  unsigned lanes_per_cpc() const { return m_cfg.lanes_per_cpc; }
  unsigned num_routes() const {
    return m_cfg.gx_planes * m_cfg.lanes_per_cpc;
  }
  bool sm_eligible(unsigned local_sm, unsigned long long cycle) const;
  dsm_route_t hash_route(uint64_t addr, unsigned src, unsigned dst,
                         unsigned uid) const;
  unsigned occupancy_flits(unsigned src, dsm_vc_t vc, unsigned dst) const;
  unsigned credit_remaining(unsigned dst, dsm_vc_t vc) const;
  const dsm_fabric_stats_t &stats() const { return m_stats; }

 private:
  enum shaper_t { SHAPER_SKIP_MOD, SHAPER_FIXED_TDM, SHAPER_HARD_RATE_CAP };

  unsigned vci(dsm_vc_t v) const { return static_cast<unsigned>(v); }
  bool shaper_allows(unsigned sm, unsigned long long cycle) const;
  unsigned peek_dst(unsigned sm, unsigned vc) const;
  bool pick_head(unsigned sm, unsigned *vc, unsigned *dst) const;
  void note_hw(unsigned vc);

  dsm_fabric_config_t m_cfg;
  unsigned m_gpc = 0;
  unsigned m_n = 0;
  shaper_t m_shaper = SHAPER_SKIP_MOD;
  bool m_index_is_slot = false;
  bool m_rsp_priority = true;
  unsigned long long m_now = 0;
  unsigned m_next_pkt_id = 0;

  std::vector<unsigned> m_cpc;
  std::vector<unsigned> m_slot;
  std::vector<unsigned> m_hrc_acc;
  std::vector<unsigned> m_consec_rsp;
  std::vector<unsigned> m_dst_next[2];
  std::vector<unsigned char> m_last_eligible;

  std::vector<bounded_voq_t<dsm_packet_t>> m_in[2];
  std::vector<std::deque<dsm_packet_t>> m_ej[2];
  flit_credit_counters_t m_credit[2];
  std::vector<round_robin_arbiter_t> m_route_rr;
  std::vector<round_robin_arbiter_t> m_cpc_rr;
  dsm_fabric_stats_t m_stats;
};

#endif
