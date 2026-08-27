#ifndef GPGPU_SIM_DSM_ENDPOINT_H
#define GPGPU_SIM_DSM_ENDPOINT_H

#include <stdio.h>
#include <deque>
#include <unordered_map>
#include <vector>

#include "dsm_fabric.h"

// Per-enabled-SM DSM endpoint protocol sitting outside the lane arbiter.
// Outstanding-tx cap is not a VC/link credit. ACK coalescing is not in
// the physical scheduler. SRAM service is a stub until the load/store
// scoreboard path is wired.

struct dsm_endpoint_config_t {
  unsigned max_outstanding_per_sm = 16;
  unsigned ack_coalesce_threshold = 4;
  unsigned ack_timeout_cycles = 64;
};

struct dsm_tx_t {
  unsigned id = 0;
  unsigned requester = 0;
  unsigned target = 0;
  dsm_packet_class_t cls = dsm_packet_class_t::write_data;
  unsigned remaining_responses = 1;
  unsigned payload_bytes = 0;
  uint64_t addr = 0;
  unsigned long long created_cycle = 0;
};

struct dsm_endpoint_stats_t {
  unsigned long long store_packets = 0;
  unsigned long long ack_packets = 0;
  unsigned long long ack_completions = 0;
  unsigned long long load_commands = 0;
  unsigned long long load_data_packets = 0;
  unsigned long long timeout_flushes = 0;
  unsigned long long threshold_flushes = 0;
  unsigned long long idle_flushes = 0;
  unsigned long long sram_store_bytes = 0;
  unsigned long long sram_load_bytes = 0;
  unsigned outstanding_high_water = 0;
};

class dsm_endpoint_protocol_t {
 public:
  dsm_endpoint_protocol_t(dsm_fabric_t *fabric,
                          const dsm_endpoint_config_t &cfg);

  bool can_store(unsigned src, unsigned dst, unsigned bytes) const;
  bool can_load(unsigned src, unsigned dst, unsigned bytes) const;
  bool issue_store(unsigned src, unsigned dst, unsigned bytes, uint64_t addr);
  bool issue_load(unsigned src, unsigned dst, unsigned bytes, uint64_t addr);

  void cycle(unsigned long long now);
  bool busy() const;
  void display_state(FILE *fp) const;

  unsigned outstanding(unsigned sm) const;
  unsigned ack_debt(unsigned sm) const;
  unsigned ack_debt(unsigned sm, unsigned requester) const;
  const dsm_endpoint_stats_t &stats() const { return m_stats; }
  double coalescing_ratio() const;
  const std::unordered_map<unsigned, dsm_tx_t> &tx_table(unsigned sm) const {
    return m_tx[sm];
  }

 private:
  struct pending_t {
    dsm_packet_class_t cls;
    unsigned src, dst, txid, bytes, count;
    uint64_t addr;
  };

  bool window_open(unsigned sm) const;
  unsigned flits_of(dsm_packet_class_t cls, unsigned bytes) const;
  void note_outstanding(unsigned sm);
  void sram_store(unsigned sm, unsigned bytes);
  void sram_load(unsigned sm, unsigned bytes);
  void accrue_ack(unsigned target, unsigned requester);
  void complete_stores(unsigned requester, unsigned target, unsigned count);
  void complete_load(unsigned requester, unsigned txid);
  void harvest();
  void try_send_pending();
  void try_flush_acks();
  bool inject_response(const pending_t &p);
  unsigned response_occ(unsigned sm) const;
  unsigned in_flight_writes(unsigned src, unsigned dst) const;

  dsm_fabric_t *m_fab;
  dsm_endpoint_config_t m_cfg;
  unsigned m_n = 0;
  unsigned long long m_now = 0;
  unsigned m_next_tx = 0;
  std::vector<std::unordered_map<unsigned, dsm_tx_t>> m_tx;
  std::vector<std::vector<std::deque<unsigned>>> m_store_q;
  std::vector<std::vector<unsigned>> m_acks_owed;
  std::vector<std::vector<unsigned long long>> m_oldest_ack;
  std::vector<std::vector<unsigned long long>> m_last_write;
  std::deque<pending_t> m_pending;
  dsm_endpoint_stats_t m_stats;
};

#endif
