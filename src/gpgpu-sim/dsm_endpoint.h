#ifndef GPGPU_SIM_DSM_ENDPOINT_H
#define GPGPU_SIM_DSM_ENDPOINT_H

#include <stdio.h>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#include "dsm_fabric.h"

// Per-enabled-SM DSM endpoint protocol sitting outside the lane arbiter.
// Outstanding-tx cap is not a VC/link credit. ACK coalescing is not in
// the physical scheduler. SRAM hooks are optional (unit tests leave them
// null and only count bytes).

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
  unsigned cta_slot = 0;
  unsigned cta_gen = 0;
  bool applied = false;
};

struct dsm_endpoint_stats_t {
  unsigned long long store_packets = 0;
  unsigned long long ack_packets = 0;
  unsigned long long ack_completions = 0;
  unsigned long long load_commands = 0;
  unsigned long long load_data_packets = 0;
  unsigned long long atom_requests = 0;
  unsigned long long atom_responses = 0;
  unsigned long long timeout_flushes = 0;
  unsigned long long threshold_flushes = 0;
  unsigned long long idle_flushes = 0;
  unsigned long long sram_store_bytes = 0;
  unsigned long long sram_load_bytes = 0;
  unsigned outstanding_high_water = 0;
};

class dsm_endpoint_protocol_t {
 public:
  using sram_fn = void (*)(void *ctx, unsigned local_sm, unsigned cta,
                           uint64_t addr, uint8_t *bytes, unsigned n);
  using tx_done_fn = void (*)(void *ctx, unsigned txid);

  dsm_endpoint_protocol_t(dsm_fabric_t *fabric,
                          const dsm_endpoint_config_t &cfg);

  void set_sram(void *ctx, sram_fn write, sram_fn read) {
    m_sram_ctx = ctx;
    m_sram_write = write;
    m_sram_read = read;
  }
  void set_on_tx_done(void *ctx, tx_done_fn fn) {
    m_done_ctx = ctx;
    m_on_tx_done = fn;
  }
  void set_on_load_data(void *ctx, void (*fn)(void *ctx, unsigned txid,
                                              const uint8_t *p, unsigned n)) {
    m_load_ctx = ctx;
    m_on_load_data = fn;
  }

  bool can_store(unsigned src, unsigned dst, unsigned bytes) const;
  bool can_load(unsigned src, unsigned dst, unsigned bytes) const;
  bool can_atom(unsigned src, unsigned dst, unsigned bytes) const;
  bool issue_store(unsigned src, unsigned dst, unsigned bytes, uint64_t addr);
  bool issue_load(unsigned src, unsigned dst, unsigned bytes, uint64_t addr);
  bool issue_store(unsigned src, unsigned dst, unsigned bytes, uint64_t addr,
                   unsigned cta_slot, unsigned cta_gen, const void *data);
  bool issue_load(unsigned src, unsigned dst, unsigned bytes, uint64_t addr,
                  unsigned cta_slot, unsigned cta_gen);
  bool issue_atom(unsigned src, unsigned dst, unsigned bytes, uint64_t addr,
                  unsigned cta_slot, unsigned cta_gen, const void *addend);

  void cycle(unsigned long long now);
  bool busy() const;
  void display_state(FILE *fp) const;

  unsigned outstanding(unsigned sm) const;
  unsigned ack_debt(unsigned sm) const;
  unsigned ack_debt(unsigned sm, unsigned requester) const;
  unsigned last_txid() const { return m_last_txid; }
  bool write_unapplied(unsigned src, unsigned dst) const;
  void bump_cta_gen(unsigned local_sm, unsigned cta);
  unsigned cta_gen(unsigned local_sm, unsigned cta) const;
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
    unsigned cta_slot, cta_gen;
    std::vector<uint8_t> payload;
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
  bool issue_req(unsigned src, unsigned dst, unsigned bytes, uint64_t addr,
                 unsigned cta_slot, unsigned cta_gen, dsm_packet_class_t cls,
                 const void *data);

  dsm_fabric_t *m_fab;
  dsm_endpoint_config_t m_cfg;
  unsigned m_n = 0;
  unsigned long long m_now = 0;
  unsigned m_next_tx = 0;
  unsigned m_last_txid = 0;
  std::vector<std::unordered_map<unsigned, dsm_tx_t>> m_tx;
  std::vector<std::vector<std::deque<unsigned>>> m_store_q;
  std::vector<std::vector<unsigned>> m_acks_owed;
  std::vector<std::vector<unsigned long long>> m_oldest_ack;
  std::vector<std::vector<unsigned long long>> m_last_write;
  std::vector<std::vector<unsigned>> m_cta_gen;
  std::deque<pending_t> m_pending;
  std::unordered_map<unsigned, std::vector<uint8_t>> m_data;
  dsm_endpoint_stats_t m_stats;
  void *m_sram_ctx = nullptr;
  sram_fn m_sram_write = nullptr;
  sram_fn m_sram_read = nullptr;
  void *m_done_ctx = nullptr;
  tx_done_fn m_on_tx_done = nullptr;
  void *m_load_ctx = nullptr;
  void (*m_on_load_data)(void *, unsigned, const uint8_t *, unsigned) = nullptr;
};

#endif
