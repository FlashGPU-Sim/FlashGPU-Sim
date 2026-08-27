#include "dsm_endpoint.h"

#include <assert.h>
#include <string.h>
#include <memory>

dsm_endpoint_protocol_t::dsm_endpoint_protocol_t(
    dsm_fabric_t *fabric, const dsm_endpoint_config_t &cfg)
    : m_fab(fabric), m_cfg(cfg) {
  assert(m_fab);
  if (!m_cfg.max_outstanding_per_sm) m_cfg.max_outstanding_per_sm = 16;
  if (!m_cfg.ack_coalesce_threshold) m_cfg.ack_coalesce_threshold = 4;
  if (!m_cfg.ack_timeout_cycles) m_cfg.ack_timeout_cycles = 64;
  m_n = m_fab->num_sms();
  m_tx.resize(m_n);
  m_store_q.assign(m_n, std::vector<std::deque<unsigned>>(m_n));
  m_acks_owed.assign(m_n, std::vector<unsigned>(m_n, 0));
  m_oldest_ack.assign(m_n, std::vector<unsigned long long>(m_n, 0));
  m_last_write.assign(m_n, std::vector<unsigned long long>(m_n, 0));
  m_cta_gen.assign(m_n, std::vector<unsigned>(32, 0));
}

bool dsm_endpoint_protocol_t::window_open(unsigned sm) const {
  return sm < m_n && outstanding(sm) < m_cfg.max_outstanding_per_sm;
}

unsigned dsm_endpoint_protocol_t::flits_of(dsm_packet_class_t cls,
                                           unsigned bytes) const {
  return dsm_payload_flits(cls, bytes, m_fab->flit_payload_bytes());
}

unsigned dsm_endpoint_protocol_t::outstanding(unsigned sm) const {
  if (sm >= m_n) return 0;
  return (unsigned)m_tx[sm].size();
}

unsigned dsm_endpoint_protocol_t::ack_debt(unsigned sm) const {
  if (sm >= m_n) return 0;
  unsigned s = 0;
  for (unsigned r = 0; r < m_n; r++) s += m_acks_owed[sm][r];
  return s;
}

unsigned dsm_endpoint_protocol_t::ack_debt(unsigned sm,
                                           unsigned requester) const {
  if (sm >= m_n || requester >= m_n) return 0;
  return m_acks_owed[sm][requester];
}

double dsm_endpoint_protocol_t::coalescing_ratio() const {
  if (!m_stats.ack_packets) return 0;
  return (double)m_stats.ack_completions / (double)m_stats.ack_packets;
}

void dsm_endpoint_protocol_t::bump_cta_gen(unsigned local_sm, unsigned cta) {
  if (local_sm >= m_n || cta >= 32) return;
  m_cta_gen[local_sm][cta]++;
}

unsigned dsm_endpoint_protocol_t::cta_gen(unsigned local_sm,
                                          unsigned cta) const {
  if (local_sm >= m_n || cta >= 32) return 0;
  return m_cta_gen[local_sm][cta];
}

bool dsm_endpoint_protocol_t::write_unapplied(unsigned src,
                                              unsigned dst) const {
  if (src >= m_n || dst >= m_n) return false;
  for (const auto &kv : m_tx[src]) {
    const dsm_tx_t &t = kv.second;
    if (t.target == dst && t.cls == dsm_packet_class_t::write_data &&
        !t.applied)
      return true;
  }
  return in_flight_writes(src, dst) > 0;
}

bool dsm_endpoint_protocol_t::can_store(unsigned src, unsigned dst,
                                        unsigned bytes) const {
  if (!window_open(src) || src == dst) return false;
  return m_fab->can_inject(src, dsm_vc_t::request, dst,
                           flits_of(dsm_packet_class_t::write_data, bytes));
}

bool dsm_endpoint_protocol_t::can_load(unsigned src, unsigned dst,
                                       unsigned bytes) const {
  if (!window_open(src) || src == dst) return false;
  return m_fab->can_inject(src, dsm_vc_t::request, dst,
                           flits_of(dsm_packet_class_t::read_command, bytes));
}

bool dsm_endpoint_protocol_t::can_atom(unsigned src, unsigned dst,
                                       unsigned bytes) const {
  if (!window_open(src) || src == dst) return false;
  return m_fab->can_inject(src, dsm_vc_t::request, dst,
                           flits_of(dsm_packet_class_t::atomic_request, bytes));
}

void dsm_endpoint_protocol_t::note_outstanding(unsigned sm) {
  const unsigned o = outstanding(sm);
  if (o > m_stats.outstanding_high_water) m_stats.outstanding_high_water = o;
}

bool dsm_endpoint_protocol_t::issue_req(unsigned src, unsigned dst,
                                        unsigned bytes, uint64_t addr,
                                        unsigned cta_slot, unsigned cta_gen,
                                        dsm_packet_class_t cls,
                                        const void *data) {
  const bool is_store = cls == dsm_packet_class_t::write_data;
  const bool is_load = cls == dsm_packet_class_t::read_command;
  const bool is_atom = cls == dsm_packet_class_t::atomic_request;
  if (is_store && !can_store(src, dst, bytes)) return false;
  if (is_load && !can_load(src, dst, bytes)) return false;
  if (is_atom && !can_atom(src, dst, bytes)) return false;
  dsm_tx_t tx;
  tx.id = ++m_next_tx;
  tx.requester = src;
  tx.target = dst;
  tx.cls = cls;
  tx.remaining_responses = 1;
  tx.payload_bytes = bytes;
  tx.addr = addr;
  tx.created_cycle = m_now;
  tx.cta_slot = cta_slot;
  tx.cta_gen = cta_gen;
  auto p = std::make_unique<dsm_packet_t>();
  p->transaction_id = tx.id;
  p->network_src_sm_id = src;
  p->network_dst_sm_id = dst;
  p->transaction_requester_sm_id = src;
  p->transaction_target_sm_id = dst;
  p->packet_class = cls;
  p->vc = dsm_vc_of(cls);
  p->payload_bytes = bytes;
  p->payload_address = addr;
  p->created_cycle = m_now;
  p->cta_slot = cta_slot;
  p->cta_gen = cta_gen;
  if (data && bytes) {
    m_data[tx.id].assign((const uint8_t *)data, (const uint8_t *)data + bytes);
  }
  m_fab->inject(std::move(p));
  m_tx[src][tx.id] = tx;
  m_last_txid = tx.id;
  if (is_store) {
    m_store_q[src][dst].push_back(tx.id);
    m_stats.store_packets++;
  } else if (is_load) {
    m_stats.load_commands++;
  } else {
    m_stats.atom_requests++;
  }
  note_outstanding(src);
  return true;
}

bool dsm_endpoint_protocol_t::issue_store(unsigned src, unsigned dst,
                                          unsigned bytes, uint64_t addr) {
  return issue_req(src, dst, bytes, addr, 0, 0, dsm_packet_class_t::write_data,
                   nullptr);
}

bool dsm_endpoint_protocol_t::issue_load(unsigned src, unsigned dst,
                                         unsigned bytes, uint64_t addr) {
  return issue_req(src, dst, bytes, addr, 0, 0,
                   dsm_packet_class_t::read_command, nullptr);
}

bool dsm_endpoint_protocol_t::issue_store(unsigned src, unsigned dst,
                                          unsigned bytes, uint64_t addr,
                                          unsigned cta_slot, unsigned cta_gen,
                                          const void *data) {
  return issue_req(src, dst, bytes, addr, cta_slot, cta_gen,
                   dsm_packet_class_t::write_data, data);
}

bool dsm_endpoint_protocol_t::issue_load(unsigned src, unsigned dst,
                                         unsigned bytes, uint64_t addr,
                                         unsigned cta_slot, unsigned cta_gen) {
  return issue_req(src, dst, bytes, addr, cta_slot, cta_gen,
                   dsm_packet_class_t::read_command, nullptr);
}

bool dsm_endpoint_protocol_t::issue_atom(unsigned src, unsigned dst,
                                         unsigned bytes, uint64_t addr,
                                         unsigned cta_slot, unsigned cta_gen,
                                         const void *addend) {
  return issue_req(src, dst, bytes, addr, cta_slot, cta_gen,
                   dsm_packet_class_t::atomic_request, addend);
}

void dsm_endpoint_protocol_t::sram_store(unsigned sm, unsigned bytes) {
  (void)sm;
  m_stats.sram_store_bytes += bytes;
}

void dsm_endpoint_protocol_t::sram_load(unsigned sm, unsigned bytes) {
  (void)sm;
  m_stats.sram_load_bytes += bytes;
}

void dsm_endpoint_protocol_t::accrue_ack(unsigned target, unsigned requester) {
  if (target >= m_n || requester >= m_n) return;
  if (!m_acks_owed[target][requester]) m_oldest_ack[target][requester] = m_now;
  m_acks_owed[target][requester]++;
  m_last_write[target][requester] = m_now;
}

void dsm_endpoint_protocol_t::complete_stores(unsigned requester,
                                              unsigned target, unsigned count) {
  if (requester >= m_n || target >= m_n) return;
  for (unsigned i = 0; i < count; i++) {
    if (m_store_q[requester][target].empty()) break;
    const unsigned id = m_store_q[requester][target].front();
    m_store_q[requester][target].pop_front();
    m_tx[requester].erase(id);
    if (m_on_tx_done) m_on_tx_done(m_done_ctx, id);
  }
}

void dsm_endpoint_protocol_t::complete_load(unsigned requester, unsigned txid) {
  m_tx[requester].erase(txid);
  if (m_on_tx_done) m_on_tx_done(m_done_ctx, txid);
}

unsigned dsm_endpoint_protocol_t::response_occ(unsigned sm) const {
  unsigned s = 0;
  for (unsigned d = 0; d < m_n; d++)
    s += m_fab->occupancy_flits(sm, dsm_vc_t::response, d);
  return s;
}

unsigned dsm_endpoint_protocol_t::in_flight_writes(unsigned src,
                                                   unsigned dst) const {
  return m_fab->occupancy_flits(src, dsm_vc_t::request, dst);
}

bool dsm_endpoint_protocol_t::inject_response(const pending_t &p) {
  const unsigned flits = flits_of(p.cls, p.bytes);
  if (!m_fab->can_inject(p.src, dsm_vc_of(p.cls), p.dst, flits)) return false;
  auto pkt = std::make_unique<dsm_packet_t>();
  pkt->transaction_id = p.txid;
  pkt->network_src_sm_id = p.src;
  pkt->network_dst_sm_id = p.dst;
  pkt->transaction_requester_sm_id = p.dst;
  pkt->transaction_target_sm_id = p.src;
  pkt->packet_class = p.cls;
  pkt->vc = dsm_vc_of(p.cls);
  pkt->payload_bytes = p.bytes;
  pkt->payload_address = p.addr;
  pkt->completion_count = p.count ? p.count : 1;
  pkt->created_cycle = m_now;
  pkt->cta_slot = p.cta_slot;
  pkt->cta_gen = p.cta_gen;
  if (!p.payload.empty()) m_data[p.txid] = p.payload;
  m_fab->inject(std::move(pkt));
  return true;
}

bool dsm_endpoint_protocol_t::apply_sram(pending_t &p) {
  const bool gen_ok =
      p.cta_slot < 32 && p.cta_gen == m_cta_gen[p.src][p.cta_slot];
  if (p.cls == dsm_packet_class_t::write_data) {
    sram_store(p.src, p.bytes);
    auto dit = m_data.find(p.txid);
    if (gen_ok && m_sram_write && dit != m_data.end() && !dit->second.empty())
      m_sram_write(m_sram_ctx, p.src, p.cta_slot, p.addr, dit->second.data(),
                   (unsigned)dit->second.size());
    accrue_ack(p.src, p.dst);
    if (p.dst < m_n) {
      auto it = m_tx[p.dst].find(p.txid);
      if (it != m_tx[p.dst].end()) it->second.applied = true;
    }
    return true;
  }
  if (p.cls == dsm_packet_class_t::read_command) {
    p.payload.assign(p.bytes, 0);
    if (gen_ok && m_sram_read && p.bytes)
      m_sram_read(m_sram_ctx, p.src, p.cta_slot, p.addr, p.payload.data(),
                  p.bytes);
    sram_load(p.src, p.bytes);
    p.cls = dsm_packet_class_t::read_data;
    return true;
  }
  if (p.cls == dsm_packet_class_t::atomic_request) {
    p.payload.assign(p.bytes, 0);
    auto tit = m_data.find(p.txid);
    const std::vector<uint8_t> *addend =
        (tit != m_data.end()) ? &tit->second : nullptr;
    if (gen_ok && m_sram_read && p.bytes) {
      m_sram_read(m_sram_ctx, p.src, p.cta_slot, p.addr, p.payload.data(),
                  p.bytes);
      if (addend && p.bytes == 4 && addend->size() >= 4) {
        uint32_t old = 0, add = 0, neu = 0;
        memcpy(&old, p.payload.data(), 4);
        memcpy(&add, addend->data(), 4);
        neu = old + add;
        if (m_sram_write)
          m_sram_write(m_sram_ctx, p.src, p.cta_slot, p.addr,
                       reinterpret_cast<uint8_t *>(&neu), 4);
        memcpy(p.payload.data(), &old, 4);
      } else if (addend && p.bytes == 8 && addend->size() >= 8) {
        uint64_t old = 0, add = 0, neu = 0;
        memcpy(&old, p.payload.data(), 8);
        memcpy(&add, addend->data(), 8);
        neu = old + add;
        if (m_sram_write)
          m_sram_write(m_sram_ctx, p.src, p.cta_slot, p.addr,
                       reinterpret_cast<uint8_t *>(&neu), 8);
        memcpy(p.payload.data(), &old, 8);
      }
    }
    sram_store(p.src, p.bytes);
    sram_load(p.src, p.bytes);
    p.cls = dsm_packet_class_t::atomic_response;
    return true;
  }
  return true;
}

void dsm_endpoint_protocol_t::process_ingress() {
  std::deque<pending_t> keep;
  while (!m_ingress.empty()) {
    pending_t p = std::move(m_ingress.front());
    m_ingress.pop_front();
    if (p.arrive_cycle >= m_now) {
      keep.push_back(std::move(p));
      continue;
    }
    if (!p.sram_done) {
      unsigned need = p.bytes ? p.bytes - p.sram_got : 0;
      unsigned got = 0;
      if (need && m_sram_take)
        got = m_sram_take(m_flow_ctx, p.src, need);
      else
        got = need;
      p.sram_got += got;
      if (p.bytes && p.sram_got < p.bytes) {
        if (m_sram_expose)
          m_sram_expose(m_flow_ctx, p.src, p.bytes - p.sram_got);
        keep.push_back(std::move(p));
        continue;
      }
      apply_sram(p);
      p.sram_done = true;
      p.inject_after = m_now;
      if (p.cls == dsm_packet_class_t::write_data) {
        continue;
      }
    }
    if (p.inject_after >= m_now) {
      keep.push_back(std::move(p));
      continue;
    }
    m_pending.push_back(std::move(p));
  }
  m_ingress.swap(keep);
}

void dsm_endpoint_protocol_t::harvest() {
  for (unsigned sm = 0; sm < m_n; sm++) {
    while (m_fab->top(sm, dsm_vc_t::request)) {
      auto p = m_fab->pop(sm, dsm_vc_t::request);
      if (!p) break;
      pending_t in;
      in.cls = p->packet_class;
      in.src = sm;
      in.dst = p->transaction_requester_sm_id;
      in.txid = p->transaction_id;
      in.bytes = p->payload_bytes;
      in.count = 1;
      in.addr = p->payload_address;
      in.cta_slot = p->cta_slot;
      in.cta_gen = p->cta_gen;
      in.arrive_cycle = m_now;
      if (m_sram_expose && in.bytes)
        m_sram_expose(m_flow_ctx, sm, in.bytes);
      m_ingress.push_back(std::move(in));
    }
    while (m_fab->top(sm, dsm_vc_t::response)) {
      auto p = m_fab->pop(sm, dsm_vc_t::response);
      if (!p) break;
      if (p->packet_class == dsm_packet_class_t::write_ack)
        complete_stores(sm, p->transaction_target_sm_id,
                        p->completion_count ? p->completion_count : 1);
      else if (p->packet_class == dsm_packet_class_t::read_data ||
               p->packet_class == dsm_packet_class_t::atomic_response) {
        auto dit = m_data.find(p->transaction_id);
        if (m_on_load_data && dit != m_data.end() && !dit->second.empty())
          m_on_load_data(m_load_ctx, p->transaction_id, dit->second.data(),
                         (unsigned)dit->second.size());
        if (dit != m_data.end()) m_data.erase(dit);
        if (p->packet_class == dsm_packet_class_t::atomic_response)
          m_stats.atom_responses++;
        complete_load(sm, p->transaction_id);
      }
    }
  }
}

void dsm_endpoint_protocol_t::try_send_pending() {
  std::deque<pending_t> keep;
  while (!m_pending.empty()) {
    pending_t p = m_pending.front();
    m_pending.pop_front();
    if (inject_response(p)) {
      if (p.cls == dsm_packet_class_t::read_data) m_stats.load_data_packets++;
    } else {
      keep.push_back(std::move(p));
    }
  }
  m_pending.swap(keep);
}

void dsm_endpoint_protocol_t::try_flush_acks() {
  for (unsigned sm = 0; sm < m_n; sm++) {
    for (unsigned req = 0; req < m_n; req++) {
      unsigned debt = m_acks_owed[sm][req];
      if (!debt) continue;
      const bool thresh = debt >= m_cfg.ack_coalesce_threshold;
      const bool timed =
          (m_now - m_oldest_ack[sm][req]) >= m_cfg.ack_timeout_cycles;
      bool ingress_store = false;
      for (const auto &in : m_ingress)
        if (in.src == sm && in.dst == req &&
            in.cls == dsm_packet_class_t::write_data)
          ingress_store = true;
      const bool idle = in_flight_writes(req, sm) == 0 &&
                        response_occ(sm) == 0 && !ingress_store &&
                        (m_now - m_last_write[sm][req]) >= 3;
      if (!thresh && !timed && !idle) continue;
      pending_t p;
      p.cls = dsm_packet_class_t::write_ack;
      p.src = sm;
      p.dst = req;
      p.txid = 0;
      p.bytes = 0;
      p.count = debt;
      p.addr = 0;
      p.cta_slot = 0;
      p.cta_gen = 0;
      if (!inject_response(p)) continue;
      m_acks_owed[sm][req] = 0;
      m_oldest_ack[sm][req] = 0;
      m_stats.ack_packets++;
      m_stats.ack_completions += debt;
      if (thresh)
        m_stats.threshold_flushes++;
      else if (timed)
        m_stats.timeout_flushes++;
      else
        m_stats.idle_flushes++;
    }
  }
}

void dsm_endpoint_protocol_t::cycle(unsigned long long now) {
  m_now = now;
  m_fab->cycle(now);
  harvest();
  process_ingress();
  try_send_pending();
  try_flush_acks();
}

bool dsm_endpoint_protocol_t::busy() const {
  if (m_fab->busy()) return true;
  if (!m_pending.empty() || !m_ingress.empty()) return true;
  for (unsigned sm = 0; sm < m_n; sm++) {
    if (!m_tx[sm].empty()) return true;
    if (ack_debt(sm)) return true;
  }
  return false;
}

void dsm_endpoint_protocol_t::display_state(FILE *fp) const {
  if (!fp) return;
  unsigned out_sum = 0, debt_sum = 0;
  for (unsigned sm = 0; sm < m_n; sm++) {
    out_sum += outstanding(sm);
    debt_sum += ack_debt(sm);
  }
  fprintf(fp,
          "dsm_endpoint outstanding=%u ack_debt=%u coalescing_ratio=%.3f "
          "timeout_flushes=%llu threshold_flushes=%llu idle_flushes=%llu\n",
          out_sum, debt_sum, coalescing_ratio(), m_stats.timeout_flushes,
          m_stats.threshold_flushes, m_stats.idle_flushes);
  fprintf(fp,
          "  stores=%llu acks=%llu ack_completions=%llu loads=%llu "
          "read_data=%llu atoms=%llu sram_store=%llu sram_load=%llu\n",
          m_stats.store_packets, m_stats.ack_packets, m_stats.ack_completions,
          m_stats.load_commands, m_stats.load_data_packets,
          m_stats.atom_requests, m_stats.sram_store_bytes,
          m_stats.sram_load_bytes);
  for (unsigned sm = 0; sm < m_n; sm++) {
    for (const auto &kv : m_tx[sm]) {
      const dsm_tx_t &t = kv.second;
      fprintf(fp,
              "  tx id=%u requester=%u target=%u class=%d remaining=%u "
              "bytes=%u addr=%llu created=%llu\n",
              t.id, t.requester, t.target, (int)t.cls, t.remaining_responses,
              t.payload_bytes, (unsigned long long)t.addr, t.created_cycle);
    }
  }
  m_fab->display_state(fp);
}
