// Intra-cluster SM↔SM Network-on-Chip for FlashGPU-Sim.
//
// Models hop latency (and optional simple bandwidth) for:
//   - TMA .shared::cluster multicast data + peer mbarrier complete_tx
//   - Distributed shared memory (DSM) loads/stores and atom (owner-smem RMW)
//   - Remote mbarrier operations
//
// Design rules (race-freedom):
//   1. One cluster_noc_t per simt_core_cluster; only that cluster's thread
//      runs cycle() after core_cycle() (OpenMP is across clusters).
//   2. Cross-SM shared state is mutated only in deliver(), never by a foreign
//      SM pipeline stage.
//   3. Per (src_cid, dst_cid) FIFO for messages with the same stream key.
//
// See docs/cluster_noc.md for the full model.

#ifndef FLASH_GPGPU_SIM_CLUSTER_NOC_H
#define FLASH_GPGPU_SIM_CLUSTER_NOC_H

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "../../abstract_hardware_model.h" // addr_t

class memory_space;
class shader_core_ctx;
class simt_core_cluster;
class shader_core_config;
class gpgpu_sim;

namespace flash_gpgpu_sim {

enum class cluster_noc_msg_type : uint8_t {
  TMA_MCAST_DATA = 0, // payload → peer CTA smem
  TMA_MCAST_MBAR,     // try_complete_tx_if_pending on peer
  DSM_STORE,          // write peer smem
  DSM_LOAD_REQ,       // read peer smem, enqueue DSM_LOAD_RSP
  DSM_LOAD_RSP,       // complete remote load (functional data path)
  MBAR_REMOTE_OP,     // remote arrive / expect_tx / complete_tx
};

enum class cluster_mbar_op : uint8_t {
  ARRIVE = 0,
  EXPECT_TX,
  COMPLETE_TX,
  TRY_COMPLETE_TX,
  // Remote try_wait interest: register waiter on owner; DONE replies to src.
  WAIT_REG,
  WAIT_DONE,
};

// Payload for in-flight messages. Large TMA tiles store bytes here so the
// issuer may overwrite local smem after inject without corrupting peers.
struct cluster_noc_payload {
  std::vector<uint8_t> bytes;
};

struct cluster_noc_message {
  cluster_noc_msg_type type = cluster_noc_msg_type::TMA_MCAST_DATA;
  unsigned src_cid = 0; // core id within physical cluster
  unsigned dst_cid = 0;
  unsigned src_sid = 0; // global SM id (debug / matrix remap)
  unsigned dst_sid = 0;
  unsigned dst_hw_cta = 0;
  uint32_t smem_addr = 0;
  uint32_t size_in_bytes = 0;
  uint32_t mbar_addr = 0;
  uint32_t mbar_count = 0;
  cluster_mbar_op mbar_op = cluster_mbar_op::TRY_COMPLETE_TX;
  unsigned long long inject_cycle = 0;
  unsigned long long ready_cycle = 0;
  uint64_t stream_key = 0; // ordering: same (src,dst,stream) is FIFO
  uint64_t seq = 0;        // monotonic inject order within fabric
  // DSM load completion bookkeeping (optional; functional path may ignore).
  unsigned req_hw_cta = 0;
  unsigned req_warp_id = 0;
  unsigned req_lane = 0;
  std::shared_ptr<cluster_noc_payload> payload;
};

struct cluster_noc_stats {
  uint64_t injected = 0;
  uint64_t delivered = 0;
  uint64_t bytes_delivered = 0;
  uint64_t max_inflight = 0;
  uint64_t dropped_on_cta_exit = 0;
};

// Latency matrix indexed by cluster-local core id [src][dst].
class cluster_noc_latency_matrix {
public:
  void init(unsigned n_cores, unsigned local_lat, unsigned remote_lat);
  // Load dense N×N CSV of integers (whitespace or comma separated).
  // Returns false on parse failure (matrix left unchanged).
  bool load_from_file(const std::string &path, unsigned n_cores);
  unsigned hop(unsigned src_cid, unsigned dst_cid) const;
  unsigned n_cores() const { return m_n; }
  bool empty() const { return m_n == 0; }

private:
  unsigned m_n = 0;
  std::vector<unsigned> m_lat; // row-major n*n
};

class cluster_noc_t {
public:
  cluster_noc_t(simt_core_cluster *cluster, const shader_core_config *config);

  // Rebuild latency matrix from config (call after config is fully parsed).
  void reconfigure();

  bool enabled() const;

  // Current absolute simulation cycle (gpu_sim + gpu_tot).
  unsigned long long now() const;

  // Hop latency between two cores in this physical cluster.
  unsigned hop_latency(unsigned src_cid, unsigned dst_cid) const;

  // TMA multicast hop (may use dedicated knob or DSM matrix).
  unsigned tma_mcast_hop(unsigned src_cid, unsigned dst_cid) const;

  // Inject a message. ready_cycle = now + hop(+optional BW term).
  // Returns false if NoC disabled (caller should apply effect immediately).
  bool inject(cluster_noc_message msg);

  // Convenience: inject TMA data + ordered mbar for one peer.
  // Data message is injected first; mbar ready_cycle >= data ready_cycle.
  bool inject_tma_mcast_to_peer(unsigned src_cid, unsigned dst_cid,
                                unsigned dst_hw_cta, uint32_t smem_addr,
                                const uint8_t *data, uint32_t size_in_bytes,
                                uint32_t mbar_addr, uint32_t mbar_bytes,
                                uint64_t stream_key);

  // DSM store: enqueue peer write for deliver after hop. Issuer also writes
  // at issue only when -gpgpu_dsm_store_immediate is 1 (default is 0).
  bool inject_dsm_store(unsigned src_cid, unsigned dst_cid, unsigned dst_hw_cta,
                        uint32_t smem_addr, const void *data,
                        uint32_t size_in_bytes, uint64_t stream_key = 0);

  // DSM load request (RTT = hop to peer + hop return). Optional timing
  // bookkeeping.
  bool inject_dsm_load_req(unsigned src_cid, unsigned dst_cid,
                           unsigned dst_hw_cta, uint32_t smem_addr,
                           uint32_t size_in_bytes, unsigned req_hw_cta,
                           unsigned req_warp_id, unsigned req_lane,
                           uint64_t stream_key = 0);

  // Remote mbarrier op toward owner CTA (arrive / expect / complete /
  // wait_reg). For WAIT_REG: req_* identify the waiting warp on the source SM.
  // For WAIT_DONE: delivered back to requester; mbar_count holds success (1/0).
  bool inject_mbar_remote(unsigned src_cid, unsigned dst_cid,
                          unsigned dst_hw_cta, uint32_t mbar_addr,
                          cluster_mbar_op op, uint32_t count,
                          unsigned req_hw_cta = 0, unsigned req_warp_id = 0,
                          int parity = 0);

  // Deliver any messages whose ready_cycle has already arrived (also called
  // from cycle()). Useful before a remote load so due stores are visible.
  void deliver_ready();

  // Advance fabric: deliver all messages with ready_cycle <= now.
  void cycle();

  // Drop in-flight messages targeting a CTA (slot exit / recycle).
  void drop_messages_to_cta(unsigned cid, unsigned hw_cta);

  const cluster_noc_stats &stats() const { return m_stats; }
  size_t inflight() const { return m_inflight.size(); }

  simt_core_cluster *cluster() const { return m_cluster; }

private:
  void deliver(cluster_noc_message &msg);
  void deliver_tma_mcast_data(cluster_noc_message &msg);
  void deliver_tma_mcast_mbar(cluster_noc_message &msg);
  void deliver_dsm_store(cluster_noc_message &msg);
  void deliver_dsm_load_req(cluster_noc_message &msg);
  void deliver_mbar_remote(cluster_noc_message &msg);

  unsigned resolve_hop(unsigned src_cid, unsigned dst_cid, bool tma_path) const;
  // BW term: TMA uses tma_mcast_bytes_per_cycle; DSM store/load uses
  // dsm_bytes_per_cycle. Mbar/remote control msgs get 0 (size_in_bytes may
  // hold parity/flags, not payload length).
  unsigned bw_extra_cycles(cluster_noc_msg_type type,
                           uint32_t size_in_bytes) const;

  simt_core_cluster *m_cluster;
  const shader_core_config *m_config;
  cluster_noc_latency_matrix m_matrix;
  std::deque<cluster_noc_message> m_inflight;
  // Messages injected during deliver() are staged here and moved after the
  // delivery sweep so we never invalidate the inflight iterator mid-loop.
  std::deque<cluster_noc_message> m_deferred_inject;
  uint64_t m_next_seq = 1;
  cluster_noc_stats m_stats;
  bool m_in_deliver_sweep = false;
};

// Helpers for generic shared address windows (per-SM).
// Returns true if addr is in any SM's shared generic window.
bool decode_shared_generic(addr_t addr, unsigned *out_smid, addr_t *out_offset);
// True if addr is shared of some SM and that SM != local_smid.
bool is_remote_shared_generic(unsigned local_smid, addr_t addr,
                              unsigned *out_owner_smid = nullptr,
                              addr_t *out_offset = nullptr);

// CTA-exit drop predicate + queue filter (used by cluster_noc_t and unit
// tests).
inline bool cluster_noc_msg_targets_cta(const cluster_noc_message &msg,
                                        unsigned cid, unsigned hw_cta) {
  return msg.dst_cid == cid && msg.dst_hw_cta == hw_cta;
}
// Erase messages targeting (cid, hw_cta). Returns the number removed.
size_t cluster_noc_drop_queue_to_cta(std::deque<cluster_noc_message> *q,
                                     unsigned cid, unsigned hw_cta);

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_CLUSTER_NOC_H
