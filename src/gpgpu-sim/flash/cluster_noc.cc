#include "cluster_noc.h"

#include "../../abstract_hardware_model.h"
#include "../../cuda-sim/ptx_sim.h"
#include "../gpu-sim.h"
#include "../shader.h"

namespace flash_gpgpu_sim {

// ---------------------------------------------------------------------------
// cluster_noc_t
// ---------------------------------------------------------------------------

cluster_noc_t::cluster_noc_t(simt_core_cluster *cluster,
                             const shader_core_config *config)
    : m_cluster(cluster), m_config(config) {
  reconfigure();
}

void cluster_noc_t::reconfigure() {
  if (!m_config)
    return;
  const unsigned n =
      m_cluster ? m_cluster->num_cores() : m_config->n_simt_cores_per_cluster;
  const unsigned local_lat = m_config->gpgpu_dsm_local_latency;
  const unsigned remote_lat = m_config->gpgpu_dsm_remote_latency;
  m_matrix.init(n, local_lat, remote_lat);
  if (m_config->gpgpu_dsm_latency_matrix_file &&
      m_config->gpgpu_dsm_latency_matrix_file[0] != '\0') {
    m_matrix.load_from_file(m_config->gpgpu_dsm_latency_matrix_file, n);
  }
}

bool cluster_noc_t::enabled() const {
  return m_config && m_config->gpgpu_cluster_noc_enable;
}

unsigned long long cluster_noc_t::now() const {
  if (!m_cluster || !m_cluster->get_gpu())
    return 0;
  gpgpu_sim *gpu = m_cluster->get_gpu();
  return gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
}

unsigned cluster_noc_t::hop_latency(unsigned src_cid, unsigned dst_cid) const {
  if (src_cid == dst_cid)
    return m_config ? m_config->gpgpu_dsm_local_latency : 0;
  if (!m_matrix.empty())
    return m_matrix.hop(src_cid, dst_cid);
  return m_config ? m_config->gpgpu_dsm_remote_latency : 0;
}

unsigned cluster_noc_t::resolve_hop(unsigned src_cid, unsigned dst_cid) const {
  if (!m_config)
    return 0;
  if (src_cid == dst_cid)
    return 0;
  if (m_config->gpgpu_mbarrier_remote_hop_latency != 0)
    return m_config->gpgpu_mbarrier_remote_hop_latency;
  return hop_latency(src_cid, dst_cid);
}

unsigned cluster_noc_t::bw_extra_cycles(cluster_noc_msg_type type,
                                        uint32_t size_in_bytes) const {
  if (!m_config || size_in_bytes == 0)
    return 0;
  unsigned bpc = 0;
  switch (type) {
  case cluster_noc_msg_type::DSM_STORE:
  case cluster_noc_msg_type::DSM_LOAD_REQ:
  case cluster_noc_msg_type::DSM_LOAD_RSP:
    bpc = m_config->gpgpu_dsm_bytes_per_cycle;
    break;
  case cluster_noc_msg_type::MBAR_REMOTE_OP:
    // size_in_bytes may encode parity; not a payload length.
    return 0;
  }
  if (bpc == 0)
    return 0;
  // ceil(size/bpc) - 1 so the first flit is covered by hop alone.
  return (size_in_bytes + bpc - 1) / bpc - 1;
}

bool cluster_noc_t::inject(cluster_noc_message msg) {
  if (!enabled())
    return false;
  const unsigned long long t = now();
  msg.inject_cycle = t;
  if (msg.ready_cycle < t) {
    // Caller did not pre-set ready; compute hop.
    unsigned hop = resolve_hop(msg.src_cid, msg.dst_cid);
    hop += bw_extra_cycles(msg.type, msg.size_in_bytes);
    msg.ready_cycle = t + hop;
  }
  msg.seq = m_next_seq++;
  if (m_in_deliver_sweep) {
    m_deferred_inject.push_back(std::move(msg));
  } else {
    m_inflight.push_back(std::move(msg));
  }
  m_stats.injected++;
  const size_t inflight_now = m_inflight.size() + m_deferred_inject.size();
  if (inflight_now > m_stats.max_inflight)
    m_stats.max_inflight = inflight_now;
  return true;
}

bool cluster_noc_t::inject_dsm_store(unsigned src_cid, unsigned dst_cid,
                                     unsigned dst_hw_cta, uint32_t smem_addr,
                                     const void *data, uint32_t size_in_bytes,
                                     uint64_t stream_key) {
  if (!enabled())
    return false;
  cluster_noc_message msg;
  msg.type = cluster_noc_msg_type::DSM_STORE;
  msg.src_cid = src_cid;
  msg.dst_cid = dst_cid;
  msg.dst_hw_cta = dst_hw_cta;
  msg.smem_addr = smem_addr;
  msg.size_in_bytes = size_in_bytes;
  msg.stream_key = stream_key;
  msg.payload = std::make_shared<cluster_noc_payload>();
  if (data && size_in_bytes > 0) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    msg.payload->bytes.assign(bytes, bytes + size_in_bytes);
  }
  return inject(std::move(msg));
}

bool cluster_noc_t::inject_dsm_load_req(unsigned src_cid, unsigned dst_cid,
                                        unsigned dst_hw_cta, uint32_t smem_addr,
                                        uint32_t size_in_bytes,
                                        unsigned req_hw_cta,
                                        unsigned req_warp_id, unsigned req_lane,
                                        uint64_t stream_key) {
  if (!enabled())
    return false;
  cluster_noc_message msg;
  msg.type = cluster_noc_msg_type::DSM_LOAD_REQ;
  msg.src_cid = src_cid;
  msg.dst_cid = dst_cid;
  msg.dst_hw_cta = dst_hw_cta;
  msg.smem_addr = smem_addr;
  msg.size_in_bytes = size_in_bytes;
  msg.req_hw_cta = req_hw_cta;
  msg.req_warp_id = req_warp_id;
  msg.req_lane = req_lane;
  msg.stream_key = stream_key;
  return inject(std::move(msg));
}

bool cluster_noc_t::inject_mbar_remote(unsigned src_cid, unsigned dst_cid,
                                       unsigned dst_hw_cta, uint32_t mbar_addr,
                                       cluster_mbar_op op, uint32_t count,
                                       unsigned req_hw_cta,
                                       unsigned req_warp_id, int parity) {
  if (!enabled())
    return false;
  cluster_noc_message msg;
  msg.type = cluster_noc_msg_type::MBAR_REMOTE_OP;
  msg.src_cid = src_cid;
  msg.dst_cid = dst_cid;
  msg.dst_hw_cta = dst_hw_cta;
  msg.mbar_addr = mbar_addr;
  msg.mbar_op = op;
  msg.mbar_count = count;
  msg.req_hw_cta = req_hw_cta;
  msg.req_warp_id = req_warp_id;
  // Reuse size_in_bytes for parity on WAIT_REG / success on WAIT_DONE.
  msg.size_in_bytes = static_cast<uint32_t>(parity);
  return inject(std::move(msg));
}

void cluster_noc_t::deliver_ready() {
  if (!enabled())
    return;
  if (m_inflight.empty() && m_deferred_inject.empty())
    return;
  const unsigned long long t = now();

  while (!m_deferred_inject.empty()) {
    m_inflight.push_back(std::move(m_deferred_inject.front()));
    m_deferred_inject.pop_front();
  }

  m_in_deliver_sweep = true;
  auto *ep = m_cluster ? m_cluster->get_dsm_endpoint() : nullptr;
  for (auto it = m_inflight.begin(); it != m_inflight.end();) {
    if (it->ready_cycle <= t) {
      if (it->type == cluster_noc_msg_type::MBAR_REMOTE_OP && ep &&
          ep->write_unapplied(it->src_cid, it->dst_cid)) {
        ++it;
        continue;
      }
      deliver(*it);
      m_stats.delivered++;
      m_stats.bytes_delivered += it->size_in_bytes;
      it = m_inflight.erase(it);
    } else {
      ++it;
    }
  }
  m_in_deliver_sweep = false;

  while (!m_deferred_inject.empty()) {
    m_inflight.push_back(std::move(m_deferred_inject.front()));
    m_deferred_inject.pop_front();
  }
}

void cluster_noc_t::cycle() { deliver_ready(); }

void cluster_noc_t::drop_messages_to_cta(unsigned cid, unsigned hw_cta) {
  m_stats.dropped_on_cta_exit +=
      cluster_noc_drop_queue_to_cta(&m_inflight, cid, hw_cta);
  m_stats.dropped_on_cta_exit +=
      cluster_noc_drop_queue_to_cta(&m_deferred_inject, cid, hw_cta);
}

void cluster_noc_t::deliver(cluster_noc_message &msg) {
  switch (msg.type) {
  case cluster_noc_msg_type::DSM_STORE:
    deliver_dsm_store(msg);
    break;
  case cluster_noc_msg_type::DSM_LOAD_REQ:
    deliver_dsm_load_req(msg);
    break;
  case cluster_noc_msg_type::DSM_LOAD_RSP:
    // Functional path applies data at inject-time read; timing scoreboard
    // completion is a follow-up. No-op payload for now.
    break;
  case cluster_noc_msg_type::MBAR_REMOTE_OP:
    deliver_mbar_remote(msg);
    break;
  }
}

void cluster_noc_t::deliver_dsm_store(cluster_noc_message &msg) {
  if (!m_cluster || !msg.payload)
    return;
  shader_core_ctx *peer = m_cluster->get_core(msg.dst_cid);
  if (!peer)
    return;
  memory_space *peer_smem = peer->get_cta_smem(msg.dst_hw_cta);
  if (!peer_smem)
    return;
  const auto &bytes = msg.payload->bytes;
  if (!bytes.empty()) {
    peer_smem->write(msg.smem_addr, bytes.size(), bytes.data(), nullptr,
                     nullptr);
  }
}

void cluster_noc_t::deliver_dsm_load_req(cluster_noc_message &msg) {
  // Read peer smem and inject a response hop back to requester.
  if (!m_cluster)
    return;
  shader_core_ctx *peer = m_cluster->get_core(msg.dst_cid);
  if (!peer)
    return;
  memory_space *peer_smem = peer->get_cta_smem(msg.dst_hw_cta);
  auto payload = std::make_shared<cluster_noc_payload>();
  payload->bytes.resize(msg.size_in_bytes);
  if (peer_smem && msg.size_in_bytes > 0) {
    peer_smem->read(msg.smem_addr, msg.size_in_bytes, payload->bytes.data());
  }

  cluster_noc_message rsp;
  rsp.type = cluster_noc_msg_type::DSM_LOAD_RSP;
  rsp.src_cid = msg.dst_cid;
  rsp.dst_cid = msg.src_cid;
  rsp.dst_hw_cta = msg.req_hw_cta;
  rsp.req_hw_cta = msg.req_hw_cta;
  rsp.req_warp_id = msg.req_warp_id;
  rsp.req_lane = msg.req_lane;
  rsp.size_in_bytes = msg.size_in_bytes;
  rsp.stream_key = msg.stream_key;
  rsp.payload = payload;
  // Return hop.
  const unsigned long long t = now();
  unsigned hop = hop_latency(msg.dst_cid, msg.src_cid);
  rsp.inject_cycle = t;
  rsp.ready_cycle = t + hop;
  inject(std::move(rsp));
}

void cluster_noc_t::deliver_mbar_remote(cluster_noc_message &msg) {
  if (!m_cluster)
    return;
  shader_core_ctx *dst_core = m_cluster->get_core(msg.dst_cid);
  if (!dst_core)
    return;

  switch (msg.mbar_op) {
  case cluster_mbar_op::TRY_COMPLETE_TX:
  case cluster_mbar_op::COMPLETE_TX:
    dst_core->try_complete_cluster_peer_mbarrier(msg.dst_hw_cta, msg.mbar_addr,
                                                 msg.mbar_count);
    // Peer complete may advance phase — notify remote waiters.
    dst_core->notify_remote_mbarrier_waiters(msg.dst_hw_cta, msg.mbar_addr);
    break;
  case cluster_mbar_op::ARRIVE:
    dst_core->remote_mbarrier_arrive(msg.dst_hw_cta, msg.mbar_addr,
                                     msg.mbar_count);
    dst_core->notify_remote_mbarrier_waiters(msg.dst_hw_cta, msg.mbar_addr);
    break;
  case cluster_mbar_op::EXPECT_TX:
    dst_core->remote_mbarrier_expect_tx(msg.dst_hw_cta, msg.mbar_addr,
                                        msg.mbar_count);
    break;
  case cluster_mbar_op::WAIT_REG: {
    // Register remote try_wait on owner; reply DONE if already satisfied.
    const int parity = static_cast<int>(msg.size_in_bytes);
    const bool satisfied = dst_core->register_remote_mbarrier_wait(
        msg.dst_hw_cta, msg.mbar_addr, parity, msg.src_cid, msg.req_hw_cta,
        msg.req_warp_id);
    if (satisfied) {
      inject_mbar_remote(/*src=*/msg.dst_cid, /*dst=*/msg.src_cid,
                         msg.req_hw_cta, msg.mbar_addr,
                         cluster_mbar_op::WAIT_DONE, /*count=*/1,
                         msg.req_hw_cta, msg.req_warp_id, /*parity=*/1);
    }
    break;
  }
  case cluster_mbar_op::WAIT_DONE:
    // Release waiting warp on the requesting SM.
    dst_core->release_remote_mbarrier_waiter(msg.req_warp_id);
    break;
  }
}

} // namespace flash_gpgpu_sim
