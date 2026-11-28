#include "clc.h"
#include "../../cuda-sim/memory.h"
#include "../shader.h"
#include "panic.h"
#include "tb_cluster.h"
#include <algorithm>
#include <tuple>

void shader_core_ctx::enqueue_clc(unsigned warp, uint32_t response,
                                  uint32_t barrier, bool broadcast) {
  using namespace flash_gpgpu_sim;
  const unsigned cta = m_warp[warp]->get_cta_id();
  if ((response & 15) || (barrier & 7) || !is_cta_slot_active(cta))
    panic("CLC requires aligned addresses and a live requesting CTA");
  clc_request req;
  req.kernel = m_warp[warp]->get_kernel_info();
  req.cta = cta;
  req.warp = warp;
  req.generation = m_cta_lifecycle[cta].generation;
  req.logical_cta = get_logical_cta_id(warp);
  req.response_address = response & 0xffffff;
  req.barrier_address = barrier & 0xffffff;
  req.broadcast = broadcast;
  m_clc_requests.push_back(req);
}

bool shader_core_ctx::clc_pending_for_cta(unsigned cta) const {
  return std::any_of(m_clc_requests.begin(), m_clc_requests.end(),
                     [cta](const auto &r) { return r.cta == cta; });
}

void shader_core_ctx::arbitrate_clc(uint64_t now) {
  // The GPU calls every core in ascending physical order outside OpenMP.
  // Within a core, CTA/warp ordering is independent of hash-table layout.
  std::stable_sort(m_clc_requests.begin(), m_clc_requests.end(),
                   [](const auto &a, const auto &b) {
                     return std::tie(a.logical_cta, a.warp) <
                            std::tie(b.logical_cta, b.warp);
                   });
  for (auto &req : m_clc_requests) {
    if (req.arbitrated)
      continue;
    dim3 canceled;
    const bool granted = req.kernel->try_cancel_pending_cluster(canceled);
    req.response = flash_gpgpu_sim::clc_response(granted, canceled.x,
                                                 canceled.y, canceled.z);
    req.ready_cycle = now + std::max(1u, m_config->gpgpu_clc_latency);
    req.arbitrated = true;
  }
}

void shader_core_ctx::deliver_clc(uint64_t now) {
  using namespace flash_gpgpu_sim;
  auto it = m_clc_requests.begin();
  while (it != m_clc_requests.end()) {
    if (!it->arbitrated || now < it->ready_cycle) {
      ++it;
      continue;
    }
    if (!is_cta_slot_active(it->cta) ||
        m_cta_lifecycle[it->cta].generation != it->generation)
      panic("CLC response outlived its requesting CTA");
    auto deliver = [&](shader_core_ctx *target, unsigned slot) {
      auto *memory = target->get_cta_smem(slot);
      if (!memory)
        panic("CLC destination has no shared storage");
      memory->write(it->response_address, 16, it->response.data(), nullptr,
                    nullptr);
      target->complete_clc_architectural(slot, it->barrier_address);
      target->try_complete_cluster_peer_mbarrier(slot, it->barrier_address, 16);
    };
    if (it->broadcast)
      for_each_tb_cluster_peer(this, it->cta, deliver, true);
    else
      deliver(this, it->cta);
    it = m_clc_requests.erase(it);
  }
}
