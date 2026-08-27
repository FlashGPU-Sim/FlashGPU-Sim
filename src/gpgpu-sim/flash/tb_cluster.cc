#include "tb_cluster.h"

#include <stdio.h>
#include <stdlib.h>

#include "../gpu_topology.h"
#include "../shader.h"

namespace flash_gpgpu_sim {

static bool fill_target(shader_core_ctx *peer, unsigned slot,
                        tb_cluster_target_t *out) {
  if (!peer || !out)
    return false;
  memory_space *smem = peer->get_cta_smem(slot);
  if (!smem)
    return false;
  out->core = peer;
  out->sm_id = peer->get_sid();
  out->local_sm = peer->get_config()->sid_to_cid(peer->get_sid());
  out->cta_slot = slot;
  out->smem = smem;
  return true;
}

static bool match_group(shader_core_ctx *requester, unsigned req_slot,
                        shader_core_ctx *peer, unsigned slot) {
  if (!peer->is_cta_slot_active(slot))
    return false;
  const unsigned group = requester->get_cta_cluster_group(req_slot);
  if (group != (unsigned)-1 && peer->get_cta_cluster_group(slot) != group)
    return false;
  return true;
}

bool resolve_tb_cluster_rank(shader_core_ctx *requester,
                             unsigned requester_cta_slot, unsigned target_rank,
                             tb_cluster_target_t *out) {
  if (!requester || !out)
    return false;
  auto *cluster = requester->get_cluster();
  if (!cluster) {
    if (target_rank == 0)
      return fill_target(requester, requester_cta_slot, out);
    return false;
  }
  for (unsigned cid = 0; cid < cluster->num_cores(); cid++) {
    auto *peer = cluster->get_core(cid);
    for (unsigned slot = 0; slot < MAX_CTA_PER_SHADER; slot++) {
      if (!match_group(requester, requester_cta_slot, peer, slot))
        continue;
      if (peer->get_cta_cluster_rank(slot) != target_rank)
        continue;
      return fill_target(peer, slot, out);
    }
  }
  return false;
}

bool resolve_tb_cluster_owner_sm(shader_core_ctx *requester,
                                 unsigned requester_cta_slot,
                                 unsigned owner_sm_id,
                                 tb_cluster_target_t *out) {
  if (!requester || !out)
    return false;
  auto *cluster = requester->get_cluster();
  if (!cluster)
    return false;
  const gpu_topology_t &topo = requester->get_config()->topology();
  if (owner_sm_id >= topo.num_sms())
    return false;
  if (topo.gpc_id_of_sm(owner_sm_id) != topo.gpc_id_of_sm(requester->get_sid()))
    return false;
  for (unsigned cid = 0; cid < cluster->num_cores(); cid++) {
    auto *peer = cluster->get_core(cid);
    if (peer->get_sid() != owner_sm_id)
      continue;
    for (unsigned slot = 0; slot < MAX_CTA_PER_SHADER; slot++) {
      if (!match_group(requester, requester_cta_slot, peer, slot))
        continue;
      return fill_target(peer, slot, out);
    }
    return false;
  }
  return false;
}

void abort_tb_cluster_dead_rank(unsigned rank, unsigned issuer_sm,
                                unsigned hw_cta) {
  const char *fmt =
      "GPGPU-Sim ERROR: mapa: cluster rank %u is not an active CTA "
      "(issuer smid=%u hw_cta=%u). The target CTA has exited or was "
      "never co-resident; refusing to alias the issuer's shared memory.\n";
  printf(fmt, rank, issuer_sm, hw_cta);
  fprintf(stderr, fmt, rank, issuer_sm, hw_cta);
  fflush(stdout);
  fflush(stderr);
  abort();
}

void abort_tb_cluster_dead_owner(unsigned owner_sm, unsigned issuer_sm,
                                 unsigned hw_cta, addr_t addr) {
  printf("GPGPU-Sim ERROR: DSM remote shared access to smid=%u failed "
         "(addr=0x%llx)\n",
         owner_sm, (unsigned long long)addr);
  abort_tb_cluster_dead_rank(owner_sm, issuer_sm, hw_cta);
}

bool dsm_fabric_enabled(shader_core_ctx *core) {
  return core && core->get_config() && core->get_config()->gpgpu_dsm_enable &&
         core->get_cluster() && core->get_cluster()->get_dsm_endpoint();
}

} // namespace flash_gpgpu_sim
