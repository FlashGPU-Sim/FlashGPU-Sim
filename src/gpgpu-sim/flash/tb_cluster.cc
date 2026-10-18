#include "tb_cluster.h"
#include "cluster_fatal.h"

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
  // No TB-cluster launch: mapa rank 0 is this CTA, not "first rank-0 in the
  // GPC".
  if (group == (unsigned)-1)
    return peer == requester && slot == req_slot;
  if (peer->get_cta_cluster_group(slot) != group)
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
  const unsigned group = requester->get_cta_cluster_group(requester_cta_slot);
  if (group == (unsigned)-1 || cluster->tb_cluster_group_size(group) <= 1) {
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
      // An address produced while the owner was live remains valid while its
      // cluster resources are retained for a peer, even after its threads exit.
      if (!peer->get_cta_smem(slot))
        continue;
      const unsigned group =
          requester->get_cta_cluster_group(requester_cta_slot);
      if (group != (unsigned)-1 && peer->get_cta_cluster_group(slot) != group)
        continue;
      return fill_target(peer, slot, out);
    }
    return false;
  }
  return false;
}

namespace {
cluster_fatal_observer_fn g_cluster_fatal_observer = nullptr;
bool g_cluster_fatal_latched = false;
} // namespace

void set_cluster_fatal_observer(cluster_fatal_observer_fn observer) {
  g_cluster_fatal_observer = observer;
  g_cluster_fatal_latched = false;
}

bool cluster_fatal_latched() { return g_cluster_fatal_latched; }

} // namespace flash_gpgpu_sim

// Integration tests link the system libcudart and pick up this simulator
// library only at run time. Keep a C name they can call as a weak symbol.
extern "C" void flashgpu_set_cluster_fatal_observer(
    flash_gpgpu_sim::cluster_fatal_observer_fn observer) {
  flash_gpgpu_sim::set_cluster_fatal_observer(observer);
}

namespace flash_gpgpu_sim {

void abort_tb_cluster_dead_rank(unsigned rank, unsigned issuer_sm,
                                unsigned hw_cta) {
  const char *fmt =
      "GPGPU-Sim ERROR: mapa: cluster rank %u is not an active CTA "
      "(issuer smid=%u hw_cta=%u). The target CTA has exited or was "
      "never co-resident; refusing to alias the issuer's shared memory.\n";
  char message[512];
  snprintf(message, sizeof(message), fmt, rank, issuer_sm, hw_cta);
  printf("%s", message);
  fprintf(stderr, "%s", message);
  fflush(stdout);
  fflush(stderr);
  // A death-test re-exec of the integration binary does not fit in the CI
  // memory cgroup. Tests observe the same message. Do not throw: core_cycle
  // runs inside an OpenMP region.
  if (g_cluster_fatal_observer) {
    g_cluster_fatal_latched = true;
    g_cluster_fatal_observer(message);
    return;
  }
  abort();
}

void abort_tb_cluster_dead_owner(unsigned owner_sm, unsigned issuer_sm,
                                 unsigned hw_cta, addr_t addr) {
  printf("GPGPU-Sim ERROR: DSM remote shared access to smid=%u failed "
         "(addr=0x%llx)\n",
         owner_sm, (unsigned long long)addr);
  abort_tb_cluster_dead_rank(owner_sm, issuer_sm, hw_cta);
}

void abort_dsm_disabled(const char *what, unsigned issuer_sm, addr_t addr) {
  const char *fmt =
      "GPGPU-Sim ERROR: remote DSM %s requires -gpgpu_dsm_enable 1 "
      "(issuer smid=%u addr=0x%llx).\n";
  const char *op = what ? what : "access";
  printf(fmt, op, issuer_sm, (unsigned long long)addr);
  fprintf(stderr, fmt, op, issuer_sm, (unsigned long long)addr);
  fflush(stdout);
  fflush(stderr);
  abort();
}

bool dsm_fabric_enabled(shader_core_ctx *core) {
  return core && core->get_config() && core->get_config()->gpgpu_dsm_enable &&
         core->get_cluster() && core->get_cluster()->get_dsm_endpoint();
}

} // namespace flash_gpgpu_sim
