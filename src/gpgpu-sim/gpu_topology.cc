#include "gpu_topology.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const gpu_topology_t *g_live_topology = nullptr;

void gpu_topology_t::build(unsigned num_gpcs, unsigned num_sms_per_gpc,
                           unsigned cpcs_per_gpc) {
  assert(num_gpcs > 0);
  assert(num_sms_per_gpc > 0);
  assert(cpcs_per_gpc > 0);
  m_slots_per_gpc = k_sm_slots_per_cpc * cpcs_per_gpc;
  if (num_sms_per_gpc > m_slots_per_gpc) {
    fprintf(stderr,
            "GPGPU-Sim ** ERROR: enabled SMs per GPC (%u) exceed CPC slots "
            "(%u = %u CPCs * %d slots). Increase -gpgpu_dsm_cpcs_per_gpc.\n",
            num_sms_per_gpc, m_slots_per_gpc, cpcs_per_gpc, k_sm_slots_per_cpc);
    abort();
  }
  m_num_gpcs = num_gpcs;
  m_sms_per_gpc = num_sms_per_gpc;
  m_cpcs_per_gpc = cpcs_per_gpc;
  m_num_sms = num_gpcs * num_sms_per_gpc;
  m_sm.resize(m_num_sms);
  m_slot_sm.assign((size_t)num_gpcs * m_slots_per_gpc, ~0u);

  for (unsigned g = 0; g < num_gpcs; g++) {
    for (unsigned local = 0; local < num_sms_per_gpc; local++) {
      const sm_id_t sm = g * num_sms_per_gpc + local;
      sm_location_t loc;
      loc.sm_id = sm;
      loc.gpc_id = g;
      loc.local_sm_id = local;
      loc.cpc_id = local / k_sm_slots_per_cpc;
      loc.cpc_slot = local % k_sm_slots_per_cpc;
      m_sm[sm] = loc;
      m_slot_sm[(size_t)g * m_slots_per_gpc + local] = sm;
    }
  }
}

unsigned gpu_topology_t::num_sms_in_gpc(gpc_id_t gpc) const {
  assert(gpc < m_num_gpcs);
  return m_sms_per_gpc;
}

unsigned gpu_topology_t::num_cpc_slots_in_gpc(gpc_id_t gpc) const {
  assert(gpc < m_num_gpcs);
  return m_slots_per_gpc;
}

sm_location_t gpu_topology_t::locate_sm(sm_id_t sm_id) const {
  assert(sm_id < m_num_sms);
  return m_sm[sm_id];
}

bool gpu_topology_t::slot_is_enabled(gpc_id_t gpc, unsigned cpc_id,
                                     unsigned cpc_slot) const {
  assert(gpc < m_num_gpcs);
  assert(cpc_id < m_cpcs_per_gpc);
  assert(cpc_slot < (unsigned)k_sm_slots_per_cpc);
  const unsigned linear = cpc_id * k_sm_slots_per_cpc + cpc_slot;
  return m_slot_sm[(size_t)gpc * m_slots_per_gpc + linear] != ~0u;
}

sm_id_t gpu_topology_t::sm_id_at(gpc_id_t gpc, local_sm_id_t local_sm) const {
  assert(gpc < m_num_gpcs);
  // local_sm == num_sms_in_gpc is the exclusive end of the GPC's SM range
  // (legacy cid_to_sid(n_cores, gpc) used by stats aggregation).
  assert(local_sm <= m_sms_per_gpc);
  return gpc * m_sms_per_gpc + local_sm;
}

gpc_id_t gpu_topology_t::gpc_id_of_sm(sm_id_t sm_id) const {
  return locate_sm(sm_id).gpc_id;
}

local_sm_id_t gpu_topology_t::local_sm_of_sm(sm_id_t sm_id) const {
  return locate_sm(sm_id).local_sm_id;
}

global_icnt_node_id_t gpu_topology_t::global_sm_node_id(sm_id_t sm_id) const {
  assert(sm_id < m_num_sms);
  return sm_id;
}

global_icnt_node_id_t gpu_topology_t::global_l2_node_id(
    unsigned subpartition_id) const {
  return m_num_sms + subpartition_id;
}

void gpu_topology_t::set_live() const { g_live_topology = this; }

const gpu_topology_t &gpu_topology_live() {
  assert(g_live_topology);
  return *g_live_topology;
}

bool gpc_resolve_topology_aliases(bool old_gpcs_set, unsigned n_clusters,
                                  bool new_gpcs_set, unsigned num_gpcs,
                                  bool old_sms_set, unsigned n_cores,
                                  bool new_sms_set, unsigned num_sms_per_gpc,
                                  unsigned *out_gpcs, unsigned *out_sms,
                                  char *err, unsigned err_len) {
  auto pair = [&](bool old_set, unsigned old_val, bool new_set,
                  unsigned new_val, unsigned *out, const char *old_name,
                  const char *new_name) -> bool {
    if (old_set && new_set && old_val != new_val) {
      if (err && err_len) {
        snprintf(err, err_len,
                 "GPGPU-Sim ** ERROR: topology knobs disagree: %s=%u vs "
                 "%s=%u\n",
                 old_name, old_val, new_name, new_val);
      }
      return false;
    }
    if (new_set)
      *out = new_val;
    else
      *out = old_val;
    return true;
  };
  if (!pair(old_gpcs_set, n_clusters, new_gpcs_set, num_gpcs, out_gpcs,
            "-gpgpu_n_clusters", "-gpgpu_num_gpcs"))
    return false;
  if (!pair(old_sms_set, n_cores, new_sms_set, num_sms_per_gpc, out_sms,
            "-gpgpu_n_cores_per_cluster", "-gpgpu_num_sms_per_gpc"))
    return false;
  return true;
}

void gpc_apply_topology_aliases(option_parser_t opp, unsigned *n_clusters,
                                unsigned *n_cores, unsigned *num_gpcs_alias,
                                unsigned *num_sms_per_gpc_alias) {
  unsigned out_gpcs = *n_clusters;
  unsigned out_sms = *n_cores;
  char err[256];
  err[0] = 0;
  const bool old_gpcs = opp && option_parser_was_set(opp, "-gpgpu_n_clusters");
  const bool new_gpcs = opp && option_parser_was_set(opp, "-gpgpu_num_gpcs");
  const bool old_sms =
      opp && option_parser_was_set(opp, "-gpgpu_n_cores_per_cluster");
  const bool new_sms =
      opp && option_parser_was_set(opp, "-gpgpu_num_sms_per_gpc");
  if (!gpc_resolve_topology_aliases(old_gpcs, *n_clusters, new_gpcs,
                                    *num_gpcs_alias, old_sms, *n_cores, new_sms,
                                    *num_sms_per_gpc_alias, &out_gpcs, &out_sms,
                                    err, sizeof(err))) {
    fprintf(stderr, "%s", err);
    abort();
  }
  *n_clusters = out_gpcs;
  *n_cores = out_sms;
  *num_gpcs_alias = out_gpcs;
  *num_sms_per_gpc_alias = out_sms;
}
