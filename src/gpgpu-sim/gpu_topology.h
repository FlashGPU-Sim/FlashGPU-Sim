#ifndef GPU_TOPOLOGY_H
#define GPU_TOPOLOGY_H

#include <vector>

#include "../option_parser.h"

// SM / GPC / CPC-slot map. All SM↔GPC↔slot conversion goes through this
// object. A CPC is six SM slots; leftover slots are PG'd (no shader core).
//
// Shader icnt nodes remain one per GPC (global_sm_node_id returns that
// GPC's node). Per-SM endpoints are a later change.

typedef unsigned sm_id_t;
typedef unsigned gpc_id_t;
typedef unsigned local_sm_id_t;
typedef unsigned global_icnt_node_id_t;

enum { k_sm_slots_per_cpc = 6 };

struct sm_location_t {
  sm_id_t sm_id;
  gpc_id_t gpc_id;
  local_sm_id_t local_sm_id;
  unsigned cpc_id;
  unsigned cpc_slot;  // 0..5; unused indices are PG'd
};

class gpu_topology_t {
 public:
  gpu_topology_t() = default;

  // Uniform map: first num_sms_per_gpc CPC slots in each GPC are enabled.
  // Aborts if enabled SMs do not fit in 6 * cpcs_per_gpc slots.
  void build(unsigned num_gpcs, unsigned num_sms_per_gpc,
             unsigned cpcs_per_gpc);

  unsigned num_gpcs() const { return m_num_gpcs; }
  unsigned num_sms() const { return m_num_sms; }
  unsigned num_sms_in_gpc(gpc_id_t gpc) const;
  unsigned num_cpc_slots_in_gpc(gpc_id_t gpc) const;
  unsigned cpcs_per_gpc() const { return m_cpcs_per_gpc; }

  sm_location_t locate_sm(sm_id_t sm_id) const;
  bool slot_is_enabled(gpc_id_t gpc, unsigned cpc_id,
                       unsigned cpc_slot) const;
  // local_sm == num_sms_in_gpc(gpc) is allowed as an exclusive range end.
  sm_id_t sm_id_at(gpc_id_t gpc, local_sm_id_t local_sm) const;
  gpc_id_t gpc_id_of_sm(sm_id_t sm_id) const;
  local_sm_id_t local_sm_of_sm(sm_id_t sm_id) const;

  // Today's shader icnt node is the GPC, not the SM.
  global_icnt_node_id_t global_sm_node_id(sm_id_t sm_id) const;
  global_icnt_node_id_t global_l2_node_id(unsigned subpartition_id) const;

  void set_live() const;

 private:
  unsigned m_num_gpcs = 0;
  unsigned m_sms_per_gpc = 0;
  unsigned m_cpcs_per_gpc = 0;
  unsigned m_num_sms = 0;
  unsigned m_slots_per_gpc = 0;
  std::vector<sm_location_t> m_sm;          // enabled SMs, index = sm_id
  std::vector<unsigned> m_slot_sm;          // gpc*slots + linear slot → sm or ~0u
};

const gpu_topology_t &gpu_topology_live();

// Canonicalize old/new topology knobs. Returns false on disagreement.
bool gpc_resolve_topology_aliases(bool old_gpcs_set, unsigned n_clusters,
                                  bool new_gpcs_set, unsigned num_gpcs,
                                  bool old_sms_set, unsigned n_cores,
                                  bool new_sms_set, unsigned num_sms_per_gpc,
                                  unsigned *out_gpcs, unsigned *out_sms,
                                  char *err, unsigned err_len);

// Reads option_parser_was_set for the four knobs and writes *n_clusters /
// *n_cores. Aborts if old and new disagree.
void gpc_apply_topology_aliases(option_parser_t opp, unsigned *n_clusters,
                                unsigned *n_cores, unsigned *num_gpcs_alias,
                                unsigned *num_sms_per_gpc_alias);

#endif
