#ifndef GPU_TOPOLOGY_H
#define GPU_TOPOLOGY_H

#include <list>
#include <vector>

#include "../option_parser.h"

class mem_fetch;

// SM / GPC / CPC-slot map. All SM↔GPC↔slot conversion goes through this
// object. A CPC is six SM slots; leftover slots are PG'd (no shader core).
//
// Each enabled SM is a global interconnect node. L2 subpartitions start at
// num_sms. A GPC id is never a node.

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

  // Per-GPC enabled SM counts (e.g. H200 6×16 + 2×18). Length == num_gpcs.
  void build(unsigned num_gpcs, const std::vector<unsigned> &sms_per_gpc,
             unsigned cpcs_per_gpc);

  // Parse "-gpgpu_gpc_sms 16,16,16,16,16,16,18,18". Returns false on error.
  static bool parse_gpc_sms(const char *s, unsigned num_gpcs,
                            std::vector<unsigned> *out, char *err,
                            unsigned err_len);

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

  // Shader icnt node of an enabled SM; equal to sm_id (dense enabled set).
  global_icnt_node_id_t global_sm_node_id(sm_id_t sm_id) const;
  // L2 subpartition node; does not overlap any SM node.
  global_icnt_node_id_t global_l2_node_id(unsigned subpartition_id) const;

  void set_live() const;

 private:
  unsigned m_num_gpcs = 0;
  unsigned m_sms_per_gpc = 0;  // uniform count, or max if hetero
  unsigned m_cpcs_per_gpc = 0;
  unsigned m_num_sms = 0;
  unsigned m_slots_per_gpc = 0;
  std::vector<unsigned> m_sms_in_gpc;
  std::vector<unsigned> m_gpc_sm_base;      // prefix: first sm_id of GPC
  std::vector<sm_location_t> m_sm;          // enabled SMs, index = sm_id
  std::vector<unsigned> m_slot_sm;          // gpc*slots + linear slot → sm or ~0u
};

const gpu_topology_t &gpu_topology_live();

// Local SM visited at step i of a GPC CTA-issue pass. rr_start is the
// previous-cycle cursor; snapshot it before the loop so issuing on one SM
// does not skip a sibling.
inline unsigned gpc_cta_issue_visit(unsigned i, unsigned rr_start,
                                    unsigned n_local_sms) {
  return (i + rr_start + 1) % n_local_sms;
}

// Per-enabled-SM icnt ejection buffers on one GPC. One SM at capacity does
// not head-of-line block a sibling SM in the same GPC.
struct gpc_sm_response_fifos_t {
  void init(unsigned n_local_sms, unsigned ejection_limit) {
    m_limit = ejection_limit;
    m_q.assign(n_local_sms, std::list<mem_fetch *>());
  }
  unsigned ejection_limit() const { return m_limit; }
  unsigned n_local_sms() const { return (unsigned)m_q.size(); }
  bool full(unsigned local_sm) const {
    return m_q[local_sm].size() >= m_limit;
  }
  bool empty(unsigned local_sm) const { return m_q[local_sm].empty(); }
  std::size_t size(unsigned local_sm) const { return m_q[local_sm].size(); }
  void push(unsigned local_sm, mem_fetch *mf) { m_q[local_sm].push_back(mf); }
  mem_fetch *front(unsigned local_sm) const { return m_q[local_sm].front(); }
  void pop(unsigned local_sm) { m_q[local_sm].pop_front(); }
  const std::list<mem_fetch *> &at(unsigned local_sm) const {
    return m_q[local_sm];
  }

 private:
  unsigned m_limit = 0;
  std::vector<std::list<mem_fetch *>> m_q;
};

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
