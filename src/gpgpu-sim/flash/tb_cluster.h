#ifndef FLASH_GPGPU_SIM_TB_CLUSTER_H
#define FLASH_GPGPU_SIM_TB_CLUSTER_H

#include <cstdint>
#include <vector>

#include "../../abstract_hardware_model.h"

class memory_space;
class shader_core_ctx;
class ptx_thread_info;
class ptx_instruction;

namespace flash_gpgpu_sim {

struct tb_cluster_target_t {
  shader_core_ctx *core = nullptr;
  unsigned sm_id = 0;
  unsigned local_sm = 0;
  unsigned cta_slot = 0;
  memory_space *smem = nullptr;
};

// One TB-cluster lookup: active CTA, same group, same GPC, smem object.
// Rank form is used by mapa / TMA / mbarrier peer walks. Owner-SM form is
// used after generic-address decode (logical window kept, offset stripped).
bool resolve_tb_cluster_rank(shader_core_ctx *requester,
                             unsigned requester_cta_slot, unsigned target_rank,
                             tb_cluster_target_t *out);
bool resolve_tb_cluster_owner_sm(shader_core_ctx *requester,
                                 unsigned requester_cta_slot,
                                 unsigned owner_sm_id,
                                 tb_cluster_target_t *out);

[[noreturn]] void abort_tb_cluster_dead_rank(unsigned rank, unsigned issuer_sm,
                                             unsigned hw_cta);
[[noreturn]] void abort_tb_cluster_dead_owner(unsigned owner_sm,
                                              unsigned issuer_sm,
                                              unsigned hw_cta, addr_t addr);

bool dsm_fabric_enabled(shader_core_ctx *core);

template <typename Fn>
void for_each_tb_cluster_peer(shader_core_ctx *core, unsigned issuer_hw_cta,
                              Fn &&fn, bool include_issuer = false,
                              bool use_mask = false,
                              uint16_t cta_mask = 0xFFFF) {
  if (!core)
    return;
  for (unsigned rank = 0; rank < 16; rank++) {
    if (use_mask && ((cta_mask >> rank) & 1u) == 0)
      continue;
    tb_cluster_target_t t;
    if (!resolve_tb_cluster_rank(core, issuer_hw_cta, rank, &t))
      continue;
    if (!include_issuer && t.core == core && t.cta_slot == issuer_hw_cta)
      continue;
    fn(t.core, t.cta_slot);
  }
}

enum class dsm_op_kind { store, load, atom_add };

struct dsm_lane_op_t {
  dsm_op_kind kind = dsm_op_kind::store;
  unsigned dst_local = 0;
  unsigned cta_slot = 0;
  unsigned cta_gen = 0;
  addr_t offset = 0;
  unsigned bytes = 0;
  std::vector<uint8_t> data;
  ptx_thread_info *thread = nullptr;
  const ptx_instruction *pI = nullptr;
  unsigned type = 0;
};

} // namespace flash_gpgpu_sim

#endif
