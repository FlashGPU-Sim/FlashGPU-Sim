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
[[noreturn]] void abort_dsm_disabled(const char *what, unsigned issuer_sm,
                                     addr_t addr);

bool dsm_fabric_enabled(shader_core_ctx *core);

// Generic shared window: SHARED_GENERIC_START + sm_id * SHARED_MEM_SIZE_MAX +
// off
inline bool decode_shared_generic(addr_t addr, unsigned *out_smid,
                                  addr_t *out_offset) {
  if (addr < SHARED_GENERIC_START)
    return false;
  const addr_t rel = addr - SHARED_GENERIC_START;
  if (rel >= TOTAL_SHARED_MEM)
    return false;
  const unsigned smid = static_cast<unsigned>(rel / SHARED_MEM_SIZE_MAX);
  const addr_t offset = static_cast<addr_t>(rel % SHARED_MEM_SIZE_MAX);
  if (out_smid)
    *out_smid = smid;
  if (out_offset)
    *out_offset = offset;
  return true;
}

inline bool is_remote_shared_generic(unsigned local_smid, addr_t addr,
                                     unsigned *out_owner_smid = nullptr,
                                     addr_t *out_offset = nullptr) {
  unsigned owner = 0;
  addr_t off = 0;
  if (!decode_shared_generic(addr, &owner, &off))
    return false;
  if (owner == local_smid)
    return false;
  if (out_owner_smid)
    *out_owner_smid = owner;
  if (out_offset)
    *out_offset = off;
  return true;
}

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
