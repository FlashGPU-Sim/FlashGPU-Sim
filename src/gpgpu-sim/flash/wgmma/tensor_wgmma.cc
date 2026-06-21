#include "tensor_wgmma.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../../gpu-sim.h"
#include "../../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

unsigned wgmma_warp_base_tid(core_t *core, const warp_inst_t &inst) {
  if (core->get_gpu()->is_functional_sim())
    return inst.warp_id_func() * core->get_warp_size();
  return inst.warp_id() * core->get_warp_size();
}

int wgmma_scalar_type_at(const ptx_instruction *pI, unsigned index,
                         int default_type) {
  const std::list<int> &scalar_types = pI->get_scalar_type();
  if (scalar_types.size() <= index)
    return default_type;

  std::list<int>::const_iterator it = scalar_types.begin();
  for (unsigned i = 0; i < index; ++i)
    ++it;
  return *it;
}

int wgmma_opcode(const warp_inst_t *inst) {
  assert(inst != NULL);
  const ptx_instruction *ptx_inst = static_cast<const ptx_instruction *>(inst);
  return ptx_inst->get_opcode();
}

bool is_wgmma_mma_async_opcode(int opcode) {
  return opcode == WGMMA_MMA_ASYNC_OP || opcode == WGMMA_MMA_ASYNC_SP_OP;
}

bool wgmma_uses_register_a_operand(const warp_inst_t *inst) {
  assert(inst != NULL);
  const ptx_instruction *ptx_inst = static_cast<const ptx_instruction *>(inst);
  if (ptx_inst->get_num_operands() < 2)
    return false;
  return ptx_inst->operand_lookup(1).is_vector();
}

class wgmma_group_manager_t {
public:
  struct wait_result_t {
    bool satisfied = true;
    std::vector<unsigned> released_warps;
  };

  void add_op(unsigned cta_id, unsigned warpgroup_id, unsigned op_uid);
  void commit_group(unsigned cta_id, unsigned warpgroup_id);
  wait_result_t wait_group(unsigned cta_id, unsigned warpgroup_id,
                           unsigned max_pending_groups,
                           const unsigned *warp_ids, unsigned count);
  wait_result_t complete_op(unsigned cta_id, unsigned warpgroup_id,
                            unsigned op_uid);
  void cleanup_cta(unsigned cta_id);

private:
  typedef std::pair<unsigned, unsigned> key_t;

  struct group_t {
    unsigned group_id = 0;
    std::set<unsigned> op_uids;
  };

  struct warpgroup_info_t {
    unsigned next_group_id = 1;
    std::set<unsigned> pending_ops;
    std::map<unsigned, unsigned> op_to_group;
    std::map<unsigned, group_t> pending_groups;
    bool is_waiting = false;
    unsigned wait_allowance = 0;
    std::vector<unsigned> waiting_warps;

    void add_op(unsigned op_uid);
    void commit_group();
    wait_result_t wait_group(unsigned max_pending_groups,
                             const unsigned *warp_ids, unsigned count);
    wait_result_t complete_op(unsigned op_uid);
    bool wait_satisfied() const;
    wait_result_t release_if_satisfied();
  };

  std::map<key_t, warpgroup_info_t> m_warpgroup_info;
};

void wgmma_group_manager_t::warpgroup_info_t::add_op(unsigned op_uid) {
  pending_ops.insert(op_uid);
}

void wgmma_group_manager_t::warpgroup_info_t::commit_group() {
  group_t group;
  group.group_id = next_group_id++;
  group.op_uids.swap(pending_ops);

  if (group.op_uids.empty())
    return;

  for (std::set<unsigned>::const_iterator it = group.op_uids.begin();
       it != group.op_uids.end(); ++it) {
    op_to_group[*it] = group.group_id;
  }
  pending_groups[group.group_id] = group;
}

bool wgmma_group_manager_t::warpgroup_info_t::wait_satisfied() const {
  return pending_groups.size() <= wait_allowance;
}

wgmma_group_manager_t::wait_result_t
wgmma_group_manager_t::warpgroup_info_t::release_if_satisfied() {
  wait_result_t result;
  if (!is_waiting || !wait_satisfied())
    return result;

  result.satisfied = true;
  result.released_warps = waiting_warps;
  waiting_warps.clear();
  is_waiting = false;
  return result;
}

wgmma_group_manager_t::wait_result_t
wgmma_group_manager_t::warpgroup_info_t::wait_group(unsigned max_pending_groups,
                                                    const unsigned *warp_ids,
                                                    unsigned count) {
  assert(!is_waiting && "Warpgroup is already waiting on a WGMMA group");
  wait_allowance = max_pending_groups;

  wait_result_t result;
  if (wait_satisfied())
    return result;

  result.satisfied = false;
  is_waiting = true;
  waiting_warps.assign(warp_ids, warp_ids + count);
  return result;
}

wgmma_group_manager_t::wait_result_t
wgmma_group_manager_t::warpgroup_info_t::complete_op(unsigned op_uid) {
  std::set<unsigned>::iterator pending_it = pending_ops.find(op_uid);
  if (pending_it != pending_ops.end()) {
    pending_ops.erase(pending_it);
    return release_if_satisfied();
  }

  std::map<unsigned, unsigned>::iterator op_it = op_to_group.find(op_uid);
  if (op_it == op_to_group.end())
    return release_if_satisfied();

  unsigned group_id = op_it->second;
  op_to_group.erase(op_it);

  std::map<unsigned, group_t>::iterator group_it =
      pending_groups.find(group_id);
  assert(group_it != pending_groups.end());
  group_it->second.op_uids.erase(op_uid);
  if (group_it->second.op_uids.empty())
    pending_groups.erase(group_it);

  return release_if_satisfied();
}

void wgmma_group_manager_t::add_op(unsigned cta_id, unsigned warpgroup_id,
                                   unsigned op_uid) {
  m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].add_op(op_uid);
}

void wgmma_group_manager_t::commit_group(unsigned cta_id,
                                         unsigned warpgroup_id) {
  m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].commit_group();
}

wgmma_group_manager_t::wait_result_t
wgmma_group_manager_t::wait_group(unsigned cta_id, unsigned warpgroup_id,
                                  unsigned max_pending_groups,
                                  const unsigned *warp_ids, unsigned count) {
  return m_warpgroup_info[std::make_pair(cta_id, warpgroup_id)].wait_group(
      max_pending_groups, warp_ids, count);
}

wgmma_group_manager_t::wait_result_t
wgmma_group_manager_t::complete_op(unsigned cta_id, unsigned warpgroup_id,
                                   unsigned op_uid) {
  std::map<key_t, warpgroup_info_t>::iterator it =
      m_warpgroup_info.find(std::make_pair(cta_id, warpgroup_id));
  if (it == m_warpgroup_info.end())
    return wait_result_t();
  return it->second.complete_op(op_uid);
}

void wgmma_group_manager_t::cleanup_cta(unsigned cta_id) {
  for (std::map<key_t, warpgroup_info_t>::iterator it =
           m_warpgroup_info.begin();
       it != m_warpgroup_info.end();) {
    if (it->first.first == cta_id) {
      it = m_warpgroup_info.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace

unsigned wgmma_thread_count(core_t *core, const warp_inst_t &inst) {
  if (inst.is_wgmma_warpgroup())
    return inst.wgmma_warpgroup_size() * core->get_warp_size();
  return core->get_warp_size();
}

unsigned wgmma_hw_tid(core_t *core, const warp_inst_t &inst,
                      unsigned thread_idx) {
  unsigned warp_size = core->get_warp_size();
  if (inst.is_wgmma_warpgroup()) {
    unsigned warpgroup_slot = thread_idx / warp_size;
    unsigned lane = thread_idx % warp_size;
    return inst.wgmma_warpgroup_warp_id(warpgroup_slot) * warp_size + lane;
  }
  return wgmma_warp_base_tid(core, inst) + thread_idx;
}

ptx_thread_info *wgmma_thread(core_t *core, const warp_inst_t &inst,
                              unsigned thread_idx) {
  ptx_thread_info *thread =
      core->get_thread_info()[wgmma_hw_tid(core, inst, thread_idx)];
  assert(thread != NULL);
  return thread;
}

unsigned wgmma_lane(const ptx_thread_info *thread) {
  return static_cast<unsigned>(thread->get_flat_tid()) % 128;
}

uint64_t wgmma_gmma_desc_base(uint64_t desc) { return (desc & 0x3FFFULL) << 4; }

uint64_t wgmma_gmma_desc_leading_byte_offset(uint64_t desc) {
  return ((desc >> 16) & 0x3FFFULL) << 4;
}

uint64_t wgmma_gmma_desc_stride_byte_offset(uint64_t desc) {
  return ((desc >> 32) & 0x3FFFULL) << 4;
}

namespace {

unsigned wgmma_gmma_desc_swizzle_mode(uint64_t desc) {
  return static_cast<unsigned>((desc >> 62) & 0x3ULL);
}

uint64_t wgmma_apply_gmma_swizzle(uint64_t byte_offset, unsigned swizzle_mode) {
  unsigned bits = 0;
  switch (swizzle_mode) {
  case 1:
    bits = 3; // 128-byte swizzle.
    break;
  case 2:
    bits = 2; // 64-byte swizzle.
    break;
  case 3:
    bits = 1; // 32-byte swizzle.
    break;
  default:
    return byte_offset;
  }

  uint64_t y_mask = ((1ULL << bits) - 1ULL) << (4 + 3);
  return byte_offset ^ ((byte_offset & y_mask) >> 3);
}

unsigned wgmma_swizzle_bytes(unsigned swizzle_mode) {
  switch (swizzle_mode) {
  case 1:
    return 128;
  case 2:
    return 64;
  case 3:
    return 32;
  default:
    return 0;
  }
}

uint64_t wgmma_swizzle_period_bytes(unsigned swizzle_mode) {
  unsigned swizzle_bytes = wgmma_swizzle_bytes(swizzle_mode);
  return swizzle_bytes == 0 ? 0 : static_cast<uint64_t>(swizzle_bytes) * 8;
}

uint64_t wgmma_apply_gmma_swizzle(uint64_t base, uint64_t byte_offset,
                                  unsigned swizzle_mode) {
  uint64_t period = wgmma_swizzle_period_bytes(swizzle_mode);
  if (period == 0)
    return base + byte_offset;

  // K-slice descriptors can start inside a swizzle period, so keep the base
  // low bits in the swizzle coordinate instead of swizzling only byte_offset.
  uint64_t period_base = base & ~(period - 1);
  uint64_t period_offset = (base - period_base) + byte_offset;
  return period_base + wgmma_apply_gmma_swizzle(period_offset, swizzle_mode);
}

} // namespace

unsigned wgmma_gmma_k_major_smem_addr(uint64_t desc, int col, int k,
                                      unsigned element_size,
                                      unsigned default_contiguous_k) {
  uint64_t base = wgmma_gmma_desc_base(desc);
  uint64_t leading = wgmma_gmma_desc_leading_byte_offset(desc);
  uint64_t stride = wgmma_gmma_desc_stride_byte_offset(desc);
  unsigned swizzle_mode = wgmma_gmma_desc_swizzle_mode(desc);
  unsigned swizzle_bytes = wgmma_swizzle_bytes(swizzle_mode);

  if (swizzle_bytes != 0) {
    unsigned elements_per_128b = 16 / element_size;
    uint64_t row_group = static_cast<uint64_t>(col / 8);
    unsigned row_in_group = static_cast<unsigned>(col % 8);
    uint64_t k_group = static_cast<uint64_t>(k / elements_per_128b);
    unsigned k_in_group = static_cast<unsigned>(k % elements_per_128b);
    uint64_t leading_offset =
        leading == 0 ? static_cast<uint64_t>(16) : leading;

    uint64_t offset = row_group * stride + row_in_group * swizzle_bytes +
                      k_group * leading_offset + k_in_group * element_size;
    return static_cast<unsigned>(
        wgmma_apply_gmma_swizzle(base, offset, swizzle_mode));
  }

  unsigned contiguous_k = stride == 0
                              ? default_contiguous_k
                              : static_cast<unsigned>(stride / element_size);
  if (contiguous_k == 0)
    contiguous_k = default_contiguous_k;

  return static_cast<unsigned>(base + (k / contiguous_k) * leading +
                               col * stride +
                               (k % contiguous_k) * element_size);
}

unsigned wgmma_gmma_mn_major_smem_addr(uint64_t desc, int row, int k,
                                       unsigned element_size) {
  uint64_t base = wgmma_gmma_desc_base(desc);
  uint64_t leading = wgmma_gmma_desc_leading_byte_offset(desc);
  uint64_t stride = wgmma_gmma_desc_stride_byte_offset(desc);
  unsigned swizzle_mode = wgmma_gmma_desc_swizzle_mode(desc);
  unsigned swizzle_bytes = wgmma_swizzle_bytes(swizzle_mode);
  unsigned elements_per_128b = 16 / element_size;

  if (swizzle_bytes == 0) {
    unsigned contiguous_m = elements_per_128b;
    uint64_t row_group = static_cast<uint64_t>(row / contiguous_m);
    unsigned row_in_group = static_cast<unsigned>(row % contiguous_m);
    uint64_t k_group = static_cast<uint64_t>(k / 8);
    unsigned k_in_group = static_cast<unsigned>(k % 8);
    uint64_t offset = row_group * stride + row_in_group * element_size +
                      k_group * leading + k_in_group * 16;
    return static_cast<unsigned>(base + offset);
  }

  unsigned rows_per_swizzle = swizzle_bytes / element_size;
  uint64_t row_group = static_cast<uint64_t>(row / rows_per_swizzle);
  unsigned row_in_group = static_cast<unsigned>(row % rows_per_swizzle);
  uint64_t k_group = static_cast<uint64_t>(k / 8);
  unsigned k_in_group = static_cast<unsigned>(k % 8);
  uint64_t offset = row_group * leading + row_in_group * element_size +
                    k_group * stride + k_in_group * swizzle_bytes;
  return static_cast<unsigned>(
      wgmma_apply_gmma_swizzle(base, offset, swizzle_mode));
}

void wgmma_m64n8_accumulator_coord(unsigned lane, int reg, int &row, int &col) {
  int tid_mma_col = lane % 4;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;

  int reg_col = reg & 0x1;
  int reg_row_block = (reg >> 1) & 0x1;

  row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  col = 2 * tid_mma_col + reg_col;
}

void tensor_wgmma_impl(const ptx_instruction *pI, core_t *core,
                       warp_inst_t &inst) {
  if (pI->is_wgmma_sparse()) {
    fprintf(stderr,
            "GPGPU-Sim: ERROR - sparse WGMMA is not functionally supported\n");
    return;
  }

  int accumulator_type = wgmma_scalar_type_at(pI, 0, F32_TYPE);
  int a_type = wgmma_scalar_type_at(pI, 1, F16_TYPE);
  int b_type = wgmma_scalar_type_at(pI, 2, a_type);

  if (accumulator_type == F32_TYPE && a_type == F16_TYPE &&
      b_type == F16_TYPE && pI->get_wgmma_shape_n() >= 8 &&
      pI->get_wgmma_shape_n() <= 256 && (pI->get_wgmma_shape_n() % 8) == 0 &&
      pI->get_wgmma_shape_k() == 16) {
    wgmma_m64nXk16_f16_impl(pI, core, inst);
    return;
  }

  if (accumulator_type == F32_TYPE && a_type == BF16_TYPE &&
      b_type == BF16_TYPE && pI->get_wgmma_shape_n() == 8 &&
      pI->get_wgmma_shape_k() == 16) {
    wgmma_m64n8k16_bf16_impl(pI, core, inst);
    return;
  }

  if (accumulator_type == F32_TYPE && a_type == TF32_TYPE &&
      b_type == TF32_TYPE && pI->get_wgmma_shape_n() == 8 &&
      pI->get_wgmma_shape_k() == 8) {
    wgmma_m64n8k8_tf32_impl(pI, core, inst);
    return;
  }

  if (accumulator_type == F32_TYPE &&
      (a_type == E4M3_TYPE || a_type == E5M2_TYPE) &&
      (b_type == E4M3_TYPE || b_type == E5M2_TYPE) &&
      pI->get_wgmma_shape_n() == 8 && pI->get_wgmma_shape_k() == 32) {
    wgmma_m64n8k32_fp8_impl(pI, core, inst, a_type == E4M3_TYPE,
                            b_type == E4M3_TYPE);
    return;
  }

  if (accumulator_type == S32_TYPE &&
      (a_type == S8_TYPE || a_type == U8_TYPE) &&
      (b_type == S8_TYPE || b_type == U8_TYPE) &&
      pI->get_wgmma_shape_n() == 8 && pI->get_wgmma_shape_k() == 32) {
    wgmma_m64n8k32_int8_impl(pI, core, inst, a_type == S8_TYPE,
                             b_type == S8_TYPE);
    return;
  }

  if (accumulator_type == S32_TYPE && a_type == B1_TYPE && b_type == B1_TYPE &&
      pI->get_wgmma_shape_n() == 8 && pI->get_wgmma_shape_k() == 256) {
    wgmma_m64n8k256_b1_impl(pI, core, inst);
    return;
  }

  fprintf(stderr,
          "GPGPU-Sim: ERROR - unsupported WGMMA variant "
          "m64n%dk%d type tuple (%d, %d, %d)\n",
          pI->get_wgmma_shape_n(), pI->get_wgmma_shape_k(), accumulator_type,
          a_type, b_type);
  assert(0 && "unsupported WGMMA variant");
  std::abort();
}

void wgmma_mma_async_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  tensor_wgmma_impl(pI, core, inst);
}

void wgmma_mma_async_sp_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
  fprintf(stderr,
          "GPGPU-Sim: ERROR - sparse WGMMA is not functionally supported\n");
}

void wgmma_fence_impl(const ptx_instruction *pI, core_t *core,
                      warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void wgmma_commit_group_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void wgmma_wait_group_impl(const ptx_instruction *pI, core_t *core,
                           warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void setmaxnreg_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

class wgmma_unit_t::impl_t {
public:
  impl_t(barrier_set_t *barriers, const shader_core_config *config)
      : m_barriers(barriers), m_config(config) {
    assert(m_barriers != NULL);
    assert(m_config != NULL);
  }

  bool issue_chain_ready(const warp_inst_t *inst,
                         unsigned long long cycle) const;
  void record_issue_chain(const warp_inst_t *inst, unsigned long long cycle);
  void add_op(unsigned cta_id, unsigned warpgroup_id, unsigned op_uid,
              unsigned compute_latency, unsigned completion_tail_latency,
              unsigned long long rf_traffic_tokens);
  unsigned long long drain_rf_traffic(unsigned long long bytes);
  unsigned long long rf_traffic_backlog() const;
  void commit_group(unsigned cta_id, unsigned warpgroup_id);
  void wait_group(unsigned cta_id, unsigned warpgroup_id,
                  unsigned max_pending_groups, const unsigned *warp_ids,
                  unsigned count);
  void cycle();
  void cleanup_cta(unsigned cta_id);

private:
  typedef std::pair<unsigned, unsigned> key_t;

  struct pending_completion_t {
    key_t key;
    unsigned op_uid = 0;
    unsigned compute_remaining = 0;
    unsigned remaining = 0;
    unsigned long long rf_traffic_remaining = 0;
  };

  barrier_set_t *m_barriers;
  const shader_core_config *m_config;
  wgmma_group_manager_t m_group_manager;
  std::vector<pending_completion_t> m_pending_completions;
  unsigned long long m_issue_chain_next_cycle = 0;
  unsigned long long m_issue_chain_last_cycle = 0;
  unsigned m_issue_chain_depth = 0;
};

bool wgmma_unit_t::impl_t::issue_chain_ready(const warp_inst_t *inst,
                                             unsigned long long cycle) const {
  if (!is_wgmma_mma_async_opcode(wgmma_opcode(inst)))
    return true;

  const unsigned *chain_config =
      wgmma_uses_register_a_operand(inst)
          ? m_config->gpgpu_wgmma_issue_chain_rs_config
          : m_config->gpgpu_wgmma_issue_chain_ss_config;
  if (chain_config[0] == 0)
    return true;

  return cycle >= m_issue_chain_next_cycle;
}

void wgmma_unit_t::impl_t::record_issue_chain(const warp_inst_t *inst,
                                              unsigned long long cycle) {
  if (!is_wgmma_mma_async_opcode(wgmma_opcode(inst)))
    return;

  const unsigned *chain_config =
      wgmma_uses_register_a_operand(inst)
          ? m_config->gpgpu_wgmma_issue_chain_rs_config
          : m_config->gpgpu_wgmma_issue_chain_ss_config;
  const unsigned depth_limit = chain_config[0];
  if (depth_limit == 0)
    return;

  const unsigned startup_gap = chain_config[1];
  const unsigned fast_gap = chain_config[2];
  const unsigned slow_gap = chain_config[3];
  const unsigned reset_gap = chain_config[4];

  if (m_issue_chain_depth != 0 && reset_gap != 0 &&
      cycle >= m_issue_chain_last_cycle + reset_gap) {
    m_issue_chain_depth = 0;
  }

  const unsigned current_ordinal = m_issue_chain_depth + 1;
  unsigned next_gap = fast_gap;
  if (m_issue_chain_depth == 0 && startup_gap != 0) {
    next_gap = startup_gap;
  } else if (current_ordinal >= depth_limit) {
    next_gap = slow_gap;
  }

  m_issue_chain_depth = current_ordinal;
  m_issue_chain_last_cycle = cycle;
  m_issue_chain_next_cycle = cycle + next_gap;
}

void wgmma_unit_t::impl_t::add_op(unsigned cta_id, unsigned warpgroup_id,
                                  unsigned op_uid, unsigned compute_latency,
                                  unsigned completion_tail_latency,
                                  unsigned long long rf_traffic_tokens) {
  m_group_manager.add_op(cta_id, warpgroup_id, op_uid);

  compute_latency = std::max(1u, compute_latency);

  unsigned compute_tail_remaining = 0;
  for (std::vector<pending_completion_t>::const_iterator it =
           m_pending_completions.begin();
       it != m_pending_completions.end(); ++it) {
    compute_tail_remaining =
        std::max(compute_tail_remaining, it->compute_remaining);
  }
  const unsigned compute_done_remaining =
      compute_tail_remaining + compute_latency;

  pending_completion_t pending;
  pending.key = std::make_pair(cta_id, warpgroup_id);
  pending.op_uid = op_uid;
  // The tensor compute segment consumes the per-SM WGMMA backend and is
  // serialized. The async completion tail models scoreboard/writeback time that
  // may overlap with later WGMMA compute.
  pending.compute_remaining = compute_done_remaining;
  pending.remaining = compute_done_remaining + completion_tail_latency;
  pending.rf_traffic_remaining = rf_traffic_tokens;
  m_pending_completions.push_back(pending);
}

unsigned long long
wgmma_unit_t::impl_t::drain_rf_traffic(unsigned long long bytes) {
  unsigned long long drained = 0;
  for (std::vector<pending_completion_t>::iterator it =
           m_pending_completions.begin();
       it != m_pending_completions.end() && bytes > 0; ++it) {
    if (it->rf_traffic_remaining == 0)
      continue;
    const unsigned long long take = std::min(bytes, it->rf_traffic_remaining);
    it->rf_traffic_remaining -= take;
    bytes -= take;
    drained += take;
  }
  return drained;
}

unsigned long long wgmma_unit_t::impl_t::rf_traffic_backlog() const {
  unsigned long long backlog = 0;
  for (std::vector<pending_completion_t>::const_iterator it =
           m_pending_completions.begin();
       it != m_pending_completions.end(); ++it) {
    backlog += it->rf_traffic_remaining;
  }
  return backlog;
}

void wgmma_unit_t::impl_t::commit_group(unsigned cta_id,
                                        unsigned warpgroup_id) {
  m_group_manager.commit_group(cta_id, warpgroup_id);
}

void wgmma_unit_t::impl_t::wait_group(unsigned cta_id, unsigned warpgroup_id,
                                      unsigned max_pending_groups,
                                      const unsigned *warp_ids,
                                      unsigned count) {
  wgmma_group_manager_t::wait_result_t result = m_group_manager.wait_group(
      cta_id, warpgroup_id, max_pending_groups, warp_ids, count);
  if (!result.satisfied)
    m_barriers->set_wgmma_waiting_warps(warp_ids, count);
}

void wgmma_unit_t::impl_t::cycle() {
  wgmma_group_manager_t::wait_result_t result;
  for (std::vector<pending_completion_t>::iterator it =
           m_pending_completions.begin();
       it != m_pending_completions.end(); ++it) {
    if (it->compute_remaining > 0)
      it->compute_remaining--;
    if (it->remaining > 0)
      it->remaining--;
  }

  for (int i = static_cast<int>(m_pending_completions.size()) - 1; i >= 0;
       --i) {
    if (m_pending_completions[i].remaining != 0 ||
        m_pending_completions[i].rf_traffic_remaining != 0)
      continue;
    wgmma_group_manager_t::wait_result_t completed =
        m_group_manager.complete_op(m_pending_completions[i].key.first,
                                    m_pending_completions[i].key.second,
                                    m_pending_completions[i].op_uid);
    result.released_warps.insert(result.released_warps.end(),
                                 completed.released_warps.begin(),
                                 completed.released_warps.end());
    m_pending_completions.erase(m_pending_completions.begin() + i);
  }

  m_barriers->release_wgmma_warps(result.released_warps);
}

void wgmma_unit_t::impl_t::cleanup_cta(unsigned cta_id) {
  for (std::vector<pending_completion_t>::iterator it =
           m_pending_completions.begin();
       it != m_pending_completions.end();) {
    if (it->key.first == cta_id) {
      it = m_pending_completions.erase(it);
    } else {
      ++it;
    }
  }

  m_group_manager.cleanup_cta(cta_id);
}

wgmma_unit_t::wgmma_unit_t(barrier_set_t *barriers,
                           const shader_core_config *config)
    : m_impl(new impl_t(barriers, config)) {}

wgmma_unit_t::~wgmma_unit_t() = default;

bool wgmma_unit_t::issue_chain_ready(const warp_inst_t *inst,
                                     unsigned long long cycle) const {
  return m_impl->issue_chain_ready(inst, cycle);
}

void wgmma_unit_t::record_issue_chain(const warp_inst_t *inst,
                                      unsigned long long cycle) {
  m_impl->record_issue_chain(inst, cycle);
}

void wgmma_unit_t::add_op(unsigned cta_id, unsigned warpgroup_id,
                          unsigned op_uid, unsigned compute_latency,
                          unsigned completion_tail_latency,
                          unsigned long long rf_traffic_tokens) {
  m_impl->add_op(cta_id, warpgroup_id, op_uid, compute_latency,
                 completion_tail_latency, rf_traffic_tokens);
}

unsigned long long wgmma_unit_t::drain_rf_traffic(unsigned long long bytes) {
  return m_impl->drain_rf_traffic(bytes);
}

unsigned long long wgmma_unit_t::rf_traffic_backlog() const {
  return m_impl->rf_traffic_backlog();
}

void wgmma_unit_t::commit_group(unsigned cta_id, unsigned warpgroup_id) {
  m_impl->commit_group(cta_id, warpgroup_id);
}

void wgmma_unit_t::wait_group(unsigned cta_id, unsigned warpgroup_id,
                              unsigned max_pending_groups,
                              const unsigned *warp_ids, unsigned count) {
  m_impl->wait_group(cta_id, warpgroup_id, max_pending_groups, warp_ids, count);
}

void wgmma_unit_t::cycle() { m_impl->cycle(); }

void wgmma_unit_t::cleanup_cta(unsigned cta_id) { m_impl->cleanup_cta(cta_id); }

} // namespace flash_gpgpu_sim

void barrier_set_t::release_wgmma_warps(
    const std::vector<unsigned> &released_warps) {
  for (std::vector<unsigned>::const_iterator it = released_warps.begin();
       it != released_warps.end(); ++it) {
    unsigned warp_id = *it;
    clear_warp_waiting(warp_id, BARRIER_WAIT_WGMMA_GROUP, "WGMMA release");
  }
}

void barrier_set_t::set_wgmma_waiting_warps(const unsigned *warp_ids,
                                            unsigned count) {
  for (unsigned i = 0; i < count; ++i) {
    if (m_warp_at_barrier.test(warp_ids[i])) {
      printf("GPGPU-Sim ERROR: warp %u reached WGMMA wait while already "
             "waiting. warp_at_barrier=%s type=%d\n",
             warp_ids[i], m_warp_at_barrier.to_string().c_str(),
             (int)m_warp_barrier_type[warp_ids[i]]);
      dump();
      assert(false && "warp reached WGMMA wait while already waiting");
    }
    m_warp_at_barrier.set(warp_ids[i]);
    m_warp_barrier_type[warp_ids[i]] = BARRIER_WAIT_WGMMA_GROUP;
    m_warp_named_barrier_id[warp_ids[i]] = (unsigned)-1;
  }
}
