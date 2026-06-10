// Copyright (c) 2026 University of British Columbia
// All rights reserved.

#ifndef FLASH_GPGPU_SIM_TENSOR_WGMMA_H
#define FLASH_GPGPU_SIM_TENSOR_WGMMA_H

#include <cstdint>
#include <memory>

class ptx_instruction;
class ptx_thread_info;
class core_t;
class barrier_set_t;
struct warp_inst_t;

namespace flash_gpgpu_sim {

// WGMMA functional entry and data-type dispatch.
void tensor_wgmma_impl(const ptx_instruction *pI, core_t *core,
                       warp_inst_t &inst);
void wgmma_m64n8k16_f16_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst);
void wgmma_m64n8k16_bf16_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst);
void wgmma_m64n8k8_tf32_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst);
void wgmma_m64n8k32_fp8_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst, bool a_is_e4m3, bool b_is_e4m3);
void wgmma_m64n8k32_int8_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst, bool a_is_signed,
                              bool b_is_signed);
void wgmma_m64n8k256_b1_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst);

// Shared WGMMA fragment helpers used by data-type-specific implementations.
unsigned wgmma_thread_count(core_t *core, const warp_inst_t &inst);
ptx_thread_info *wgmma_thread(core_t *core, const warp_inst_t &inst,
                              unsigned thread_idx);
unsigned wgmma_lane(const ptx_thread_info *thread);
uint64_t wgmma_gmma_desc_base(uint64_t desc);
uint64_t wgmma_gmma_desc_leading_byte_offset(uint64_t desc);
uint64_t wgmma_gmma_desc_stride_byte_offset(uint64_t desc);
unsigned wgmma_gmma_k_major_smem_addr(uint64_t desc, int col, int k,
                                      unsigned element_size,
                                      unsigned default_contiguous_k);
void wgmma_m64n8_accumulator_coord(unsigned lane, int reg, int &row, int &col);

// WGMMA opcode hooks. These are wired through OP_W_DEF because WGMMA is a
// warpgroup instruction and must receive the active warp instance.
void wgmma_mma_async_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst);
void wgmma_mma_async_sp_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst);
void wgmma_fence_impl(const ptx_instruction *pI, core_t *core,
                      warp_inst_t &inst);
void wgmma_commit_group_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst);
void wgmma_wait_group_impl(const ptx_instruction *pI, core_t *core,
                           warp_inst_t &inst);
void setmaxnreg_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t &inst);

class wgmma_unit_t {
public:
  explicit wgmma_unit_t(barrier_set_t *barriers);
  ~wgmma_unit_t();

  void add_op(unsigned cta_id, unsigned warpgroup_id, unsigned op_uid,
              unsigned completion_latency);
  void commit_group(unsigned cta_id, unsigned warpgroup_id);
  void wait_group(unsigned cta_id, unsigned warpgroup_id,
                  unsigned max_pending_groups, const unsigned *warp_ids,
                  unsigned count);
  void cycle();
  void cleanup_cta(unsigned cta_id);

private:
  class impl_t;
  std::unique_ptr<impl_t> m_impl;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TENSOR_WGMMA_H
