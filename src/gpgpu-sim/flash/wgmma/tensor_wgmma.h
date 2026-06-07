// Copyright (c) 2026 University of British Columbia
// All rights reserved.

#ifndef FLASH_GPGPU_SIM_TENSOR_WGMMA_H
#define FLASH_GPGPU_SIM_TENSOR_WGMMA_H

class ptx_instruction;
class core_t;
struct warp_inst_t;

namespace flash_gpgpu_sim {

// Warpgroup-level MMA placeholders. These are wired through OP_W_DEF because
// WGMMA is a warpgroup instruction and must receive the active warp instance.
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

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TENSOR_WGMMA_H
