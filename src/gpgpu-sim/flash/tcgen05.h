#ifndef FLASH_GPGPU_SIM_TCGEN05_H
#define FLASH_GPGPU_SIM_TCGEN05_H

#include "tcgen05/descriptor.h"
#include "tcgen05/mma.h"
#include "tcgen05/timing.h"
#include "tcgen05/tmem.h"

class ptx_instruction;
class ptx_thread_info;
class shader_core_config;
struct warp_inst_t;

namespace flash_gpgpu_sim {

// Adapt PTX instructions and shader configuration to the timing model.
int tcgen05_opcode(const warp_inst_t *inst);
bool tcgen05_has_option(const ptx_instruction *inst, int option);
bool is_tcgen05_timing_data_op(int opcode);
tcgen05_op_kind_t tcgen05_timing_kind(int opcode);
tcgen05_timing_config_t tcgen05_timing_config(const shader_core_config *config);
tcgen05_op_t tcgen05_timing_op(const ptx_instruction *inst,
                               const ptx_instruction *dynamic_inst,
                               const shader_core_config *config);

// PTX functional execution handlers.

void handle_tcgen05_alloc_inst(const ptx_instruction *inst,
                               ptx_thread_info *thread);
void handle_tcgen05_dealloc_inst(const ptx_instruction *inst,
                                 ptx_thread_info *thread);
void handle_tcgen05_relinq_inst(const ptx_instruction *inst,
                                ptx_thread_info *thread);
void handle_tcgen05_mma_inst(const ptx_instruction *inst,
                             ptx_thread_info *thread);
void handle_tcgen05_commit_inst(const ptx_instruction *inst,
                                ptx_thread_info *thread);
void handle_tcgen05_ld_inst(const ptx_instruction *inst,
                            ptx_thread_info *thread);
void handle_tcgen05_st_inst(const ptx_instruction *inst,
                            ptx_thread_info *thread);
void handle_tcgen05_wait_inst(const ptx_instruction *inst,
                              ptx_thread_info *thread);
void handle_tcgen05_cp_inst(const ptx_instruction *inst,
                            ptx_thread_info *thread);
void handle_tcgen05_shift_inst(const ptx_instruction *inst,
                               ptx_thread_info *thread);
void handle_tcgen05_fence_inst(const ptx_instruction *inst,
                               ptx_thread_info *thread);

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_H
