#ifndef FLASH_GPGPU_SIM_LDMATRIX_H
#define FLASH_GPGPU_SIM_LDMATRIX_H

// Forward declarations for mbarrier_impl
class ptx_instruction;
class ptx_thread_info;
class core_t;
class warp_inst_t;
void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst);
#endif