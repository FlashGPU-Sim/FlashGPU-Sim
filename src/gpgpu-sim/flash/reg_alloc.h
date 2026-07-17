#ifndef FLASH_GPGPU_SIM_REG_ALLOC_H
#define FLASH_GPGPU_SIM_REG_ALLOC_H

class function_info;

namespace flash_gpgpu_sim {

void run_ptx_register_allocation(function_info *func);

} // namespace flash_gpgpu_sim

#endif
