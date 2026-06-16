#ifndef FLASH_GPGPU_SIM_PTX_SCHEDULER_H
#define FLASH_GPGPU_SIM_PTX_SCHEDULER_H

class function_info;

namespace flash_gpgpu_sim {

void run_ptx_reorder(function_info *func);

} // namespace flash_gpgpu_sim

#endif
