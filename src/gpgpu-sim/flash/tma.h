#ifndef FLASH_GPGPU_SIM_TMA_H
#define FLASH_GPGPU_SIM_TMA_H

#include "mbarrier.h"

namespace flash_gpgpu_sim {

class tensor_memory_accelerator_t {
public:
  tensor_memory_accelerator_t(mbarrier_manager_t *mbarrier_manager)
      : m_mbarrier_manager(mbarrier_manager) {}

private:
  mbarrier_manager_t *m_mbarrier_manager;
};

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);

#endif