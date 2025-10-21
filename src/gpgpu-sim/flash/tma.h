#ifndef FLASH_GPGPU_SIM_TMA_H
#define FLASH_GPGPU_SIM_TMA_H

#include <memory>

class shader_core_ctx;
class warp_inst_t;
class barrier_set_t;
class ptx_instruction;
class ptx_thread_info;

namespace flash_gpgpu_sim {

class tma_unit_impl_t;
class tma_unit_t {
public:
  tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers);
  ~tma_unit_t();

  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);

private:
  std::unique_ptr<tma_unit_impl_t> m_impl;
};

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);

#endif