#ifndef FLASH_GPGPU_SIM_TMA_H
#define FLASH_GPGPU_SIM_TMA_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>

#include "tensormap.h"

class shader_core_ctx;
class warp_inst_t;
class barrier_set_t;
class ptx_instruction;
class ptx_thread_info;
class mem_fetch_interface;
class shader_core_mem_fetch_allocator;
class mem_fetch;
class memory_space;

namespace flash_gpgpu_sim {

struct tma_progress_counters_t {
  unsigned long long tx_started = 0;
  unsigned long long read_tx_started = 0;
  unsigned long long write_tx_started = 0;
  unsigned long long tx_completed = 0;
  unsigned long long read_tx_completed = 0;
  unsigned long long write_tx_completed = 0;
  unsigned long long mf_issued = 0;
  unsigned long long read_mf_issued = 0;
  unsigned long long write_mf_issued = 0;
  unsigned long long mf_responses = 0;
  unsigned long long read_mf_responses = 0;
  unsigned long long write_mf_responses = 0;
  unsigned long long bytes_issued = 0;
  unsigned long long bytes_completed = 0;
};

tma_progress_counters_t get_global_tma_progress_counters();

class tma_unit_impl_t;
class tma_unit_t {
public:
  tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
             mem_fetch_interface *icnt,
             shader_core_mem_fetch_allocator *mf_allocator);
  ~tma_unit_t();

  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
  void cycle();

  void fill(mem_fetch *mf);
  bool response_buffer_full() const;
  bool has_pending_for_cta(unsigned cta_id) const;

private:
  std::unique_ptr<tma_unit_impl_t> m_impl;
};

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);

#endif
