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
  unsigned long long max_active_transactions = 0;
  // Maximum child requests awaiting responses on any one SM.
  unsigned long long max_mf_inflight = 0;
  // Sum of SM-cycles in which the cap prevented full TMA issue progress.
  unsigned long long issue_blocked_inflight_cycles = 0;
};

tma_progress_counters_t get_global_tma_progress_counters();

struct cp_async_debug_counters_t {
  unsigned long long tx_started = 0;
  unsigned long long tx_completed = 0;
  unsigned long long mf_issued = 0;
  unsigned long long mf_responses = 0;
  unsigned long long bytes_issued = 0;
  unsigned long long bytes_completed = 0;
  unsigned long long issue_queue_cycles = 0;
  unsigned long long issue_active_cycles = 0;
  unsigned long long issue_width_limited_cycles = 0;
  unsigned long long issue_blocked_inflight_cycles = 0;
  unsigned long long issue_blocked_icnt_cycles = 0;
  unsigned long long max_issue_queue = 0;
  unsigned long long max_inflight = 0;
  unsigned long long wait_calls = 0;
  unsigned long long wait_immediate = 0;
  unsigned long long wait_blocked = 0;
  unsigned long long wait_releases = 0;
  unsigned long long waiting_warp_cycles = 0;
  unsigned long long response_fifo_nonempty_cycles = 0;
  unsigned long long response_width_limited_cycles = 0;
  unsigned long long max_response_fifo = 0;
};

cp_async_debug_counters_t get_global_cp_async_debug_counters();

class tma_unit_impl_t;
class tma_unit_t {
public:
  tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
             mem_fetch_interface *icnt,
             shader_core_mem_fetch_allocator *mf_allocator);
  ~tma_unit_t();

  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
  void warp_reaches_cp_async(unsigned cta_id, unsigned warp_id,
                             const warp_inst_t &inst,
                             const ptx_instruction *static_inst);
  void
  warp_reaches_cp_async_mbarrier_arrive(unsigned cta_id, unsigned warp_id,
                                        const warp_inst_t &inst,
                                        const ptx_instruction &dynamic_inst);
  void commit_cp_async_group(unsigned cta_id, unsigned warp_id);
  void wait_cp_async_group(unsigned cta_id, unsigned warp_id,
                           unsigned max_pending_groups);
  void cleanup_cta(unsigned cta_id);
  void cycle();

  void fill(mem_fetch *mf);
  bool can_accept_transaction() const;
  bool response_buffer_full() const;
  bool has_pending_for_cta(unsigned cta_id) const;

private:
  std::unique_ptr<tma_unit_impl_t> m_impl;
};

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);

#endif
