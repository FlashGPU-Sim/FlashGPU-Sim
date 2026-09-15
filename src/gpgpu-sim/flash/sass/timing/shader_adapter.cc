// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh, Vijay Kandiah, Nikos
// Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The University of
// British Columbia, Northwestern University, Purdue University All rights
// reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "../../../gpu-sim.h"
#include "../../../shader.h"
#include "../../frontend/execution_frontend.h"
#include "../../panic.h"
#include "../runtime/runtime_adapter.h"
#include "timing_runtime.h"
#include <climits>

namespace flash_gpgpu_sim {
namespace sass {
class shader_frontend final : public execution_frontend {
public:
  explicit shader_frontend(exec_shader_core_ctx &core)
      : execution_frontend(core, frontend_kind::sass),
        runtime_(core_.m_config, core_.m_sid),
        active_masks_(core_.m_config->max_warps_per_shader) {}
  void control_flow_top(unsigned warp_id, const warp_inst_t *pI, unsigned *pc,
                        unsigned *rpc) override {

    *pc = static_cast<unsigned>(runtime_.pc(warp_id));
    *rpc = static_cast<unsigned>(-1);
    return;
  }

  const active_mask_t &active_mask(unsigned warp_id,
                                   const warp_inst_t *pI) override {

    active_mask_t &mask = active_masks_.at(warp_id);
    mask.reset();
    const uint32_t sass_mask = runtime_.active_mask(warp_id);
    for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane)
      mask.set(lane, (sass_mask & (uint32_t{1} << lane)) != 0);
    return mask;
  }

  void execute(warp_inst_t &inst) override {

    const flash_gpgpu_sim::sass::timing_step_result result = runtime_.step(
        inst.warp_id(), inst,
        core_.m_gpu->gpu_tot_sim_cycle + core_.m_gpu->gpu_sim_cycle,
        core_.m_gpu->get_config().get_core_freq());
    if (result.instruction_refill_delay != 0) {
      // A target can be fetched two cycles before issue. Start the refill
      // after the native control delay so the configured value remains an
      // additional backedge bubble rather than overlapping that delay.
      constexpr unsigned kFetchDecodeLeadCycles = 2;
      const unsigned control_delay = inst.get_dependency_control().stall_cycles;
      const unsigned fetch_delay =
          control_delay + result.instruction_refill_delay >
                  kFetchDecodeLeadCycles
              ? control_delay + result.instruction_refill_delay -
                    kFetchDecodeLeadCycles
              : 0;
      core_.m_warp[inst.warp_id()]->delay_instruction_fetch(
          core_.m_gpu->gpu_tot_sim_cycle + core_.m_gpu->gpu_sim_cycle,
          fetch_delay);
    }
    for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane) {
      if ((result.exited_mask & (uint32_t{1} << lane)) != 0)
        core_.m_warp[inst.warp_id()]->set_completed(lane);
    }
    if (result.active_after == 0) {
      // SASS cubins may contain padding instructions after a terminal EXIT.
      // Decode can prefetch one of them before EXIT executes, so retire that
      // buffered instruction just as the PTX thread-completion path does.
      core_.m_warp[inst.warp_id()]->ibuffer_flush();
      core_.m_barriers.warp_exit(inst.warp_id());
    }
    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      const unsigned num_shader = core_.m_config->n_simt_clusters *
                                  core_.m_config->n_simt_cores_per_cluster;
      for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane) {
        if (!inst.active(lane))
          continue;
        const unsigned tid = inst.warp_id() * core_.m_config->warp_size + lane;
        new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
        const unsigned num_addrs = core_.translate_local_memaddr(
            inst.get_addr(lane), tid, num_shader, inst.data_size, localaddrs);
        inst.set_addr(lane, localaddrs, num_addrs);
      }
    }
    if (inst.is_load() || inst.is_store())
      inst.generate_mem_accesses();
    return;
  }

  void execute_collective_participant(warp_inst_t &inst) override {

    // A native GMMA is issued once for the four-warp group, but each SASS warp
    // owns a distinct architectural register state. Execute the already-issued
    // instruction for every non-representative warp so its PC and functional
    // outputs advance without creating another timing operation.
    execute(inst);
    return;
  }

  const warp_inst_t *fetch(unsigned warp, address_type pc) override {
    return runtime_.fetch(warp, pc);
  }

  void start_cta(unsigned cta_id, unsigned start_thread, unsigned end_thread,
                 unsigned ctaid, int cta_size, kernel_info_t &kernel) override {

    if (!flash_gpgpu_sim::sass::runtime_timing_requested())
      flash_gpgpu_sim::panic(
          "SASS kernel reached a timing shader without timing mode");

    const unsigned start_warp = start_thread / core_.m_config->warp_size;
    const unsigned end_warp =
        end_thread / core_.m_config->warp_size +
        ((end_thread % core_.m_config->warp_size) != 0 ? 1 : 0);
    const unsigned warp_count = end_warp - start_warp;
    runtime_.bind(kernel, core_.m_cluster->get_gpu()->get_global_memory());
    runtime_.start_cta(cta_id, start_warp, warp_count, ctaid);

    for (unsigned warp_id = start_warp; warp_id < end_warp; ++warp_id) {
      core_.reset_shared_proxy_warp(warp_id);
      active_mask_t active_threads;
      unsigned n_active = 0;
      for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane) {
        const unsigned hwtid = warp_id * core_.m_config->warp_size + lane;
        if (hwtid >= end_thread)
          continue;
        active_threads.set(lane);
        core_.m_active_threads.set(hwtid);
        ++n_active;
      }
      core_.m_warp[warp_id]->init(runtime_.pc(warp_id), cta_id, warp_id,
                                  active_threads, core_.m_dynamic_warp_id,
                                  kernel.get_streamID());
      ++core_.m_dynamic_warp_id;
      core_.m_not_completed += n_active;
      ++core_.m_active_warps;
    }
  }

  void complete_lane(warp_inst_t &, unsigned, unsigned) override {}

  int logical_cta_id(unsigned warp_id) const override {

    const uint64_t cta_id = runtime_.linear_cta_id(warp_id);
    if (cta_id > INT_MAX)
      flash_gpgpu_sim::panic("SASS logical CTA id exceeds timing backend ABI");
    return static_cast<int>(cta_id);
  }

  int cta_warp_id(unsigned warp_id) const override {

    const unsigned local_warp = runtime_.local_warp_id(warp_id);
    if (local_warp > INT_MAX)
      flash_gpgpu_sim::panic("SASS CTA warp id exceeds timing backend ABI");
    return static_cast<int>(local_warp);
  }

  bool instruction_text(unsigned warp_id, address_type pc,
                        std::string &text) const override {

    const flash_gpgpu_sim::sass::timing_instruction *inst =
        runtime_.fetch(warp_id, pc);
    if (inst == nullptr)
      return false;
    text = inst->native_text();
    return true;
  }

  void complete_barrier_wait(unsigned warp_id) override {

    if (runtime_.complete_mbarrier_try_wait(warp_id) ==
        flash_gpgpu_sim::sass::mbarrier_try_wait_state::kNone)
      flash_gpgpu_sim::panic(
          "SASS mbarrier released a warp without a pending try-wait");
  }

  bool can_issue(unsigned warp_id) override {
    return runtime_.deferred_barrier_can_issue(
        warp_id, core_.m_gpu->gpu_tot_sim_cycle + core_.m_gpu->gpu_sim_cycle);
  }
  void release_cta(unsigned cta_id) override { runtime_.release_cta(cta_id); }
  void reset() override { runtime_.reset(); }
  void update_control_flow(unsigned, warp_inst_t *) override {}
  unsigned initialize_thread(kernel_info_t &kernel, unsigned,
                             unsigned threads_left, unsigned) override {
    if (!runtime_timing_requested())
      panic("SASS kernel reached timing thread initialization without timing "
            "mode");
    if (threads_left == 1)
      kernel.increment_cta_id();
    return 1;
  }

private:
  timing_runtime runtime_;
  std::vector<active_mask_t> active_masks_;
};
} // namespace sass
std::unique_ptr<execution_frontend>
make_sass_frontend(exec_shader_core_ctx &core) {
  return std::make_unique<sass::shader_frontend>(core);
}
} // namespace flash_gpgpu_sim
