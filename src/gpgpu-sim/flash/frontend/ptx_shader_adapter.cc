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

#include "../../../../libcuda/gpgpu_context.h"
#include "../../../cuda-sim/cuda-sim.h"
#include "../../gpu-sim.h"
#include "../../shader.h"
#include "../../stat-tool.h"
#include "execution_frontend.h"

namespace flash_gpgpu_sim {
class ptx_shader_frontend final : public execution_frontend {
public:
  explicit ptx_shader_frontend(exec_shader_core_ctx &core)
      : execution_frontend(core, frontend_kind::ptx) {}
  void control_flow_top(unsigned warp_id, const warp_inst_t *pI, unsigned *pc,
                        unsigned *rpc) override {

    core_.m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
  }

  const active_mask_t &active_mask(unsigned warp_id,
                                   const warp_inst_t *pI) override {

    return core_.m_simt_stack[warp_id]->get_active_mask();
  }

  void execute(warp_inst_t &inst) override {

    // PTX functional handlers record a few per-lane results on their dynamic
    // instruction object.  Copy those results across the frontend boundary here
    // so the timing pipeline only ever sees warp_inst_t.
    const bool needs_named_barrier_info = inst.op == BARRIER_OP;
    const bool needs_mbarrier_info =
        inst.op == MBARRIER_OP || inst.m_is_cp_async_mbarrier_arrive;
    const bool needs_tma_info = inst.op == TENSOR_MEMORY_ACCELERATOR_OP;
    ptx_instruction *frontend_dynamic_inst = nullptr;
    if (needs_named_barrier_info || needs_mbarrier_info || needs_tma_info) {
      frontend_dynamic_inst =
          const_cast<ptx_instruction *>(dynamic_cast<const ptx_instruction *>(
              fetch(inst.warp_id(), inst.pc)));
      assert(frontend_dynamic_inst != nullptr);
      if (needs_named_barrier_info) {
        frontend_dynamic_inst->set_bar_id((unsigned)-1);
        frontend_dynamic_inst->set_bar_count((unsigned)-1);
      }
      if (needs_mbarrier_info)
        frontend_dynamic_inst->reset_mbarrier_info();
      if (needs_tma_info)
        frontend_dynamic_inst->reset_tma_dyn_info();
    }

    core_.execute_warp_inst_t(inst);

    if (frontend_dynamic_inst != nullptr) {
      if (needs_named_barrier_info) {
        inst.set_bar_id(frontend_dynamic_inst->bar_id);
        inst.set_bar_count(frontend_dynamic_inst->bar_count);
      }
      if (needs_tma_info)
        inst.set_tma_static_info(frontend_dynamic_inst->get_tma_static_info());
      for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane) {
        if (needs_mbarrier_info) {
          inst.set_mbarrier_info(
              lane, frontend_dynamic_inst->get_mbarrier_info(lane));
        }
        if (needs_tma_info) {
          inst.set_tma_dyn_info(lane,
                                frontend_dynamic_inst->get_tma_dyn_info(lane));
        }
      }
    }

    if (inst.is_load() || inst.is_store()) {
      inst.generate_mem_accesses();
      // inst.print_m_accessq();
    }
  }

  void execute_collective_participant(warp_inst_t &inst) override {

    // PTX implementation of the generic frontend-state hook. WGMMA functional
    // semantics are evaluated once for the representative warpgroup
    // instruction; participating warps still need their architectural PCs and
    // execution status advanced.
    for (unsigned lane = 0; lane < core_.m_config->warp_size; ++lane) {
      if (!inst.active(lane))
        continue;
      const unsigned tid = inst.warp_id() * core_.m_config->warp_size + lane;
      ptx_thread_info *thread = core_.m_thread[tid];
      assert(thread != NULL);
      const addr_t pc = thread->next_instr();
      assert(pc == inst.pc);
      thread->set_npc(inst.pc + inst.isize);
      thread->update_pc();
      complete_lane(inst, lane, tid);
    }
  }

  const warp_inst_t *fetch(unsigned, address_type pc) override {
    return core_.m_gpu->gpgpu_ctx->ptx_fetch_inst(pc);
  }

  void start_cta(unsigned cta_id, unsigned start_thread, unsigned end_thread,
                 unsigned ctaid, int cta_size, kernel_info_t &kernel) override {
    core_.shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid,
                                      cta_size, kernel);
  }

  void complete_lane(warp_inst_t &inst, unsigned t, unsigned tid) override {

    if (inst.isatomic())
      core_.m_warp[inst.warp_id()]->inc_n_atomic();
    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = core_.translate_local_memaddr(
          inst.get_addr(t), tid,
          core_.m_config->n_simt_clusters *
              core_.m_config->n_simt_cores_per_cluster,
          inst.data_size, (new_addr_type *)localaddrs);
      inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    }
    if (core_.ptx_thread_done(tid)) {
      core_.m_warp[inst.warp_id()]->set_completed(t);
      core_.m_warp[inst.warp_id()]->ibuffer_flush();
    }

    // PC-Histogram Update
    unsigned warp_id = inst.warp_id();
    unsigned pc = inst.pc;
    for (unsigned t = 0; t < core_.m_config->warp_size; t++) {
      if (inst.active(t)) {
        int tid = warp_id * core_.m_config->warp_size + t;
        cflog_update_thread_pc(core_.m_sid, tid, pc);
      }
    }
  }

  int logical_cta_id(unsigned warp_id) const override {
    return core_.shader_core_ctx::get_logical_cta_id(warp_id);
  }

  int cta_warp_id(unsigned warp_id) const override {
    return core_.shader_core_ctx::get_cta_warp_id(warp_id);
  }

  bool can_issue(unsigned) override { return true; }
  void release_cta(unsigned) override {}
  void reset() override {}
  void complete_barrier_wait(unsigned) override {}
  void update_control_flow(unsigned warp, warp_inst_t *inst) override {
    core_.core_t::updateSIMTStack(warp, inst);
  }
  unsigned initialize_thread(kernel_info_t &kernel, unsigned tid,
                             unsigned threads_left,
                             unsigned hardware_cta) override {
    return ptx_sim_init_thread(
        kernel, &core_.m_thread[tid], core_.m_sid, tid, threads_left,
        core_.m_config->n_thread_per_shader, &core_, hardware_cta,
        tid / core_.m_config->warp_size, core_.m_cluster->get_gpu());
  }
};
std::unique_ptr<execution_frontend>
make_ptx_frontend(exec_shader_core_ctx &core) {
  return std::make_unique<ptx_shader_frontend>(core);
}
} // namespace flash_gpgpu_sim
