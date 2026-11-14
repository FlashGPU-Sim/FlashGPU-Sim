#include "execution_frontend.h"
#include "../../shader.h"
#include "../panic.h"

namespace flash_gpgpu_sim {
std::unique_ptr<execution_frontend>
make_execution_frontend(exec_shader_core_ctx &core, frontend_kind kind) {
  switch (kind) {
  case frontend_kind::ptx:
    return make_ptx_frontend(core);
  case frontend_kind::sass:
    return make_sass_frontend(core);
  }
  panic("unsupported execution frontend");
}
} // namespace flash_gpgpu_sim

exec_shader_core_ctx::~exec_shader_core_ctx() = default;

void exec_shader_core_ctx::select_frontend(const kernel_info_t &kernel) {
  if (m_frontend->kind() == kernel.frontend())
    return;
  if (get_n_active_cta() != 0)
    flash_gpgpu_sim::panic("cannot change execution frontend with live CTAs");
  m_frontend =
      flash_gpgpu_sim::make_execution_frontend(*this, kernel.frontend());
}

int exec_shader_core_ctx::get_logical_cta_id(unsigned warp_id) const {
  return m_kernel ? m_frontend->logical_cta_id(warp_id)
                  : shader_core_ctx::get_logical_cta_id(warp_id);
}

int exec_shader_core_ctx::get_cta_warp_id(unsigned warp_id) const {
  return m_kernel ? m_frontend->cta_warp_id(warp_id)
                  : shader_core_ctx::get_cta_warp_id(warp_id);
}

void exec_shader_core_ctx::complete_mbarrier_try_wait(unsigned warp_id) {
  m_frontend->complete_barrier_wait(warp_id);
}

void exec_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                      unsigned end_thread, unsigned ctaid,
                                      int cta_size, kernel_info_t &kernel) {
  select_frontend(kernel);
  m_frontend->start_cta(cta_id, start_thread, end_thread, ctaid, cta_size,
                        kernel);
}

void exec_shader_core_ctx::release_frontend_cta(unsigned cta_id) {
  m_frontend->release_cta(cta_id);
}

void exec_shader_core_ctx::reset_frontend() { m_frontend->reset(); }

const warp_inst_t *exec_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                       address_type pc) {
  return m_frontend->fetch(warp_id, pc);
}

bool exec_shader_core_ctx::get_frontend_instruction_text(
    unsigned warp_id, address_type pc, std::string &text) const {
  return m_kernel && m_frontend->instruction_text(warp_id, pc, text);
}

void exec_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                   const warp_inst_t *pI,
                                                   unsigned *pc,
                                                   unsigned *rpc) {
  m_frontend->control_flow_top(warp_id, pI, pc, rpc);
}

const active_mask_t &
exec_shader_core_ctx::get_active_mask(unsigned warp_id, const warp_inst_t *pI) {
  return m_frontend->active_mask(warp_id, pI);
}

bool exec_shader_core_ctx::frontend_instruction_can_issue(unsigned warp_id) {
  return m_frontend->can_issue(warp_id);
}

void exec_shader_core_ctx::updateSIMTStack(unsigned warp_id,
                                           warp_inst_t *inst) {
  m_frontend->update_control_flow(warp_id, inst);
}

void exec_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  m_frontend->execute(inst);
}

void exec_shader_core_ctx::execute_collective_participant(warp_inst_t &inst) {
  m_frontend->execute_collective_participant(inst);
}

void exec_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                         unsigned t,
                                                         unsigned tid) {
  m_frontend->complete_lane(inst, t, tid);
}

unsigned exec_shader_core_ctx::initialize_frontend_thread(
    kernel_info_t &kernel, unsigned tid, unsigned threads_left,
    unsigned hardware_cta) {
  select_frontend(kernel);
  return m_frontend->initialize_thread(kernel, tid, threads_left, hardware_cta);
}
