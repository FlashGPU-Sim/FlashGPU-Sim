#ifndef FLASH_GPGPU_SIM_EXECUTION_FRONTEND_H_
#define FLASH_GPGPU_SIM_EXECUTION_FRONTEND_H_

#include "../../../abstract_hardware_model.h"
#include "launch_types.h"
#include <memory>
#include <string>

class exec_shader_core_ctx;

namespace flash_gpgpu_sim {

// Architectural state for one execution-driven shader. Timing issue, resource
// arbitration and completion scheduling remain in shader_core_ctx. Adapters
// may apply architectural effects here but must not issue pipeline operations.
// This is a SIMT backend contract, not a claim that every NPU uses SIMT.
class execution_frontend {
public:
  virtual ~execution_frontend() = default;
  frontend_kind kind() const { return kind_; }

  virtual unsigned initialize_thread(kernel_info_t &kernel, unsigned tid,
                                     unsigned threads_left,
                                     unsigned hardware_cta) = 0;
  virtual void start_cta(unsigned cta_id, unsigned start_thread,
                         unsigned end_thread, unsigned linear_cta_id,
                         int cta_size, kernel_info_t &kernel) = 0;
  virtual void release_cta(unsigned cta_id) = 0;
  virtual void reset() = 0;

  virtual const warp_inst_t *fetch(unsigned warp, address_type pc) = 0;
  virtual void control_flow_top(unsigned warp, const warp_inst_t *inst,
                                unsigned *pc, unsigned *reconvergence_pc) = 0;
  virtual const active_mask_t &active_mask(unsigned warp,
                                           const warp_inst_t *inst) = 0;
  virtual bool can_issue(unsigned warp) = 0;
  virtual void execute(warp_inst_t &inst) = 0;
  // Apply an already-issued collective to another architectural participant;
  // this may execute semantics or only advance PC, depending on the frontend.
  virtual void execute_collective_participant(warp_inst_t &inst) = 0;
  virtual void update_control_flow(unsigned warp, warp_inst_t *inst) = 0;
  virtual void complete_lane(warp_inst_t &inst, unsigned lane,
                             unsigned tid) = 0;
  virtual void complete_barrier_wait(unsigned warp) = 0;
  virtual int logical_cta_id(unsigned warp) const = 0;
  virtual int cta_warp_id(unsigned warp) const = 0;
  // Optional diagnostic; false requests the caller's existing fallback.
  virtual bool instruction_text(unsigned warp, address_type pc,
                                std::string &text) const {
    return false;
  }

protected:
  execution_frontend(exec_shader_core_ctx &core, frontend_kind kind)
      : core_(core), kind_(kind) {}
  exec_shader_core_ctx &core_;

private:
  frontend_kind kind_;
};

std::unique_ptr<execution_frontend>
make_execution_frontend(exec_shader_core_ctx &core, frontend_kind kind);
std::unique_ptr<execution_frontend>
make_ptx_frontend(exec_shader_core_ctx &core);
std::unique_ptr<execution_frontend>
make_sass_frontend(exec_shader_core_ctx &core);

} // namespace flash_gpgpu_sim
#endif
