#ifndef FLASH_GPGPU_SIM_SASS_TIMING_RUNTIME_H_
#define FLASH_GPGPU_SIM_SASS_TIMING_RUNTIME_H_

#include <cstdint>
#include <memory>

#include "../frontend.h"
#include "timing_projection.h"

class kernel_info_t;
class memory_space;

namespace flash_gpgpu_sim {
namespace sass {

struct timing_step_result {
  step_result frontend_result;
  uint32_t active_before = 0;
  uint32_t executed_mask = 0;
  uint32_t active_after = 0;
  uint32_t exited_mask = 0;
  unsigned instruction_refill_delay = 0;
};

// Per-shader adapter between execution-driven architectural SASS state and
// the common timing instruction contract. Static images are shared across
// shader cores; CTA and warp state is private to this object.
class timing_runtime {
public:
  timing_runtime(const core_config *config, unsigned shader_id);
  ~timing_runtime();

  timing_runtime(const timing_runtime &) = delete;
  timing_runtime &operator=(const timing_runtime &) = delete;

  void bind(kernel_info_t &launch, ::memory_space *global_memory);
  void start_cta(unsigned hardware_cta_id, unsigned first_hardware_warp,
                 unsigned warp_count, uint64_t linear_cta_id);
  void release_cta(unsigned hardware_cta_id);
  void reset();

  const timing_instruction *fetch(unsigned hardware_warp_id, uint64_t pc) const;
  uint64_t pc(unsigned hardware_warp_id) const;
  uint32_t active_mask(unsigned hardware_warp_id) const;
  uint64_t linear_cta_id(unsigned hardware_warp_id) const;
  unsigned local_warp_id(unsigned hardware_warp_id) const;
  bool deferred_barrier_can_issue(unsigned hardware_warp_id,
                                  uint64_t architectural_cycle);
  mbarrier_try_wait_state complete_mbarrier_try_wait(unsigned hardware_warp_id);
  timing_step_result step(unsigned hardware_warp_id,
                          warp_inst_t &dynamic_instruction,
                          uint64_t architectural_cycle,
                          uint64_t architectural_core_frequency_hz);

private:
  class impl;
  std::unique_ptr<impl> impl_;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_TIMING_RUNTIME_H_
