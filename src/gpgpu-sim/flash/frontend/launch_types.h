#ifndef FLASH_GPGPU_SIM_FRONTEND_LAUNCH_TYPES_H_
#define FLASH_GPGPU_SIM_FRONTEND_LAUNCH_TYPES_H_

namespace flash_gpgpu_sim {

// Explicit program identity; absence of a PTX object is not an ISA selector.
enum class frontend_kind { ptx, sass };

// Launch-time resource requirements consumed by occupancy and allocation.
// Counts are registers/thread and bytes (shared/CTA, local and stack/thread).
struct kernel_resource_usage {
  unsigned registers = 0;
  unsigned static_shared = 0;
  unsigned local_memory = 0;
  unsigned stack_size = 0;
};

} // namespace flash_gpgpu_sim
#endif
