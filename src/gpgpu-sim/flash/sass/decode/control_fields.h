#ifndef FLASH_GPGPU_SIM_SASS_CONTROL_FIELDS_H_
#define FLASH_GPGPU_SIM_SASS_CONTROL_FIELDS_H_

#include "../ir.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace decode_detail {

// Shared scheduling fields at instruction bits [121:105]. The architecture
// entry points retain ownership of the distinct register-reuse layouts.
inline control_code decode_scheduling_fields(uint64_t raw_hi) {
  constexpr unsigned kControlShift = 105 - 64;
  constexpr uint64_t kControlMask = (uint64_t{1} << 17) - 1;
  const uint32_t packed =
      static_cast<uint32_t>((raw_hi >> kControlShift) & kControlMask);
  control_code result;
  result.stall = packed & 0xf;
  // Zero requests yield; the hardware field has inverted polarity.
  result.yield_flag = ((packed >> 4) & 1) == 0;
  result.write_barrier = (packed >> 5) & 0x7;
  result.read_barrier = (packed >> 8) & 0x7;
  result.wait_mask = (packed >> 11) & 0x3f;
  return result;
}

} // namespace decode_detail
} // namespace sass
} // namespace flash_gpgpu_sim

#endif
