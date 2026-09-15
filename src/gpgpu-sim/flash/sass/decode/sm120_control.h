#ifndef FLASH_GPGPU_SIM_SASS_SM120_CONTROL_H_
#define FLASH_GPGPU_SIM_SASS_SM120_CONTROL_H_

#include <cstdint>

#include "../ir.h"

namespace flash_gpgpu_sim {
namespace sass {

// Decode the standard SM120 control layout from bits [121:105].  Register
// reuse flags occupy bits [124:122].  Some opcode classes overlay parts of
// these positions; the raw word is retained in instruction::raw so an opcode-
// aware decoder can refine those cases without losing information.
control_code decode_sm120_control(uint64_t raw_hi);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_SM120_CONTROL_H_
