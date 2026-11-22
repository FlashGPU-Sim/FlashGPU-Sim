#ifndef FLASH_GPGPU_SIM_SASS_SM90_CONTROL_H_
#define FLASH_GPGPU_SIM_SASS_SM90_CONTROL_H_

#include <cstdint>

#include "../ir.h"

namespace flash_gpgpu_sim {
namespace sass {

// Decode Hopper's standard scheduling layout from bits [121:105]. Hopper
// retains four register-reuse flags in bits [125:122].
control_code decode_sm90_control(uint64_t raw_hi);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_SM90_CONTROL_H_
