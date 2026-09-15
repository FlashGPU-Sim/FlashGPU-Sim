#include "sm90_control.h"

#include "control_fields.h"

namespace flash_gpgpu_sim {
namespace sass {

control_code decode_sm90_control(uint64_t raw_hi) {
  control_code result = decode_detail::decode_scheduling_fields(raw_hi);
  result.reuse_mask = (raw_hi >> (122 - 64)) & 0xf;
  return result;
}

} // namespace sass
} // namespace flash_gpgpu_sim
