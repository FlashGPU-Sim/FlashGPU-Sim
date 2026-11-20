#ifndef FLASH_GPGPU_SIM_SASS_DECODER_H_
#define FLASH_GPGPU_SIM_SASS_DECODER_H_

#include <string>

#include "../ir.h"

namespace flash_gpgpu_sim {
namespace sass {

// Stable boundary between an evolving reverse-engineered decoder and the
// simulator.  A decoder runs once per static kernel; it is not an instruction
// trace source.
class decoder {
public:
  virtual ~decoder() = default;
  virtual kernel decode_kernel(const std::string &image_path,
                               const std::string &kernel_name) const = 0;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_DECODER_H_
