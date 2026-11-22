#ifndef FLASH_GPGPU_SIM_SASSIR_DECODER_H_
#define FLASH_GPGPU_SIM_SASSIR_DECODER_H_

#include "decoder.h"

namespace flash_gpgpu_sim {
namespace sass {

// Reads the versioned, line-oriented interchange format. Current sassirs are
// produced from nvdisasm JSON plus raw cubin code/static-constant bytes; v1
// remains readable for historical CuBit sassirs. The simulator itself has no
// JSON, nvdisasm, or CuBit runtime dependency.
class sassir_decoder : public decoder {
public:
  kernel decode_kernel(const std::string &sassir_path,
                       const std::string &kernel_name) const override;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASSIR_DECODER_H_
