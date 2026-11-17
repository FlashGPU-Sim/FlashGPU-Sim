#ifndef FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H
#define FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H

#include <cstdint>

namespace flash_gpgpu_sim {

enum tcgen05_mma_type_field_t {
  TCGEN05_MMA_TYPE_FIELD_F16 = 0,
  TCGEN05_MMA_TYPE_FIELD_ONE = 1,
};

struct tcgen05_shared_descriptor_t {
  uint32_t start_address;
  uint32_t leading_dimension_byte_offset;
  uint32_t stride_dimension_byte_offset;
  uint8_t fixed_constant;
  uint8_t base_offset;
  bool leading_dimension_absolute;
  uint8_t swizzle_mode;
};

struct tcgen05_mma_descriptor_t {
  bool sparse;
  uint8_t sparsity_selector;
  uint8_t d_type;
  uint8_t a_type;
  uint8_t b_type;
  bool negate_a;
  bool negate_b;
  bool transpose_a;
  bool transpose_b;
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

tcgen05_shared_descriptor_t tcgen05_decode_shared_descriptor(uint64_t desc);
tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group);

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H
