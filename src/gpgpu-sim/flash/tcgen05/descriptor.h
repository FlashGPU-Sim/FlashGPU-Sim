#ifndef FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H
#define FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H

#include <cstdint>

namespace flash_gpgpu_sim {

enum tcgen05_mma_type_field_t {
  TCGEN05_MMA_TYPE_FIELD_F16 = 0,
  TCGEN05_MMA_TYPE_FIELD_ONE = 1,
};

enum tcgen05_mxf4_format_t {
  TCGEN05_MXF4_FORMAT_E2M1 = 1,
};

enum tcgen05_scale_format_t {
  TCGEN05_SCALE_FORMAT_UE4M3 = 0,
  TCGEN05_SCALE_FORMAT_UE8M0 = 1,
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
  uint8_t a_scale_factor_id;
  uint8_t b_scale_factor_id;
  uint8_t scale_format;
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

tcgen05_shared_descriptor_t tcgen05_decode_shared_descriptor(uint64_t desc);
uint32_t tcgen05_shared_k_major_packed_byte_address(
    const tcgen05_shared_descriptor_t &desc, uint32_t matrix_row,
    uint32_t packed_k, uint32_t default_packed_k_per_row);
bool tcgen05_mxf4_dense_shape_supported(uint32_t m, uint32_t n, uint32_t k,
                                        unsigned cta_group);
tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group);
tcgen05_mma_descriptor_t tcgen05_decode_mxf4_mma_descriptor(uint32_t idesc,
                                                            unsigned cta_group);

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_DESCRIPTOR_H
