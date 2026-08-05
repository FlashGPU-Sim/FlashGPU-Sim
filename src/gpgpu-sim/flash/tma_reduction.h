#ifndef FLASH_GPGPU_SIM_TMA_REDUCTION_H
#define FLASH_GPGPU_SIM_TMA_REDUCTION_H

#include <cstddef>
#include <cstdint>

enum class tma_reduction_op_t {
  NONE,
  ADD,
  MIN,
  MAX,
  INC,
  DEC,
  BIT_AND,
  BIT_OR,
  BIT_XOR,
};

const char *tma_reduction_op_name(tma_reduction_op_t op);

bool tma_tensor_reduction_supported(tma_reduction_op_t op,
                                    uint32_t tensor_data_type);

// Applies the element-wise tensor reduction in place to dst. The caller must
// validate the operation/type pair and provide a whole number of elements.
void apply_tma_tensor_reduction(tma_reduction_op_t op,
                                uint32_t tensor_data_type, void *dst,
                                const void *src, size_t size_in_bytes);

#endif
