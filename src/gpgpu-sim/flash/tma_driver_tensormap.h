#ifndef FLASH_GPGPU_SIM_TMA_DRIVER_TENSORMAP_H
#define FLASH_GPGPU_SIM_TMA_DRIVER_TENSORMAP_H

#include "tensormap.h"

namespace flash_gpgpu_sim {
// Captured CUDA Driver SM100 descriptors, not a documented portable ABI.
// Only observed dense controls are accepted. TF32 conversion and packed
// sub-byte payloads require separate transfer semantics and remain rejected.
inline bool sm100_driver_tma_format(uint32_t control, unsigned rank,
                                    uint32_t &dtype, uint32_t &swizzle) {
  swizzle = (control >> 13) & 3;
  if (rank == 2) {
    switch (control) {
    case 0x60510:
    case 0x62510:
    case 0x264510:
    case 0x46510:
    case 0x246510:
    case 0x66510:
    case 0x266510:
      dtype = TMA_DTYPE_BF16;
      return true;
    case 0x266390:
    case 0x264390:
    case 0x64390:
    case 0x60390:
    case 0x260390:
      dtype = TMA_DTYPE_F32;
      return true;
    case 0x46010:
    case 0x246010:
    case 0x66010:
    case 0x266010:
    case 0x240010:
    case 0x264010:
      dtype = TMA_DTYPE_U8;
      return true;
    case 0x40290:
    case 0x240290:
      dtype = TMA_DTYPE_U64;
      return true;
    case 0x260110:
    case 0x260190:
    case 0x60190:
      dtype = TMA_DTYPE_U32;
      return true;
    default:
      return false;
    }
  }
  if (rank == 3) {
    switch (control) {
    case 0x40020:
    case 0x240020:
    case 0x66020:
    case 0x266020:
      dtype = TMA_DTYPE_U8;
      return true;
    case 0x66520:
    case 0x266520:
      dtype = TMA_DTYPE_BF16;
      return true;
    case 0x663a0:
    case 0x2663a0:
      dtype = TMA_DTYPE_F32;
      return true;
    default:
      return false;
    }
  }
  if (rank == 4 && control == 0x266030) {
    dtype = TMA_DTYPE_U8;
    return true;
  }
  if (rank == 5 && (control == 0x266540 || control == 0x2663c0)) {
    dtype = control == 0x266540 ? TMA_DTYPE_BF16 : TMA_DTYPE_F32;
    return true;
  }
  return false;
}

inline bool decode_sm100_driver_tensormap(const tensormap_descriptor_t &encoded,
                                          unsigned rank,
                                          tensormap_descriptor_t &decoded) {
  const auto &w = encoded.raw_u64;
  uint32_t dtype, swizzle;
  if (!w[0] || !sm100_driver_tma_format(uint32_t(w[1]), rank, dtype, swizzle))
    return false;
  // No interleave, element-stride modifiers or reserved higher-rank fields.
  const uint64_t dim_mask =
      rank == 5 ? 0x00ffffff00000000ull : 0x00ffffffffffffffull;
  if ((w[6] & dim_mask) || (w[7] >> (8 * (rank - 1))))
    return false;
  const uint32_t dims[5] = {uint32_t(w[4]), uint32_t(w[4] >> 32),
                            uint32_t(w[5]), uint32_t(w[5] >> 32),
                            uint32_t(w[6])};
  const uint32_t strides[5] = {uint32_t(w[1] >> 32), uint32_t(w[2]),
                               uint32_t(w[2] >> 32), uint32_t(w[3]),
                               uint32_t(w[3] >> 32)};
  for (unsigned i = rank; i < 5; ++i)
    if (dims[i])
      return false;
  for (unsigned i = rank - 1; i < 5; ++i)
    if (strides[i])
      return false;
  for (unsigned i = 10; i < 16; ++i)
    if (w[i])
      return false;
  tensormap_descriptor_t result{};
  result.fields.globalAddress = w[0];
  result.fields.tensorRank = rank - 1;
  result.fields.tensorDataType = dtype;
  result.fields.swizzle = swizzle;
  result.fields.interleave = TMA_INTERLEAVE_NONE;
  result.fields.oobFill = TMA_OOB_ZERO;
  uint64_t bytes = result.get_element_size();
  for (unsigned i = 0; i < rank; ++i) {
    if (dims[i] == UINT32_MAX || (i && !strides[i - 1]))
      return false;
    result.fields.globalDim[i] = dims[i] + 1;
    result.fields.boxDim[i] =
        1 + (i ? ((w[7] >> (8 * (i - 1))) & 255) : (w[6] >> 56));
    result.fields.elementStrides[i] = 1;
    if (i)
      result.fields.globalStrides[i - 1] = uint64_t(strides[i - 1]) * 16;
    bytes *= result.fields.boxDim[i];
  }
  const uint64_t swizzle_bytes = swizzle ? 16u << swizzle : 0;
  if (w[8] != bytes || w[9] != (swizzle ? swizzle_bytes * 8 : 0x10))
    return false;
  decoded = result;
  return true;
}
} // namespace flash_gpgpu_sim
#endif
