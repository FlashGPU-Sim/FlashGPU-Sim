#pragma once
#include "tensormap.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace flash_gpgpu_sim {
inline bool tma_float_conversion(unsigned type) {
  return type == TMA_DTYPE_F32_FTZ || type == TMA_DTYPE_TF32 ||
         type == TMA_DTYPE_TF32_FTZ;
}
// Existing gather4 semantics, also used for ordinary tiled TMA loads. The
// TF32 conversion rounds to nearest-even and preserves signed zero/infinity.
inline uint32_t tma_convert_float_bits(uint32_t bits, unsigned type) {
  const uint32_t mag = bits & 0x7fffffffu;
  if ((type == TMA_DTYPE_F32_FTZ || type == TMA_DTYPE_TF32_FTZ) &&
      mag < 0x800000u)
    return bits & 0x80000000u;
  if (type != TMA_DTYPE_TF32 && type != TMA_DTYPE_TF32_FTZ)
    return bits;
  if (mag > 0x7f800000u)
    return 0x7fffe000u;
  if (mag == 0x7f800000u)
    return bits;
  return (bits + 0xfffu + ((bits >> 13) & 1u)) & ~0x1fffu;
}
inline uint32_t tma_float_oob_bits(unsigned type) {
  return type == TMA_DTYPE_TF32 || type == TMA_DTYPE_TF32_FTZ ? 0x7fffe000u
                                                              : 0x7fffffffu;
}
inline void convert_tma_load(unsigned type, void *data, size_t bytes) {
  if (!tma_float_conversion(type))
    return;
  assert(bytes % 4 == 0);
  for (size_t offset = 0; offset < bytes; offset += 4) {
    auto *address = static_cast<uint8_t *>(data) + offset;
    uint32_t bits;
    std::memcpy(&bits, address, 4);
    bits = tma_convert_float_bits(bits, type);
    std::memcpy(address, &bits, 4);
  }
}
// Explicit simulator Driver descriptors are distinct from opaque hardware
// payloads. Only floating conversion formats are handled here; legacy host
// formats retain their existing frontend validation.
inline bool decode_tma_float_host(const tensormap_descriptor_t &encoded,
                                  unsigned rank,
                                  tensormap_descriptor_t &decoded) {
  const auto &f = encoded.fields;
  if (rank < 1 || rank > 5 || !encoded.is_valid() ||
      encoded.num_dims() != rank || !tma_float_conversion(f.tensorDataType) ||
      f.interleave != TMA_INTERLEAVE_NONE || f.swizzle > TMA_SWIZZLE_128B ||
      f.oobFill > TMA_OOB_NAN)
    return false;
  for (unsigned i = 0; i < 5; ++i) {
    if (i < rank) {
      if (!f.globalDim[i] || !f.boxDim[i] || f.boxDim[i] > 256 ||
          f.elementStrides[i] != 1 || (i && !f.globalStrides[i - 1]))
        return false;
    } else if (f.globalDim[i] || f.boxDim[i] || f.elementStrides[i])
      return false;
    if (i >= rank - 1 && f.globalStrides[i])
      return false;
  }
  decoded = encoded;
  return true;
}
} // namespace flash_gpgpu_sim
