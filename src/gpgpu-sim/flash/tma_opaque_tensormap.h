#ifndef FLASH_GPGPU_SIM_TMA_OPAQUE_TENSORMAP_H
#define FLASH_GPGPU_SIM_TMA_OPAQUE_TENSORMAP_H

#include "tensormap.h"

#include <cstdint>
#include <cstring>

namespace flash_gpgpu_sim {

inline uint32_t tma_opaque_low32(uint64_t value) {
  return static_cast<uint32_t>(value & 0xffffffffu);
}

inline uint32_t tma_opaque_high32(uint64_t value) {
  return static_cast<uint32_t>(value >> 32);
}

inline bool decode_sm100_opaque_tensormap(const tensormap_descriptor_t &encoded,
                                          tensormap_descriptor_t &decoded) {
  if (encoded.is_valid())
    return false;

  // Current CuTeDSL AOT export packs CUDA's opaque CUtensorMap by value in
  // kernel parameter memory instead of calling cuTensorMapEncodeTiled().  CUDA
  // does not document the byte layout, so this decoder is intentionally limited
  // to the tiled 4D descriptor shapes used by the FA4 SM100 forward kernel.
  //
  // The first 64 bytes use the same field positions in CUDA 12.9's public
  // cuTensorMapEncodeTiled() output and CUDA 13.1 / CuTeDSL 4.5.2's AOT output.
  // The descriptor magic differs between those encoders.
  //   raw_u64[0]      full global base address
  //   raw_u64[1].lo   descriptor magic plus operand flags
  //   raw_u64[1].hi   stride for dim1 in 16-byte units
  //   raw_u64[2].lo   stride for dim2 in 16-byte units
  //   raw_u64[2].hi   stride for dim3 in 16-byte units
  //   raw_u64[4]      dim0-1, dim1-1
  //   raw_u64[5]      dim2-1, dim3-1
  //   raw_u64[6].hi8  box dim0-1
  //   raw_u64[7].lo   box dim1-1
  constexpr uint32_t kMagicMask = 0x000fffffu;
  constexpr uint32_t kCudaApiMagic = 0x06330u;
  constexpr uint32_t kCuteDslMagic = 0x46332u;
  const uint32_t magic = tma_opaque_low32(encoded.raw_u64[1]) & kMagicMask;
  if (magic != kCudaApiMagic && magic != kCuteDslMagic)
    return false;
  if (encoded.raw_u64[0] == 0)
    return false;

  const uint64_t stride_dim1 =
      static_cast<uint64_t>(tma_opaque_high32(encoded.raw_u64[1])) * 16ull;
  const uint64_t stride_dim2 =
      static_cast<uint64_t>(tma_opaque_low32(encoded.raw_u64[2])) * 16ull;
  const uint64_t stride_dim3 =
      static_cast<uint64_t>(tma_opaque_high32(encoded.raw_u64[2])) * 16ull;
  if (stride_dim1 == 0 || stride_dim2 == 0 || stride_dim3 == 0)
    return false;

  const uint32_t global_dim0 = tma_opaque_low32(encoded.raw_u64[4]) + 1u;
  const uint32_t global_dim1 = tma_opaque_high32(encoded.raw_u64[4]) + 1u;
  const uint32_t global_dim2 = tma_opaque_low32(encoded.raw_u64[5]) + 1u;
  const uint32_t global_dim3 = tma_opaque_high32(encoded.raw_u64[5]) + 1u;
  const uint32_t box_dim0 =
      static_cast<uint32_t>((encoded.raw_u64[6] >> 56) & 0xffu) + 1u;
  const uint32_t box_dim1 = tma_opaque_low32(encoded.raw_u64[7]) + 1u;

  std::memset(&decoded, 0, sizeof(decoded));
  decoded.fields.globalAddress = encoded.raw_u64[0];
  decoded.fields.tensorRank = 3; // CUDA tensorRank=4, internal rank is 0-based.
  // FA4 SM100 forward currently uses fp16/bf16 TMA operands.  The observed
  // opaque descriptor bytes do not expose a stable public dtype field; for TMA
  // address generation both types are 2 bytes, so model the transfer as fp16.
  decoded.fields.tensorDataType = TMA_DTYPE_F16;
  decoded.fields.globalDim[0] = global_dim0;
  decoded.fields.globalDim[1] = global_dim1;
  decoded.fields.globalDim[2] = global_dim2;
  decoded.fields.globalDim[3] = global_dim3;
  decoded.fields.boxDim[0] = box_dim0;
  decoded.fields.boxDim[1] = box_dim1;
  decoded.fields.boxDim[2] = 1;
  decoded.fields.boxDim[3] = 1;
  for (unsigned i = 0; i < 4; ++i)
    decoded.fields.elementStrides[i] = 1;

  decoded.fields.globalStrides[0] = stride_dim1;
  decoded.fields.globalStrides[1] = stride_dim2;
  decoded.fields.globalStrides[2] = stride_dim3;
  decoded.fields.interleave = TMA_INTERLEAVE_NONE;
  decoded.fields.swizzle = TMA_SWIZZLE_128B;
  decoded.fields.oobFill = TMA_OOB_ZERO;

  return decoded.is_valid();
}

inline bool
decode_fa4_cutedsl_sm100_opaque_tensormap(const tensormap_descriptor_t &encoded,
                                          tensormap_descriptor_t &decoded) {
  return decode_sm100_opaque_tensormap(encoded, decoded);
}

} // namespace flash_gpgpu_sim

#endif
