#ifndef FLASH_GPGPU_SIM_SASS_TENSOR_MAP_H_
#define FLASH_GPGPU_SIM_SASS_TENSOR_MAP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {

// Result of decoding the compact hardware tensor-map payload consumed by TMA
// load/store. This is intentionally separate from the expanded PTX functional
// descriptor in flash/tensormap.h.
struct tensor_map_decode_result {
  bool supported = false;
  functional_tensor_map_2d descriptor;
  std::string detail;
};

template <size_t Rank> struct functional_tensor_map_nd {
  uint64_t global_address = 0;
  uint32_t element_bytes = 0;
  tensor_map_element_type element_type = tensor_map_element_type::kUnknown;
  std::array<uint32_t, Rank> global_dim{};
  std::array<uint32_t, Rank> box_dim{};
  std::array<uint64_t, Rank> global_stride_bytes{};
  std::array<uint32_t, Rank> element_stride{};
  uint32_t swizzle_bytes = 0;
  bool oob_zero = true;

  functional_tensor_map_nd() { element_stride.fill(1); }
};

template <size_t Rank> struct tensor_map_nd_decode_result {
  bool supported = false;
  functional_tensor_map_nd<Rank> descriptor;
  std::string detail;
};

using functional_tensor_map_3d = functional_tensor_map_nd<3>;
using functional_tensor_map_4d = functional_tensor_map_nd<4>;
using functional_tensor_map_5d = functional_tensor_map_nd<5>;
using tensor_map_3d_decode_result = tensor_map_nd_decode_result<3>;
using tensor_map_4d_decode_result = tensor_map_nd_decode_result<4>;
using tensor_map_5d_decode_result = tensor_map_nd_decode_result<5>;

// Decodes only the differentially validated SM120 tiled-1D U32/F32, tiled-2D
// FP16/F32, and tiled-3D/4D/5D F32 subsets. The first eight qwords contain all
// fields used by those subsets. Unknown ranks, data types, control modes, and
// element strides fail closed.
tensor_map_decode_result
decode_sm120_tensor_map_1d(const std::array<uint64_t, 8> &words);

// The simulator-side cuTensorMapEncodeTiled implementation exposes a stable,
// explicit 128-byte descriptor ABI rather than NVIDIA's opaque hardware
// encoding. Keep these rank-1/rank-2 compatibility formats separate from the
// official SM120 decoder so neither format can be mistaken for the other.
tensor_map_decode_result
decode_flashgpu_host_tensor_map_1d(const std::array<uint64_t, 16> &words);

tensor_map_decode_result
decode_flashgpu_host_tensor_map_2d(const std::array<uint64_t, 16> &words);

tensor_map_4d_decode_result
decode_flashgpu_host_tensor_map_4d(const std::array<uint64_t, 16> &words);

tensor_map_5d_decode_result
decode_flashgpu_host_tensor_map_5d(const std::array<uint64_t, 16> &words);

tensor_map_decode_result
decode_sm120_tensor_map_2d(const std::array<uint64_t, 8> &words);

tensor_map_3d_decode_result
decode_sm120_tensor_map_3d(const std::array<uint64_t, 8> &words);

tensor_map_4d_decode_result
decode_sm120_tensor_map_4d(const std::array<uint64_t, 8> &words);

tensor_map_5d_decode_result
decode_sm120_tensor_map_5d(const std::array<uint64_t, 8> &words);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_TENSOR_MAP_H_
