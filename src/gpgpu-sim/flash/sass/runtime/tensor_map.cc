#include "tensor_map.h"

#include "../../panic.h"
#include "../../tensormap.h"

#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace {

tensor_map_decode_result rejected(const char *detail) {
  tensor_map_decode_result result;
  result.detail = detail;
  return result;
}

template <size_t Rank>
tensor_map_nd_decode_result<Rank> rejected_nd(const char *detail) {
  tensor_map_nd_decode_result<Rank> result;
  result.detail = detail;
  return result;
}

uint32_t packed_u32(const std::array<uint64_t, 16> &words, size_t byte_offset) {
  const size_t index = byte_offset / sizeof(uint64_t);
  if (index >= words.size())
    panic("SASS tensor-map field offset is out of range");
  const uint64_t word = words[index];
  const unsigned shift =
      static_cast<unsigned>(byte_offset % sizeof(uint64_t)) * 8;
  return static_cast<uint32_t>(word >> shift);
}

uint64_t packed_u64(const std::array<uint64_t, 16> &words, size_t byte_offset) {
  return static_cast<uint64_t>(packed_u32(words, byte_offset)) |
         (static_cast<uint64_t>(packed_u32(words, byte_offset + 4)) << 32);
}

} // namespace

tensor_map_decode_result
decode_sm120_tensor_map_1d(const std::array<uint64_t, 8> &words) {
  // CUDA 13.3's device-side tensormap.replace sequence for rank-1 U32/F32
  // maps produces this compact layout: type/rank in word 1, global_dim-1 in
  // word 4, and box_dim-1 in the high byte of word 6.
  const uint16_t configuration = static_cast<uint16_t>(words[1]);
  const uint16_t swizzle_bits = configuration & 0x6000u;
  const uint16_t format = configuration & ~uint16_t{0x6000};
  if (format != 0x0100u && format != 0x0380u)
    return rejected("raw tensor map is not a validated 1D U32/F32 form");
  if (((words[1] >> 16) & 0xffffu) != 0 || (words[1] >> 32) != 0)
    return rejected("raw 1D tensor map has unsupported control fields");
  if (words[2] != 0 || words[3] != 0 || words[5] != 0 ||
      (words[4] >> 32) != 0 || (words[6] & 0x00ffffffffffffffull) != 0 ||
      words[7] != 0)
    return rejected("raw 1D tensor map has unsupported higher-rank fields");

  const uint32_t encoded_global_x = static_cast<uint32_t>(words[4]);
  if (encoded_global_x == std::numeric_limits<uint32_t>::max())
    return rejected("raw tensor-map dimension exceeds the functional range");

  tensor_map_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = format == 0x0100u
                                       ? tensor_map_element_type::kUnsigned32
                                       : tensor_map_element_type::kFloat32;
  result.descriptor.global_dim = {{encoded_global_x + 1, 1}};
  result.descriptor.box_dim = {
      {static_cast<uint32_t>((words[6] >> 56) + 1), 1}};
  result.descriptor.row_stride_bytes = 0;
  result.descriptor.element_stride = {{1, 1}};
  result.descriptor.swizzle_bytes = swizzle_bits == 0x2000u   ? 32
                                    : swizzle_bits == 0x4000u ? 64
                                    : swizzle_bits == 0x6000u ? 128
                                                              : 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_decode_result
decode_flashgpu_host_tensor_map_1d(const std::array<uint64_t, 16> &words) {
  // libcuda's cuTensorMapEncodeTiled compatibility implementation stores a
  // packed descriptor with rank-minus-one at byte 8, five box dimensions at
  // byte 12, five global dimensions at byte 32, five 64-bit global strides at
  // byte 52, five element strides at byte 92, and control words at byte 112.
  if (words[0] == 0 || packed_u32(words, 8) != 0)
    return rejected("host tensor map is not rank-1");
  const uint32_t box_x = packed_u32(words, 12);
  const uint32_t global_x = packed_u32(words, 32);
  if (box_x == 0 || global_x == 0)
    return rejected("host tensor map has an empty dimension");
  for (size_t dimension = 1; dimension < 5; ++dimension) {
    if (packed_u32(words, 12 + dimension * sizeof(uint32_t)) != 0 ||
        packed_u32(words, 32 + dimension * sizeof(uint32_t)) != 0 ||
        packed_u32(words, 92 + dimension * sizeof(uint32_t)) != 0)
      return rejected("host tensor map has unsupported higher-rank fields");
  }
  for (size_t stride_word = 0; stride_word < 5; ++stride_word) {
    const size_t offset = 52 + stride_word * sizeof(uint64_t);
    if (packed_u32(words, offset) != 0 ||
        packed_u32(words, offset + sizeof(uint32_t)) != 0)
      return rejected("rank-1 host tensor map has a global stride");
  }
  if (packed_u32(words, 92) != 1 || packed_u32(words, 112) != 7 ||
      packed_u32(words, 116) != 0 || packed_u32(words, 120) != 0 ||
      packed_u32(words, 124) != 0)
    return rejected("host tensor map is not contiguous rank-1 F32");

  tensor_map_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = tensor_map_element_type::kFloat32;
  result.descriptor.global_dim = {{global_x, 1}};
  result.descriptor.box_dim = {{box_x, 1}};
  result.descriptor.row_stride_bytes = 0;
  result.descriptor.element_stride = {{1, 1}};
  result.descriptor.swizzle_bytes = 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_decode_result
decode_flashgpu_host_tensor_map_2d(const std::array<uint64_t, 16> &words) {
  if (words[0] == 0 || packed_u32(words, 8) != 1)
    return rejected("host tensor map is not rank-2");
  const std::array<uint32_t, 2> box_dim{
      {packed_u32(words, 12), packed_u32(words, 16)}};
  const std::array<uint32_t, 2> global_dim{
      {packed_u32(words, 32), packed_u32(words, 36)}};
  if (box_dim[0] == 0 || box_dim[1] == 0 || global_dim[0] == 0 ||
      global_dim[1] == 0)
    return rejected("host tensor map has an empty dimension");
  for (size_t dimension = 2; dimension < 5; ++dimension) {
    if (packed_u32(words, 12 + dimension * sizeof(uint32_t)) != 0 ||
        packed_u32(words, 32 + dimension * sizeof(uint32_t)) != 0 ||
        packed_u32(words, 92 + dimension * sizeof(uint32_t)) != 0)
      return rejected("rank-2 host tensor map has higher-rank fields");
  }
  const uint64_t row_stride = packed_u64(words, 52);
  if (row_stride == 0)
    return rejected("rank-2 host tensor map has no row stride");
  for (size_t stride = 1; stride < 5; ++stride) {
    if (packed_u64(words, 52 + stride * sizeof(uint64_t)) != 0)
      return rejected("rank-2 host tensor map has higher-rank strides");
  }
  if (packed_u32(words, 92) != 1 || packed_u32(words, 96) != 1)
    return rejected("host tensor map has non-unit element strides");

  const uint32_t data_type = packed_u32(words, 112);
  if ((data_type != TMA_DTYPE_F32 && data_type != TMA_DTYPE_F32_FTZ) ||
      packed_u32(words, 116) != TMA_INTERLEAVE_NONE ||
      packed_u32(words, 120) != TMA_SWIZZLE_NONE ||
      packed_u32(words, 124) != TMA_OOB_ZERO)
    return rejected("host tensor map is not an unswizzled rank-2 F32 form");

  tensor_map_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = data_type == TMA_DTYPE_F32_FTZ
                                       ? tensor_map_element_type::kFloat32Ftz
                                       : tensor_map_element_type::kFloat32;
  result.descriptor.global_dim = global_dim;
  result.descriptor.box_dim = box_dim;
  result.descriptor.row_stride_bytes = row_stride;
  result.descriptor.element_stride = {{1, 1}};
  result.descriptor.swizzle_bytes = 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_4d_decode_result
decode_flashgpu_host_tensor_map_4d(const std::array<uint64_t, 16> &words) {
  // cuTensorMapEncodeTiled in the simulator stores its explicit compatibility
  // descriptor ABI in kernel parameters. Hopper UTMALDG addresses that
  // 128-byte payload through [URaddr],desc[URspace], rather than consuming the
  // compact device-generated encoding handled below.
  if (words[0] == 0 || packed_u32(words, 8) != 3)
    return rejected_nd<4>("host tensor map is not rank-4");

  functional_tensor_map_4d descriptor;
  descriptor.global_address = words[0];
  for (size_t dimension = 0; dimension < 4; ++dimension) {
    descriptor.box_dim[dimension] =
        packed_u32(words, 12 + dimension * sizeof(uint32_t));
    descriptor.global_dim[dimension] =
        packed_u32(words, 32 + dimension * sizeof(uint32_t));
    descriptor.element_stride[dimension] =
        packed_u32(words, 92 + dimension * sizeof(uint32_t));
    if (descriptor.box_dim[dimension] == 0 ||
        descriptor.global_dim[dimension] == 0 ||
        descriptor.element_stride[dimension] != 1)
      return rejected_nd<4>(
          "host rank-4 tensor map has empty dimensions or non-unit element "
          "strides");
  }
  if (packed_u32(words, 28) != 0 || packed_u32(words, 48) != 0 ||
      packed_u32(words, 108) != 0 || packed_u64(words, 76) != 0 ||
      packed_u64(words, 84) != 0)
    return rejected_nd<4>("host rank-4 tensor map has higher-rank fields");

  descriptor.element_bytes = 2;
  descriptor.element_type = tensor_map_element_type::kFloat16;
  descriptor.global_stride_bytes[0] = descriptor.element_bytes;
  for (size_t dimension = 1; dimension < 4; ++dimension) {
    descriptor.global_stride_bytes[dimension] =
        packed_u64(words, 52 + (dimension - 1) * sizeof(uint64_t));
    if (descriptor.global_stride_bytes[dimension] == 0)
      return rejected_nd<4>("host rank-4 tensor map has a zero global stride");
  }

  const uint32_t data_type = packed_u32(words, 112);
  const uint32_t interleave = packed_u32(words, 116);
  const uint32_t swizzle = packed_u32(words, 120);
  const uint32_t oob_fill = packed_u32(words, 124);
  if (data_type != TMA_DTYPE_F16 || interleave != TMA_INTERLEAVE_NONE ||
      swizzle > TMA_SWIZZLE_128B || oob_fill != TMA_OOB_ZERO)
    return rejected_nd<4>(
        "host rank-4 tensor map is not a validated tiled F16 form");
  descriptor.swizzle_bytes = swizzle == TMA_SWIZZLE_32B    ? 32
                             : swizzle == TMA_SWIZZLE_64B  ? 64
                             : swizzle == TMA_SWIZZLE_128B ? 128
                                                           : 0;
  descriptor.oob_zero = true;

  tensor_map_4d_decode_result result;
  result.supported = true;
  result.descriptor = descriptor;
  return result;
}

tensor_map_5d_decode_result
decode_flashgpu_host_tensor_map_5d(const std::array<uint64_t, 16> &words) {
  if (words[0] == 0 || packed_u32(words, 8) != 4)
    return rejected_nd<5>("host tensor map is not rank-5");

  functional_tensor_map_5d descriptor;
  descriptor.global_address = words[0];
  for (size_t dimension = 0; dimension < 5; ++dimension) {
    descriptor.box_dim[dimension] =
        packed_u32(words, 12 + dimension * sizeof(uint32_t));
    descriptor.global_dim[dimension] =
        packed_u32(words, 32 + dimension * sizeof(uint32_t));
    descriptor.element_stride[dimension] =
        packed_u32(words, 92 + dimension * sizeof(uint32_t));
    if (descriptor.box_dim[dimension] == 0 ||
        descriptor.global_dim[dimension] == 0 ||
        descriptor.element_stride[dimension] != 1)
      return rejected_nd<5>(
          "host rank-5 tensor map has empty dimensions or non-unit element "
          "strides");
  }
  if (packed_u64(words, 84) != 0)
    return rejected_nd<5>("host rank-5 tensor map has an extra global stride");

  descriptor.element_bytes = 2;
  descriptor.element_type = tensor_map_element_type::kFloat16;
  descriptor.global_stride_bytes[0] = descriptor.element_bytes;
  for (size_t dimension = 1; dimension < 5; ++dimension) {
    descriptor.global_stride_bytes[dimension] =
        packed_u64(words, 52 + (dimension - 1) * sizeof(uint64_t));
    if (descriptor.global_stride_bytes[dimension] == 0 &&
        descriptor.global_dim[dimension] != 1)
      return rejected_nd<5>(
          "host rank-5 tensor map has a zero stride for a non-unit dimension");
  }

  const uint32_t data_type = packed_u32(words, 112);
  const uint32_t interleave = packed_u32(words, 116);
  const uint32_t swizzle = packed_u32(words, 120);
  const uint32_t oob_fill = packed_u32(words, 124);
  if (data_type != TMA_DTYPE_F16 || interleave != TMA_INTERLEAVE_NONE ||
      swizzle > TMA_SWIZZLE_128B || oob_fill != TMA_OOB_ZERO)
    return rejected_nd<5>(
        "host rank-5 tensor map is not a validated tiled F16 form");
  descriptor.swizzle_bytes = swizzle == TMA_SWIZZLE_32B    ? 32
                             : swizzle == TMA_SWIZZLE_64B  ? 64
                             : swizzle == TMA_SWIZZLE_128B ? 128
                                                           : 0;
  descriptor.oob_zero = true;

  tensor_map_5d_decode_result result;
  result.supported = true;
  result.descriptor = descriptor;
  return result;
}

tensor_map_decode_result
decode_sm120_tensor_map_2d(const std::array<uint64_t, 8> &words) {
  // CUDA Driver cuTensorMapEncodeTiled fixtures and descriptors constructed by
  // the checked-in SM120 Triton cubin agree on this compact tiled-2D layout.
  // Keep the accepted subset narrow until each additional field is validated
  // by independent differential fixtures.
  const uint16_t configuration = static_cast<uint16_t>(words[1]);
  const uint16_t swizzle_bits = configuration & 0x6000u;
  const uint16_t format = configuration & ~uint16_t{0x6000};
  if (format != 0x0310u && format != 0x0390u)
    return rejected("raw tensor map is not a validated 2D FP16/F32 form");

  // Bits 16..31 are zero in device-constructed descriptors. The Driver API
  // sets 0x20 in this field for its equivalent host-created descriptor.
  const uint32_t control = static_cast<uint32_t>((words[1] >> 16) & 0xffffu);
  if (control != 0 && control != 0x20u)
    return rejected("raw tensor map has unsupported control fields");
  if (words[2] != 0 || words[3] != 0 || words[5] != 0)
    return rejected("raw tensor map has unsupported higher-rank fields");
  if ((words[6] & 0x00ffffffffffffffull) != 0 ||
      (words[7] & 0xffffffffffffff00ull) != 0)
    return rejected("raw tensor map has non-unit element strides or options");

  const uint64_t encoded_row_stride = words[1] >> 32;
  if (encoded_row_stride == 0 ||
      encoded_row_stride > std::numeric_limits<uint64_t>::max() / 16)
    return rejected("raw tensor map has an invalid row stride");
  const uint32_t encoded_global_x = static_cast<uint32_t>(words[4]);
  const uint32_t encoded_global_y = static_cast<uint32_t>(words[4] >> 32);
  if (encoded_global_x == std::numeric_limits<uint32_t>::max() ||
      encoded_global_y == std::numeric_limits<uint32_t>::max())
    return rejected("raw tensor-map dimension exceeds the functional range");

  tensor_map_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = format == 0x0310u ? 2 : 4;
  result.descriptor.element_type = format == 0x0310u
                                       ? tensor_map_element_type::kFloat16
                                       : tensor_map_element_type::kFloat32;
  result.descriptor.global_dim = {{encoded_global_x + 1, encoded_global_y + 1}};
  result.descriptor.box_dim = {{static_cast<uint32_t>((words[6] >> 56) + 1),
                                static_cast<uint32_t>((words[7] & 0xffu) + 1)}};
  result.descriptor.row_stride_bytes = encoded_row_stride * 16;
  result.descriptor.element_stride = {{1, 1}};
  result.descriptor.swizzle_bytes = swizzle_bits == 0x2000u   ? 32
                                    : swizzle_bits == 0x4000u ? 64
                                    : swizzle_bits == 0x6000u ? 128
                                                              : 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_3d_decode_result
decode_sm120_tensor_map_3d(const std::array<uint64_t, 8> &words) {
  // CUDA 13.3's device-side rank-3 F32 construction extends the compact 2D
  // layout with the second encoded global stride in word 2, global dimension
  // 2 in word 5, and box dimension 2 in the second byte of word 7.
  const uint16_t configuration = static_cast<uint16_t>(words[1]);
  const uint16_t swizzle_bits = configuration & 0x6000u;
  if ((configuration & ~uint16_t{0x6000}) != 0x03a0u)
    return rejected_nd<3>("raw tensor map is not the validated 3D F32 form");
  if (((words[1] >> 16) & 0xffffu) != 0)
    return rejected_nd<3>("raw 3D tensor map has unsupported control fields");
  if ((words[2] >> 32) != 0 || words[3] != 0 || (words[5] >> 32) != 0)
    return rejected_nd<3>(
        "raw 3D tensor map has unsupported higher-rank fields");
  if ((words[6] & 0x00ffffffffffffffull) != 0 ||
      (words[7] & 0xffffffffffff0000ull) != 0)
    return rejected_nd<3>(
        "raw 3D tensor map has non-unit element strides or options");

  const uint64_t encoded_row_stride = words[1] >> 32;
  const uint64_t encoded_plane_stride = static_cast<uint32_t>(words[2]);
  if (encoded_row_stride == 0 || encoded_plane_stride == 0 ||
      encoded_row_stride > std::numeric_limits<uint64_t>::max() / 16 ||
      encoded_plane_stride > std::numeric_limits<uint64_t>::max() / 16)
    return rejected_nd<3>("raw 3D tensor map has an invalid global stride");
  const std::array<uint32_t, 3> encoded_global_dim{{
      static_cast<uint32_t>(words[4]),
      static_cast<uint32_t>(words[4] >> 32),
      static_cast<uint32_t>(words[5]),
  }};
  for (uint32_t dimension : encoded_global_dim) {
    if (dimension == std::numeric_limits<uint32_t>::max())
      return rejected_nd<3>(
          "raw tensor-map dimension exceeds the functional range");
  }

  tensor_map_3d_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = tensor_map_element_type::kFloat32;
  for (size_t dimension = 0; dimension < encoded_global_dim.size(); ++dimension)
    result.descriptor.global_dim[dimension] = encoded_global_dim[dimension] + 1;
  result.descriptor.box_dim = {{
      static_cast<uint32_t>((words[6] >> 56) + 1),
      static_cast<uint32_t>((words[7] & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 8) & 0xffu) + 1),
  }};
  result.descriptor.global_stride_bytes = {
      {4, encoded_row_stride * 16, encoded_plane_stride * 16}};
  result.descriptor.element_stride = {{1, 1, 1}};
  result.descriptor.swizzle_bytes = swizzle_bits == 0x2000u   ? 32
                                    : swizzle_bits == 0x4000u ? 64
                                    : swizzle_bits == 0x6000u ? 128
                                                              : 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_4d_decode_result
decode_sm120_tensor_map_4d(const std::array<uint64_t, 8> &words) {
  // Rank 4 packs its third encoded global stride into the upper half of word
  // 2, dimensions 2/3 into word 5, and box dimensions 1..3 into word 7.
  const uint16_t configuration = static_cast<uint16_t>(words[1]);
  const uint16_t swizzle_bits = configuration & 0x6000u;
  if ((configuration & ~uint16_t{0x6000}) != 0x03b0u)
    return rejected_nd<4>("raw tensor map is not the validated 4D F32 form");
  if (((words[1] >> 16) & 0xffffu) != 0)
    return rejected_nd<4>("raw 4D tensor map has unsupported control fields");
  if (words[3] != 0 || (words[6] & 0x00ffffffffffffffull) != 0 ||
      (words[7] & 0xffffffffff000000ull) != 0)
    return rejected_nd<4>("raw 4D tensor map has higher-rank fields, element "
                          "strides, or options");

  const std::array<uint64_t, 3> encoded_global_stride{{
      words[1] >> 32,
      static_cast<uint32_t>(words[2]),
      words[2] >> 32,
  }};
  for (uint64_t stride : encoded_global_stride) {
    if (stride == 0 || stride > std::numeric_limits<uint64_t>::max() / 16)
      return rejected_nd<4>("raw 4D tensor map has an invalid global stride");
  }
  const std::array<uint32_t, 4> encoded_global_dim{{
      static_cast<uint32_t>(words[4]),
      static_cast<uint32_t>(words[4] >> 32),
      static_cast<uint32_t>(words[5]),
      static_cast<uint32_t>(words[5] >> 32),
  }};
  for (uint32_t dimension : encoded_global_dim) {
    if (dimension == std::numeric_limits<uint32_t>::max())
      return rejected_nd<4>(
          "raw tensor-map dimension exceeds the functional range");
  }

  tensor_map_4d_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = tensor_map_element_type::kFloat32;
  for (size_t dimension = 0; dimension < encoded_global_dim.size(); ++dimension)
    result.descriptor.global_dim[dimension] = encoded_global_dim[dimension] + 1;
  result.descriptor.box_dim = {{
      static_cast<uint32_t>((words[6] >> 56) + 1),
      static_cast<uint32_t>((words[7] & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 8) & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 16) & 0xffu) + 1),
  }};
  result.descriptor.global_stride_bytes = {{4, encoded_global_stride[0] * 16,
                                            encoded_global_stride[1] * 16,
                                            encoded_global_stride[2] * 16}};
  result.descriptor.swizzle_bytes = swizzle_bits == 0x2000u   ? 32
                                    : swizzle_bits == 0x4000u ? 64
                                    : swizzle_bits == 0x6000u ? 128
                                                              : 0;
  result.descriptor.oob_zero = true;
  return result;
}

tensor_map_5d_decode_result
decode_sm120_tensor_map_5d(const std::array<uint64_t, 8> &words) {
  // Rank 5 uses the lower half of word 3 for its fourth encoded global
  // stride, the lower half of word 6 for global dimension 4, and all four
  // low bytes of word 7 for box dimensions 1..4.
  const uint16_t configuration = static_cast<uint16_t>(words[1]);
  const uint16_t swizzle_bits = configuration & 0x6000u;
  if ((configuration & ~uint16_t{0x6000}) != 0x03c0u)
    return rejected_nd<5>("raw tensor map is not the validated 5D F32 form");
  if (((words[1] >> 16) & 0xffffu) != 0)
    return rejected_nd<5>("raw 5D tensor map has unsupported control fields");
  if ((words[3] >> 32) != 0 || (words[6] & 0x00ffffff00000000ull) != 0 ||
      (words[7] & 0xffffffff00000000ull) != 0)
    return rejected_nd<5>(
        "raw 5D tensor map has element strides or unsupported options");

  const std::array<uint64_t, 4> encoded_global_stride{{
      words[1] >> 32,
      static_cast<uint32_t>(words[2]),
      words[2] >> 32,
      static_cast<uint32_t>(words[3]),
  }};
  for (uint64_t stride : encoded_global_stride) {
    if (stride == 0 || stride > std::numeric_limits<uint64_t>::max() / 16)
      return rejected_nd<5>("raw 5D tensor map has an invalid global stride");
  }
  const std::array<uint32_t, 5> encoded_global_dim{{
      static_cast<uint32_t>(words[4]),
      static_cast<uint32_t>(words[4] >> 32),
      static_cast<uint32_t>(words[5]),
      static_cast<uint32_t>(words[5] >> 32),
      static_cast<uint32_t>(words[6]),
  }};
  for (uint32_t dimension : encoded_global_dim) {
    if (dimension == std::numeric_limits<uint32_t>::max())
      return rejected_nd<5>(
          "raw tensor-map dimension exceeds the functional range");
  }

  tensor_map_5d_decode_result result;
  result.supported = true;
  result.descriptor.global_address = words[0];
  result.descriptor.element_bytes = 4;
  result.descriptor.element_type = tensor_map_element_type::kFloat32;
  for (size_t dimension = 0; dimension < encoded_global_dim.size(); ++dimension)
    result.descriptor.global_dim[dimension] = encoded_global_dim[dimension] + 1;
  result.descriptor.box_dim = {{
      static_cast<uint32_t>((words[6] >> 56) + 1),
      static_cast<uint32_t>((words[7] & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 8) & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 16) & 0xffu) + 1),
      static_cast<uint32_t>(((words[7] >> 24) & 0xffu) + 1),
  }};
  result.descriptor.global_stride_bytes = {
      {4, encoded_global_stride[0] * 16, encoded_global_stride[1] * 16,
       encoded_global_stride[2] * 16, encoded_global_stride[3] * 16}};
  result.descriptor.swizzle_bytes = swizzle_bits == 0x2000u   ? 32
                                    : swizzle_bits == 0x4000u ? 64
                                    : swizzle_bits == 0x6000u ? 128
                                                              : 0;
  result.descriptor.oob_zero = true;
  return result;
}

} // namespace sass
} // namespace flash_gpgpu_sim
