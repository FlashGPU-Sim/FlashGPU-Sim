#include "descriptor.h"

#include <cassert>

namespace flash_gpgpu_sim {

namespace {

constexpr uint32_t kTcgen05DescriptorMask = 0x3fff;

uint32_t decode_matrix_descriptor_field(uint64_t desc, unsigned shift) {
  return static_cast<uint32_t>(((desc >> shift) & kTcgen05DescriptorMask) << 4);
}

uint32_t tcgen05_swizzle_bytes(uint8_t swizzle_mode) {
  switch (swizzle_mode) {
  case 2:
    return 128;
  case 4:
    return 64;
  case 6:
    return 32;
  default:
    return 0;
  }
}

uint64_t apply_tcgen05_swizzle(uint64_t base, uint64_t byte_offset,
                               uint32_t swizzle_bytes) {
  if (swizzle_bytes == 0)
    return base + byte_offset;

  const uint64_t period = static_cast<uint64_t>(swizzle_bytes) * 8;
  const uint64_t period_base = base & ~(period - 1);
  const uint64_t unswizzled = (base - period_base) + byte_offset;
  const unsigned swizzle_bits = swizzle_bytes == 128  ? 3
                                : swizzle_bytes == 64 ? 2
                                                      : 1;
  const uint64_t row_mask = ((1ull << swizzle_bits) - 1) << 7;
  return period_base + (unswizzled ^ ((unswizzled & row_mask) >> 3));
}

} // namespace

tcgen05_shared_descriptor_t tcgen05_decode_shared_descriptor(uint64_t desc) {
  tcgen05_shared_descriptor_t decoded;
  decoded.start_address = decode_matrix_descriptor_field(desc, 0);
  decoded.leading_dimension_byte_offset =
      decode_matrix_descriptor_field(desc, 16);
  decoded.stride_dimension_byte_offset =
      decode_matrix_descriptor_field(desc, 32);
  decoded.fixed_constant = static_cast<uint8_t>((desc >> 46) & 0x7);
  decoded.base_offset = static_cast<uint8_t>((desc >> 49) & 0x7);
  decoded.leading_dimension_absolute = ((desc >> 52) & 0x1) != 0;
  decoded.swizzle_mode = static_cast<uint8_t>((desc >> 61) & 0x7);

  assert(decoded.fixed_constant == 1 &&
         "TCGen05 shared descriptor fixed bits must be 0b001");
  assert((decoded.swizzle_mode == 0 || decoded.swizzle_mode == 1 ||
          decoded.swizzle_mode == 2 || decoded.swizzle_mode == 4 ||
          decoded.swizzle_mode == 6) &&
         "TCGen05 shared descriptor swizzle mode is invalid");
  return decoded;
}

uint32_t tcgen05_shared_k_major_packed_byte_address(
    const tcgen05_shared_descriptor_t &desc, uint32_t matrix_row,
    uint32_t packed_k, uint32_t default_packed_k_per_row) {
  assert(!desc.leading_dimension_absolute &&
         "TCGen05 absolute leading dimension is not supported");
  assert(desc.swizzle_mode != 1 &&
         "TCGen05 K-major does not support 128B/32B-atomic swizzle");
  assert(default_packed_k_per_row > 0);

  const uint32_t swizzle_bytes = tcgen05_swizzle_bytes(desc.swizzle_mode);
  if (swizzle_bytes != 0) {
    constexpr uint32_t kRowsPerAtom = 8;
    constexpr uint32_t kPackedBytesPer128Bits = 16;
    const uint64_t row_group = matrix_row / kRowsPerAtom;
    const uint32_t row_in_group = matrix_row % kRowsPerAtom;
    const uint64_t k_group = packed_k / kPackedBytesPer128Bits;
    const uint32_t k_in_group = packed_k % kPackedBytesPer128Bits;
    const uint64_t leading = desc.leading_dimension_byte_offset == 0
                                 ? kPackedBytesPer128Bits
                                 : desc.leading_dimension_byte_offset;
    const uint64_t offset = row_group * desc.stride_dimension_byte_offset +
                            row_in_group * swizzle_bytes + k_group * leading +
                            k_in_group;
    return static_cast<uint32_t>(
        apply_tcgen05_swizzle(desc.start_address, offset, swizzle_bytes));
  }

  if (desc.leading_dimension_byte_offset == 0 &&
      desc.stride_dimension_byte_offset == 0) {
    return desc.start_address + matrix_row * default_packed_k_per_row +
           packed_k;
  }

  uint32_t packed_k_per_atom = desc.stride_dimension_byte_offset;
  if (packed_k_per_atom == 0)
    packed_k_per_atom = default_packed_k_per_row;
  const uint64_t leading = desc.leading_dimension_byte_offset;
  return static_cast<uint32_t>(
      desc.start_address + (packed_k / packed_k_per_atom) * leading +
      static_cast<uint64_t>(matrix_row) * desc.stride_dimension_byte_offset +
      packed_k % packed_k_per_atom);
}

bool tcgen05_mxf4_dense_shape_supported(uint32_t m, uint32_t n, uint32_t k,
                                        unsigned cta_group) {
  if (cta_group == 1)
    return m == 128 && k == 64 && n >= 8 && n <= 256 && n % 8 == 0;
  if (cta_group != 2 || n < 16 || n > 256 || n % 16 != 0)
    return false;
  if (k == 64)
    return m == 128 || m == 256;
  return m == 256 && k == 96;
}

tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group) {
  tcgen05_mma_descriptor_t decoded = {};
  decoded.sparsity_selector = idesc & 0x3;
  decoded.sparse = ((idesc >> 2) & 0x1) != 0;
  decoded.d_type = static_cast<uint8_t>((idesc >> 4) & 0x3);
  decoded.a_type = static_cast<uint8_t>((idesc >> 7) & 0x7);
  decoded.b_type = static_cast<uint8_t>((idesc >> 10) & 0x7);
  decoded.negate_a = ((idesc >> 13) & 0x1) != 0;
  decoded.negate_b = ((idesc >> 14) & 0x1) != 0;
  decoded.transpose_a = ((idesc >> 15) & 0x1) != 0;
  decoded.transpose_b = ((idesc >> 16) & 0x1) != 0;
  decoded.n = ((idesc >> 17) & 0x3f) << 3;
  decoded.m = ((idesc >> 24) & 0x1f) << 4;
  decoded.k = decoded.sparse ? 32 : 16;

  assert(cta_group == 1 && "Only TCGen05 cta_group::1 is supported");
  assert(!decoded.sparse && "Sparse TCGen05 f16 MMA is not supported");
  assert(((idesc >> 3) & 0x1) == 0 &&
         "TCGen05 f16 MMA saturate bit must be zero");
  assert(((idesc >> 6) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 6 must be zero");
  assert(((idesc >> 23) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 23 must be zero");
  assert(((idesc >> 29) & 0x1) == 0 &&
         "TCGen05 f16 MMA reserved bit 29 must be zero");
  assert(decoded.d_type == TCGEN05_MMA_TYPE_FIELD_ONE &&
         "Only TCGen05 f16 MMA with f32 D is supported");
  assert((decoded.a_type == TCGEN05_MMA_TYPE_FIELD_F16 ||
          decoded.a_type == TCGEN05_MMA_TYPE_FIELD_ONE) &&
         "Only TCGen05 f16/BF16 A input is supported");
  assert((decoded.b_type == TCGEN05_MMA_TYPE_FIELD_F16 ||
          decoded.b_type == TCGEN05_MMA_TYPE_FIELD_ONE) &&
         "Only TCGen05 f16/BF16 B input is supported");
  assert((decoded.m == 64 || decoded.m == 128) &&
         "TCGen05 f16 dense cta_group::1 supports M=64 or M=128");
  assert(decoded.n >= 8 && decoded.n <= 256 && decoded.n % 8 == 0 &&
         "TCGen05 f16 dense cta_group::1 supports N in steps of 8");
  return decoded;
}

tcgen05_mma_descriptor_t
tcgen05_decode_mxf4_mma_descriptor(uint32_t idesc, unsigned cta_group) {
  tcgen05_mma_descriptor_t decoded = {};
  decoded.sparsity_selector = idesc & 0x3;
  decoded.sparse = ((idesc >> 2) & 0x1) != 0;
  decoded.b_scale_factor_id = static_cast<uint8_t>((idesc >> 4) & 0x3);
  decoded.a_type = static_cast<uint8_t>((idesc >> 7) & 0x7);
  decoded.b_type = static_cast<uint8_t>((idesc >> 10) & 0x7);
  decoded.negate_a = ((idesc >> 13) & 0x1) != 0;
  decoded.negate_b = ((idesc >> 14) & 0x1) != 0;
  decoded.transpose_a = ((idesc >> 15) & 0x1) != 0;
  decoded.transpose_b = ((idesc >> 16) & 0x1) != 0;
  decoded.n = ((idesc >> 17) & 0x3f) << 3;
  decoded.scale_format = static_cast<uint8_t>((idesc >> 23) & 0x1);
  decoded.m = ((idesc >> 27) & 0x3) << 7;
  decoded.a_scale_factor_id = static_cast<uint8_t>((idesc >> 29) & 0x3);
  decoded.k = ((idesc >> 31) & 0x1) ? 96 : 64;
  decoded.d_type = TCGEN05_MMA_TYPE_FIELD_ONE;

  assert((cta_group == 1 || cta_group == 2) &&
         "TCGen05 MXFP4 CTA group must be 1 or 2");
  assert(!decoded.sparse && "Sparse TCGen05 MXFP4 MMA is not supported");
  assert(((idesc >> 3) & 0x1) == 0 &&
         "TCGen05 MXFP4 reserved bit 3 must be zero");
  assert(((idesc >> 6) & 0x1) == 0 &&
         "TCGen05 MXFP4 reserved bit 6 must be zero");
  assert(((idesc >> 12) & 0x1) == 0 &&
         "TCGen05 MXFP4 reserved bit 12 must be zero");
  assert(((idesc >> 24) & 0x7) == 0 &&
         "TCGen05 MXFP4 reserved bits 24-26 must be zero");
  assert(decoded.a_type == TCGEN05_MXF4_FORMAT_E2M1 &&
         decoded.b_type == TCGEN05_MXF4_FORMAT_E2M1 &&
         "Only E2M1 TCGen05 MXFP4 inputs are supported");
  assert(!decoded.transpose_a && !decoded.transpose_b &&
         "TCGen05 MXFP4 E2M1 inputs must be K-major");
  assert(decoded.scale_format == TCGEN05_SCALE_FORMAT_UE8M0 &&
         "Only UE8M0 TCGen05 MXFP4 scale factors are supported");
  if (decoded.k == 64) {
    assert((decoded.a_scale_factor_id == 0 || decoded.a_scale_factor_id == 2) &&
           (decoded.b_scale_factor_id == 0 || decoded.b_scale_factor_id == 2) &&
           "TCGen05 MXFP4 K=64 scale-factor IDs must be 0 or 2");
  }
  assert(tcgen05_mxf4_dense_shape_supported(decoded.m, decoded.n, decoded.k,
                                            cta_group) &&
         "Unsupported dense TCGen05 MXFP4 shape");
  return decoded;
}

} // namespace flash_gpgpu_sim
