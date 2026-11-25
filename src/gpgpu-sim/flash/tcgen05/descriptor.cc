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
  case 1:
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
                               uint32_t swizzle_bytes, bool atom32 = false) {
  if (swizzle_bytes == 0)
    return base + byte_offset;

  const uint64_t period =
      static_cast<uint64_t>(swizzle_bytes) * (atom32 ? 4 : 8);
  const uint64_t period_base = base & ~(period - 1);
  const uint64_t unswizzled = (base - period_base) + byte_offset;
  const unsigned swizzle_bits = atom32                 ? 2
                                : swizzle_bytes == 128 ? 3
                                : swizzle_bytes == 64  ? 2
                                                       : 1;
  const uint64_t row_mask = ((1ull << swizzle_bits) - 1) << 7;
  return period_base +
         (unswizzled ^ ((unswizzled & row_mask) >> (atom32 ? 2 : 3)));
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

bool tcgen05_mma_shared_descriptor_supported(uint64_t raw, bool mn_major) {
  const unsigned mode = raw >> 61;
  // 14-bit address/stride fields on Blackwell, not Rubin's widened fields.
  const uint64_t reserved = (3ull << 14) | (3ull << 30) | (0xffull << 53);
  // Pattern-base offsets are not implemented by the address helpers below.
  // Reject them in both adapters rather than silently using pattern base zero.
  return !(raw & reserved) && ((raw >> 46) & 7) == 1 && !((raw >> 49) & 0xf) &&
         (mode == 0 || mode == 2 || mode == 4 || mode == 6 ||
          (mode == 1 && mn_major));
}

tcgen05_shared_descriptor_t
tcgen05_decode_mma_shared_descriptor(uint64_t raw, bool mn_major) {
  assert(tcgen05_mma_shared_descriptor_supported(raw, mn_major) &&
         "Unsupported Blackwell MMA shared descriptor");
  return tcgen05_decode_shared_descriptor(raw);
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
    const uint64_t row_group = matrix_row / kRowsPerAtom;
    const uint32_t row_in_group = matrix_row % kRowsPerAtom;
    // PTX canonical swizzled K-major layout has contiguous K bytes.
    // LBO is ignored (assumed one 128-bit unit), even when cuBLAS leaves
    // a non-unit value in the descriptor's relative leading field.
    const uint64_t offset = row_group * desc.stride_dimension_byte_offset +
                            row_in_group * swizzle_bytes + packed_k;
    return static_cast<uint32_t>(
        apply_tcgen05_swizzle(desc.start_address, offset, swizzle_bytes));
  }

  if (desc.leading_dimension_byte_offset == 0 &&
      desc.stride_dimension_byte_offset == 0) {
    return desc.start_address + matrix_row * default_packed_k_per_row +
           packed_k;
  }

  // Canonical interleaved K-major layout in 16-byte units:
  // ((8,m),(1,k)):((1,SBO),(1,LBO)). SBO steps eight rows,
  // not a single row; LBO steps one 16-byte K segment.
  return static_cast<uint32_t>(
      desc.start_address +
      (matrix_row / 8) * uint64_t(desc.stride_dimension_byte_offset) +
      (matrix_row % 8) * 16 +
      (packed_k / 16) * uint64_t(desc.leading_dimension_byte_offset) +
      packed_k % 16);
}

uint32_t
tcgen05_shared_mn_major_byte_address(const tcgen05_shared_descriptor_t &desc,
                                     uint32_t matrix_mn, uint32_t matrix_k,
                                     uint32_t element_size,
                                     uint32_t default_mn_elements_per_row) {
  assert(!desc.leading_dimension_absolute &&
         "TCGen05 absolute leading dimension is only valid for K-major");
  assert(element_size > 0 && 16 % element_size == 0);
  assert(default_mn_elements_per_row > 0);

  // A descriptor with no strides is useful for small hand-written PTX tests.
  // Its natural MN-major representation is K rows with MN contiguous.
  if (desc.leading_dimension_byte_offset == 0 &&
      desc.stride_dimension_byte_offset == 0) {
    return desc.start_address +
           (static_cast<uint64_t>(matrix_k) * default_mn_elements_per_row +
            matrix_mn) *
               element_size;
  }

  // PTX ISA 9.7.17.3.3 canonical MN-major layout, expressed in bytes.  T is
  // the number of input elements in a normalized 128-bit element.  K is
  // grouped in eights; the descriptor's leading/stride fields advance the MN
  // and K outer groups respectively for swizzled layouts.
  const uint32_t swizzle_bytes = tcgen05_swizzle_bytes(desc.swizzle_mode);
  const uint32_t elements_per_128_bits = 16 / element_size;
  const uint32_t mn_per_group =
      swizzle_bytes == 0 ? elements_per_128_bits : swizzle_bytes / element_size;
  const uint64_t mn_group = matrix_mn / mn_per_group;
  const uint32_t mn_in_group = matrix_mn % mn_per_group;
  // CUTLASS UMMA::make_umma_desc uses four K rows for SW128_BASE32B,
  // eight otherwise. Swizzle<2,5,2> preserves each 32-byte atom.
  const unsigned k_atom = desc.swizzle_mode == 1 ? 4 : 8;
  const uint64_t k_group = matrix_k / k_atom;
  const uint32_t k_in_group = matrix_k % k_atom;

  uint64_t offset = 0;
  if (swizzle_bytes == 0) {
    offset = mn_group * desc.stride_dimension_byte_offset +
             static_cast<uint64_t>(mn_in_group) * element_size +
             k_group * desc.leading_dimension_byte_offset +
             static_cast<uint64_t>(k_in_group) * 16;
  } else {
    offset = mn_group * desc.leading_dimension_byte_offset +
             static_cast<uint64_t>(mn_in_group) * element_size +
             k_group * desc.stride_dimension_byte_offset +
             static_cast<uint64_t>(k_in_group) * swizzle_bytes;
  }
  return static_cast<uint32_t>(apply_tcgen05_swizzle(
      desc.start_address, offset, swizzle_bytes, desc.swizzle_mode == 1));
}

bool tcgen05_mxf4_dense_shape_supported(uint32_t m, uint32_t n, uint32_t k,
                                        unsigned cta_group) {
  if (cta_group == 1)
    return m == 128 && (k == 64 || k == 128) && n >= 8 && n <= 256 &&
           n % 8 == 0;
  if (cta_group != 2 || n < 16 || n > 256 || n % 16 != 0)
    return false;
  if (k == 64)
    return m == 128 || m == 256;
  return m == 256 && k == 96;
}

bool tcgen05_mxf8f6f4_format_supported(uint8_t format) {
  switch (format) {
  case TCGEN05_MXF8F6F4_FORMAT_E4M3:
  case TCGEN05_MXF8F6F4_FORMAT_E5M2:
  case TCGEN05_MXF8F6F4_FORMAT_E2M3:
  case TCGEN05_MXF8F6F4_FORMAT_E3M2:
  case TCGEN05_MXF8F6F4_FORMAT_E2M1:
    return true;
  default:
    return false;
  }
}

bool tcgen05_blackwell_mixed_descriptor_supported(uint32_t idesc,
                                                  unsigned group) {
  // This is the execution subset, not the full ISA decoder below. Both
  // adapters currently gather padded K-major operands only; accepting the
  // otherwise legal eight-bit transpose would feed a different layout to MMA.
  constexpr uint32_t unsupported =
      (1u << 31) | (7u << 24) | (3u << 15) | (1u << 6) | 0xfu;
  return !(idesc & unsupported) && (idesc & (1u << 23)) &&
         tcgen05_mxf8f6f4_format_supported((idesc >> 7) & 7) &&
         tcgen05_mxf8f6f4_format_supported((idesc >> 10) & 7) &&
         tcgen05_mxf8f6f4_dense_shape_supported(
             ((idesc >> 27) & 3) * 128, ((idesc >> 17) & 63) * 8, 32, group);
}

uint32_t tcgen05_mxf8f6f4_format_bits(uint8_t format) {
  switch (format) {
  case TCGEN05_MXF8F6F4_FORMAT_E4M3:
  case TCGEN05_MXF8F6F4_FORMAT_E5M2:
    return 8;
  case TCGEN05_MXF8F6F4_FORMAT_E2M3:
  case TCGEN05_MXF8F6F4_FORMAT_E3M2:
    return 6;
  case TCGEN05_MXF8F6F4_FORMAT_E2M1:
    return 4;
  default:
    assert(false && "Invalid TCGen05 MXF8F6F4 input format");
    return 0;
  }
}

tcgen05_mxf8f6f4_shared_location_t
tcgen05_mxf8f6f4_shared_location(uint32_t logical_k, uint8_t format) {
  const uint32_t element_bits = tcgen05_mxf8f6f4_format_bits(format);
  const uint32_t element_in_group = logical_k % 16;
  const uint32_t bit_offset = element_in_group * element_bits;
  return {/*byte_offset=*/(logical_k / 16) * 16 + bit_offset / 8,
          /*bit_offset=*/static_cast<uint8_t>(bit_offset % 8)};
}

bool tcgen05_mxf8f6f4_dense_shape_supported(uint32_t m, uint32_t n, uint32_t k,
                                            unsigned cta_group) {
  if (k != 32 && k != 64)
    return false;
  if (cta_group == 1)
    return m == 128 && n >= 8 && n <= 256 && n % 8 == 0;
  return cta_group == 2 && (m == 256 || (m == 128 && k == 32)) && n >= 16 &&
         n <= 256 && n % 16 == 0;
}

tcgen05_mma_descriptor_t tcgen05_decode_fp8_mma_descriptor(uint32_t idesc) {
  tcgen05_mma_descriptor_t d{};
  d.m = ((idesc >> 24) & 31) * 16;
  d.n = ((idesc >> 17) & 63) * 8;
  d.k = 32;
  d.a_type = (idesc >> 7) & 7;
  d.b_type = (idesc >> 10) & 7;
  d.d_type = (idesc >> 4) & 3;
  assert(!(idesc & 0xe081804fu) && d.d_type == 1 && d.a_type <= 1 &&
         d.b_type <= 1 && d.m == 128 && d.n >= 16 && d.n <= 256 &&
         d.n % 16 == 0);
  d.negate_a = idesc & (1u << 13);
  d.negate_b = idesc & (1u << 14);
  d.scale_format = TCGEN05_SCALE_FORMAT_UE8M0;
  return d;
}

bool tcgen05_mma_tmem_alignment_supported(unsigned local_m, bool folded,
                                          uint32_t d, bool tmem_a, uint32_t a) {
  const unsigned lane = d >> 16;
  const bool half_datapaths = local_m == 64 && !folded;
  if (lane != 0 && !(half_datapaths && lane == 16))
    return false;
  return !tmem_a || ((a >> 16) == lane && (a & 7) == 0);
}

bool tcgen05_tf32_descriptor_supported(uint32_t idesc) {
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  // Qualified dense CTA1 K-major TF32 -> F32 forms (shared or TMEM A).
  return !(idesc & 0xe081804fu) && ((idesc >> 4) & 3) == 1 &&
         ((idesc >> 7) & 7) == 2 && ((idesc >> 10) & 7) == 2 &&
         (m == 64 || m == 128) && n >= 8 && n <= 256 && n % 8 == 0;
}

bool tcgen05_f16_descriptor_supported(uint32_t idesc, unsigned group, bool ws) {
  if (group != 1 && group != 2)
    return false;
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  const unsigned a = (idesc >> 7) & 7, b = (idesc >> 10) & 7;
  const unsigned d = (idesc >> 4) & 3;
  if ((idesc & 0xe080004fu) || d > 1 || a > 1 || b > 1 ||
      (d == 0 && (a != 0 || b != 0)))
    return false;
  if (ws)
    return group == 1 && m == 64 && (n == 64 || n == 128 || n == 256);
  return (group == 1 ? (m == 64 || m == 128) : (m == 128 || m == 256)) &&
         n >= 8 * group && n <= 256 && n % (8 * group) == 0;
}

tcgen05_mma_descriptor_t
tcgen05_decode_f16_mma_descriptor(uint32_t idesc, unsigned cta_group, bool ws) {
  assert(tcgen05_f16_descriptor_supported(idesc, cta_group, ws) &&
         "Unsupported dense F16/BF16 instruction descriptor");
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

  return decoded;
}

tcgen05_mma_descriptor_t tcgen05_decode_mxf4_mma_descriptor(uint32_t idesc,
                                                            unsigned cta_group,
                                                            bool sm107) {
  tcgen05_mma_descriptor_t decoded = {};
  decoded.sparsity_selector = idesc & 0x3;
  decoded.sparse = ((idesc >> 2) & 0x1) != 0;
  decoded.b_scale_factor_id = static_cast<uint8_t>((idesc >> 4) & 0x3);
  decoded.a_type = static_cast<uint8_t>((idesc >> 7) & 0x7);
  decoded.b_type = static_cast<uint8_t>((idesc >> 10) & 0x3);
  decoded.negate_a = ((idesc >> 13) & 0x1) != 0;
  decoded.negate_b = ((idesc >> 14) & 0x1) != 0;
  decoded.transpose_a = ((idesc >> 15) & 0x1) != 0;
  decoded.transpose_b = ((idesc >> 16) & 0x1) != 0;
  decoded.n = ((idesc >> 17) & 0x3f) << 3;
  decoded.scale_format = static_cast<uint8_t>((idesc >> 23) & 0x1);
  decoded.m = ((idesc >> 27) & 0x3) << 7;
  decoded.a_scale_factor_id = static_cast<uint8_t>((idesc >> 29) & 0x3);
  const unsigned k_code = ((idesc >> 31) & 1) | (((idesc >> 3) & 1) << 1);
  decoded.k = k_code == 2 ? 128 : k_code == 1 ? 96 : 64;
  decoded.d_type = TCGEN05_MMA_TYPE_FIELD_ONE;

  assert((cta_group == 1 || cta_group == 2) &&
         "TCGen05 MXFP4 CTA group must be 1 or 2");
  assert(!decoded.sparse && "Sparse TCGen05 MXFP4 MMA is not supported");
  assert(k_code < 3 && (sm107 || k_code < 2) &&
         "TCGen05 MXFP4 K128 requires SM107; K code3 is reserved");
  assert(((idesc >> 6) & 0x1) == 0 &&
         "TCGen05 MXFP4 reserved bit 6 must be zero");
  assert(((idesc >> 12) & 1) == unsigned(sm107) &&
         "TCGen05 MXFP4 sparsity version must match target architecture");
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
  if (decoded.k == 128)
    assert(decoded.a_scale_factor_id == 0 && decoded.b_scale_factor_id == 0 &&
           "K128 consumes all four block32 scale bytes from ID0");
  assert(tcgen05_mxf4_dense_shape_supported(decoded.m, decoded.n, decoded.k,
                                            cta_group) &&
         "Unsupported dense TCGen05 MXFP4 shape");
  return decoded;
}

tcgen05_mma_descriptor_t
tcgen05_decode_mxf8f6f4_mma_descriptor(uint32_t idesc, unsigned cta_group) {
  tcgen05_mma_descriptor_t decoded = {};
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
  decoded.k = (idesc & 0x80000000u) ? 64 : 32;
  if (decoded.sparse)
    decoded.k *= 2;
  decoded.d_type = TCGEN05_MMA_TYPE_FIELD_ONE;

  assert((cta_group == 1 || cta_group == 2) &&
         "TCGen05 MXF8F6F4 CTA group must be 1 or 2");
  assert(!decoded.sparse && "Sparse TCGen05 MXF8F6F4 MMA is not supported");
  assert((idesc & 0x3) == 0 &&
         "TCGen05 MXF8F6F4 reserved bits 0-1 must be zero");
  assert(((idesc >> 3) & 0x1) == 0 &&
         "TCGen05 MXF8F6F4 reserved bit 3 must be zero");
  assert(((idesc >> 6) & 0x1) == 0 &&
         "TCGen05 MXF8F6F4 reserved bit 6 must be zero");
  assert(((idesc >> 24) & 0x7) == 0 &&
         "TCGen05 MXF8F6F4 reserved bits 24-26 must be zero");
  assert((decoded.k != 64 || (decoded.a_scale_factor_id % 2 == 0 &&
                              decoded.b_scale_factor_id % 2 == 0)) &&
         "TCGen05 MXF8F6F4 K64 requires aligned pairs of block32 scales");
  assert(tcgen05_mxf8f6f4_format_supported(decoded.a_type) &&
         tcgen05_mxf8f6f4_format_supported(decoded.b_type) &&
         "Unsupported TCGen05 MXF8F6F4 input format");
  assert(decoded.scale_format == TCGEN05_SCALE_FORMAT_UE8M0 &&
         "TCGen05 MXF8F6F4 requires UE8M0 scale factors");

  // PTX only defines MN-major (transpose) narrow inputs for eight-bit element
  // formats.  DeepGEMM uses K-major operands, but validate the descriptor here
  // so other callers do not accidentally reinterpret padded FP4/FP6 storage.
  assert((!decoded.transpose_a ||
          tcgen05_mxf8f6f4_format_bits(decoded.a_type) == 8) &&
         (!decoded.transpose_b ||
          tcgen05_mxf8f6f4_format_bits(decoded.b_type) == 8) &&
         "TCGen05 MXF8F6F4 transpose requires an eight-bit operand");
  assert(tcgen05_mxf8f6f4_dense_shape_supported(decoded.m, decoded.n, decoded.k,
                                                cta_group) &&
         "Unsupported dense TCGen05 MXF8F6F4 shape");
  if (decoded.transpose_b) {
    const uint32_t n_step = cta_group == 1 ? 16 : 32;
    assert(decoded.n % n_step == 0 &&
           "TCGen05 MXF8F6F4 transposed eight-bit B has a wider N step");
  }
  return decoded;
}

} // namespace flash_gpgpu_sim
