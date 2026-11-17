#include "descriptor.h"

#include <cassert>

namespace flash_gpgpu_sim {

namespace {

constexpr uint32_t kTcgen05DescriptorMask = 0x3fff;

uint32_t decode_matrix_descriptor_field(uint64_t desc, unsigned shift) {
  return static_cast<uint32_t>(((desc >> shift) & kTcgen05DescriptorMask) << 4);
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

tcgen05_mma_descriptor_t tcgen05_decode_f16_mma_descriptor(uint32_t idesc,
                                                           unsigned cta_group) {
  tcgen05_mma_descriptor_t decoded;
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

} // namespace flash_gpgpu_sim
