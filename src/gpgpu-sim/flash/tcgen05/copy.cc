#include "copy.h"
namespace flash_gpgpu_sim {
bool tcgen05_copy_addresses(uint64_t raw, tcgen05_copy_shape shape,
                            std::vector<uint32_t> &out) {
  out.clear();
  const unsigned mode = raw >> 61;
  if (((raw >> 46) & 7) != 1 || ((raw >> 49) & 0xfff) || ((raw >> 14) & 3) ||
      ((raw >> 30) & 3) || (mode != 0 && mode != 2 && mode != 4 && mode != 6))
    return false;
  const unsigned start = (raw & 0x3fff) << 4;
  const unsigned lbo = ((raw >> 16) & 0x3fff) << 4;
  const unsigned sbo = ((raw >> 32) & 0x3fff) << 4;
  const unsigned swizzle = mode == 2   ? 128
                           : mode == 4 ? 64
                           : mode == 6 ? 32
                                       : 0;
  const bool wide = shape == tcgen05_copy_shape::rows128_bits256 ||
                    shape == tcgen05_copy_shape::rows128_bits128;
  const unsigned words = shape == tcgen05_copy_shape::rows128_bits256 ? 8 : 4;
  if (wide) {
    if (start % (4 * words) ||
        (swizzle ? lbo != 16 || sbo != 8 * swizzle : lbo != 2048 || sbo != 128))
      return false;
  } else {
    // Broadcast scale copies use a contiguous 16-byte row. Retain the zero
    // stride encoding and accept the explicit eight-row (128B) stride emitted
    // by DeepGEMM. Other strides/swizzles still require layout qualification.
    if (mode != 0 || lbo != 0 || (sbo != 0 && sbo != 128))
      return false;
  }
  out.reserve(128 * words);
  for (unsigned row = 0; row < 128; ++row) {
    unsigned source_row = row;
    if (shape == tcgen05_copy_shape::warpx4)
      source_row = row % 32;
    if (shape == tcgen05_copy_shape::warpx2_02_13)
      source_row = row % 64;
    if (shape == tcgen05_copy_shape::warpx2_01_23)
      source_row = (row / 64) * 32 + row % 32;
    for (unsigned col = 0; col < words; ++col) {
      unsigned address;
      if (!wide)
        address = start + source_row * 16 + col * 4;
      else if (!swizzle)
        address = start + (row / 8) * sbo + (row % 8) * 16 + (col / 4) * lbo +
                  (col % 4) * 4;
      else {
        address = start + (row / 8) * sbo + (row % 8) * swizzle + col * 4;
        address ^= ((address >> 7) & (swizzle / 16 - 1)) << 4;
      }
      out.push_back(address);
    }
  }
  return true;
}
} // namespace flash_gpgpu_sim
