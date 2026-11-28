#pragma once

#include "descriptor.h"
#include <cassert>
#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {

// Frontends supply memory access (and optional access accounting). Matrix
// layout, CTA partition assembly and packed-element decoding live here.
// read(rank, address, destination, bytes) returns false on an unmapped access.
template <typename T, typename Read>
bool tcgen05_gather_shared(const tcgen05_shared_descriptor_t &sd, unsigned rows,
                           unsigned k, unsigned group, bool transpose,
                           std::vector<T> &values, Read read) {
  assert(group && rows % group == 0);
  assert(!sd.leading_dimension_absolute);
  const unsigned local_rows = rows / group;
  values.resize(rows * k);
  for (unsigned rank = 0; rank < group; ++rank)
    for (unsigned row = 0; row < local_rows; ++row)
      for (unsigned col = 0; col < k; ++col) {
        const auto address = transpose
                                 ? tcgen05_shared_mn_major_byte_address(
                                       sd, row, col, sizeof(T), local_rows)
                                 : tcgen05_shared_k_major_packed_byte_address(
                                       sd, row, col * sizeof(T), k * sizeof(T));
        const unsigned global_row = rank * local_rows + row;
        if (!read(rank, address,
                  &values[transpose ? col * rows + global_row
                                    : global_row * k + col],
                  sizeof(T)))
          return false;
      }
  return true;
}

template <typename Read>
bool tcgen05_gather_mixed_partition(const tcgen05_shared_descriptor_t &sd,
                                    unsigned rows, unsigned k, uint8_t format,
                                    bool compact, unsigned rank,
                                    std::vector<uint8_t> &values, Read read) {
  assert(values.size() >= (rank + 1) * rows * k);
  const auto bits = tcgen05_mxf8f6f4_format_bits(format);
  const unsigned row_bytes = compact ? k * bits / 8 : k;
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned col = 0; col < k; ++col) {
      auto loc = tcgen05_mxf8f6f4_shared_location(col, format);
      if (compact)
        loc = {col * bits / 8, static_cast<uint8_t>((col * bits) % 8)};
      uint8_t lo = 0, hi = 0;
      auto address = tcgen05_shared_k_major_packed_byte_address(
          sd, row, loc.byte_offset, row_bytes);
      if (!read(rank, address, &lo, 1))
        return false;
      if (loc.bit_offset + bits > 8) {
        address = tcgen05_shared_k_major_packed_byte_address(
            sd, row, loc.byte_offset + 1, row_bytes);
        if (!read(rank, address, &hi, 1))
          return false;
      }
      values[(rank * rows + row) * k + col] =
          ((uint16_t(lo) | (uint16_t(hi) << 8)) >> loc.bit_offset) &
          ((1u << bits) - 1);
    }
  return true;
}

template <typename Read>
bool tcgen05_gather_fp4(const tcgen05_shared_descriptor_t &sd, unsigned rows,
                        unsigned k, std::vector<uint8_t> &values, Read read) {
  assert(k % 2 == 0);
  values.resize(rows * k);
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned col = 0; col < k; col += 2) {
      uint8_t pair = 0;
      const auto address =
          tcgen05_shared_k_major_packed_byte_address(sd, row, col / 2, k / 2);
      if (!read(0, address, &pair, 1))
        return false;
      values[row * k + col] = pair & 15;
      values[row * k + col + 1] = pair >> 4;
    }
  return true;
}
} // namespace flash_gpgpu_sim
