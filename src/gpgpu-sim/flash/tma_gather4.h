#pragma once
#include "tensormap.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>
namespace flash_gpgpu_sim {
inline std::array<int32_t, 5> tma_gather4_row(const int32_t coords[5],
                                              unsigned slot) {
  assert(slot < 4);
  return {{coords[0], coords[slot + 1], 0, 0, 0}};
}
inline bool tma_gather4_packed(const tensormap_descriptor_t &tm) {
  return tm.fields.tensorDataType >= TMA_DTYPE_16U4_ALIGN8B &&
         tm.fields.tensorDataType <= TMA_DTYPE_16U6_ALIGN16B;
}
inline unsigned tma_gather4_gmem_row_bytes(const tensormap_descriptor_t &tm) {
  const unsigned bits = tm.fields.tensorDataType == TMA_DTYPE_16U6_ALIGN16B ? 6
                        : tma_gather4_packed(tm)                            ? 4
                                                 : 8 * tm.get_element_size();
  return (tm.fields.boxDim[0] * bits + 7) / 8;
}
inline unsigned tma_gather4_smem_row_bytes(const tensormap_descriptor_t &tm) {
  return tm.fields.tensorDataType >= TMA_DTYPE_16U4_ALIGN16B
             ? tm.fields.boxDim[0]
             : tma_gather4_gmem_row_bytes(tm);
}
inline bool tma_gather4_supported(const tensormap_descriptor_t &tm) {
  const auto &f = tm.fields;
  const unsigned bytes = tma_gather4_gmem_row_bytes(tm);
  const bool floating = f.tensorDataType >= TMA_DTYPE_F16 &&
                        f.tensorDataType <= TMA_DTYPE_TF32_FTZ;
  const bool expanded = f.tensorDataType == TMA_DTYPE_16U4_ALIGN16B ||
                        f.tensorDataType == TMA_DTYPE_16U6_ALIGN16B;
  const unsigned span = f.swizzle == 1 ? 32 : f.swizzle == 2 ? 64 : 128;
  return tm.is_valid() && tm.num_dims() == 2 && f.boxDim[0] > 0 &&
         f.boxDim[0] <= 256 && f.boxDim[1] == 1 && f.elementStrides[0] == 1 &&
         f.elementStrides[1] >= 1 && f.elementStrides[1] <= 8 &&
         f.interleave == TMA_INTERLEAVE_NONE &&
         f.tensorDataType <= TMA_DTYPE_16U6_ALIGN16B &&
         (f.oobFill == TMA_OOB_ZERO ||
          (floating && f.oobFill == TMA_OOB_NAN)) &&
         f.swizzle <= 6 &&
         (!f.swizzle || tma_gather4_smem_row_bytes(tm) <= span) &&
         bytes % 16 == 0 &&
         (!tma_gather4_packed(tm) || f.globalDim[0] % 2 == 0) &&
         (!expanded ||
          (f.boxDim[0] == 128 && f.globalDim[0] % 128 == 0 &&
           f.globalAddress % 32 == 0 && f.globalStrides[0] % 32 == 0 &&
           (f.swizzle == 0 || f.swizzle == 3 || f.swizzle == 4)));
}
// The phase is determined by the shared address, not the source row index.
// Gather slots form one contiguous logical destination before swizzling.
inline uint64_t tma_gather4_shared_offset(uint64_t offset, uint32_t base,
                                          unsigned swizzle) {
  if (!swizzle)
    return offset;
  const uint64_t row = (base + offset) >> 7;
  if (swizzle == 4 || swizzle == 5)
    return offset ^ ((row & 3) << 5) ^ (swizzle == 5 ? ((row & 1) << 3) : 0);
  if (swizzle == 6)
    return offset ^ ((row & 1) << 6);
  return offset ^ ((row & ((1u << swizzle) - 1)) << 4);
}
inline uint32_t tma_gather4_convert(uint32_t bits, unsigned type) {
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
// Each source destination contributes one completion. Paired multicast may
// credit the same owner twice; the upper half retains that second credit.
inline bool tma_gather4_completion_targets(unsigned cluster_size,
                                           uint16_t destinations,
                                           unsigned barrier_rank, bool paired,
                                           bool multicast,
                                           uint32_t &completions) {
  completions = 0;
  if (!cluster_size || cluster_size > 16 || barrier_rank >= cluster_size ||
      (uint32_t(destinations) >> cluster_size))
    return false;
  for (unsigned rank = 0; rank < cluster_size; ++rank)
    if (destinations & (1u << rank)) {
      const unsigned owner =
          paired
              ? (multicast ? (rank & ~1u) | (barrier_rank & 1u) : barrier_rank)
              : rank;
      if (owner >= cluster_size || (paired && owner / 2 != rank / 2) ||
          (!paired && !multicast && owner != barrier_rank))
        return false;
      const uint32_t bit = 1u << owner;
      completions |= completions & bit ? bit << 16 : bit;
    }
  return true;
}

// Architectural payload staging is shared by both frontends. No destination
// or barrier changes occur until every source read succeeds. Cache policy is
// a timing hint; the existing TMA AGU/queue remains the timing implementation.
template <class Read>
bool tma_gather4_load(const tensormap_descriptor_t &tm, const int32_t coords[5],
                      Read read, std::vector<uint8_t> &tile) {
  if (!tma_gather4_supported(tm))
    return false;
  const unsigned bytes = tm.get_element_size(), cols = tm.fields.boxDim[0];
  const unsigned row_bytes = tma_gather4_smem_row_bytes(tm);
  tile.assign(4 * row_bytes, 0);
  if (tma_gather4_packed(tm)) {
    const unsigned bits =
        tm.fields.tensorDataType == TMA_DTYPE_16U6_ALIGN16B ? 6 : 4;
    const bool expanded = tm.fields.tensorDataType != TMA_DTYPE_16U4_ALIGN8B;
    for (unsigned row = 0; row < 4; ++row)
      for (unsigned col = 0; col < cols; ++col) {
        const int64_t x = int64_t(coords[0]) + col, y = coords[row + 1];
        if (x < 0 || y < 0 || uint64_t(x) >= tm.fields.globalDim[0] ||
            uint64_t(y) >= tm.fields.globalDim[1])
          continue;
        const uint64_t row_offset = uint64_t(y) * tm.fields.globalStrides[0];
        if (uint64_t(y) &&
            row_offset / uint64_t(y) != tm.fields.globalStrides[0])
          return false;
        const uint64_t offset = row_offset + uint64_t(x) * bits / 8;
        if (offset < row_offset ||
            tm.fields.globalAddress > UINT64_MAX - offset)
          return false;
        uint16_t word = 0;
        const unsigned shift = (uint64_t(x) * bits) % 8;
        const unsigned read_bytes = (shift + bits + 7) / 8;
        if (read_bytes - 1 > UINT64_MAX - tm.fields.globalAddress - offset ||
            !read(tm.fields.globalAddress + offset, &word, read_bytes))
          return false;
        const unsigned value = (word >> shift) & ((1u << bits) - 1);
        const unsigned dstbit =
            (expanded ? (col / 16) * 128 + (col % 16) * bits : col * bits) +
            row * row_bytes * 8;
        for (unsigned bit = 0; bit < bits; ++bit)
          tile[(dstbit + bit) / 8] |= ((value >> bit) & 1u)
                                      << ((dstbit + bit) % 8);
      }
    return true;
  }
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned col = 0; col < cols; ++col) {
      const int64_t x = int64_t(coords[0]) + col, y = coords[row + 1];
      auto *out = tile.data() + (row * cols + col) * bytes;
      if (x < 0 || y < 0 || uint64_t(x) >= tm.fields.globalDim[0] ||
          uint64_t(y) >= tm.fields.globalDim[1]) {
        if (tm.fields.oobFill == TMA_OOB_NAN) {
          std::fill_n(out, bytes, 0xff);
          out[bytes - 1] = 0x7f;
          if (tm.fields.tensorDataType == TMA_DTYPE_TF32 ||
              tm.fields.tensorDataType == TMA_DTYPE_TF32_FTZ) {
            out[0] = 0;
            out[1] = 0xe0;
          }
        }
        continue;
      }
      const auto stride = tm.fields.globalStrides[0];
      if (uint64_t(y) && stride > UINT64_MAX / uint64_t(y))
        return false;
      const auto offset = uint64_t(y) * stride;
      if (offset > UINT64_MAX - uint64_t(x) * bytes ||
          tm.fields.globalAddress > UINT64_MAX - offset - uint64_t(x) * bytes)
        return false;
      const uint64_t address =
          tm.fields.globalAddress + offset + uint64_t(x) * bytes;
      if (bytes - 1 > UINT64_MAX - address || !read(address, out, bytes))
        return false;
      if (bytes == 4) {
        uint32_t bits;
        std::memcpy(&bits, out, 4);
        bits = tma_gather4_convert(bits, tm.fields.tensorDataType);
        std::memcpy(out, &bits, 4);
      }
    }
  return true;
}
} // namespace flash_gpgpu_sim
