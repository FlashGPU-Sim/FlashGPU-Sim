#pragma once
#include "tensormap.h"
#include <algorithm>
#include <vector>

namespace flash_gpgpu_sim {
// Ordinary tiled U4 loads. Gather4 keeps its separately qualified U4/U6 path.
inline bool tma_packed4_supported(const tensormap_descriptor_t &tm) {
  const auto &f = tm.fields;
  const bool expanded = f.tensorDataType == TMA_DTYPE_16U4_ALIGN16B;
  if (!tm.is_valid() || !tm.is_packed_u4() ||
      f.interleave != TMA_INTERLEAVE_NONE || f.oobFill != TMA_OOB_ZERO ||
      f.swizzle > 3 || f.globalAddress % (expanded ? 32 : 16) ||
      f.globalDim[0] % (expanded ? 128 : 2) || f.boxDim[0] % 32 ||
      (expanded && (f.boxDim[0] != 128 || (f.swizzle && f.swizzle != 3))))
    return false;
  const unsigned row_bytes = expanded ? f.boxDim[0] : f.boxDim[0] / 2;
  if (f.swizzle && row_bytes > (16u << f.swizzle))
    return false;
  for (unsigned d = 0; d < tm.num_dims(); ++d) {
    if (!f.globalDim[d] || f.globalDim[d] > 0x80000000u || !f.boxDim[d] ||
        f.boxDim[d] > 256 || f.elementStrides[d] != 1)
      return false;
    if (d && (!f.globalStrides[d - 1] ||
              f.globalStrides[d - 1] % (expanded ? 32 : 16) ||
              f.globalStrides[d - 1] >= (1ull << 40)))
      return false;
  }
  return tm.get_tile_size_bytes() && tm.get_tile_smem_size_bytes();
}

// Produces the unswizzled shared image, staging all reads before the caller
// changes shared memory or completion state. Padding is deterministically zero
// in the model; only the packed payload contributes to completion/HBM bytes.
template <class Read>
bool tma_packed4_load(const tensormap_descriptor_t &tm, const int32_t coords[5],
                      Read read, std::vector<uint8_t> &tile) {
  if (!tma_packed4_supported(tm) || coords[0] % 16)
    return false;
  const auto &f = tm.fields;
  const bool expanded = f.tensorDataType == TMA_DTYPE_16U4_ALIGN16B;
  const unsigned row_bytes = expanded ? f.boxDim[0] : f.boxDim[0] / 2;
  std::vector<uint8_t> staged(tm.get_tile_smem_size_bytes(), 0);
  for (uint64_t row = 0; row < staged.size() / row_bytes; ++row) {
    uint64_t rem = row, base = f.globalAddress;
    bool in_bounds = true;
    for (unsigned d = 1; d < tm.num_dims(); ++d) {
      const int64_t c = int64_t(coords[d]) + int64_t(rem % f.boxDim[d]);
      rem /= f.boxDim[d];
      if (c < 0 || uint64_t(c) >= f.globalDim[d]) {
        in_bounds = false;
        break;
      }
      const uint64_t stride = f.globalStrides[d - 1];
      if (uint64_t(c) > (UINT64_MAX - base) / stride)
        return false;
      base += uint64_t(c) * stride;
    }
    if (!in_bounds)
      continue;
    for (unsigned col = 0; col < f.boxDim[0]; col += 16) {
      const int64_t x = int64_t(coords[0]) + col;
      const int64_t lo = std::max<int64_t>(x, 0);
      const int64_t hi = std::min<int64_t>(x + 16, f.globalDim[0]);
      if (lo >= hi)
        continue;
      const uint64_t offset = uint64_t(lo) / 2;
      const unsigned bytes = (hi - lo) / 2;
      if (offset > UINT64_MAX - base || bytes - 1 > UINT64_MAX - base - offset)
        return false;
      const uint64_t destination =
          row * row_bytes + (expanded ? col : col / 2) + (lo - x) / 2;
      if (!read(base + offset, staged.data() + destination, bytes))
        return false;
    }
  }
  tile.swap(staged);
  return true;
}
} // namespace flash_gpgpu_sim
