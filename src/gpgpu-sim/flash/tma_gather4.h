#pragma once
#include "tensormap.h"
#include <array>
#include <cassert>
namespace flash_gpgpu_sim {
// PTX 9.5: cp.async.bulk.tensor.2d.tile::gather4 uses {x,y0,y1,y2,y3}.
// Four independent bounding boxes, in operand order; repeated rows are legal.
inline std::array<int32_t, 5> tma_gather4_row(const int32_t coords[5],
                                              unsigned slot) {
  assert(slot < 4);
  return {{coords[0], coords[slot + 1], 0, 0, 0}};
}
inline bool tma_gather4_supported(const tensormap_descriptor_t &tm) {
  return tm.is_valid() && tm.num_dims() == 2 && tm.fields.boxDim[0] > 0 &&
         tm.fields.boxDim[1] == 1 && tm.fields.elementStrides[0] == 1 &&
         tm.fields.elementStrides[1] == 1 &&
         tm.fields.interleave == TMA_INTERLEAVE_NONE &&
         tm.fields.tensorDataType <= TMA_DTYPE_BF16 &&
         tm.fields.oobFill == TMA_OOB_ZERO &&
         tm.fields.swizzle == TMA_SWIZZLE_NONE &&
         uint64_t(tm.fields.boxDim[0]) * tm.get_element_size() <=
             UINT32_MAX / 4;
}
} // namespace flash_gpgpu_sim
