#pragma once
#include <cassert>

namespace flash_gpgpu_sim {
struct matrix_byte_location {
  unsigned address_lane;
  unsigned offset;
};
// PTX stmatrix.m16n8.trans.b8: each matrix uses eight independently
// addressed 16-byte columns. Four packed bytes span two columns/two halves.
inline matrix_byte_location stmatrix_m16n8_trans_byte(unsigned lane,
                                                      unsigned element,
                                                      unsigned matrix = 0) {
  assert(lane < 32 && element < 4 && matrix < 4);
  return {matrix * 8 + 2 * (lane % 4) + element % 2,
          lane / 4 + 8 * (element / 2)};
}
} // namespace flash_gpgpu_sim
