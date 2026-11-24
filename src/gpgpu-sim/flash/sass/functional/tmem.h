#pragma once
#include "../../tcgen05/tmem.h"
namespace flash_gpgpu_sim {
namespace sass {
struct functional_tcgen_state {
  tcgen05_tmem_manager_t memory;
  uint32_t allocated = 0;
  functional_tcgen_state() { memory.init_native_backing({0, 0}, 512); }
  bool contains(uint32_t address, unsigned words) const {
    const auto a = tcgen05_decode_tmem_address(address);
    if (a.lane >= 128 || !words || a.column >= 512 || words > 512 - a.column)
      return false;
    for (unsigned col = a.column; col < a.column + words; ++col)
      if (!(allocated & (1u << (col / 32))))
        return false;
    return true;
  }
};
} // namespace sass
} // namespace flash_gpgpu_sim
