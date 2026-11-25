#pragma once
#include "../../tcgen05/mma.h"
#include "../../tcgen05/operands.h"
#include "tmem_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
inline bool shared_descriptor(uint64_t raw, tcgen05_shared_descriptor_t &desc,
                              const execution_context &,
                              bool mn_major = false) {
  if (!tcgen05_mma_shared_descriptor_supported(raw, mn_major))
    return false;
  desc = tcgen05_decode_mma_shared_descriptor(raw, mn_major);
  return true;
}

using mma_shared_reads = tcgen05_shared_reads;

inline std::vector<uint32_t> read_accumulator(functional_tcgen_state &tm,
                                              uint32_t base, unsigned rows,
                                              unsigned n, bool folded) {
  return tm.memory.read_accumulator({0, 0}, base, rows, n, folded);
}
inline void write_accumulator(functional_tcgen_state &tm, uint32_t base,
                              const std::vector<uint32_t> &values,
                              unsigned rows, unsigned n, bool folded) {
  tm.memory.write_accumulator({0, 0}, base, values, rows, n, folded);
}

step_result half_mma(const instruction &, warp_state &, execution_context &);
step_result mixed_mma(const instruction &, warp_state &, execution_context &);
step_result fp4_mma(const instruction &, warp_state &, execution_context &);
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
