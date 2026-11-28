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

inline bool read_output_lane_mask(const operand &mask, const warp_state &state,
                                  std::vector<uint32_t> &words) {
  if (mask.kind != operand_kind::kUniformRegister ||
      (mask.index != kZeroUniformRegister &&
       mask.index + words.size() > kZeroUniformRegister))
    return false;
  for (unsigned i = 0; i < words.size(); ++i)
    words[i] = mask.index == kZeroUniformRegister
                   ? 0
                   : state.read_uniform_register(mask.index + i);
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
                              unsigned rows, unsigned n, bool folded,
                              const uint32_t *disable_lanes = nullptr) {
  tm.memory.write_accumulator({0, 0}, base, values, rows, n, folded,
                              disable_lanes);
}

step_result tf32_mma(const instruction &, warp_state &, execution_context &,
                     uint32_t idesc, bool tmem_a, uint32_t a_address,
                     uint32_t destination, const operand &ar, const operand &br,
                     const uint32_t *disable_lanes);
step_result half_mma(const instruction &, warp_state &, execution_context &);
step_result mixed_mma(const instruction &, warp_state &, execution_context &);
step_result fp4_mma(const instruction &, warp_state &, execution_context &);
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
