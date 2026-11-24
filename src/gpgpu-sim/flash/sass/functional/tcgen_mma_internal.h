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

// Record physical shared addresses while the functional gather decodes the
// descriptor. Coalesce adjacent bytes; packed FP4/FP6 overlapping scalar
// reads must not double-charge the bytes shared by neighboring elements.
struct mma_shared_reads {
  std::vector<functional_tcgen_shared_read> reads;
  bool new_operand = true;
  void begin_operand() { new_operand = true; }
  void add(uint32_t address, unsigned bytes, unsigned rank = 0) {
    if (!new_operand && !reads.empty()) {
      auto &last = reads.back();
      if (last.rank == rank && address >= last.address &&
          uint64_t(address) <= uint64_t(last.address) + last.bytes) {
        last.bytes = std::max(last.bytes, address - last.address + bytes);
        return;
      }
    }
    new_operand = false;
    reads.push_back({address, bytes, rank});
  }
};

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
