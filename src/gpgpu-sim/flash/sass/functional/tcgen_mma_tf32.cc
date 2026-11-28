#include "tcgen_mma_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
step_result tf32_mma(const instruction &inst, warp_state &state,
                     execution_context &context, uint32_t idesc, bool tmem_a,
                     uint32_t a_address, uint32_t destination,
                     const operand &ar, const operand &br,
                     const uint32_t *disable_lanes) {
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  if (inst.opcode != "UTCHMMA" || !tcgen05_tf32_descriptor_supported(idesc) ||
      (a_address >> 16) || (a_address & 7) || (destination >> 16))
    return unsupported(inst, "unqualified single-CTA TF32 descriptor");
  tcgen05_shared_descriptor_t ad, bd;
  if ((!tmem_a &&
       !shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                          context)) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context))
    return unsupported(inst, "unqualified TF32 shared descriptor");
  if (!context.cta->tcgen)
    return unsupported(inst, "TF32 MMA requires live TMEM storage");
  auto &tm = *context.cta->tcgen;
  for (unsigned row = 0; row < m; ++row) {
    const unsigned offset = (m == 64 ? (row / 16) * 32 + row % 16 : row) << 16;
    if ((tmem_a && !tm.contains(a_address + offset, 8)) ||
        !tm.contains(destination + offset, n))
      return unsupported(inst, "TF32 MMA operand exceeds live TMEM allocation");
  }
  mma_shared_reads shared_reads;
  auto gather = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                    std::vector<uint32_t> &values) {
    shared_reads.begin_operand();
    return tcgen05_gather_shared(
        sd, rows, 8, 1, false, values,
        [&](unsigned, uint32_t address, void *dst, unsigned bytes) {
          shared_reads.add(address, bytes);
          return context.memory->read(memory_space::kShared, address, dst,
                                      bytes);
        });
  };
  std::vector<uint32_t> a, b;
  if ((!tmem_a && !gather(ad, m, a)) || !gather(bd, n, b))
    return unsupported(inst, "TF32 input is outside mapped shared memory");
  if (tmem_a)
    a = read_accumulator(tm, a_address, m, 8, false);
  const auto &predicate = inst.operands.back();
  const bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  const auto input = accumulate ? read_accumulator(tm, destination, m, n, false)
                                : std::vector<uint32_t>{};
  const auto output = tcgen05_mma_tf32_compute_words(
      m, n, a, b, input, accumulate, idesc & (1u << 13), idesc & (1u << 14));
  // Gather all inputs before writing, including when TMEM A aliases D.
  write_accumulator(tm, destination, output, m, n, false, disable_lanes);
  context.tcgen_mma_effect = {true, false, m, n, 8, 1};
  tcgen05_mma_descriptor_t timing_desc{};
  timing_desc.m = m;
  timing_desc.n = n;
  timing_desc.k = 8;
  timing_desc.d_type = 1;
  context.tcgen_mma_effect.timing =
      tcgen05_metadata(tcgen05_mma_kind::tf32, timing_desc, 1, tmem_a);
  context.tcgen_mma_effect.tmem_a = tmem_a;
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
