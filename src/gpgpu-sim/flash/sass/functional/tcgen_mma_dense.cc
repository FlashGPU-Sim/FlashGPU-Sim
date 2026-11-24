#include "tcgen_mma_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
step_result tf32_tmem_mma(const instruction &inst, warp_state &state,
                          execution_context &context, uint32_t idesc,
                          uint32_t a_address, uint32_t destination,
                          const operand &br) {
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  constexpr uint32_t unsupported_bits = 0xe080004fu;
  if (inst.opcode != "UTCHMMA" || (idesc & unsupported_bits) ||
      ((idesc >> 4) & 3) != 1 || ((idesc >> 7) & 7) != 2 ||
      ((idesc >> 10) & 7) != 2 || (idesc & (3u << 15)) ||
      (m != 64 && m != 128) || n < 8 || n > 256 || n % 8 || (a_address >> 16) ||
      (a_address & 7) || (destination >> 16))
    return unsupported(inst, "unqualified single-CTA TF32 TMEM-A descriptor");
  tcgen05_shared_descriptor_t bd;
  if (!shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context) ||
      bd.base_offset != 0)
    return unsupported(inst, "unqualified TF32 B shared descriptor");
  if (!context.cta->tcgen)
    return unsupported(inst, "TF32 MMA requires live TMEM storage");
  auto &tm = *context.cta->tcgen;
  auto physical_row = [m](unsigned row) {
    return m == 64 ? (row / 16) * 32 + row % 16 : row;
  };
  for (unsigned row = 0; row < m; ++row) {
    const unsigned offset = physical_row(row) << 16;
    if (!tm.contains(a_address + offset, 8) ||
        !tm.contains(destination + offset, n))
      return unsupported(inst, "TF32 MMA operand exceeds live TMEM allocation");
  }
  mma_shared_reads shared_reads;
  std::vector<uint32_t> b(n * 8), output(m * n);
  for (unsigned col = 0; col < n; ++col)
    for (unsigned k = 0; k < 8; ++k) {
      const auto address =
          tcgen05_shared_k_major_packed_byte_address(bd, col, k * 4, 32);
      shared_reads.add(address, 4);
      if (!context.memory->read(memory_space::kShared, address, &b[col * 8 + k],
                                4))
        return memory_fault(inst, memory_space::kShared, address, 4);
    }
  const auto &predicate = inst.operands[5];
  const bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  std::vector<uint32_t> a, input;
  for (unsigned row = 0; row < m; ++row) {
    const unsigned offset = physical_row(row) << 16;
    const auto values = tm.memory.read_words({0, 0}, a_address + offset, 8);
    a.insert(a.end(), values.begin(), values.end());
    if (accumulate) {
      const auto previous =
          tm.memory.read_words({0, 0}, destination + offset, n);
      input.insert(input.end(), previous.begin(), previous.end());
    }
  }
  output = tcgen05_mma_tf32_compute_words(
      m, n, a, b, input, accumulate, idesc & (1u << 13), idesc & (1u << 14));
  // All reads and checks precede writes, including when A aliases D.
  for (unsigned row = 0; row < m; ++row)
    tm.memory.write_words(
        {0, 0}, destination + (physical_row(row) << 16),
        std::vector<uint32_t>(output.begin() + row * n,
                              output.begin() + (row + 1) * n));
  context.tcgen_mma_effect = {true, false, m, n, 8, 1};
  context.tcgen_mma_effect.tmem_a = true;
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result half_mma(const instruction &inst, warp_state &state,
                     execution_context &context) {
  const bool ws = inst.opcode == "UTCHMMA.WS";
  const unsigned group = 1;
  if (inst.operands.size() != 6 || !context.cta || !context.memory)
    return unsupported(inst, "half MMA requires six operands and CTA memory");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (group == 2 &&
      (!context.cluster ||
       context.cluster->ctas.size() < (context.cga_cta_id & ~1u) + 2 ||
       context.cluster->memories.size() < (context.cga_cta_id & ~1u) + 2))
    return unsupported(inst, "paired half MMA requires two resident CTAs");
  operand ar, br, dr, metadata;
  operand b_operand = inst.operands[1];
  uint32_t destination = 0, a_address = 0;
  const bool tmem_a = tmem_address(inst.operands[0], state, a_address);
  if ((!tmem_a && !wrapped_uniform(inst.operands[0], "gdesc", ar)) ||
      !wrapped_uniform(b_operand, "gdesc", br) ||
      !tmem_address(inst.operands[2], state, destination) ||
      !wrapped_uniform(inst.operands[3], "tmem", metadata) ||
      !wrapped_uniform(inst.operands[4], "idesc", dr) ||
      (!tmem_a && ar.index >= kZeroUniformRegister - 1) ||
      br.index >= kZeroUniformRegister - 1 ||
      inst.operands[5].kind != operand_kind::kUniformPredicate)
    return unsupported(inst, "unqualified UTCHMMA operand layout");
  if (state.read_uniform_register(metadata.index) != 0)
    return unsupported(inst, "half MMA sparse/masked metadata is unqualified");
  const uint32_t idesc = state.read_uniform_register(dr.index);
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  const unsigned af = (idesc >> 7) & 7, bf = (idesc >> 10) & 7;
  if (tmem_a && (af == 2 || bf == 2)) {
    return tf32_tmem_mma(inst, state, context, idesc, a_address, destination,
                         br);
  }
  if (!tcgen05_f16_descriptor_supported(idesc, group, ws))
    return unsupported(inst,
                       "unqualified dense F16/BF16 instruction descriptor");
  const auto desc = tcgen05_decode_f16_mma_descriptor(idesc, group, ws);
  tcgen05_shared_descriptor_t ad, bd;
  if ((!tmem_a &&
       !shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                          context)) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context))
    return unsupported(inst, "unqualified half MMA shared descriptor");
  const unsigned local_m = m / group, local_n = n / group;
  const bool folded = (group == 2 && m == 128) || ws;
  if (tmem_a && ((!ws && group != 2) || n % 32 || desc.transpose_a ||
                 (a_address >> 16) || (a_address & 7)))
    return unsupported(inst, "unqualified paired TMEM-A layout");
  functional_tcgen_state *tmem[2] = {nullptr, nullptr};
  functional_memory *memories[2] = {nullptr, nullptr};
  for (unsigned rank = 0; rank < group; ++rank) {
    auto *cta = group == 2
                    ? context.cluster->ctas[(context.cga_cta_id & ~1u) + rank]
                    : context.cta;
    memories[rank] =
        group == 2
            ? context.cluster->memories[(context.cga_cta_id & ~1u) + rank]
            : context.memory;
    if (!cta || !cta->tcgen || !memories[rank])
      return unsupported(inst, "half MMA requires allocated peer storage");
    tmem[rank] = cta->tcgen.get();
    if (tmem_a)
      for (unsigned row = 0; row < 128; ++row)
        if (!tmem[rank]->contains(a_address + (row << 16), 8))
          return unsupported(inst, "half MMA TMEM-A exceeds allocation");
    for (unsigned row = 0; row < (folded ? 128u : local_m); ++row) {
      const unsigned physical_row =
          !folded && local_m == 64 ? (row / 16) * 32 + row % 16 : row;
      if (!tmem[rank]->contains(destination + (physical_row << 16),
                                folded ? n / 2 : n))
        return unsupported(inst, "half MMA destination exceeds allocation");
    }
  }
  mma_shared_reads shared_reads;
  std::vector<uint16_t> a(m * 16), b(n * 16);
  auto gather = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                    bool transpose, std::vector<uint16_t> &values) {
    shared_reads.begin_operand();
    return tcgen05_gather_shared(
        sd, rows, 16, group, transpose, values,
        [&](unsigned rank, uint32_t address, void *dst, unsigned bytes) {
          shared_reads.add(address, bytes, rank);
          return memories[rank]->read(memory_space::kShared, address, dst,
                                      bytes);
        });
  };
  if (!tmem_a && !gather(ad, m, desc.transpose_a, a))
    return unsupported(inst, "half MMA input is outside mapped shared memory");
  if (!gather(bd, n, desc.transpose_b, b))
    return unsupported(inst, "half MMA B is outside mapped shared memory");
  const auto &predicate = inst.operands[5];
  const bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  std::vector<uint32_t> input;
  if (accumulate)
    for (unsigned rank = 0; rank < group; ++rank) {
      const auto part =
          read_accumulator(*tmem[rank], destination, local_m, n, folded);
      input.insert(input.end(), part.begin(), part.end());
    }
  std::vector<uint32_t> output;
  if (!tmem_a) {
    output = tcgen05_mma_f16_compute_words(desc, a, b, input, accumulate);
  } else {
    std::vector<std::vector<uint16_t>> physical_a(group);
    for (unsigned rank = 0; rank < group; ++rank)
      physical_a[rank] =
          tmem[rank]->memory.read_matrix_packed_u16({0, 0}, a_address, 128, 16);
    output = tcgen05_mma_f16_tmem_a_compute_words(desc, physical_a, b, input,
                                                  accumulate, folded);
  }
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto begin = output.begin() + rank * local_m * n;
    write_accumulator(*tmem[rank], destination,
                      std::vector<uint32_t>(begin, begin + local_m * n),
                      local_m, n, folded);
  }
  context.tcgen_mma_effect = {true, false, desc.m, desc.n, desc.k, group};
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  // Default-discard WS: charge each shared operand once, not per FMA.
  context.tcgen_mma_effect.ws = ws;
  context.tcgen_mma_effect.tmem_a = tmem_a;
  context.tcgen_mma_effect.shared_operand_bytes =
      ws ? ((tmem_a ? 0 : desc.m) + desc.n) * desc.k * 2 : 0;
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
