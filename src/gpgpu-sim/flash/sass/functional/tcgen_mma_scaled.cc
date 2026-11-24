#include "tcgen_mma_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
step_result dense_fp8_mma(const instruction &inst, warp_state &state,
                          execution_context &context) {
  if (inst.opcode != "UTCQMMA" || inst.operands.size() != 6 || !context.cta ||
      !context.memory)
    return unsupported(inst, "unqualified dense FP8 MMA CTA/operand form");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  operand ar, br, dr, metadata;
  uint32_t destination = 0;
  if (!wrapped_uniform(inst.operands[0], "gdesc", ar) ||
      !wrapped_uniform(inst.operands[1], "gdesc", br) ||
      !tmem_address(inst.operands[2], state, destination) ||
      !wrapped_uniform(inst.operands[3], "tmem", metadata) ||
      !wrapped_uniform(inst.operands[4], "idesc", dr) ||
      ar.index >= kZeroUniformRegister - 1 ||
      br.index >= kZeroUniformRegister - 1 ||
      inst.operands[5].kind != operand_kind::kUniformPredicate ||
      state.read_uniform_register(metadata.index) != 0 || (destination >> 16))
    return unsupported(inst, "unqualified dense FP8 MMA operands");
  const uint32_t idesc = state.read_uniform_register(dr.index);
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  const unsigned af = (idesc >> 7) & 7, bf = (idesc >> 10) & 7;
  if ((idesc & 0xe081804fu) || ((idesc >> 4) & 3) != 1 || af > 1 || bf > 1 ||
      m != 128 || n < 16 || n > 256 || n % 16)
    return unsupported(inst, "unqualified dense K-major FP8 descriptor");
  tcgen05_shared_descriptor_t ad, bd;
  if (!shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                         context) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context) ||
      ad.base_offset || bd.base_offset || !context.cta->tcgen)
    return unsupported(
        inst, "dense FP8 requires qualified shared operands and live TMEM");
  auto &tm = *context.cta->tcgen;
  for (unsigned row = 0; row < m; ++row)
    if (!tm.contains(destination + (row << 16), n))
      return unsupported(inst, "dense FP8 MMA exceeds destination allocation");
  mma_shared_reads shared_reads;
  std::vector<uint8_t> a(m * 32), b(n * 32);
  auto gather = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                    std::vector<uint8_t> &values) {
    shared_reads.begin_operand();
    return tcgen05_gather_shared(
        sd, rows, 32, 1, false, values,
        [&](unsigned rank, uint32_t address, void *dst, unsigned bytes) {
          shared_reads.add(address, bytes, rank);
          return context.memory->read(memory_space::kShared, address, dst,
                                      bytes);
        });
  };
  if (!gather(ad, m, a) || !gather(bd, n, b))
    return unsupported(inst,
                       "dense FP8 operand is outside mapped shared memory");
  const auto desc = tcgen05_decode_fp8_mma_descriptor(idesc);
  const auto &predicate = inst.operands[5];
  const bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  const auto input =
      accumulate ? tm.memory.read_matrix_words({0, 0}, destination, m, n)
                 : std::vector<uint32_t>{};
  // Reuse only the arithmetic core with multiplicative identity scales.
  // There are no scale-memory reads or block-scale descriptor decoding here.
  const auto output = tcgen05_mma_mxf8f6f4_compute_words(
      desc, a, b, std::vector<uint8_t>(m, 127), std::vector<uint8_t>(n, 127),
      input, accumulate);
  tm.memory.write_matrix_words({0, 0}, destination, output, m, n);
  context.tcgen_mma_effect = {true, true, m, n, 32, 1};
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result fp4_mma(const instruction &inst, warp_state &state,
                    execution_context &context) {
  if (inst.opcode != "UTCOMMA" || inst.operands.size() != 7 || !context.cta ||
      !context.memory)
    return unsupported(inst, "MXFP4 MMA requires CTA1 and seven operands");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  operand ar, br, dr, sr, metadata;
  uint32_t destination = 0;
  if (!wrapped_uniform(inst.operands[0], "gdesc", ar) ||
      !wrapped_uniform(inst.operands[1], "gdesc", br) ||
      !tmem_address(inst.operands[2], state, destination) ||
      !wrapped_uniform(inst.operands[3], "tmem", metadata) ||
      !wrapped_uniform(inst.operands[4], "idesc", dr) ||
      !wrapped_uniform(inst.operands[5], "tmem", sr) ||
      ar.index >= kZeroUniformRegister - 1 ||
      br.index >= kZeroUniformRegister - 1 ||
      sr.index >= kZeroUniformRegister - 1 ||
      inst.operands[6].kind != operand_kind::kUniformPredicate)
    return unsupported(inst, "unqualified UTCOMMA operands");
  uint32_t idesc = state.read_uniform_register(dr.index);
  const bool sm107 = false;
  const unsigned mma_k = (idesc & 8u) ? 128 : 64;
  unsigned m = ((idesc >> 27) & 3) * 128, n = ((idesc >> 17) & 63) * 8;
  unsigned aid = (idesc >> 29) & 3, bid = (idesc >> 4) & 3;
  constexpr uint32_t rejected_bits =
      0x80000000u | (7u << 24) | (3u << 15) | (1u << 6) | 7u;
  if (state.read_uniform_register(metadata.index) != 0 ||
      (idesc & rejected_bits) || !(idesc & (1u << 23)) ||
      ((idesc >> 12) & 1) != unsigned(sm107) || (mma_k == 128 && !sm107) ||
      ((idesc >> 7) & 7) != TCGEN05_MXF4_FORMAT_E2M1 ||
      ((idesc >> 10) & 3) != TCGEN05_MXF4_FORMAT_E2M1 || (aid & 1) ||
      (bid & 1) || (mma_k == 128 && (aid || bid)) ||
      !tcgen05_mxf4_dense_shape_supported(m, n, mma_k, 1))
    return unsupported(inst, "unqualified dense block32 MXFP4 descriptor");
  const auto desc = tcgen05_decode_mxf4_mma_descriptor(idesc, 1, sm107);
  tcgen05_shared_descriptor_t ad, bd;
  if (!shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                         context) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context))
    return unsupported(inst, "unqualified MXFP4 shared descriptors");
  uint32_t sa = state.read_uniform_register(sr.index),
           sb = state.read_uniform_register(sr.index + 1);
  if (!context.cta->tcgen || destination >> 16 || sa >> 16 || sb >> 16)
    return unsupported(inst, "MXFP4 requires allocated row-zero TMEM bases");
  auto &tm = *context.cta->tcgen;
  for (unsigned row = 0; row < m; ++row)
    if (!tm.contains(destination + (row << 16), n))
      return unsupported(inst, "MXFP4 accumulator exceeds allocation");
  for (unsigned row = 0; row < 32; ++row)
    if (!tm.contains(sa + (row << 16), (m + 31) / 32) ||
        !tm.contains(sb + (row << 16), (n + 31) / 32))
      return unsupported(inst, "MXFP4 scales exceed allocation");
  mma_shared_reads shared_reads;
  std::vector<uint8_t> a(m * mma_k), b(n * mma_k);
  auto read = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                  std::vector<uint8_t> &out) {
    shared_reads.begin_operand();
    return tcgen05_gather_fp4(
        sd, rows, mma_k, out,
        [&](unsigned rank, uint32_t address, void *dst, unsigned bytes) {
          shared_reads.add(address, bytes, rank);
          return context.memory->read(memory_space::kShared, address, dst,
                                      bytes);
        });
  };
  if (!read(ad, m, a) || !read(bd, n, b))
    return unsupported(inst, "MXFP4 inputs outside mapped shared memory");
  auto scales_a =
      tm.memory.read_mxf4_scale_matrix({0, 0}, sa, m, mma_k / 32, aid);
  auto scales_b =
      tm.memory.read_mxf4_scale_matrix({0, 0}, sb, n, mma_k / 32, bid);
  const auto &predicate = inst.operands[6];
  bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  std::vector<uint32_t> input;
  if (accumulate)
    input = tm.memory.read_matrix_words({0, 0}, destination, m, n);
  auto out = tcgen05_mma_mxf4_compute_words(desc, a, b, scales_a, scales_b, 32,
                                            input, accumulate);
  tm.memory.write_matrix_words({0, 0}, destination, out, m, n);
  context.tcgen_mma_effect = {true, true, m, n, mma_k, 1, true};
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result mixed_mma(const instruction &inst, warp_state &state,
                      execution_context &context) {
  if (inst.operands.size() == 6)
    return dense_fp8_mma(inst, state, context);
  const unsigned group = inst.opcode == "UTCQMMA.2CTA" ? 2 : 1;
  if (inst.operands.size() != 7 || !context.cta || !context.memory)
    return unsupported(inst,
                       "mixed MMA requires seven operands and CTA memory");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (group == 2 &&
      (!context.cluster ||
       context.cluster->ctas.size() < (context.cga_cta_id & ~1u) + 2 ||
       context.cluster->memories.size() < (context.cga_cta_id & ~1u) + 2))
    return unsupported(inst, "paired mixed MMA requires two resident CTAs");
  operand ar, br, dr, sr, metadata;
  operand b_operand = inst.operands[1];
  uint32_t destination = 0;
  if (!wrapped_uniform(inst.operands[0], "gdesc", ar) ||
      !wrapped_uniform(b_operand, "gdesc", br) ||
      !tmem_address(inst.operands[2], state, destination) ||
      !wrapped_uniform(inst.operands[3], "tmem", metadata) ||
      !wrapped_uniform(inst.operands[4], "idesc", dr) ||
      !wrapped_uniform(inst.operands[5], "tmem", sr) ||
      ar.index >= kZeroUniformRegister - 1 ||
      br.index >= kZeroUniformRegister - 1 ||
      sr.index >= kZeroUniformRegister - 1 ||
      inst.operands[6].kind != operand_kind::kUniformPredicate)
    return unsupported(inst, "unqualified UTCQMMA operand layout");
  if (state.read_uniform_register(metadata.index) != 0)
    return unsupported(inst, "sparse UTCQMMA metadata is not supported");
  const uint32_t idesc = state.read_uniform_register(dr.index);
  const unsigned m = ((idesc >> 27) & 3) * 128;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  const unsigned af = (idesc >> 7) & 7, bf = (idesc >> 10) & 7;
  const unsigned mma_k = (idesc & 0x80000000u) ? 64 : 32;
  if (mma_k != 32)
    return unsupported(inst, "MXF8F6F4 K64 requires SM107");
  // Reject sparse/reserved/transpose bits before calling the ISA-neutral
  // decoder, whose internal assertions are not a malformed-input boundary.
  constexpr uint32_t unsupported_descriptor_bits =
      (7u << 24) | (3u << 15) | (1u << 6) | 0xfu;
  if ((idesc & unsupported_descriptor_bits) || !(idesc & (1u << 23)) ||
      !tcgen05_mxf8f6f4_format_supported(af) ||
      !tcgen05_mxf8f6f4_format_supported(bf) ||
      !tcgen05_mxf8f6f4_dense_shape_supported(m, n, mma_k, group) ||
      (mma_k == 64 && (idesc & ((1u << 29) | (1u << 4)))))
    return unsupported(inst, "unqualified dense K-major MXF8F6F4 descriptor");
  const auto desc = tcgen05_decode_mxf8f6f4_mma_descriptor(idesc, group);
  tcgen05_shared_descriptor_t ad, bd;
  if (!shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                         context) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context))
    return unsupported(inst, "unqualified mixed MMA shared descriptor");
  // CuTe carries the scale byte selector in address bits 31:30;
  // the instruction descriptor already contains the corresponding SFA/SFB ID.
  const uint32_t scale_address_mask = 0xffffffffu;
  const uint32_t sfa =
      state.read_uniform_register(sr.index) & scale_address_mask;
  // Native SASS prints only the first register of this address pair.
  const uint32_t sfb =
      state.read_uniform_register(sr.index + 1) & scale_address_mask;
  functional_tcgen_state *tmem[2] = {nullptr, nullptr};
  functional_memory *memories[2] = {nullptr, nullptr};
  const unsigned local_m = m / group, local_n = n / group;
  const bool folded = group == 2 && m == 128;
  if (destination >> 16 || sfa >> 16 || sfb >> 16)
    return unsupported(inst, "unqualified mixed MMA TMEM row base");
  for (unsigned rank = 0; rank < group; ++rank) {
    auto *cta = group == 2
                    ? context.cluster->ctas[(context.cga_cta_id & ~1u) + rank]
                    : context.cta;
    memories[rank] =
        group == 2
            ? context.cluster->memories[(context.cga_cta_id & ~1u) + rank]
            : context.memory;
    if (!cta || !cta->tcgen || !memories[rank])
      return unsupported(inst, "mixed MMA requires allocated peer storage");
    tmem[rank] = cta->tcgen.get();
    for (unsigned r = 0; r < (folded ? 128u : local_m); ++r)
      if (!tmem[rank]->contains(destination + (r << 16), folded ? n / 2 : n))
        return unsupported(inst, "mixed MMA destination exceeds allocation");
    for (unsigned r = 0; r < 32; ++r)
      if (!tmem[rank]->contains(sfa + (r << 16), (local_m + 31) / 32) ||
          !tmem[rank]->contains(sfb + (r << 16), (n + 31) / 32))
        return unsupported(inst, "mixed MMA scales exceed allocation");
  }
  mma_shared_reads shared_reads;
  std::vector<uint8_t> a(m * mma_k), b(n * mma_k);
  auto read_input = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                        uint8_t format, std::vector<uint8_t> &values,
                        unsigned rank) {
    shared_reads.begin_operand();
    return tcgen05_gather_mixed_partition(
        sd, rows, mma_k, format, mma_k == 64, rank, values,
        [&](unsigned owner, uint32_t address, void *dst, unsigned bytes) {
          shared_reads.add(address, bytes, owner);
          return memories[owner]->read(memory_space::kShared, address, dst,
                                       bytes);
        });
  };
  for (unsigned rank = 0; rank < group; ++rank)
    if (!read_input(ad, local_m, af, a, rank))
      return unsupported(inst, "mixed MMA A outside mapped shared memory");
  for (unsigned rank = 0; rank < group; ++rank)
    if (!read_input(bd, local_n, bf, b, rank))
      return unsupported(inst, "mixed MMA B outside mapped shared memory");
  std::vector<uint8_t> sa;
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto part = tmem[rank]->memory.read_mxf4_scale_matrix(
        {0, 0}, sfa, local_m, mma_k / 32, desc.a_scale_factor_id);
    sa.insert(sa.end(), part.begin(), part.end());
  }
  // The existing qualified 2x1SM layout replicates the complete SFB image;
  // it must not concatenate the lower N/2 scales from both replicas.
  const auto sb = tmem[0]->memory.read_mxf4_scale_matrix(
      {0, 0}, sfb, n, mma_k / 32, desc.b_scale_factor_id);

  const auto &predicate = inst.operands[6];
  const bool accumulate =
      state.read_uniform_predicate(predicate.index) != predicate.negated;
  std::vector<uint32_t> input;
  if (accumulate)
    for (unsigned rank = 0; rank < group; ++rank) {
      const auto part =
          read_accumulator(*tmem[rank], destination, local_m, n, folded);
      input.insert(input.end(), part.begin(), part.end());
    }
  const auto output =
      tcgen05_mma_mxf8f6f4_compute_words(desc, a, b, sa, sb, input, accumulate);
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto begin = output.begin() + rank * local_m * n;
    write_accumulator(*tmem[rank], destination,
                      std::vector<uint32_t>(begin, begin + local_m * n),
                      local_m, n, folded);
  }
  context.tcgen_mma_effect = {true, true, desc.m, desc.n, desc.k, group};
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
