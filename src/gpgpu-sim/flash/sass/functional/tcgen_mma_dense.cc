#include "../../tcgen05/ws_mask.h"
#include "../decode/tcgen_collector.h"
#include "tcgen_mma_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
step_result half_mma(const instruction &inst, warp_state &state,
                     execution_context &context) {
  const bool ws = inst.opcode == "UTCHMMA.WS";
  const unsigned group = inst.opcode == "UTCHMMA.2CTA" ? 2 : 1;
  if ((inst.operands.size() != 6 && inst.operands.size() != 7) ||
      !context.cta || !context.memory)
    return unsupported(inst,
                       "half MMA requires six/seven operands and CTA memory");
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
  tcgen05_ws_collector_control collector;
  if (ws && !decode_ws_collector_operand(b_operand.text, collector))
    return unsupported(inst, "invalid WS B collector control");
  uint32_t destination = 0, a_address = 0;
  const bool tmem_a = tmem_address(inst.operands[0], state, a_address);
  if ((!tmem_a && !wrapped_uniform(inst.operands[0], "gdesc", ar)) ||
      !wrapped_uniform(b_operand, "gdesc", br) ||
      !tmem_address(inst.operands[2], state, destination) ||
      !wrapped_uniform(inst.operands[3], "tmem", metadata) ||
      !wrapped_uniform(inst.operands[4], "idesc", dr) ||
      (!tmem_a && ar.index >= kZeroUniformRegister - 1) ||
      br.index >= kZeroUniformRegister - 1 ||
      inst.operands.back().kind != operand_kind::kUniformPredicate)
    return unsupported(inst, "unqualified UTCHMMA operand layout");
  if (state.read_uniform_register(metadata.index) != 0)
    return unsupported(inst, "half MMA sparse/masked metadata is unqualified");
  std::vector<uint32_t> disable_lanes(4 * group, 0);
  uint64_t ws_mask = 0;
  if (inst.operands.size() == 7) {
    const auto &mask = inst.operands[5];
    const unsigned words = ws ? 2 : disable_lanes.size();
    if (mask.kind != operand_kind::kUniformRegister ||
        (mask.index != kZeroUniformRegister &&
         mask.index + words > kZeroUniformRegister))
      return unsupported(inst, "invalid MMA mask register group");
    if (ws) {
      ws_mask = read_uniform_register_pair(state, mask.index);
    } else if (!read_output_lane_mask(mask, state, disable_lanes)) {
      return unsupported(inst, "invalid MMA output-lane mask");
    }
  }
  const uint32_t idesc = state.read_uniform_register(dr.index);
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  const unsigned af = (idesc >> 7) & 7, bf = (idesc >> 10) & 7;
  if (af == 2 || bf == 2)
    return tf32_mma(inst, state, context, idesc, tmem_a, a_address, destination,
                    ar, br, disable_lanes.data());
  if (!tcgen05_f16_descriptor_supported(idesc, group, ws))
    return unsupported(inst,
                       "unqualified dense F16/BF16 instruction descriptor");
  const auto desc = tcgen05_decode_f16_mma_descriptor(idesc, group, ws);
  const unsigned max_shift = ws ? tcgen05_ws_max_shift(idesc) : 0;
  if (ws && !tcgen05_ws_mask_supported(ws_mask, m, n, max_shift))
    return unsupported(inst, "invalid WS B zero-column descriptor");
  tcgen05_shared_descriptor_t ad, bd;
  if ((!tmem_a &&
       !shared_descriptor(read_uniform_register_pair(state, ar.index), ad,
                          context, desc.transpose_a)) ||
      !shared_descriptor(read_uniform_register_pair(state, br.index), bd,
                         context, desc.transpose_b))
    return unsupported(inst, "unqualified half MMA shared descriptor");
  const unsigned local_m = m / group, local_n = n / group;
  const bool folded = (group == 2 && m == 128) || ws;
  if (!tcgen05_mma_tmem_alignment_supported(local_m, folded, destination,
                                            tmem_a, a_address) ||
      (tmem_a && (desc.transpose_a || (folded && n % 32))))
    return unsupported(inst, "unqualified dense TMEM operand alignment");
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
        if (!tmem[rank]->contains((a_address & 0xffff) + (row << 16), 8))
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
  if ((!ws || !collector.reuses()) &&
      !gather(bd, n + max_shift, desc.transpose_b, b))
    return unsupported(inst, "half MMA B is outside mapped shared memory");
  if (ws) {
    unsigned owner = context.warp_id * kWarpLanes;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      if ((state.active_mask & (1u << lane)) &&
          lane_executes(inst, state, lane)) {
        owner += lane;
        break;
      }
    if (!tmem[0]->memory.ws_half_b(
            {0, 0}, owner, collector,
            {n, desc.k, desc.b_type, desc.transpose_b, max_shift}, b))
      return unsupported(inst, "WS B reuse requires a compatible prior fill");
    if (!tcgen05_ws_select_b(ws_mask, m, n, desc.k, max_shift, desc.transpose_b,
                             b))
      return unsupported(inst, "invalid WS B collector window");
  }
  const auto &predicate = inst.operands.back();
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
      physical_a[rank] = tmem[rank]->memory.read_matrix_packed_u16(
          {0, 0}, a_address & 0xffff, 128, 16);
    output = tcgen05_mma_f16_tmem_a_compute_words(
        desc, physical_a, b, input, accumulate, folded, a_address >> 16);
  }
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto begin = output.begin() + rank * local_m * n;
    write_accumulator(*tmem[rank], destination,
                      std::vector<uint32_t>(begin, begin + local_m * n),
                      local_m, n, folded, disable_lanes.data() + rank * 4);
  }
  context.tcgen_mma_effect = {true, false, desc.m, desc.n, desc.k, group};
  context.tcgen_mma_effect.timing = tcgen05_metadata(
      tcgen05_mma_kind::half, desc, group, tmem_a, ws, collector.reuses());
  context.tcgen_mma_effect.shared_reads = std::move(shared_reads.reads);
  // Reused B comes from the collector; it produces no shared-memory reads.
  context.tcgen_mma_effect.ws = ws;
  context.tcgen_mma_effect.tmem_a = tmem_a;
  context.tcgen_mma_effect.shared_operand_bytes =
      ws ? ((tmem_a ? 0 : desc.m) +
            (collector.reuses() ? 0 : desc.n + max_shift)) *
               desc.k * 2
         : 0;
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
