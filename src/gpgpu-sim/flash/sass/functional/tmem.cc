#include "../../tcgen05/copy.h"
#include "../decode/operand_parser.h"
#include "../decode/tmem_instruction.h"
#include "internal.h"
#include "tmem_internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

step_result copy_shared(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (!context.cta || !context.memory || !has_typed_operands(inst, 2))
    return unsupported(inst, "TMEM copy requires CTA memory and two operands");
  uint32_t base;
  operand source;
  if (!tmem_address(inst.operands[0], state, base) || (base >> 16) ||
      !wrapped_uniform(inst.operands[1], "gdesc", source) ||
      source.index + 1 >= kUniformRegisters)
    return unsupported(inst, "invalid TMEM copy operands");
  auto shape = tcgen05_copy_shape::warpx4;
  const bool paired = inst.opcode.find(".2CTA") != std::string::npos;
  std::string opcode = inst.opcode;
  if (paired)
    opcode.erase(opcode.find(".2CTA"), 5);
  if (opcode == "UTCCP.T.S")
    shape = tcgen05_copy_shape::rows128_bits256;
  else if (opcode == "UTCCP.T.S.128dp128bit")
    shape = tcgen05_copy_shape::rows128_bits128;
  else if (opcode == "UTCCP.T.S.2x64dp128bit_lw02_lw13")
    shape = tcgen05_copy_shape::warpx2_02_13;
  else if (opcode == "UTCCP.T.S.2x64dp128bit_lw01_lw23")
    shape = tcgen05_copy_shape::warpx2_01_23;
  std::vector<uint32_t> addresses;
  if (!tcgen05_copy_addresses(read_uniform_register_pair(state, source.index),
                              shape, addresses))
    return unsupported(inst, "unsupported TMEM copy descriptor");
  const unsigned width = addresses.size() / 128;
  if (base % width)
    return unsupported(inst, "unaligned TMEM copy");
  const unsigned first = context.cga_cta_id & ~1u;
  if (paired && (!context.cluster || context.cluster->ctas.size() < first + 2 ||
                 context.cluster->memories.size() < first + 2))
    return unsupported(inst, "paired copy requires two resident CTAs");
  const unsigned count = paired ? 2 : 1;
  functional_tcgen_state *targets[2] = {};
  std::vector<uint32_t> images[2];
  for (unsigned rank = 0; rank < count; ++rank) {
    auto *cta = paired ? context.cluster->ctas[first + rank] : context.cta;
    auto *mem =
        paired ? context.cluster->memories[first + rank] : context.memory;
    if (!cta || !cta->tcgen || !mem)
      return unsupported(inst, "TMEM copy requires live peer storage");
    targets[rank] = cta->tcgen.get();
    for (unsigned row = 0; row < 128; ++row)
      if (!targets[rank]->contains(base + (row << 16), width))
        return unsupported(inst, "TMEM copy exceeds a live allocation");
    images[rank].resize(addresses.size());
    for (unsigned i = 0; i < addresses.size(); ++i)
      if (!mem->read(memory_space::kShared, addresses[i], &images[rank][i], 4))
        return memory_fault(inst, memory_space::kShared, addresses[i], 4);
  }
  // Validate and gather both halves before publishing either image.
  for (unsigned rank = 0; rank < count; ++rank)
    targets[rank]->memory.write_matrix_words({0, 0}, base, images[rank], 128,
                                             width);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result commit(const instruction &inst, warp_state &state,
                   execution_context &context) {
  const bool multicast = inst.opcode.find(".MULTICAST") != std::string::npos;
  const bool paired = inst.opcode.find(".2CTA") != std::string::npos;
  if (!has_typed_operands(inst, multicast ? 3 : 2) || !context.cta ||
      inst.operands[0].kind != operand_kind::kMemoryAddress ||
      inst.operands[1].kind != operand_kind::kUniformRegister ||
      inst.operands[1].index != kZeroUniformRegister ||
      (multicast && inst.operands[2].kind != operand_kind::kUniformRegister))
    return unsupported(inst, "unqualified UTCBAR operand form");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  uint64_t address = 0;
  if (!evaluate_shared_address(inst.operands[0], state, 0, address) ||
      address % 8)
    return unsupported(inst, "unaligned UTCBAR address");
  const unsigned owner = context.cluster ? address >> 24 : 0;
  const uint32_t local = context.cluster ? address & 0xffffff : address;
  const unsigned size = context.cluster ? context.cluster->ctas.size() : 1;
  if (owner >= size || size > 16)
    return unsupported(inst, "invalid UTCBAR cluster owner");
  const unsigned mask =
      multicast ? state.read_uniform_register(inst.operands[2].index)
                : 1u << owner;
  if (owner >= size || !mask || size > 16 || (mask >> size) ||
      (paired && (!context.cluster || size < (context.cga_cta_id & ~1u) + 2)))
    return unsupported(inst, "invalid UTCBAR cluster destinations");
  if (!multicast && context.cluster && owner != context.cga_cta_id)
    return unsupported(inst, "remote unicast UTCBAR is not qualified");
  for (unsigned rank = 0; rank < size; ++rank) {
    if (!(mask & (1u << rank)))
      continue;
    auto *cta = context.cluster ? context.cluster->ctas[rank] : context.cta;
    auto *mem =
        context.cluster ? context.cluster->memories[rank] : context.memory;
    if (!cta || !mem)
      return unsupported(inst, "UTCBAR target is not resident");
    auto *saved_cta = context.cta;
    auto *saved_mem = context.memory;
    const auto imported = context.imported_mbarriers.size();
    context.cta = cta;
    context.memory = mem;
    const bool valid = ensure_mbarrier_imported(local, context) &&
                       cta->mbarrier_pending_arrivals(local);
    context.cta = saved_cta;
    context.memory = saved_mem;
    if (context.cluster)
      for (unsigned i = imported; i < context.imported_mbarriers.size(); ++i)
        context.imported_mbarriers[i].address |= rank << 24;
    if (!valid)
      return unsupported(inst, "UTCBAR requires pending initialized targets");
  }
  for (unsigned rank = 0; rank < size; ++rank)
    if (mask & (1u << rank))
      (context.cluster ? context.cluster->ctas[rank] : context.cta)
          ->arrive_mbarrier(local, 1, 0);
  functional_mbarrier_effect effect;
  effect.valid = true;
  effect.address = address;
  effect.count = 1;
  effect.multicast_mask = multicast ? mask : 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane)
    if ((state.active_mask & (1u << lane)) &&
        lane_executes(inst, state, lane)) {
      context.record_mbarrier_effect(lane, effect);
      break;
    }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
step_result allocate(const instruction &inst, warp_state &state,
                     execution_context &context) {
  if (!context.cta || !has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kUniformPredicate ||
      inst.operands[1].kind != operand_kind::kUniformRegister ||
      inst.operands[2].kind != operand_kind::kUniformRegister)
    return unsupported(inst, "unsupported native TMEM allocation form");
  if (any_lane_executes(inst, state)) {
    const uint32_t blocks = state.read_uniform_register(inst.operands[2].index);
    const bool full = inst.opcode == "UTCATOMSWS.FIND_AND_SET" ||
                      inst.opcode == "UTCATOMSWS.2CTA.FIND_AND_SET";
    if (!blocks || blocks > 16 || (blocks & (blocks - 1)))
      return unsupported(inst, "invalid native TMEM allocation block count");
    if (full &&
        (blocks != 16 || inst.operands[1].index != kZeroUniformRegister))
      return unsupported(inst, "unqualified non-ALIGN allocation form");
    const uint32_t mask = (1u << blocks) - 1;
    auto &tmem = storage(context);
    functional_tcgen_state *peer = nullptr;
    if (inst.opcode.find(".2CTA") != std::string::npos) {
      if (!context.cluster || (context.cga_cta_id & 1) ||
          context.cluster->ctas.size() <= (context.cga_cta_id ^ 1u) ||
          !context.cluster->ctas[context.cga_cta_id ^ 1u])
        return unsupported(
            inst,
            "paired physical allocation requires even CTA and resident peer");
      auto &p = context.cluster->ctas[context.cga_cta_id ^ 1u]->tcgen;
      if (!p)
        p = std::make_shared<functional_tcgen_state>();
      peer = p.get();
    }
    bool found = false;
    for (unsigned base = 0; base + blocks <= 16; base += blocks) {
      if (!((tmem.allocated | (peer ? peer->allocated : 0)) & (mask << base))) {
        tmem.allocated |= mask << base;
        if (peer)
          peer->allocated |= mask << base;
        state.write_uniform_register(inst.operands[1].index, base);
        found = true;
        break;
      }
    }
    state.write_uniform_predicate(inst.operands[0].index, found);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result free_blocks(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!context.cta || !has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[1].kind != operand_kind::kUniformRegister)
    return unsupported(inst, "unsupported native TMEM bitmap AND");
  if (any_lane_executes(inst, state)) {
    auto &tmem = storage(context);
    const uint32_t mask = state.read_uniform_register(inst.operands[1].index);
    const uint32_t prior = tmem.allocated;
    tmem.allocated &= mask;
    state.write_uniform_register(inst.operands[0].index, prior);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result transfer(const instruction &inst, warp_state &state,
                     execution_context &context) {
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (!context.cta || inst.operands.size() != 2 ||
      state.active_mask != 0xffffffffu)
    return unsupported(inst,
                       "TMEM transfer requires CTA state and a full warp");
  const bool store = inst.opcode.rfind("STTM", 0) == 0;
  const bool packed = tmem_packed_transfer_words(inst.opcode) != 0;
  const int half = tmem_transfer_half(inst.opcode);
  const bool shape16 =
      (!packed && half < 0 && inst.opcode.rfind("LDTM.16dp", 0) == 0) ||
      inst.opcode.rfind("STTM.16dp256bit", 0) == 0;
  const unsigned shape_bits = inst.opcode == "LDTM.16dp64bit.x4" ? 64
                              : (inst.opcode == "LDTM.16dp128bit.x4" ||
                                 inst.opcode == "LDTM.16dp128bit.x2")
                                  ? 128
                              : shape16 ? 256
                                        : 32;
  const unsigned words = tmem_transfer_words(inst.opcode);
  if (words == 0)
    return unsupported(inst, "unqualified TMEM transfer width");
  const auto &reg = inst.operands[store ? 1 : 0];
  const unsigned address_index = store ? 0 : 1;
  const std::string &text = inst.operands[address_index].text;
  const bool zero_store = store && reg.index == kZeroRegister;
  if (reg.kind != operand_kind::kRegister ||
      (!zero_store && reg.index + words > kZeroRegister) ||
      text.rfind("tmem[", 0) || text.back() != ']')
    return unsupported(inst, "unsupported TMEM transfer operand");
  uint32_t base = 0;
  if (!tmem_address(inst.operands[address_index], state, base))
    return unsupported(inst, "TMEM transfer requires a uniform address");
  // Register transfers are restricted to the issuing warp's 32-datapath
  // partition. Address row bits 5:6 do not select another warp's partition.
  base = tcgen05_tmem_warp_base(base, context.warp_id);
  const bool rows16 = shape16 || half >= 0;
  if ((base >> 16) > (rows16 ? 112u : 96u) ||
      ((base >> 16) & 31) > (rows16 ? 16u : 0u) ||
      (base & 0xffff) > 512u - ((shape16 || packed) ? words * 2 : words))
    return unsupported(inst, "TMEM transfer row span exceeds 128 lanes");
  auto &tmem = storage(context);
  auto participates = [&](unsigned lane) {
    return half < 0 || lane / 16 == unsigned(half);
  };
  auto address = [&](unsigned lane, unsigned word) {
    if (packed)
      return tcgen05_tmem_register_address(base, half < 0 ? lane : lane % 16,
                                           half < 0 ? 32 : 16, 32, word, true);
    if (half >= 0)
      return base + ((lane % 16) << 16) + word;
    return tcgen05_tmem_register_address(base, lane, shape16 ? 16 : 32,
                                         shape_bits, word);
  };
  // Validate all lanes first, so a malformed span cannot partially write.
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "TMEM transfer is divergent");
    if (!participates(lane))
      continue;
    for (unsigned w = 0; w < words; ++w) {
      const auto location = address(lane, w);
      if (!tmem.contains(location, packed ? 2 : 1))
        return unsupported(
            inst, "TMEM transfer is divergent or outside a live allocation");
    }
  }
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!participates(lane))
      continue;
    for (unsigned w = 0; w < words; ++w) {
      const auto location = address(lane, w);
      if (store) {
        const uint32_t value =
            zero_store ? 0 : state.read_register(lane, reg.index + w);
        // One FP16 per physical 32-bit column; no numeric conversion. Upper
        // unused bits are deterministically zeroed, never interpreted as FP32.
        if (packed)
          tmem.memory.write_words({0, 0}, location,
                                  {value & 0xffffu, value >> 16});
        else
          tmem.memory.write_words({0, 0}, location, {value});
      } else if (packed) {
        const auto pair = tmem.memory.read_words({0, 0}, location, 2);
        state.write_register(lane, reg.index + w,
                             (pair[0] & 0xffffu) | (pair[1] << 16));
      } else {
        const auto value = tmem.memory.read_words({0, 0}, location, 1)[0];
        state.write_register(lane, reg.index + w, value);
      }
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result relinquish_pool(const instruction &inst, warp_state &state,
                            execution_context &) {
  if (!has_typed_operands(inst, 1) ||
      inst.operands[0].kind != operand_kind::kImmediate ||
      inst.operands[0].immediate != 0x80)
    return unsupported(inst, "unqualified TMEM SMPOOL virtual count");
  // Scheduler accounting only; the compiler separately updates the shared
  // allocation-permit byte. No instruction timing or physical occupancy here.
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
} // namespace
void register_tmem_semantics(frontend &target) {
  target.register_semantics("UTCBAR", commit);
  target.register_semantics("UTCBAR.MULTICAST", commit);
  target.register_semantics("UTCBAR.2CTA", commit);
  target.register_semantics("UTCBAR.2CTA.MULTICAST", commit);
  for (const char *op :
       {"UTCCP.T.S.2CTA", "UTCCP.T.S.2CTA.128dp128bit",
        "UTCCP.T.S.2CTA.2x64dp128bit_lw02_lw13",
        "UTCCP.T.S.2CTA.2x64dp128bit_lw01_lw23", "UTCCP.T.S.2CTA.4x32dp128bit"})
    target.register_semantics(op, copy_shared);
  target.register_semantics("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", allocate);
  target.register_semantics("UTCATOMSWS.2CTA.FIND_AND_SET", allocate);
  for (const char *op :
       {"UTCCP.T.S", "UTCCP.T.S.128dp128bit", "UTCCP.T.S.4x32dp128bit",
        "UTCCP.T.S.2x64dp128bit_lw02_lw13", "UTCCP.T.S.2x64dp128bit_lw01_lw23"})
    target.register_semantics(op, copy_shared);
  target.register_semantics("LDTM.16dp128bit.x2", transfer);
  for (const char *shape : {"", ".16dp32bit_t0_t15", ".16dp32bit_t16_t31"})
    for (unsigned words = 1; words <= 64; words *= 2) {
      const std::string width = words == 1 ? "" : ".x" + std::to_string(words);
      target.register_semantics(
          std::string("STTM") + shape + width + ".EXPAND16BIT", transfer);
      target.register_semantics(
          std::string("LDTM") + shape + width + ".PACK16BIT", transfer);
      if (shape[0] != '\0') {
        target.register_semantics(std::string("STTM") + shape + width,
                                  transfer);
        target.register_semantics(std::string("LDTM") + shape + width,
                                  transfer);
      }
    }
  target.register_semantics("UTCATOMSWS.FIND_AND_SET.ALIGN", allocate);
  target.register_semantics("UTCATOMSWS.FIND_AND_SET", allocate);
  target.register_semantics("UTCATOMSWS.AND", free_blocks);
  target.register_semantics("LDTM", transfer);
  target.register_semantics("LDTM.x2", transfer);
  target.register_semantics("LDTM.x128", transfer);
  target.register_semantics("STTM.x8", transfer);
  target.register_semantics("STTM.x16", transfer);
  target.register_semantics("STTM.x128", transfer);
  target.register_semantics("STTM", transfer);
  target.register_semantics("STTM.x2", transfer);
  target.register_semantics("STTM.x4", transfer);
  target.register_semantics("STTM.16dp256bit", transfer);
  target.register_semantics("STTM.16dp256bit.x2", transfer);
  target.register_semantics("STTM.16dp256bit.x4", transfer);
  target.register_semantics("STTM.16dp256bit.x8", transfer);
  target.register_semantics("STTM.16dp256bit.x16", transfer);
  target.register_semantics("STTM.x32", transfer);
  target.register_semantics("STTM.x64", transfer);
  target.register_semantics("LDTM.x4", transfer);
  target.register_semantics("LDTM.16dp64bit.x4", transfer);
  target.register_semantics("LDTM.16dp128bit.x4", transfer);
  target.register_semantics("LDTM.x8", transfer);
  target.register_semantics("LDTM.x32", transfer);
  target.register_semantics("LDTM.x64", transfer);
  target.register_semantics("LDTM.x16", transfer);
  target.register_semantics("LDTM.16dp256bit", transfer);
  target.register_semantics("LDTM.16dp256bit.x2", transfer);
  target.register_semantics("LDTM.16dp256bit.x4", transfer);
  target.register_semantics("LDTM.16dp256bit.x8", transfer);
  target.register_semantics("LDTM.16dp256bit.x16", transfer);
  target.register_semantics("UVIRTCOUNT.DEALLOC.SMPOOL", relinquish_pool);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
