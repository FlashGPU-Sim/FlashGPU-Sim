#include "tmem.h"
#include "../../tcgen05/copy.h"
#include "../decode/operand_parser.h"
#include "../decode/tmem_instruction.h"
#include "internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
functional_tcgen_state &storage(execution_context &context) {
  if (!context.cta->tcgen)
    context.cta->tcgen = std::make_shared<functional_tcgen_state>();
  return *context.cta->tcgen;
}

bool tmem_address(const operand &op, const warp_state &state, uint32_t &value) {
  if (op.text.rfind("tmem[", 0) || op.text.back() != ']')
    return false;
  const auto address = parse_operand(op.text.substr(4));
  if (address.kind != operand_kind::kMemoryAddress || address.has_prefix ||
      address.address_groups.size() != 1)
    return false;
  uint64_t result = 0;
  for (const auto &term : address.address_groups[0].terms) {
    if (term.wide || term.negated)
      return false;
    if (term.kind == address_term_kind::kUniformRegister)
      result += state.read_uniform_register(term.index);
    else if (term.kind == address_term_kind::kImmediate) {
      if (term.immediate < 0) {
        const uint64_t magnitude = uint64_t(-(term.immediate + 1)) + 1;
        if (result < magnitude)
          return false;
        result -= magnitude;
      } else {
        if (uint64_t(term.immediate) > UINT32_MAX - result)
          return false;
        result += uint64_t(term.immediate);
      }
    } else
      return false;
    if (result > UINT32_MAX)
      return false;
  }
  value = result;
  return true;
}

bool wrapped_uniform(const operand &op, const char *prefix, operand &inner) {
  const std::string start = std::string(prefix) + "[";
  if (op.text.rfind(start, 0) != 0 || op.text.back() != ']')
    return false;
  inner = parse_operand(
      op.text.substr(start.size(), op.text.size() - start.size() - 1));
  return inner.kind == operand_kind::kUniformRegister && !inner.negated &&
         !inner.absolute && !inner.wide && !inner.bitwise_complement;
}
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
  if (inst.opcode == "UTCCP.T.S")
    shape = tcgen05_copy_shape::rows128_bits256;
  else if (inst.opcode == "UTCCP.T.S.128dp128bit")
    shape = tcgen05_copy_shape::rows128_bits128;
  else if (inst.opcode == "UTCCP.T.S.2x64dp128bit_lw02_lw13")
    shape = tcgen05_copy_shape::warpx2_02_13;
  else if (inst.opcode == "UTCCP.T.S.2x64dp128bit_lw01_lw23")
    shape = tcgen05_copy_shape::warpx2_01_23;
  std::vector<uint32_t> addresses;
  if (!tcgen05_copy_addresses(read_uniform_register_pair(state, source.index),
                              shape, addresses))
    return unsupported(inst, "unsupported TMEM copy descriptor");
  const unsigned width = addresses.size() / 128;
  auto &tm = storage(context);
  if (base % width)
    return unsupported(inst, "unaligned TMEM copy");
  for (unsigned row = 0; row < 128; ++row)
    if (!tm.contains(base + (row << 16), width))
      return unsupported(inst, "TMEM copy exceeds a live allocation");
  std::vector<uint32_t> values(addresses.size());
  for (unsigned i = 0; i < addresses.size(); ++i)
    if (!context.memory->read(memory_space::kShared, addresses[i], &values[i],
                              4))
      return memory_fault(inst, memory_space::kShared, addresses[i], 4);
  tm.memory.write_matrix_words({0, 0}, base, values, 128, width);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result commit(const instruction &inst, warp_state &state,
                   execution_context &context) {
  if (!has_typed_operands(inst, 2) || !context.cta ||
      inst.operands[0].kind != operand_kind::kMemoryAddress ||
      inst.operands[1].kind != operand_kind::kUniformRegister ||
      inst.operands[1].index != kZeroUniformRegister)
    return unsupported(inst, "unqualified UTCBAR operand form");
  if (any_lane_executes(inst, state)) {
    uint64_t address = 0;
    if (!evaluate_shared_address(inst.operands[0], state, 0, address) ||
        address % 8)
      return unsupported(inst, "unaligned UTCBAR address");
    if (!ensure_mbarrier_imported(address, context) ||
        !context.cta->arrive_mbarrier(address, 1, 0))
      return unsupported(
          inst, "UTCBAR requires a pending initialized local mbarrier");
    // All qualified TCGEN operations execute synchronously in functional mode.
    // This arrival is after their data effects; no timing latency is implied.
    functional_mbarrier_effect effect;
    effect.valid = true;
    effect.address = static_cast<uint32_t>(address);
    effect.count = 1;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      if ((state.active_mask & (1u << lane)) &&
          lane_executes(inst, state, lane)) {
        context.record_mbarrier_effect(lane, effect);
        break;
      }
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
    const bool full = inst.opcode == "UTCATOMSWS.FIND_AND_SET";
    if (!blocks || blocks > 16 || (blocks & (blocks - 1)))
      return unsupported(inst, "invalid native TMEM allocation block count");
    if (full &&
        (blocks != 16 || inst.operands[1].index != kZeroUniformRegister))
      return unsupported(inst, "unqualified non-ALIGN allocation form");
    const uint32_t mask = (1u << blocks) - 1;
    auto &tmem = storage(context);
    bool found = false;
    for (unsigned base = 0; base + blocks <= 16; base += blocks) {
      if (!(tmem.allocated & (mask << base))) {
        tmem.allocated |= mask << base;
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
