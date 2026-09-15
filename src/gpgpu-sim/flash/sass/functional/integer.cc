#include "internal.h"

#include <algorithm>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.1/9.7.6-9.7.8: integer, logic, and selection.

step_result execute_iadd(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported IADD operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t a = 0;
    uint32_t b = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, a) ||
        !read_u32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported IADD source");
    state.write_register(lane, inst.operands[0].index, a + b);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_iadd3(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool hopper_compact = has_typed_operands(inst, 4);
  const bool hopper_carry = has_typed_operands(inst, 5);
  if (!hopper_compact && !hopper_carry && !has_typed_operands(inst, 6))
    return unsupported(inst, "expected four, five, or six typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  if (destination == nullptr)
    return unsupported(inst, "IADD3 destination must be a vector register");
  size_t first_source = 1;
  const operand *carry_output0 = nullptr;
  const operand *carry_output1 = nullptr;
  if (hopper_carry) {
    carry_output0 = get_operand(inst, 1, operand_kind::kPredicate);
    if (carry_output0 == nullptr)
      return unsupported(inst, "IADD3 carry output must be a predicate");
    first_source = 2;
  } else if (!hopper_compact) {
    carry_output0 = get_operand(inst, 1, operand_kind::kPredicate);
    carry_output1 = get_operand(inst, 2, operand_kind::kPredicate);
    if (carry_output0 == nullptr || carry_output1 == nullptr)
      return unsupported(inst, "IADD3 carry outputs must be predicates");
    first_source = 3;
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    if (!read_u32_operand(inst.operands[first_source], state, lane, a) ||
        !read_u32_operand(inst.operands[first_source + 1], state, lane, b) ||
        !read_u32_operand(inst.operands[first_source + 2], state, lane, c))
      return unsupported(inst, "unsupported IADD3 source");
    const uint64_t first_sum = static_cast<uint64_t>(a) + b;
    const uint64_t second_sum =
        static_cast<uint32_t>(first_sum) + static_cast<uint64_t>(c);
    const uint64_t sum = static_cast<uint64_t>(a) + b + c;
    state.write_register(lane, destination->index, static_cast<uint32_t>(sum));
    if (carry_output0 != nullptr)
      state.write_predicate(lane, carry_output0->index,
                            carry_output1 == nullptr ? (sum >> 32) != 0
                                                     : (first_sum >> 32) != 0);
    if (carry_output1 != nullptr)
      state.write_predicate(lane, carry_output1->index,
                            (second_sum >> 32) != 0);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_iadd3x(const instruction &inst, warp_state &state,
                           execution_context &) {
  if (!has_typed_operands(inst, 6) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported IADD3.X operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    bool carry0 = false;
    bool carry1 = false;
    if (!read_u32_operand(inst.operands[1], state, lane, a) ||
        !read_u32_operand(inst.operands[2], state, lane, b) ||
        !read_u32_operand(inst.operands[3], state, lane, c) ||
        !read_predicate_operand(inst.operands[4], state, lane, carry0) ||
        !read_predicate_operand(inst.operands[5], state, lane, carry1))
      return unsupported(inst, "unsupported IADD3.X source");
    state.write_register(lane, inst.operands[0].index,
                         a + b + c + static_cast<uint32_t>(carry0) +
                             static_cast<uint32_t>(carry1));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_uiadd3(const instruction &inst, warp_state &state,
                           execution_context &) {
  const bool hopper_compact =
      (inst.opcode == "UIADD3" || inst.opcode == "UIADD3.64") &&
      has_typed_operands(inst, 4);
  const bool hopper_single_carry =
      inst.opcode == "UIADD3" && has_typed_operands(inst, 5);
  if (!hopper_compact && !hopper_single_carry && !has_typed_operands(inst, 6))
    return unsupported(inst, "expected four, five, or six typed operands");
  const operand *destination =
      get_operand(inst, 0, operand_kind::kUniformRegister);
  const size_t first_source = hopper_compact ? 1 : hopper_single_carry ? 2 : 3;
  const operand *carry_output0 =
      hopper_compact ? nullptr
                     : get_operand(inst, 1, operand_kind::kUniformPredicate);
  const operand *carry_output1 =
      hopper_compact || hopper_single_carry
          ? nullptr
          : get_operand(inst, 2, operand_kind::kUniformPredicate);
  if (destination == nullptr ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate) ||
      inst.operands[first_source].kind == operand_kind::kRegister ||
      inst.operands[first_source + 1].kind == operand_kind::kRegister ||
      inst.operands[first_source + 2].kind == operand_kind::kRegister)
    return unsupported(inst, "UIADD3 source is not warp uniform");
  if ((!hopper_compact && carry_output0 == nullptr) ||
      (!hopper_compact && !hopper_single_carry && carry_output1 == nullptr))
    return unsupported(inst, "UIADD3 carry outputs must be predicates");
  if (any_lane_executes(inst, state)) {
    if (inst.opcode == "UIADD3.64") {
      uint64_t a = 0;
      uint64_t b = 0;
      uint64_t c = 0;
      if (!read_u64_operand(inst.operands[first_source], state, 0, a) ||
          !read_u64_operand(inst.operands[first_source + 1], state, 0, b) ||
          !read_u64_operand(inst.operands[first_source + 2], state, 0, c))
        return unsupported(inst, "unsupported UIADD3.64 source");
      const uint64_t result = a + b + c;
      write_uniform_register_pair(state, destination->index, result);
    } else {
      uint32_t a = 0;
      uint32_t b = 0;
      uint32_t c = 0;
      if (!read_u32_operand(inst.operands[first_source], state, 0, a) ||
          !read_u32_operand(inst.operands[first_source + 1], state, 0, b) ||
          !read_u32_operand(inst.operands[first_source + 2], state, 0, c))
        return unsupported(inst, "unsupported UIADD3 source");
      const uint64_t first_sum = static_cast<uint64_t>(a) + b;
      const uint64_t second_sum =
          static_cast<uint32_t>(first_sum) + static_cast<uint64_t>(c);
      state.write_uniform_register(destination->index, a + b + c);
      if (hopper_single_carry)
        state.write_uniform_predicate(
            carry_output0->index,
            ((static_cast<uint64_t>(a) + b + c) >> 32) != 0);
      else if (carry_output0 != nullptr)
        state.write_uniform_predicate(carry_output0->index,
                                      (first_sum >> 32) != 0);
      if (carry_output1 != nullptr)
        state.write_uniform_predicate(carry_output1->index,
                                      (second_sum >> 32) != 0);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_uiadd3x(const instruction &inst, warp_state &state,
                            execution_context &) {
  if (!has_typed_operands(inst, 6) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[1].kind == operand_kind::kRegister ||
      inst.operands[2].kind == operand_kind::kRegister ||
      inst.operands[3].kind == operand_kind::kRegister ||
      inst.operands[4].kind != operand_kind::kUniformPredicate ||
      inst.operands[5].kind != operand_kind::kUniformPredicate ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported UIADD3.X operand form");
  if (any_lane_executes(inst, state)) {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    bool carry0 = false;
    bool carry1 = false;
    if (!read_u32_operand(inst.operands[1], state, 0, a) ||
        !read_u32_operand(inst.operands[2], state, 0, b) ||
        !read_u32_operand(inst.operands[3], state, 0, c) ||
        !read_predicate_operand(inst.operands[4], state, 0, carry0) ||
        !read_predicate_operand(inst.operands[5], state, 0, carry1))
      return unsupported(inst, "unsupported UIADD3.X source");
    state.write_uniform_register(inst.operands[0].index,
                                 a + b + c + static_cast<uint32_t>(carry0) +
                                     static_cast<uint32_t>(carry1));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_iadd64(const instruction &inst, warp_state &state,
                           execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported IADD.64 operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t a = 0;
    uint64_t b = 0;
    if (!read_u64_operand(inst.operands[1], state, lane, a) ||
        !read_u64_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported IADD.64 source");
    write_register_pair(state, lane, inst.operands[0].index, a + b);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_shf(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected four typed operands");
  const operand &shift = inst.operands[2];
  if (shift.kind == operand_kind::kImmediate &&
      (shift.immediate < 0 || shift.immediate >= 32))
    return unsupported(inst, "invalid SHF shift amount");
  const bool uniform = inst.opcode.rfind("USHF.", 0) == 0;
  if ((uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && inst.has_guard &&
       inst.guard.kind != operand_kind::kUniformPredicate) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   shift.kind == operand_kind::kRegister ||
                   inst.operands[3].kind == operand_kind::kRegister)))
    return unsupported(inst, "unsupported SHF destination or guard");

  const auto compute = [&](unsigned lane, uint32_t &result) {
    uint32_t low = 0;
    uint32_t high = 0;
    uint32_t shift_value = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, low) ||
        !read_u32_operand(shift, state, lane, shift_value) ||
        !read_u32_operand(inst.operands[3], state, lane, high))
      return false;
    const unsigned amount = shift_value & 31u;
    if (inst.opcode == "SHF.L.U32" || inst.opcode == "USHF.L.U32") {
      result = amount == 0 ? low : (low << amount) | (high >> (32 - amount));
    } else if (inst.opcode == "SHF.R.S32.HI" ||
               inst.opcode == "USHF.R.S32.HI") {
      if (low != 0)
        return false;
      result = static_cast<uint32_t>(static_cast<int32_t>(high) >> amount);
    } else if (inst.opcode == "SHF.R.U32.HI" ||
               inst.opcode == "USHF.R.U32.HI") {
      if (low != 0)
        return false;
      result = high >> amount;
    } else if (inst.opcode == "SHF.R.U64") {
      const uint64_t pair = (static_cast<uint64_t>(high) << 32) | low;
      result = static_cast<uint32_t>(pair >> amount);
    } else if (inst.opcode == "SHF.L.U64.HI" ||
               inst.opcode == "USHF.L.U64.HI") {
      const uint64_t pair = (static_cast<uint64_t>(high) << 32) | low;
      result = static_cast<uint32_t>((pair << amount) >> 32);
    } else {
      return false;
    }
    return true;
  };

  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported uniform SHF operand form");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported SHF operand form");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_lop3(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool uniform = inst.opcode == "ULOP3.LUT";
  const bool predicate_output = inst.operands.size() == 7;
  const size_t expected = predicate_output ? 7 : 6;
  if (!has_typed_operands(inst, expected))
    return unsupported(inst, "unexpected LOP3.LUT operand count");
  const size_t destination_position = predicate_output ? 1 : 0;
  const size_t first_source = predicate_output ? 2 : 1;
  const size_t lut_position = predicate_output ? 5 : 4;
  const size_t predicate_position = predicate_output ? 6 : 5;
  const operand *destination = get_operand(
      inst, destination_position,
      uniform ? operand_kind::kUniformRegister : operand_kind::kRegister);
  const operand *output_predicate =
      predicate_output ? get_operand(inst, 0,
                                     uniform ? operand_kind::kUniformPredicate
                                             : operand_kind::kPredicate)
                       : nullptr;
  const operand *lut =
      get_operand(inst, lut_position, operand_kind::kImmediate);
  const operand *predicate = get_operand(
      inst, predicate_position,
      uniform ? operand_kind::kUniformPredicate : operand_kind::kPredicate);
  if (destination == nullptr || lut == nullptr || predicate == nullptr ||
      (predicate_output && output_predicate == nullptr) ||
      predicate->index != kTruePredicate || !predicate->negated ||
      lut->immediate < 0 || lut->immediate > 0xff ||
      (uniform &&
       (inst.operands[first_source].kind == operand_kind::kRegister ||
        inst.operands[first_source + 1].kind == operand_kind::kRegister ||
        inst.operands[first_source + 2].kind == operand_kind::kRegister ||
        (inst.has_guard &&
         inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported LOP3.LUT operand form");
  const auto compute = [&](unsigned lane, uint32_t &value) {
    uint32_t av = 0;
    uint32_t bv = 0;
    uint32_t cv = 0;
    if (!read_u32_operand(inst.operands[first_source], state, lane, av) ||
        !read_u32_operand(inst.operands[first_source + 1], state, lane, bv) ||
        !read_u32_operand(inst.operands[first_source + 2], state, lane, cv))
      return false;
    value = 0;
    for (unsigned bit = 0; bit < 32; ++bit) {
      const unsigned index = (((av >> bit) & 1u) << 2) |
                             (((bv >> bit) & 1u) << 1) | ((cv >> bit) & 1u);
      value |= ((static_cast<uint64_t>(lut->immediate) >> index) & 1u) << bit;
    }
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t value = 0;
      if (!compute(0, value))
        return unsupported(inst, "unsupported ULOP3.LUT source");
      state.write_uniform_register(destination->index, value);
      if (predicate_output)
        state.write_uniform_predicate(output_predicate->index, value != 0);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t value = 0;
      if (!compute(lane, value))
        return unsupported(inst, "unsupported LOP3.LUT source");
      state.write_register(lane, destination->index, value);
      if (predicate_output)
        state.write_predicate(lane, output_predicate->index, value != 0);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_plop3(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool uniform = inst.opcode == "UPLOP3.LUT";
  if (!has_typed_operands(inst, 7))
    return unsupported(inst,
                       "expected two outputs, three inputs, and two LUTs");
  const operand_kind predicate_kind =
      uniform ? operand_kind::kUniformPredicate : operand_kind::kPredicate;
  if (inst.operands[0].kind != predicate_kind ||
      inst.operands[1].kind != predicate_kind)
    return unsupported(inst, "PLOP3 outputs must be predicates");
  for (size_t position = 2; position < 5; ++position) {
    if ((uniform &&
         inst.operands[position].kind != operand_kind::kUniformPredicate) ||
        (!uniform && inst.operands[position].kind != operand_kind::kPredicate &&
         inst.operands[position].kind != operand_kind::kUniformPredicate))
      return unsupported(inst, "PLOP3 inputs must be predicates");
  }
  const operand *first_lut = get_operand(inst, 5, operand_kind::kImmediate);
  const operand *second_lut = get_operand(inst, 6, operand_kind::kImmediate);
  if (first_lut == nullptr || second_lut == nullptr ||
      first_lut->immediate < 0 || first_lut->immediate > 0xff ||
      second_lut->immediate < 0 || second_lut->immediate > 0xff)
    return unsupported(inst, "PLOP3 LUT must be an 8-bit immediate");

  const auto execute = [&](unsigned lane) {
    bool first = false;
    bool second = false;
    bool third = false;
    if (!read_predicate_operand(inst.operands[2], state, lane, first) ||
        !read_predicate_operand(inst.operands[3], state, lane, second) ||
        !read_predicate_operand(inst.operands[4], state, lane, third))
      return false;
    const unsigned index = (static_cast<unsigned>(first) << 2) |
                           (static_cast<unsigned>(second) << 1) |
                           static_cast<unsigned>(third);
    const bool first_result =
        (static_cast<uint64_t>(first_lut->immediate) >> index) & 1u;
    const bool second_result =
        (static_cast<uint64_t>(second_lut->immediate) >> index) & 1u;
    if (uniform) {
      state.write_uniform_predicate(inst.operands[0].index, first_result);
      state.write_uniform_predicate(inst.operands[1].index, second_result);
    } else {
      state.write_predicate(lane, inst.operands[0].index, first_result);
      state.write_predicate(lane, inst.operands[1].index, second_result);
    }
    return true;
  };

  if (uniform) {
    if (any_lane_executes(inst, state) && !execute(0))
      return unsupported(inst, "unsupported UPLOP3 input predicate");
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      if (!execute(lane))
        return unsupported(inst, "unsupported PLOP3 input predicate");
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_prmt(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool uniform = inst.opcode == "UPRMT";
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != (uniform ? operand_kind::kUniformRegister
                                        : operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   inst.operands[2].kind == operand_kind::kRegister ||
                   inst.operands[3].kind == operand_kind::kRegister)))
    return unsupported(inst, "unsupported PRMT operand form");
  const auto permute = [](uint32_t low, uint32_t selector, uint32_t high) {
    const uint64_t bytes = (static_cast<uint64_t>(high) << 32) | low;
    uint32_t result = 0;
    for (unsigned destination_byte = 0; destination_byte < 4;
         ++destination_byte) {
      const unsigned selection = (selector >> (destination_byte * 4)) & 0xfu;
      uint8_t value = static_cast<uint8_t>(bytes >> ((selection & 7u) * 8));
      if ((selection & 8u) != 0)
        value = (value & 0x80u) != 0 ? 0xffu : 0u;
      result |= static_cast<uint32_t>(value) << (destination_byte * 8);
    }
    return result;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t low = 0;
      uint32_t high = 0;
      uint32_t selector = 0;
      if (!read_u32_operand(inst.operands[1], state, 0, low) ||
          !read_u32_operand(inst.operands[2], state, 0, selector) ||
          !read_u32_operand(inst.operands[3], state, 0, high))
        return unsupported(inst, "unsupported UPRMT source");
      state.write_uniform_register(inst.operands[0].index,
                                   permute(low, selector, high));
    }
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t low = 0;
    uint32_t high = 0;
    uint32_t selector = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, low) ||
        !read_u32_operand(inst.operands[2], state, lane, selector) ||
        !read_u32_operand(inst.operands[3], state, lane, high))
      return unsupported(inst, "unsupported PRMT source");
    state.write_register(lane, inst.operands[0].index,
                         permute(low, selector, high));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_imad(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool carry_input = inst.opcode == "IMAD.X";
  const size_t operand_count = carry_input ? 5 : 4;
  if (!has_typed_operands(inst, operand_count))
    return unsupported(inst, "unexpected IMAD operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *carry =
      carry_input ? get_operand(inst, 4, operand_kind::kPredicate) : nullptr;
  if (destination == nullptr || (carry_input && carry == nullptr))
    return unsupported(inst, "IMAD destination must be a vector register");
  const bool high = inst.opcode == "IMAD.HI" || inst.opcode == "IMAD.HI.U32";
  const bool wide = inst.opcode.rfind("IMAD.WIDE", 0) == 0;
  const bool unsigned_product = inst.opcode.find(".U32") != std::string::npos;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t a = 0;
    uint32_t b = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, a) ||
        !read_u32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported IMAD multiplicand");
    if (wide || high) {
      // SM120 HI forms consume the addend as a 64-bit register pair even
      // though nvdisasm prints only its low register. Integer-division
      // refinement sequences depend on the implicit c + 1 high half.
      uint64_t addend = 0;
      if (!read_u64_operand(inst.operands[3], state, lane, addend))
        return unsupported(inst, "unsupported IMAD 64-bit addend");
      uint64_t product = 0;
      if (unsigned_product)
        product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
      else
        product = static_cast<uint64_t>(
            static_cast<int64_t>(static_cast<int32_t>(a)) *
            static_cast<int64_t>(static_cast<int32_t>(b)));
      const uint64_t result = product + addend;
      if (high)
        state.write_register(lane, destination->index,
                             static_cast<uint32_t>(result >> 32));
      else
        write_register_pair(state, lane, destination->index, result);
    } else {
      uint32_t addend = 0;
      if (!read_u32_operand(inst.operands[3], state, lane, addend))
        return unsupported(inst, "unsupported IMAD addend");
      bool carry_value = false;
      if (carry_input &&
          !read_predicate_operand(*carry, state, lane, carry_value))
        return unsupported(inst, "unsupported IMAD carry predicate");
      const uint64_t product = static_cast<uint64_t>(a) * b;
      const uint32_t result = static_cast<uint32_t>(product) + addend +
                              static_cast<uint32_t>(carry_value);
      state.write_register(lane, destination->index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_uimad(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate) ||
      inst.operands[1].kind == operand_kind::kRegister ||
      inst.operands[2].kind == operand_kind::kRegister ||
      inst.operands[3].kind == operand_kind::kRegister)
    return unsupported(inst, "unsupported UIMAD operand form");
  if (any_lane_executes(inst, state)) {
    uint32_t a = 0;
    uint32_t b = 0;
    if (!read_u32_operand(inst.operands[1], state, 0, a) ||
        !read_u32_operand(inst.operands[2], state, 0, b))
      return unsupported(inst, "unsupported UIMAD source");
    const bool high = inst.opcode == "UIMAD.HI.U32";
    const bool wide = inst.opcode.rfind("UIMAD.WIDE", 0) == 0;
    const bool unsigned_product = inst.opcode.find(".U32") != std::string::npos;
    if (wide || high) {
      uint64_t c = 0;
      if (!read_u64_operand(inst.operands[3], state, 0, c))
        return unsupported(inst, "unsupported UIMAD 64-bit addend");
      const uint64_t product =
          unsigned_product ? static_cast<uint64_t>(a) * static_cast<uint64_t>(b)
                           : static_cast<uint64_t>(
                                 static_cast<int64_t>(static_cast<int32_t>(a)) *
                                 static_cast<int64_t>(static_cast<int32_t>(b)));
      const uint64_t result = product + c;
      if (high)
        state.write_uniform_register(inst.operands[0].index,
                                     static_cast<uint32_t>(result >> 32));
      else
        write_uniform_register_pair(state, inst.operands[0].index, result);
    } else {
      uint32_t c = 0;
      if (!read_u32_operand(inst.operands[3], state, 0, c))
        return unsupported(inst, "unsupported UIMAD addend");
      state.write_uniform_register(inst.operands[0].index, a * b + c);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_iabs(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported IABS operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported IABS source");
    const int32_t signed_source = static_cast<int32_t>(source);
    const uint32_t result = signed_source < 0 ? uint32_t{0} - source : source;
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_popc(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported POPC operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported POPC source");
    state.write_register(lane, inst.operands[0].index,
                         static_cast<uint32_t>(__builtin_popcount(source)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_flo(const instruction &inst, warp_state &state,
                        execution_context &) {
  const bool uniform = inst.opcode == "UFLO.U32";
  if (!has_typed_operands(inst, 2) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && inst.has_guard))
    return unsupported(inst, "unsupported FLO.U32 operand form");
  const auto find_leading_one = [](uint32_t source) {
    return source == 0 ? 0xffffffffu
                       : 31u - static_cast<uint32_t>(__builtin_clz(source));
  };
  if (uniform) {
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, 0, source))
      return unsupported(inst, "unsupported UFLO.U32 source");
    state.write_uniform_register(inst.operands[0].index,
                                 find_leading_one(source));
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t source = 0;
      if (!read_u32_operand(inst.operands[1], state, lane, source))
        return unsupported(inst, "unsupported FLO.U32 source");
      state.write_register(lane, inst.operands[0].index,
                           find_leading_one(source));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

bool extend_comparison(const instruction &inst, uint64_t a, uint64_t b,
                       bool unsigned_compare, bool extension, bool &compared) {
  if (inst.opcode.find(".EQ.") != std::string::npos) {
    compared = compared && extension;
    return true;
  }
  if (inst.opcode.find(".NE.") != std::string::npos) {
    compared = compared || extension;
    return true;
  }

  const bool equal = a == b;
  const bool less = unsigned_compare
                        ? a < b
                        : static_cast<int32_t>(a) < static_cast<int32_t>(b);
  const bool greater = unsigned_compare
                           ? a > b
                           : static_cast<int32_t>(a) > static_cast<int32_t>(b);
  if (inst.opcode.find(".LT.") != std::string::npos ||
      inst.opcode.find(".LE.") != std::string::npos) {
    compared = less || (equal && extension);
    return true;
  }
  if (inst.opcode.find(".GT.") != std::string::npos ||
      inst.opcode.find(".GE.") != std::string::npos) {
    compared = greater || (equal && extension);
    return true;
  }
  return false;
}

step_result execute_isetp(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool extended = inst.opcode.find(".EX") != std::string::npos;
  if (!has_typed_operands(inst, extended ? 6 : 5))
    return unsupported(inst, extended ? "expected six typed operands"
                                      : "expected five typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kPredicate);
  const operand *second_destination =
      get_operand(inst, 1, operand_kind::kPredicate);
  const bool combine_and = inst.opcode.find(".AND") != std::string::npos;
  const bool combine_or = inst.opcode.find(".OR") != std::string::npos;
  const bool combine_xor = inst.opcode.find(".XOR") != std::string::npos;
  if (destination == nullptr || second_destination == nullptr ||
      second_destination->index != kTruePredicate ||
      (!combine_and && !combine_or && !combine_xor))
    return unsupported(
        inst, "only ISETP AND/OR/XOR with a discarded second output is "
              "implemented");
  const bool wide_compare = inst.opcode.find(".U64.") != std::string::npos;
  const bool unsigned_compare =
      wide_compare || inst.opcode.find(".U32.") != std::string::npos;
  enum class comparison { kEq, kNe, kLt, kLe, kGt, kGe };
  comparison operation;
  if (inst.opcode.find(".EQ.") != std::string::npos)
    operation = comparison::kEq;
  else if (inst.opcode.find(".NE.") != std::string::npos)
    operation = comparison::kNe;
  else if (inst.opcode.find(".LT.") != std::string::npos)
    operation = comparison::kLt;
  else if (inst.opcode.find(".LE.") != std::string::npos)
    operation = comparison::kLe;
  else if (inst.opcode.find(".GT.") != std::string::npos)
    operation = comparison::kGt;
  else if (inst.opcode.find(".GE.") != std::string::npos)
    operation = comparison::kGe;
  else
    return unsupported(inst, "unknown ISETP comparison");

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    bool input = false;
    if (!read_predicate_operand(inst.operands[4], state, lane, input))
      return unsupported(inst, "unsupported ISETP source");
    bool extension = false;
    if (extended &&
        !read_predicate_operand(inst.operands[5], state, lane, extension))
      return unsupported(inst, "unsupported extended ISETP source");
    uint64_t a = 0;
    uint64_t b = 0;
    if (wide_compare) {
      if (!read_u64_operand(inst.operands[2], state, lane, a) ||
          !read_u64_operand(inst.operands[3], state, lane, b))
        return unsupported(inst, "unsupported ISETP.U64 source");
    } else {
      uint32_t a32 = 0;
      uint32_t b32 = 0;
      if (!read_u32_operand(inst.operands[2], state, lane, a32) ||
          !read_u32_operand(inst.operands[3], state, lane, b32))
        return unsupported(inst, "unsupported ISETP source");
      a = a32;
      b = b32;
    }
    bool compared = false;
    if (operation == comparison::kEq)
      compared = a == b;
    else if (operation == comparison::kNe)
      compared = a != b;
    else if (unsigned_compare && operation == comparison::kLt)
      compared = a < b;
    else if (unsigned_compare && operation == comparison::kLe)
      compared = a <= b;
    else if (unsigned_compare && operation == comparison::kGt)
      compared = a > b;
    else if (unsigned_compare && operation == comparison::kGe)
      compared = a >= b;
    else if (operation == comparison::kLt)
      compared = static_cast<int32_t>(a) < static_cast<int32_t>(b);
    else if (operation == comparison::kLe)
      compared = static_cast<int32_t>(a) <= static_cast<int32_t>(b);
    else if (operation == comparison::kGt)
      compared = static_cast<int32_t>(a) > static_cast<int32_t>(b);
    else
      compared = static_cast<int32_t>(a) >= static_cast<int32_t>(b);
    if (extended &&
        !extend_comparison(inst, a, b, unsigned_compare, extension, compared))
      return unsupported(inst, "unsupported extended ISETP comparison");
    const bool result = combine_xor  ? compared != input
                        : combine_or ? compared || input
                                     : compared && input;
    state.write_predicate(lane, destination->index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_uisetp(const instruction &inst, warp_state &state,
                           execution_context &) {
  enum class comparison { kEq, kNe, kLt, kLe, kGt, kGe };
  comparison operation;
  if (inst.opcode.find(".EQ.") != std::string::npos)
    operation = comparison::kEq;
  else if (inst.opcode.find(".NE.") != std::string::npos)
    operation = comparison::kNe;
  else if (inst.opcode.find(".LT.") != std::string::npos)
    operation = comparison::kLt;
  else if (inst.opcode.find(".LE.") != std::string::npos)
    operation = comparison::kLe;
  else if (inst.opcode.find(".GT.") != std::string::npos)
    operation = comparison::kGt;
  else if (inst.opcode.find(".GE.") != std::string::npos)
    operation = comparison::kGe;
  else
    return unsupported(inst, "unknown UISETP comparison");
  const bool wide_compare = inst.opcode.find(".U64.") != std::string::npos;
  const bool unsigned_compare =
      wide_compare || inst.opcode.find(".U32.") != std::string::npos;
  const bool combine_or = inst.opcode.find(".OR") != std::string::npos;
  const bool extended = inst.opcode.find(".EX") != std::string::npos;
  if (!has_typed_operands(inst, extended ? 6 : 5) ||
      inst.operands[0].kind != operand_kind::kUniformPredicate ||
      inst.operands[1].kind != operand_kind::kUniformPredicate ||
      inst.operands[1].index != kTruePredicate ||
      (inst.opcode.find(".AND") == std::string::npos && !combine_or) ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported UISETP operand form");
  if (any_lane_executes(inst, state)) {
    bool input = false;
    if (!read_predicate_operand(inst.operands[4], state, 0, input))
      return unsupported(inst, "unsupported UISETP source");
    bool extension = false;
    if (extended &&
        !read_predicate_operand(inst.operands[5], state, 0, extension))
      return unsupported(inst, "unsupported extended UISETP source");
    uint64_t a = 0;
    uint64_t b = 0;
    if (wide_compare) {
      if (!read_u64_operand(inst.operands[2], state, 0, a) ||
          !read_u64_operand(inst.operands[3], state, 0, b))
        return unsupported(inst, "unsupported UISETP.U64 source");
    } else {
      uint32_t a32 = 0;
      uint32_t b32 = 0;
      if (!read_u32_operand(inst.operands[2], state, 0, a32) ||
          !read_u32_operand(inst.operands[3], state, 0, b32))
        return unsupported(inst, "unsupported UISETP source");
      a = a32;
      b = b32;
    }
    bool compared = false;
    if (operation == comparison::kEq)
      compared = a == b;
    else if (operation == comparison::kNe)
      compared = a != b;
    else if (unsigned_compare && operation == comparison::kLt)
      compared = a < b;
    else if (unsigned_compare && operation == comparison::kLe)
      compared = a <= b;
    else if (unsigned_compare && operation == comparison::kGt)
      compared = a > b;
    else if (unsigned_compare && operation == comparison::kGe)
      compared = a >= b;
    else if (operation == comparison::kLt)
      compared = static_cast<int32_t>(a) < static_cast<int32_t>(b);
    else if (operation == comparison::kLe)
      compared = static_cast<int32_t>(a) <= static_cast<int32_t>(b);
    else if (operation == comparison::kGt)
      compared = static_cast<int32_t>(a) > static_cast<int32_t>(b);
    else
      compared = static_cast<int32_t>(a) >= static_cast<int32_t>(b);
    if (extended &&
        !extend_comparison(inst, a, b, unsigned_compare, extension, compared))
      return unsupported(inst, "unsupported extended UISETP comparison");
    state.write_uniform_predicate(inst.operands[0].index,
                                  combine_or ? compared || input
                                             : compared && input);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_lea(const instruction &inst, warp_state &state,
                        execution_context &) {
  const bool signed_high_carry =
      inst.opcode == "LEA.HI.X.SX32" || inst.opcode == "ULEA.HI.X.SX32";
  const bool high_carry = inst.opcode == "LEA.HI.X" || signed_high_carry;
  const bool uniform_high = inst.opcode == "ULEA.HI";
  const bool high = inst.opcode == "LEA.HI" || uniform_high || high_carry;
  const bool sx32 = inst.opcode == "LEA.HI.SX32" ||
                    inst.opcode == "ULEA.HI.SX32" || signed_high_carry;
  const bool uniform = inst.opcode.rfind("ULEA", 0) == 0;
  const bool uniform_high_carry =
      inst.opcode == "ULEA.HI.X" || inst.opcode == "ULEA.HI.X.SX32";
  const bool uniform_carry_output =
      inst.opcode == "ULEA" && inst.operands.size() == 5;
  const bool carry_output = inst.opcode == "LEA" && inst.operands.size() == 5;
  const size_t expected = signed_high_carry                        ? 5
                          : (uniform_high_carry || high_carry)     ? 6
                          : (uniform_carry_output || carry_output) ? 5
                          : high                                   ? 5
                                                                   : 4;
  if (!has_typed_operands(inst, expected))
    return unsupported(inst, "unexpected LEA operand count");
  if ((uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister))
    return unsupported(inst, "unsupported LEA destination");
  const size_t first_source = (uniform_carry_output || carry_output) ? 2 : 1;
  if (uniform &&
      (inst.operands[first_source].kind == operand_kind::kRegister ||
       inst.operands[first_source + 1].kind == operand_kind::kRegister))
    return unsupported(inst, "ULEA source is not warp uniform");
  const size_t shift_position = signed_high_carry ? 3
                                : (uniform_high_carry || high_carry)
                                    ? 4
                                    : expected - 1;
  const operand *shift =
      get_operand(inst, shift_position, operand_kind::kImmediate);
  if (shift == nullptr || shift->immediate < 0 || shift->immediate >= 32)
    return unsupported(inst, "invalid LEA shift amount");
  const auto compute = [&](unsigned lane, uint32_t &result) {
    uint32_t a = 0;
    uint32_t b = 0;
    if (!read_u32_operand(inst.operands[first_source], state, lane, a) ||
        !read_u32_operand(inst.operands[first_source + 1], state, lane, b))
      return false;
    const unsigned amount = static_cast<unsigned>(shift->immediate);
    if (uniform_high_carry || high_carry) {
      uint32_t c = 0;
      bool carry = false;
      const size_t predicate_position = signed_high_carry ? 4 : 5;
      if ((!signed_high_carry &&
           !read_u32_operand(inst.operands[3], state, lane, c)) ||
          !read_predicate_operand(inst.operands[predicate_position], state,
                                  lane, carry))
        return false;
      const uint32_t carry_fragment =
          amount == 0 ? 0
          : signed_high_carry
              ? static_cast<uint32_t>(static_cast<int32_t>(a) >> (32 - amount))
              : a >> (32 - amount);
      result = carry_fragment + b + c + static_cast<uint32_t>(carry);
    } else if (high) {
      uint32_t c = 0;
      if (!read_u32_operand(inst.operands[3], state, lane, c))
        return false;
      // LEA.HI adds the carry fragment selected from the first source to
      // the second and third sources.  Triton's signed power-of-two cdiv
      // lowering exposes the mapping directly:
      //   sign = value >> 31
      //   bias = sign >> (32 - shift)
      //   adjusted = value + bias
      // becomes
      //   LEA.HI adjusted, sign, value, RZ, shift
      const uint32_t carry_fragment = amount == 0 ? 0 : a >> (32 - amount);
      result = carry_fragment + b + c;
    } else if (sx32) {
      const unsigned right = 32 - amount;
      // Widen before shifting: shift=0 selects the sign-extension word,
      // and shifting an int32_t by 32 would be undefined C++ behavior.
      const int64_t extended = static_cast<int32_t>(a);
      const uint32_t upper = static_cast<uint32_t>(extended >> right);
      result = upper + b;
    } else {
      result = (a << amount) + b;
    }
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported ULEA operand form");
      if (uniform_carry_output) {
        const operand *carry =
            get_operand(inst, 1, operand_kind::kUniformPredicate);
        uint32_t a = 0;
        uint32_t b = 0;
        if (carry == nullptr ||
            !read_u32_operand(inst.operands[2], state, 0, a) ||
            !read_u32_operand(inst.operands[3], state, 0, b))
          return unsupported(inst, "unsupported ULEA carry output");
        const unsigned amount =
            static_cast<unsigned>(inst.operands[4].immediate);
        const uint64_t wide = (static_cast<uint64_t>(a) << amount) + b;
        state.write_uniform_predicate(carry->index, (wide >> 32) != 0);
      }
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported LEA operand form");
      if (carry_output) {
        const operand *carry = get_operand(inst, 1, operand_kind::kPredicate);
        uint32_t a = 0;
        uint32_t b = 0;
        if (carry == nullptr ||
            !read_u32_operand(inst.operands[2], state, lane, a) ||
            !read_u32_operand(inst.operands[3], state, lane, b))
          return unsupported(inst, "unsupported LEA carry output");
        const unsigned amount =
            static_cast<unsigned>(inst.operands[4].immediate);
        const uint64_t wide = (static_cast<uint64_t>(a) << amount) + b;
        state.write_predicate(lane, carry->index, (wide >> 32) != 0);
      }
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_gxt(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[2].kind != operand_kind::kImmediate ||
      inst.operands[2].immediate <= 0 || inst.operands[2].immediate > 32)
    return unsupported(inst, "unsupported GXT operand form");
  const bool uniform =
      inst.opcode == "USGXT" || inst.opcode.rfind("USGXT.", 0) == 0;
  const bool unsigned_result = inst.opcode.find(".U32") != std::string::npos;
  if ((uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && inst.operands[1].kind == operand_kind::kRegister))
    return unsupported(inst, "unsupported GXT register class");
  const unsigned width = static_cast<unsigned>(inst.operands[2].immediate);
  const uint32_t mask = width == 32 ? 0xffffffffu : (uint32_t{1} << width) - 1;
  const auto compute = [&](unsigned lane, uint32_t &result) {
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, source))
      return false;
    result = source & mask;
    if (!unsigned_result && width < 32 &&
        (result & (uint32_t{1} << (width - 1))) != 0)
      result |= ~mask;
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported USGXT source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported SGXT source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_vimnmx(const instruction &inst, warp_state &state,
                           execution_context &) {
  const bool uniform = inst.opcode.rfind("UVIMNMX.", 0) == 0;
  if (!has_typed_operands(inst, 4) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   inst.operands[2].kind == operand_kind::kRegister ||
                   inst.operands[3].kind == operand_kind::kPredicate ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported VIMNMX operand form");
  const auto compute = [&](unsigned lane, uint32_t &result) {
    uint32_t a = 0;
    uint32_t b = 0;
    bool choose_minimum = false;
    if (!read_u32_operand(inst.operands[1], state, lane, a) ||
        !read_u32_operand(inst.operands[2], state, lane, b) ||
        !read_predicate_operand(inst.operands[3], state, lane, choose_minimum))
      return false;
    if (inst.opcode == "VIMNMX.U32" || inst.opcode == "UVIMNMX.U32") {
      result = choose_minimum ? std::min(a, b) : std::max(a, b);
    } else {
      const int32_t signed_a = static_cast<int32_t>(a);
      const int32_t signed_b = static_cast<int32_t>(b);
      result =
          static_cast<uint32_t>(choose_minimum ? std::min(signed_a, signed_b)
                                               : std::max(signed_a, signed_b));
    }
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported UVIMNMX source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported VIMNMX source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_viaddmnmx(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 5) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported VIADDMNMX operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t bound = 0;
    bool choose_minimum = false;
    if (!read_u32_operand(inst.operands[1], state, lane, a) ||
        !read_u32_operand(inst.operands[2], state, lane, b) ||
        !read_u32_operand(inst.operands[3], state, lane, bound) ||
        !read_predicate_operand(inst.operands[4], state, lane, choose_minimum))
      return unsupported(inst, "unsupported VIADDMNMX source");
    const uint32_t sum = a + b;
    uint32_t result = 0;
    if (inst.opcode == "VIADDMNMX.U32") {
      result = choose_minimum ? std::min(sum, bound) : std::max(sum, bound);
    } else {
      const int32_t signed_sum = static_cast<int32_t>(sum);
      const int32_t signed_bound = static_cast<int32_t>(bound);
      result = static_cast<uint32_t>(choose_minimum
                                         ? std::min(signed_sum, signed_bound)
                                         : std::max(signed_sum, signed_bound));
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_p2r(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[1].kind != operand_kind::kRegister ||
      inst.operands[2].kind != operand_kind::kImmediate ||
      inst.operands[2].immediate < 0 || inst.operands[2].immediate > 0x7f)
    return unsupported(inst, "unsupported P2R operand form");
  const uint32_t mask = static_cast<uint32_t>(inst.operands[2].immediate);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t value = state.read_register(lane, inst.operands[1].index);
    for (unsigned predicate = 0; predicate < 7; ++predicate) {
      const uint32_t bit = 1u << predicate;
      if ((mask & bit) == 0)
        continue;
      if (state.read_predicate(lane, predicate))
        value |= bit;
      else
        value &= ~bit;
    }
    state.write_register(lane, inst.operands[0].index, value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_r2p(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kPredicateRegisterFile ||
      inst.operands[1].kind != operand_kind::kRegister ||
      inst.operands[2].kind != operand_kind::kImmediate ||
      inst.operands[2].immediate < 0 || inst.operands[2].immediate > 0x7f)
    return unsupported(inst, "unsupported R2P operand form");
  const uint32_t mask = static_cast<uint32_t>(inst.operands[2].immediate);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t value = state.read_register(lane, inst.operands[1].index);
    for (unsigned predicate = 0; predicate < 7; ++predicate) {
      const uint32_t bit = 1u << predicate;
      if ((mask & bit) != 0)
        state.write_predicate(lane, predicate, (value & bit) != 0);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_usel(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported USEL operand form");
  if (any_lane_executes(inst, state)) {
    bool select_a = false;
    if (!read_predicate_operand(inst.operands[3], state, 0, select_a))
      return unsupported(inst, "unsupported USEL predicate");
    if (inst.opcode == "USEL.64") {
      uint64_t a = 0;
      uint64_t b = 0;
      if (!read_u64_operand(inst.operands[1], state, 0, a) ||
          !read_u64_operand(inst.operands[2], state, 0, b))
        return unsupported(inst, "unsupported USEL.64 source");
      write_uniform_register_pair(state, inst.operands[0].index,
                                  select_a ? a : b);
    } else {
      uint32_t a = 0;
      uint32_t b = 0;
      if (!read_u32_operand(inst.operands[1], state, 0, a) ||
          !read_u32_operand(inst.operands[2], state, 0, b))
        return unsupported(inst, "unsupported USEL source");
      state.write_uniform_register(inst.operands[0].index, select_a ? a : b);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_sel(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported SEL operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    bool select_a = false;
    if (!read_predicate_operand(inst.operands[3], state, lane, select_a))
      return unsupported(inst, "unsupported SEL source");
    if (inst.opcode == "SEL.64") {
      uint64_t a = 0;
      uint64_t b = 0;
      if (!read_u64_operand(inst.operands[1], state, lane, a) ||
          !read_u64_operand(inst.operands[2], state, lane, b))
        return unsupported(inst, "unsupported SEL.64 source");
      write_register_pair(state, lane, inst.operands[0].index,
                          select_a ? a : b);
    } else {
      uint32_t a = 0;
      uint32_t b = 0;
      if (!read_u32_operand(inst.operands[1], state, lane, a) ||
          !read_u32_operand(inst.operands[2], state, lane, b))
        return unsupported(inst, "unsupported SEL source");
      state.write_register(lane, inst.operands[0].index, select_a ? a : b);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_integer_semantics(frontend &target) {
  target.register_semantics("IADD", execute_iadd);
  target.register_semantics("VIADD", execute_iadd);
  target.register_semantics("IADD3", execute_iadd3);
  target.register_semantics("IADD3.X", execute_iadd3x);
  target.register_semantics("IADD.64", execute_iadd64);
  target.register_semantics("UIADD3", execute_uiadd3);
  target.register_semantics("UIADD3.64", execute_uiadd3);
  target.register_semantics("UIADD3.X", execute_uiadd3x);
  target.register_semantics("SHF.L.U32", execute_shf);
  target.register_semantics("SHF.R.S32.HI", execute_shf);
  target.register_semantics("SHF.R.U32.HI", execute_shf);
  target.register_semantics("SHF.R.U64", execute_shf);
  target.register_semantics("SHF.L.U64.HI", execute_shf);
  target.register_semantics("USHF.L.U32", execute_shf);
  target.register_semantics("USHF.R.U32.HI", execute_shf);
  target.register_semantics("USHF.R.S32.HI", execute_shf);
  target.register_semantics("USHF.L.U64.HI", execute_shf);
  target.register_semantics("LOP3.LUT", execute_lop3);
  target.register_semantics("ULOP3.LUT", execute_lop3);
  target.register_semantics("PLOP3.LUT", execute_plop3);
  target.register_semantics("UPLOP3.LUT", execute_plop3);
  target.register_semantics("PRMT", execute_prmt);
  target.register_semantics("UPRMT", execute_prmt);
  target.register_semantics("IMAD", execute_imad);
  target.register_semantics("IMAD.X", execute_imad);
  target.register_semantics("IMAD.IADD", execute_imad);
  target.register_semantics("IMAD.MOV", execute_imad);
  target.register_semantics("IMAD.MOV.U32", execute_imad);
  target.register_semantics("IMAD.U32", execute_imad);
  target.register_semantics("IMAD.SHL.U32", execute_imad);
  target.register_semantics("IMAD.HI", execute_imad);
  target.register_semantics("IMAD.HI.U32", execute_imad);
  target.register_semantics("IMAD.WIDE", execute_imad);
  target.register_semantics("IMAD.WIDE.U32", execute_imad);
  target.register_semantics("UIMAD", execute_uimad);
  target.register_semantics("UIMAD.HI.U32", execute_uimad);
  target.register_semantics("UIMAD.WIDE", execute_uimad);
  target.register_semantics("UIMAD.WIDE.U32", execute_uimad);
  target.register_semantics("IABS", execute_iabs);
  target.register_semantics("POPC", execute_popc);
  target.register_semantics("FLO.U32", execute_flo);
  target.register_semantics("UFLO.U32", execute_flo);
  target.register_semantics("ISETP.EQ.AND", execute_isetp);
  target.register_semantics("ISETP.EQ.OR", execute_isetp);
  target.register_semantics("ISETP.EQ.U32.AND", execute_isetp);
  target.register_semantics("ISETP.EQ.U32.OR", execute_isetp);
  target.register_semantics("ISETP.NE.U32.AND", execute_isetp);
  target.register_semantics("ISETP.NE.AND", execute_isetp);
  target.register_semantics("ISETP.NE.XOR", execute_isetp);
  target.register_semantics("ISETP.NE.AND.EX", execute_isetp);
  target.register_semantics("ISETP.NE.U32.AND.EX", execute_isetp);
  target.register_semantics("ISETP.EQ.OR.EX", execute_isetp);
  target.register_semantics("ISETP.NE.U64.AND", execute_isetp);
  target.register_semantics("ISETP.GE.U64.AND", execute_isetp);
  target.register_semantics("ISETP.LT.U32.AND", execute_isetp);
  target.register_semantics("ISETP.LT.AND", execute_isetp);
  target.register_semantics("ISETP.LE.AND", execute_isetp);
  target.register_semantics("ISETP.GT.U32.AND", execute_isetp);
  target.register_semantics("ISETP.GT.U32.AND.EX", execute_isetp);
  target.register_semantics("ISETP.GT.AND", execute_isetp);
  target.register_semantics("ISETP.GE.U32.AND", execute_isetp);
  target.register_semantics("ISETP.GE.U32.AND.EX", execute_isetp);
  target.register_semantics("ISETP.GE.AND", execute_isetp);
  target.register_semantics("ISETP.GE.OR", execute_isetp);
  target.register_semantics("ISETP.GT.OR", execute_isetp);
  target.register_semantics("ISETP.LT.OR", execute_isetp);
  target.register_semantics("ISETP.LT.U32.OR.EX", execute_isetp);
  target.register_semantics("ISETP.NE.OR", execute_isetp);
  target.register_semantics("ISETP.GT.U32.OR", execute_isetp);
  target.register_semantics("ISETP.NE.U32.OR", execute_isetp);
  target.register_semantics("UISETP.NE.U32.AND", execute_uisetp);
  target.register_semantics("UISETP.NE.U64.AND", execute_uisetp);
  target.register_semantics("UISETP.NE.AND", execute_uisetp);
  target.register_semantics("UISETP.NE.AND.EX", execute_uisetp);
  target.register_semantics("UISETP.EQ.OR.EX", execute_uisetp);
  target.register_semantics("UISETP.NE.OR", execute_uisetp);
  target.register_semantics("UISETP.NE.U32.OR", execute_uisetp);
  target.register_semantics("UISETP.EQ.AND", execute_uisetp);
  target.register_semantics("UISETP.EQ.U32.AND", execute_uisetp);
  target.register_semantics("UISETP.EQ.U32.OR", execute_uisetp);
  target.register_semantics("UISETP.LT.AND", execute_uisetp);
  target.register_semantics("UISETP.GT.U32.AND", execute_uisetp);
  target.register_semantics("UISETP.GT.U32.OR", execute_uisetp);
  target.register_semantics("UISETP.GT.AND", execute_uisetp);
  target.register_semantics("UISETP.GE.AND", execute_uisetp);
  target.register_semantics("UISETP.GE.U32.AND", execute_uisetp);
  target.register_semantics("UISETP.GE.U64.AND", execute_uisetp);
  target.register_semantics("LEA", execute_lea);
  target.register_semantics("LEA.HI", execute_lea);
  target.register_semantics("LEA.HI.X", execute_lea);
  target.register_semantics("LEA.HI.SX32", execute_lea);
  target.register_semantics("LEA.HI.X.SX32", execute_lea);
  target.register_semantics("ULEA", execute_lea);
  target.register_semantics("ULEA.HI", execute_lea);
  target.register_semantics("ULEA.HI.SX32", execute_lea);
  target.register_semantics("ULEA.HI.X", execute_lea);
  target.register_semantics("ULEA.HI.X.SX32", execute_lea);
  target.register_semantics("SGXT.U32", execute_gxt);
  target.register_semantics("USGXT.U32", execute_gxt);
  target.register_semantics("SGXT", execute_gxt);
  target.register_semantics("USGXT", execute_gxt);
  target.register_semantics("VIMNMX", execute_vimnmx);
  target.register_semantics("VIMNMX.S32", execute_vimnmx);
  target.register_semantics("VIMNMX.U32", execute_vimnmx);
  target.register_semantics("UVIMNMX.S32", execute_vimnmx);
  target.register_semantics("UVIMNMX.U32", execute_vimnmx);
  target.register_semantics("VIADDMNMX", execute_viaddmnmx);
  target.register_semantics("VIADDMNMX.S32", execute_viaddmnmx);
  target.register_semantics("VIADDMNMX.U32", execute_viaddmnmx);
  target.register_semantics("P2R", execute_p2r);
  target.register_semantics("R2P", execute_r2p);
  target.register_semantics("SEL", execute_sel);
  target.register_semantics("SEL.64", execute_sel);
  target.register_semantics("USEL", execute_usel);
  target.register_semantics("USEL.64", execute_usel);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
