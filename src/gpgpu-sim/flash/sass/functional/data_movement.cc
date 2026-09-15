#include "internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.9: data movement and conversion.

uint64_t read_clock(const execution_context &context, bool global_timer) {
  // Timing execution supplies the architectural issue cycle. Functional-only
  // execution retains its deterministic instruction-count fallback.
  const uint64_t cycle = context.architectural_cycle_valid
                             ? context.architectural_cycle
                             : context.instructions_executed;
  if (global_timer && context.architectural_cycle_valid &&
      context.architectural_core_frequency_hz != 0)
    return static_cast<uint64_t>(
        (static_cast<long double>(cycle) * 1000000000.0L) /
        static_cast<long double>(context.architectural_core_frequency_hz));
  return cycle;
}

step_result execute_move(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand &destination = inst.operands[0];
  const operand &source = inst.operands[1];

  if (inst.opcode == "UMOV" || inst.opcode == "UMOV.64") {
    if (destination.kind != operand_kind::kUniformRegister ||
        source.kind == operand_kind::kRegister ||
        (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
      return unsupported(inst, "unsupported UMOV operand form");
    if (any_lane_executes(inst, state)) {
      if (inst.opcode == "UMOV.64") {
        uint64_t value = 0;
        if (destination.index >= kUniformRegisters - 1 ||
            !read_u64_operand(source, state, 0, value))
          return unsupported(inst, "unsupported UMOV.64 source");
        write_uniform_register_pair(state, destination.index, value);
      } else {
        uint32_t value = 0;
        if (!read_u32_operand(source, state, 0, value))
          return unsupported(inst, "unsupported UMOV source");
        state.write_uniform_register(destination.index, value);
      }
    }
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }

  if (destination.kind != operand_kind::kRegister)
    return unsupported(inst, "MOV destination must be a vector register");
  const bool wide = inst.opcode == "MOV.64";
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    if (wide) {
      uint64_t value = 0;
      if (!read_u64_operand(source, state, lane, value))
        return unsupported(inst, "unsupported MOV.64 source");
      write_register_pair(state, lane, destination.index, value);
    } else {
      uint32_t value = 0;
      if (!read_u32_operand(source, state, lane, value))
        return unsupported(inst, "unsupported MOV source");
      state.write_register(lane, destination.index, value);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_r2ur(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool reports_nonuniform = inst.operands.size() == 3;
  if (!has_typed_operands(inst, reports_nonuniform ? 3 : 2))
    return unsupported(inst, "expected two or three typed operands");
  const operand *nonuniform =
      reports_nonuniform ? get_operand(inst, 0, operand_kind::kPredicate)
                         : nullptr;
  const operand *destination = get_operand(inst, reports_nonuniform ? 1 : 0,
                                           operand_kind::kUniformRegister);
  const operand *source =
      get_operand(inst, reports_nonuniform ? 2 : 1, operand_kind::kRegister);
  if ((reports_nonuniform && nonuniform == nullptr) || destination == nullptr ||
      source == nullptr ||
      (inst.has_guard && inst.guard.kind != operand_kind::kPredicate &&
       inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported R2UR operand form");

  bool found = false;
  uint32_t uniform_value = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t value = state.read_register(lane, source->index);
    if (!found) {
      uniform_value = value;
      found = true;
    } else if (value != uniform_value) {
      return unsupported(inst,
                         "R2UR source is not uniform across active lanes");
    }
  }
  if (found) {
    state.write_uniform_register(destination->index, uniform_value);
    if (reports_nonuniform) {
      // The closed kernels reach these status forms with one uniform source
      // value across their executing lanes. In that case the nonuniform flag
      // is false; .OR retains a flag accumulated by an earlier component.
      // A genuinely nonuniform source is rejected above until its grouping
      // behavior is independently validated.
      const bool combine_or = inst.opcode == "R2UR.OR";
      for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
        if ((state.active_mask & (1u << lane)) == 0 ||
            !lane_executes(inst, state, lane))
          continue;
        const bool previous =
            combine_or && state.read_predicate(lane, nonuniform->index);
        state.write_predicate(lane, nonuniform->index, previous);
      }
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_up2ur(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[1].kind != operand_kind::kUniformPredicateRegisterFile ||
      inst.operands[2].kind != operand_kind::kUniformRegister ||
      inst.operands[3].kind != operand_kind::kImmediate ||
      inst.operands[3].immediate < 0 || inst.operands[3].immediate > 0x7f ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported UP2UR operand form");

  if (any_lane_executes(inst, state)) {
    uint32_t value = state.read_uniform_register(inst.operands[2].index);
    const uint32_t mask = static_cast<uint32_t>(inst.operands[3].immediate);
    for (unsigned predicate = 0; predicate < 7; ++predicate) {
      const uint32_t bit = 1u << predicate;
      if ((mask & bit) == 0)
        continue;
      if (state.read_uniform_predicate(predicate))
        value |= bit;
      else
        value &= ~bit;
    }
    state.write_uniform_register(inst.operands[0].index, value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_s2r(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *special = get_operand(inst, 1, operand_kind::kSpecialRegister);
  if (destination == nullptr || special == nullptr)
    return unsupported(inst, "unsupported S2R operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane)) {
      uint32_t value = 0;
      if (!read_special_register(*special, context, lane, value))
        return unsupported(inst, "unsupported S2R special register");
      state.write_register(lane, destination->index, value);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_s2ur(const instruction &inst, warp_state &state,
                         execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination =
      get_operand(inst, 0, operand_kind::kUniformRegister);
  const operand *special = get_operand(inst, 1, operand_kind::kSpecialRegister);
  if (destination == nullptr || special == nullptr ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported S2UR operand form");
  if (any_lane_executes(inst, state)) {
    uint32_t value = 0;
    if (special->text == "SR_CLOCKLO") {
      value = static_cast<uint32_t>(read_clock(context, false));
    } else if (!read_special_register(*special, context, 0, value) ||
               special->text.rfind("SR_TID.", 0) == 0 ||
               special->text == "SR_LANEID") {
      return unsupported(inst, "S2UR source is not warp uniform");
    }
    state.write_uniform_register(destination->index, value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_cs2r(const instruction &inst, warp_state &state,
                         execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *special = get_operand(inst, 1, operand_kind::kSpecialRegister);
  if (destination == nullptr || special == nullptr ||
      (special->text != "SRZ" && special->text != "SR_GLOBALTIMERLO" &&
       special->text != "SR_CLOCKLO"))
    return unsupported(inst, "unsupported CS2R operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint64_t value =
        special->text == "SRZ"
            ? 0
            : read_clock(context, special->text == "SR_GLOBALTIMERLO");
    if (inst.opcode == "CS2R.32")
      state.write_register(lane, destination->index,
                           static_cast<uint32_t>(value));
    else
      write_register_pair(state, lane, destination->index, value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_cs2ur(const instruction &inst, warp_state &state,
                          execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination =
      get_operand(inst, 0, operand_kind::kUniformRegister);
  const operand *special = get_operand(inst, 1, operand_kind::kSpecialRegister);
  if (destination == nullptr || special == nullptr ||
      (special->text != "SRZ" && special->text != "SR_GLOBALTIMERLO" &&
       special->text != "SR_CLOCKLO") ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported CS2UR operand form");
  if (any_lane_executes(inst, state)) {
    const uint64_t value =
        special->text == "SRZ"
            ? 0
            : read_clock(context, special->text == "SR_GLOBALTIMERLO");
    if (inst.opcode == "CS2UR.32")
      state.write_uniform_register(destination->index,
                                   static_cast<uint32_t>(value));
    else
      write_uniform_register_pair(state, destination->index, value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_shfl(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 5) ||
      inst.operands[0].kind != operand_kind::kPredicate ||
      inst.operands[1].kind != operand_kind::kRegister ||
      inst.operands[2].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported SHFL operand form");
  const bool butterfly = inst.opcode == "SHFL.BFLY";
  if (!butterfly && inst.opcode != "SHFL.IDX")
    return unsupported(inst, "unsupported SHFL mode");
  std::array<uint32_t, kWarpLanes> source{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane)
    source[lane] = state.read_register(lane, inst.operands[2].index);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t lane_selector = 0;
    uint32_t configuration = 0;
    if (!read_u32_operand(inst.operands[3], state, lane, lane_selector) ||
        !read_u32_operand(inst.operands[4], state, lane, configuration))
      return unsupported(inst, "unsupported SHFL lane selector");
    const unsigned segment_mask = (configuration >> 8) & 0x1f;
    const unsigned clamp = configuration & 0x1f;
    const unsigned min_lane = lane & segment_mask;
    const unsigned max_lane = (lane & segment_mask) | (clamp & ~segment_mask);
    unsigned source_lane =
        butterfly
            ? lane ^ (lane_selector & 0x1f)
            : (lane & segment_mask) | (lane_selector & 0x1f & ~segment_mask);
    const bool valid = source_lane >= min_lane && source_lane <= max_lane &&
                       source_lane < kWarpLanes &&
                       (state.active_mask & (1u << source_lane)) != 0;
    if (!valid)
      source_lane = lane;
    state.write_register(lane, inst.operands[1].index, source[source_lane]);
    state.write_predicate(lane, inst.operands[0].index, valid);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_data_movement_semantics(frontend &target) {
  target.register_semantics("MOV", execute_move);
  target.register_semantics("MOV.64", execute_move);
  target.register_semantics("UMOV", execute_move);
  target.register_semantics("UMOV.64", execute_move);
  target.register_semantics("R2UR", execute_r2ur);
  target.register_semantics("R2UR.OR", execute_r2ur);
  target.register_semantics("R2UR.BROADCAST", execute_r2ur);
  target.register_semantics("UP2UR", execute_up2ur);
  target.register_semantics("S2R", execute_s2r);
  target.register_semantics("S2UR", execute_s2ur);
  target.register_semantics("CS2R", execute_cs2r);
  target.register_semantics("CS2R.32", execute_cs2r);
  target.register_semantics("CS2UR", execute_cs2ur);
  target.register_semantics("CS2UR.32", execute_cs2ur);
  target.register_semantics("SHFL.BFLY", execute_shfl);
  target.register_semantics("SHFL.IDX", execute_shfl);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
