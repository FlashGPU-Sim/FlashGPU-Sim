#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.3-9.7.5: floating-point and mixed-precision conversion.

float flush_subnormal(float value) {
  return std::fpclassify(value) == FP_SUBNORMAL ? std::copysign(0.0f, value)
                                                : value;
}

step_result execute_i2f(const instruction &inst, warp_state &state,
                        execution_context &) {
  const bool uniform = inst.opcode.rfind("UI2F.", 0) == 0;
  if (!has_typed_operands(inst, 2) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported I2F operand form");
  const auto convert = [&](unsigned lane, uint32_t &result_bits) {
    if (inst.opcode == "I2F.U64.RP" || inst.opcode == "UI2F.U64.RP") {
      uint64_t source = 0;
      if (!read_u64_operand(inst.operands[1], state, lane, source))
        return false;
      const long double exact = static_cast<long double>(source);
      float result = static_cast<float>(exact);
      if (static_cast<long double>(result) < exact)
        result = std::nextafter(result, INFINITY);
      result_bits = float_bits(result);
      return true;
    }
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, source))
      return false;
    if (inst.opcode == "I2F.U8" || inst.opcode == "UI2F.U8") {
      result_bits =
          float_bits(static_cast<float>(static_cast<uint8_t>(source)));
      return true;
    }
    if (inst.opcode == "I2F.U16" || inst.opcode == "I2F.U16.RZ" ||
        inst.opcode == "UI2F.U16" || inst.opcode == "UI2F.U16.RZ") {
      result_bits =
          float_bits(static_cast<float>(static_cast<uint16_t>(source)));
      return true;
    }
    const bool unsigned_source =
        inst.opcode == "I2F.U32.RP" || inst.opcode == "UI2F.U32.RP";
    const double exact =
        unsigned_source ? static_cast<double>(source)
                        : static_cast<double>(static_cast<int32_t>(source));
    float result = static_cast<float>(exact);
    if (static_cast<double>(result) < exact)
      result = std::nextafter(result, INFINITY);
    result_bits = float_bits(result);
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!convert(0, result))
        return unsupported(inst, "unsupported UI2F source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!convert(lane, result))
        return unsupported(inst, "unsupported I2F source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_i2fp_f32(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported I2FP.F32 operand form");
  const bool signed_source = inst.opcode.find(".S32") != std::string::npos;
  const bool round_to_zero =
      inst.opcode.size() >= 3 &&
      inst.opcode.compare(inst.opcode.size() - 3, 3, ".RZ") == 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t source = 0;
    if (!read_u32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported I2FP.F32 source");
    const double exact = signed_source
                             ? static_cast<double>(static_cast<int32_t>(source))
                             : static_cast<double>(source);
    float result = static_cast<float>(exact);
    if (round_to_zero) {
      const double rounded = static_cast<double>(result);
      if ((exact > 0.0 && rounded > exact) || (exact < 0.0 && rounded < exact))
        result = std::nextafter(result, 0.0f);
    }
    state.write_register(lane, inst.operands[0].index, float_bits(result));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_mufu_rcp(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.RCP operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.RCP source");
    // MUFU.RCP is approximate, but a correctly rounded host reciprocal stays
    // inside its error contract and preserves the sign of ptxas' integer-
    // division refinement remainder. A fixed ULP bias does not: for example,
    // SM120 returns the correctly rounded 0x3bc0c0c1 for 1 / 170.
    const float result = 1.0f / source;
    state.write_register(lane, inst.operands[0].index, float_bits(result));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_f2i(const instruction &inst, warp_state &state,
                        execution_context &) {
  const bool uniform = inst.opcode.rfind("UF2I.", 0) == 0;
  if (!has_typed_operands(inst, 2) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported F2I operand form");
  const auto convert = [&](unsigned lane, uint32_t &result) {
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return false;
    if (std::fpclassify(source) == FP_SUBNORMAL)
      source = 0.0f;
    result = 0;
    if (!std::isnan(source) && source > 0.0f) {
      constexpr double kUint32Limit = 4294967296.0;
      result = static_cast<double>(source) >= kUint32Limit
                   ? 0xffffffffu
                   : static_cast<uint32_t>(source);
    }
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!convert(0, result))
        return unsupported(inst, "unsupported UF2I source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!convert(lane, result))
        return unsupported(inst, "unsupported F2I source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_f2fp_pack_ab(const instruction &inst, warp_state &state,
                                 execution_context &) {
  if (!has_typed_operands(inst, 3))
    return unsupported(inst, "expected destination and two FP32 sources");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  if (destination == nullptr)
    return unsupported(inst, "packed FP16 destination must be a register");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a = 0.0f;
    float b = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported packed FP16 source");
    // PACK_AB places B in the low halfword and A in the high halfword.
    const uint32_t packed =
        float_half_rn(float_bits(b)) |
        (static_cast<uint32_t>(float_half_rn(float_bits(a))) << 16);
    state.write_register(lane, destination->index, packed);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ffma(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool uniform = inst.opcode.rfind("UFFMA", 0) == 0;
  if (!has_typed_operands(inst, 4) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   inst.operands[2].kind == operand_kind::kRegister ||
                   inst.operands[3].kind == operand_kind::kRegister ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "expected four typed operands");
  const auto compute = [&](unsigned lane, uint32_t &result_bits) {
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b) ||
        !read_f32_operand(inst.operands[3], state, lane, c))
      return false;
    const bool ftz = inst.opcode.find(".FTZ") != std::string::npos;
    if (ftz) {
      a = flush_subnormal(a);
      b = flush_subnormal(b);
      c = flush_subnormal(c);
    }
    float result = std::fma(a, b, c);
    if (ftz)
      result = flush_subnormal(result);
    result_bits = float_bits(result);
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported UFFMA source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported FFMA source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

bool read_half2_operand(const operand &source, const warp_state &state,
                        unsigned lane, float &low, float &high) {
  uint32_t packed = 0;
  if (source.kind == operand_kind::kRegister)
    packed = state.read_register(lane, source.index);
  else if (source.kind == operand_kind::kUniformRegister)
    packed = state.read_uniform_register(source.index);
  else if (source.kind == operand_kind::kImmediate)
    packed = static_cast<uint32_t>(source.immediate);
  else
    return false;
  low = half_float(static_cast<uint16_t>(packed));
  high = half_float(static_cast<uint16_t>(packed >> 16));
  if (source.absolute) {
    low = std::fabs(low);
    high = std::fabs(high);
  }
  if (source.negated) {
    low = -low;
    high = -high;
  }
  return true;
}

step_result execute_hfma2(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 5) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported packed FP16 FMA operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a_low = 0.0f;
    float a_high = 0.0f;
    float b_low = 0.0f;
    float b_high = 0.0f;
    float c_high = 0.0f;
    float c_low = 0.0f;
    if (!read_half2_operand(inst.operands[1], state, lane, a_low, a_high) ||
        !read_half2_operand(inst.operands[2], state, lane, b_low, b_high) ||
        !read_f32_operand(inst.operands[3], state, lane, c_high) ||
        !read_f32_operand(inst.operands[4], state, lane, c_low))
      return unsupported(inst, "unsupported packed FP16 FMA source");
    const uint16_t low =
        float_half_rn(float_bits(std::fma(a_low, b_low, c_low)));
    const uint16_t high =
        float_half_rn(float_bits(std::fma(a_high, b_high, c_high)));
    state.write_register(lane, inst.operands[0].index,
                         low | (static_cast<uint32_t>(high) << 16));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hadd2(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported packed FP16 add operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a_low = 0.0f;
    float a_high = 0.0f;
    float b_low = 0.0f;
    float b_high = 0.0f;
    if (!read_half2_operand(inst.operands[1], state, lane, a_low, a_high) ||
        !read_half2_operand(inst.operands[2], state, lane, b_low, b_high))
      return unsupported(inst, "unsupported packed FP16 add source");
    const uint16_t low = float_half_rn(float_bits(a_low + b_low));
    const uint16_t high = float_half_rn(float_bits(a_high + b_high));
    state.write_register(lane, inst.operands[0].index,
                         low | (static_cast<uint32_t>(high) << 16));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hadd2_f32_conversion(const instruction &inst,
                                         warp_state &state,
                                         execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[1].kind != operand_kind::kRegister ||
      inst.operands[1].index != kZeroRegister || !inst.operands[1].negated ||
      inst.operands[2].kind != operand_kind::kRegister ||
      inst.operands[2].half_selection == operand_half_selection::kNone)
    return unsupported(inst, "expected half-to-FP32 conversion form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t packed = state.read_register(lane, inst.operands[2].index);
    const uint16_t selected =
        inst.operands[2].half_selection == operand_half_selection::kLowReplicate
            ? static_cast<uint16_t>(packed)
            : static_cast<uint16_t>(packed >> 16);
    state.write_register(lane, inst.operands[0].index,
                         float_bits(half_float(selected)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_binary_float(const instruction &inst, warp_state &state,
                                 execution_context &) {
  const bool uniform = inst.opcode.rfind("UF", 0) == 0;
  if (!has_typed_operands(inst, 3) ||
      (uniform && inst.operands[0].kind != operand_kind::kUniformRegister) ||
      (!uniform && inst.operands[0].kind != operand_kind::kRegister) ||
      (uniform && (inst.operands[1].kind == operand_kind::kRegister ||
                   inst.operands[2].kind == operand_kind::kRegister ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported binary FP32 operand form");
  const auto compute = [&](unsigned lane, uint32_t &result_bits) {
    float a = 0.0f;
    float b = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b))
      return false;
    const bool ftz = inst.opcode.find(".FTZ") != std::string::npos;
    if (ftz) {
      a = flush_subnormal(a);
      b = flush_subnormal(b);
    }
    float result = 0.0f;
    const bool multiply =
        inst.opcode.rfind("FMUL", 0) == 0 || inst.opcode.rfind("UFMUL", 0) == 0;
    if (multiply && inst.opcode.find(".RZ") != std::string::npos) {
      const double exact = static_cast<double>(a) * static_cast<double>(b);
      result = static_cast<float>(exact);
      if (std::isfinite(exact) &&
          std::fabs(static_cast<double>(result)) > std::fabs(exact))
        result = std::nextafter(result, 0.0f);
    } else {
      result = multiply ? a * b : a + b;
    }
    if (ftz)
      result = flush_subnormal(result);
    result_bits = float_bits(result);
    return true;
  };
  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!compute(0, result))
        return unsupported(inst, "unsupported uniform binary FP32 source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!compute(lane, result))
        return unsupported(inst, "unsupported binary FP32 source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_fmnmx(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported FMNMX operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a = 0.0f;
    float b = 0.0f;
    bool choose_minimum = false;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b) ||
        !read_predicate_operand(inst.operands[3], state, lane, choose_minimum))
      return unsupported(inst, "unsupported FMNMX source");
    if (inst.opcode.find(".FTZ") != std::string::npos) {
      a = flush_subnormal(a);
      b = flush_subnormal(b);
    }
    const float result = choose_minimum ? std::fmin(a, b) : std::fmax(a, b);
    state.write_register(lane, inst.operands[0].index, float_bits(result));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_fsel(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool uniform = inst.opcode == "UFSEL";
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != (uniform ? operand_kind::kUniformRegister
                                        : operand_kind::kRegister) ||
      (uniform && inst.has_guard &&
       inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported FSEL operand form");

  const auto select = [&](unsigned lane, uint32_t &result) {
    float a = 0.0f;
    float b = 0.0f;
    bool select_a = false;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b) ||
        !read_predicate_operand(inst.operands[3], state, lane, select_a))
      return false;
    result = float_bits(select_a ? a : b);
    return true;
  };

  if (uniform) {
    if (any_lane_executes(inst, state)) {
      uint32_t result = 0;
      if (!select(0, result))
        return unsupported(inst, "unsupported UFSEL source");
      state.write_uniform_register(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint32_t result = 0;
      if (!select(lane, result))
        return unsupported(inst, "unsupported FSEL source");
      state.write_register(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_fsetp(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool uniform = inst.opcode.rfind("UFSETP.", 0) == 0;
  if (!has_typed_operands(inst, 5) ||
      inst.operands[0].kind != (uniform ? operand_kind::kUniformPredicate
                                        : operand_kind::kPredicate) ||
      inst.operands[1].kind != (uniform ? operand_kind::kUniformPredicate
                                        : operand_kind::kPredicate) ||
      inst.operands[1].index != kTruePredicate ||
      inst.opcode.find(".AND") == std::string::npos ||
      (uniform && inst.has_guard &&
       inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported FSETP operand form");

  const auto compare = [&](unsigned lane, bool &result) {
    float a = 0.0f;
    float b = 0.0f;
    bool input = false;
    if (!read_f32_operand(inst.operands[2], state, lane, a) ||
        !read_f32_operand(inst.operands[3], state, lane, b) ||
        !read_predicate_operand(inst.operands[4], state, lane, input))
      return false;
    if (inst.opcode.find(".FTZ") != std::string::npos) {
      a = flush_subnormal(a);
      b = flush_subnormal(b);
    }
    bool compared = false;
    if (inst.opcode.find(".GEU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a >= b;
    else if (inst.opcode.find(".NEU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a != b;
    else if (inst.opcode.find(".NE.") != std::string::npos)
      compared = !std::isnan(a) && !std::isnan(b) && a != b;
    else if (inst.opcode.find(".GT.") != std::string::npos)
      compared = !std::isnan(a) && !std::isnan(b) && a > b;
    else
      return false;
    result = compared && input;
    return true;
  };

  if (uniform) {
    if (any_lane_executes(inst, state)) {
      bool result = false;
      if (!compare(0, result))
        return unsupported(inst, "unsupported UFSETP source or comparison");
      state.write_uniform_predicate(inst.operands[0].index, result);
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      bool result = false;
      if (!compare(lane, result))
        return unsupported(inst, "unsupported FSETP source or comparison");
      state.write_predicate(lane, inst.operands[0].index, result);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_mufu_ex2(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.EX2 operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.EX2 source");
    state.write_register(lane, inst.operands[0].index,
                         float_bits(std::exp2(source)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_mufu_lg2(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.LG2 operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.LG2 source");
    state.write_register(lane, inst.operands[0].index,
                         float_bits(std::log2(source)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_float_semantics(frontend &target) {
  target.register_semantics("I2F.RP", execute_i2f);
  target.register_semantics("I2F.U8", execute_i2f);
  target.register_semantics("I2F.U16", execute_i2f);
  target.register_semantics("I2F.U16.RZ", execute_i2f);
  target.register_semantics("I2F.U32.RP", execute_i2f);
  target.register_semantics("I2F.U64.RP", execute_i2f);
  target.register_semantics("UI2F.U16", execute_i2f);
  target.register_semantics("UI2F.U16.RZ", execute_i2f);
  target.register_semantics("UI2F.RP", execute_i2f);
  target.register_semantics("UI2F.U8", execute_i2f);
  target.register_semantics("UI2F.U32.RP", execute_i2f);
  target.register_semantics("UI2F.U64.RP", execute_i2f);
  target.register_semantics("I2FP.F32.S32", execute_i2fp_f32);
  target.register_semantics("I2FP.F32.S32.RZ", execute_i2fp_f32);
  target.register_semantics("I2FP.F32.U32", execute_i2fp_f32);
  target.register_semantics("I2FP.F32.U32.RZ", execute_i2fp_f32);
  target.register_semantics("MUFU.RCP", execute_mufu_rcp);
  target.register_semantics("F2I.FTZ.U32.TRUNC.NTZ", execute_f2i);
  target.register_semantics("F2I.U32.TRUNC.NTZ", execute_f2i);
  target.register_semantics("UF2I.FTZ.U32.TRUNC.NTZ", execute_f2i);
  target.register_semantics("F2FP.F16.F32.PACK_AB", execute_f2fp_pack_ab);
  target.register_semantics("FFMA", execute_ffma);
  target.register_semantics("FFMA.FTZ", execute_ffma);
  target.register_semantics("UFFMA.FTZ", execute_ffma);
  target.register_semantics("HFMA2", execute_hfma2);
  target.register_semantics("HFMA2.MMA", execute_hfma2);
  target.register_semantics("HADD2", execute_hadd2);
  target.register_semantics("HADD2.F32", execute_hadd2_f32_conversion);
  target.register_semantics("FMUL", execute_binary_float);
  target.register_semantics("FMUL.FTZ", execute_binary_float);
  target.register_semantics("FMUL.RZ", execute_binary_float);
  target.register_semantics("UFMUL", execute_binary_float);
  target.register_semantics("UFMUL.FTZ", execute_binary_float);
  target.register_semantics("UFADD", execute_binary_float);
  target.register_semantics("UFADD.FTZ", execute_binary_float);
  target.register_semantics("FADD", execute_binary_float);
  target.register_semantics("FADD.FTZ", execute_binary_float);
  target.register_semantics("FMNMX", execute_fmnmx);
  target.register_semantics("FMNMX.FTZ", execute_fmnmx);
  target.register_semantics("FSEL", execute_fsel);
  target.register_semantics("UFSEL", execute_fsel);
  target.register_semantics("FSETP.GEU.AND", execute_fsetp);
  target.register_semantics("FSETP.NEU.AND", execute_fsetp);
  target.register_semantics("FSETP.NEU.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.NE.AND", execute_fsetp);
  target.register_semantics("FSETP.NE.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.GT.AND", execute_fsetp);
  target.register_semantics("UFSETP.GEU.AND", execute_fsetp);
  target.register_semantics("UFSETP.NEU.AND", execute_fsetp);
  target.register_semantics("UFSETP.NEU.FTZ.AND", execute_fsetp);
  target.register_semantics("UFSETP.NE.FTZ.AND", execute_fsetp);
  target.register_semantics("UFSETP.GT.AND", execute_fsetp);
  target.register_semantics("MUFU.EX2", execute_mufu_ex2);
  target.register_semantics("MUFU.LG2", execute_mufu_lg2);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
