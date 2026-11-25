#include "../../bfloat_arithmetic.h"
#include "../../fp32_fma.h"
#include "../../half_sign.h"
#include "../decode/fp32x2_operand.h"
#include "../decode/operand_parser.h"
#include "../decode/scaled_unpack_operand.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// Scalar and packed floating-point arithmetic, comparisons and SFU operations.

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

step_result execute_fp32x2(const instruction &inst, warp_state &state,
                           execution_context &) {
  // Keep these native type selectors opaque to ordinary scalar handlers.
  // A source independently selects a scalar broadcast or consecutive pair.
  const bool fused = inst.opcode == "FFMA2" || inst.opcode == "FFMA2.FTZ";
  const bool multiply = inst.opcode == "FMUL2" || inst.opcode == "FMUL2.FTZ";
  const bool ftz = inst.opcode.find(".FTZ") != std::string::npos;
  const unsigned count = fused ? 3 : 2;
  if (inst.operands.size() != count + 1 ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].index >= kZeroRegister - 1)
    return unsupported(inst, "unsupported FP32x2 destination");
  fp32x2_source sources[3];
  for (unsigned i = 0; i < count; ++i)
    if (!decode_fp32x2_source(inst.operands[i + 1].text, sources[i]))
      return unsupported(inst, "unsupported FP32x2 source selector");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float values[3][2];
    // Read both results' sources before writing an aliased destination pair.
    for (unsigned i = 0; i < count; ++i)
      for (unsigned word = 0; word < 2; ++word) {
        operand source = sources[i].value;
        source.index +=
            sources[i].paired ? (sources[i].reversed ? 1 - word : word) : 0;
        if (!read_f32_operand(source, state, lane, values[i][word]))
          return unsupported(inst, "unsupported FP32x2 source value");
      }
    for (unsigned word = 0; word < 2; ++word) {
      if (ftz)
        for (unsigned i = 0; i < count; ++i)
          values[i][word] = flush_subnormal(values[i][word]);
      float result =
          fused ? std::fma(values[0][word], values[1][word], values[2][word])
          : multiply ? values[0][word] * values[1][word]
                     : values[0][word] + values[1][word];
      if (ftz)
        result = flush_subnormal(result);
      state.write_register(lane, inst.operands[0].index + word,
                           float_bits(result));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Native SM120 exponent-grid and random-bit qualification: FCHK selects
// the division slow path outside these biased-exponent limits.
step_result execute_fchk(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kPredicate)
    return unsupported(inst, "FCHK requires predicate and two FP32 sources");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a, b;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported FCHK source");
    const int ea = (float_bits(a) >> 23) & 255;
    const int eb = (float_bits(b) >> 23) & 255;
    const bool slow = ea < 25 || ea > 254 || eb < 2 || eb > 251 ||
                      ea - eb < -124 || ea - eb > 126;
    state.write_predicate(lane, inst.operands[0].index, slow);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Use the same rounding primitive as PTX. fesetround is thread-local and
// the helper restores the caller's mode before returning.
int fp32_rounding(const instruction &inst) {
  if (inst.opcode.find(".RM") != std::string::npos)
    return FE_DOWNWARD;
  if (inst.opcode.find(".RP") != std::string::npos)
    return FE_UPWARD;
  if (inst.opcode.find(".RZ") != std::string::npos)
    return FE_TOWARDZERO;
  return FE_TONEAREST;
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
    float result = rounded_f32_fma(a, b, c, fp32_rounding(inst), false);
    if (ftz)
      result = flush_subnormal(result);
    if (inst.opcode.find(".SAT") != std::string::npos)
      result = !(result > 0.0f) ? 0.0f : std::min(result, 1.0f);
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
                        unsigned lane, float &low, float &high,
                        bool ftz = false) {
  uint32_t packed = 0;
  if (source.kind == operand_kind::kRegister)
    packed = state.read_register(lane, source.index);
  else if (source.kind == operand_kind::kUniformRegister)
    packed = state.read_uniform_register(source.index);
  else if (source.kind == operand_kind::kImmediate)
    packed = static_cast<uint32_t>(source.immediate);
  else
    return false;
  if (source.wide || source.bitwise_complement)
    return false;
  auto extract = [&](unsigned half) {
    const auto selected =
        source.half_selection == operand_half_selection::kLowReplicate ? 0
        : source.half_selection == operand_half_selection::kHighReplicate
            ? 1
            : half;
    auto bits = half_sign_bits(packed >> (16 * selected), source.absolute,
                               source.negated);
    if (ftz)
      bits = half_flush_subnormal(bits);
    return half_float(bits);
  };
  low = extract(0);
  high = extract(1);
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
  const bool ftz = inst.opcode == "HADD2.FTZ";
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a_low = 0.0f;
    float a_high = 0.0f;
    float b_low = 0.0f;
    float b_high = 0.0f;
    if (!read_half2_operand(inst.operands[1], state, lane, a_low, a_high,
                            ftz) ||
        !read_half2_operand(inst.operands[2], state, lane, b_low, b_high, ftz))
      return unsupported(inst, "unsupported packed FP16 add source");
    uint16_t low = float_half_rn(float_bits(a_low + b_low));
    uint16_t high = float_half_rn(float_bits(a_high + b_high));
    if (ftz) {
      low = half_flush_subnormal(low);
      high = half_flush_subnormal(high);
    }
    state.write_register(lane, inst.operands[0].index,
                         low | (static_cast<uint32_t>(high) << 16));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bfloat_fma_register(const instruction &inst,
                                        warp_state &state) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected three packed BF16 FMA sources");
  for (unsigned i = 0; i < 4; ++i) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.wide ||
        op.bitwise_complement ||
        (i == 0 && (op.negated || op.absolute ||
                    op.half_selection != operand_half_selection::kNone)))
      return unsupported(inst, "unsupported packed BF16 FMA register modifier");
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      float values[3];
      for (unsigned i = 0; i < 3; ++i) {
        const auto &op = inst.operands[i + 1];
        const unsigned selected =
            op.half_selection == operand_half_selection::kLowReplicate ? 0
            : op.half_selection == operand_half_selection::kHighReplicate
                ? 1
                : half;
        const float value = bfloat_float(half_sign_bits(
            state.read_register(lane, op.index) >> (16 * selected), op.absolute,
            op.negated));
        values[i] = value;
      }
      result |= uint32_t(bf16_fma_rn(values[0], values[1], values[2]))
                << (16 * half);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bfloat_fma_immediate(const instruction &inst,
                                         warp_state &state,
                                         execution_context &) {
  if (inst.operands.size() == 4)
    return execute_bfloat_fma_register(inst, state);
  if (!has_typed_operands(inst, 5))
    return unsupported(inst, "expected packed FMA immediate form");
  for (unsigned i : {0u, 1u, 4u}) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.negated || op.absolute ||
        op.bitwise_complement || op.wide ||
        op.half_selection != operand_half_selection::kNone)
      return unsupported(inst, "packed FMA requires plain registers");
  }
  for (unsigned i : {2u, 3u})
    if (inst.operands[i].kind != operand_kind::kImmediate &&
        inst.operands[i].kind != operand_kind::kFloatImmediate)
      return unsupported(inst, "only literal BF16 multipliers supported");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float high, low;
    if (!read_f32_operand(inst.operands[2], state, lane, high) ||
        !read_f32_operand(inst.operands[3], state, lane, low))
      return unsupported(inst, "invalid packed FMA literals");
    const uint32_t a = state.read_register(lane, inst.operands[1].index);
    const uint32_t c = state.read_register(lane, inst.operands[4].index);
    const uint32_t lo =
        bf16_fma_rn(bfloat_float(a & 0xffff), low, bfloat_float(c & 0xffff));
    const uint32_t hi =
        bf16_fma_rn(bfloat_float(a >> 16), high, bfloat_float(c >> 16));
    state.write_register(lane, inst.operands[0].index, lo | (hi << 16));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bfloat_minmax2(const instruction &inst, warp_state &state,
                                   execution_context &) {
  const unsigned sources = inst.opcode == "VHMNMX.BF16_V2" ? 3 : 2;
  const bool immediate = sources == 2 && inst.operands.size() == 5;
  if (!has_typed_operands(inst, sources + 2 + unsigned(immediate)) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].half_selection != operand_half_selection::kNone)
    return unsupported(inst, "unsupported packed BF16 min/max form");
  for (unsigned i = 1; i <= (immediate ? 1 : sources); ++i) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.wide || op.bitwise_complement)
      return unsupported(inst, "packed BF16 min/max requires register sources");
  }
  if (immediate)
    for (unsigned i : {2u, 3u})
      if (inst.operands[i].kind != operand_kind::kImmediate &&
          inst.operands[i].kind != operand_kind::kFloatImmediate)
        return unsupported(inst, "expected split BF16 min/max literals");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    bool minimum;
    if (!read_predicate_operand(
            inst.operands[sources + 1 + unsigned(immediate)], state, lane,
            minimum))
      return unsupported(inst, "invalid min/max predicate");
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      float values[3];
      for (unsigned i = 0; i < sources; ++i) {
        if (immediate && i == 1) {
          float literal;
          if (!read_f32_operand(inst.operands[half == 0 ? 3 : 2], state, lane,
                                literal))
            return unsupported(inst, "invalid BF16 min/max literal");
          values[i] = bfloat_float(float_bfloat_rn(literal));
          continue;
        }
        const auto &op = inst.operands[i + 1];
        unsigned selected =
            op.half_selection == operand_half_selection::kLowReplicate ? 0
            : op.half_selection == operand_half_selection::kHighReplicate
                ? 1
                : half;
        const float value = bfloat_float(half_sign_bits(
            state.read_register(lane, op.index) >> (16 * selected), op.absolute,
            op.negated));
        if (sources == 3 && std::isnan(value)) {
          std::ostringstream detail;
          detail << "unqualified three-input BF16 NaN at lane " << lane
                 << " source " << i << " half " << half << ": bits=0x"
                 << std::hex << (float_bits(value) >> 16);
          return unsupported(inst, detail.str());
        }
        values[i] = value;
      }
      float value = values[0];
      for (unsigned i = 1; i < sources; ++i) {
        // CUDA min/max orders -0 below +0 irrespective of operand order.
        if (value == 0 && values[i] == 0) {
          const bool negative =
              minimum ? (std::signbit(value) || std::signbit(values[i]))
                      : (std::signbit(value) && std::signbit(values[i]));
          value = negative ? -0.0f : 0.0f;
        } else {
          // Host fmin/fmax may propagate signaling NaNs. Native min/max
          // without .NaN selects the numeric operand for either NaN kind.
          if (std::isnan(value))
            value = values[i];
          else if (!std::isnan(values[i]))
            value = minimum ? std::fmin(value, values[i])
                            : std::fmax(value, values[i]);
        }
      }
      result |= uint32_t(float_bfloat_rn(value)) << (half * 16);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bfloat_mul2(const instruction &inst, warp_state &state,
                                execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].half_selection != operand_half_selection::kNone)
    return unsupported(inst, "expected BF16 pair destination and two sources");
  for (unsigned i : {1u, 2u}) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.wide || op.bitwise_complement)
      return unsupported(inst,
                         "packed BF16 multiply requires register sources");
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      float values[2];
      for (unsigned i = 0; i < 2; ++i) {
        const auto &op = inst.operands[i + 1];
        unsigned selected =
            op.half_selection == operand_half_selection::kLowReplicate ? 0
            : op.half_selection == operand_half_selection::kHighReplicate
                ? 1
                : half;
        const float value = bfloat_float(half_sign_bits(
            state.read_register(lane, op.index) >> (16 * selected), op.absolute,
            op.negated));
        values[i] = value;
      }
      result |= uint32_t(bf16_mul_rn(values[0], values[1])) << (half * 16);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bfloat_add2(const instruction &inst, warp_state &state,
                                execution_context &) {
  if (!has_typed_operands(inst, 3) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].half_selection != operand_half_selection::kNone)
    return unsupported(inst, "expected BF16 pair destination and two sources");
  for (unsigned i : {1u, 2u}) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.wide || op.bitwise_complement)
      return unsupported(inst, "invalid BF16 add source");
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      float values[2];
      for (unsigned i = 0; i < 2; ++i) {
        const auto &op = inst.operands[i + 1];
        unsigned selected =
            op.half_selection == operand_half_selection::kLowReplicate ? 0
            : op.half_selection == operand_half_selection::kHighReplicate
                ? 1
                : half;
        const float value = bfloat_float(half_sign_bits(
            state.read_register(lane, op.index) >> (16 * selected), op.absolute,
            op.negated));
        values[i] = value;
      }
      result |= uint32_t(bf16_add_rn(values[0], values[1])) << (16 * half);
    }
    state.write_register(lane, inst.operands[0].index, result);
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
    // A same-sign zero preserves FMUL's signed zero through the FMA helper.
    result = multiply ? rounded_f32_fma(a, b, std::copysign(0.0f, a * b),
                                        fp32_rounding(inst), false)
                      : rounded_f32_fma(a, 1.0f, b, fp32_rounding(inst), false);
    if (ftz)
      result = flush_subnormal(result);
    if (inst.opcode.find(".SAT") != std::string::npos)
      result = !(result > 0.0f) ? 0.0f : std::min(result, 1.0f);
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
  const bool ternary = inst.opcode.rfind("FMNMX3", 0) == 0;
  if (!has_typed_operands(inst, ternary ? 5 : 4) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported FMNMX operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    bool choose_minimum = false;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b) ||
        (ternary && !read_f32_operand(inst.operands[3], state, lane, c)) ||
        !read_predicate_operand(inst.operands[ternary ? 4 : 3], state, lane,
                                choose_minimum))
      return unsupported(inst, "unsupported FMNMX source");
    if (inst.opcode.find(".FTZ") != std::string::npos) {
      a = flush_subnormal(a);
      b = flush_subnormal(b);
      c = flush_subnormal(c);
    }
    float result = choose_minimum ? std::fmin(a, b) : std::fmax(a, b);
    if (ternary)
      result = choose_minimum ? std::fmin(result, c) : std::fmax(result, c);
    if (inst.opcode == "FMNMX3.NAN" &&
        (std::isnan(a) || std::isnan(b) || std::isnan(c)))
      result = bits_float(0x7fffffff);
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

// Comparison and Boolean combination are shared by FSET and FSETP.
bool fp_compare(const instruction &inst, float a, float b) {
  if (inst.opcode.find(".FTZ") != std::string::npos) {
    a = flush_subnormal(a);
    b = flush_subnormal(b);
  }
  const bool nan = std::isnan(a) || std::isnan(b);
  const auto has = [&](const char *cmp) {
    return inst.opcode.find(std::string(".") + cmp + ".") != std::string::npos;
  };
  if (has("NAN"))
    return nan;
  if (has("NUM"))
    return !nan;
  if (has("LTU"))
    return nan || a < b;
  if (has("LEU"))
    return nan || a <= b;
  if (has("GTU"))
    return nan || a > b;
  if (has("GEU"))
    return nan || a >= b;
  if (has("EQU"))
    return nan || a == b;
  if (has("NEU"))
    return nan || a != b;
  if (has("LT"))
    return !nan && a < b;
  if (has("LE"))
    return !nan && a <= b;
  if (has("GT"))
    return !nan && a > b;
  if (has("GE"))
    return !nan && a >= b;
  if (has("EQ"))
    return !nan && a == b;
  return !nan && a != b; // NE, qualified by registration
}
bool fp_combine(const instruction &inst, bool cmp, bool input) {
  if (inst.opcode.find(".XOR") != std::string::npos)
    return cmp != input;
  if (inst.opcode.find(".OR") != std::string::npos)
    return cmp || input;
  return cmp && input;
}
step_result execute_fset_mask(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst,
                       "expected mask destination and comparison sources");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a, b;
    bool input;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b) ||
        !read_predicate_operand(inst.operands[3], state, lane, input))
      return unsupported(inst, "unsupported mask comparison source");
    state.write_register(
        lane, inst.operands[0].index,
        fp_combine(inst, fp_compare(inst, a, b), input) ? 0xffffffffu : 0u);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
step_result execute_fsetp(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool uniform = inst.opcode.rfind("UFSETP.", 0) == 0;
  const auto kind =
      uniform ? operand_kind::kUniformPredicate : operand_kind::kPredicate;
  if (!has_typed_operands(inst, 5) || inst.operands[0].kind != kind ||
      inst.operands[1].kind != kind ||
      (uniform && (inst.operands[2].kind == operand_kind::kRegister ||
                   inst.operands[3].kind == operand_kind::kRegister ||
                   inst.operands[4].kind != operand_kind::kUniformPredicate ||
                   (inst.has_guard &&
                    inst.guard.kind != operand_kind::kUniformPredicate))))
    return unsupported(inst, "unsupported FSETP operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a, b;
    bool input;
    if (!read_f32_operand(inst.operands[2], state, lane, a) ||
        !read_f32_operand(inst.operands[3], state, lane, b) ||
        !read_predicate_operand(inst.operands[4], state, lane, input))
      return unsupported(inst, "unsupported FSETP source");
    const bool cmp = fp_compare(inst, a, b);
    const bool first = fp_combine(inst, cmp, input),
               second = fp_combine(inst, !cmp, input);
    if (uniform) {
      state.write_uniform_predicate(inst.operands[0].index, first);
      state.write_uniform_predicate(inst.operands[1].index, second);
      break;
    }
    state.write_predicate(lane, inst.operands[0].index, first);
    state.write_predicate(lane, inst.operands[1].index, second);
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

step_result execute_mufu_tanh(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.TANH operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.TANH source");
    // Mathematical functional approximation, not a bit-exact SFU polynomial.
    state.write_register(lane, inst.operands[0].index,
                         float_bits(std::tanh(source)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_mufu_rsq(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.RSQ operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.RSQ source");
    // Functional approximation, not a bit-exact SFU polynomial model.
    state.write_register(lane, inst.operands[0].index,
                         float_bits(1.0f / std::sqrt(flush_subnormal(source))));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_mufu_sqrt(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported MUFU.SQRT operand form");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0.0f;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported MUFU.SQRT source");
    // Functional approximation, not a bit-exact SFU polynomial model.
    state.write_register(lane, inst.operands[0].index,
                         float_bits(std::sqrt(flush_subnormal(source))));
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
  register_float_conversion_semantics(target);
  for (const std::string base : {"FSETP", "UFSETP", "FSET.BF"})
    for (const std::string cmp :
         {"LT", "LE", "GT", "GE", "EQ", "NE", "LTU", "LEU", "GTU", "GEU", "EQU",
          "NEU", "NUM", "NAN"})
      for (const std::string ftz : {"", ".FTZ"})
        for (const std::string combine : {"AND", "OR", "XOR"})
          target.register_semantics(base + "." + cmp + ftz + "." + combine,
                                    base == "FSET.BF" ? execute_fset_mask
                                                      : execute_fsetp);

  for (const std::string base :
       {"FADD", "FMUL", "FFMA", "UFADD", "UFMUL", "UFFMA"})
    for (const std::string ftz : {"", ".FTZ"})
      for (const std::string round : {"", ".RM", ".RP", ".RZ"})
        for (const std::string sat : {"", ".SAT"})
          target.register_semantics(base + ftz + round + sat,
                                    base.find("FFMA") != std::string::npos
                                        ? execute_ffma
                                        : execute_binary_float);

  target.register_semantics("MUFU.RCP", execute_mufu_rcp);

  register_mixed_precision_semantics(target);

  target.register_semantics("FCHK", execute_fchk);
  target.register_semantics("FFMA2", execute_fp32x2);
  target.register_semantics("FFMA2.FTZ", execute_fp32x2);
  target.register_semantics("FADD2", execute_fp32x2);
  target.register_semantics("FADD2.FTZ", execute_fp32x2);
  target.register_semantics("FMUL2", execute_fp32x2);
  target.register_semantics("FMUL2.FTZ", execute_fp32x2);
  target.register_semantics("HFMA2", execute_hfma2);
  target.register_semantics("HFMA2.MMA", execute_hfma2);
  target.register_semantics("HADD2", execute_hadd2);
  target.register_semantics("HADD2.FTZ", execute_hadd2);
  target.register_semantics("HADD2.BF16_V2", execute_bfloat_add2);
  target.register_semantics("HMUL2.BF16_V2", execute_bfloat_mul2);
  target.register_semantics("HMNMX2.BF16_V2", execute_bfloat_minmax2);
  target.register_semantics("VHMNMX.BF16_V2", execute_bfloat_minmax2);
  target.register_semantics("HFMA2.BF16_V2", execute_bfloat_fma_immediate);
  target.register_semantics("HADD2.F32", execute_hadd2_f32_conversion);
  target.register_semantics("FMNMX", execute_fmnmx);
  target.register_semantics("FMNMX.FTZ", execute_fmnmx);
  target.register_semantics("FMNMX3.FTZ", execute_fmnmx);
  target.register_semantics("FMNMX3", execute_fmnmx);
  target.register_semantics("FMNMX3.NAN", execute_fmnmx);

  target.register_semantics("FSEL", execute_fsel);
  target.register_semantics("UFSEL", execute_fsel);
  target.register_semantics("MUFU.EX2", execute_mufu_ex2);
  target.register_semantics("MUFU.LG2", execute_mufu_lg2);
  target.register_semantics("MUFU.RSQ", execute_mufu_rsq);
  target.register_semantics("MUFU.SQRT", execute_mufu_sqrt);
  target.register_semantics("MUFU.TANH", execute_mufu_tanh);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
