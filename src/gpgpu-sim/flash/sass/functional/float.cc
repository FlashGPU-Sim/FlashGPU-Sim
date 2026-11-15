#include "../decode/fp32x2_operand.h"
#include "../decode/operand_parser.h"
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

step_result execute_f2i_u64(const instruction &inst, warp_state &state,
                            execution_context &) {
  const auto *destination = get_operand(inst, 0, operand_kind::kRegister);
  if (!has_typed_operands(inst, 2) || destination == nullptr ||
      (destination->index != kZeroRegister &&
       destination->index >= kZeroRegister - 1))
    return unsupported(inst, "unsupported F2I.U64 destination pair");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    float source = 0;
    if (!read_f32_operand(inst.operands[1], state, lane, source))
      return unsupported(inst, "unsupported F2I.U64 source");
    // Qualified finite range used by ptxas unsigned-division sequences.
    // Do not guess unmeasured SASS NaN/overflow conversion behavior.
    if (!std::isfinite(source) || source < 0 ||
        static_cast<double>(source) >= 18446744073709551616.0)
      return unsupported(inst, "F2I.U64 source outside qualified finite range");
    const uint64_t value = static_cast<uint64_t>(source);
    if (destination->index != kZeroRegister) {
      state.write_register(lane, destination->index, uint32_t(value));
      state.write_register(lane, destination->index + 1, uint32_t(value >> 32));
    }
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

uint16_t float_bfloat_rn(float value) {
  if (std::isnan(value))
    return 0x7fff;
  const uint32_t bits = float_bits(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

uint8_t float_e4m3_satfinite_rn(float value) {
  if (std::isnan(value))
    return 0x7f;
  const uint8_t sign = std::signbit(value) ? 0x80 : 0;
  const float magnitude = std::fabs(value);
  if (magnitude >= 448.0f)
    return sign | 0x7e;
  if (magnitude == 0)
    return sign;
  // Quantize in units of the current exponent's ULP, including subnormals.
  int exponent = 0;
  std::frexp(magnitude, &exponent);
  const int encoded_exponent = std::max(0, exponent + 6);
  const float scaled =
      std::ldexp(magnitude, 10 - std::max(1, encoded_exponent));
  unsigned rounded = static_cast<unsigned>(scaled);
  const float remainder = scaled - rounded;
  if (remainder > 0.5f || (remainder == 0.5f && (rounded & 1)))
    ++rounded;
  const unsigned encoded =
      encoded_exponent ? (encoded_exponent - 1) * 8 + rounded : rounded;
  return sign | static_cast<uint8_t>(encoded);
}

uint8_t float_e2m1_satfinite_rn(float value) {
  // CUDA's FP4 reference conversion: NaN saturates to positive MAXNORM.
  if (std::isnan(value))
    return 7;
  const uint8_t sign = std::signbit(value) ? 8 : 0;
  const float magnitude = std::fabs(value);
  constexpr float levels[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  unsigned code = 7;
  for (unsigned i = 0; i < 7; ++i) {
    const float midpoint = (levels[i] + levels[i + 1]) * 0.5f;
    if (magnitude < midpoint || (magnitude == midpoint && !(i & 1))) {
      code = i;
      break;
    }
  }
  return sign | code;
}

step_result execute_f2fp_e4m3(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[3].kind != operand_kind::kRegister)
    return unsupported(inst,
                       "FP8 PACK_AB_MERGE_C requires a register merge source");
  const auto &merge = inst.operands[3];
  if (merge.index > kZeroRegister || merge.negated || merge.absolute ||
      merge.bitwise_complement || merge.wide ||
      merge.half_selection != operand_half_selection::kNone ||
      merge.matrix_layout != operand_matrix_layout::kNone)
    return unsupported(inst, "unsupported FP8 merge source modifier");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a = 0, b = 0;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unsupported FP8 packed source");
    const bool fp4 = inst.opcode == "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C";
    const auto convert =
        fp4 ? float_e2m1_satfinite_rn : float_e4m3_satfinite_rn;
    state.write_register(
        lane, inst.operands[0].index,
        (state.read_register(lane, merge.index) << (fp4 ? 8 : 16)) |
            (uint32_t(convert(a)) << (fp4 ? 4 : 8)) | convert(b));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Native UE8M0 conversion encodes the magnitude, including negative inputs.
uint8_t float_ue8m0_satfinite_rp(float value) {
  const uint32_t bits = float_bits(value) & 0x7fffffff;
  const unsigned exponent = bits >> 23, fraction = bits & 0x7fffff;
  if (exponent == 255)
    return fraction ? 255 : 254;
  if (exponent == 0)
    return fraction > 0x400000 ? 1 : 0;
  return std::min(254u, exponent + (fraction != 0));
}

step_result execute_f2fp_ue8m0(const instruction &inst, warp_state &state,
                               execution_context &) {
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[3].kind != operand_kind::kRegister)
    return unsupported(inst,
                       "UE8M0 pack requires register destination and merge");
  const auto &merge = inst.operands[3];
  if (merge.negated || merge.absolute || merge.bitwise_complement ||
      merge.wide || merge.half_selection != operand_half_selection::kNone ||
      merge.matrix_layout != operand_matrix_layout::kNone)
    return unsupported(inst, "unqualified UE8M0 merge modifier");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a, b;
    if (!read_f32_operand(inst.operands[1], state, lane, a) ||
        !read_f32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "unqualified UE8M0 conversion source");
    const uint32_t result = (state.read_register(lane, merge.index) << 16) |
                            (uint32_t(float_ue8m0_satfinite_rp(a)) << 8) |
                            float_ue8m0_satfinite_rp(b);
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// The BF16 pair supplies the low two FP8 bytes; C's low half is merged
// into the upper half. Read both operands before an aliased destination write.
step_result execute_bfloat_to_e4m3(const instruction &inst, warp_state &state,
                                   execution_context &) {
  if (!has_typed_operands(inst, 3))
    return unsupported(inst, "expected BF16 pair and FP8 merge register");
  for (const auto &op : inst.operands)
    if (op.kind != operand_kind::kRegister || op.negated || op.absolute ||
        op.wide || op.bitwise_complement ||
        op.half_selection != operand_half_selection::kNone ||
        op.matrix_layout != operand_matrix_layout::kNone)
      return unsupported(inst, "unsupported BF16 to FP8 register modifier");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t source = state.read_register(lane, inst.operands[1].index);
    const uint32_t merge = state.read_register(lane, inst.operands[2].index);
    const bool fp4 = inst.opcode == "F2FP.SATFINITE.E2M1.BF16.UNPACK_B_MERGE_C";
    const auto convert =
        fp4 ? float_e2m1_satfinite_rn : float_e4m3_satfinite_rn;
    const uint32_t low = convert(bfloat_float(source & 0xffff));
    const uint32_t high = convert(bfloat_float(source >> 16));
    state.write_register(lane, inst.operands[0].index,
                         (merge << (fp4 ? 8 : 16)) | (high << (fp4 ? 4 : 8)) |
                             low);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// PTX fma.rn.f32.bf16: scalar low BF16 inputs, FP32 accumulator/result.
step_result execute_fhadd_bfloat(const instruction &inst, warp_state &state,
                                 execution_context &) {
  if (inst.operands.size() != 3 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid BF16-to-FP32 add operands");
  auto text = inst.operands[1].text;
  unsigned shift = 0;
  if (text.size() > 3 && text.compare(text.size() - 3, 3, ".H1") == 0) {
    shift = 16;
    text.resize(text.size() - 3);
  }
  const auto src = parse_operand(text);
  if (src.kind != operand_kind::kRegister || src.wide || src.negated ||
      src.absolute || src.bitwise_complement ||
      src.half_selection != operand_half_selection::kNone)
    return unsupported(inst, "invalid BF16 add source");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float b;
    if (!read_f32_operand(inst.operands[2], state, lane, b))
      return unsupported(inst, "invalid FP32 add source");
    const float a =
        bfloat_float((state.read_register(lane, src.index) >> shift) & 0xffff);
    state.write_register(lane, inst.operands[0].index, float_bits(a + b));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_fhfma_bfloat(const instruction &native, warp_state &state,
                                 execution_context &) {
  instruction inst = native;
  unsigned shifts[2] = {};
  if (inst.operands.size() == 4) {
    for (unsigned i = 1; i <= 2; ++i) {
      auto text = inst.operands[i].text;
      if (text.size() > 3 && text.compare(text.size() - 3, 3, ".H1") == 0) {
        shifts[i - 1] = 16;
        inst.operands[i] = parse_operand(text.substr(0, text.size() - 3));
      }
    }
    inst.operands_structured = std::all_of(
        inst.operands.begin(), inst.operands.end(),
        [](const operand &o) { return o.kind != operand_kind::kOpaque; });
  }
  if (!has_typed_operands(inst, 4) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported BF16-to-FP32 FMA form");
  for (unsigned i : {1u, 2u}) {
    const auto &op = inst.operands[i];
    if (op.kind != operand_kind::kRegister || op.wide || op.negated ||
        op.absolute || op.bitwise_complement ||
        op.half_selection != operand_half_selection::kNone)
      return unsupported(inst,
                         "BF16-to-FP32 FMA requires unmodified scalar inputs");
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float c;
    if (!read_f32_operand(inst.operands[3], state, lane, c))
      return unsupported(inst, "invalid FMA accumulator");
    const float a = bfloat_float(
        (state.read_register(lane, inst.operands[1].index) >> shifts[0]) &
        0xffff);
    const float b = bfloat_float(
        (state.read_register(lane, inst.operands[2].index) >> shifts[1]) &
        0xffff);
    state.write_register(lane, inst.operands[0].index,
                         float_bits(std::fma(a, b, c)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Unpack two UE8M0 scale bytes to BF16 or two E4M3 bytes to FP16.
// Native conversions canonicalize NaNs, including negative E4M3 NaN.
step_result execute_packed_unpack(const instruction &inst, warp_state &state,
                                  execution_context &) {
  if (inst.operands.size() != 2 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid packed byte unpack");
  std::string text = inst.operands[1].text;
  const bool high =
      text.size() > 3 && text.compare(text.size() - 3, 3, ".H1") == 0;
  if (high)
    text.resize(text.size() - 3);
  const auto src = parse_operand(text);
  if (src.kind != operand_kind::kRegister || src.wide || src.negated ||
      src.absolute || src.bitwise_complement ||
      src.half_selection != operand_half_selection::kNone)
    return unsupported(inst, "unqualified packed byte source");
  const bool fp8 = inst.opcode == "F2FP.F16.E4M3.UNPACK_B";
  const auto unpack = [fp8](unsigned x) -> uint32_t {
    if (!fp8)
      return x == 255 ? 0x7fffu : x == 0 ? 0x40u : x << 7;
    const unsigned magnitude = x & 127, exponent = magnitude >> 3;
    if (magnitude == 127)
      return 0x7fff; // canonical half NaN, either sign
    const uint32_t sign = (x & 128) << 8;
    if (exponent)
      return sign | ((exponent + 8) << 10) | ((x & 7) << 7);
    return sign | float_half_rn(float_bits(std::ldexp(float(x & 7), -9)));
  };
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t x = state.read_register(lane, src.index) >> (high ? 16 : 0);
    state.write_register(lane, inst.operands[0].index,
                         unpack(x & 255) | (unpack((x >> 8) & 255) << 16));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Native cvt.rn.bf16x2.e4m3x2 uses UE8M0 factors 0x7f7f, printed by
// nvdisasm as the bit-equivalent FP32 subnormal immediate plus H0.
step_result execute_fp8_bfloat_unpack(const instruction &inst,
                                      warp_state &state, execution_context &) {
  if (inst.operands.size() != 4 ||
      inst.operands[0].kind != operand_kind::kRegister ||
      (inst.operands[3].text != "H0" && inst.operands[3].text != "H1"))
    return unsupported(inst,
                       "expected packed FP8 payload, scale and half selector");
  std::string text = inst.operands[1].text;
  const bool high =
      text.size() > 3 && text.compare(text.size() - 3, 3, ".H1") == 0;
  if (high)
    text.resize(text.size() - 3);
  const auto payload = parse_operand(text);
  if (payload.kind != operand_kind::kRegister || payload.wide ||
      payload.negated || payload.absolute || payload.bitwise_complement ||
      payload.half_selection != operand_half_selection::kNone)
    return unsupported(inst, "invalid packed FP8 source");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t scale = 0;
    if (inst.operands[2].kind == operand_kind::kFloatImmediate)
      scale = float_bits(static_cast<float>(inst.operands[2].float_immediate));
    else if (!read_u32_operand(inst.operands[2], state, lane, scale))
      return unsupported(inst, "invalid packed FP8 scale");
    if (inst.operands[3].text == "H1")
      scale >>= 16;
    const uint32_t packed =
        state.read_register(lane, payload.index) >> (high ? 16 : 0);
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      const unsigned code = (packed >> (8 * half)) & 255,
                     factor = (scale >> (8 * half)) & 255;
      const unsigned mag = code & 127, exp = mag >> 3;
      float value = exp ? std::ldexp(float(8 + (mag & 7)), int(exp) - 10)
                        : std::ldexp(float(mag & 7), -9);
      if (code & 128)
        value = -value;
      const uint16_t bits =
          (mag == 127 || factor == 255)
              ? 0x7fff
              : float_bfloat_rn(std::ldexp(value, int(factor) - 127));
      result |= uint32_t(bits) << (16 * half);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// FP4 pair scaled by two UE8M0 bytes (.H0), or one broadcast byte (.Bn).
step_result execute_fp4_bfloat_unpack(const instruction &native,
                                      warp_state &state, execution_context &) {
  instruction inst = native;
  if (inst.operands.size() == 4 &&
      (inst.operands[3].text == "H0" || inst.operands[3].text == "H1")) {
    inst.operands[2].text += "." + inst.operands[3].text;
    inst.operands.pop_back();
  }
  if (inst.operands.size() != 3 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid scaled FP4 unpack");
  std::string payload_text = inst.operands[1].text;
  unsigned payload_shift = 0;
  const auto byte_selector = [](std::string &text, unsigned &shift) {
    if (text.size() > 3 && text[text.size() - 3] == '.' &&
        text[text.size() - 2] == 'B' && text.back() >= '0' &&
        text.back() <= '3') {
      shift = 8 * (text.back() - '0');
      text.resize(text.size() - 3);
      return true;
    }
    return false;
  };
  byte_selector(payload_text, payload_shift);
  const operand packed = parse_operand(payload_text);
  std::string text = inst.operands[2].text;
  unsigned scale_shift = 0;
  const bool broadcast = byte_selector(text, scale_shift);
  if (!broadcast &&
      (text.size() <= 3 || (text.compare(text.size() - 3, 3, ".H0") != 0 &&
                            text.compare(text.size() - 3, 3, ".H1") != 0)))
    return unsupported(inst, "unqualified FP4 scale selector");
  if (!broadcast) {
    scale_shift = text.back() == '1' ? 16 : 0;
    text.resize(text.size() - 3);
  }
  const operand scale = parse_operand(text);
  if (packed.kind != operand_kind::kRegister || packed.wide || packed.negated ||
      packed.absolute || packed.bitwise_complement ||
      packed.half_selection != operand_half_selection::kNone ||
      (scale.kind != operand_kind::kRegister &&
       scale.kind != operand_kind::kFloatImmediate) ||
      scale.wide || scale.negated || scale.absolute ||
      scale.bitwise_complement ||
      scale.half_selection != operand_half_selection::kNone)
    return unsupported(inst, "unqualified scaled FP4 source");
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t codes =
        state.read_register(lane, packed.index) >> payload_shift;
    const uint32_t factors =
        (scale.kind == operand_kind::kFloatImmediate
             ? float_bits(static_cast<float>(scale.float_immediate))
             : state.read_register(lane, scale.index)) >>
        scale_shift;
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      const unsigned code = (codes >> (4 * half)) & 15;
      const unsigned factor = (factors >> (broadcast ? 0 : 8 * half)) & 255;
      const float value = std::ldexp(
          (code & 8) ? -values[code & 7] : values[code & 7], int(factor) - 127);
      const uint16_t bits = factor == 255 ? 0x7fff : float_bfloat_rn(value);
      result |= uint32_t(bits) << (16 * half);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Scalar FP32 -> FP16/BF16 conversion writes a zero-extended half word.
step_result execute_f2f_half(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported scalar half conversion");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float value;
    if (!read_f32_operand(inst.operands[1], state, lane, value))
      return unsupported(inst, "unsupported scalar half source");
    state.write_register(lane, inst.operands[0].index,
                         inst.opcode == "F2F.BF16.F32"
                             ? float_bfloat_rn(value)
                             : float_half_rn(float_bits(value)));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

// Numeric warp maximum (including infinities) used by DSV4 KV quantization.
step_result execute_credux_max(const instruction &inst, warp_state &state,
                               execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[1].kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported CREDUX.MAX.F32 form");
  float result = -std::numeric_limits<float>::infinity();
  bool participating = false;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float value;
    if (!read_f32_operand(inst.operands[1], state, lane, value))
      return unsupported(inst, "unsupported CREDUX.MAX.F32 source");
    if (std::isnan(value)) {
      std::ostringstream detail;
      detail << "unqualified CREDUX NaN at lane " << lane << ": bits=0x"
             << std::hex << float_bits(value);
      return unsupported(inst, detail.str());
    }
    if (result == 0 && value == 0)
      result = std::signbit(result) && std::signbit(value) ? -0.0f : 0.0f;
    else
      result = std::fmax(result, value);
    participating = true;
  }
  if (participating)
    state.write_uniform_register(inst.operands[0].index, float_bits(result));
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
    const auto convert = [&](float value) -> uint16_t {
      const uint32_t bits = float_bits(value);
      if (inst.opcode == "F2FP.BF16.F32.PACK_AB" ||
          inst.opcode == "F2FP.SATFINITE.BF16.F32.PACK_AB" ||
          inst.opcode == "F2FP.RELU.BF16.F32.PACK_AB") {
        if (inst.opcode == "F2FP.RELU.BF16.F32.PACK_AB" && !std::isnan(value) &&
            value < 0)
          value = 0;
        const uint16_t rounded = float_bfloat_rn(value);
        if (inst.opcode == "F2FP.SATFINITE.BF16.F32.PACK_AB" &&
            (rounded & 0x7fff) == 0x7f80)
          return (rounded & 0x8000) | 0x7f7f;
        return rounded;
      }
      return float_half_rn(bits);
    };
    const uint32_t packed =
        convert(b) | (static_cast<uint32_t>(convert(a)) << 16);
    state.write_register(lane, destination->index, packed);
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

// FP32 products are exact in FP64. TwoSum retains the addition residual,
// including tiny addends lost by a plain double sum. No host rounding-mode
// mutation is needed, so concurrent simulator threads remain independent.
float directed_fma(float a, float b, float c, const std::string &opcode) {
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
    return std::fma(a, b, c);
  const double product = static_cast<double>(a) * b;
  const double sum = product + static_cast<double>(c);
  const double recovered = sum - product;
  const double residual = (product - (sum - recovered)) + (c - recovered);
  const bool down = opcode == "FFMA.RM" || opcode == "FFMA.FTZ.RM";
  const bool up = opcode == "FFMA.RP";
  if (sum == 0.0 && residual == 0.0) {
    const bool product_negative = std::signbit(a) != std::signbit(b);
    const bool both_zero = product == 0.0 && c == 0.0f;
    const bool negative = both_zero && product_negative == std::signbit(c)
                              ? product_negative
                              : down;
    return negative ? -0.0f : 0.0f;
  }
  float result = static_cast<float>(sum);
  const double rounded = result;
  const bool above = rounded > sum || (rounded == sum && residual < 0.0);
  const bool below = rounded < sum || (rounded == sum && residual > 0.0);
  if ((down && above) || (up && below))
    result = std::nextafter(result, down ? -INFINITY : INFINITY);
  else if (!down && !up && ((sum > 0 && above) || (sum < 0 && below)))
    result = std::nextafter(result, 0.0f);
  return result;
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
    const bool directed = inst.opcode == "FFMA.RM" ||
                          inst.opcode == "FFMA.FTZ.RM" ||
                          inst.opcode == "FFMA.RP" || inst.opcode == "FFMA.RZ";
    float result =
        directed ? directed_fma(a, b, c, inst.opcode) : std::fma(a, b, c);
    if (ftz)
      result = flush_subnormal(result);
    if (inst.opcode == "FFMA.SAT")
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

// BF16 inputs have exact FP64 products. Correct the rare FP32 midpoint
// before BF16 rounding so a tiny addend cannot produce double rounding.
uint16_t bfloat_fma_rn(float a, float b, float c) {
  float rounded = std::fma(a, b, c);
  if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
      std::isfinite(rounded) && (float_bits(rounded) & 0xffff) == 0x8000) {
    const double product = double(a) * double(b);
    const double sum = product + double(c);
    const double recovered = sum - product;
    const double residual =
        (product - (sum - recovered)) + (double(c) - recovered);
    const double difference = sum - double(rounded);
    const double direction = difference != 0 ? difference : residual;
    if (direction != 0)
      rounded = std::nextafter(rounded, direction > 0 ? INFINITY : -INFINITY);
  }
  return float_bfloat_rn(rounded);
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
        float value = bfloat_float(
            (state.read_register(lane, op.index) >> (16 * selected)) & 0xffff);
        if (op.absolute)
          value = std::fabs(value);
        if (op.negated)
          value = -value;
        values[i] = value;
      }
      result |= uint32_t(bfloat_fma_rn(values[0], values[1], values[2]))
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
    const uint32_t lo = float_bfloat_rn(
        std::fma(bfloat_float(a & 0xffff), low, bfloat_float(c & 0xffff)));
    const uint32_t hi = float_bfloat_rn(
        std::fma(bfloat_float(a >> 16), high, bfloat_float(c >> 16)));
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
        float value = bfloat_float(
            (state.read_register(lane, op.index) >> (16 * selected)) & 0xffff);
        if (op.absolute)
          value = std::fabs(value);
        if (op.negated)
          value = -value;
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
        float value = bfloat_float(
            (state.read_register(lane, op.index) >> (16 * selected)) & 0xffff);
        if (op.absolute)
          value = std::fabs(value);
        if (op.negated)
          value = -value;
        values[i] = value;
      }
      result |= uint32_t(float_bfloat_rn(values[0] * values[1])) << (half * 16);
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
        float value = bfloat_float(
            (state.read_register(lane, op.index) >> (16 * selected)) & 0xffff);
        if (op.absolute)
          value = std::fabs(value);
        if (op.negated)
          value = -value;
        values[i] = value;
      }
      result |= uint32_t(float_bfloat_rn(values[0] + values[1])) << (16 * half);
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
    if (inst.opcode.find(".RZ") != std::string::npos) {
      const double x = a, y = b;
      const double exact = multiply ? x * y : x + y;
      // TwoSum retains the sign of a tiny addend lost even in double. This
      // matters for directed rounding, e.g. 1 + (-2^-100) must round below 1.
      const double recovered_y = exact - x;
      const double error =
          multiply ? 0.0 : (x - (exact - recovered_y)) + (y - recovered_y);
      result = static_cast<float>(exact);
      if (std::isfinite(exact) &&
          (std::fabs(static_cast<double>(result)) > std::fabs(exact) ||
           (static_cast<double>(result) == exact &&
            ((exact > 0 && error < 0) || (exact < 0 && error > 0)))))
        result = std::nextafter(result, 0.0f);
    } else {
      result = multiply ? a * b : a + b;
    }
    if (ftz)
      result = flush_subnormal(result);
    if (inst.opcode == "FADD.FTZ.SAT")
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
    bool cmp = inst.opcode == "FSET.BF.EQ.AND" ? a == b : a > b;
    state.write_register(lane, inst.operands[0].index,
                         cmp && input ? 0xffffffffu : 0u);
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
      (inst.opcode.find(".AND") == std::string::npos &&
       inst.opcode.find(".OR") == std::string::npos) ||
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
    if (inst.opcode.find(".NAN.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b);
    else if (inst.opcode.find(".GEU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a >= b;
    else if (inst.opcode.find(".GTU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a > b;
    else if (inst.opcode.find(".LEU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a <= b;
    else if (inst.opcode.find(".NEU.") != std::string::npos)
      compared = std::isnan(a) || std::isnan(b) || a != b;
    else if (inst.opcode.find(".NE.") != std::string::npos)
      compared = !std::isnan(a) && !std::isnan(b) && a != b;
    else if (inst.opcode.find(".GT.") != std::string::npos)
      compared = !std::isnan(a) && !std::isnan(b) && a > b;
    else if (inst.opcode.find(".GE.") != std::string::npos)
      compared = !std::isnan(a) && !std::isnan(b) && a >= b;
    else if (inst.opcode.find(".EQ.") != std::string::npos)
      compared = a == b;
    else
      return false;
    result = inst.opcode.find(".OR") != std::string::npos ? compared || input
                                                          : compared && input;
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
  target.register_semantics("F2I.U64.TRUNC", execute_f2i_u64);
  target.register_semantics("UF2I.FTZ.U32.TRUNC.NTZ", execute_f2i);
  target.register_semantics("F2FP.BF16.E8.UNPACK_B", execute_packed_unpack);
  target.register_semantics("F2FP.F16.E4M3.UNPACK_B", execute_packed_unpack);
  target.register_semantics("F2FP.BF16.E4M3.UNPACK_B",
                            execute_fp8_bfloat_unpack);
  target.register_semantics("F2FP.BF16.E2M1.UNPACK_B",
                            execute_fp4_bfloat_unpack);
  target.register_semantics("F2FP.SATFINITE.E4M3.BF16.UNPACK_B_MERGE_C",
                            execute_bfloat_to_e4m3);
  target.register_semantics("F2FP.SATFINITE.E2M1.BF16.UNPACK_B_MERGE_C",
                            execute_bfloat_to_e4m3);
  target.register_semantics("F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C",
                            execute_f2fp_e4m3);
  target.register_semantics("FHFMA.BF16", execute_fhfma_bfloat);
  target.register_semantics("FHADD.BF16", execute_fhadd_bfloat);
  target.register_semantics("F2F.F16.F32", execute_f2f_half);
  target.register_semantics("F2F.BF16.F32", execute_f2f_half);
  target.register_semantics("CREDUX.MAX.F32", execute_credux_max);
  target.register_semantics("F2FP.F16.F32.PACK_AB", execute_f2fp_pack_ab);
  target.register_semantics("F2FP.BF16.F32.PACK_AB", execute_f2fp_pack_ab);
  target.register_semantics("F2FP.SATFINITE.BF16.F32.PACK_AB",
                            execute_f2fp_pack_ab);
  target.register_semantics("F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
                            execute_f2fp_e4m3);
  target.register_semantics("F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP",
                            execute_f2fp_ue8m0);
  target.register_semantics("FCHK", execute_fchk);
  target.register_semantics("FFMA", execute_ffma);
  target.register_semantics("FFMA.RM", execute_ffma);
  target.register_semantics("FFMA.RP", execute_ffma);
  target.register_semantics("FFMA.RZ", execute_ffma);
  target.register_semantics("FFMA2", execute_fp32x2);
  target.register_semantics("FFMA2.FTZ", execute_fp32x2);
  target.register_semantics("FADD2", execute_fp32x2);
  target.register_semantics("FADD2.FTZ", execute_fp32x2);
  target.register_semantics("FMUL2", execute_fp32x2);
  target.register_semantics("FMUL2.FTZ", execute_fp32x2);
  target.register_semantics("FFMA.FTZ", execute_ffma);
  target.register_semantics("FFMA.SAT", execute_ffma);
  target.register_semantics("FFMA.RM", execute_ffma);
  target.register_semantics("FFMA.FTZ.RM", execute_ffma);
  target.register_semantics("FFMA.RP", execute_ffma);
  target.register_semantics("FFMA.RZ", execute_ffma);
  target.register_semantics("UFFMA.FTZ", execute_ffma);
  target.register_semantics("HFMA2", execute_hfma2);
  target.register_semantics("HFMA2.MMA", execute_hfma2);
  target.register_semantics("HADD2", execute_hadd2);
  target.register_semantics("HADD2.BF16_V2", execute_bfloat_add2);
  target.register_semantics("HMUL2.BF16_V2", execute_bfloat_mul2);
  target.register_semantics("HMNMX2.BF16_V2", execute_bfloat_minmax2);
  target.register_semantics("VHMNMX.BF16_V2", execute_bfloat_minmax2);
  target.register_semantics("HFMA2.BF16_V2", execute_bfloat_fma_immediate);
  target.register_semantics("HADD2.F32", execute_hadd2_f32_conversion);
  target.register_semantics("FMUL", execute_binary_float);
  target.register_semantics("FMUL.FTZ", execute_binary_float);
  target.register_semantics("FMUL.RZ", execute_binary_float);
  target.register_semantics("UFMUL", execute_binary_float);
  target.register_semantics("UFMUL.FTZ", execute_binary_float);
  target.register_semantics("UFADD", execute_binary_float);
  target.register_semantics("UFADD.FTZ", execute_binary_float);
  target.register_semantics("FADD", execute_binary_float);
  target.register_semantics("FADD.RZ", execute_binary_float);
  target.register_semantics("FADD.FTZ", execute_binary_float);
  target.register_semantics("FADD.FTZ.SAT", execute_binary_float);
  target.register_semantics("FADD.FTZ.RZ", execute_binary_float);
  target.register_semantics("FMNMX", execute_fmnmx);
  target.register_semantics("FMNMX.FTZ", execute_fmnmx);
  target.register_semantics("FMNMX3.FTZ", execute_fmnmx);
  target.register_semantics("FMNMX3", execute_fmnmx);
  target.register_semantics("FMNMX3.NAN", execute_fmnmx);
  target.register_semantics("FSETP.EQ.AND", execute_fsetp);
  target.register_semantics("FSET.BF.EQ.AND", execute_fset_mask);
  target.register_semantics("FSET.BF.GT.AND", execute_fset_mask);
  target.register_semantics("FSETP.LEU.AND", execute_fsetp);
  target.register_semantics("FSETP.NEU.OR", execute_fsetp);
  target.register_semantics("F2FP.RELU.BF16.F32.PACK_AB", execute_f2fp_pack_ab);
  target.register_semantics("FSETP.NAN.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.NAN.AND", execute_fsetp);
  target.register_semantics("FSEL", execute_fsel);
  target.register_semantics("UFSEL", execute_fsel);
  target.register_semantics("FSETP.GEU.AND", execute_fsetp);
  target.register_semantics("FSETP.GEU.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.GTU.AND", execute_fsetp);
  target.register_semantics("FSETP.GTU.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.GE.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.GE.AND", execute_fsetp);
  target.register_semantics("FSETP.EQ.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.NEU.AND", execute_fsetp);
  target.register_semantics("FSETP.NEU.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.NE.AND", execute_fsetp);
  target.register_semantics("FSETP.NE.FTZ.AND", execute_fsetp);
  target.register_semantics("FSETP.GT.AND", execute_fsetp);
  target.register_semantics("FSETP.GT.FTZ.AND", execute_fsetp);
  target.register_semantics("UFSETP.GEU.AND", execute_fsetp);
  target.register_semantics("UFSETP.NEU.AND", execute_fsetp);
  target.register_semantics("UFSETP.NEU.FTZ.AND", execute_fsetp);
  target.register_semantics("UFSETP.NE.FTZ.AND", execute_fsetp);
  target.register_semantics("UFSETP.GT.AND", execute_fsetp);
  target.register_semantics("MUFU.EX2", execute_mufu_ex2);
  target.register_semantics("MUFU.LG2", execute_mufu_lg2);
  target.register_semantics("MUFU.RSQ", execute_mufu_rsq);
  target.register_semantics("MUFU.SQRT", execute_mufu_sqrt);
  target.register_semantics("MUFU.TANH", execute_mufu_tanh);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
