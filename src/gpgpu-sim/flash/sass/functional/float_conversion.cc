#include "../decode/scaled_unpack_operand.h"
#include "internal.h"

#include <cmath>
#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// Numeric conversions and packed low-precision formats. Arithmetic remains
// in float.cc; operand selector decoding is shared with timing projection.
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

// Unpack two UE8M0 scale bytes to BF16 or two E4M3 bytes to FP16.
// Native conversions canonicalize NaNs, including negative E4M3 NaN.
step_result execute_packed_unpack(const instruction &inst, warp_state &state,
                                  execution_context &) {
  if (inst.operands.size() != 2 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid packed byte unpack");
  operand src;
  unsigned shift = 0;
  if (!decode_halfword_source(inst.operands[1], src, shift))
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
    const uint32_t x = state.read_register(lane, src.index) >> shift;
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
  scaled_unpack_operands decoded;
  if (!decode_scaled_unpack(inst, decoded))
    return unsupported(inst, "invalid scaled FP8 unpack");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint32_t scale = 0;
    if (decoded.scale.kind == operand_kind::kFloatImmediate)
      scale = float_bits(static_cast<float>(decoded.scale.float_immediate));
    else if (!read_u32_operand(decoded.scale, state, lane, scale))
      return unsupported(inst, "invalid packed FP8 scale");
    scale >>= decoded.scale_shift;
    const uint32_t packed = state.read_register(lane, decoded.payload.index) >>
                            decoded.payload_shift;
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
  const instruction &inst = native;
  scaled_unpack_operands decoded;
  if (!decode_scaled_unpack(inst, decoded))
    return unsupported(inst, "invalid scaled FP4 unpack");
  const operand &packed = decoded.payload, &scale = decoded.scale;
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    const uint32_t codes =
        state.read_register(lane, packed.index) >> decoded.payload_shift;
    const uint32_t factors =
        (scale.kind == operand_kind::kFloatImmediate
             ? float_bits(static_cast<float>(scale.float_immediate))
             : state.read_register(lane, scale.index)) >>
        decoded.scale_shift;
    uint32_t result = 0;
    for (unsigned half = 0; half < 2; ++half) {
      const unsigned code = (codes >> (4 * half)) & 15;
      const unsigned factor =
          (factors >> (decoded.broadcast_scale ? 0 : 8 * half)) & 255;
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

// Scalar FP32 -> FP16 conversion writes a zero-extended half word.
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
                         float_half_rn(float_bits(value)));
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
    const auto convert = [&](float value) -> uint16_t {
      const uint32_t bits = float_bits(value);
      return float_half_rn(bits);
    };
    const uint32_t packed =
        convert(b) | (static_cast<uint32_t>(convert(a)) << 16);
    state.write_register(lane, destination->index, packed);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_float_conversion_semantics(frontend &target) {
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
  target.register_semantics("F2F.F16.F32", execute_f2f_half);
  register_bfloat_conversion_semantics(target);
  target.register_semantics("F2FP.F16.F32.PACK_AB", execute_f2fp_pack_ab);
  target.register_semantics("F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
                            execute_f2fp_e4m3);
  target.register_semantics("F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP",
                            execute_f2fp_ue8m0);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
