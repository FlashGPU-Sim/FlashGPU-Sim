#pragma once
#include "operand_parser.h"

namespace flash_gpgpu_sim {
namespace sass {

// Decode once for functional values and timing dependencies. The printed
// selector is not a register; stripping it must retain its underlying source.
struct scaled_unpack_operands {
  operand payload, scale;
  unsigned payload_shift = 0, scale_shift = 0;
  bool broadcast_scale = false;
};

inline bool take_unpack_selector(std::string &text, char kind,
                                 unsigned &shift) {
  const unsigned limit = kind == 'B' ? 4 : 2;
  if (text.size() <= 3 || text[text.size() - 3] != '.' ||
      text[text.size() - 2] != kind || text.back() < '0' ||
      unsigned(text.back() - '0') >= limit)
    return false;
  shift = (kind == 'B' ? 8 : 16) * (text.back() - '0');
  text.resize(text.size() - 3);
  return true;
}

inline bool decode_halfword_source(const operand &source, operand &value,
                                   unsigned &shift) {
  shift = 0;
  std::string text = source.text;
  take_unpack_selector(text, 'H', shift);
  value = parse_operand(text);
  return value.kind == operand_kind::kRegister && !value.wide &&
         !value.negated && !value.absolute && !value.bitwise_complement &&
         value.half_selection == operand_half_selection::kNone &&
         value.matrix_layout == operand_matrix_layout::kNone;
}

inline bool decode_scaled_unpack(const instruction &inst,
                                 scaled_unpack_operands &out) {
  out = {};
  const bool fp4 = inst.opcode == "F2FP.BF16.E2M1.UNPACK_B";
  if ((!fp4 && inst.opcode != "F2FP.BF16.E4M3.UNPACK_B") ||
      inst.operands.size() < 3 || inst.operands.size() > 4 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return false;
  std::string payload = inst.operands[1].text;
  std::string scale = inst.operands[2].text;
  if (inst.operands.size() == 4) {
    const auto &selector = inst.operands[3].text;
    if (selector != "H0" && selector != "H1")
      return false;
    scale += "." + selector;
  } else if (!fp4) {
    return false;
  }
  take_unpack_selector(payload, fp4 ? 'B' : 'H', out.payload_shift);
  out.broadcast_scale =
      fp4 && take_unpack_selector(scale, 'B', out.scale_shift);
  if (!out.broadcast_scale &&
      !take_unpack_selector(scale, 'H', out.scale_shift))
    return false;
  out.payload = parse_operand(payload);
  out.scale = parse_operand(scale);
  const auto plain = [](const operand &v) {
    return !v.wide && !v.negated && !v.absolute && !v.bitwise_complement &&
           v.half_selection == operand_half_selection::kNone &&
           v.matrix_layout == operand_matrix_layout::kNone;
  };
  if (out.payload.kind != operand_kind::kRegister || !plain(out.payload))
    return false;
  if (fp4)
    return (out.scale.kind == operand_kind::kRegister ||
            out.scale.kind == operand_kind::kFloatImmediate) &&
           plain(out.scale);
  // FP8 also accepts the integer/uniform scale forms handled by
  // read_u32_operand.
  return out.scale.kind == operand_kind::kFloatImmediate ||
         ((out.scale.kind == operand_kind::kRegister ||
           out.scale.kind == operand_kind::kUniformRegister ||
           out.scale.kind == operand_kind::kImmediate) &&
          !out.scale.absolute &&
          out.scale.half_selection == operand_half_selection::kNone &&
          out.scale.matrix_layout == operand_matrix_layout::kNone);
}

} // namespace sass
} // namespace flash_gpgpu_sim
