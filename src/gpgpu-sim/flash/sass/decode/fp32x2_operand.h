#pragma once
#include "../frontend.h"
#include "operand_parser.h"

namespace flash_gpgpu_sim {
namespace sass {
struct fp32x2_source {
  operand value;
  bool paired = false;
  bool reversed = false;
};
inline bool decode_fp32x2_source(const std::string &text, fp32x2_source &out) {
  out = {};
  std::string base = text;
  bool selected = false;
  for (const char *tag : {".F32x2.HI_LO", ".F32x2.LO_HI", ".F32"}) {
    const std::string suffix(tag);
    if (text.size() > suffix.size() &&
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
      base.resize(text.size() - suffix.size());
      out.paired = suffix != ".F32";
      out.reversed = suffix == ".F32x2.LO_HI";
      selected = true;
      break;
    }
  }
  out.value = parse_operand(base);
  const auto &v = out.value;
  if (!selected)
    return v.kind == operand_kind::kImmediate ||
           v.kind == operand_kind::kFloatImmediate;
  const bool uniform = v.kind == operand_kind::kUniformRegister;
  if (v.kind != operand_kind::kRegister && !uniform)
    return false;
  const unsigned zero = uniform ? kZeroUniformRegister : kZeroRegister;
  return !v.bitwise_complement && !v.wide &&
         v.half_selection == operand_half_selection::kNone &&
         v.matrix_layout == operand_matrix_layout::kNone && v.index <= zero &&
         (!out.paired || v.index < zero - 1);
}
} // namespace sass
} // namespace flash_gpgpu_sim
