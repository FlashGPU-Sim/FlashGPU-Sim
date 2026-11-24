#pragma once
#include "scaled_unpack_operand.h"
#include <cfenv>
namespace flash_gpgpu_sim {
namespace sass {
struct mixed_scalar_opcode {
  bool fused = false, bfloat = false, saturate = false;
  int rounding = FE_TONEAREST;
};
inline bool decode_mixed_scalar_opcode(const std::string &opcode,
                                       mixed_scalar_opcode &out) {
  out = {};
  std::string suffix;
  if (opcode.rfind("FHFMA", 0) == 0) {
    out.fused = true;
    suffix = opcode.substr(5);
  } else if (opcode.rfind("FHADD", 0) == 0)
    suffix = opcode.substr(5);
  else
    return false;
  if (suffix.rfind(".BF16", 0) == 0) {
    out.bfloat = true;
    suffix.erase(0, 5);
  }
  if (suffix.rfind(".RM", 0) == 0) {
    out.rounding = FE_DOWNWARD;
    suffix.erase(0, 3);
  } else if (suffix.rfind(".RP", 0) == 0) {
    out.rounding = FE_UPWARD;
    suffix.erase(0, 3);
  } else if (suffix.rfind(".RZ", 0) == 0) {
    out.rounding = FE_TOWARDZERO;
    suffix.erase(0, 3);
  }
  if (suffix == ".SAT") {
    out.saturate = true;
    suffix.clear();
  }
  return suffix.empty();
}
inline bool decode_mixed_half_source(const operand &source, operand &value,
                                     unsigned &shift) {
  operand plain = source;
  bool negative = false, absolute = false;
  if (!plain.text.empty() && plain.text.front() == '-') {
    negative = true;
    plain.text.erase(0, 1);
  }
  if (plain.text.size() > 2 && plain.text.front() == '|' &&
      plain.text.back() == '|') {
    absolute = true;
    plain.text = plain.text.substr(1, plain.text.size() - 2);
  }
  if (!decode_halfword_source(plain, value, shift))
    return false;
  value.negated = negative;
  value.absolute = absolute;
  return true;
}
} // namespace sass
} // namespace flash_gpgpu_sim
