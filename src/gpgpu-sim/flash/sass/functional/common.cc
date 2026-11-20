#include "internal.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {

std::mutex global_atomic_mutex;

step_result unsupported(const instruction &inst, const std::string &reason) {
  std::ostringstream detail;
  detail << "unsupported " << inst.opcode << " at 0x" << std::hex << inst.pc
         << ": " << reason;
  return {step_status::kUnsupported, detail.str()};
}

step_result memory_fault(const instruction &inst, memory_space space,
                         uint64_t address, size_t bytes) {
  const char *space_name = "shared";
  if (space == memory_space::kConstant)
    space_name = "constant";
  else if (space == memory_space::kGlobal)
    space_name = "global";
  else if (space == memory_space::kLocal)
    space_name = "local";
  std::ostringstream detail;
  detail << space_name << " memory fault at 0x" << std::hex << address
         << " for " << std::dec << bytes << " bytes while executing "
         << inst.opcode << " at 0x" << std::hex << inst.pc;
  return {step_status::kUnsupported, detail.str()};
}

const operand *get_operand(const instruction &inst, size_t position,
                           operand_kind kind) {
  if (position >= inst.operands.size() || inst.operands[position].kind != kind)
    return nullptr;
  return &inst.operands[position];
}

bool has_typed_operands(const instruction &inst, size_t count) {
  return inst.operands_structured && inst.operands.size() == count;
}

bool lane_executes(const instruction &inst, const warp_state &state,
                   unsigned lane) {
  if (!inst.has_guard)
    return true;
  bool value = false;
  if (inst.guard.kind == operand_kind::kPredicate)
    value = state.read_predicate(lane, inst.guard.index);
  else if (inst.guard.kind == operand_kind::kUniformPredicate)
    value = state.read_uniform_predicate(inst.guard.index);
  return value != inst.guard.negated;
}

bool any_lane_executes(const instruction &inst, const warp_state &state) {
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane))
      return true;
  }
  return false;
}

bool read_predicate_operand(const operand &source, const warp_state &state,
                            unsigned lane, bool &value) {
  if (source.bitwise_complement || source.absolute ||
      source.half_selection != operand_half_selection::kNone ||
      source.matrix_layout != operand_matrix_layout::kNone)
    return false;
  if (source.kind == operand_kind::kPredicate)
    value = state.read_predicate(lane, source.index);
  else if (source.kind == operand_kind::kUniformPredicate)
    value = state.read_uniform_predicate(source.index);
  else
    return false;
  if (source.negated)
    value = !value;
  return true;
}

void advance(const instruction &inst, warp_state &state) {
  state.pc = inst.pc + kInstructionBytes;
}

uint64_t read_register_pair(const warp_state &state, unsigned lane,
                            unsigned low_register) {
  if (low_register == kZeroRegister)
    return 0;
  const uint64_t low = state.read_register(lane, low_register);
  const uint64_t high = state.read_register(lane, low_register + 1);
  return low | (high << 32);
}

uint64_t read_uniform_register_pair(const warp_state &state,
                                    unsigned low_register) {
  if (low_register == kZeroUniformRegister)
    return 0;
  const uint64_t low = state.read_uniform_register(low_register);
  const uint64_t high = state.read_uniform_register(low_register + 1);
  return low | (high << 32);
}

void write_register_pair(warp_state &state, unsigned lane,
                         unsigned low_register, uint64_t value) {
  if (low_register == kZeroRegister)
    return;
  state.write_register(lane, low_register, static_cast<uint32_t>(value));
  state.write_register(lane, low_register + 1,
                       static_cast<uint32_t>(value >> 32));
}

void write_uniform_register_pair(warp_state &state, unsigned low_register,
                                 uint64_t value) {
  if (low_register == kZeroUniformRegister)
    return;
  state.write_uniform_register(low_register, static_cast<uint32_t>(value));
  state.write_uniform_register(low_register + 1,
                               static_cast<uint32_t>(value >> 32));
}

bool read_u32_operand(const operand &source, const warp_state &state,
                      unsigned lane, uint32_t &value) {
  // Layout/absolute modifiers need instruction-specific semantics; never
  // silently treat a modified operand as an ordinary integer register.
  if (source.absolute ||
      source.half_selection != operand_half_selection::kNone ||
      source.matrix_layout != operand_matrix_layout::kNone)
    return false;
  switch (source.kind) {
  case operand_kind::kRegister:
    value = state.read_register(lane, source.index);
    break;
  case operand_kind::kUniformRegister:
    value = state.read_uniform_register(source.index);
    break;
  case operand_kind::kImmediate:
    value = static_cast<uint32_t>(source.immediate);
    break;
  default:
    return false;
  }
  if (source.bitwise_complement)
    value = ~value;
  else if (source.negated)
    value = uint32_t{0} - value;
  return true;
}

bool read_u64_operand(const operand &source, const warp_state &state,
                      unsigned lane, uint64_t &value) {
  if (source.absolute ||
      source.half_selection != operand_half_selection::kNone ||
      source.matrix_layout != operand_matrix_layout::kNone)
    return false;
  switch (source.kind) {
  case operand_kind::kRegister:
    value = read_register_pair(state, lane, source.index);
    break;
  case operand_kind::kUniformRegister:
    value = read_uniform_register_pair(state, source.index);
    break;
  case operand_kind::kImmediate:
    value = static_cast<uint64_t>(source.immediate);
    break;
  default:
    return false;
  }
  if (source.bitwise_complement)
    value = ~value;
  else if (source.negated)
    value = uint64_t{0} - value;
  return true;
}

bool read_f32_operand(const operand &source, const warp_state &state,
                      unsigned lane, float &value) {
  if (source.bitwise_complement ||
      source.half_selection != operand_half_selection::kNone ||
      source.matrix_layout != operand_matrix_layout::kNone)
    return false;
  if (source.kind == operand_kind::kRegister) {
    value = bits_float(state.read_register(lane, source.index));
  } else if (source.kind == operand_kind::kUniformRegister) {
    value = bits_float(state.read_uniform_register(source.index));
  } else if (source.kind == operand_kind::kFloatImmediate) {
    value = static_cast<float>(source.float_immediate);
  } else if (source.kind == operand_kind::kImmediate) {
    value = static_cast<float>(source.immediate);
  } else {
    return false;
  }
  if (source.absolute)
    value = std::fabs(value);
  if (source.negated)
    value = -value;
  return true;
}

uint64_t evaluate_address_terms(const address_expression &address,
                                const warp_state &state, unsigned lane) {
  uint64_t value = 0;
  for (const address_term &term : address.terms) {
    uint64_t component = 0;
    if (term.kind == address_term_kind::kRegister) {
      component = term.wide ? read_register_pair(state, lane, term.index)
                            : state.read_register(lane, term.index);
    } else if (term.kind == address_term_kind::kUniformRegister) {
      component = term.wide ? read_uniform_register_pair(state, term.index)
                            : state.read_uniform_register(term.index);
    } else {
      component = static_cast<uint64_t>(term.immediate);
    }
    value += term.negated ? uint64_t{0} - component : component;
  }
  return value;
}

bool evaluate_address(const operand &address, const warp_state &state,
                      unsigned lane, uint64_t &value) {
  if (address.kind != operand_kind::kMemoryAddress ||
      address.address_groups.size() != 1)
    return false;
  value = evaluate_address_terms(address.address_groups[0], state, lane);
  return true;
}

bool evaluate_shared_address(const operand &address, const warp_state &state,
                             unsigned lane, uint64_t &value) {
  if (!evaluate_address(address, state, lane, value))
    return false;
  value = static_cast<uint32_t>(value);
  return true;
}

bool read_special_register(const operand &source,
                           const execution_context &context, unsigned lane,
                           uint32_t &value) {
  if (source.kind != operand_kind::kSpecialRegister)
    return false;
  if (source.text == "SRZ")
    value = 0;
  else if (source.text == "SR_TID.X")
    value = context.thread_idx_x[lane];
  else if (source.text == "SR_TID.Y")
    value = context.thread_idx_y[lane];
  else if (source.text == "SR_TID.Z")
    value = context.thread_idx_z[lane];
  else if (source.text == "SR_LANEID")
    value = lane;
  else if (source.text == "SR_VIRTID")
    // SM120 lowers PTX %warpid to SR_VIRTID[14:8]: S2R is followed by
    // SHF.R.U32.HI(..., 8) and SGXT.U32(..., 7). Preserve that raw SASS
    // register layout instead of returning the already-decoded warp id.
    value = context.warp_id << 8;
  else if (source.text == "SR_LTMASK")
    value = lane == 0 ? 0 : (uint32_t{1} << lane) - 1;
  else if (source.text == "SR_CTAID.X")
    value = context.cta_id_x;
  else if (source.text == "SR_CTAID.Y")
    value = context.cta_id_y;
  else if (source.text == "SR_CTAID.Z")
    value = context.cta_id_z;
  else if (source.text == "SR_CgaCtaId")
    value = context.cga_cta_id;
  else if (source.text == "SR_VIRTUALSMID")
    value = context.virtual_smid;
  else if (source.text == "SR_SWINHI")
    // The flat functional shared-memory aperture starts at address zero.
    value = 0;
  else
    return false;
  return true;
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float width");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(uint32_t bits) {
  float value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float half_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x3ffu;
  if (exponent == 0) {
    if (mantissa == 0)
      return bits_float(sign);
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return sign == 0 ? magnitude : -magnitude;
  }
  if (exponent == 0x1f)
    return bits_float(sign | 0x7f800000u | (mantissa << 13));
  return bits_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

float bfloat_float(uint16_t bits) {
  return bits_float(static_cast<uint32_t>(bits) << 16);
}

float tf32_float(uint32_t bits) { return bits_float(bits & 0xffffe000u); }

uint16_t float_half_rn(uint32_t bits) {
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    if (mantissa == 0)
      return sign | 0x7c00u;
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
    payload |= 0x0200u;
    return sign | 0x7c00u | payload;
  }

  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31)
    return sign | 0x7c00u;
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    mantissa |= 0x800000u;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
      ++rounded;
    return sign | static_cast<uint16_t>(rounded);
  }

  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      ++half_exponent;
      if (half_exponent >= 31)
        return sign | 0x7c00u;
    }
  }
  return sign | static_cast<uint16_t>(half_exponent << 10) |
         static_cast<uint16_t>(rounded);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
