#include "../../bfloat_conversion.h"
#include "internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result convert_bfloat(const instruction &inst, warp_state &state,
                           execution_context &) {
  const bool scalar = inst.opcode.rfind("F2F.", 0) == 0;
  const bool merge = inst.opcode.find(".MERGE_C") != std::string::npos;
  if (!has_typed_operands(inst, scalar ? 2 : 3) ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid BF16 conversion operands");
  if (merge &&
      (inst.operands[2].kind != operand_kind::kRegister ||
       inst.operands[2].index > kZeroRegister || inst.operands[2].negated ||
       inst.operands[2].absolute || inst.operands[2].bitwise_complement ||
       inst.operands[2].wide ||
       inst.operands[2].half_selection != operand_half_selection::kNone ||
       inst.operands[2].matrix_layout != operand_matrix_layout::kNone))
    return unsupported(inst, "invalid BF16 merge register");
  const bool rz = inst.opcode.find(".RZ") != std::string::npos;
  const bool relu = inst.opcode.find(".RELU") != std::string::npos;
  const bool satfinite = inst.opcode.find(".SATFINITE") != std::string::npos;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float a, b;
    if (!read_f32_operand(inst.operands[1], state, lane, a))
      return unsupported(inst, "invalid BF16 source");
    uint32_t result = f32_bits_to_bf16(float_bits(a), rz, relu, satfinite);
    if (merge)
      result |= state.read_register(lane, inst.operands[2].index) << 16;
    else if (!scalar) {
      if (!read_f32_operand(inst.operands[2], state, lane, b))
        return unsupported(inst, "invalid BF16 second source");
      result =
          (result << 16) | f32_bits_to_bf16(float_bits(b), rz, relu, satfinite);
    }
    state.write_register(lane, inst.operands[0].index, result);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
} // namespace
void register_bfloat_conversion_semantics(frontend &target) {
  for (const std::string rounding : {"", ".RZ"}) {
    target.register_semantics("F2F.BF16.F32" + rounding, convert_bfloat);
    for (const std::string modifier :
         {"", ".RELU", ".SATFINITE", ".SATFINITE.RELU"})
      for (const std::string packing : {".PACK_AB", ".MERGE_C"})
        target.register_semantics("F2FP" + modifier + ".BF16.F32" + packing +
                                      rounding,
                                  convert_bfloat);
  }
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
