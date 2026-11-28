#include "../../mixed_precision.h"
#include "../decode/mixed_precision_operand.h"
#include "internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result execute_mixed_scalar(const instruction &inst, warp_state &state,
                                 execution_context &) {
  mixed_scalar_opcode desc;
  if (!decode_mixed_scalar_opcode(inst.opcode, desc) ||
      inst.operands.size() != (desc.fused ? 4u : 3u) ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].wide || inst.operands[0].negated ||
      inst.operands[0].absolute)
    return unsupported(inst, "invalid mixed scalar opcode/operands");
  operand halves[2];
  unsigned shifts[2]{};
  for (unsigned i = 0; i < (desc.fused ? 2u : 1u); ++i)
    if (!decode_mixed_half_source(inst.operands[i + 1], halves[i], shifts[i]))
      return unsupported(inst, "invalid mixed scalar half source");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    float values[2] = {0, 1};
    for (unsigned i = 0; i < (desc.fused ? 2u : 1u); ++i) {
      values[i] = mixed_input_to_f32(
          state.read_register(lane, halves[i].index) >> shifts[i], desc.bfloat);
      if (halves[i].absolute)
        values[i] = std::fabs(values[i]);
      if (halves[i].negated)
        values[i] = -values[i];
    }
    float c;
    if (!read_f32_operand(inst.operands[desc.fused ? 3 : 2], state, lane, c))
      return unsupported(inst, "invalid mixed scalar accumulator");
    const float result = mixed_f32_arithmetic(
        desc.fused ? mixed_operation::fma : mixed_operation::add, values[0],
        values[1], c, desc.rounding, desc.saturate);
    state.write_register(lane, inst.operands[0].index, float_bits(result));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
} // namespace
void register_mixed_precision_semantics(frontend &target) {
  for (const std::string base : {"FHADD", "FHFMA"})
    for (const std::string type : {"", ".BF16"})
      for (const std::string rounding : {"", ".RM", ".RP", ".RZ"})
        for (const std::string sat : {"", ".SAT"})
          target.register_semantics(base + type + rounding + sat,
                                    execute_mixed_scalar);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
