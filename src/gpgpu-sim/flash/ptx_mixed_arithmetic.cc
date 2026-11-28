#include "ptx_mixed_arithmetic.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "mixed_precision.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_mixed_arithmetic(const ptx_instruction *inst,
                                  ptx_thread_info *thread) {
  const auto types = inst->get_scalar_type();
  if (types.size() < 2)
    return false;
  // Only the scalar SM100 forms. Newer packed/Rubin forms are not aliases.
  const auto op = inst->get_opcode();
  const bool fused = op == FMA_OP;
  if (types.size() != 2 || inst->get_type() != F32_TYPE ||
      (inst->get_type2() != F16_TYPE && inst->get_type2() != BF16_TYPE) ||
      (op != ADD_OP && op != SUB_OP && !fused) ||
      inst->get_num_operands() != (fused ? 4u : 3u))
    panic("unqualified PTX mixed arithmetic: " + inst->to_string());
  bool explicit_round = false;
  for (const auto modifier : inst->get_options()) {
    if (modifier == RN_OPTION || modifier == RZ_OPTION ||
        modifier == RM_OPTION || modifier == RP_OPTION)
      explicit_round = true;
    else if (modifier != SAT_OPTION)
      panic("unqualified PTX mixed arithmetic modifier: " + inst->to_string());
  }
  if (fused && !explicit_round)
    panic("mixed FMA requires explicit rounding");
  const bool bfloat = inst->get_type2() == BF16_TYPE;
  auto half = [&](const operand_info &src) {
    const auto value =
        thread->get_operand_value(src, inst->dst(), B16_TYPE, thread, 1);
    return mixed_input_to_f32(value.u16, bfloat);
  };
  const float a = half(inst->src1());
  const float b = fused ? half(inst->src2()) : 1.0f;
  const auto &src_c = fused ? inst->src3() : inst->src2();
  const float c =
      thread->get_operand_value(src_c, inst->dst(), F32_TYPE, thread, 1).f32;
  const auto mode = inst->rounding_mode();
  const int rounding = mode == RM_OPTION   ? FE_DOWNWARD
                       : mode == RP_OPTION ? FE_UPWARD
                       : mode == RZ_OPTION ? FE_TOWARDZERO
                                           : FE_TONEAREST;
  ptx_reg_t result;
  result.f32 = mixed_f32_arithmetic(fused          ? mixed_operation::fma
                                    : op == SUB_OP ? mixed_operation::subtract
                                                   : mixed_operation::add,
                                    a, b, c, rounding, inst->saturation_mode());
  thread->set_operand_value(inst->dst(), result, F32_TYPE, thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
