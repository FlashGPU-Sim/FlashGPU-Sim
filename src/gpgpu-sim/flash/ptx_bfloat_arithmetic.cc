#include "ptx_bfloat_arithmetic.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "bfloat_arithmetic.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_bfloat_arithmetic(const ptx_instruction *inst,
                                   ptx_thread_info *thread) {
  const auto type = inst->get_type();
  if (type != BF16_TYPE && type != BF16X2_TYPE)
    return false;
  const auto op = inst->get_opcode();
  const bool fused = op == FMA_OP;
  if (inst->get_scalar_type().size() != 1 ||
      (op != ADD_OP && op != SUB_OP && op != MUL_OP && !fused) ||
      inst->get_num_operands() != (fused ? 4u : 3u) ||
      inst->rounding_mode() != RN_OPTION || inst->saturation_mode())
    panic("unqualified PTX BF16 arithmetic: " + inst->to_string());
  for (const auto modifier : inst->get_options())
    if (modifier != RN_OPTION)
      panic("unqualified PTX BF16 arithmetic modifier: " + inst->to_string());
  const unsigned reg_type = type == BF16X2_TYPE ? B32_TYPE : B16_TYPE;
  const auto read = [&](const operand_info &source) {
    return thread->get_operand_value(source, inst->dst(), reg_type, thread, 1)
        .u32;
  };
  const uint32_t a = read(inst->src1()), b = read(inst->src2());
  const uint32_t c = fused ? read(inst->src3()) : 0;
  ptx_reg_t result;
  result.u32 = 0;
  for (unsigned lane = 0; lane < (type == BF16X2_TYPE ? 2u : 1u); ++lane) {
    const unsigned shift = 16 * lane;
    const float x = bf16_to_f32(a >> shift), y = bf16_to_f32(b >> shift);
    const uint16_t value = fused ? bf16_fma_rn(x, y, bf16_to_f32(c >> shift))
                           : op == MUL_OP
                               ? bf16_mul_rn(x, y)
                               : bf16_add_rn(x, op == SUB_OP ? -y : y);
    result.u32 |= uint32_t(value) << shift;
  }
  thread->set_operand_value(inst->dst(), result, reg_type, thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
