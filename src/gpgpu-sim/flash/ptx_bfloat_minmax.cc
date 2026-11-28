#include "ptx_bfloat_minmax.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "bfloat_minmax.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_bfloat_minmax(const ptx_instruction *inst,
                               ptx_thread_info *thread) {
  const auto type = inst->get_type();
  if (type != BF16_TYPE && type != BF16X2_TYPE)
    return false;
  const auto op = inst->get_opcode();
  if ((op != MIN_OP && op != MAX_OP) || inst->get_num_operands() != 3 ||
      inst->get_scalar_type().size() != 1 || !inst->get_options().empty())
    panic("unqualified PTX BF16 min/max form: " + inst->to_string());
  const bool packed = type == BF16X2_TYPE;
  const auto reg_type = packed ? B32_TYPE : B16_TYPE;
  const auto read = [&](const operand_info &src) {
    return thread->get_operand_value(src, inst->dst(), reg_type, thread, 1).u32;
  };
  const uint32_t a = read(inst->src1()), b = read(inst->src2());
  ptx_reg_t result;
  result.u32 = bf16_minmax(a, b, op == MIN_OP);
  if (packed)
    result.u32 |= uint32_t(bf16_minmax(a >> 16, b >> 16, op == MIN_OP)) << 16;
  thread->set_operand_value(inst->dst(), result, reg_type, thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
