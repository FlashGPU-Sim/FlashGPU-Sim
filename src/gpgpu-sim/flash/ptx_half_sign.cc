#include "ptx_half_sign.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "half_sign.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_half_sign(const ptx_instruction *inst,
                           ptx_thread_info *thread) {
  const auto type = inst->get_type();
  const bool bf16 = type == BF16_TYPE || type == BF16X2_TYPE;
  if (!bf16 && type != F16_TYPE && type != F16X2_TYPE)
    return false;
  const auto op = inst->get_opcode();
  if ((op != ABS_OP && op != NEG_OP) || inst->get_num_operands() != 2 ||
      inst->get_scalar_type().size() != 1)
    panic("unqualified PTX half sign operation: " + inst->to_string());
  bool ftz = false;
  for (auto option : inst->get_options()) {
    if (option != FTZ_OPTION || bf16)
      panic("unqualified PTX half sign modifier: " + inst->to_string());
    ftz = true;
  }
  const bool packed = type == F16X2_TYPE || type == BF16X2_TYPE;
  const auto reg_type = packed ? B32_TYPE : B16_TYPE;
  const auto input =
      thread->get_operand_value(inst->src1(), inst->dst(), reg_type, thread, 1)
          .u32;
  ptx_reg_t result;
  result.u32 = 0;
  for (unsigned lane = 0; lane < (packed ? 2u : 1u); ++lane) {
    uint16_t bits = input >> (16 * lane);
    if (ftz)
      bits = half_flush_subnormal(bits);
    bits = half_sign_bits(bits, op == ABS_OP, op == NEG_OP);
    result.u32 |= uint32_t(bits) << (16 * lane);
  }
  thread->set_operand_value(inst->dst(), result, reg_type, thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
