#include "ptx_bfloat_conversion.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "bfloat_conversion.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
bool execute_ptx_bfloat_cvt(const ptx_instruction *inst,
                            ptx_thread_info *thread) {
  const auto to = inst->get_type(), from = inst->get_type2();
  if (to != BF16_TYPE && to != BF16X2_TYPE && from != BF16_TYPE &&
      from != BF16X2_TYPE) {
    for (auto option : inst->get_options())
      if (option == RELU_OPTION)
        panic("unqualified PTX relu conversion: " + inst->to_string());
    return false;
  }
  const auto &dst = inst->dst();
  if (to == F32_TYPE && from == BF16_TYPE && inst->get_num_operands() == 2 &&
      inst->get_options().empty()) {
    auto value =
        thread->get_operand_value(inst->src1(), dst, B16_TYPE, thread, 1);
    value.u32 = uint32_t(value.u16) << 16;
    thread->set_operand_value(dst, value, F32_TYPE, thread, inst);
    return true;
  }
  if (from != F32_TYPE || (to != BF16_TYPE && to != BF16X2_TYPE) ||
      inst->get_num_operands() != (to == BF16X2_TYPE ? 3u : 2u))
    panic("unqualified PTX BF16 conversion: " + inst->to_string());
  bool relu = false, satfinite = false;
  for (auto option : inst->get_options()) {
    if (option == RELU_OPTION)
      relu = true;
    else if (option == SATFINITE_OPTION)
      satfinite = true;
    else if (option != RN_OPTION && option != RZ_OPTION)
      panic("unqualified PTX BF16 conversion modifier: " + inst->to_string());
  }
  const auto rounding = inst->rounding_mode();
  if (rounding != RN_OPTION && rounding != RZ_OPTION)
    panic("unqualified PTX BF16 rounding: " + inst->to_string());
  const auto convert = [&](const operand_info &source) {
    const auto value =
        thread->get_operand_value(source, dst, F32_TYPE, thread, 1);
    return f32_bits_to_bf16(value.u32, rounding == RZ_OPTION, relu, satfinite);
  };
  ptx_reg_t result;
  result.u32 = convert(inst->src1());
  if (to == BF16X2_TYPE)
    result.u32 = (result.u32 << 16) | convert(inst->src2());
  thread->set_operand_value(
      dst, result, to == BF16X2_TYPE ? B32_TYPE : B16_TYPE, thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
