#include "ptx_narrow_conversion.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "narrow_conversion.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
namespace {
bool narrow(unsigned type) {
  return type == E4M3X2_TYPE || type == UE8M0X2_TYPE || type == E2M1X2_TYPE;
}
} // namespace
bool execute_ptx_narrow_cvt(const ptx_instruction *inst,
                            ptx_thread_info *thread) {
  const unsigned to = inst->get_type(), from = inst->get_type2();
  if (!narrow(to) && !narrow(from))
    return false;
  const auto fail = [&]() {
    panic("unqualified PTX packed conversion: " + inst->to_string());
  };
  bool satfinite = false;
  for (auto opt : inst->get_options()) {
    if (opt == SATFINITE_OPTION)
      satfinite = true;
    else if (opt != RN_OPTION && opt != RP_OPTION)
      fail();
  }
  const unsigned rounding = inst->rounding_mode();
  const auto &dst = inst->dst();
  auto read = [&](const operand_info &op, unsigned type) {
    return thread->get_operand_value(op, dst, type, thread, 1);
  };
  ptx_reg_t result;
  result.u32 = 0;
  if ((to == BF16X2_TYPE || to == F16X2_TYPE) &&
      (from == E4M3X2_TYPE || from == UE8M0X2_TYPE)) {
    if (rounding != RN_OPTION || satfinite || inst->get_num_operands() != 2 ||
        (to == F16X2_TYPE && from != E4M3X2_TYPE))
      fail();
    const unsigned input = read(inst->src1(), B16_TYPE).u16;
    for (unsigned half = 0; half < 2; ++half) {
      const uint8_t code = input >> (half * 8);
      const uint16_t converted = to == F16X2_TYPE ? e4m3_to_f16(code)
                                 : from == E4M3X2_TYPE
                                     ? e4m3_scaled_to_bf16(code)
                                     : ue8m0_to_bf16(code);
      result.u32 |= uint32_t(converted) << (16 * half);
    }
    thread->set_operand_value(dst, result, B32_TYPE, thread, inst);
    return true;
  }
  if (!narrow(to) || (from != F32_TYPE && from != BF16X2_TYPE) || !satfinite ||
      inst->get_num_operands() != (from == F32_TYPE ? 3u : 2u))
    fail();
  if (rounding != (to == UE8M0X2_TYPE ? RP_OPTION : RN_OPTION) ||
      (to == UE8M0X2_TYPE && from != F32_TYPE))
    fail();
  float high, low;
  if (from == BF16X2_TYPE) {
    const unsigned input = read(inst->src1(), B32_TYPE).u32;
    low = bf16_to_f32(input);
    high = bf16_to_f32(input >> 16);
  } else {
    high = read(inst->src1(), F32_TYPE).f32;
    low = read(inst->src2(), F32_TYPE).f32;
  }
  const auto convert = to == E4M3X2_TYPE   ? f32_to_e4m3_satfinite_rn
                       : to == E2M1X2_TYPE ? f32_to_e2m1_satfinite_rn
                                           : f32_to_ue8m0_satfinite_rp;
  result.u32 =
      (unsigned(convert(high)) << (to == E2M1X2_TYPE ? 4 : 8)) | convert(low);
  thread->set_operand_value(dst, result, to == E2M1X2_TYPE ? B8_TYPE : B16_TYPE,
                            thread, inst);
  return true;
}
} // namespace flash_gpgpu_sim
