#include "ptx_ternary_logic.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "panic.h"
#include "ternary_logic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
void execute_ptx_ternary_logic(const ptx_instruction *inst,
                               ptx_thread_info *thread) {
  const auto &ops = inst->get_operands();
  const auto &dst = inst->dst();
  const auto options = inst->get_options();
  const bool pred =
      inst->get_atomic() == ATOMIC_AND || inst->get_atomic() == ATOMIC_OR;
  if (inst->get_scalar_type().size() != 1 || inst->get_type() != B32_TYPE ||
      ops.size() != (pred ? 6u : 5u) || options.size() != (pred ? 1u : 0u) ||
      (pred ? !dst.is_vector() || dst.get_vect_nelem() != 2
            : !dst.is_reg() || dst.is_vector()) ||
      !ops[4].is_literal() || ops[4].get_type() != int_t ||
      ops[4].get_literal_value().u64 > 255)
    panic("unsupported PTX lop3 form: " + inst->to_string());
  auto read = [&](unsigned n) {
    return thread->get_operand_value(ops[n], dst, B32_TYPE, thread, 1).u32;
  };
  const auto value =
      ternary_logic(read(1), read(2), read(3), ops[4].get_literal_value().u32);
  // Resolve q before writing p: p and q may be the same register.
  const bool q =
      pred &&
      !(thread->get_operand_value(ops[5], dst, PRED_TYPE, thread, 1).pred & 1);
  ptx_reg_t data;
  data.u64 = value;
  if (pred) {
    thread->set_reg(dst.vec_symbol(0), data); // set_reg discards the '_' sink.
    data.pred =
        !ternary_logic_predicate(value, q, inst->get_atomic() == ATOMIC_AND);
    thread->set_reg(dst.vec_symbol(1), data);
  } else {
    thread->set_operand_value(dst, data, B32_TYPE, thread, inst);
  }
}
} // namespace flash_gpgpu_sim
