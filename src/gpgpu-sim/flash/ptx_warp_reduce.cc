#include "ptx_warp_reduce.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "panic.h"
#include "warp_reduce.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
void execute_ptx_redux(const ptx_instruction *pI, core_t *core,
                       warp_inst_t &inst) {
  const auto fail = [&](const char *why) {
    panic(std::string(why) + ": " + pI->to_string());
  };
  if (pI->get_num_operands() != 3)
    fail("redux requires destination, source, membermask");
  const unsigned type = pI->get_type(), operation = pI->get_atomic();
  reduce32_operation op{reduce32_kind::add};
  bool sync = false;
  for (auto opt : pI->get_options()) {
    if (opt == SYNC_OPTION)
      sync = true;
    else if (opt == ABS_OPTION && type == F32_TYPE)
      op.absolute = true;
    else if (opt == NAN_OPTION && type == F32_TYPE)
      op.propagate_nan = true;
    else if (opt != ATOMIC_ADD && opt != ATOMIC_MIN && opt != ATOMIC_MAX &&
             opt != ATOMIC_AND && opt != ATOMIC_OR && opt != ATOMIC_XOR)
      fail("unsupported redux modifier");
  }
  if (!sync)
    fail("redux requires sync");
  if (type == B32_TYPE) {
    if (operation == ATOMIC_AND)
      op.kind = reduce32_kind::bit_and;
    else if (operation == ATOMIC_OR)
      op.kind = reduce32_kind::bit_or;
    else if (operation == ATOMIC_XOR)
      op.kind = reduce32_kind::bit_xor;
    else
      fail("redux.b32 requires bitwise operation");
  } else if (type == S32_TYPE || type == U32_TYPE || type == F32_TYPE) {
    if (operation == ATOMIC_ADD && type != F32_TYPE)
      op.kind = reduce32_kind::add;
    else if (operation == ATOMIC_MIN)
      op.kind = type == F32_TYPE   ? reduce32_kind::min_f
                : type == S32_TYPE ? reduce32_kind::min_s
                                   : reduce32_kind::min_u;
    else if (operation == ATOMIC_MAX)
      op.kind = type == F32_TYPE   ? reduce32_kind::max_f
                : type == S32_TYPE ? reduce32_kind::max_s
                                   : reduce32_kind::max_u;
    else
      fail("unsupported redux arithmetic operation");
  } else
    fail("unsupported redux type");
  auto threads =
      core->get_thread_info() + inst.warp_id_func() * core->get_warp_size();
  uint32_t active = 0, exited = 0, values[32]{}, masks[32]{}, result[32]{};
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!threads[lane] || threads[lane]->is_done()) {
      exited |= 1u << lane;
      continue;
    }
    if (inst.get_active_mask().test(lane) &&
        threads[lane]->predicate_allows(*pI))
      active |= 1u << lane;
  }
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!(active & (1u << lane)))
      continue;
    auto t = threads[lane];
    values[lane] = t->get_operand_value(pI->src1(), pI->dst(), type, t, 1).u32;
    masks[lane] =
        t->get_operand_value(pI->src2(), pI->dst(), U32_TYPE, t, 1).u32 &
        ~exited;
  }
  if (!reduce32_groups(op, active, values, masks, result))
    fail("redux requires matching groups of co-issued lanes; divergent "
         "rendezvous is not supported");
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (!(active & (1u << lane)))
      continue;
    ptx_reg_t value;
    value.u32 = result[lane];
    threads[lane]->set_operand_value(pI->dst(), value, type, threads[lane], pI);
  }
}
} // namespace flash_gpgpu_sim
