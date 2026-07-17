#include "elect.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

void handle_elect_inst(const ptx_instruction *pI, core_t *core,
                       const warp_inst_t &inst) {
  // elect.sync instruction - elect a leader thread from active threads
  // Format: elect.sync d|p, membermask
  // Returns:
  //   d = laneid of the elected thread (same value for all threads)
  //   p = True for the elected thread, False for all others
  //
  // Election is deterministic: lowest numbered lane in membermask is elected.

  // Get the destination operand (vector: d|p where d is laneid, p is predicate)
  const operand_info &dst = pI->dst();

  // Get the membership mask from src1
  const operand_info &src1 = pI->src1();
  const auto active_mask = inst.get_active_mask();
  if (!active_mask.any())
    return;

  unsigned warp_id = inst.warp_id_func();
  unsigned warp_size = core->get_warp_size();
  ptx_thread_info **threads = core->get_thread_info();

  ptx_reg_t membership;
  membership.u32 = 0;
  for (unsigned lane = 0; lane < warp_size; lane++) {
    if (!active_mask.test(lane))
      continue;
    ptx_thread_info *thread = threads[warp_id * warp_size + lane];
    membership = thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
    break;
  }

  // Elect the lowest numbered active lane that's in the membership mask.
  unsigned elected_lane = 0;
  bool found = false;
  for (unsigned i = 0; i < warp_size; i++) {
    if (active_mask.test(i) && (membership.u32 & (1u << i))) {
      elected_lane = i;
      found = true;
      break;
    }
  }

  // Set the output values per PTX ISA:
  // "laneid of the elected thread is returned in the 32-bit destination operand
  // d" All threads return the same elected_lane value in d
  for (unsigned lane = 0; lane < warp_size; lane++) {
    if (!active_mask.test(lane))
      continue;

    ptx_thread_info *thread = threads[warp_id * warp_size + lane];
    bool is_elected = found && (lane == elected_lane);

    ptx_reg_t laneid_val;
    ptx_reg_t pred_val;
    laneid_val.u32 = elected_lane;
    pred_val.pred = is_elected ? 0 : 1; // 0 = true (leader), 1 = false

    if (dst.is_vector()) {
      const symbol *laneid_sym = dst.vec_symbol(0);
      const symbol *pred_sym = dst.vec_symbol(1);

      thread->set_reg(laneid_sym, laneid_val);
      thread->set_reg(pred_sym, pred_val);
    } else {
      thread->set_operand_value(dst, laneid_val, U32_TYPE, thread, pI);
    }
  }
}
