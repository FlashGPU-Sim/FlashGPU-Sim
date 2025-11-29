#include "elect.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

void handle_elect_inst(const ptx_instruction *pI, ptx_thread_info *thread) {
  DPRINTF_INST_EXEC(WIP, "elect.sync instruction%s\n", "");
  fflush(stdout);
  
  // elect.sync instruction - elect a leader thread from active threads
  // Format: elect.sync d|p, membermask
  // Returns: 
  //   d = laneid of the elected thread (same value for all threads)
  //   p = True for the elected thread, False for all others
  // 
  // Election is deterministic: lowest numbered lane in membermask is elected.
  
  unsigned laneid = thread->get_hw_tid() % 32;
  
  // Get the destination operand (vector: d|p where d is laneid, p is predicate)
  const operand_info &dst = pI->dst();
  
  // Get the membership mask from src1
  const operand_info &src1 = pI->src1();
  ptx_reg_t membership = thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
  
  // Elect the lowest numbered lane that's in the membership mask
  unsigned elected_lane = 0;
  for (unsigned i = 0; i < 32; i++) {
    if (membership.u32 & (1 << i)) {
      elected_lane = i;
      break;
    }
  }
  
  bool is_elected = (laneid == elected_lane);
  
  // Set the output values per PTX ISA:
  // "laneid of the elected thread is returned in the 32-bit destination operand d"
  // All threads return the same elected_lane value in d
  ptx_reg_t laneid_val;
  ptx_reg_t pred_val;
  
  laneid_val.u32 = elected_lane;  // All threads get the elected lane's id
  pred_val.pred = is_elected ? 0 : 1;  // 0 = true (leader), 1 = false (others)
  
  // The destination is a vector operand (d|p)
  if (dst.is_vector()) {
    const symbol *laneid_sym = dst.vec_symbol(0);  // d - elected laneid
    const symbol *pred_sym = dst.vec_symbol(1);    // p - predicate
    
    thread->set_reg(laneid_sym, laneid_val);
    thread->set_reg(pred_sym, pred_val);
  } else {
    // Fallback: just set the destination as laneid
    thread->set_operand_value(dst, laneid_val, U32_TYPE, thread, pI);
  }
}
