#include "tma.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread) {
  // Currently no TMA instructions are defined.

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);

  DPRINTF_INST_EXEC(TMA, "inst %s\n", pI->to_string().c_str());

  bool is_commit_group = false;
  bool is_wait_group = false;
  for (auto op : pI->get_options()) {
    if (op == COMMIT_GROUP_OPTION) {
      is_commit_group = true;
    }
    if (op == WAIT_GROUP_OPTION) {
      is_wait_group = true;
    }
  }

  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };
  auto get_u64_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
    return reg.u64;
  };

  if (!is_commit_group && !is_wait_group) {
    // This is TMA copy instruction.
    const auto &options = pI->get_options();
    if (options.size() != 3) {
      for (auto op : options) {
        DPRINTF_INST_EXEC(TMA, "option: %d\n", op);
      }
    }
    assert(options.size() >= 3 &&
           "TMA copy must have at least dst, src, completion_mechanism.");
    // ! Ignore cache hint for now.
    auto option_iter = options.begin();
    auto dst_option = *option_iter;
    ++option_iter;
    auto src_option = *option_iter;
    ++option_iter;
    auto completion_option = *option_iter;
    ++option_iter;

    if (dst_option == CTA_OPTION && src_option == GLOBAL_OPTION &&
        completion_option == TMA_MBAR_COMPLETE_BYTES) {
      // Handle shared::cta <- global TMA copy inst with MBAR completion.

      // Extract operands.
      auto dst_addr = get_u32_value(pI->dst());
      auto src_addr = get_u64_value(pI->src1());
      auto size_in_bytes = get_u32_value(pI->src2());
      auto mbar_addr = get_u32_value(pI->src3());

      // Check alignment to 16 bytes.
      if (dst_addr % 16 != 0 || src_addr % 16 != 0 || size_in_bytes % 16 != 0) {
        DPRINTF_INST_EXEC(
            TMA, "unaligned TMA copy dst=0x%x, src=0x%llx, size_in_bytes=%u\n",
            dst_addr, src_addr, size_in_bytes);
        abort();
      }

      DPRINTF_INST_EXEC(TMA,
                        "TMA shared::cta <- global dst=0x%x, src=0x%llx, "
                        "size_in_bytes=%u, mbar=0x%x\n",
                        dst_addr, src_addr, size_in_bytes, mbar_addr);

      inst_t::tma_dyn_info tma_info{
          .dst_addr = dst_addr,
          .src_addr = src_addr,
          .size_in_bytes = size_in_bytes,
          .mbar_addr = mbar_addr,
      };
      pI->set_tma_info(tma_info);

    } else {
      assert(false && "Unsupported TMA copy instruction");
    }

  } else if (is_commit_group) {
    // Handle TMA commit group instruction.
    assert(false && "TMA instructions are not implemented yet");
  } else if (is_wait_group) {
    // Handle TMA wait group instruction.
    assert(false && "TMA instructions are not implemented yet");
  } else {
    assert(false && "Unrecognized TMA instruction");
  }
}