#include "mbarrier.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

mbarrier_t *mbarrier_manager_t::init(uint64_t addr, int expected_count) {
  auto id = m_next_id++;
  auto ret = addr_to_mbarrier_map.emplace(
      addr, std::make_unique<mbarrier_t>(id, addr, expected_count));
  assert(ret.second && "mbarrier at the same address already exists");
  return ret.first->second.get();
}
void mbarrier_manager_t::inval(uint64_t addr) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it != addr_to_mbarrier_map.end()) {
    addr_to_mbarrier_map.erase(it);
  } else {
    assert(false && "mbarrier to be invalidated does not exist");
  }
}

} // namespace flash_gpgpu_sim

void handle_mbarrier_inst(const ptx_instruction *pIin,
                          ptx_thread_info *thread) {
  DPRINTF_GPU(thread->get_gpu(), MBAR, "handling mbarrier instruction %s\n",
              pIin->to_string().c_str());

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned ctaid = thread->get_cta_uid();

  if (bar_op == INIT_OPTION) {
    assert(pI->get_num_operands() == 2);
    // So weird, pI->dst() is always the first operand.
    const operand_info &addr_op = pI->dst();
    const operand_info &expected_count_op = pI->src1();
    ptx_reg_t addr_reg =
        thread->get_operand_value(addr_op, addr_op, U32_TYPE, thread, 0);
    ptx_reg_t expected_count_reg = thread->get_operand_value(
        expected_count_op, expected_count_op, U32_TYPE, thread, 0);
    auto addr = addr_reg.u32;
    auto expected_count = expected_count_reg.u32;
    DPRINTF_GPU(thread->get_gpu(), MBAR,
                "mbarrier init at address 0x%x with expected count %u\n", addr,
                expected_count);
    pI->set_bar_id(addr);
    pI->set_bar_count(expected_count);
  } else {
    // Placeholder implementation for mbarrier instruction
    // TODO: Implement the mbarrier logic
    printf(
        "GPGPU-Sim: mbarrier instruction encountered (not yet implemented)\n");
    assert(false && "mbarrier not implemented");
  }
}

void barrier_set_t::warp_reaches_mbarrier(unsigned cta_id, unsigned warp_id,
                                          warp_inst_t *inst) {

  auto pI = dynamic_cast<ptx_instruction *>(inst);
  assert(pI && "mbarrier instruction is not ptx_instruction");

  auto bar_op = pI->barrier_op();
  if (bar_op == INIT_OPTION) {
    auto addr = pI->bar_id;
    auto expected_count = pI->bar_count;
    m_mbarrier_manager.init(addr, expected_count);
    DPRINTF_GPU(m_shader->get_gpu(), MBAR,
                "CTA %u Warp %u reached mbarrier init at address 0x%x with "
                "expected count %u\n",
                cta_id, warp_id, addr, expected_count);
    return;
  }

  assert(false && "mbarrier in barrier_set_t not implemented");
}
