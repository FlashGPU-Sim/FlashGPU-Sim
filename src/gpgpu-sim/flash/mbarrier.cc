#include "mbarrier.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

void mbarrier_manager_t::init(gpgpu_sim *gpu, int cta_id, int warp_id,
                              uint64_t addr, int expected_count) {
  auto id = m_next_id++;
  auto ret = addr_to_mbarrier_map.emplace(
      addr, std::make_unique<mbarrier_t>(id, addr, expected_count));
  assert(ret.second && "mbarrier at the same address already exists");

  DPRINTF_GPU(gpu, MBAR,
              "CTA %u Warp %u reached mbarrier init at address 0x%x with "
              "expected count %u\n",
              cta_id, warp_id, addr, expected_count);
}

void mbarrier_manager_t::inval(gpgpu_sim *gpu, int cta_id, int warp_id,
                               uint64_t addr) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it != addr_to_mbarrier_map.end()) {
    addr_to_mbarrier_map.erase(it);
  } else {
    assert(false && "mbarrier to be invalidated does not exist");
  }
}

bool mbarrier_manager_t::try_wait(gpgpu_sim *gpu, int cta_id, int warp_id,
                                  uint64_t addr, int parity) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to wait on does not exist");
  }
  auto mbarrier = it->second.get();
  auto current_parity = mbarrier->m_phase & 1;

  DPRINTF_GPU(
      gpu, MBAR,
      "CTA %d Warp %d mbarrier.try_wait id %d at 0x%x with parity %d "
      "(current phase %d parity %d) arrived count %d/%d tx_count %d/%d\n",
      cta_id, warp_id, mbarrier->m_id, (unsigned)addr, parity,
      mbarrier->m_phase, current_parity, mbarrier->m_arrived_count,
      mbarrier->m_expected_count, mbarrier->m_arrived_tx_count,
      mbarrier->m_expected_tx_count);

  if (parity != current_parity) {
    // This is waiting for previous phase, return true immediately.
    return true;
  }

  // Add the warp_id to the waiting set.
  mbarrier->m_waiting_warps.insert(warp_id);
  return false;
}
} // namespace flash_gpgpu_sim

void handle_mbarrier_inst(const ptx_instruction *pIin,
                          ptx_thread_info *thread) {
  DPRINTF_GPU(thread->get_gpu(), MBAR, "handling mbarrier instruction %s\n",
              pIin->to_string().c_str());

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned ctaid = thread->get_cta_uid();

  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  if (bar_op == INIT_OPTION) {
    assert(pI->get_num_operands() == 2);
    assert(pI->membar_level() == CTA_OPTION &&
           "Only support shared::cta mbarrier");
    // So weird, pI->dst() is always the first operand.
    const operand_info &addr_op = pI->dst();
    const operand_info &expected_count_op = pI->src1();
    auto addr = get_u32_value(addr_op);
    auto expected_count = get_u32_value(expected_count_op);
    DPRINTF_GPU(thread->get_gpu(), MBAR,
                "mbarrier init at address 0x%x with expected count %u\n", addr,
                expected_count);
    pI->set_bar_id(addr);
    pI->set_bar_count(expected_count);
  } else if (bar_op == TRY_WAIT_OPTION) {

    assert(pI->parity_op() && "Only support parity op of mbarrier.try_wait");
    assert(pI->membar_level() == CTA_OPTION &&
           "Only support shared::cta mbarrier");

    assert(pI->get_num_operands() == 3);

    const operand_info &addr_op = pI->src1();
    const operand_info &parity_op = pI->src2();
    auto addr = get_u32_value(addr_op);
    auto parity = get_u32_value(parity_op) & 1;

    DPRINTF_GPU(thread->get_gpu(), MBAR,
                "mbarrier.try_wait at address 0x%x with parity %u\n", addr,
                parity);
    pI->set_bar_id(addr);
    pI->set_bar_parity(parity);

    /**
     * So far we always block until the mbarrier is released,
     * so the destination predication reg is always set to true.
     * ! PTXPlus inverts the zero flag -- 0 means true, 1 means false !
     *
     * TODO: Support timeout feature of mbarrier.try_wait.
     */
    ptx_reg_t true_pred;
    true_pred.pred = 0;
    thread->set_operand_value(pI->dst(), true_pred, PRED_TYPE, thread, pI);

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
    m_mbarrier_manager.init(m_shader->get_gpu(), cta_id, warp_id, addr,
                            expected_count);
    return;

  } else if (bar_op == TRY_WAIT_OPTION) {

    auto addr = pI->bar_id;
    auto parity = pI->bar_parity;
    bool released = m_mbarrier_manager.try_wait(m_shader->get_gpu(), cta_id,
                                                warp_id, addr, parity);
    if (!released) {
      m_warp_at_barrier.set(warp_id);
    }

    return;
  }

  assert(false && "mbarrier in barrier_set_t not implemented");
}
