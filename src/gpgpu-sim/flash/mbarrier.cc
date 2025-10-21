#include "mbarrier.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

void mbarrier_manager_t::init(gpgpu_sim *gpu,
                              const thread_index_t &thread_index, uint64_t addr,
                              int expected_count) {
  auto id = m_next_id++;
  auto ret = addr_to_mbarrier_map.emplace(
      addr, std::make_unique<mbarrier_t>(id, addr, expected_count));
  assert(ret.second && "mbarrier at the same address already exists");

  DPRINTF_GPU(gpu, MBAR,
              "CTA %u Warp %u reached mbarrier init at address 0x%x with "
              "expected count %u\n",
              thread_index.sw_cta_id, thread_index.sw_warp_id, addr,
              expected_count);
}

void mbarrier_manager_t::inval(gpgpu_sim *gpu,
                               const thread_index_t &thread_index,
                               uint64_t addr) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it != addr_to_mbarrier_map.end()) {
    addr_to_mbarrier_map.erase(it);
  } else {
    assert(false && "mbarrier to be invalidated does not exist");
  }
}

bool mbarrier_manager_t::try_wait(gpgpu_sim *gpu,
                                  const thread_index_t &thread_index,
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
      thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
      (uint32_t)addr, parity, mbarrier->m_phase, current_parity,
      mbarrier->m_arrived_count, mbarrier->m_expected_count,
      mbarrier->m_arrived_tx_count, mbarrier->m_expected_tx_count);

  if (parity != current_parity) {
    // This is waiting for previous phase, return true immediately.
    return true;
  }

  // Add the warp_id to the waiting set.
  mbarrier->m_waiting_warps.insert(thread_index.hw_warp_id);
  return false;
}

std::set<int> mbarrier_manager_t::try_advance(
    gpgpu_sim *gpu, const thread_index_t &thread_index, mbarrier_t *mbarrier) {

  if (mbarrier->m_arrived_count == mbarrier->m_expected_count &&
      mbarrier->m_arrived_tx_count == mbarrier->m_expected_tx_count) {
    // Release all waiting warps.
    std::set<int> released_warps = mbarrier->m_waiting_warps;
    mbarrier->m_waiting_warps.clear();
    mbarrier->m_arrived_count = 0;
    mbarrier->m_arrived_tx_count = 0;
    mbarrier->m_expected_tx_count = 0;
    mbarrier->m_phase++;
    DPRINTF_GPU(gpu, MBAR,
                "CTA %d Warp %d mbarrier.id %d at 0x%x all arrived, "
                "releasing %zu warps, moving to phase %d\n",
                thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
                mbarrier->m_addr, released_warps.size(), mbarrier->m_phase);
    return released_warps;
  } else {
    return {};
  }
}

std::set<int> mbarrier_manager_t::arrive(gpgpu_sim *gpu,
                                         const thread_index_t &thread_index,
                                         uint64_t addr, int arrival_count) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to arrive at does not exist");
  }
  auto mbarrier = it->second.get();

  DPRINTF_GPU(
      gpu, MBAR,
      "CTA %d Warp %d mbarrier.arrive id %d at 0x%x with arrival_count %d "
      "arrived count %d/%d tx_count %d/%d\n",
      thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
      (unsigned)addr, arrival_count, mbarrier->m_arrived_count,
      mbarrier->m_expected_count, mbarrier->m_arrived_tx_count,
      mbarrier->m_expected_tx_count);

  mbarrier->m_arrived_count += arrival_count;
  return try_advance(gpu, thread_index, mbarrier);
}

std::set<int>
mbarrier_manager_t::complete_tx(gpgpu_sim *gpu,
                                const thread_index_t &thread_index,
                                uint64_t addr, int completed_tx_count) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to complete tx at does not exist");
  }
  auto mbarrier = it->second.get();

  DPRINTF_GPU(gpu, MBAR,
              "CTA %d Warp %d mbarrier.complete_tx id %d at 0x%x with "
              "completed_tx_count %d arrived tx count %d/%d\n",
              thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
              (unsigned)addr, completed_tx_count, mbarrier->m_arrived_tx_count,
              mbarrier->m_expected_tx_count);

  mbarrier->m_arrived_tx_count += completed_tx_count;
  return try_advance(gpu, thread_index, mbarrier);
}

void mbarrier_manager_t::expect_tx(gpgpu_sim *gpu,
                                   const thread_index_t &thread_index,
                                   uint64_t addr, int expected_tx_count) {
  auto it = addr_to_mbarrier_map.find(addr);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to expect tx does not exist");
  }
  auto mbarrier = it->second.get();
  mbarrier->m_expected_tx_count += expected_tx_count;
  DPRINTF_GPU(gpu, MBAR,
              "CTA %d Warp %d mbarrier.expect_tx id %d at 0x%x increasing "
              "expected tx count by %d to %d\n",
              thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
              (unsigned)addr, expected_tx_count, mbarrier->m_expected_tx_count);
}

} // namespace flash_gpgpu_sim

namespace {
// Some helper functions
std::pair<bool, bool>
parse_mbarrier_arrive_expect_tx_options(const ptx_instruction *pI) {
  bool is_arrive = false;
  bool is_expect_tx = false;
  for (auto op : pI->get_options()) {
    if (op == ARRIVE_OPTION) {
      is_arrive = true;
    }
    if (op == EXPECT_TX_OPTION) {
      is_expect_tx = true;
    }
  }
  assert(is_arrive || is_expect_tx);
  return {is_arrive, is_expect_tx};
}
} // namespace

void handle_mbarrier_inst(const ptx_instruction *pIin,
                          ptx_thread_info *thread) {
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned ctaid = thread->get_cta_uid();
  auto hw_tid = thread->get_hw_tid();

  DPRINTF_GPU(thread->get_gpu(), MBAR,
              "CTA %d Thread %d handling mbarrier inst %s\n", ctaid, hw_tid,
              pIin->to_string().c_str());

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
    assert(expected_count > 0 && "expected count must be positive");
    DPRINTF_GPU(thread->get_gpu(), MBAR,
                "CTA %d Thread %d mbarrier init at address 0x%x with expected "
                "count %u\n",
                ctaid, hw_tid, addr, expected_count);
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

    DPRINTF_GPU(
        thread->get_gpu(), MBAR,
        "CTA %d Thread %d mbarrier.try_wait at address 0x%x with parity %u\n",
        ctaid, hw_tid, addr, parity);
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

  } else if (bar_op == ARRIVE_OPTION || bar_op == EXPECT_TX_OPTION) {

    assert(pI->membar_level() == CTA_OPTION &&
           "Only support shared::cta mbarrier");
    /**
     * arrive and expect_tx may be combined into single instruction.
     */
    auto [is_arrive, is_expect_tx] =
        parse_mbarrier_arrive_expect_tx_options(pI);

    // Now parse the operands.
    auto addr = 0;
    auto arrival_count = 1;
    auto expected_tx_count = 0;
    if (is_arrive && is_expect_tx) {
      assert(pI->get_num_operands() == 3);
      const operand_info &phase_op = pI->dst();
      assert(phase_op.name() == "_" &&
             "Only support sink reg for mbarrier.arrive");

      addr = get_u32_value(pI->src1());
      expected_tx_count = get_u32_value(pI->src2());

      DPRINTF_GPU(thread->get_gpu(), MBAR,
                  "CTA %d Thread %d mbarrier.arrive.expect_tx at address 0x%x "
                  "with expected_tx_count %u\n",
                  ctaid, hw_tid, addr, expected_tx_count);
      pI->set_bar_id(addr);
      pI->set_bar_count(expected_tx_count);

    } else if (is_arrive) {
      assert(pI->get_num_operands() == 3 || pI->get_num_operands() == 2);
      const operand_info &phase_op = pI->dst();
      assert(phase_op.name() == "_" &&
             "Only support sink reg for mbarrier.arrive");

      addr = get_u32_value(pI->src1());
      if (pI->get_num_operands() == 3) {
        arrival_count = get_u32_value(pI->src2());
      }

      if (arrival_count == 0) {
        printf("GPGPU-Sim: mbarrier.arrive with arrival_count 0\n");
        abort();
      }

      DPRINTF_GPU(thread->get_gpu(), MBAR,
                  "CTA %d Thread %d mbarrier.arrive at address 0x%x with "
                  "arrival_count %u\n",
                  ctaid, hw_tid, addr, arrival_count);
      pI->set_bar_id(addr);
      pI->set_bar_count(arrival_count);

    } else if (is_expect_tx) {
      assert(pI->get_num_operands() == 2);

      addr = get_u32_value(pI->dst());
      expected_tx_count = get_u32_value(pI->src1());

      DPRINTF_GPU(thread->get_gpu(), MBAR,
                  "CTA %d Thread %d mbarrier.expect_tx at address 0x%x "
                  "with expected_tx_count %u\n",
                  ctaid, hw_tid, addr, expected_tx_count);
      pI->set_bar_id(addr);
      pI->set_bar_count(expected_tx_count);

    } else {
      printf("GPGPU-Sim: mbarrier.arrive/expect_tx inst invalid options\n");
      abort();
    }

  } else {
    // Placeholder implementation for mbarrier instruction
    // TODO: Implement the mbarrier logic
    printf(
        "GPGPU-Sim: mbarrier instruction encountered (not yet implemented)\n");
    assert(false && "mbarrier not implemented");
  }
}

void barrier_set_t::complete_tx(unsigned cta_id, unsigned warp_id,
                                uint32_t mbarrier_addr,
                                uint32_t completed_tx_count) {

  // We use the logical CTA ID here.
  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);

  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      cta_id, warp_id, logical_cta_id, logical_warp_id};

  auto released_warps = m_mbarrier_manager.complete_tx(
      m_shader->get_gpu(), thread_index, mbarrier_addr, completed_tx_count);
  for (auto w : released_warps) {
    m_warp_at_barrier.reset(w);
  }
}

void barrier_set_t::warp_reaches_mbarrier(unsigned cta_id, unsigned warp_id,
                                          warp_inst_t *inst) {

  auto pI = dynamic_cast<ptx_instruction *>(inst);
  assert(pI && "mbarrier instruction is not ptx_instruction");

  // We use the logical CTA ID here.
  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);

  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      cta_id, warp_id, logical_cta_id, logical_warp_id};

  auto bar_op = pI->barrier_op();
  if (bar_op == INIT_OPTION) {

    auto addr = pI->bar_id;
    auto expected_count = pI->bar_count;
    m_mbarrier_manager.init(m_shader->get_gpu(), thread_index, addr,
                            expected_count);
    return;

  } else if (bar_op == TRY_WAIT_OPTION) {

    auto addr = pI->bar_id;
    auto parity = pI->bar_parity;
    bool released = m_mbarrier_manager.try_wait(m_shader->get_gpu(),
                                                thread_index, addr, parity);
    if (!released) {
      m_warp_at_barrier.set(warp_id);
    }

    return;
  } else if (bar_op == ARRIVE_OPTION || bar_op == EXPECT_TX_OPTION) {

    auto [is_arrive, is_expect_tx] =
        parse_mbarrier_arrive_expect_tx_options(pI);

    if (is_expect_tx && is_arrive) {

      // We have to do expect_tx first, in case arrive releases the barrier.
      auto addr = pI->bar_id;
      auto expected_tx_count = pI->bar_count;
      auto arrival_count = 1;
      m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, addr,
                                   expected_tx_count);

      auto released_warps = m_mbarrier_manager.arrive(
          m_shader->get_gpu(), thread_index, addr, arrival_count);
      for (auto w : released_warps) {
        m_warp_at_barrier.reset(w);
      }

    } else if (is_arrive) {
      auto addr = pI->bar_id;
      auto arrival_count = pI->bar_count;

      auto released_warps = m_mbarrier_manager.arrive(
          m_shader->get_gpu(), thread_index, addr, arrival_count);
      for (auto w : released_warps) {
        m_warp_at_barrier.reset(w);
      }

    } else if (is_expect_tx) {

      auto addr = pI->bar_id;
      auto expected_tx_count = pI->bar_count;
      m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, addr,
                                   expected_tx_count);
    }

    return;
  }

  assert(false && "mbarrier in barrier_set_t not implemented");
}
