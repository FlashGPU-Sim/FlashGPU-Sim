#include "mbarrier.h"
#include "cluster_noc.h"

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"
#include <cstdlib>

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

void mbarrier_manager_t::init(gpgpu_sim *gpu,
                              const thread_index_t &thread_index, uint64_t addr,
                              int expected_count) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto existing = addr_to_mbarrier_map.find(key);
  if (existing != addr_to_mbarrier_map.end()) {
    auto *mbarrier = existing->second.get();
    if (mbarrier->m_expected_count == expected_count) {
      return;
    }

    printf("MBARRIER INIT COLLISION: CTA %u (hw_cta=%u) Warp %u trying to init "
           "mbarrier at addr 0x%lx, but it already exists!\n",
           thread_index.sw_cta_id, thread_index.hw_cta_id,
           thread_index.sw_warp_id, (unsigned long)addr);
    printf("  Existing mbarrier: id=%d, addr=0x%lx, expected_count=%d\n",
           mbarrier->m_id, (unsigned long)mbarrier->m_addr,
           mbarrier->m_expected_count);
    printf("  New mbarrier: expected_count=%d\n", expected_count);
    fflush(stdout);
    assert(false && "mbarrier at the same address already exists");
  }

  auto id = m_next_id++;
  auto ret = addr_to_mbarrier_map.emplace(
      key, std::make_unique<mbarrier_t>(id, thread_index.hw_cta_id,
                                        thread_index.sw_cta_id, addr,
                                        expected_count));
  assert(ret.second && "mbarrier at the same address already exists");

  GPPRINTF_GPU(gpu, MBAR,
               "CTA %u Warp %u reached mbarrier init at address 0x%llx with "
               "expected count %u\n",
               thread_index.sw_cta_id, thread_index.sw_warp_id,
               (unsigned long long)addr, expected_count);
}

void mbarrier_manager_t::inval(gpgpu_sim *gpu,
                               const thread_index_t &thread_index,
                               uint64_t addr) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it != addr_to_mbarrier_map.end()) {
    addr_to_mbarrier_map.erase(it);
  } else {
    assert(false && "mbarrier to be invalidated does not exist");
  }
}

void mbarrier_manager_t::cleanup_cta(unsigned hw_cta_id) {
  // Remove all mbarriers for this hw_cta_id to prevent stale barriers when
  // the hw_cta_id gets recycled for a new CTA.
  for (auto it = addr_to_mbarrier_map.begin();
       it != addr_to_mbarrier_map.end();) {
    if (it->second->m_hw_cta_id == (int)hw_cta_id) {
      it = addr_to_mbarrier_map.erase(it);
    } else {
      ++it;
    }
  }
}

void mbarrier_manager_t::dump() const {
  printf("  mbarriers: %zu\n", addr_to_mbarrier_map.size());
  for (const auto &entry : addr_to_mbarrier_map) {
    const auto *mbarrier = entry.second.get();
    printf("    sw_cta=%d hw_cta=%d id=%d addr=0x%llx expected=%d "
           "pending_arrivals=%d tx_count=%d phase=%d parity=%d "
           "waiting_warps:",
           mbarrier->m_sw_cta_id, mbarrier->m_hw_cta_id, mbarrier->m_id,
           (unsigned long long)mbarrier->m_addr, mbarrier->m_expected_count,
           mbarrier->m_pending_arrival_count, mbarrier->m_tx_count,
           mbarrier->m_phase, mbarrier->m_phase & 1);
    for (int warp_id : mbarrier->m_waiting_warps) {
      printf(" %d", warp_id);
    }
    printf("\n");
  }
}

bool mbarrier_manager_t::try_wait(gpgpu_sim *gpu,
                                  const thread_index_t &thread_index,
                                  uint64_t addr, int parity) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to wait on does not exist");
  }
  auto mbarrier = it->second.get();
  auto current_parity = mbarrier->m_phase & 1;

  GPPRINTF_GPU(
      gpu, MBAR,
      "CTA %d Warp %d mbarrier.try_wait id %d at 0x%x with parity %d "
      "(current phase %d parity %d) pending arrivals %d/%d tx_count %d\n",
      thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
      (uint32_t)addr, parity, mbarrier->m_phase, current_parity,
      mbarrier->m_pending_arrival_count, mbarrier->m_expected_count,
      mbarrier->m_tx_count);

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

  if (mbarrier->m_pending_arrival_count == 0 && mbarrier->m_tx_count == 0) {
    // Release all waiting warps.
    std::set<int> released_warps = mbarrier->m_waiting_warps;
    mbarrier->m_waiting_warps.clear();
    mbarrier->m_pending_arrival_count = mbarrier->m_expected_count;
    mbarrier->m_phase++;
    GPPRINTF_GPU(gpu, MBAR,
                 "CTA %d Warp %d mbarrier.id %d at 0x%llx all arrived, "
                 "releasing %zu warps, moving to phase %d\n",
                 thread_index.sw_cta_id, thread_index.sw_warp_id,
                 mbarrier->m_id, (unsigned long long)mbarrier->m_addr,
                 released_warps.size(), mbarrier->m_phase);
    return released_warps;
  } else {
    return {};
  }
}

bool mbarrier_manager_t::register_remote_wait(
    gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
    int parity, unsigned src_cid, unsigned src_hw_cta, unsigned src_warp_id) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    // Barrier not yet inited / already cleaned: treat as not satisfied so
    // waiter stays blocked (caller may retry) — or success if gone after
    // complete? Prefer false (keep waiting) only if we expect late init.
    // If barrier is gone after phase done, success is safer for try_wait.
    GPPRINTF_GPU(gpu, MBAR,
                 "remote try_wait: no barrier at sw_cta=%d addr=0x%llx "
                 "(treat as satisfied)\n",
                 thread_index.sw_cta_id, (unsigned long long)addr);
    return true;
  }
  auto *mbarrier = it->second.get();
  const int current_parity = mbarrier->m_phase & 1;
  if (parity != current_parity) {
    return true; // already advanced past this parity
  }
  remote_waiter_t w;
  w.src_cid = src_cid;
  w.src_hw_cta = src_hw_cta;
  w.src_warp_id = src_warp_id;
  w.parity = parity;
  mbarrier->m_remote_waiters.push_back(w);
  GPPRINTF_GPU(gpu, MBAR,
               "remote try_wait registered: owner sw_cta=%d addr=0x%llx "
               "from cid=%u warp=%u parity=%d\n",
               thread_index.sw_cta_id, (unsigned long long)addr, src_cid,
               src_warp_id, parity);
  return false;
}

std::vector<mbarrier_manager_t::remote_waiter_t>
mbarrier_manager_t::take_satisfied_remote_waiters(int sw_cta_id,
                                                  uint64_t addr) {
  std::vector<remote_waiter_t> out;
  auto key = std::make_pair(sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end())
    return out;
  auto *mbarrier = it->second.get();
  const int current_parity = mbarrier->m_phase & 1;
  for (auto wit = mbarrier->m_remote_waiters.begin();
       wit != mbarrier->m_remote_waiters.end();) {
    if (wit->parity != current_parity) {
      out.push_back(*wit);
      wit = mbarrier->m_remote_waiters.erase(wit);
    } else {
      ++wit;
    }
  }
  return out;
}

std::set<int> mbarrier_manager_t::arrive(gpgpu_sim *gpu,
                                         const thread_index_t &thread_index,
                                         uint64_t addr, int arrival_count) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    // Remote NoC arrives can race ahead of init visibility across SMs.
    // Soft-fail rather than aborting the whole simulation.
    GPPRINTF_GPU(gpu, MBAR,
                 "CTA %d Warp %d mbarrier.arrive at 0x%llx: barrier missing "
                 "(ignored)\n",
                 thread_index.sw_cta_id, thread_index.sw_warp_id,
                 (unsigned long long)addr);
    printf("GPGPU-Sim WARNING: mbarrier.arrive at missing barrier "
           "sw_cta=%d addr=0x%llx (ignored)\n",
           thread_index.sw_cta_id, (unsigned long long)addr);
    return {};
  }
  auto mbarrier = it->second.get();

  GPPRINTF_GPU(
      gpu, MBAR,
      "CTA %d Warp %d mbarrier.arrive id %d at 0x%x with arrival_count %d "
      "pending arrivals %d/%d tx_count %d\n",
      thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
      (unsigned)addr, arrival_count, mbarrier->m_pending_arrival_count,
      mbarrier->m_expected_count, mbarrier->m_tx_count);

  if (arrival_count >= mbarrier->m_pending_arrival_count) {
    mbarrier->m_pending_arrival_count = 0;
  } else {
    mbarrier->m_pending_arrival_count -= arrival_count;
  }
  return try_advance(gpu, thread_index, mbarrier);
}

std::set<int>
mbarrier_manager_t::complete_tx(gpgpu_sim *gpu,
                                const thread_index_t &thread_index,
                                uint64_t addr, int completed_tx_count) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    // Delayed TMA complete_tx can race with mbarrier.inval after the phase
    // already advanced (e.g. peer try_complete or prior local complete).
    // Treat as a no-op rather than hard-failing the simulator.
    GPPRINTF_GPU(gpu, MBAR,
                 "CTA %d Warp %d mbarrier.complete_tx at 0x%x: barrier gone "
                 "(already inval'd or never armed); ignoring\n",
                 thread_index.sw_cta_id, thread_index.sw_warp_id,
                 (unsigned)addr);
    return {};
  }
  auto mbarrier = it->second.get();

  GPPRINTF_GPU(gpu, MBAR,
               "CTA %d Warp %d mbarrier.complete_tx id %d at 0x%x with "
               "completed_tx_count %d pending tx count %d\n",
               thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
               (unsigned)addr, completed_tx_count, mbarrier->m_tx_count);

  if (completed_tx_count >= mbarrier->m_tx_count) {
    mbarrier->m_tx_count = 0;
  } else {
    mbarrier->m_tx_count -= completed_tx_count;
  }
  return try_advance(gpu, thread_index, mbarrier);
}

std::set<int> mbarrier_manager_t::try_complete_tx_if_pending(
    gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
    int completed_tx_count) {
  // Map is keyed by sw_cta_id (same as init / complete_tx / expect_tx).
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    return {};
  }
  auto mbarrier = it->second.get();
  // Only apply peer completion when this CTA still has outstanding TMA
  // transaction bytes (m_tx_count). After a local complete_tx has drained
  // m_tx_count to 0, a second completion must not re-apply.
  if (mbarrier->m_tx_count <= 0) {
    return {};
  }
  return complete_tx(gpu, thread_index, addr, completed_tx_count);
}

void mbarrier_manager_t::expect_tx(gpgpu_sim *gpu,
                                   const thread_index_t &thread_index,
                                   uint64_t addr, int expected_tx_count) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to expect tx does not exist");
  }
  auto mbarrier = it->second.get();
  mbarrier->m_tx_count += expected_tx_count;
  GPPRINTF_GPU(gpu, MBAR,
               "CTA %d Warp %d mbarrier.expect_tx id %d at 0x%x increasing "
               "expected tx count by %d to %d\n",
               thread_index.sw_cta_id, thread_index.sw_warp_id, mbarrier->m_id,
               (unsigned)addr, expected_tx_count, mbarrier->m_tx_count);
}

} // namespace flash_gpgpu_sim

namespace {
// Some helper functions
bool is_valid_mbarrier_info(const inst_t::mbarrier_info_t &info) {
  return info.bar_id != (unsigned)-1;
}

bool mbarrier_trace_enabled() {
  const char *trace = getenv("FLASHGPU_SIM_MBARRIER_TRACE");
  return trace != nullptr && trace[0] != '\0' && trace[0] != '0';
}

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
  auto laneid = thread->get_laneid();

  // printf("handling mbarrier inst %s\n", pIin->to_string().c_str());
  // fflush(stdout);

  GPPRINTF_GPU(thread->get_gpu(), MBAR,
               "CTA %d Thread %d (lane %u) handling mbarrier inst %s\n", ctaid,
               hw_tid, laneid, pIin->to_string().c_str());
  fflush(stdout);

  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  // Helper to check if membar_level indicates shared memory scope.
  // .shared::cta is parsed as CTA_OPTION and sets membar_level.
  // .shared (without ::cta) is parsed as SHARED_DIRECTIVE which sets
  // Only support shared memory in the same CTA for now.
  auto is_shared_level = [&](uint32_t *addr = nullptr) {
    bool is_shared =
        (pI->membar_level() == CTA_OPTION) || (pI->get_space() == shared_space);

    if (is_shared && addr != nullptr) {
      // Convert relative shared memory offset to absolute generic address
      addr_t absolute_addr = shared_to_generic(thread->get_hw_sid(), *addr);
      if (!isspace_shared(thread->get_hw_sid(), absolute_addr)) {
        // Remote DSM mbarrier: allowed when cluster mbarrier is enabled.
        unsigned owner = 0;
        addr_t off = 0;
        const bool remote = flash_gpgpu_sim::decode_shared_generic(
                                absolute_addr, &owner, &off) &&
                            owner != thread->get_hw_sid();
        const auto *cfg =
            thread->get_core()
                ? static_cast<shader_core_ctx *>(thread->get_core())
                      ->get_config()
                : nullptr;
        if (!(remote && cfg && cfg->gpgpu_mbarrier_cluster_enable &&
              cfg->gpgpu_cluster_noc_enable)) {
          printf("GPGPU-Sim ERROR: mbarrier address 0x%x (absolute 0x%llx) is "
                 "not in SM %u's shared memory.\n"
                 "Enable -gpgpu_cluster_noc_enable and "
                 "-gpgpu_mbarrier_cluster_enable for remote mbarrier.\n",
                 *addr, (unsigned long long)absolute_addr,
                 thread->get_hw_sid());
          fflush(stdout);
          abort();
        }
        // Remote path is allowed; timing routes via cluster NoC.
      }
    }
    return is_shared;
  };

  // Resolve mbarrier address: local smem offset, or remote via generic mapa
  // addr. Sets remote_* fields when the barrier lives on a peer CTA.
  auto resolve_and_fill_mbar_info = [&](uint64_t raw_addr, unsigned count,
                                        bool parity) {
    inst_t::mbarrier_info_t info;
    info.bar_count = count;
    info.bar_parity = parity;
    info.is_remote = false;
    info.bar_id = static_cast<unsigned>(raw_addr);

    auto *core = dynamic_cast<shader_core_ctx *>(thread->get_core());
    const auto *cfg = core ? core->get_config() : nullptr;
    if (core && cfg && cfg->gpgpu_cluster_noc_enable &&
        cfg->gpgpu_mbarrier_cluster_enable && core->get_cluster()) {
      unsigned owner_smid = 0;
      addr_t offset = 0;
      // Prefer full generic (mapa.u64 result). Fallback: local-relative only.
      if (flash_gpgpu_sim::decode_shared_generic(raw_addr, &owner_smid,
                                                 &offset) &&
          owner_smid != thread->get_hw_sid()) {
        auto *cluster = core->get_cluster();
        for (unsigned cid = 0; cid < cluster->num_cores(); cid++) {
          auto *peer = cluster->get_core(cid);
          if (peer->get_sid() != owner_smid)
            continue;
          unsigned group = core->get_cta_cluster_group(thread->get_hw_ctaid());
          for (unsigned slot = 0; slot < MAX_CTA_PER_SHADER; slot++) {
            if (!peer->is_cta_slot_active(slot))
              continue;
            if (group != (unsigned)-1 &&
                peer->get_cta_cluster_group(slot) != group)
              continue;
            info.is_remote = true;
            info.remote_cid = cid;
            info.remote_hw_cta = slot;
            info.bar_id = static_cast<unsigned>(offset);
            break;
          }
          break;
        }
      }
    }
    pI->set_mbarrier_info(laneid, info);
  };

  // Helper to set per-thread mbarrier info (local)
  auto set_thread_mbarrier_info = [&](unsigned addr, unsigned count,
                                      bool parity) {
    resolve_and_fill_mbar_info(addr, count, parity);
  };

  if (bar_op == INIT_OPTION) {
    assert(pI->get_num_operands() == 2);
    // So weird, pI->dst() is always the first operand.
    const operand_info &addr_op = pI->dst();
    const operand_info &expected_count_op = pI->src1();
    auto addr = get_u32_value(addr_op);
    assert(is_shared_level(&addr) && "Only support shared mbarrier");
    auto expected_count = get_u32_value(expected_count_op);
    assert(expected_count > 0 && "expected count must be positive");
    resolve_and_fill_mbar_info(addr, expected_count, false);
    if (pI->get_mbarrier_info(laneid).is_remote) {
      printf("GPGPU-Sim ERROR: mbarrier.init on remote DSM address is not "
             "supported (init must run on the owner CTA).\n");
      abort();
    }
    GPPRINTF_GPU(thread->get_gpu(), MBAR,
                 "CTA %d Thread %d (lane %u) mbarrier init at address 0x%x "
                 "with expected "
                 "count %u\n",
                 ctaid, hw_tid, laneid, addr, expected_count);
    fflush(stdout);
  } else if (bar_op == TRY_WAIT_OPTION) {

    assert(pI->parity_op() && "Only support parity op of mbarrier.try_wait");

    assert((pI->get_num_operands() == 3 || pI->get_num_operands() == 4) &&
           "mbarrier.try_wait expects predicate, address, parity, and optional "
           "timeout");

    const operand_info &addr_op = pI->src1();
    const operand_info &parity_op = pI->src2();
    // Prefer u64 so mapa.u64 generic remote addresses are preserved.
    uint64_t raw_addr =
        thread->get_operand_value(addr_op, addr_op, U64_TYPE, thread, 0).u64;
    uint32_t addr32 = static_cast<uint32_t>(raw_addr);
    // Local-offset path still validates shared level when not remote-generic.
    unsigned owner = 0;
    addr_t off = 0;
    const bool looks_remote =
        flash_gpgpu_sim::decode_shared_generic(raw_addr, &owner, &off) &&
        owner != thread->get_hw_sid();
    if (!looks_remote) {
      assert(is_shared_level(&addr32) && "Only support shared mbarrier");
      raw_addr = addr32;
    }
    auto parity = get_u32_value(parity_op) & 1;

    GPPRINTF_GPU(
        thread->get_gpu(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier.try_wait at address 0x%llx "
        "with parity %u\n",
        ctaid, hw_tid, laneid, (unsigned long long)raw_addr, parity);
    resolve_and_fill_mbar_info(raw_addr, (unsigned)-1, parity);

    /**
     * Inline PTX commonly lowers mbarrier waits to an explicit software loop:
     *
     *   mbarrier.try_wait.parity ..., complete;
     *   @!complete bra waitLoop;
     *
     * The timing model below already blocks and releases the warp at this
     * instruction, so functional execution must let the loop exit. Otherwise,
     * the warp re-enters the same try_wait forever after timing release.
     *
     * ! PTXPlus inverts the zero flag -- 0 means true, 1 means false !
     */
    ptx_reg_t pred;
    pred.pred = 0;
    thread->set_operand_value(pI->dst(), pred, PRED_TYPE, thread, pI);

  } else if (bar_op == COMPLETE_TX_OPTION) {

    assert(pI->get_num_operands() == 2);
    const operand_info &addr_op = pI->dst();
    const operand_info &tx_count_op = pI->src1();
    uint64_t raw_addr =
        thread->get_operand_value(addr_op, addr_op, U64_TYPE, thread, 0).u64;
    auto completed_tx_count = get_u32_value(tx_count_op);
    if (completed_tx_count == 0) {
      printf("GPGPU-Sim: mbarrier.complete_tx with completed_tx_count 0\n");
      abort();
    }

    GPPRINTF_GPU(
        thread->get_gpu(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier.complete_tx at address 0x%llx "
        "with completed_tx_count %u\n",
        ctaid, hw_tid, laneid, (unsigned long long)raw_addr,
        completed_tx_count);

    resolve_and_fill_mbar_info(raw_addr, completed_tx_count, false);

  } else if (bar_op == ARRIVE_OPTION || bar_op == EXPECT_TX_OPTION) {

    /**
     * arrive and expect_tx may be combined into single instruction.
     */
    auto [is_arrive, is_expect_tx] =
        parse_mbarrier_arrive_expect_tx_options(pI);

    // Now parse the operands (u64 preserves mapa generic addresses).
    uint64_t raw_addr = 0;
    auto arrival_count = 1;
    auto expected_tx_count = 0;
    if (is_arrive && is_expect_tx) {
      assert(pI->get_num_operands() == 3);
      const operand_info &phase_op = pI->dst();
      assert(phase_op.name() == "_" &&
             "Only support sink reg for mbarrier.arrive");

      raw_addr =
          thread->get_operand_value(pI->src1(), pI->src1(), U64_TYPE, thread, 0)
              .u64;
      expected_tx_count = get_u32_value(pI->src2());

      GPPRINTF_GPU(thread->get_gpu(), MBAR,
                   "CTA %d Thread %d (lane %u) mbarrier.arrive.expect_tx at "
                   "address 0x%llx "
                   "with expected_tx_count %u\n",
                   ctaid, hw_tid, laneid, (unsigned long long)raw_addr,
                   expected_tx_count);
      resolve_and_fill_mbar_info(raw_addr, expected_tx_count, false);

    } else if (is_arrive) {
      assert(pI->get_num_operands() == 3 || pI->get_num_operands() == 2);
      const operand_info &phase_op = pI->dst();
      assert(phase_op.name() == "_" &&
             "Only support sink reg for mbarrier.arrive");

      raw_addr =
          thread->get_operand_value(pI->src1(), pI->src1(), U64_TYPE, thread, 0)
              .u64;
      if (pI->get_num_operands() == 3) {
        arrival_count = get_u32_value(pI->src2());
      }

      if (arrival_count == 0) {
        printf("GPGPU-Sim: mbarrier.arrive with arrival_count 0\n");
        abort();
      }

      GPPRINTF_GPU(
          thread->get_gpu(), MBAR,
          "CTA %d Thread %d (lane %u) mbarrier.arrive at address 0x%llx with "
          "arrival_count %u\n",
          ctaid, hw_tid, laneid, (unsigned long long)raw_addr, arrival_count);
      resolve_and_fill_mbar_info(raw_addr, arrival_count, false);

    } else if (is_expect_tx) {
      assert(pI->get_num_operands() == 2);

      raw_addr =
          thread->get_operand_value(pI->dst(), pI->dst(), U64_TYPE, thread, 0)
              .u64;
      expected_tx_count = get_u32_value(pI->src1());

      GPPRINTF_GPU(
          thread->get_gpu(), MBAR,
          "CTA %d Thread %d (lane %u) mbarrier.expect_tx at address 0x%llx "
          "with expected_tx_count %u\n",
          ctaid, hw_tid, laneid, (unsigned long long)raw_addr,
          expected_tx_count);
      resolve_and_fill_mbar_info(raw_addr, expected_tx_count, false);

    } else {
      printf("GPGPU-Sim: mbarrier.arrive/expect_tx inst invalid options\n");
      abort();
    }

  } else if (bar_op == INVAL_OPTION) {

    assert(pI->get_num_operands() == 1);
    const operand_info &addr_op = pI->dst();
    auto addr = get_u32_value(addr_op);
    GPPRINTF_GPU(thread->get_gpu(), MBAR,
                 "CTA %d Thread %d (lane %u) mbarrier inval at address 0x%x\n",
                 ctaid, hw_tid, laneid, addr);
    // Set per-thread info
    set_thread_mbarrier_info(addr, (unsigned)-1, false);

  } else {
    // TODO: Implement remaining mbarrier variants as needed
    printf(
        "GPGPU-Sim: mbarrier instruction not implemented: bar_op=%u, inst=%s\n",
        bar_op, pIin->to_string().c_str());
    assert(false && "mbarrier not implemented");
  }
}

void barrier_set_t::release_warps(const std::set<int> &released_warps) {
  if (released_warps.empty())
    return;
  unsigned trywait_latency =
      m_shader->get_config()->gpgpu_mbarrier_trywait_latency;
  const char *trace = getenv("FLASHGPU_SIM_BARRIER_TRACE");
  bool trace_barrier = trace != nullptr && trace[0] != '\0' && trace[0] != '0';
  if (trywait_latency > 0) {
    for (auto w : released_warps) {
      assert_warp_waiting(w, BARRIER_WAIT_MBARRIER, "mbarrier release");
      if (trace_barrier) {
        printf("GPGPU-Sim Cycle %llu: MBAR_RELEASE - schedule warp=%d "
               "latency=%u warp_at_barrier=%s type=%d\n",
               m_shader->get_gpu()->gpu_sim_cycle +
                   m_shader->get_gpu()->gpu_tot_sim_cycle,
               w, trywait_latency, m_warp_at_barrier.to_string().c_str(),
               (w >= 0 && (unsigned)w < m_warp_barrier_type.size())
                   ? (int)m_warp_barrier_type[w]
                   : -1);
      }
      m_pending_warp_releases.push_back(
          {trywait_latency, w, BARRIER_WAIT_MBARRIER});
    }
  } else {
    for (auto w : released_warps) {
      if (trace_barrier) {
        printf("GPGPU-Sim Cycle %llu: MBAR_RELEASE - reset warp=%d "
               "warp_at_barrier_before=%s type=%d\n",
               m_shader->get_gpu()->gpu_sim_cycle +
                   m_shader->get_gpu()->gpu_tot_sim_cycle,
               w, m_warp_at_barrier.to_string().c_str(),
               (w >= 0 && (unsigned)w < m_warp_barrier_type.size())
                   ? (int)m_warp_barrier_type[w]
                   : -1);
      }
      clear_warp_waiting(w, BARRIER_WAIT_MBARRIER, "mbarrier release");
    }
  }
}

void barrier_set_t::cycle() {
  for (auto &entry : m_pending_warp_releases) {
    entry.remaining--;
  }
  // Release warps whose countdown reached zero
  for (int i = m_pending_warp_releases.size() - 1; i >= 0; i--) {
    if (m_pending_warp_releases[i].remaining == 0) {
      int warp_id = m_pending_warp_releases[i].warp_id;
      const char *trace = getenv("FLASHGPU_SIM_BARRIER_TRACE");
      if (trace != nullptr && trace[0] != '\0' && trace[0] != '0') {
        printf("GPGPU-Sim Cycle %llu: MBAR_RELEASE - delayed reset warp=%d "
               "warp_at_barrier_before=%s type=%d\n",
               m_shader->get_gpu()->gpu_sim_cycle +
                   m_shader->get_gpu()->gpu_tot_sim_cycle,
               warp_id, m_warp_at_barrier.to_string().c_str(),
               (warp_id >= 0 && (unsigned)warp_id < m_warp_barrier_type.size())
                   ? (int)m_warp_barrier_type[warp_id]
                   : -1);
      }
      barrier_wait_type_t type = m_pending_warp_releases[i].type;
      const char *reason = (type == BARRIER_WAIT_CP_ASYNC_GROUP)
                               ? "delayed cp.async wait_group release"
                               : "delayed mbarrier release";
      clear_warp_waiting(warp_id, type, reason);
      m_pending_warp_releases.erase(m_pending_warp_releases.begin() + i);
    }
  }
}

void barrier_set_t::complete_tx(unsigned cta_id, unsigned warp_id,
                                uint32_t mbarrier_addr,
                                uint32_t completed_tx_count) {

  // We use the logical CTA ID here.
  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);

  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, logical_cta_id, logical_warp_id};

  auto released_warps = m_mbarrier_manager.complete_tx(
      m_shader->get_gpu(), thread_index, mbarrier_addr, completed_tx_count);
  release_warps(released_warps);
}

void barrier_set_t::try_complete_tx_if_pending(unsigned cta_id,
                                               uint32_t mbarrier_addr,
                                               uint32_t completed_tx_count) {
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  if (w == m_cta_to_warps.end())
    return;

  // Pick any warp in the CTA for logical-id lookup / logging.
  unsigned warp_id = (unsigned)-1;
  for (unsigned i = 0; i < m_max_warps_per_core; i++) {
    if (w->second.test(i)) {
      warp_id = i;
      break;
    }
  }
  if (warp_id == (unsigned)-1)
    return;

  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, logical_cta_id, logical_warp_id};

  auto released_warps = m_mbarrier_manager.try_complete_tx_if_pending(
      m_shader->get_gpu(), thread_index, mbarrier_addr, completed_tx_count);
  release_warps(released_warps);
  notify_remote_waiters(cta_id, mbarrier_addr);
}

// Helper: build thread_index for a CTA using any live warp in that CTA.
static bool barrier_pick_cta_thread_index(
    barrier_set_t *self, shader_core_ctx *shader, unsigned cta_id,
    unsigned max_warps_per_core,
    const std::map<unsigned, warp_set_t> &cta_to_warps,
    flash_gpgpu_sim::mbarrier_manager_t::thread_index_t *out) {
  auto w = cta_to_warps.find(cta_id);
  if (w == cta_to_warps.end())
    return false;
  unsigned warp_id = (unsigned)-1;
  for (unsigned i = 0; i < max_warps_per_core; i++) {
    if (w->second.test(i)) {
      warp_id = i;
      break;
    }
  }
  if (warp_id == (unsigned)-1)
    return false;
  out->hw_cta_id = (int)cta_id;
  out->hw_warp_id = (int)warp_id;
  out->sw_cta_id = shader->get_logical_cta_id(warp_id);
  out->sw_warp_id = shader->get_cta_warp_id(warp_id);
  return true;
}

void barrier_set_t::remote_arrive(unsigned cta_id, uint32_t mbarrier_addr,
                                  uint32_t arrival_count) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{};
  if (!barrier_pick_cta_thread_index(this, m_shader, cta_id,
                                     m_max_warps_per_core, m_cta_to_warps,
                                     &thread_index))
    return;
  auto released = m_mbarrier_manager.arrive(m_shader->get_gpu(), thread_index,
                                            mbarrier_addr, (int)arrival_count);
  release_warps(released);
  notify_remote_waiters(cta_id, mbarrier_addr);
}

void barrier_set_t::remote_expect_tx(unsigned cta_id, uint32_t mbarrier_addr,
                                     uint32_t expected_tx_count) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{};
  if (!barrier_pick_cta_thread_index(this, m_shader, cta_id,
                                     m_max_warps_per_core, m_cta_to_warps,
                                     &thread_index))
    return;
  m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, mbarrier_addr,
                               (int)expected_tx_count);
}

bool barrier_set_t::register_remote_wait(unsigned cta_id,
                                         uint32_t mbarrier_addr, int parity,
                                         unsigned src_cid, unsigned src_hw_cta,
                                         unsigned src_warp_id) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{};
  if (!barrier_pick_cta_thread_index(this, m_shader, cta_id,
                                     m_max_warps_per_core, m_cta_to_warps,
                                     &thread_index))
    return true;
  return m_mbarrier_manager.register_remote_wait(
      m_shader->get_gpu(), thread_index, mbarrier_addr, parity, src_cid,
      src_hw_cta, src_warp_id);
}

void barrier_set_t::notify_remote_waiters(unsigned cta_id,
                                          uint32_t mbarrier_addr) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{};
  if (!barrier_pick_cta_thread_index(this, m_shader, cta_id,
                                     m_max_warps_per_core, m_cta_to_warps,
                                     &thread_index))
    return;
  auto waiters = m_mbarrier_manager.take_satisfied_remote_waiters(
      thread_index.sw_cta_id, mbarrier_addr);
  if (waiters.empty())
    return;
  auto *cluster = m_shader->get_cluster();
  auto *noc = cluster ? cluster->get_cluster_noc() : nullptr;
  if (!noc || !noc->enabled())
    return;
  const unsigned owner_cid =
      m_shader->get_config()->sid_to_cid(m_shader->get_sid());
  for (const auto &w : waiters) {
    noc->inject_mbar_remote(owner_cid, w.src_cid, w.src_hw_cta, mbarrier_addr,
                            flash_gpgpu_sim::cluster_mbar_op::WAIT_DONE, 1,
                            w.src_hw_cta, w.src_warp_id, /*parity=*/1);
  }
}

void barrier_set_t::release_remote_waiter(unsigned warp_id) {
  if (warp_id >= m_max_warps_per_core)
    return;
  // Pay trywait observation latency like a local successful try_wait.
  release_warps({static_cast<int>(warp_id)});
}

void barrier_set_t::warp_reaches_mbarrier(unsigned cta_id, unsigned warp_id,
                                          const ptx_instruction *pI,
                                          const warp_inst_t *dynamic_inst,
                                          const active_mask_t &active_mask) {

  // We use the logical CTA ID here.
  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);

  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, logical_cta_id, logical_warp_id};

  auto bar_op = pI->barrier_op();

  unsigned warp_size = m_shader->get_config()->warp_size;
  const bool trace_mbarrier = mbarrier_trace_enabled();

  // mbarrier.complete_tx is modeled once per warp and therefore requires one
  // uniform set of lane parameters. Other mbarrier operations are thread-level
  // and are handled per lane below.
  auto get_uniform_mbarrier_info = [&](inst_t::mbarrier_info_t &mbar_info,
                                       unsigned &mbar_lane) -> bool {
    bool found = false;
    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(info))
        continue;

      if (!found) {
        mbar_info = info;
        mbar_lane = lane;
        found = true;
        continue;
      }

      const bool matches = info.bar_id == mbar_info.bar_id &&
                           info.bar_count == mbar_info.bar_count &&
                           info.bar_parity == mbar_info.bar_parity &&
                           info.is_remote == mbar_info.is_remote &&
                           info.remote_cid == mbar_info.remote_cid &&
                           info.remote_hw_cta == mbar_info.remote_hw_cta;
      if (!matches) {
        fprintf(stderr,
                "GPGPU-Sim ERROR: non-uniform mbarrier params in CTA %u "
                "warp %u inst %s. lane %u has addr=0x%x count=%u parity=%u; "
                "lane %u has addr=0x%x count=%u parity=%u; active=%s\n",
                cta_id, warp_id, pI->to_string().c_str(), mbar_lane,
                mbar_info.bar_id, mbar_info.bar_count,
                (unsigned)mbar_info.bar_parity, lane, info.bar_id,
                info.bar_count, (unsigned)info.bar_parity,
                active_mask.to_string().c_str());
        fflush(stderr);
      }
      assert(matches && "mbarrier parameters must be uniform across lanes");
    }
    return found;
  };

  if (bar_op == INIT_OPTION) {

    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      auto addr = mbar_info.bar_id;
      auto expected_count = mbar_info.bar_count;

      m_mbarrier_manager.init(m_shader->get_gpu(), thread_index, addr,
                              expected_count);
    }
    return;

  } else if (bar_op == TRY_WAIT_OPTION) {
    // Aggregate per-lane results: SIMT warp waits if any active lane is not
    // released. Call release_warps at most once (multiple lanes must not
    // re-queue the same warp — that trips assert_warp_waiting on delayed
    // clear).
    bool any_active = false;
    bool any_blocking = false;
    bool any_released = false;
    bool any_remote = false;

    auto *cluster = m_shader->get_cluster();
    auto *noc = cluster ? cluster->get_cluster_noc() : nullptr;
    const unsigned src_cid =
        m_shader->get_config()->sid_to_cid(m_shader->get_sid());

    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      any_active = true;
      unsigned addr = mbar_info.bar_id;
      bool parity = mbar_info.bar_parity;

      if (mbar_info.is_remote) {
        any_remote = true;
        any_blocking = true; // always wait for NoC DONE (or hop if immediate)
        if (noc && noc->enabled()) {
          noc->inject_mbar_remote(src_cid, mbar_info.remote_cid,
                                  mbar_info.remote_hw_cta, addr,
                                  flash_gpgpu_sim::cluster_mbar_op::WAIT_REG,
                                  /*count=*/0, cta_id, warp_id, parity ? 1 : 0);
        } else {
          // NoC off: fall back to local try_wait (will assert if bar missing).
          bool released = m_mbarrier_manager.try_wait(
              m_shader->get_gpu(), thread_index, addr, parity);
          if (released)
            any_released = true;
          else
            any_blocking = true;
        }
        continue;
      }

      bool released = m_mbarrier_manager.try_wait(m_shader->get_gpu(),
                                                  thread_index, addr, parity);
      if (trace_mbarrier) {
        printf("GPGPU-Sim Cycle %llu: MBAR_TRY_WAIT - CTA %u Warp %u lane=%u "
               "addr=0x%x parity=%u released=%s active=%s\n",
               m_shader->get_gpu()->gpu_sim_cycle +
                   m_shader->get_gpu()->gpu_tot_sim_cycle,
               cta_id, warp_id, lane, addr, (unsigned)parity,
               released ? "yes" : "no", active_mask.to_string().c_str());
      }
      if (released) {
        any_released = true;
      } else {
        any_blocking = true;
      }
    }

    if (any_active) {
      // Always mark the warp as waiting at the barrier first.
      // - Incomplete: stays blocked until complete_tx/arrive releases it.
      // - Already complete (immediate success): still pay
      //   gpgpu_mbarrier_trywait_latency via release_warps(). Without this,
      //   clock64 microbenches measure only opcode overhead (~15 cycles) vs
      //   ~130 cycles on real Hopper/Blackwell hardware.
      // - Remote: released when WAIT_DONE arrives (release_remote_waiter).
      m_warp_at_barrier.set(warp_id);
      m_warp_barrier_type[warp_id] = BARRIER_WAIT_MBARRIER;
      m_warp_named_barrier_id[warp_id] = (unsigned)-1;
      if (!any_remote && any_released && !any_blocking) {
        release_warps({static_cast<int>(warp_id)});
      }
    }

    return;
  } else if (bar_op == COMPLETE_TX_OPTION) {

    inst_t::mbarrier_info_t mbar_info;
    unsigned lane = 0;
    if (!get_uniform_mbarrier_info(mbar_info, lane))
      return;

    auto addr = mbar_info.bar_id;
    auto completed_tx_count = mbar_info.bar_count;

    if (mbar_info.is_remote) {
      auto *cluster = m_shader->get_cluster();
      auto *noc = cluster ? cluster->get_cluster_noc() : nullptr;
      if (noc && noc->enabled()) {
        const unsigned src_cid =
            m_shader->get_config()->sid_to_cid(m_shader->get_sid());
        noc->inject_mbar_remote(
            src_cid, mbar_info.remote_cid, mbar_info.remote_hw_cta, addr,
            flash_gpgpu_sim::cluster_mbar_op::COMPLETE_TX, completed_tx_count);
      }
      return;
    }

    auto released_warps = m_mbarrier_manager.complete_tx(
        m_shader->get_gpu(), thread_index, addr, completed_tx_count);
    release_warps(released_warps);
    notify_remote_waiters(cta_id, addr);

    return;
  } else if (bar_op == ARRIVE_OPTION || bar_op == EXPECT_TX_OPTION) {

    auto [is_arrive, is_expect_tx] =
        parse_mbarrier_arrive_expect_tx_options(pI);

    auto *cluster = m_shader->get_cluster();
    auto *noc = cluster ? cluster->get_cluster_noc() : nullptr;
    const unsigned src_cid =
        m_shader->get_config()->sid_to_cid(m_shader->get_sid());

    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      auto addr = mbar_info.bar_id;
      auto count = mbar_info.bar_count;

      if (mbar_info.is_remote) {
        if (noc && noc->enabled()) {
          if (is_expect_tx && is_arrive) {
            noc->inject_mbar_remote(
                src_cid, mbar_info.remote_cid, mbar_info.remote_hw_cta, addr,
                flash_gpgpu_sim::cluster_mbar_op::EXPECT_TX, count);
            noc->inject_mbar_remote(
                src_cid, mbar_info.remote_cid, mbar_info.remote_hw_cta, addr,
                flash_gpgpu_sim::cluster_mbar_op::ARRIVE, 1);
          } else if (is_arrive) {
            noc->inject_mbar_remote(
                src_cid, mbar_info.remote_cid, mbar_info.remote_hw_cta, addr,
                flash_gpgpu_sim::cluster_mbar_op::ARRIVE, count);
          } else if (is_expect_tx) {
            noc->inject_mbar_remote(
                src_cid, mbar_info.remote_cid, mbar_info.remote_hw_cta, addr,
                flash_gpgpu_sim::cluster_mbar_op::EXPECT_TX, count);
          }
        }
        continue;
      }

      if (is_expect_tx && is_arrive) {
        // We have to do expect_tx first, in case arrive releases the barrier.
        auto arrival_count = 1;
        m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, addr,
                                     count);

        auto released_warps = m_mbarrier_manager.arrive(
            m_shader->get_gpu(), thread_index, addr, arrival_count);
        release_warps(released_warps);
        notify_remote_waiters(cta_id, addr);

      } else if (is_arrive) {
        auto released_warps = m_mbarrier_manager.arrive(
            m_shader->get_gpu(), thread_index, addr, count);
        release_warps(released_warps);
        notify_remote_waiters(cta_id, addr);

      } else if (is_expect_tx) {
        m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, addr,
                                     count);
      }
    }

    return;
  } else if (bar_op == INVAL_OPTION) {

    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      auto addr = mbar_info.bar_id;
      m_mbarrier_manager.inval(m_shader->get_gpu(), thread_index, addr);
    }
    return;
  }

  assert(false && "mbarrier in barrier_set_t not implemented");
}
