#include "mbarrier.h"
#include "tb_cluster.h"

#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_sim.h"
#include "../gpu-sim.h"
#include "../shader.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

uint64_t mbarrier_saturating_add(uint64_t cycle, uint64_t delta) {
  return cycle > std::numeric_limits<uint64_t>::max() - delta
             ? std::numeric_limits<uint64_t>::max()
             : cycle + delta;
}

uint64_t mbarrier_hint_ns_to_cycles(uint32_t hint_ns, unsigned core_freq_hz) {
  const long double cycles =
      (static_cast<long double>(hint_ns) * core_freq_hz) / 1000000000.0L;
  return static_cast<uint64_t>(std::ceil(cycles));
}

mbarrier_recheck_action_t mbarrier_classify_recheck(bool phase_complete,
                                                    uint64_t cycle,
                                                    uint64_t deadline_cycle) {
  if (phase_complete)
    return mbarrier_recheck_action_t::RETURN_TRUE;
  if (cycle >= deadline_cycle)
    return mbarrier_recheck_action_t::RETURN_FALSE;
  return mbarrier_recheck_action_t::KEEP_SLEEPING;
}

uint64_t mbarrier_wake_on_phase_notification(uint64_t scheduled_wake_cycle,
                                             uint64_t notification_cycle) {
  return scheduled_wake_cycle < notification_cycle ? scheduled_wake_cycle
                                                   : notification_cycle;
}

bool mbarrier_should_delay_phase_wakeup(bool suspended,
                                        bool phase_notification_pending,
                                        bool all_true, unsigned latency) {
  return suspended && phase_notification_pending && all_true && latency > 0;
}

bool mbarrier_latches_initial_predicate(bool has_time_hint,
                                        unsigned predicate_latency) {
  return !has_time_hint && predicate_latency > 0;
}

uint64_t mbarrier_result_release_cycle(uint64_t issue_cycle,
                                       uint64_t resolution_cycle,
                                       unsigned predicate_latency,
                                       unsigned phase_wakeup_latency) {
  return std::max(
      mbarrier_saturating_add(issue_cycle, predicate_latency),
      mbarrier_saturating_add(resolution_cycle, phase_wakeup_latency));
}

void mbarrier_manager_t::init(gpgpu_sim *gpu,
                              const thread_index_t &thread_index, uint64_t addr,
                              int expected_count) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto existing = addr_to_mbarrier_map.find(key);
  if (existing != addr_to_mbarrier_map.end()) {
    // PTX mbarrier.init reinitializes the object. A no-op left a completed
    // phase with pending_arrival reset to expected, so a later try_wait
    // on parity 0 parked forever (init/arrive/try_wait loops).
    auto *mbarrier = existing->second.get();
    mbarrier->m_expected_count = expected_count;
    mbarrier->m_pending_arrival_count = expected_count;
    mbarrier->m_tx_count = 0;
    mbarrier->m_phase = 0;
    mbarrier->m_waiting_warps.clear();
    mbarrier->m_remote_waiters.clear();
    GPPRINTF_GPU(gpu, MBAR,
                 "CTA %u Warp %u re-init mbarrier at address 0x%llx "
                 "expected count %u\n",
                 thread_index.sw_cta_id, thread_index.sw_warp_id,
                 (unsigned long long)addr, expected_count);
    return;
  }

  auto id = m_next_id++;
  auto ret = addr_to_mbarrier_map.emplace(
      key, std::make_unique<mbarrier_t>(id, thread_index.hw_cta_id,
                                        thread_index.sw_cta_id, addr,
                                        expected_count));
  assert(ret.second && "mbarrier at the same address already exists");

  GPPRINTF_GPU_CORE(
      gpu, thread_index.sm_id, MBAR,
      "CTA %u Warp %u reached mbarrier init at address 0x%llx with "
      "expected count %u\n",
      thread_index.sw_cta_id, thread_index.sw_warp_id, (unsigned long long)addr,
      expected_count);
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

void mbarrier_manager_t::cancel_wait(int hw_warp_id) {
  for (auto &entry : addr_to_mbarrier_map)
    entry.second->m_waiting_warps.erase(hw_warp_id);
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

bool mbarrier_manager_t::test_wait(gpgpu_sim *gpu,
                                   const thread_index_t &thread_index,
                                   uint64_t addr, int parity) const {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  if (it == addr_to_mbarrier_map.end()) {
    assert(false && "mbarrier to wait on does not exist");
  }
  auto mbarrier = it->second.get();
  auto current_parity = mbarrier->m_phase & 1;

  GPPRINTF_GPU_CORE(
      gpu, thread_index.sm_id, MBAR,
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
  return false;
}

void mbarrier_manager_t::register_wait(const thread_index_t &thread_index,
                                       uint64_t addr) {
  auto *mbarrier = get_mbarrier(thread_index.sw_cta_id, addr);
  assert(mbarrier && "mbarrier to wait on does not exist");
  mbarrier->m_waiting_warps.insert(thread_index.hw_warp_id);
}

void mbarrier_manager_t::cancel_wait(int sw_cta_id, uint64_t addr,
                                     int hw_warp_id) {
  if (auto *mbarrier = get_mbarrier(sw_cta_id, addr)) {
    mbarrier->m_waiting_warps.erase(hw_warp_id);
  }
}

bool mbarrier_manager_t::try_wait(gpgpu_sim *gpu,
                                  const thread_index_t &thread_index,
                                  uint64_t addr, int parity,
                                  std::set<int> *released_on_advance) {
  if (test_wait(gpu, thread_index, addr, parity)) return true;
  auto *mbarrier = get_mbarrier(thread_index.sw_cta_id, addr);
  if (mbarrier && mbarrier->m_pending_arrival_count == 0 &&
      mbarrier->m_tx_count == 0 && (mbarrier->m_phase & 1) == parity) {
    std::set<int> released = try_advance(gpu, thread_index, mbarrier);
    if (released_on_advance)
      released_on_advance->insert(released.begin(), released.end());
    return true;
  }
  return false;
}

void mbarrier_manager_t::enqueue_wait(const thread_index_t &thread_index,
                                      uint64_t addr) {
  auto *mbarrier = get_mbarrier(thread_index.sw_cta_id, addr);
  if (!mbarrier) return;
  mbarrier->m_waiting_warps.insert(thread_index.hw_warp_id);
}

std::set<int> mbarrier_manager_t::try_advance(
    gpgpu_sim *gpu, const thread_index_t &thread_index, mbarrier_t *mbarrier) {

  if (mbarrier->m_pending_arrival_count == 0 && mbarrier->m_tx_count == 0) {
    // Release all waiting warps.
    std::set<int> released_warps = mbarrier->m_waiting_warps;
    mbarrier->m_waiting_warps.clear();
    mbarrier->m_pending_arrival_count = mbarrier->m_expected_count;
    mbarrier->m_phase++;
    GPPRINTF_GPU_CORE(gpu, thread_index.sm_id, MBAR,
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

  GPPRINTF_GPU_CORE(
      gpu, thread_index.sm_id, MBAR,
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

void mbarrier_manager_t::prepare_async_arrival(
    gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
    bool increment_pending) {
  auto key = std::make_pair(thread_index.sw_cta_id, addr);
  auto it = addr_to_mbarrier_map.find(key);
  assert(it != addr_to_mbarrier_map.end() &&
         "mbarrier for asynchronous arrival does not exist");

  auto *mbarrier = it->second.get();
  if (increment_pending) {
    assert(mbarrier->m_pending_arrival_count < (1 << 20) - 1);
    mbarrier->m_pending_arrival_count++;
  }

  GPPRINTF_GPU_CORE(gpu, thread_index.sm_id, MBAR,
                    "CTA %d Warp %d prepared asynchronous arrival at 0x%x "
                    "increment_pending=%d pending arrivals %d/%d\n",
                    thread_index.sw_cta_id, thread_index.sw_warp_id,
                    (unsigned)addr, (int)increment_pending,
                    mbarrier->m_pending_arrival_count,
                    mbarrier->m_expected_count);
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

  GPPRINTF_GPU_CORE(gpu, thread_index.sm_id, MBAR,
                    "CTA %d Warp %d mbarrier.complete_tx id %d at 0x%x with "
                    "completed_tx_count %d pending tx count %d\n",
                    thread_index.sw_cta_id, thread_index.sw_warp_id,
                    mbarrier->m_id, (unsigned)addr, completed_tx_count,
                    mbarrier->m_tx_count);

  // PTX tx-count is signed: completion may precede expect_tx. Clamping at
  // zero loses that credit and can strand a later arrive.expect_tx phase.
  mbarrier->m_tx_count -= completed_tx_count;
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
  // A selected, initialized peer receives every completion exactly once,
  // even if its expect_tx instruction has not executed yet.
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
  GPPRINTF_GPU_CORE(
      gpu, thread_index.sm_id, MBAR,
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

bool is_tcgen05_mbarrier_arrive_one(const ptx_instruction *pI) {
  for (auto op : pI->get_options()) {
    if (op == TCGEN05_MBARRIER_ARRIVE_ONE_OPTION)
      return true;
  }
  return false;
}

bool mbar_has_remote_path(shader_core_ctx *core) {
  return flash_gpgpu_sim::dsm_fabric_enabled(core);
}

void inject_remote_mbar(shader_core_ctx *shader, unsigned src_cid,
                        unsigned dst_cid, unsigned dst_hw_cta,
                        uint32_t mbar_addr, flash_gpgpu_sim::cluster_mbar_op op,
                        uint32_t count, unsigned req_hw_cta = 0,
                        unsigned req_warp_id = 0, int parity = 0) {
  if (!shader)
    return;
  auto *cluster = shader->get_cluster();
  if (!cluster)
    return;
  if (!flash_gpgpu_sim::dsm_fabric_enabled(shader)) {
    flash_gpgpu_sim::abort_dsm_disabled("mbarrier", shader->get_sid(),
                                        mbar_addr);
  }
  const unsigned gen = cluster->dsm_cta_gen(dst_cid, dst_hw_cta);
  if (!cluster->dsm_issue_mbar(src_cid, dst_cid, dst_hw_cta, gen, mbar_addr,
                               (unsigned)op, count, req_hw_cta, req_warp_id,
                               parity))
    cluster->dsm_queue_mbar_retry(src_cid, dst_cid, dst_hw_cta, gen, mbar_addr,
                                  (unsigned)op, count, req_hw_cta, req_warp_id,
                                  parity);
}
} // namespace

void handle_mbarrier_inst(const ptx_instruction *pIin,
                          ptx_thread_info *thread) {
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned ctaid = thread->get_cta_uid();
  auto hw_tid = thread->get_hw_tid();
  auto laneid = thread->get_laneid();

  GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), MBAR,
                    "CTA %d Thread %d (lane %u) handling mbarrier inst %s\n",
                    ctaid, hw_tid, laneid, pIin->to_string().c_str());

  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };

  // Local .shared::cta offsets are u32. Reading them as u64 can pick up
  // stale high bits from the register union (add.s32 only writes u32).
  // Keep a full 64-bit value only when it is a real generic mapa pointer.
  auto get_mbar_addr = [&](const operand_info &op) -> uint64_t {
    uint64_t raw = thread->get_operand_value(op, op, U64_TYPE, thread, 0).u64;
    unsigned owner = 0;
    addr_t off = 0;
    if ((raw >> 32) != 0 &&
        flash_gpgpu_sim::decode_shared_generic(raw, &owner, &off))
      return raw;
    return static_cast<uint32_t>(raw);
  };

  // Shared operands may be local offsets, mapa.u64 generic pointers, or
  // mapa.u32/cvta.to.shared compact owner/offset values.
  auto decode_mbar_address = [&](uint64_t raw, unsigned *owner, addr_t *off) {
    if (flash_gpgpu_sim::decode_shared_generic(raw, owner, off))
      return true;
    if (raw < SHARED_MEM_SIZE_MAX) {
      *owner = thread->get_hw_sid();
      *off = raw;
      return true;
    }
    auto *core = static_cast<shader_core_ctx *>(thread->get_core());
    if (core && raw / SHARED_MEM_SIZE_MAX <= core->get_config()->num_shader()) {
      *owner = raw / SHARED_MEM_SIZE_MAX - 1;
      *off = raw % SHARED_MEM_SIZE_MAX;
      return true;
    }
    return false;
  };

  // Helper to check if membar_level indicates shared memory scope.
  // .shared::cta is parsed as CTA_OPTION and sets membar_level.
  // .shared (without ::cta) is parsed as SHARED_DIRECTIVE which sets
  // Shared operands may still be local CTA offsets.
  auto is_shared_level = [&](uint32_t *addr = nullptr) {
    bool is_shared = (pI->membar_level() == CTA_OPTION) ||
                     (pI->membar_level() == CLUSTER_OPTION) ||
                     (pI->get_space() == shared_space);

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
              cfg->gpgpu_dsm_enable)) {
          printf("GPGPU-Sim ERROR: mbarrier address 0x%x (absolute 0x%llx) is "
                 "not in SM %u's shared memory.\n"
                 "Enable -gpgpu_mbarrier_cluster_enable with "
                 "-gpgpu_dsm_enable for remote mbarrier.\n",
                 *addr, (unsigned long long)absolute_addr,
                 thread->get_hw_sid());
          fflush(stdout);
          abort();
        }
        // Remote path is allowed; timing routes via the intra-GPC fabric.
      }
    }
    return is_shared;
  };

  // Resolve mbarrier address: local smem offset, or remote via generic mapa
  // addr. Sets remote_* fields when the barrier lives on a peer CTA.
  auto resolve_and_fill_mbar_info = [&](uint64_t raw_addr, unsigned count,
                                        bool parity,
                                        bool has_time_hint = false,
                                        uint32_t time_hint_ns = 0) {
    inst_t::mbarrier_info_t info;
    info.bar_count = count;
    info.bar_parity = parity;
    info.bar_has_time_hint = has_time_hint;
    info.bar_time_hint_ns = time_hint_ns;
    info.is_remote = false;
    info.bar_id = static_cast<unsigned>(raw_addr);

    unsigned owner_smid = 0;
    addr_t offset = 0;
    if (decode_mbar_address(raw_addr, &owner_smid, &offset)) {
      // mapa compact/generic and local offsets must share one key.
      info.bar_id = static_cast<unsigned>(offset);
      auto *core = dynamic_cast<shader_core_ctx *>(thread->get_core());
      const auto *cfg = core ? core->get_config() : nullptr;
      if (core && cfg && cfg->gpgpu_mbarrier_cluster_enable &&
          flash_gpgpu_sim::dsm_fabric_enabled(core) &&
          owner_smid != thread->get_hw_sid()) {
        flash_gpgpu_sim::tb_cluster_target_t tgt;
        if (flash_gpgpu_sim::resolve_tb_cluster_owner_sm(
                core, thread->get_hw_ctaid(), owner_smid, &tgt)) {
          info.is_remote = true;
          info.remote_cid = tgt.local_sm;
          info.remote_hw_cta = tgt.cta_slot;
        }
      }
    }
    pI->set_mbarrier_info(laneid, info);
  };

  // Helper to set per-thread mbarrier info (local)
  auto set_thread_mbarrier_info = [&](unsigned addr, unsigned count,
                                      bool parity, bool has_time_hint = false,
                                      uint32_t time_hint_ns = 0) {
    resolve_and_fill_mbar_info(addr, count, parity, has_time_hint,
                               time_hint_ns);
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
    set_thread_mbarrier_info(addr, expected_count, false);
    if (pI->get_mbarrier_info(laneid).is_remote) {
      printf("GPGPU-Sim ERROR: mbarrier.init on remote DSM address is not "
             "supported (init must run on the owner CTA).\n");
      abort();
    }
    GPPRINTF_GPU_CORE(
        thread->get_gpu(), thread->get_hw_sid(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier init at address 0x%x "
        "with expected "
        "count %u\n",
        ctaid, hw_tid, laneid, addr, expected_count);
  } else if (bar_op == TRY_WAIT_OPTION) {

    assert(pI->parity_op() && "Only support parity op of mbarrier.try_wait");

    assert((pI->get_num_operands() == 3 || pI->get_num_operands() == 4) &&
           "mbarrier.try_wait expects predicate, address, parity, and optional "
           "suspendTimeHint");

    const operand_info &addr_op = pI->src1();
    const operand_info &parity_op = pI->src2();
    const bool has_timeout = pI->get_num_operands() == 4;
    unsigned timeout_hint = 0;
    if (has_timeout)
      timeout_hint = get_u32_value(pI->src3());
    uint64_t raw_addr = get_mbar_addr(addr_op);
    uint32_t addr32 = static_cast<uint32_t>(raw_addr);
    unsigned owner = 0;
    addr_t off = 0;
    const bool looks_mapped = raw_addr >= SHARED_MEM_SIZE_MAX &&
                              decode_mbar_address(raw_addr, &owner, &off);
    if (!looks_mapped) {
      assert(is_shared_level(&addr32) && "Only support shared mbarrier");
      raw_addr = addr32;
    }
    auto parity = get_u32_value(parity_op) & 1;
    const bool has_time_hint = pI->get_num_operands() == 4;
    const uint32_t time_hint_ns = has_time_hint ? get_u32_value(pI->src3()) : 0;

    GPPRINTF_GPU_CORE(
        thread->get_gpu(), thread->get_hw_sid(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier.try_wait at address 0x%x "
        "with parity %u has_time_hint=%u time_hint_ns=%u\n",
        ctaid, hw_tid, laneid, addr, parity, (unsigned)has_time_hint,
        time_hint_ns);
    // Set per-thread info
    set_thread_mbarrier_info(addr, (unsigned)-1, parity, has_time_hint,
                             time_hint_ns);

    // Predicate writeback is owned by barrier_set_t's initial test / pending
    // wait state.  Writing a provisional value here would let functional and
    // timing state disagree.

  } else if (bar_op == COMPLETE_TX_OPTION) {

    assert(pI->get_num_operands() == 2);
    const operand_info &addr_op = pI->dst();
    const operand_info &tx_count_op = pI->src1();
    uint64_t raw_addr = get_mbar_addr(addr_op);
    auto completed_tx_count = get_u32_value(tx_count_op);
    if (completed_tx_count == 0) {
      printf("GPGPU-Sim: mbarrier.complete_tx with completed_tx_count 0\n");
      abort();
    }

    GPPRINTF_GPU_CORE(
        thread->get_gpu(), thread->get_hw_sid(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier.complete_tx at address 0x%x "
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
      // Dest is the phase token; `_` or a discarded register are both fine.

      raw_addr = get_mbar_addr(pI->src1());
      expected_tx_count = get_u32_value(pI->src2());

      GPPRINTF_GPU_CORE(
          thread->get_gpu(), thread->get_hw_sid(), MBAR,
          "CTA %d Thread %d (lane %u) mbarrier.arrive.expect_tx at "
          "address 0x%x "
          "with expected_tx_count %u\n",
          ctaid, hw_tid, laneid, addr, expected_tx_count);
      // Set per-thread info (for arrive.expect_tx, store expected_tx_count in
      // bar_count)
      set_thread_mbarrier_info(addr, expected_tx_count, false);

    } else if (is_arrive) {
      assert(pI->get_num_operands() == 3 || pI->get_num_operands() == 2);

      raw_addr = get_mbar_addr(pI->src1());
      if (pI->get_num_operands() == 3) {
        arrival_count = get_u32_value(pI->src2());
      }

      if (arrival_count == 0) {
        printf("GPGPU-Sim: mbarrier.arrive with arrival_count 0\n");
        abort();
      }

      GPPRINTF_GPU_CORE(
          thread->get_gpu(), thread->get_hw_sid(), MBAR,
          "CTA %d Thread %d (lane %u) mbarrier.arrive at address 0x%x with "
          "arrival_count %u\n",
          ctaid, hw_tid, laneid, (unsigned long long)raw_addr, arrival_count);
      resolve_and_fill_mbar_info(raw_addr, arrival_count, false);

    } else if (is_expect_tx) {
      assert(pI->get_num_operands() == 2);

      raw_addr = get_mbar_addr(pI->dst());
      expected_tx_count = get_u32_value(pI->src1());

      GPPRINTF_GPU_CORE(
          thread->get_gpu(), thread->get_hw_sid(), MBAR,
          "CTA %d Thread %d (lane %u) mbarrier.expect_tx at address 0x%x "
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
    GPPRINTF_GPU_CORE(
        thread->get_gpu(), thread->get_hw_sid(), MBAR,
        "CTA %d Thread %d (lane %u) mbarrier inval at address 0x%x\n", ctaid,
        hw_tid, laneid, addr);
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

void barrier_set_t::notify_mbarrier_phase_change(
    const std::set<int> &notified_warps) {
  const uint64_t now = m_shader->get_gpu()->gpu_sim_cycle +
                       m_shader->get_gpu()->gpu_tot_sim_cycle;
  for (int warp_id : notified_warps) {
    auto it = m_pending_mbarrier_waits.find(warp_id);
    if (it == m_pending_mbarrier_waits.end())
      continue;
    if (it->second.result_ready_delay_pending)
      continue;
    it->second.phase_notification_pending = true;
    it->second.next_recheck_cycle =
        flash_gpgpu_sim::mbarrier_wake_on_phase_notification(
            it->second.next_recheck_cycle, now);
    if (GPTRACE_CORE(MBAR, m_shader->get_sid())) {
      GPPRINTF_GPU_CORE(m_shader->get_gpu(), m_shader->get_sid(), MBAR,
                        "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%d "
                        "pc=0x%llx state=sleeping event=phase_notification "
                        "next_recheck=%llu\n",
                        it->second.hw_cta_id, it->second.sw_cta_id, warp_id,
                        (unsigned long long)it->second.pc,
                        (unsigned long long)it->second.next_recheck_cycle);
    }
  }
}

void barrier_set_t::apply_trywait_pred(unsigned warp_id, bool phase_complete) {
  if (warp_id >= m_mbar_trywait_inst.size())
    return;
  const ptx_instruction *pI = m_mbar_trywait_inst[warp_id];
  if (!pI)
    return;
  ptx_thread_info **threads = m_shader->get_thread_info();
  if (!threads)
    return;
  // PTXPlus dest pred: 0 = true, 1 = false.
  ptx_reg_t pred;
  pred.pred = phase_complete ? 0 : 1;
  const unsigned warp_size = m_warp_size;
  const active_mask_t &mask = m_mbar_trywait_mask[warp_id];
  for (unsigned lane = 0; lane < warp_size; lane++) {
    if (!mask.test(lane))
      continue;
    ptx_thread_info *thd = threads[warp_id * warp_size + lane];
    if (!thd)
      continue;
    thd->set_operand_value(pI->dst(), pred, PRED_TYPE, thd, pI);
  }
}

void barrier_set_t::arm_trywait_timeout(unsigned warp_id, unsigned timeout_hint,
                                        const ptx_instruction *static_inst,
                                        const active_mask_t &active_mask) {
  if (warp_id >= m_mbar_trywait_has_timeout.size())
    return;
  m_mbar_trywait_has_timeout[warp_id] = true;
  m_mbar_trywait_inst[warp_id] = static_inst;
  m_mbar_trywait_mask[warp_id] = active_mask;
  // PTX suspendTimeHint is in nanoseconds, shader_clock() in kHz. Round
  // upward and retain a one-cycle minimum for a zero hint. Widen before
  // multiplying: the full u32 nanosecond range exceeds u32 core cycles.
  const unsigned long long delay =
      std::max(1ULL, (static_cast<unsigned long long>(timeout_hint) *
                          m_shader->get_gpu()->shader_clock() +
                      999999ULL) /
                         1000000ULL);
  const unsigned long long now = m_shader->get_gpu()->gpu_sim_cycle +
                                 m_shader->get_gpu()->gpu_tot_sim_cycle;
  m_mbar_timeout_cycle[warp_id] = now + delay;
}

void barrier_set_t::clear_trywait_timeout(unsigned warp_id) {
  if (warp_id >= m_mbar_trywait_has_timeout.size())
    return;
  m_mbar_trywait_has_timeout[warp_id] = false;
  m_mbar_timeout_cycle[warp_id] = 0;
  m_mbar_trywait_inst[warp_id] = nullptr;
}

void barrier_set_t::release_warps(const std::set<int> &released_warps) {
  if (released_warps.empty())
    return;
  for (auto w : released_warps) {
    if (w >= 0 && (unsigned)w < m_mbar_trywait_has_timeout.size() &&
        m_mbar_trywait_has_timeout[w]) {
      // Phase completed before the timeout expired.
      apply_trywait_pred((unsigned)w, /*phase_complete=*/true);
      clear_trywait_timeout((unsigned)w);
    }
  }
  unsigned trywait_latency =
      m_shader->get_config()->gpgpu_mbarrier_trywait_latency;
  const char *trace = getenv("FLASHGPU_SIM_BARRIER_TRACE");
  bool trace_barrier = trace != nullptr && trace[0] != '\0' && trace[0] != '0';
  if (trywait_latency > 0) {
    for (auto w : released_warps) {
      // A query that did not park (or has already timed out) must not be
      // treated as a live barrier waiter by a later arrive.
      if (w < 0 || (unsigned)w >= m_warp_barrier_type.size() ||
          !m_warp_at_barrier.test(w) ||
          m_warp_barrier_type[w] != BARRIER_WAIT_MBARRIER)
        continue;
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
  }
}

void barrier_set_t::finish_mbarrier_wait(unsigned warp_id, const char *reason) {
  auto it = m_pending_mbarrier_waits.find(warp_id);
  assert(it != m_pending_mbarrier_waits.end());
  const pending_mbarrier_wait_t wait = it->second;
  const uint64_t now = m_shader->get_gpu()->gpu_sim_cycle +
                       m_shader->get_gpu()->gpu_tot_sim_cycle;

  assert(wait.static_inst != nullptr);
  bool all_true = true;
  for (unsigned lane = 0; lane < m_warp_size; ++lane) {
    if (!wait.lanes[lane].active)
      continue;
    assert(wait.lanes[lane].resolved);
    all_true = all_true && wait.lanes[lane].result;
    ptx_thread_info *thread =
        m_shader->get_thread_info()[warp_id * m_warp_size + lane];
    assert(thread != nullptr);
    ptx_reg_t pred;
    // PTXPlus stores predicate truth using the inverse zero flag.
    pred.pred = wait.lanes[lane].result ? 0 : 1;
    thread->set_operand_value(wait.static_inst->dst(), pred, PRED_TYPE, thread,
                              wait.static_inst);
    m_mbarrier_manager.cancel_wait(wait.sw_cta_id, wait.lanes[lane].addr,
                                   warp_id);
  }

  if (m_warp_at_barrier.test(warp_id)) {
    clear_warp_waiting(warp_id, BARRIER_WAIT_MBARRIER, reason);
  }
  m_shader->m_stats->mbarrier_sleep_cycles += now - wait.issue_cycle;
  if (!wait.suspended && all_true) {
    m_shader->m_stats->mbarrier_immediate_true++;
  } else if (wait.suspended && all_true) {
    m_shader->m_stats->mbarrier_true_after_suspend++;
  } else {
    m_shader->m_stats->mbarrier_timeout_false++;
  }

  if (GPTRACE_CORE(MBAR, m_shader->get_sid())) {
    GPPRINTF_GPU_CORE(m_shader->get_gpu(), m_shader->get_sid(), MBAR,
                      "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%u "
                      "dynamic_warp=%u pc=0x%llx active=%s state=complete "
                      "all_true=%u reason=%s sleep_cycles=%llu\n",
                      wait.hw_cta_id, wait.sw_cta_id, warp_id,
                      wait.dynamic_warp_id, (unsigned long long)wait.pc,
                      wait.active_mask.to_string().c_str(), (unsigned)all_true,
                      reason, (unsigned long long)(now - wait.issue_cycle));
    for (unsigned lane = 0; lane < m_warp_size; ++lane) {
      if (!wait.lanes[lane].active)
        continue;
      GPPRINTF_GPU_CORE(
          m_shader->get_gpu(), m_shader->get_sid(), MBAR,
          "MBAR_WAIT warp=%u pc=0x%llx lane=%u "
          "addr=0x%llx parity=%u hint=%s%u state=lane_complete result=%s "
          "deadline=%llu\n",
          warp_id, (unsigned long long)wait.pc, lane,
          (unsigned long long)wait.lanes[lane].addr,
          (unsigned)wait.lanes[lane].parity,
          wait.lanes[lane].has_time_hint ? "" : "none/",
          wait.lanes[lane].time_hint_ns,
          wait.lanes[lane].result ? "true" : "false",
          (unsigned long long)wait.lanes[lane].deadline_cycle);
    }
  }
  m_pending_mbarrier_waits.erase(it);
}

void barrier_set_t::cancel_mbarrier_wait(unsigned warp_id, const char *reason,
                                         bool clear_wait_bit) {
  auto it = m_pending_mbarrier_waits.find(warp_id);
  if (it == m_pending_mbarrier_waits.end())
    return;
  const pending_mbarrier_wait_t wait = it->second;
  for (unsigned lane = 0; lane < m_warp_size; ++lane) {
    if (wait.lanes[lane].active) {
      m_mbarrier_manager.cancel_wait(wait.sw_cta_id, wait.lanes[lane].addr,
                                     warp_id);
    }
  }
  if (clear_wait_bit && m_warp_at_barrier.test(warp_id) &&
      m_warp_barrier_type[warp_id] == BARRIER_WAIT_MBARRIER) {
    clear_warp_waiting(warp_id, BARRIER_WAIT_MBARRIER, reason);
  }
  if (GPTRACE_CORE(MBAR, m_shader->get_sid())) {
    GPPRINTF_GPU_CORE(m_shader->get_gpu(), m_shader->get_sid(), MBAR,
                      "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%u "
                      "dynamic_warp=%u pc=0x%llx state=cancelled reason=%s\n",
                      wait.hw_cta_id, wait.sw_cta_id, warp_id,
                      wait.dynamic_warp_id, (unsigned long long)wait.pc,
                      reason);
  }
  m_pending_mbarrier_waits.erase(it);
}

void barrier_set_t::cleanup_cta_pending_mbarrier_waits(unsigned hw_cta_id) {
  std::vector<unsigned> warp_ids;
  for (const auto &entry : m_pending_mbarrier_waits) {
    if (entry.second.hw_cta_id == hw_cta_id)
      warp_ids.push_back(entry.first);
  }
  for (unsigned warp_id : warp_ids) {
    cancel_mbarrier_wait(warp_id, "mbarrier CTA cleanup", true);
  }
}

void barrier_set_t::hold_warp(unsigned warp_id, unsigned latency,
                              barrier_wait_type_t type) {
  if (latency == 0)
    return;
  m_warp_at_barrier.set(warp_id);
  if (warp_id < m_warp_barrier_type.size())
    m_warp_barrier_type[warp_id] = type;
  if (warp_id < m_warp_named_barrier_id.size())
    m_warp_named_barrier_id[warp_id] = (unsigned)-1;
  m_pending_warp_releases.push_back({latency, static_cast<int>(warp_id), type});
}

void barrier_set_t::cycle() {
  const unsigned long long now = m_shader->get_gpu()->gpu_sim_cycle +
                                 m_shader->get_gpu()->gpu_tot_sim_cycle;
  for (unsigned w = 0; w < m_max_warps_per_core; w++) {
    if (!m_mbar_trywait_has_timeout[w])
      continue;
    if (!m_warp_at_barrier.test(w) ||
        m_warp_barrier_type[w] != BARRIER_WAIT_MBARRIER)
      continue;
    if (now < m_mbar_timeout_cycle[w])
      continue;
    apply_trywait_pred(w, /*phase_complete=*/false);
    clear_trywait_timeout(w);
    m_mbarrier_manager.cancel_wait((int)w);
    release_warps({static_cast<int>(w)});
  }

  for (auto &entry : m_pending_warp_releases) {
    entry.remaining--;
  }
  // Release warps whose countdown reached zero
  for (int i = m_pending_warp_releases.size() - 1; i >= 0; i--) {
    if (m_pending_warp_releases[i].remaining == 0) {
      int warp_id = m_pending_warp_releases[i].warp_id;
      if (GPTRACE_CORE(NAMED_BARRIER, m_shader->get_sid())) {
        GPPRINTF_GPU_CORE(
            m_shader->get_gpu(), m_shader->get_sid(), NAMED_BARRIER,
            "delayed_release warp=%d "
            "warp_at_barrier_before=%s type=%d\n",
            warp_id, m_warp_at_barrier.to_string().c_str(),
            (warp_id >= 0 && (unsigned)warp_id < m_warp_barrier_type.size())
                ? (int)m_warp_barrier_type[warp_id]
                : -1);
      }
      barrier_wait_type_t type = m_pending_warp_releases[i].type;
      const char *reason = type == BARRIER_WAIT_BAR_SYNC
                               ? "delayed CTA barrier release"
                               : "delayed cp.async wait_group release";
      assert(type == BARRIER_WAIT_BAR_SYNC ||
             type == BARRIER_WAIT_CP_ASYNC_GROUP);
      clear_warp_waiting(warp_id, type, reason);
      m_pending_warp_releases.erase(m_pending_warp_releases.begin() + i);
    }
  }

  const uint64_t now = m_shader->get_gpu()->gpu_sim_cycle +
                       m_shader->get_gpu()->gpu_tot_sim_cycle;
  std::vector<unsigned> due_warps;
  for (const auto &entry : m_pending_mbarrier_waits) {
    if (now >= entry.second.next_recheck_cycle) {
      due_warps.push_back(entry.first);
    }
  }

  for (unsigned warp_id : due_warps) {
    auto it = m_pending_mbarrier_waits.find(warp_id);
    if (it == m_pending_mbarrier_waits.end())
      continue;
    pending_mbarrier_wait_t &wait = it->second;

    const bool same_generation =
        warp_id < m_shader->m_warp.size() &&
        m_shader->m_warp[warp_id]->get_dynamic_warp_id() ==
            wait.dynamic_warp_id &&
        m_shader->m_warp[warp_id]->get_cta_id() == wait.hw_cta_id;
    if (!same_generation) {
      cancel_mbarrier_wait(warp_id, "mbarrier stale warp generation", false);
      continue;
    }

    if (wait.result_ready_delay_pending) {
      finish_mbarrier_wait(warp_id, "mbarrier predicate result visible");
      continue;
    }

    flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
        (int)wait.hw_cta_id, (int)wait.hw_warp_id, wait.sw_cta_id,
        wait.sw_warp_id, static_cast<int>(m_shader->get_sid())};
    m_shader->m_stats->mbarrier_rechecks++;
    bool all_resolved = true;
    uint64_t next_recheck = std::numeric_limits<uint64_t>::max();
    for (unsigned lane = 0; lane < m_warp_size; ++lane) {
      pending_mbarrier_wait_t::lane_wait_t &lane_wait = wait.lanes[lane];
      if (!lane_wait.active || lane_wait.resolved)
        continue;

      const bool complete = m_mbarrier_manager.test_wait(
          m_shader->get_gpu(), thread_index, lane_wait.addr, lane_wait.parity);
      const flash_gpgpu_sim::mbarrier_recheck_action_t action =
          flash_gpgpu_sim::mbarrier_classify_recheck(complete, now,
                                                     lane_wait.deadline_cycle);

      if (GPTRACE_CORE(MBAR, m_shader->get_sid())) {
        GPPRINTF_GPU_CORE(m_shader->get_gpu(), m_shader->get_sid(), MBAR,
                          "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%u "
                          "dynamic_warp=%u pc=0x%llx lane=%u state=recheck "
                          "complete=%u notified=%u deadline=%llu\n",
                          wait.hw_cta_id, wait.sw_cta_id, warp_id,
                          wait.dynamic_warp_id, (unsigned long long)wait.pc,
                          lane, (unsigned)complete,
                          (unsigned)wait.phase_notification_pending,
                          (unsigned long long)lane_wait.deadline_cycle);
      }

      // Completion visible at the start of this recheck wins at the inclusive
      // deadline.  A phase update later in the simulator cycle cannot change
      // a result committed here.
      if (action == flash_gpgpu_sim::mbarrier_recheck_action_t::RETURN_TRUE) {
        lane_wait.resolved = true;
        lane_wait.result = true;
      } else if (action ==
                 flash_gpgpu_sim::mbarrier_recheck_action_t::RETURN_FALSE) {
        lane_wait.resolved = true;
        lane_wait.result = false;
      } else {
        all_resolved = false;
        next_recheck = std::min(next_recheck, lane_wait.deadline_cycle);
        // A phase transition clears the manager's waiter set. A lane whose
        // requested phase is still incomplete must register for the next
        // transition while retaining its original deadline.
        m_mbarrier_manager.register_wait(thread_index, lane_wait.addr);
      }
    }

    if (all_resolved) {
      bool all_true = true;
      for (unsigned lane = 0; lane < m_warp_size; ++lane) {
        if (wait.lanes[lane].active)
          all_true = all_true && wait.lanes[lane].result;
      }
      const bool delay_phase_wakeup =
          flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(
              wait.suspended, wait.phase_notification_pending, all_true,
              m_shader->get_config()->gpgpu_mbarrier_phase_wakeup_latency);
      const unsigned phase_wakeup_latency =
          delay_phase_wakeup
              ? m_shader->get_config()->gpgpu_mbarrier_phase_wakeup_latency
              : 0;
      const uint64_t release_cycle =
          flash_gpgpu_sim::mbarrier_result_release_cycle(
              wait.issue_cycle, now,
              wait.has_no_hint
                  ? m_shader->get_config()->gpgpu_mbarrier_predicate_latency
                  : 0,
              phase_wakeup_latency);
      if (release_cycle > now) {
        wait.phase_notification_pending = false;
        wait.result_ready_delay_pending = true;
        wait.next_recheck_cycle = release_cycle;
        if (delay_phase_wakeup) {
          m_shader->m_stats->mbarrier_phase_wakeups++;
          m_shader->m_stats->mbarrier_phase_wakeup_cycles +=
              phase_wakeup_latency;
        }
        if (GPTRACE_CORE(MBAR, m_shader->get_sid())) {
          GPPRINTF_GPU_CORE(
              m_shader->get_gpu(), m_shader->get_sid(), MBAR,
              "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%u "
              "pc=0x%llx state=result_ready_delay release_cycle=%llu\n",
              wait.hw_cta_id, wait.sw_cta_id, warp_id,
              (unsigned long long)wait.pc,
              (unsigned long long)wait.next_recheck_cycle);
        }
      } else {
        finish_mbarrier_wait(warp_id, "mbarrier lanes resolved");
      }
    } else {
      wait.phase_notification_pending = false;
      wait.next_recheck_cycle = next_recheck;
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
      (int)cta_id, (int)warp_id, logical_cta_id, logical_warp_id,
      static_cast<int>(m_shader->get_sid())};

  auto released_warps = m_mbarrier_manager.complete_tx(
      m_shader->get_gpu(), thread_index, mbarrier_addr, completed_tx_count);
  notify_mbarrier_phase_change(released_warps);
}

void barrier_set_t::prepare_mbarrier_async_arrival(unsigned cta_id,
                                                   unsigned warp_id,
                                                   uint32_t mbarrier_addr,
                                                   bool increment_pending) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, m_shader->get_logical_cta_id(warp_id),
      m_shader->get_cta_warp_id(warp_id),
      static_cast<int>(m_shader->get_sid())};
  m_mbarrier_manager.prepare_async_arrival(m_shader->get_gpu(), thread_index,
                                           mbarrier_addr, increment_pending);
}

void barrier_set_t::arrive_mbarrier_async(unsigned cta_id, unsigned warp_id,
                                          uint32_t mbarrier_addr) {
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, m_shader->get_logical_cta_id(warp_id),
      m_shader->get_cta_warp_id(warp_id),
      static_cast<int>(m_shader->get_sid())};
  notify_mbarrier_phase_change(m_mbarrier_manager.arrive(
      m_shader->get_gpu(), thread_index, mbarrier_addr, 1));
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
  if (!cluster || !mbar_has_remote_path(m_shader))
    return;
  const unsigned owner_cid =
      m_shader->get_config()->sid_to_cid(m_shader->get_sid());
  for (const auto &w : waiters) {
    inject_remote_mbar(m_shader, owner_cid, w.src_cid, w.src_hw_cta,
                       mbarrier_addr,
                       flash_gpgpu_sim::cluster_mbar_op::WAIT_DONE, 1,
                       w.src_hw_cta, w.src_warp_id, /*parity=*/1);
  }
}

void barrier_set_t::release_remote_waiter(unsigned warp_id) {
  if (warp_id >= m_max_warps_per_core)
    return;
  apply_trywait_pred(warp_id, /*phase_complete=*/true);
  // Pay trywait observation latency like a local successful try_wait.
  release_warps({static_cast<int>(warp_id)});
}

void barrier_set_t::warp_reaches_mbarrier(unsigned cta_id, unsigned warp_id,
                                          const ptx_instruction *pI,
                                          const warp_inst_t *dynamic_inst,
                                          const active_mask_t &active_mask,
                                          unsigned dynamic_warp_id) {

  // We use the logical CTA ID here.
  auto logical_cta_id = m_shader->get_logical_cta_id(warp_id);
  auto logical_warp_id = m_shader->get_cta_warp_id(warp_id);

  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      (int)cta_id, (int)warp_id, logical_cta_id, logical_warp_id,
      static_cast<int>(m_shader->get_sid())};

  if (is_tcgen05_mbarrier_arrive_one(pI)) {
    for (unsigned lane = 0; lane < m_shader->get_config()->warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      auto released_warps = m_mbarrier_manager.arrive(
          m_shader->get_gpu(), thread_index, mbar_info.bar_id, 1);
      notify_mbarrier_phase_change(released_warps);
      return;
    }
    return;
  }

  auto bar_op = pI->barrier_op();

  unsigned warp_size = m_shader->get_config()->warp_size;
  const bool trace_mbarrier = GPTRACE_CORE(MBAR, m_shader->get_sid());

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

      const bool matches =
          info.bar_id == mbar_info.bar_id &&
          info.bar_count == mbar_info.bar_count &&
          info.bar_parity == mbar_info.bar_parity &&
          info.bar_has_time_hint == mbar_info.bar_has_time_hint &&
          info.bar_time_hint_ns == mbar_info.bar_time_hint_ns &&
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
    assert(m_pending_mbarrier_waits.find(warp_id) ==
               m_pending_mbarrier_waits.end() &&
           "warp already has a pending mbarrier.try_wait");

    const uint64_t now = m_shader->get_gpu()->gpu_sim_cycle +
                         m_shader->get_gpu()->gpu_tot_sim_cycle;
    pending_mbarrier_wait_t wait;
    wait.hw_cta_id = cta_id;
    wait.sw_cta_id = logical_cta_id;
    wait.hw_warp_id = warp_id;
    wait.sw_warp_id = logical_warp_id;
    wait.dynamic_warp_id = dynamic_warp_id;
    wait.pc = pI->pc;
    wait.active_mask = active_mask;
    wait.issue_cycle = now;
    wait.static_inst = pI;

    bool found_lane = false;
    bool has_unresolved_lane = false;
    bool has_latched_false_lane = false;
    wait.next_recheck_cycle = std::numeric_limits<uint64_t>::max();
    for (unsigned lane = 0; lane < warp_size; ++lane) {
      if (!active_mask.test(lane))
        continue;
      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      assert(is_valid_mbarrier_info(mbar_info) &&
             "active mbarrier.try_wait lane is missing operand metadata");
      found_lane = true;

      pending_mbarrier_wait_t::lane_wait_t &lane_wait = wait.lanes[lane];
      lane_wait.active = true;
      lane_wait.addr = mbar_info.bar_id;
      lane_wait.parity = mbar_info.bar_parity;
      lane_wait.has_time_hint = mbar_info.bar_has_time_hint;
      lane_wait.time_hint_ns = mbar_info.bar_time_hint_ns;
      wait.has_no_hint = wait.has_no_hint || !lane_wait.has_time_hint;

      const bool complete = m_mbarrier_manager.test_wait(
          m_shader->get_gpu(), thread_index, lane_wait.addr, lane_wait.parity);
      const bool latch_initial_predicate =
          flash_gpgpu_sim::mbarrier_latches_initial_predicate(
              lane_wait.has_time_hint,
              m_shader->get_config()->gpgpu_mbarrier_predicate_latency);
      if (complete || latch_initial_predicate) {
        lane_wait.resolved = true;
        lane_wait.result = complete;
        lane_wait.deadline_cycle = now;
        has_latched_false_lane = has_latched_false_lane || !complete;
        continue;
      }

      uint64_t suspension_cycles =
          m_shader->get_config()->gpgpu_mbarrier_trywait_latency;
      if (lane_wait.has_time_hint) {
        suspension_cycles = flash_gpgpu_sim::mbarrier_hint_ns_to_cycles(
            lane_wait.time_hint_ns,
            m_shader->get_gpu()->get_config().get_core_freq());
      }
      lane_wait.deadline_cycle =
          flash_gpgpu_sim::mbarrier_saturating_add(now, suspension_cycles);
      if (suspension_cycles == 0) {
        lane_wait.resolved = true;
        lane_wait.result = false;
        continue;
      }

      has_unresolved_lane = true;
      m_mbarrier_manager.register_wait(thread_index, lane_wait.addr);
      wait.next_recheck_cycle =
          std::min(wait.next_recheck_cycle, lane_wait.deadline_cycle);
    }
    assert(found_lane);
    wait.suspended = has_unresolved_lane || has_latched_false_lane;
    m_pending_mbarrier_waits.emplace(warp_id, wait);
    m_shader->m_stats->mbarrier_logical_trywait++;
    if (wait.suspended)
      m_shader->m_stats->mbarrier_suspended_waits++;

    if (!has_unresolved_lane) {
      const uint64_t release_cycle =
          flash_gpgpu_sim::mbarrier_result_release_cycle(
              wait.issue_cycle, now,
              wait.has_no_hint
                  ? m_shader->get_config()->gpgpu_mbarrier_predicate_latency
                  : 0,
              0);
      if (release_cycle == now) {
        finish_mbarrier_wait(warp_id, "mbarrier initial lanes resolved");
        return;
      }
      pending_mbarrier_wait_t &pending =
          m_pending_mbarrier_waits.find(warp_id)->second;
      pending.result_ready_delay_pending = true;
      pending.next_recheck_cycle = release_cycle;
      m_warp_at_barrier.set(warp_id);
      m_warp_barrier_type[warp_id] = BARRIER_WAIT_MBARRIER;
      m_warp_named_barrier_id[warp_id] = (unsigned)-1;
      return;
    }

    m_warp_at_barrier.set(warp_id);
    m_warp_barrier_type[warp_id] = BARRIER_WAIT_MBARRIER;
    m_warp_named_barrier_id[warp_id] = (unsigned)-1;
    if (trace_mbarrier) {
      GPPRINTF_GPU_CORE(m_shader->get_gpu(), m_shader->get_sid(), MBAR,
                        "MBAR_WAIT hw_cta=%u sw_cta=%d warp=%u "
                        "dynamic_warp=%u pc=0x%llx active=%s state=sleeping "
                        "next_recheck=%llu\n",
                        cta_id, logical_cta_id, warp_id, dynamic_warp_id,
                        (unsigned long long)wait.pc,
                        active_mask.to_string().c_str(),
                        (unsigned long long)wait.next_recheck_cycle);
      for (unsigned lane = 0; lane < warp_size; ++lane) {
        const pending_mbarrier_wait_t::lane_wait_t &lane_wait =
            wait.lanes[lane];
        if (!lane_wait.active)
          continue;
        GPPRINTF_GPU_CORE(
            m_shader->get_gpu(), m_shader->get_sid(), MBAR,
            "MBAR_WAIT warp=%u pc=0x%llx lane=%u "
            "addr=0x%llx parity=%u hint=%s%u state=%s result=%s "
            "deadline=%llu\n",
            warp_id, (unsigned long long)wait.pc, lane,
            (unsigned long long)lane_wait.addr, (unsigned)lane_wait.parity,
            lane_wait.has_time_hint ? "" : "none/", lane_wait.time_hint_ns,
            lane_wait.resolved ? "resolved" : "sleeping",
            lane_wait.resolved ? (lane_wait.result ? "true" : "false")
                               : "pending",
            (unsigned long long)lane_wait.deadline_cycle);
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
      if (mbar_has_remote_path(m_shader)) {
        const unsigned src_cid =
            m_shader->get_config()->sid_to_cid(m_shader->get_sid());
        inject_remote_mbar(m_shader, src_cid, mbar_info.remote_cid,
                           mbar_info.remote_hw_cta, addr,
                           flash_gpgpu_sim::cluster_mbar_op::COMPLETE_TX,
                           completed_tx_count);
      }
      return;
    }

    auto released_warps = m_mbarrier_manager.complete_tx(
        m_shader->get_gpu(), thread_index, addr, completed_tx_count);
    notify_mbarrier_phase_change(released_warps);
    release_warps(released_warps);
    notify_remote_waiters(cta_id, addr);

    return;
  } else if (bar_op == ARRIVE_OPTION || bar_op == EXPECT_TX_OPTION) {

    auto [is_arrive, is_expect_tx] =
        parse_mbarrier_arrive_expect_tx_options(pI);

    const unsigned src_cid =
        m_shader->get_config()->sid_to_cid(m_shader->get_sid());
    bool expect_tx_zero = false;

    for (unsigned lane = 0; lane < warp_size; lane++) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbar_info = dynamic_inst->get_mbarrier_info(lane);
      if (!is_valid_mbarrier_info(mbar_info))
        continue;

      auto addr = mbar_info.bar_id;
      auto count = mbar_info.bar_count;
      if (is_expect_tx && count == 0)
        expect_tx_zero = true;

      if (mbar_info.is_remote) {
        if (mbar_has_remote_path(m_shader)) {
          if (is_expect_tx && is_arrive) {
            inject_remote_mbar(m_shader, src_cid, mbar_info.remote_cid,
                               mbar_info.remote_hw_cta, addr,
                               flash_gpgpu_sim::cluster_mbar_op::EXPECT_TX,
                               count);
            inject_remote_mbar(m_shader, src_cid, mbar_info.remote_cid,
                               mbar_info.remote_hw_cta, addr,
                               flash_gpgpu_sim::cluster_mbar_op::ARRIVE, 1);
          } else if (is_arrive) {
            inject_remote_mbar(m_shader, src_cid, mbar_info.remote_cid,
                               mbar_info.remote_hw_cta, addr,
                               flash_gpgpu_sim::cluster_mbar_op::ARRIVE, count);
          } else if (is_expect_tx) {
            inject_remote_mbar(m_shader, src_cid, mbar_info.remote_cid,
                               mbar_info.remote_hw_cta, addr,
                               flash_gpgpu_sim::cluster_mbar_op::EXPECT_TX,
                               count);
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
        notify_mbarrier_phase_change(released_warps);
        release_warps(released_warps);
        notify_remote_waiters(cta_id, addr);

      } else if (is_arrive) {
        auto released_warps = m_mbarrier_manager.arrive(
            m_shader->get_gpu(), thread_index, addr, count);
        notify_mbarrier_phase_change(released_warps);
        release_warps(released_warps);
        notify_remote_waiters(cta_id, addr);

      } else if (is_expect_tx) {
        m_mbarrier_manager.expect_tx(m_shader->get_gpu(), thread_index, addr,
                                     count);
      }
    }

    const unsigned arrive_lat =
        m_shader->get_config()->gpgpu_mbarrier_arrive_latency;
    // Isolated arrive is 6; H200 arrive.expect_tx(0)+wait is 71 vs 6+43.
    unsigned hold = arrive_lat;
    if (is_arrive && is_expect_tx && expect_tx_zero)
      hold += 20;
    if (is_arrive && hold > 0)
      hold_warp(warp_id, hold, BARRIER_WAIT_MBARRIER);
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
