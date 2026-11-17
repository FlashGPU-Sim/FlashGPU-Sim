#ifndef FLASH_GPGPU_SIM_MBARRIER_H
#define FLASH_GPGPU_SIM_MBARRIER_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

class gpgpu_sim;
namespace flash_gpgpu_sim {

enum class mbarrier_recheck_action_t {
  KEEP_SLEEPING,
  RETURN_TRUE,
  RETURN_FALSE,
};

uint64_t mbarrier_saturating_add(uint64_t cycle, uint64_t delta);
uint64_t mbarrier_hint_ns_to_cycles(uint32_t hint_ns, unsigned core_freq_hz);
mbarrier_recheck_action_t mbarrier_classify_recheck(bool phase_complete,
                                                    uint64_t cycle,
                                                    uint64_t deadline_cycle);
uint64_t mbarrier_wake_on_phase_notification(uint64_t scheduled_wake_cycle,
                                             uint64_t notification_cycle);
bool mbarrier_should_delay_phase_wakeup(bool suspended,
                                        bool phase_notification_pending,
                                        bool all_true, unsigned latency);
bool mbarrier_latches_initial_predicate(bool has_time_hint,
                                        unsigned predicate_latency);
uint64_t mbarrier_result_release_cycle(uint64_t issue_cycle,
                                       uint64_t resolution_cycle,
                                       unsigned predicate_latency,
                                       unsigned phase_wakeup_latency);

enum class cluster_mbar_op : uint8_t {
  ARRIVE = 0,
  EXPECT_TX,
  COMPLETE_TX,
  TRY_COMPLETE_TX,
  WAIT_REG,
  WAIT_DONE,
};

class mbarrier_manager_t {

public:
  // Remote try_wait interest (other SM waiting on this barrier).
  struct remote_waiter_t {
    unsigned src_cid;
    unsigned src_hw_cta;
    unsigned src_warp_id;
    int parity;
  };

private:
  /**
   * mbarrier objects live in simulator state (not 64-bit smem contents).
   * Local CTA ops and remote (mapa) arrive/try_wait/expect/complete via
   * the intra-GPC fabric are supported. A blocked thread stalls the whole warp
   * (GPGPU-Sim SIMT). try_wait.parity optional timeout sets dest pred
   * false on expiry, true if the waited phase completed.
   */
  struct mbarrier_t {
    mbarrier_t(int id, int hw_cta_id, int sw_cta_id, uint64_t addr,
               int expected_count)
        : m_id(id), m_hw_cta_id(hw_cta_id), m_sw_cta_id(sw_cta_id),
          m_addr(addr), m_expected_count(expected_count),
          m_pending_arrival_count(expected_count), m_tx_count(0), m_phase(0) {}

    const int m_id;
    const int m_hw_cta_id;
    const int m_sw_cta_id;
    const uint64_t m_addr;
    int m_expected_count;
    int m_pending_arrival_count;
    // This is for TMA interaction. It may change every phase.
    int m_tx_count;
    int m_phase;
    std::set<int> m_waiting_warps;
    std::vector<remote_waiter_t> m_remote_waiters;
  };

public:
  mbarrier_manager_t() : m_next_id(0) {}

  mbarrier_t *get_mbarrier(int sw_cta_id, uint64_t addr) {
    auto it = addr_to_mbarrier_map.find(std::make_pair(sw_cta_id, addr));
    if (it != addr_to_mbarrier_map.end()) {
      return it->second.get();
    } else {
      return nullptr;
    }
  }

  void reset() {
    m_next_id = 0;
    addr_to_mbarrier_map.clear();
  }

  void dump() const;

  struct thread_index_t {
    int hw_cta_id;
    int hw_warp_id;
    int sw_cta_id;
    int sw_warp_id;
    int sm_id = -1;
  };

  void init(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
            int expected_count);
  void inval(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr);

  /** Query the current phase without registering or changing barrier state. */
  bool test_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
                 uint64_t addr, int parity) const;

  /** Register one warp for a phase-change notification. */
  void register_wait(const thread_index_t &thread_index, uint64_t addr);

  /** Remove a waiter after timeout or lifecycle cleanup. */
  void cancel_wait(int sw_cta_id, uint64_t addr, int hw_warp_id);

  /**
   * Query whether the waited parity has completed. Does not park the warp.
   * If the current phase is already satisfied, try_advance runs and any
   * previously parked waiters are returned in released_on_advance.
   */
  bool try_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
                uint64_t addr, int parity,
                std::set<int> *released_on_advance = nullptr);

  // Record a parked waiter so a later arrive/complete_tx can release it.
  void enqueue_wait(const thread_index_t &thread_index, uint64_t addr);

  /**
   * Arrive at the mbarrier at addr with arrival_count for warp warp_id.
   * @return the set of warp ids that are released due to this arrive.
   */
  std::set<int> arrive(gpgpu_sim *gpu, const thread_index_t &thread_index,
                       uint64_t addr, int arrival_count);

  // The default form reserves one pending arrival; .noinc only validates it.
  void prepare_async_arrival(gpgpu_sim *gpu, const thread_index_t &thread_index,
                             uint64_t addr, bool increment_pending);

  /**
   * Complete transaction at the mbarrier at addr with completed_tx_count for
   * warp warp_id.
   * @return the set of warp ids that are released due to this complete_tx.
   */
  std::set<int> complete_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                            uint64_t addr, int completed_tx_count);

  /**
   * Like complete_tx, but no-ops when the mbarrier is missing. Despite the
   * historical name, initialized peers retain early completions as negative
   * tx-count; the caller must deliver each transaction exactly once.
   */
  std::set<int> try_complete_tx_if_pending(gpgpu_sim *gpu,
                                           const thread_index_t &thread_index,
                                           uint64_t addr,
                                           int completed_tx_count);

  /**
   * Increase the expected tx count for the mbarrier at addr.
   * @return void
   */
  void expect_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                 uint64_t addr, int expected_tx_count);

  /**
   * Register a remote try_wait interest on the barrier at addr.
   * @return true if the wait is already satisfied (parity advanced).
   * If false, waiter is stored and will be notified on phase advance.
   */
  bool register_remote_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
                            uint64_t addr, int parity, unsigned src_cid,
                            unsigned src_hw_cta, unsigned src_warp_id);

  /**
   * After a phase advance, return and clear remote waiters whose parity is
   * now satisfied (waiting for previous phase).
   */
  std::vector<remote_waiter_t> take_satisfied_remote_waiters(int sw_cta_id,
                                                             uint64_t addr);

  /**
   * Clean up all mbarriers for a given hw_cta_id when the CTA completes.
   * This prevents stale barriers when hw_cta_ids get recycled.
   */
  void cleanup_cta(unsigned hw_cta_id);

  // Drop a local waiter without advancing the phase (try_wait timeout).
  void cancel_wait(int hw_warp_id);

private:
  int m_next_id;
  struct pair_hash {
    size_t operator()(const std::pair<int, uint64_t> &p) const noexcept {
      return std::hash<int>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
    }
  };
  std::unordered_map<std::pair<int, uint64_t>, std::unique_ptr<mbarrier_t>,
                     pair_hash>
      addr_to_mbarrier_map;

  /**
   * Check if the mbarrier is satisfied, and release warps if so.
   * @return the set of warp ids that are released.
   */
  std::set<int> try_advance(gpgpu_sim *gpu, const thread_index_t &thread_index,
                            mbarrier_t *mbarrier);
};
} // namespace flash_gpgpu_sim

// Forward declarations for mbarrier_impl
class ptx_instruction;
class ptx_thread_info;
void handle_mbarrier_inst(const ptx_instruction *pI, ptx_thread_info *thread);
#endif
