#ifndef FLASH_GPGPU_SIM_MBARRIER_H
#define FLASH_GPGPU_SIM_MBARRIER_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>

class gpgpu_sim;
namespace flash_gpgpu_sim {

class mbarrier_manager_t {

  /**
   * Implement the mbarrier instruction.
   * NOTE: So far, this is more like a idealized implementation with some
   * limitations:
   * 1. It does not access the shared memory,
   * 2. only support CTA level synchronization.
   * 3. Does not support thread-level barrier (this is a limitation from
   * GPGPU-Sim, we can fix later). So far if one thread in a warp is blocked by
   * a barrier, the entire warp is blocked.
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
    const int m_expected_count;
    int m_pending_arrival_count;
    // This is for TMA interaction. It may change every phase.
    int m_tx_count;
    int m_phase;
    std::set<int> m_waiting_warps;
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
  };

  void init(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr,
            int expected_count);
  void inval(gpgpu_sim *gpu, const thread_index_t &thread_index, uint64_t addr);

  /**
   * Try to wait on the mbarrier at addr with parity for warp warp_id.
   * @return true if the wait is satisfied.
   */
  bool try_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
                uint64_t addr, int parity);

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
   * Increase the expected tx count for the mbarrier at addr.
   * @return void
   */
  void expect_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                 uint64_t addr, int expected_tx_count);

  /**
   * Clean up all mbarriers for a given hw_cta_id when the CTA completes.
   * This prevents stale barriers when hw_cta_ids get recycled.
   */
  void cleanup_cta(unsigned hw_cta_id);

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
