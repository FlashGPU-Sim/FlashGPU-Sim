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
    mbarrier_t(int id, uint64_t addr, int expected_count)
        : m_id(id), m_addr(addr), m_expected_count(expected_count),
          m_arrived_count(0), m_expected_tx_count(0), m_arrived_tx_count(0),
          m_phase(0) {}

    const int m_id;
    const uint64_t m_addr;
    const int m_expected_count;
    int m_arrived_count;
    // This is for TMA interaction. It may change every phase.
    int m_expected_tx_count;
    int m_arrived_tx_count;
    int m_phase;
    std::set<int> m_waiting_warps;
  };

public:
  mbarrier_manager_t() : m_next_id(0) {}

  mbarrier_t *get_mbarrier(uint64_t addr) {
    auto it = addr_to_mbarrier_map.find(addr);
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

  void init(gpgpu_sim *gpu, int cta_id, int warp_id, uint64_t addr,
            int expected_count);
  void inval(gpgpu_sim *gpu, int cta_id, int warp_id, uint64_t addr);

  /**
   * Try to wait on the mbarrier at addr with parity for warp warp_id.
   * @return true if the wait is satisfied.
   */
  bool try_wait(gpgpu_sim *gpu, int cta_id, int warp_id, uint64_t addr,
                int parity);

  /**
   * Arrive at the mbarrier at addr with arrival_count for warp warp_id.
   * @return the set of warp ids that are released due to this arrive.
   */
  std::set<int> arrive(gpgpu_sim *gpu, int cta_id, int warp_id, uint64_t addr,
                       int arrival_count);

private:
  int m_next_id;
  std::unordered_map<uint64_t, std::unique_ptr<mbarrier_t>>
      addr_to_mbarrier_map;
};
} // namespace flash_gpgpu_sim

// Forward declarations for mbarrier_impl
class ptx_instruction;
class ptx_thread_info;
void handle_mbarrier_inst(const ptx_instruction *pI, ptx_thread_info *thread);
#endif