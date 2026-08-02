#include <cassert>

#include "../../../src/gpgpu-sim/flash/mbarrier.h"

int main() {
  flash_gpgpu_sim::mbarrier_manager_t manager;
  flash_gpgpu_sim::mbarrier_manager_t::thread_index_t thread_index{
      0, 3, 7, 1};

  constexpr uint64_t default_addr = 0x80;
  manager.init(nullptr, thread_index, default_addr, 1);
  auto *default_barrier =
      manager.get_mbarrier(thread_index.sw_cta_id, default_addr);
  manager.prepare_async_arrival(nullptr, thread_index, default_addr, true);
  assert(default_barrier->m_pending_arrival_count == 2);
  assert(!manager.try_wait(nullptr, thread_index, default_addr, 0));
  assert(manager.arrive(nullptr, thread_index, default_addr, 1).empty());
  const auto default_released =
      manager.arrive(nullptr, thread_index, default_addr, 1);
  assert(default_released.count(thread_index.hw_warp_id) == 1);
  assert(default_barrier->m_phase == 1);

  constexpr uint64_t noinc_addr = 0x88;
  manager.init(nullptr, thread_index, noinc_addr, 1);
  auto *noinc_barrier =
      manager.get_mbarrier(thread_index.sw_cta_id, noinc_addr);
  manager.prepare_async_arrival(nullptr, thread_index, noinc_addr, false);
  assert(noinc_barrier->m_pending_arrival_count == 1);
  assert(!manager.try_wait(nullptr, thread_index, noinc_addr, 0));
  const auto noinc_released =
      manager.arrive(nullptr, thread_index, noinc_addr, 1);
  assert(noinc_released.count(thread_index.hw_warp_id) == 1);
  assert(noinc_barrier->m_phase == 1);
}
