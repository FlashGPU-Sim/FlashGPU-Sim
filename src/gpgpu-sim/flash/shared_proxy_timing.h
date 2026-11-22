#ifndef FLASH_GPGPU_SIM_SHARED_PROXY_TIMING_H_
#define FLASH_GPGPU_SIM_SHARED_PROXY_TIMING_H_

#include "panic.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {

// Models the visibility path between matrix stores and an async-proxy fence.
//
// Matrix stores finish their ordinary shared-memory data service before their
// contents necessarily become visible to the async proxy.  Visibility is
// admitted independently by each scheduler partition.  A fence waits for the
// most recent matrix store issued by its warp, while unrelated fences remain
// free to complete immediately.
class shared_proxy_timing {
public:
  struct config {
    unsigned visibility_latency = 0;
    unsigned initiation_interval = 0;
  };

  shared_proxy_timing(unsigned warp_count, unsigned scheduler_count,
                      config timing)
      : timing_(timing), warp_visible_cycle_(warp_count, 0),
        scheduler_next_cycle_(scheduler_count, 0) {
    if (warp_count == 0)
      flash_gpgpu_sim::panic("shared proxy has no warps");
    if (scheduler_count == 0)
      flash_gpgpu_sim::panic("shared proxy has no scheduler partitions");
  }

  void reset() {
    std::fill(warp_visible_cycle_.begin(), warp_visible_cycle_.end(), 0);
    std::fill(scheduler_next_cycle_.begin(), scheduler_next_cycle_.end(), 0);
  }

  void reset_warp(unsigned warp) {
    if (warp >= warp_visible_cycle_.size())
      panic("shared proxy warp is outside the core");
    warp_visible_cycle_[warp] = 0;
  }

  void record_matrix_store(unsigned warp, unsigned scheduler,
                           uint64_t service_cycle) {
    if (warp >= warp_visible_cycle_.size())
      flash_gpgpu_sim::panic("shared proxy warp is outside the core");
    if (scheduler >= scheduler_next_cycle_.size())
      flash_gpgpu_sim::panic("shared proxy scheduler is outside the core");

    const uint64_t visibility_start =
        std::max(service_cycle, scheduler_next_cycle_[scheduler]);
    warp_visible_cycle_[warp] =
        std::max(warp_visible_cycle_[warp],
                 visibility_start + timing_.visibility_latency);
    scheduler_next_cycle_[scheduler] =
        visibility_start + timing_.initiation_interval;
  }

  bool fence_ready(unsigned warp, uint64_t cycle) const {
    return cycle >= visible_cycle(warp);
  }

  uint64_t visible_cycle(unsigned warp) const {
    if (warp >= warp_visible_cycle_.size())
      panic("shared proxy warp is outside the core");
    return warp_visible_cycle_[warp];
  }

private:
  config timing_;
  std::vector<uint64_t> warp_visible_cycle_;
  std::vector<uint64_t> scheduler_next_cycle_;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SHARED_PROXY_TIMING_H_
