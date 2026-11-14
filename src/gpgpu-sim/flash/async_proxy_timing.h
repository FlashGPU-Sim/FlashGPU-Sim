#ifndef FLASH_GPGPU_SIM_ASYNC_PROXY_TIMING_H_
#define FLASH_GPGPU_SIM_ASYNC_PROXY_TIMING_H_

#include "panic.h"
#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {

// CTA-scoped async-proxy visibility with per-warp synchronization. A TMA
// load advances the producer epoch for the whole CTA. Each warp observes the
// new epoch only when it issues a fence carrying a completion dependency
// token. Tokenless fences occupy the proxy frontend but cannot make the
// synchronization visible to a later dependency-barrier wait.
class async_proxy_timing {
public:
  struct config {
    unsigned completion_stall = 0;
    unsigned dirty_completion_extra_stall = 0;
    unsigned initiation_stall = 0;
    unsigned dirty_initiation_stall = 0;
  };

  async_proxy_timing(unsigned warp_count, config timing)
      : timing_(timing), observed_epoch_(warp_count, 0) {
    if (warp_count == 0)
      flash_gpgpu_sim::panic("async proxy CTA has no warps");
  }

  void record_tma_load() { ++producer_epoch_; }

  unsigned issue_fence(unsigned local_warp, bool completion_token) {
    if (local_warp >= observed_epoch_.size())
      flash_gpgpu_sim::panic("async proxy warp is outside CTA");
    const bool dirty = observed_epoch_[local_warp] != producer_epoch_;
    if (!completion_token)
      return dirty ? timing_.dirty_initiation_stall : timing_.initiation_stall;

    observed_epoch_[local_warp] = producer_epoch_;
    return timing_.completion_stall +
           (dirty ? timing_.dirty_completion_extra_stall : 0);
  }

  uint64_t producer_epoch() const { return producer_epoch_; }
  uint64_t observed_epoch(unsigned local_warp) const {
    if (local_warp >= observed_epoch_.size())
      panic("async proxy warp is outside CTA");
    return observed_epoch_[local_warp];
  }

private:
  config timing_;
  uint64_t producer_epoch_ = 0;
  std::vector<uint64_t> observed_epoch_;
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_ASYNC_PROXY_TIMING_H_
