#ifndef FLASH_GPGPU_SIM_CLUSTER_HANG_PREVENT_H
#define FLASH_GPGPU_SIM_CLUSTER_HANG_PREVENT_H

// Sim hang preventers for unsupported cluster programming patterns.
// These are not hardware detectors. Recognized waits and hop-scale
// stalls must not trip.

#include <cstdlib>

namespace flash_gpgpu_sim {

static const unsigned kHangPcHist = 8;
static const unsigned kHangTightLoopPcs = 6;
static const unsigned kHangWatchdogDefault = 8192;

inline unsigned hang_watchdog_threshold(unsigned configured) {
  const char *env = std::getenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
  if (env && env[0] != '\0')
    return static_cast<unsigned>(std::strtoul(env, nullptr, 10));
  return configured;
}

// How long a peer DSM/TMA access stays "recent" with no further peer touch.
// Hop-scale so a remote load's scoreboard stall stays armed, but a later
// local tight loop does not.
inline unsigned peer_arm_quiet_limit(unsigned hop, unsigned tma_hop,
                                     unsigned trywait,
                                     unsigned fabric_rtt = 0) {
  unsigned n = hop * 2 + 64;
  if (tma_hop + 64 > n)
    n = tma_hop + 64;
  if (trywait + 64 > n)
    n = trywait + 64;
  if (fabric_rtt + 64 > n)
    n = fabric_rtt + 64;
  return n;
}

// Armed only while the last peer access is recent. Recognized waits
// (try_wait park, bar.sync, …) drop the arm immediately. Fabric
// outstanding / live RTT keeps the arm until the path is idle.
inline bool peer_access_still_armed(bool saw_peer, bool at_recognized_wait,
                                    unsigned quiet_since_peer,
                                    unsigned quiet_limit,
                                    bool fabric_outstanding = false) {
  if (!saw_peer || at_recognized_wait)
    return false;
  if (fabric_outstanding)
    return true;
  return quiet_since_peer < quiet_limit;
}

inline unsigned unique_pc_count(const unsigned long long *hist, unsigned n) {
  unsigned unique = 0;
  for (unsigned i = 0; i < n; i++) {
    bool seen = false;
    for (unsigned j = 0; j < i; j++) {
      if (hist[j] == hist[i]) {
        seen = true;
        break;
      }
    }
    if (!seen)
      unique++;
  }
  return unique;
}

// Peer DSM/TMA access, no mbarrier interest, not parked on a recognized
// wait, and the warp is looping a small PC set (or sitting on one PC).
inline bool spin_watchdog_should_trip(bool enabled, unsigned threshold,
                                      bool at_recognized_wait,
                                      bool has_mbar_interest,
                                      bool saw_peer_access,
                                      unsigned unique_recent_pcs,
                                      unsigned watch_cycles) {
  if (!enabled || threshold == 0)
    return false;
  if (at_recognized_wait || has_mbar_interest || !saw_peer_access)
    return false;
  if (unique_recent_pcs == 0 || unique_recent_pcs > kHangTightLoopPcs)
    return false;
  return watch_cycles >= threshold;
}

// Partial-warp try_wait parked at the same time as a bar.sync waiter
// in the same CTA, for longer than hop-scale latencies.
inline bool mixed_bar_trywait_should_trip(bool enabled, unsigned threshold,
                                          bool partial_trywait,
                                          bool sibling_at_bar_sync,
                                          unsigned mix_cycles) {
  if (!enabled || threshold == 0)
    return false;
  if (!partial_trywait || !sibling_at_bar_sync)
    return false;
  return mix_cycles >= threshold;
}

} // namespace flash_gpgpu_sim

#endif
