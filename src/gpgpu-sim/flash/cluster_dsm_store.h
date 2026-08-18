#ifndef FLASH_GPGPU_SIM_CLUSTER_DSM_STORE_H
#define FLASH_GPGPU_SIM_CLUSTER_DSM_STORE_H

// Issuer-side policy for remote DSM stores (see docs/cluster_noc.md).
// Default: inject a NoC hop and also write peer smem immediately.
// `-gpgpu_dsm_store_immediate 0` writes peer smem only on DSM_STORE deliver.
// FLASHGPU_DSM_STORE_IMMEDIATE overrides the parsed knob at store time so
// tests can switch without a second config file.

#include <cstdlib>

namespace flash_gpgpu_sim {

inline const char *dsm_store_immediate_env_name() {
  return "FLASHGPU_DSM_STORE_IMMEDIATE";
}

inline bool dsm_store_immediate_enabled(bool configured) {
  const char *env = std::getenv(dsm_store_immediate_env_name());
  if (env && env[0] != '\0')
    return std::strtoul(env, nullptr, 10) != 0;
  return configured;
}

// Shipped remote-store issue path used by st_impl.
// Returns true if the NoC accepted the store (caller must not write again).
// When immediate is false, peer smem changes only on NoC deliver.
template <typename InjectFn, typename WritePeerFn>
inline bool issue_remote_dsm_store(bool is_remote, bool noc_enabled,
                                   bool immediate, InjectFn &&inject,
                                   WritePeerFn &&write_peer) {
  if (!is_remote || !noc_enabled)
    return false;
  if (!inject())
    return false;
  if (immediate)
    write_peer();
  return true;
}

} // namespace flash_gpgpu_sim

#endif
