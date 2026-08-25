#ifndef FLASH_GPGPU_SIM_CLUSTER_DSM_STORE_H
#define FLASH_GPGPU_SIM_CLUSTER_DSM_STORE_H

// Issuer-side policy for remote DSM stores
// (docs/cluster_noc/programming_model.md). Default (0): inject a NoC hop; write
// peer smem only on DSM_STORE deliver (closer to silicon). 1 = also write at
// issue (opt-in). FLASHGPU_DSM_STORE_IMMEDIATE overrides the parsed knob at
// store time so tests can switch without a second config file.

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
