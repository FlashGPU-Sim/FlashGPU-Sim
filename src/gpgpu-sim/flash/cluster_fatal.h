#ifndef FLASH_GPGPU_SIM_CLUSTER_FATAL_H
#define FLASH_GPGPU_SIM_CLUSTER_FATAL_H

namespace flash_gpgpu_sim {

// Production leaves this null and the fatal helpers abort(). A test installs
// an observer so the helper can return; core_cycle() runs inside an OpenMP
// region, so throwing there calls std::terminate.
using cluster_fatal_observer_fn = void (*)(const char *message);

void set_cluster_fatal_observer(cluster_fatal_observer_fn observer);

// True after an observer accepted a fatal check in this process.
bool cluster_fatal_latched();

} // namespace flash_gpgpu_sim

#endif
