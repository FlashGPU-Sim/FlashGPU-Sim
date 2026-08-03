// Helpers to read GPGPU-Sim topology knobs from gpgpusim.config in CWD.
// Integration tests run from the config/run directory, so these values match
// the active simulation topology.
//
// Use with GTEST_SKIP() when a test requires multi-SM clusters or multi-cluster
// isolation and would hang or give false failures on the wrong config.
//
// Documented skip matrix (which tests skip on REDUCED / CLUSTER2x1 / 2x2 / 4x4):
//   docs/cluster_cta2_explain.md §19.1
//   docs/cluster_cta2_realLaunch.md §5 (Tests / topology skips)
//
// Naming: physical packing is CLUSTERmxn (m = n_cores_per_cluster,
// n = n_clusters). TB cluster size is a launch attribute, not the config name.
// GPC-aligned full: SM120_RTX5090_CLUSTER16x11 (m=16, n=11).
// Reduced m>2:     SM120_RTX5090_REDUCED_CLUSTER4x4 (m=4, n=4).
//
// Note: some negative tests (e.g. cluster size > physical m) use a manual
// GTEST_SKIP when m is *too large*, not these LT macros.

#ifndef FLASH_TEST_GPGPUSIM_CONFIG_TOPOLOGY_H
#define FLASH_TEST_GPGPUSIM_CONFIG_TOPOLOGY_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace flash_test {

struct GpgpuSimTopology {
  unsigned n_clusters = 1;
  unsigned n_cores_per_cluster = 1;
  bool found_config = false;
};

// Parse -gpgpu_n_clusters / -gpgpu_n_cores_per_cluster from gpgpusim.config.
// Searches: CWD, ./gpgpusim.config (explicit).
inline GpgpuSimTopology read_gpgpusim_topology(
    const char *config_path = "gpgpusim.config") {
  GpgpuSimTopology topo;
  std::ifstream in(config_path);
  if (!in) {
    return topo;
  }
  topo.found_config = true;
  std::string line;
  while (std::getline(in, line)) {
    // Strip comments
    auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key)) {
      continue;
    }
    if (key == "-gpgpu_n_clusters") {
      unsigned v = 0;
      if (iss >> v) {
        topo.n_clusters = v;
      }
    } else if (key == "-gpgpu_n_cores_per_cluster") {
      unsigned v = 0;
      if (iss >> v) {
        topo.n_cores_per_cluster = v;
      }
    }
  }
  return topo;
}

// Skip if the active config has fewer than min_cores SMs per cluster.
// One-producer cluster TMA and peer mbarrier complete require >= 2.
#define SKIP_IF_N_CORES_PER_CLUSTER_LT(min_cores)                              \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (__topo.found_config &&                                                 \
        __topo.n_cores_per_cluster < static_cast<unsigned>(min_cores)) {       \
      GTEST_SKIP() << "Requires -gpgpu_n_cores_per_cluster >= " << (min_cores) \
                   << " (got " << __topo.n_cores_per_cluster                    \
                   << "). Use SM120_RTX5090_REDUCED_CLUSTER2x1 / 4x4 "         \
                      "(or CLUSTER16x11).";                                   \
    }                                                                          \
  } while (0)

// Skip if fewer than min_clusters physical clusters (multi-cluster isolation).
#define SKIP_IF_N_CLUSTERS_LT(min_clusters)                                    \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (__topo.found_config &&                                                 \
        __topo.n_clusters < static_cast<unsigned>(min_clusters)) {             \
      GTEST_SKIP() << "Requires -gpgpu_n_clusters >= " << (min_clusters)       \
                   << " (got " << __topo.n_clusters                            \
                   << "). Use SM120_RTX5090_REDUCED_CLUSTER2x2 / 4x4 "         \
                      "(or similar).";                                        \
    }                                                                          \
  } while (0)

} // namespace flash_test

#endif // FLASH_TEST_GPGPUSIM_CONFIG_TOPOLOGY_H
