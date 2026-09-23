// Helpers to read GPGPU-Sim topology knobs from gpgpusim.config in CWD.
// Integration tests run from the config/run directory, so these values match
// the active simulation topology.
//
// Use with GTEST_SKIP() when a test requires multi-SM clusters or multi-cluster
// isolation and would hang or give false failures on the wrong config.
//
// Documented skip matrix (which tests skip on REDUCED / CLUSTER2x1 / 2x2 / 4x4):
//   docs/cluster_noc/tests.md
//   docs/cluster_noc/programming_model.md
//
// Naming: physical packing is CLUSTERmxn (m = n_cores_per_cluster,
// n = n_clusters). TB cluster size is a launch attribute, not the config name.
// Reduced m>2: SM120_RTX5090_REDUCED_CLUSTER4x4 (m=4, n=4).
// Full-chip H200: SM90_H200_CLUSTER132.
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

#include <gtest/gtest.h>

namespace flash_test {

struct GpgpuSimTopology {
  unsigned n_clusters = 1;
  unsigned n_cores_per_cluster = 1;
  unsigned total_sms = 0;
  bool dsm_enable = false;
  bool hetero_gpc = false;
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
    } else if (key == "-gpgpu_dsm_enable") {
      unsigned v = 0;
      if (iss >> v && v != 0) {
        topo.dsm_enable = true;
      }
    } else if (key == "-gpgpu_gpc_sms") {
      unsigned v = 0;
      unsigned min_v = 0, max_v = 0, n = 0, sum = 0;
      char sep = 0;
      if (iss >> v) {
        min_v = max_v = v;
        sum = v;
        n = 1;
        while (iss >> sep >> v) {
          if (sep != ',') break;
          if (v < min_v) min_v = v;
          if (v > max_v) max_v = v;
          sum += v;
          n++;
        }
      }
      if (n) {
        topo.total_sms = sum;
        topo.hetero_gpc = (min_v != max_v);
        if (max_v) topo.n_cores_per_cluster = max_v;
      }
    }
  }
  if (!topo.total_sms) {
    topo.total_sms = topo.n_clusters * topo.n_cores_per_cluster;
  }
  return topo;
}

// Print to stdout and stderr so a topology skip is visible in suite logs
// (GTEST_SKIP alone is easy to miss in a long [  SKIPPED ] list).
inline void warn_topology_skip(const std::string &msg) {
  const char *suite = "?";
  const char *name = "?";
  if (const auto *info = ::testing::UnitTest::GetInstance()->current_test_info()) {
    suite = info->test_suite_name();
    name = info->name();
  }
  std::fprintf(stderr, "WARNING: skipped %s.%s: %s\n", suite, name, msg.c_str());
  std::fprintf(stdout, "WARNING: skipped %s.%s: %s\n", suite, name, msg.c_str());
  std::fflush(stderr);
  std::fflush(stdout);
}

// Skip if the active config has fewer than min_cores SMs per cluster.
// One-producer cluster TMA and peer mbarrier complete require >= 2.
#define SKIP_IF_N_CORES_PER_CLUSTER_LT(min_cores)                              \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (__topo.found_config &&                                                 \
        __topo.n_cores_per_cluster < static_cast<unsigned>(min_cores)) {       \
      std::ostringstream __skip;                                               \
      __skip << "Requires -gpgpu_n_cores_per_cluster >= " << (min_cores)       \
             << " (got " << __topo.n_cores_per_cluster                         \
             << "). Use SM120_RTX5090_REDUCED_CLUSTER2x1 / 4x4.";              \
      ::flash_test::warn_topology_skip(__skip.str());                          \
      GTEST_SKIP() << __skip.str();                                            \
    }                                                                          \
  } while (0)

// Skip unless the intra-GPC DSM fabric is on.
#define SKIP_IF_CLUSTER_NOC_OFF()                                              \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (__topo.found_config && !__topo.dsm_enable) {                           \
      std::ostringstream __skip;                                               \
      __skip << "Requires -gpgpu_dsm_enable 1. "                               \
                "Use SM120_RTX5090_REDUCED_CLUSTER2x1 / 2x2 / 4x4.";           \
      ::flash_test::warn_topology_skip(__skip.str());                          \
      GTEST_SKIP() << __skip.str();                                            \
    }                                                                          \
  } while (0)
#define SKIP_IF_NOT_CLUSTER_NOC() SKIP_IF_CLUSTER_NOC_OFF()

// Skip if fewer than min_clusters physical clusters (multi-cluster isolation).
#define SKIP_IF_NOT_HETERO_GPC()                                               \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (!__topo.hetero_gpc) {                                                  \
      std::ostringstream __skip;                                               \
      __skip << "Requires mixed -gpgpu_gpc_sms. Use SM90_H200_CLUSTER132 "     \
                "(found="                                                      \
             << __topo.found_config << " hetero=" << __topo.hetero_gpc         \
             << " sms=" << __topo.total_sms                                    \
             << " m=" << __topo.n_cores_per_cluster                            \
             << " n=" << __topo.n_clusters << ").";                            \
      ::flash_test::warn_topology_skip(__skip.str());                          \
      GTEST_SKIP() << __skip.str();                                            \
    }                                                                          \
  } while (0)

#define SKIP_IF_N_CLUSTERS_LT(min_clusters)                                    \
  do {                                                                         \
    const auto __topo = ::flash_test::read_gpgpusim_topology();                \
    if (__topo.found_config &&                                                 \
        __topo.n_clusters < static_cast<unsigned>(min_clusters)) {             \
      std::ostringstream __skip;                                               \
      __skip << "Requires -gpgpu_n_clusters >= " << (min_clusters)             \
             << " (got " << __topo.n_clusters                                  \
             << "). Use SM120_RTX5090_REDUCED_CLUSTER2x2 / 4x4 "               \
                "(or similar).";                                               \
      ::flash_test::warn_topology_skip(__skip.str());                          \
      GTEST_SKIP() << __skip.str();                                            \
    }                                                                          \
  } while (0)

} // namespace flash_test

#endif // FLASH_TEST_GPGPUSIM_CONFIG_TOPOLOGY_H
