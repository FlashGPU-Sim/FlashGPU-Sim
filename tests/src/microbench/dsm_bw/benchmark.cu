// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory dsm_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

#include "kernels.cuh"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class Method { Load, Tma, Store, Mixed };

struct TestConfig {
  std::string label;
  Method method = Method::Load;
  dsm::Pattern pattern = dsm::Pattern::PairUnidirectional;
  int cluster_size = 2;
  int param = 0;
  int active_warps = 32;
  int chunk_bytes = 16 * 1024;
  int pressure_mode = 0;
  int pressure_warps = 0;
  int threads = dsm::kThreads;
  int epoch_bytes = 64 * 1024;
  int mixed_kind = -1;
};

struct Options {
  std::string suite = "smoke";
  std::string filter;
  std::string output;
  int bytes = 0;
  int warmups = 2;
  int iterations = 5;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(cudaError_t error, const char *where) {
  if (error != cudaSuccess) {
    std::ostringstream stream;
    stream << where << ": " << cudaGetErrorString(error);
    fail(stream.str());
  }
}

const char *method_name(Method method) {
  switch (method) {
  case Method::Load:
    return "load";
  case Method::Tma:
    return "tma";
  case Method::Store:
    return "store";
  case Method::Mixed:
    return "mixed";
  }
  return "unknown";
}

const char *pattern_name(dsm::Pattern pattern) {
  switch (pattern) {
  case dsm::Pattern::PairUnidirectional:
    return "pair_unidirectional";
  case dsm::Pattern::PairBidirectional:
    return "pair_bidirectional";
  case dsm::Pattern::Ring:
    return "ring";
  case dsm::Pattern::PackedRing:
    return "packed_ring";
  }
  return "unknown";
}

unsigned long long expected_checksum(int source_rank, int epoch_bytes) {
  unsigned long long sum = 0;
  for (int word = 0; word < epoch_bytes / 4; ++word) {
    sum += static_cast<unsigned long long>(
        (0x9e3779b9u * static_cast<uint32_t>(source_rank + 1)) ^
        (0x85ebca6bu * static_cast<uint32_t>(word + 17)));
  }
  return sum;
}

uint32_t expected_store_word(int source_rank, int component) {
  return (0x6a09e667u * static_cast<uint32_t>(source_rank + 1)) ^
         (0xbb67ae85u + 0x9e3779b9u * static_cast<uint32_t>(component + 1));
}

unsigned long long expected_store_checksum(int source_rank, int epoch_bytes) {
  unsigned long long sum = 0;
  for (int word = 0; word < epoch_bytes / 4; ++word) {
    sum += expected_store_word(source_rank, word & 3);
  }
  return sum;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        fail(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (argument == "--suite") {
      options.suite = value("--suite");
    } else if (argument == "--filter") {
      options.filter = value("--filter");
    } else if (argument == "--bytes") {
      options.bytes = std::atoi(value("--bytes"));
    } else if (argument == "--warmups") {
      options.warmups = std::atoi(value("--warmups"));
    } else if (argument == "--iterations") {
      options.iterations = std::atoi(value("--iterations"));
    } else if (argument == "--output") {
      options.output = value("--output");
    } else if (argument == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--suite smoke|bandwidth|topology|all] [--filter LABEL]"
                   " [--warmups N] [--iterations N] [--output FILE]\n";
      std::exit(0);
    } else {
      fail("unknown option: " + argument);
    }
  }
  if (options.warmups < 0 || options.iterations <= 0) {
    fail("warmups must be non-negative and iterations must be positive");
  }
  if (options.suite != "smoke" && options.suite != "bandwidth" &&
      options.suite != "topology" && options.suite != "all") {
    fail("suite must be smoke, bandwidth, topology, or all");
  }
  return options;
}

TestConfig config(std::string label, Method method, dsm::Pattern pattern,
                  int cluster_size = 2, int param = 0) {
  TestConfig value;
  value.label = std::move(label);
  value.method = method;
  value.pattern = pattern;
  value.cluster_size = cluster_size;
  value.param = param;
  return value;
}

TestConfig mixed_config(std::string label, int mixed_kind) {
  auto value = config(std::move(label), Method::Mixed,
                      dsm::Pattern::PairBidirectional);
  value.active_warps = 16;
  value.mixed_kind = mixed_kind;
  return value;
}

void append_smoke(std::vector<TestConfig> &tests) {
  tests.push_back(config("load_size_uni", Method::Load,
                         dsm::Pattern::PairUnidirectional));
  tests.push_back(config("load_size_bi", Method::Load,
                         dsm::Pattern::PairBidirectional));
  tests.push_back(config("store_size_uni", Method::Store,
                         dsm::Pattern::PairUnidirectional));
  tests.push_back(config("tma_size_bi", Method::Tma,
                         dsm::Pattern::PairBidirectional));
  tests.push_back(mixed_config("mixed_load_store_same",
                               dsm::kMixedLoadStoreSameDirection));
  tests.push_back(mixed_config("mixed_load_tma_opposite",
                               dsm::kMixedLoadTmaOppositeDirection));
  tests.push_back(
      config("tma_scale_n16", Method::Tma, dsm::Pattern::Ring, 16, 1));
  auto topology =
      config("load_ring_n16_s1", Method::Load, dsm::Pattern::Ring, 16, 1);
  topology.active_warps = 16;
  tests.push_back(topology);
}

void append_bandwidth(std::vector<TestConfig> &tests) {
  constexpr int sizes_kib[] = {16, 24, 32, 40, 48, 56,
                               64, 72, 80, 88, 96};
  for (int kib : sizes_kib) {
    const int bytes = kib * 1024;
    for (auto pattern : {dsm::Pattern::PairUnidirectional,
                         dsm::Pattern::PairBidirectional}) {
      const bool bidirectional = pattern == dsm::Pattern::PairBidirectional;
      auto load = config(bidirectional ? "load_size_bi" : "load_size_uni",
                         Method::Load, pattern);
      load.epoch_bytes = bytes;
      tests.push_back(load);

      auto store =
          config(bidirectional ? "store_size_bi" : "store_size_uni",
                 Method::Store, pattern);
      store.epoch_bytes = bytes;
      tests.push_back(store);

      auto tma = config(bidirectional ? "tma_size_bi" : "tma_size_uni",
                        Method::Tma, pattern);
      tma.epoch_bytes = bytes;
      tests.push_back(tma);
    }

    const std::pair<const char *, int> mixed_cases[] = {
        {"mixed_load_store_same", dsm::kMixedLoadStoreSameDirection},
        {"mixed_load_store_opposite",
         dsm::kMixedLoadStoreOppositeDirection},
        {"mixed_load_tma_same", dsm::kMixedLoadTmaSameDirection},
        {"mixed_load_tma_opposite", dsm::kMixedLoadTmaOppositeDirection},
    };
    for (const auto &[label, kind] : mixed_cases) {
      auto mixed = mixed_config(label, kind);
      mixed.epoch_bytes = bytes;
      tests.push_back(mixed);
    }
  }

  // One source-SM SRAM control is enough: earlier sweeps showed no change from
  // 1 through 16 pressure warps in either bank-conflict-free or conflicting
  // mode.  Keep the strongest 16-warp cases and their no-pressure baseline.
  for (int pressure_mode : {0, 1, 2}) {
    const char *label = pressure_mode == 0   ? "load_pressure_none"
                        : pressure_mode == 1 ? "load_pressure_free"
                                             : "load_pressure_conflict";
    auto pressure = config(label, Method::Load,
                           dsm::Pattern::PairUnidirectional);
    pressure.active_warps = 16;
    pressure.pressure_mode = pressure_mode;
    pressure.pressure_warps = pressure_mode == 0 ? 0 : 16;
    tests.push_back(pressure);
  }

  // Every rank puts one tile to its ring neighbor.  Size slopes, rather than
  // raw 64-KiB throughput, remove the cluster-size-dependent completion cost.
  for (int cluster_size : {2, 4, 8, 16}) {
    std::ostringstream label;
    label << "tma_scale_n" << cluster_size;
    for (int kib : sizes_kib) {
      auto scaling = config(label.str(), Method::Tma, dsm::Pattern::Ring,
                            cluster_size, 1);
      scaling.epoch_bytes = kib * 1024;
      tests.push_back(scaling);
    }
  }
}

void append_topology(std::vector<TestConfig> &tests) {
  for (int cluster_size : {2, 4, 8, 16}) {
    for (int stride = 1; stride < cluster_size; ++stride) {
      std::ostringstream label;
      label << "load_ring_n" << cluster_size << "_s" << stride;
      auto value = config(label.str(), Method::Load, dsm::Pattern::Ring,
                          cluster_size, stride);
      value.active_warps = 16;
      tests.push_back(value);
    }
  }
  // Packed groups expose the unusually favorable adjacent two-SM pairing.
  // Group 16 is identical to the ring-16 sweep and is not duplicated.
  for (int group_size : {2, 4, 8}) {
    for (int stride = 1; stride < group_size; ++stride) {
      const int param = (group_size << 16) | stride;
      std::ostringstream label;
      label << "load_packed_g" << group_size << "_s" << stride;
      auto value = config(label.str(), Method::Load,
                          dsm::Pattern::PackedRing, 16, param);
      value.active_warps = 16;
      tests.push_back(value);
    }
  }
}

std::vector<TestConfig> make_tests(const std::string &suite) {
  std::vector<TestConfig> tests;
  if (suite == "smoke") {
    append_smoke(tests);
  } else if (suite == "bandwidth") {
    append_bandwidth(tests);
  } else if (suite == "topology") {
    append_topology(tests);
  } else {
    append_bandwidth(tests);
    append_topology(tests);
  }
  return tests;
}

cudaLaunchConfig_t make_launch_config(const TestConfig &test,
                                      cudaLaunchAttribute *attributes) {
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(test.cluster_size, 1, 1);
  launch.blockDim = dim3(test.threads, 1, 1);
  launch.dynamicSmemBytes = dsm::kDynamicSharedBytes;
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = test.cluster_size;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  attributes[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attributes[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;
  launch.attrs = attributes;
  launch.numAttrs = 2;
  return launch;
}

template <typename Kernel>
cudaError_t prepare_kernel(Kernel kernel) {
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dsm::kDynamicSharedBytes);
  if (error != cudaSuccess) {
    return error;
  }
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}

cudaError_t launch_test(const TestConfig &test, dsm::TimingResult *results,
                        bool occupancy_only, int *active_clusters) {
  cudaLaunchAttribute attributes[2]{};
  cudaLaunchConfig_t launch = make_launch_config(test, attributes);
  switch (test.method) {
  case Method::Load: {
    auto kernel = dsm::load_kernel;
    cudaError_t error = prepare_kernel(kernel);
    if (error != cudaSuccess)
      return error;
    if (occupancy_only)
      return cudaOccupancyMaxActiveClusters(active_clusters, kernel, &launch);
    return cudaLaunchKernelEx(&launch, kernel, results, test.pattern,
                              test.param, test.active_warps,
                              test.pressure_mode, test.pressure_warps,
                              test.epoch_bytes);
  }
  case Method::Tma: {
    auto kernel = dsm::tma_kernel;
    cudaError_t error = prepare_kernel(kernel);
    if (error != cudaSuccess)
      return error;
    if (occupancy_only)
      return cudaOccupancyMaxActiveClusters(active_clusters, kernel, &launch);
    return cudaLaunchKernelEx(&launch, kernel, results, test.pattern,
                              test.param, test.chunk_bytes, test.epoch_bytes);
  }
  case Method::Store: {
    auto kernel = dsm::store_kernel;
    cudaError_t error = prepare_kernel(kernel);
    if (error != cudaSuccess)
      return error;
    if (occupancy_only)
      return cudaOccupancyMaxActiveClusters(active_clusters, kernel, &launch);
    return cudaLaunchKernelEx(&launch, kernel, results, test.pattern,
                              test.param, test.active_warps,
                              test.epoch_bytes);
  }
  case Method::Mixed: {
    auto kernel = dsm::mixed_kernel;
    cudaError_t error = prepare_kernel(kernel);
    if (error != cudaSuccess)
      return error;
    if (occupancy_only)
      return cudaOccupancyMaxActiveClusters(active_clusters, kernel, &launch);
    return cudaLaunchKernelEx(&launch, kernel, results, test.mixed_kind,
                              test.epoch_bytes, test.active_warps,
                              test.chunk_bytes);
  }
  }
  return cudaErrorInvalidValue;
}

void emit_meta(std::ostream &out, const Options &options,
               const cudaDeviceProp &properties, int driver_version,
               int runtime_version, int max_shared_optin) {
  out << "{\"type\":\"meta\",\"suite\":\"" << options.suite
      << "\",\"gpu\":\"" << properties.name << "\",\"sm_count\":"
      << properties.multiProcessorCount << ",\"cc\":" << properties.major
      << "." << properties.minor << ",\"driver_version\":"
      << driver_version << ",\"runtime_version\":" << runtime_version
      << ",\"max_shared_optin\":" << max_shared_optin
      << ",\"max_tile_bytes\":" << dsm::kMaxTileBytes
      << ",\"dynamic_shared_bytes\":" << dsm::kDynamicSharedBytes
      << ",\"warmups\":" << options.warmups
      << ",\"iterations\":" << options.iterations << "}\n";
}

void emit_skip(std::ostream &out, const TestConfig &test,
               const std::string &reason) {
  out << "{\"type\":\"skip\",\"label\":\"" << test.label
      << "\",\"method\":\"" << method_name(test.method)
      << "\",\"cluster_size\":" << test.cluster_size << ",\"reason\":\""
      << reason << "\"}\n";
}

void emit_result(std::ostream &out, const TestConfig &test, int iteration,
                 int active_clusters, float event_ms,
                 const std::vector<dsm::TimingResult> &results) {
  unsigned long long first_ns = std::numeric_limits<unsigned long long>::max();
  unsigned long long last_ns = 0;
  unsigned long long max_cycles = 0;
  int active_count = 0;
  std::set<unsigned int> smids;
  for (const auto &result : results) {
    smids.insert(result.smid);
    if (!result.active)
      continue;
    ++active_count;
    first_ns = std::min(first_ns, result.ns_begin);
    last_ns = std::max(last_ns, result.ns_end);
    max_cycles = std::max(max_cycles, result.clock_end - result.clock_begin);
  }
  if (static_cast<int>(smids.size()) != test.cluster_size) {
    fail("cluster ranks did not map to unique SMs for " + test.label);
  }
  if (active_count == 0 || last_ns <= first_ns || max_cycles == 0) {
    fail("invalid timing interval for " + test.label);
  }

  const auto transfer_bytes = static_cast<unsigned long long>(test.epoch_bytes);
  const auto elapsed_ns = last_ns - first_ns;
  const double aggregate_gbps =
      static_cast<double>(active_count) * transfer_bytes / elapsed_ns;
  const double aggregate_bytes_per_cycle =
      static_cast<double>(active_count) * transfer_bytes / max_cycles;

  out << std::setprecision(10)
      << "{\"type\":\"result\",\"label\":\"" << test.label
      << "\",\"method\":\"" << method_name(test.method)
      << "\",\"pattern\":\"" << pattern_name(test.pattern)
      << "\",\"cluster_size\":" << test.cluster_size
      << ",\"param\":" << test.param << ",\"iteration\":" << iteration
      << ",\"active_clusters\":" << active_clusters
      << ",\"epoch_bytes_per_active_sm\":" << test.epoch_bytes
      << ",\"active_sms\":" << active_count
      << ",\"threads\":" << test.threads
      << ",\"active_warps\":" << test.active_warps
      << ",\"chunk_bytes\":" << test.chunk_bytes
      << ",\"pressure_mode\":" << test.pressure_mode
      << ",\"pressure_warps\":" << test.pressure_warps
      << ",\"mixed_kind\":" << test.mixed_kind
      << ",\"elapsed_ns\":" << elapsed_ns
      << ",\"max_cycles\":" << max_cycles
      << ",\"aggregate_bytes_per_cycle\":" << aggregate_bytes_per_cycle
      << ",\"aggregate_gbps\":" << aggregate_gbps
      << ",\"event_ms\":" << event_ms << ",\"smids\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    if (i)
      out << ',';
    out << results[i].smid;
  }
  out << "],\"ranks\":[";

  bool first = true;
  for (const auto &result : results) {
    if (!result.active)
      continue;
    unsigned long long expected = 0;
    if (test.method == Method::Store) {
      expected = expected_store_checksum(result.source_rank, test.epoch_bytes);
    } else if (test.method == Method::Mixed) {
      if (result.rank == 0 ||
          test.mixed_kind >= dsm::kMixedLoadTmaSameDirection) {
        expected = expected_checksum(result.source_rank, test.epoch_bytes);
      } else {
        expected =
            expected_store_checksum(result.source_rank, test.epoch_bytes);
      }
    } else {
      expected = expected_checksum(result.source_rank, test.epoch_bytes);
    }
    if (result.checksum != expected) {
      std::ostringstream message;
      message << "checksum mismatch for " << test.label << " rank "
              << result.rank << ": expected " << expected << ", got "
              << result.checksum;
      fail(message.str());
    }
    if (result.clock_end <= result.clock_begin ||
        result.ns_end <= result.ns_begin) {
      fail("non-monotonic timer for " + test.label);
    }
    if (!first)
      out << ',';
    first = false;
    const auto rank_cycles = result.clock_end - result.clock_begin;
    const auto rank_ns = result.ns_end - result.ns_begin;
    out << "{\"rank\":" << result.rank << ",\"smid\":" << result.smid
        << ",\"source_rank\":" << result.source_rank
        << ",\"target_rank\":" << result.target_rank
        << ",\"clock_begin\":" << result.clock_begin
        << ",\"clock_end\":" << result.clock_end
        << ",\"cycles\":" << rank_cycles << ",\"ns_begin\":"
        << result.ns_begin << ",\"ns_end\":" << result.ns_end
        << ",\"ns\":" << rank_ns << ",\"gbps\":"
        << static_cast<double>(transfer_bytes) / rank_ns
        << ",\"bytes_per_cycle\":"
        << static_cast<double>(transfer_bytes) / rank_cycles
        << ",\"checksum\":" << result.checksum << '}';
  }
  out << "]}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "get device properties");
    if (properties.major != 9) {
      fail("DSM benchmark requires a Hopper compute-capability 9.x GPU");
    }
    int driver_version = 0, runtime_version = 0, max_shared_optin = 0;
    check(cudaDriverGetVersion(&driver_version), "get driver version");
    check(cudaRuntimeGetVersion(&runtime_version), "get runtime version");
    check(cudaDeviceGetAttribute(&max_shared_optin,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, 0),
          "get opt-in shared memory limit");
    if (max_shared_optin < dsm::kDynamicSharedBytes) {
      fail("GPU does not provide the 200-KiB shared-memory allocation needed "
           "to enforce one CTA per SM");
    }

    std::ofstream output_file;
    std::ostream *out = &std::cout;
    if (!options.output.empty()) {
      output_file.open(options.output, std::ios::out | std::ios::trunc);
      if (!output_file)
        fail("cannot open output file: " + options.output);
      out = &output_file;
    }
    emit_meta(*out, options, properties, driver_version, runtime_version,
              max_shared_optin);

    dsm::TimingResult *device_results = nullptr;
    check(cudaMalloc(&device_results,
                     sizeof(dsm::TimingResult) * dsm::kMaxClusterRanks),
          "allocate results");
    std::vector<dsm::TimingResult> host_results(dsm::kMaxClusterRanks);
    const auto tests = make_tests(options.suite);
    std::cerr << "Running " << tests.size() << " DSM configurations on "
              << properties.name << '\n';

    for (size_t test_index = 0; test_index < tests.size(); ++test_index) {
      const auto &test = tests[test_index];
      if (!options.filter.empty() && test.label != options.filter) continue;
      if (options.bytes > 0 && test.epoch_bytes != options.bytes) continue;
      if (test.threads != dsm::kThreads || test.active_warps <= 0 ||
          test.active_warps * 32 > test.threads || test.epoch_bytes <= 0 ||
          test.epoch_bytes > dsm::kMaxTileBytes ||
          test.epoch_bytes % 16 != 0 || test.chunk_bytes <= 0 ||
          test.chunk_bytes > 16 * 1024 || test.chunk_bytes % 16 != 0) {
        fail("invalid configuration for " + test.label);
      }
      int active_clusters = 0;
      cudaError_t occupancy_error =
          launch_test(test, device_results, true, &active_clusters);
      if (occupancy_error != cudaSuccess || active_clusters < 1) {
        const std::string reason = cudaGetErrorString(occupancy_error);
        cudaGetLastError();
        emit_skip(*out, test, reason);
        continue;
      }

      for (int warmup = 0; warmup < options.warmups; ++warmup) {
        check(cudaMemset(device_results, 0,
                         sizeof(dsm::TimingResult) * dsm::kMaxClusterRanks),
              "clear warmup results");
        check(launch_test(test, device_results, false, &active_clusters),
              "warmup launch");
        check(cudaDeviceSynchronize(), "warmup synchronize");
      }

      cudaEvent_t begin = nullptr, end = nullptr;
      check(cudaEventCreate(&begin), "create begin event");
      check(cudaEventCreate(&end), "create end event");
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        check(cudaMemset(device_results, 0,
                         sizeof(dsm::TimingResult) * dsm::kMaxClusterRanks),
              "clear results");
        check(cudaEventRecord(begin), "record begin event");
        check(launch_test(test, device_results, false, &active_clusters),
              "timed launch");
        check(cudaEventRecord(end), "record end event");
        check(cudaEventSynchronize(end), "timed synchronize");
        float event_ms = 0.0f;
        check(cudaEventElapsedTime(&event_ms, begin, end),
              "event elapsed time");
        check(cudaMemcpy(host_results.data(), device_results,
                         sizeof(dsm::TimingResult) * test.cluster_size,
                         cudaMemcpyDeviceToHost),
              "copy results");
        host_results.resize(test.cluster_size);
        emit_result(*out, test, iteration, active_clusters, event_ms,
                    host_results);
        host_results.resize(dsm::kMaxClusterRanks);
      }
      check(cudaEventDestroy(begin), "destroy begin event");
      check(cudaEventDestroy(end), "destroy end event");
      if ((test_index + 1) % 25 == 0 || test_index + 1 == tests.size()) {
        std::cerr << "Completed " << (test_index + 1) << '/' << tests.size()
                  << " configurations\n";
      }
    }
    check(cudaFree(device_results), "free results");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DSM benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
