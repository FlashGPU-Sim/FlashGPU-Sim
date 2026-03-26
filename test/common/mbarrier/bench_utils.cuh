#ifndef TEST_COMMON_MBARRIER_BENCH_UTILS_CUH
#define TEST_COMMON_MBARRIER_BENCH_UTILS_CUH

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct MBarrierBenchOptions {
  int warmup_iterations = 5;
  int measured_iterations = 51;
  bool subtract_clock_overhead = true;
};

struct MBarrierCycleSummary {
  double min = 0.0;
  double p10 = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double max = 0.0;
  int sample_count = 0;
};

struct MBarrierComparisonSummary {
  MBarrierCycleSummary raw;
  MBarrierCycleSummary control;
  double derived_median = 0.0;
  double derived_p10 = 0.0;
  double derived_p90 = 0.0;
  uint64_t clock_overhead = 0;
};

struct MBarrierSweepPointSummary {
  uint32_t instruction_count = 0;
  MBarrierCycleSummary cycles;
};

struct MBarrierLinearFit {
  double slope = 0.0;
  double intercept = 0.0;
  double r_squared = 0.0;
};

struct MBarrierSweepSummary {
  std::vector<MBarrierSweepPointSummary> points;
  MBarrierLinearFit fit;
  uint64_t clock_overhead = 0;
};

struct MBarrierThresholdPointSummary {
  uint32_t sweep_value = 0;
  int hits = 0;
  int trials = 0;
  double hit_rate = 0.0;
};

struct MBarrierThresholdSummary {
  std::vector<MBarrierThresholdPointSummary> points;
  double filler_step_cycles = 0.0;
  int first_any_hit = -1;
  int first_all_hit = -1;
  double approx_first_any_cycles = -1.0;
  double approx_first_all_cycles = -1.0;
  uint64_t clock_overhead = 0;
};

static __global__ void mbarrier_measure_clock_overhead_kernel(
    uint64_t* cycle_start, uint64_t* cycle_end) {
  if (threadIdx.x == 0) {
    const uint64_t start = clock64();
    const uint64_t end = clock64();
    cycle_start[0] = start;
    cycle_end[0] = end;
  }
}

inline bool mbarrier_running_in_native_mode() {
  const char* sim_env = std::getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN");
  if (sim_env != nullptr && sim_env[0] != '\0') {
    return false;
  }

  const char* ld_library_path = std::getenv("LD_LIBRARY_PATH");
  if (ld_library_path == nullptr || ld_library_path[0] == '\0') {
    return true;
  }

  const std::string ld_path(ld_library_path);
  const char* gpgpusim_root = std::getenv("GPGPUSIM_ROOT");
  if (gpgpusim_root != nullptr && gpgpusim_root[0] != '\0') {
    const std::string sim_lib_prefix = std::string(gpgpusim_root) + "/lib/";
    if (ld_path.find(sim_lib_prefix) != std::string::npos) {
      return false;
    }
  }

  return ld_path.find("gpgpu-sim") == std::string::npos;
}

inline double mbarrier_percentile(const std::vector<uint64_t>& sorted_samples,
                                  double quantile) {
  if (sorted_samples.empty()) {
    return 0.0;
  }

  const double clamped = std::clamp(quantile, 0.0, 1.0);
  const size_t index = static_cast<size_t>(
      clamped * static_cast<double>(sorted_samples.size() - 1));
  return static_cast<double>(sorted_samples[index]);
}

inline double mbarrier_percentile(const std::vector<double>& sorted_samples,
                                  double quantile) {
  if (sorted_samples.empty()) {
    return 0.0;
  }

  const double clamped = std::clamp(quantile, 0.0, 1.0);
  const size_t index = static_cast<size_t>(
      clamped * static_cast<double>(sorted_samples.size() - 1));
  return sorted_samples[index];
}

inline MBarrierCycleSummary summarize_mbarrier_cycles(
    const std::vector<uint64_t>& samples) {
  MBarrierCycleSummary summary;
  summary.sample_count = static_cast<int>(samples.size());
  if (samples.empty()) {
    return summary;
  }

  std::vector<uint64_t> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end());

  summary.min = static_cast<double>(sorted_samples.front());
  summary.p10 = mbarrier_percentile(sorted_samples, 0.10);
  summary.median = mbarrier_percentile(sorted_samples, 0.50);
  summary.p90 = mbarrier_percentile(sorted_samples, 0.90);
  summary.max = static_cast<double>(sorted_samples.back());
  return summary;
}

inline void print_mbarrier_comparison(const char* test_name,
                                      const char* description,
                                      const MBarrierComparisonSummary& summary) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
  printf("│  Metric  │   Min    │   P10    │  Median  │   P90    │   Max    │\n");
  printf("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
  printf("│ Raw      │ %8.2f │ %8.2f │ %8.2f │ %8.2f │ %8.2f │\n",
         summary.raw.min, summary.raw.p10, summary.raw.median,
         summary.raw.p90, summary.raw.max);
  printf("│ Control  │ %8.2f │ %8.2f │ %8.2f │ %8.2f │ %8.2f │\n",
         summary.control.min, summary.control.p10, summary.control.median,
         summary.control.p90, summary.control.max);
  printf("│ Derived  │    n/a   │ %8.2f │ %8.2f │ %8.2f │    n/a   │\n",
         summary.derived_p10, summary.derived_median, summary.derived_p90);
  printf("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n");
  printf("Samples: %d, clock overhead: %lu cycles\n", summary.raw.sample_count,
         static_cast<unsigned long>(summary.clock_overhead));
}

inline MBarrierLinearFit mbarrier_linear_regression(
    const std::vector<uint32_t>& x, const std::vector<double>& y) {
  MBarrierLinearFit fit;
  if (x.size() != y.size() || x.size() < 2) {
    return fit;
  }

  const double n = static_cast<double>(x.size());
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xy = 0.0;
  double sum_xx = 0.0;

  for (size_t i = 0; i < x.size(); ++i) {
    const double xd = static_cast<double>(x[i]);
    sum_x += xd;
    sum_y += y[i];
    sum_xy += xd * y[i];
    sum_xx += xd * xd;
  }

  const double denom = n * sum_xx - sum_x * sum_x;
  if (denom == 0.0) {
    return fit;
  }

  fit.slope = (n * sum_xy - sum_x * sum_y) / denom;
  fit.intercept = (sum_y - fit.slope * sum_x) / n;

  const double mean_y = sum_y / n;
  double ss_tot = 0.0;
  double ss_res = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double predicted =
        fit.slope * static_cast<double>(x[i]) + fit.intercept;
    const double residual = y[i] - predicted;
    const double centered = y[i] - mean_y;
    ss_res += residual * residual;
    ss_tot += centered * centered;
  }

  fit.r_squared = ss_tot == 0.0 ? 1.0 : 1.0 - (ss_res / ss_tot);
  return fit;
}

inline void print_mbarrier_sweep(const char* test_name,
                                 const char* description,
                                 const MBarrierSweepSummary& summary) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf("┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n");
  printf("│ Instr Count  │     Min      │    Median    │     P90      │     Max      │\n");
  printf("├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n");
  for (const auto& point : summary.points) {
    printf("│ %10u   │ %10.2f   │ %10.2f   │ %10.2f   │ %10.2f   │\n",
           point.instruction_count, point.cycles.min, point.cycles.median,
           point.cycles.p90, point.cycles.max);
  }
  printf("└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n");
  printf("Regression: slope=%.2f cycles, intercept=%.2f, r2=%.4f, clock overhead=%lu cycles\n",
         summary.fit.slope, summary.fit.intercept, summary.fit.r_squared,
         static_cast<unsigned long>(summary.clock_overhead));
}

inline void print_mbarrier_threshold(const char* test_name,
                                     const char* description,
                                     const MBarrierThresholdSummary& summary) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf("┌──────────────┬──────────┬──────────┬────────────┐\n");
  printf("│ Sweep Value  │   Hits   │  Trials  │  Hit Rate  │\n");
  printf("├──────────────┼──────────┼──────────┼────────────┤\n");
  for (const auto& point : summary.points) {
    printf("│ %10u   │ %6d   │ %6d   │ %8.3f   │\n", point.sweep_value,
           point.hits, point.trials, point.hit_rate);
  }
  printf("└──────────────┴──────────┴──────────┴────────────┘\n");
  printf("Filler step: %.2f cycles, first_any: %d, first_all: %d, approx cycles: %.2f / %.2f, clock overhead: %lu cycles\n",
         summary.filler_step_cycles, summary.first_any_hit,
         summary.first_all_hit, summary.approx_first_any_cycles,
         summary.approx_first_all_cycles,
         static_cast<unsigned long>(summary.clock_overhead));
}

inline void print_mbarrier_scalar(const char* test_name,
                                  const char* description,
                                  const MBarrierCycleSummary& summary,
                                  uint64_t clock_overhead) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
  printf("│  Metric  │   Min    │   P10    │  Median  │   P90    │   Max    │\n");
  printf("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
  printf("│ Cycles   │ %8.2f │ %8.2f │ %8.2f │ %8.2f │ %8.2f │\n",
         summary.min, summary.p10, summary.median, summary.p90, summary.max);
  printf("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n");
  printf("Samples: %d, clock overhead: %lu cycles\n", summary.sample_count,
         static_cast<unsigned long>(clock_overhead));
}

inline void export_mbarrier_csv_header(std::ofstream& out) {
  out << "test_name,description,row_type,instruction_count,sample_count,"
         "warmup_iterations,measured_iterations,clock_overhead,min,p10,"
         "median,p90,max,slope,intercept,r_squared,delta,notes\n";
}

inline void export_mbarrier_sweep_csv(const char* filename,
                                      const char* test_name,
                                      const char* description,
                                      const MBarrierSweepSummary& summary,
                                      const MBarrierBenchOptions& options) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    ADD_FAILURE() << "Failed to open CSV output file: " << filename;
    return;
  }

  export_mbarrier_csv_header(out);
  for (const auto& point : summary.points) {
    out << test_name << ',' << description << ",point,"
        << point.instruction_count << ',' << point.cycles.sample_count << ','
        << options.warmup_iterations << ',' << options.measured_iterations
        << ',' << summary.clock_overhead << ',' << point.cycles.min << ','
        << point.cycles.p10 << ',' << point.cycles.median << ','
        << point.cycles.p90 << ',' << point.cycles.max << ','
        << summary.fit.slope << ',' << summary.fit.intercept << ','
        << summary.fit.r_squared << ",,\n";
  }
}

inline void export_mbarrier_scalar_csv(const char* filename,
                                       const char* test_name,
                                       const char* description,
                                       const MBarrierCycleSummary& summary,
                                       const MBarrierBenchOptions& options,
                                       uint64_t clock_overhead,
                                       const char* notes = "") {
  std::ofstream out(filename);
  if (!out.is_open()) {
    ADD_FAILURE() << "Failed to open CSV output file: " << filename;
    return;
  }

  export_mbarrier_csv_header(out);
  out << test_name << ',' << description << ",summary,-1,"
      << summary.sample_count << ',' << options.warmup_iterations << ','
      << options.measured_iterations << ',' << clock_overhead << ','
      << summary.min << ',' << summary.p10 << ',' << summary.median << ','
      << summary.p90 << ',' << summary.max << ",,,,," << notes << '\n';
}

inline void export_mbarrier_threshold_csv(
    const char* filename, const char* test_name, const char* description,
    const MBarrierThresholdSummary& summary,
    const MBarrierBenchOptions& options) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    ADD_FAILURE() << "Failed to open CSV output file: " << filename;
    return;
  }

  out << "test_name,description,sweep_value,hits,trials,hit_rate,"
         "warmup_iterations,measured_iterations,clock_overhead,"
         "filler_step_cycles,first_any_hit,first_all_hit,"
         "approx_first_any_cycles,approx_first_all_cycles\n";
  for (const auto& point : summary.points) {
    out << test_name << ',' << description << ',' << point.sweep_value << ','
        << point.hits << ',' << point.trials << ',' << point.hit_rate << ','
        << options.warmup_iterations << ',' << options.measured_iterations
        << ',' << summary.clock_overhead << ',' << summary.filler_step_cycles
        << ',' << summary.first_any_hit << ',' << summary.first_all_hit << ','
        << summary.approx_first_any_cycles << ','
        << summary.approx_first_all_cycles << '\n';
  }
}

class MBarrierBenchmarkFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaError_t err = cudaMalloc(&d_cycle_start_, sizeof(uint64_t));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate cycle_start buffer";

    err = cudaMalloc(&d_cycle_end_, sizeof(uint64_t));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate cycle_end buffer";
  }

  void TearDown() override {
    if (d_cycle_start_ != nullptr) {
      cudaFree(d_cycle_start_);
      d_cycle_start_ = nullptr;
    }
    if (d_cycle_end_ != nullptr) {
      cudaFree(d_cycle_end_);
      d_cycle_end_ = nullptr;
    }
  }

  template <typename LaunchFn>
  std::vector<uint64_t> collect_samples(LaunchFn&& launch,
                                        const MBarrierBenchOptions& options,
                                        uint64_t clock_overhead) {
    for (int i = 0; i < options.warmup_iterations; ++i) {
      launch();
      const cudaError_t warmup_err = cudaDeviceSynchronize();
      EXPECT_EQ(warmup_err, cudaSuccess)
          << "Warmup kernel failed: " << cudaGetErrorString(warmup_err);
      if (warmup_err != cudaSuccess) {
        return {};
      }
    }

    std::vector<uint64_t> samples;
    samples.reserve(options.measured_iterations);

    for (int i = 0; i < options.measured_iterations; ++i) {
      launch();

      const cudaError_t sync_err = cudaDeviceSynchronize();
      EXPECT_EQ(sync_err, cudaSuccess)
          << "Measured kernel failed: " << cudaGetErrorString(sync_err);
      if (sync_err != cudaSuccess) {
        return samples;
      }

      const cudaError_t start_copy_err =
          cudaMemcpy(&h_cycle_start_, d_cycle_start_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(start_copy_err, cudaSuccess)
          << "Failed to copy cycle_start from device";
      if (start_copy_err != cudaSuccess) {
        return samples;
      }

      const cudaError_t end_copy_err =
          cudaMemcpy(&h_cycle_end_, d_cycle_end_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(end_copy_err, cudaSuccess)
          << "Failed to copy cycle_end from device";
      if (end_copy_err != cudaSuccess) {
        return samples;
      }

      uint64_t elapsed = h_cycle_end_ - h_cycle_start_;
      if (options.subtract_clock_overhead && elapsed > clock_overhead) {
        elapsed -= clock_overhead;
      }
      samples.push_back(elapsed);
    }

    return samples;
  }

  template <typename LaunchFn>
  MBarrierCycleSummary measure_summary(LaunchFn&& launch,
                                       const MBarrierBenchOptions& options,
                                       uint64_t clock_overhead) {
    return summarize_mbarrier_cycles(collect_samples(
        std::forward<LaunchFn>(launch), options, clock_overhead));
  }

  template <typename RawLaunchFn, typename ControlLaunchFn>
  MBarrierComparisonSummary measure_with_control(
      RawLaunchFn&& raw_launch, ControlLaunchFn&& control_launch,
      const MBarrierBenchOptions& options, uint64_t clock_overhead,
      uint32_t normalization_factor, const char* test_name,
      const char* description) {
    MBarrierComparisonSummary summary;
    summary.clock_overhead = clock_overhead;

    auto measure_once = [&](auto&& launch, uint64_t& elapsed) -> bool {
      launch();

      const cudaError_t sync_err = cudaDeviceSynchronize();
      EXPECT_EQ(sync_err, cudaSuccess)
          << "Measured kernel failed: " << cudaGetErrorString(sync_err);
      if (sync_err != cudaSuccess) {
        return false;
      }

      const cudaError_t start_copy_err =
          cudaMemcpy(&h_cycle_start_, d_cycle_start_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(start_copy_err, cudaSuccess)
          << "Failed to copy cycle_start from device";
      if (start_copy_err != cudaSuccess) {
        return false;
      }

      const cudaError_t end_copy_err =
          cudaMemcpy(&h_cycle_end_, d_cycle_end_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(end_copy_err, cudaSuccess)
          << "Failed to copy cycle_end from device";
      if (end_copy_err != cudaSuccess) {
        return false;
      }

      elapsed = h_cycle_end_ - h_cycle_start_;
      if (options.subtract_clock_overhead && elapsed > clock_overhead) {
        elapsed -= clock_overhead;
      }
      return true;
    };

    for (int i = 0; i < options.warmup_iterations; ++i) {
      uint64_t elapsed = 0;
      if (!measure_once(raw_launch, elapsed) ||
          !measure_once(control_launch, elapsed)) {
        return summary;
      }
    }

    std::vector<uint64_t> raw_samples;
    std::vector<uint64_t> control_samples;
    std::vector<double> derived_samples;
    raw_samples.reserve(options.measured_iterations);
    control_samples.reserve(options.measured_iterations);
    derived_samples.reserve(options.measured_iterations);

    for (int i = 0; i < options.measured_iterations; ++i) {
      uint64_t raw_elapsed = 0;
      uint64_t control_elapsed = 0;
      if (!measure_once(raw_launch, raw_elapsed) ||
          !measure_once(control_launch, control_elapsed)) {
        break;
      }

      raw_samples.push_back(raw_elapsed);
      control_samples.push_back(control_elapsed);
      derived_samples.push_back(static_cast<double>(raw_elapsed) -
                                static_cast<double>(control_elapsed));
    }

    summary.raw = summarize_mbarrier_cycles(raw_samples);
    summary.control = summarize_mbarrier_cycles(control_samples);

    const double divisor =
        normalization_factor == 0 ? 1.0 : static_cast<double>(normalization_factor);
    if (!derived_samples.empty()) {
      std::sort(derived_samples.begin(), derived_samples.end());
      summary.derived_median =
          mbarrier_percentile(derived_samples, 0.50) / divisor;
      summary.derived_p10 =
          mbarrier_percentile(derived_samples, 0.10) / divisor;
      summary.derived_p90 =
          mbarrier_percentile(derived_samples, 0.90) / divisor;
    }

    print_mbarrier_comparison(test_name, description, summary);
    return summary;
  }

  template <typename LaunchForCountFn>
  MBarrierSweepSummary measure_with_sweep(
      LaunchForCountFn&& launch_for_count,
      const std::vector<uint32_t>& instruction_counts,
      const MBarrierBenchOptions& options, uint64_t clock_overhead,
      const char* test_name, const char* description) {
    MBarrierSweepSummary summary;
    summary.clock_overhead = clock_overhead;

    std::vector<double> medians;
    medians.reserve(instruction_counts.size());
    summary.points.reserve(instruction_counts.size());

    for (const uint32_t instruction_count : instruction_counts) {
      const MBarrierCycleSummary cycles = measure_summary(
          [&] { launch_for_count(instruction_count); }, options,
          clock_overhead);
      summary.points.push_back({instruction_count, cycles});
      medians.push_back(cycles.median);
    }

    summary.fit = mbarrier_linear_regression(instruction_counts, medians);
    print_mbarrier_sweep(test_name, description, summary);
    return summary;
  }

  uint64_t measure_clock_overhead(int iterations = 31) {
    std::vector<uint64_t> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
      mbarrier_measure_clock_overhead_kernel<<<1, 1>>>(d_cycle_start_,
                                                       d_cycle_end_);
      const cudaError_t sync_err = cudaDeviceSynchronize();
      EXPECT_EQ(sync_err, cudaSuccess)
          << "Clock overhead kernel failed: " << cudaGetErrorString(sync_err);
      if (sync_err != cudaSuccess) {
        return 0;
      }

      const cudaError_t start_copy_err =
          cudaMemcpy(&h_cycle_start_, d_cycle_start_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(start_copy_err, cudaSuccess)
          << "Failed to copy cycle_start from device";
      if (start_copy_err != cudaSuccess) {
        return 0;
      }

      const cudaError_t end_copy_err =
          cudaMemcpy(&h_cycle_end_, d_cycle_end_, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(end_copy_err, cudaSuccess)
          << "Failed to copy cycle_end from device";
      if (end_copy_err != cudaSuccess) {
        return 0;
      }

      samples.push_back(h_cycle_end_ - h_cycle_start_);
    }

    return static_cast<uint64_t>(summarize_mbarrier_cycles(samples).median);
  }

  uint64_t* cycle_start_ptr() { return d_cycle_start_; }
  uint64_t* cycle_end_ptr() { return d_cycle_end_; }

 private:
  uint64_t* d_cycle_start_ = nullptr;
  uint64_t* d_cycle_end_ = nullptr;
  uint64_t h_cycle_start_ = 0;
  uint64_t h_cycle_end_ = 0;
};

#endif  // TEST_COMMON_MBARRIER_BENCH_UTILS_CUH
