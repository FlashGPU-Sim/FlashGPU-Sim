#ifndef FLASHGPU_TEST_MBARRIER_BENCHMARK_HARNESS_CUH_
#define FLASHGPU_TEST_MBARRIER_BENCHMARK_HARNESS_CUH_

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <utility>
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

static __global__ void mbarrier_measure_clock_overhead_kernel(
    uint64_t* cycle_start, uint64_t* cycle_end) {
  if (threadIdx.x == 0) {
    const uint64_t start = clock64();
    const uint64_t end = clock64();
    cycle_start[0] = start;
    cycle_end[0] = end;
  }
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

inline void print_mbarrier_sweep(const char* test_name, const char* description,
                                 const MBarrierSweepSummary& summary) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf(
      "┌──────────────┬──────────────┬──────────────┬──────────────┬───────────"
      "───┐\n");
  printf(
      "│ Instr Count  │     Min      │    Median    │     P90      │     Max   "
      "   │\n");
  printf(
      "├──────────────┼──────────────┼──────────────┼──────────────┼───────────"
      "───┤\n");
  for (const auto& point : summary.points) {
    printf("│ %10u   │ %10.2f   │ %10.2f   │ %10.2f   │ %10.2f   │\n",
           point.instruction_count, point.cycles.min, point.cycles.median,
           point.cycles.p90, point.cycles.max);
  }
  printf(
      "└──────────────┴──────────────┴──────────────┴──────────────┴───────────"
      "───┘\n");
  printf(
      "Regression: slope=%.2f cycles, intercept=%.2f, r2=%.4f, clock "
      "overhead=%lu cycles\n",
      summary.fit.slope, summary.fit.intercept, summary.fit.r_squared,
      static_cast<unsigned long>(summary.clock_overhead));
}

inline void print_mbarrier_scalar(const char* test_name,
                                  const char* description,
                                  const MBarrierCycleSummary& summary,
                                  uint64_t clock_overhead) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf(
      "┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n");
  printf(
      "│  Metric  │   Min    │   P10    │  Median  │   P90    │   Max    │\n");
  printf(
      "├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n");
  printf("│ Cycles   │ %8.2f │ %8.2f │ %8.2f │ %8.2f │ %8.2f │\n", summary.min,
         summary.p10, summary.median, summary.p90, summary.max);
  printf(
      "└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n");
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

inline void export_mbarrier_scalar_csv(
    const char* filename, const char* test_name, const char* description,
    const MBarrierCycleSummary& summary, const MBarrierBenchOptions& options,
    uint64_t clock_overhead, const char* notes = "") {
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

  template <typename LaunchForCountFn>
  MBarrierSweepSummary measure_sweep(
      LaunchForCountFn&& launch_for_count,
      const std::vector<uint32_t>& instruction_counts,
      const MBarrierBenchOptions& options, uint64_t clock_overhead) {
    MBarrierSweepSummary summary;
    summary.clock_overhead = clock_overhead;

    std::vector<double> medians;
    medians.reserve(instruction_counts.size());
    summary.points.reserve(instruction_counts.size());

    for (const uint32_t instruction_count : instruction_counts) {
      const MBarrierCycleSummary cycles =
          measure_summary([&] { launch_for_count(instruction_count); }, options,
                          clock_overhead);
      summary.points.push_back({instruction_count, cycles});
      medians.push_back(cycles.median);
    }

    summary.fit = mbarrier_linear_regression(instruction_counts, medians);
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

#endif  // FLASHGPU_TEST_MBARRIER_BENCHMARK_HARNESS_CUH_
