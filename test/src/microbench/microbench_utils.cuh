#ifndef MICROBENCH_UTILS_CUH
#define MICROBENCH_UTILS_CUH

#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

// Statistical Analysis Results
struct BenchmarkStats {
  double mean;
  double median;
  double std_dev;
  double p95;
  double p99;
  double min;
  double max;
  int num_samples;
  int num_outliers_removed;

  BenchmarkStats()
      : mean(0),
        median(0),
        std_dev(0),
        p95(0),
        p99(0),
        min(0),
        max(0),
        num_samples(0),
        num_outliers_removed(0) {}
};

// Remove outliers using IQR method
inline std::vector<uint64_t> remove_outliers_iqr(
    const std::vector<uint64_t>& data, double multiplier = 1.5) {
  if (data.size() < 4) return data;

  std::vector<uint64_t> sorted = data;
  std::sort(sorted.begin(), sorted.end());

  size_t q1_idx = sorted.size() / 4;
  size_t q3_idx = (sorted.size() * 3) / 4;

  uint64_t q1 = sorted[q1_idx];
  uint64_t q3 = sorted[q3_idx];
  uint64_t iqr = q3 - q1;

  double lower_bound = q1 - multiplier * iqr;
  double upper_bound = q3 + multiplier * iqr;

  std::vector<uint64_t> filtered;
  for (uint64_t val : data) {
    if (static_cast<double>(val) >= lower_bound &&
        static_cast<double>(val) <= upper_bound) {
      filtered.push_back(val);
    }
  }
  return filtered;
}

// Calculate statistics from raw cycle data
inline BenchmarkStats calculate_stats(const std::vector<uint64_t>& raw_data,
                                      bool remove_outliers = true) {
  BenchmarkStats stats;
  stats.num_samples = raw_data.size();

  if (raw_data.empty()) return stats;

  std::vector<uint64_t> data;
  if (remove_outliers) {
    data = remove_outliers_iqr(raw_data);
    stats.num_outliers_removed = raw_data.size() - data.size();
  } else {
    data = raw_data;
    stats.num_outliers_removed = 0;
  }

  if (data.empty()) return stats;

  // Sort for percentiles and median
  std::sort(data.begin(), data.end());

  stats.min = static_cast<double>(data.front());
  stats.max = static_cast<double>(data.back());
  stats.median = static_cast<double>(data[data.size() / 2]);
  stats.p95 =
      static_cast<double>(data[static_cast<size_t>(data.size() * 0.95)]);
  stats.p99 =
      static_cast<double>(data[static_cast<size_t>(data.size() * 0.99)]);

  // Mean and Std Dev
  double sum = 0.0;
  for (uint64_t val : data) sum += val;
  stats.mean = sum / data.size();

  double sq_sum = 0.0;
  for (uint64_t val : data) {
    double diff = val - stats.mean;
    sq_sum += diff * diff;
  }
  stats.std_dev = std::sqrt(sq_sum / data.size());

  return stats;
}

#endif  // MICROBENCH_UTILS_CUH
