#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "common/mbarrier/bench_utils.cuh"
#include "common/mbarrier/device_kernels.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 32;
constexpr int kContentionThreads = kWarpSize * kMaxWarps;
constexpr uint32_t kStrideScanIterations = 1024;
constexpr uint32_t kContentionIterations = 1024;
constexpr uint32_t kStrideCandidates[] = {8,   16,  24,  32,  64,
                                          96,  128, 160, 192, 256};
constexpr uint32_t kStrideScanWarps[] = {2, 4, 8, 16, 32};
constexpr uint32_t kContentionWarps[] = {1, 2, 4, 8, 16, 32};
constexpr int kStrideScanBatches = 3;
constexpr int kMaxStrideBytes = 256;
constexpr int kSharedPaddingBytes = 64;
constexpr int kSharedRegionBytes =
    kMaxWarps * kMaxStrideBytes + kSharedPaddingBytes;

struct NormalizedComparisonSummary {
  MBarrierCycleSummary raw;
  MBarrierCycleSummary control;
  double derived_median = 0.0;
  double derived_p10 = 0.0;
  double derived_p90 = 0.0;
  uint64_t clock_overhead = 0;
};

struct StrideScanPoint {
  uint32_t stride_bytes = 0;
  uint32_t warp_count = 0;
  MBarrierCycleSummary raw;
};

struct StrideBatchSummary {
  std::vector<StrideScanPoint> points;
  uint32_t high_conflict_stride = 0;
  uint32_t low_conflict_stride = 0;
};

struct StrideLayoutSelection {
  std::vector<StrideBatchSummary> batches;
  uint32_t high_conflict_stride = 0;
  uint32_t low_conflict_stride = 0;
  double high_score = 0.0;
  double low_score = 0.0;
  double relative_gap_percent = 0.0;

  bool stable() const { return relative_gap_percent >= 1.0; }
  bool distinct() const {
    return high_conflict_stride != 0 && low_conflict_stride != 0 &&
           high_conflict_stride != low_conflict_stride;
  }
};

__device__ __forceinline__ uint32_t try_wait_false_control(uint32_t addr) {
  uint32_t ready = 0;
  asm volatile("{\n"
               ".reg .u64 value;\n"
               ".reg .pred p;\n"
               "ld.shared.u64 value, [%1];\n"
               "setp.ne.u64 p, value, value;\n"
               "selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(ready)
               : "r"(addr));
  return ready;
}

template <typename T>
__device__ __forceinline__ T* align_ptr(T* ptr, uintptr_t alignment) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<T*>((addr + alignment - 1) & ~(alignment - 1));
}

template <bool UseTryWait>
__global__ void contention_probe_kernel(uint32_t active_warps,
                                        uint32_t stride_bytes,
                                        uint32_t iterations,
                                        uint64_t* cycle_start,
                                        uint64_t* cycle_end,
                                        uint32_t* result_out) {
  __shared__ __align__(8) uint8_t storage[kSharedRegionBytes];
  __shared__ uint32_t ready_count_storage;
  __shared__ uint32_t start_flag_storage;
  __shared__ uint32_t done_count_storage;
  __shared__ uint32_t sink_storage;

  volatile uint32_t* ready_count = &ready_count_storage;
  volatile uint32_t* start_flag = &start_flag_storage;
  volatile uint32_t* done_count = &done_count_storage;

  uint64_t* base = align_ptr(reinterpret_cast<uint64_t*>(storage), 8);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;

  if (tid == 0) {
    ready_count_storage = 0;
    start_flag_storage = 0;
    done_count_storage = 0;
    sink_storage = 0;
    result_out[0] = 0;
    result_out[1] = 0;
  }
  __syncthreads();

  if (lane == 0 && warp < static_cast<int>(active_warps)) {
    uint64_t* slot = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uint8_t*>(base) + warp * stride_bytes);
    if constexpr (UseTryWait) {
      mbarrier_init(slot, 2);
      mbarrier_arrive(slot);
    } else {
      *slot = 0xfeedface12345678ull ^ static_cast<uint64_t>(warp + 3);
    }
  }
  __syncthreads();

  if (lane == 0 && warp < static_cast<int>(active_warps)) {
    uint64_t* slot = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uint8_t*>(base) + warp * stride_bytes);
    const uint32_t slot_addr = smem_u32_addr(slot);
    atomicAdd(&ready_count_storage, 1u);
    if (warp == 0) {
      while (*ready_count < active_warps) {
      }
      asm volatile("membar.cta;" ::: "memory");
      cycle_start[0] = clock64();
      *start_flag = 1;
    } else {
      while (*start_flag == 0) {
      }
    }

    uint32_t sink = static_cast<uint32_t>(warp);
    #pragma unroll 1
    for (uint32_t i = 0; i < iterations; ++i) {
      const uint32_t ready =
          UseTryWait ? (mbarrier_try_wait_parity(slot, 0) ? 1u : 0u)
                     : try_wait_false_control(slot_addr);
      sink ^= ready + i;
    }

    atomicAdd(&sink_storage, sink);
    atomicAdd(&done_count_storage, 1u);
    if (warp == 0) {
      while (*done_count < active_warps) {
      }
      asm volatile("membar.cta;" ::: "memory");
      cycle_end[0] = clock64();
      result_out[0] = sink_storage;
      result_out[1] = active_warps;
    }
  }
}

MBarrierCycleSummary normalize_summary(const MBarrierCycleSummary& summary,
                                       uint32_t normalization_factor) {
  MBarrierCycleSummary normalized = summary;
  const double divisor = normalization_factor == 0
                             ? 1.0
                             : static_cast<double>(normalization_factor);
  normalized.min /= divisor;
  normalized.p10 /= divisor;
  normalized.median /= divisor;
  normalized.p90 /= divisor;
  normalized.max /= divisor;
  return normalized;
}

void print_stride_scan_point(uint32_t stride_bytes, uint32_t warp_count,
                             const MBarrierCycleSummary& raw) {
  printf("│ %6u │ %5u │ %7d │ %9.2f │ %9.2f │ %9.2f │ %9.2f │ %9.2f │\n",
         stride_bytes, warp_count, raw.sample_count, raw.min, raw.p10,
         raw.median, raw.p90, raw.max);
}

void print_stride_batch_summary(int batch_index,
                                const StrideBatchSummary& summary) {
  printf("Batch %d summary: high-conflict stride=%u, low-conflict stride=%u\n",
         batch_index, summary.high_conflict_stride, summary.low_conflict_stride);
}

void print_layout_selection(const StrideLayoutSelection& selection) {
  printf("Selected layouts: high-conflict=%u (score=%.4f), low-conflict=%u (score=%.4f), relative gap=%.4f%%, stable=%d\n",
         selection.high_conflict_stride, selection.high_score,
         selection.low_conflict_stride, selection.low_score,
         selection.relative_gap_percent, selection.stable() ? 1 : 0);
}

void print_contention_summary(const char* layout_name, uint32_t stride_bytes,
                              uint32_t warp_count,
                              const NormalizedComparisonSummary& summary) {
  printf("│ %-13s │ %6u │ %5u │ %7d │ %9.2f │ %9.2f │ %9.2f │ %9.2f │ %9.2f │ %9.2f │\n",
         layout_name, stride_bytes, warp_count, summary.raw.sample_count,
         summary.raw.median, summary.control.median, summary.derived_median,
         summary.derived_p10, summary.derived_p90,
         static_cast<double>(summary.clock_overhead));
}

void export_layout_scan_csv(const char* filename,
                            const StrideLayoutSelection& selection,
                            const MBarrierBenchOptions& options,
                            uint64_t clock_overhead) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    ADD_FAILURE() << "Failed to open CSV output file: " << filename;
    return;
  }

  out << "row_type,batch_index,stride_bytes,warp_count,sample_count,"
         "warmup_iterations,measured_iterations,clock_overhead,min,p10,"
         "median,p90,max,batch_high_conflict_stride,batch_low_conflict_stride,"
         "selected_high_conflict_stride,selected_low_conflict_stride,"
         "high_score,low_score,relative_gap_percent,stable,distinct\n";

  for (size_t batch_index = 0; batch_index < selection.batches.size();
       ++batch_index) {
    const auto& batch = selection.batches[batch_index];
    for (const auto& point : batch.points) {
      out << "point," << batch_index << ',' << point.stride_bytes << ','
          << point.warp_count << ',' << point.raw.sample_count << ','
          << options.warmup_iterations << ',' << options.measured_iterations
          << ',' << clock_overhead << ',' << point.raw.min << ','
          << point.raw.p10 << ',' << point.raw.median << ',' << point.raw.p90
          << ',' << point.raw.max << ',' << batch.high_conflict_stride << ','
          << batch.low_conflict_stride << ','
          << selection.high_conflict_stride << ','
          << selection.low_conflict_stride << ',' << selection.high_score
          << ',' << selection.low_score << ','
          << selection.relative_gap_percent << ','
          << (selection.stable() ? 1 : 0) << ','
          << (selection.distinct() ? 1 : 0) << '\n';
    }
  }

  out << "selection,-1,-1,-1,-1," << options.warmup_iterations << ','
      << options.measured_iterations << ',' << clock_overhead
      << ",-1,-1,-1,-1,-1,-1,-1," << selection.high_conflict_stride << ','
      << selection.low_conflict_stride << ',' << selection.high_score << ','
      << selection.low_score << ',' << selection.relative_gap_percent << ','
      << (selection.stable() ? 1 : 0) << ','
      << (selection.distinct() ? 1 : 0) << '\n';
}

void export_contention_csv_header(std::ofstream& out) {
  out << "layout_name,stride_bytes,warp_count,sample_count,warmup_iterations,"
         "measured_iterations,clock_overhead,selected_high_conflict_stride,"
         "selected_low_conflict_stride,layout_relative_gap_percent,"
         "layout_stable,raw_min,raw_p10,raw_median,raw_p90,raw_max,"
         "control_min,control_p10,control_median,control_p90,control_max,"
         "derived_median,derived_p10,derived_p90\n";
}

void export_contention_csv_row(std::ofstream& out, const char* layout_name,
                               uint32_t stride_bytes, uint32_t warp_count,
                               const NormalizedComparisonSummary& summary,
                               const MBarrierBenchOptions& options,
                               const StrideLayoutSelection& layouts) {
  out << layout_name << ',' << stride_bytes << ',' << warp_count << ','
      << summary.raw.sample_count << ',' << options.warmup_iterations << ','
      << options.measured_iterations << ',' << summary.clock_overhead << ','
      << layouts.high_conflict_stride << ',' << layouts.low_conflict_stride
      << ',' << layouts.relative_gap_percent << ','
      << (layouts.stable() ? 1 : 0) << ',' << summary.raw.min << ','
      << summary.raw.p10 << ',' << summary.raw.median << ','
      << summary.raw.p90 << ',' << summary.raw.max << ','
      << summary.control.min << ',' << summary.control.p10 << ','
      << summary.control.median << ',' << summary.control.p90 << ','
      << summary.control.max << ',' << summary.derived_median << ','
      << summary.derived_p10 << ',' << summary.derived_p90 << '\n';
}

double aggregate_stride_score(const std::vector<StrideBatchSummary>& batches,
                              uint32_t stride_bytes) {
  double total_median = 0.0;
  int matches = 0;
  for (const auto& batch : batches) {
    for (const auto& point : batch.points) {
      if (point.stride_bytes == stride_bytes) {
        total_median += point.raw.median;
        ++matches;
      }
    }
  }
  return matches == 0 ? 0.0 : total_median / static_cast<double>(matches);
}

void expect_valid_summary(const NormalizedComparisonSummary& summary,
                          const MBarrierBenchOptions& options) {
  EXPECT_EQ(summary.raw.sample_count, options.measured_iterations);
  EXPECT_EQ(summary.control.sample_count, options.measured_iterations);
}

void launch_stride_scan(uint32_t warp_count, uint32_t stride_bytes,
                        uint64_t* cycle_start, uint64_t* cycle_end,
                        uint32_t* result_out) {
  contention_probe_kernel<true><<<1, kContentionThreads>>>(
      warp_count, stride_bytes, kStrideScanIterations, cycle_start, cycle_end,
      result_out);
}

void launch_contention_raw(uint32_t warp_count, uint32_t stride_bytes,
                           uint64_t* cycle_start, uint64_t* cycle_end,
                           uint32_t* result_out) {
  contention_probe_kernel<true><<<1, kContentionThreads>>>(
      warp_count, stride_bytes, kContentionIterations, cycle_start, cycle_end,
      result_out);
}

void launch_contention_control(uint32_t warp_count, uint32_t stride_bytes,
                               uint64_t* cycle_start, uint64_t* cycle_end,
                               uint32_t* result_out) {
  contention_probe_kernel<false><<<1, kContentionThreads>>>(
      warp_count, stride_bytes, kContentionIterations, cycle_start, cycle_end,
      result_out);
}

}  // namespace

class MBarrierContentionTest : public MBarrierBenchmarkFixture {
 protected:
  void SetUp() override {
    MBarrierBenchmarkFixture::SetUp();
    const cudaError_t err = cudaMalloc(&d_result_, 2 * sizeof(uint32_t));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate contention result buffer";
  }

  void TearDown() override {
    if (d_result_ != nullptr) {
      cudaFree(d_result_);
      d_result_ = nullptr;
    }
    MBarrierBenchmarkFixture::TearDown();
  }

  uint32_t* result_ptr() { return d_result_; }

  template <typename LaunchFn>
  MBarrierCycleSummary measure_normalized_summary(LaunchFn&& launch,
                                                  const MBarrierBenchOptions& options,
                                                  uint64_t clock_overhead,
                                                  uint32_t normalization_factor) {
    return normalize_summary(measure_summary(std::forward<LaunchFn>(launch),
                                             options, clock_overhead),
                             normalization_factor);
  }

  template <typename RawLaunchFn, typename ControlLaunchFn>
  NormalizedComparisonSummary measure_normalized_comparison(
      RawLaunchFn&& raw_launch, ControlLaunchFn&& control_launch,
      const MBarrierBenchOptions& options, uint64_t clock_overhead,
      uint32_t normalization_factor) {
    NormalizedComparisonSummary summary;
    summary.clock_overhead = clock_overhead;
    summary.raw = measure_normalized_summary(std::forward<RawLaunchFn>(raw_launch),
                                             options, clock_overhead,
                                             normalization_factor);
    summary.control = measure_normalized_summary(
        std::forward<ControlLaunchFn>(control_launch), options, clock_overhead,
        normalization_factor);
    summary.derived_median = summary.raw.median - summary.control.median;
    summary.derived_p10 = summary.raw.p10 - summary.control.p10;
    summary.derived_p90 = summary.raw.p90 - summary.control.p90;
    return summary;
  }

  StrideBatchSummary measure_stride_batch(const MBarrierBenchOptions& options,
                                          uint64_t clock_overhead) {
    StrideBatchSummary batch;
    batch.points.reserve(std::size(kStrideCandidates) *
                         std::size(kStrideScanWarps));

    double max_score = -1.0;
    double min_score = -1.0;
    for (uint32_t stride_bytes : kStrideCandidates) {
      double stride_score = 0.0;
      for (uint32_t warp_count : kStrideScanWarps) {
        const MBarrierCycleSummary summary = measure_normalized_summary(
            [&] {
              launch_stride_scan(warp_count, stride_bytes, cycle_start_ptr(),
                                 cycle_end_ptr(), result_ptr());
            },
            options, clock_overhead, kStrideScanIterations);
        batch.points.push_back({stride_bytes, warp_count, summary});
        stride_score += summary.median;
        print_stride_scan_point(stride_bytes, warp_count, summary);
      }

      if (max_score < 0.0 || stride_score > max_score) {
        max_score = stride_score;
        batch.high_conflict_stride = stride_bytes;
      }
      if (min_score < 0.0 || stride_score < min_score) {
        min_score = stride_score;
        batch.low_conflict_stride = stride_bytes;
      }
    }

    return batch;
  }

  StrideLayoutSelection classify_stride_layouts(
      const MBarrierBenchOptions& options, uint64_t clock_overhead) {
    StrideLayoutSelection selection;
    selection.batches.reserve(kStrideScanBatches);

    for (int batch = 0; batch < kStrideScanBatches; ++batch) {
      StrideBatchSummary summary = measure_stride_batch(options, clock_overhead);
      print_stride_batch_summary(batch, summary);
      selection.batches.push_back(summary);
    }

    double max_score = -1.0;
    double min_score = -1.0;
    for (uint32_t stride_bytes : kStrideCandidates) {
      const double score =
          aggregate_stride_score(selection.batches, stride_bytes);
      if (max_score < 0.0 || score > max_score) {
        max_score = score;
        selection.high_score = score;
        selection.high_conflict_stride = stride_bytes;
      }
      if (min_score < 0.0 || score < min_score) {
        min_score = score;
        selection.low_score = score;
        selection.low_conflict_stride = stride_bytes;
      }
    }

    if (selection.low_score > 0.0) {
      selection.relative_gap_percent =
          100.0 * (selection.high_score - selection.low_score) /
          selection.low_score;
    }
    print_layout_selection(selection);
    return selection;
  }

 private:
  uint32_t* d_result_ = nullptr;
};

TEST_F(MBarrierContentionTest, LayoutScan) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierContentionTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 2;
  options.measured_iterations = 17;
  options.subtract_clock_overhead = true;

  printf("\n=== P_contention Layout Scan (T4-pre) ===\n\n");
  printf("┌────────┬───────┬─────────┬───────────┬───────────┬───────────┬───────────┬───────────┐\n");
  printf("│ Stride │ Warps │ Samples │   Min     │   P10     │  Median   │   P90     │   Max     │\n");
  printf("├────────┼───────┼─────────┼───────────┼───────────┼───────────┼───────────┼───────────┤\n");

  const StrideLayoutSelection selection =
      classify_stride_layouts(options, clock_overhead);
  printf("└────────┴───────┴─────────┴───────────┴───────────┴───────────┴───────────┴───────────┘\n");
  constexpr const char* kCsvFile = "MBarrierContentionTest.LayoutScan.csv";
  export_layout_scan_csv(kCsvFile, selection, options, clock_overhead);
  printf("Results exported to: %s\n", kCsvFile);

  EXPECT_TRUE(selection.distinct())
      << "Stride scan did not produce distinct high/low layouts";
}

TEST_F(MBarrierContentionTest, PContention) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierContentionTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();

  MBarrierBenchOptions scan_options;
  scan_options.warmup_iterations = 2;
  scan_options.measured_iterations = 9;
  scan_options.subtract_clock_overhead = true;
  const StrideLayoutSelection layouts =
      classify_stride_layouts(scan_options, clock_overhead);
  if (!layouts.distinct()) {
    GTEST_SKIP() << "Stride layout classification did not produce distinct layouts";
  }
  if (!layouts.stable()) {
    printf(
        "[mbarrier] parameter=P_contention method=T4 "
        "note=weak_layout_separation selected_high=%u selected_low=%u "
        "relative_gap_pct=%.4f\n",
        layouts.high_conflict_stride, layouts.low_conflict_stride,
        layouts.relative_gap_percent);
  }

  MBarrierBenchOptions options;
  options.warmup_iterations = 3;
  options.measured_iterations = 17;
  options.subtract_clock_overhead = true;
  constexpr const char* kCsvFile = "MBarrierContentionTest.PContention.csv";
  std::ofstream csv_out(kCsvFile);
  ASSERT_TRUE(csv_out.is_open()) << "Failed to open CSV output file: "
                                 << kCsvFile;
  export_contention_csv_header(csv_out);

  printf("\n=== P_contention Summary (T4) ===\n\n");
  printf("┌───────────────┬────────┬───────┬─────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐\n");
  printf("│ Layout        │ Stride │ Warps │ Samples │ Raw Med   │ Ctl Med   │ Extra Med │ Extra P10 │ Extra P90 │ Clock OH  │\n");
  printf("├───────────────┼────────┼───────┼─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤\n");

  for (uint32_t warp_count : kContentionWarps) {
    const auto low_summary = measure_normalized_comparison(
        [&] {
          launch_contention_raw(warp_count, layouts.low_conflict_stride,
                                cycle_start_ptr(), cycle_end_ptr(),
                                result_ptr());
        },
        [&] {
          launch_contention_control(warp_count, layouts.low_conflict_stride,
                                    cycle_start_ptr(), cycle_end_ptr(),
                                    result_ptr());
        },
        options, clock_overhead, kContentionIterations);
    expect_valid_summary(low_summary, options);
    print_contention_summary("low-conflict", layouts.low_conflict_stride,
                             warp_count, low_summary);
    export_contention_csv_row(csv_out, "low-conflict",
                              layouts.low_conflict_stride, warp_count,
                              low_summary, options, layouts);

    const auto high_summary = measure_normalized_comparison(
        [&] {
          launch_contention_raw(warp_count, layouts.high_conflict_stride,
                                cycle_start_ptr(), cycle_end_ptr(),
                                result_ptr());
        },
        [&] {
          launch_contention_control(warp_count, layouts.high_conflict_stride,
                                    cycle_start_ptr(), cycle_end_ptr(),
                                    result_ptr());
        },
        options, clock_overhead, kContentionIterations);
    expect_valid_summary(high_summary, options);
    print_contention_summary("high-conflict", layouts.high_conflict_stride,
                             warp_count, high_summary);
    export_contention_csv_row(csv_out, "high-conflict",
                              layouts.high_conflict_stride, warp_count,
                              high_summary, options, layouts);
  }
  printf("└───────────────┴────────┴───────┴─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘\n");
  csv_out.close();
  printf("Results exported to: %s\n", kCsvFile);
}
