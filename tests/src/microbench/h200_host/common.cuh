#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t v = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(v)::"memory");
  return v;
}

// Device-wide nanosecond timer (synchronized across SMs). Use for cross-CTA
// simultaneity; do NOT use clock64 for that (per-SM, unsynchronized).
__device__ __forceinline__ uint64_t globaltimer_now() {
  uint64_t v = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)::"memory");
  return v;
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void *ptr) {
  uint64_t addr = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"(ptr));
  return addr;
}

__device__ __forceinline__ uint32_t smid_now() {
  uint32_t id = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// ---------------------------------------------------------------------------
// Host options / results
// ---------------------------------------------------------------------------

struct SuiteOptions {
  int device = 0;
  int samples = 48;
  int warmup = 8;
  // Shared clock64 overhead measured by device suite (0 if not yet set).
  double clock64_overhead = 0.0;
  // Observed SM clock (MHz) from busy-loop calibration; 0 if unknown.
  double sm_clock_mhz = 0.0;
  // DSM / cluster probes: cluster size in CTAs (0 = auto max).
  int dsm_cluster_size = 0;
  // Local / 5090 smoke: skip autotune, force winner tile, only smoke_tile.
  bool gemm_smoke = false;
};

struct Metric {
  std::string suite;
  std::string name;
  std::string sim_knob;
  double median = 0.0;
  double p90 = 0.0;
  double mean = 0.0;
  std::string unit = "cycles";
  std::string notes;
};

struct MetricSink {
  std::vector<Metric> metrics;

  void add(const char *suite, const char *name, const char *sim_knob,
           double median, double p90, double mean, const char *unit = "cycles",
           const char *notes = "") {
    Metric m;
    m.suite = suite;
    m.name = name;
    m.sim_knob = sim_knob;
    m.median = median;
    m.p90 = p90;
    m.mean = mean;
    m.unit = unit;
    m.notes = notes;
    metrics.push_back(std::move(m));
  }
};

struct CycleStats {
  double min = 0;
  double median = 0;
  double p90 = 0;
  double max = 0;
  double mean = 0;
  int n = 0;
};

inline double percentile_sorted(const std::vector<double> &v, double q) {
  if (v.empty()) return 0;
  q = std::clamp(q, 0.0, 1.0);
  const size_t i = static_cast<size_t>(q * (v.size() - 1));
  return v[i];
}

inline CycleStats summarize_u64(const std::vector<uint64_t> &raw,
                                double overhead = 0.0) {
  CycleStats s;
  if (raw.empty()) return s;
  std::vector<double> v;
  v.reserve(raw.size());
  for (uint64_t x : raw) {
    double d = static_cast<double>(x) - overhead;
    if (d < 0) d = 0;
    v.push_back(d);
  }
  std::sort(v.begin(), v.end());
  s.n = static_cast<int>(v.size());
  s.min = v.front();
  s.max = v.back();
  s.median = percentile_sorted(v, 0.5);
  s.p90 = percentile_sorted(v, 0.9);
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  return s;
}

inline CycleStats summarize_f64(std::vector<double> v) {
  CycleStats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.n = static_cast<int>(v.size());
  s.min = v.front();
  s.max = v.back();
  s.median = percentile_sorted(v, 0.5);
  s.p90 = percentile_sorted(v, 0.9);
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  return s;
}

inline void print_metric_line(const char *name, const CycleStats &s,
                              const char *extra = nullptr) {
  std::printf("  %-44s median=%10.2f  p90=%10.2f  mean=%10.2f  n=%d", name,
              s.median, s.p90, s.mean, s.n);
  if (extra && extra[0]) std::printf("  (%s)", extra);
  std::printf("\n");
}

// Bound a default-stream launch. On timeout or illegal instruction: do not
// DeviceReset / EventDestroy / cudaFree. Job 2111262 hung 4h in DeviceReset
// after a cluster-kernel timeout. The process should return and exit so the
// next srun starts with a clean GPU.
inline bool sync_device_timeout(float timeout_s, int device, const char *what) {
  (void)device;
  const auto t0 = std::chrono::steady_clock::now();
  cudaError_t q = cudaErrorNotReady;
  while ((q = cudaStreamQuery(0)) == cudaErrorNotReady) {
    const float dt = std::chrono::duration<float>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    if (dt > timeout_s) {
      std::printf(
          "  TIMEOUT (%.1fs) %s — skip DeviceReset (process will exit)\n",
          timeout_s, what);
      std::fflush(stdout);
      (void)cudaGetLastError();
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (q != cudaSuccess) {
    std::printf("  sync failed for %s: %s\n", what, cudaGetErrorString(q));
    std::fflush(stdout);
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

// Host wall-time of a launch body via CUDA events (ms).
template <typename F>
inline float time_kernel_ms(F &&launch, int warmup = 2) {
  for (int w = 0; w < warmup; ++w) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------

inline void print_kernel_source(const char *path, const char *origin,
                                const char *timed_region, const char *knob) {
  std::printf("kernel_source: %s\n", path);
  std::printf("kernel_origin: %s\n", origin);
  std::printf("timed_region: %s\n", timed_region);
  std::printf("sim_knob: %s\n", knob);
  std::fflush(stdout);
}
