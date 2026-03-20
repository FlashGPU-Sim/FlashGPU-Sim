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

constexpr uint32_t kVisibilityWarmupSteps = 4;
constexpr int kTmaTensorMapBytes = 128;
constexpr int kTmaGlobalScratchBytes = 256;
constexpr int kTmaThreads = 64;
constexpr int kMaxTmaTileFloats = 256;

#define VISIBILITY_FILLER_X1 "add.u32 %0, %0, %1;\n"
#define VISIBILITY_FILLER_X2 VISIBILITY_FILLER_X1 VISIBILITY_FILLER_X1
#define VISIBILITY_FILLER_X4 VISIBILITY_FILLER_X2 VISIBILITY_FILLER_X2
#define VISIBILITY_FILLER_X8 VISIBILITY_FILLER_X4 VISIBILITY_FILLER_X4

struct TmaVisibilitySummary {
  MBarrierCycleSummary data_raw;
  MBarrierCycleSummary barrier_raw;
  MBarrierCycleSummary barrier_control;
  MBarrierCycleSummary data_iters;
  MBarrierCycleSummary barrier_iters;
  double barrier_extra_median = 0.0;
  uint64_t clock_overhead = 0;
};

__host__ __device__ inline uint32_t tma_signature_word(uint32_t index) {
  return 0x3f800000u + index * 257u;
}

__device__ __forceinline__ uint32_t run_visibility_filler_step(
    uint32_t value, uint32_t delta) {
  asm volatile(VISIBILITY_FILLER_X8 : "+r"(value) : "r"(delta));
  return value;
}

__device__ __forceinline__ uint32_t run_visibility_filler_steps(
    uint32_t value, uint32_t delta, uint32_t step_count) {
  #pragma unroll 1
  for (uint32_t i = 0; i < step_count; ++i) {
    value = run_visibility_filler_step(value, delta);
  }
  return value;
}

__device__ inline void initialize_tensormap_shared(void* tmap_smem, int tid) {
  uint32_t* words = reinterpret_cast<uint32_t*>(tmap_smem);
  if (tid < 32) {
    words[tid] = 0;
  }
  __syncthreads();
}

template <int TileFloats>
__device__ __forceinline__ void configure_tensormap_1d(void* tmap_smem,
                                                       const float* source) {
  const uint64_t tmap_addr = __cvta_generic_to_shared(tmap_smem);
  tensormap_set_global_address(tmap_addr, reinterpret_cast<uint64_t>(source));
  asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x0;"
               :
               : "l"(tmap_addr));

  const uint32_t box_dim = TileFloats;
  asm volatile(
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;"
      :
      : "l"(tmap_addr), "r"(box_dim));

  const uint32_t global_dim = TileFloats;
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, %1;"
      :
      : "l"(tmap_addr), "r"(global_dim));

  const uint32_t element_stride = 1;
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 "
      "[%0], 0x0, %1;"
      :
      : "l"(tmap_addr), "r"(element_stride));

  asm volatile("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], "
               "0x7;"
               :
               : "l"(tmap_addr));
  asm volatile(
      "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 "
      "[%0], 0x0;"
      :
      : "l"(tmap_addr));
  asm volatile("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 "
               "[%0], 0x0;"
               :
               : "l"(tmap_addr));
  asm volatile("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], "
               "0x0;"
               :
               : "l"(tmap_addr));
}

template <int TileFloats>
__device__ __forceinline__ void initialize_signature_tile(uint32_t* tile_words,
                                                          int tid,
                                                          int thread_count) {
  for (int i = tid; i < TileFloats; i += thread_count) {
    tile_words[i] = tma_signature_word(static_cast<uint32_t>(i));
  }
}

template <int TileFloats>
__device__ __forceinline__ bool tile_signature_matches(
    volatile const uint32_t* tile_words) {
  #pragma unroll
  for (int i = 0; i < TileFloats; i += 4) {
    if (tile_words[i] != tma_signature_word(static_cast<uint32_t>(i))) {
      return false;
    }
  }
  return true;
}

template <int TileFloats>
__device__ __forceinline__ bool tile_sentinel_matches(
    volatile const uint32_t* tile_words) {
  constexpr int kSentinelIndex = TileFloats - 1;
  return tile_words[kSentinelIndex] ==
         tma_signature_word(static_cast<uint32_t>(kSentinelIndex));
}

__global__ void arrive_visibility_filler_calibration_kernel(
    uint32_t step_count, uint64_t* cycle_start, uint64_t* cycle_end,
    uint32_t* result_out) {
  if (threadIdx.x == 0) {
    const uint32_t seed = static_cast<uint32_t>(clock64());
    const uint32_t delta = seed | 1u;
    uint32_t sink =
        run_visibility_filler_steps(seed, delta, kVisibilityWarmupSteps);

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    sink ^= run_visibility_filler_steps(sink, delta, step_count);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    result_out[0] = sink;
    result_out[1] = delta;
  }
}

__global__ void arrive_visibility_probe_kernel(uint32_t step_count,
                                               uint32_t* result_out) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    const uint32_t seed = static_cast<uint32_t>(clock64());
    const uint32_t delta = seed | 1u;
    uint32_t sink =
        run_visibility_filler_steps(seed, delta, kVisibilityWarmupSteps);

    mbarrier_arrive_expect_tx(&barrier, 0);
    sink ^= run_visibility_filler_steps(sink, delta, step_count);

    result_out[0] = mbarrier_try_wait_parity(&barrier, 0) ? 1u : 0u;
    result_out[1] = sink;
  }
}

template <int TileFloats>
__global__ void tma_data_visibility_kernel(const float* source,
                                           uint8_t* global_scratch,
                                           uint64_t* cycle_start,
                                           uint64_t* cycle_end,
                                           uint32_t* result_out) {
  extern __shared__ uint8_t smem[];

  uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_smem =
      reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~static_cast<uintptr_t>(127));
  float* tile = reinterpret_cast<float*>(tmap_smem + kTmaTensorMapBytes);
  uint32_t* control_tile_words =
      reinterpret_cast<uint32_t*>(tile + TileFloats);
  uint64_t* false_barrier =
      reinterpret_cast<uint64_t*>(control_tile_words + TileFloats);
  uint64_t* true_barrier = false_barrier + 1;
  volatile uint32_t* observer_ready =
      reinterpret_cast<volatile uint32_t*>(true_barrier + 1);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  initialize_tensormap_shared(tmap_smem, tid);
  for (int i = tid; i < TileFloats; i += blockDim.x) {
    tile[i] = 0.0f;
  }
  if (tid == 0) {
    configure_tensormap_1d<TileFloats>(tmap_smem, source);
    mbarrier_init(false_barrier, 1);
    mbarrier_init(true_barrier, 1);
    mbarrier_arrive_expect_tx(true_barrier, TileFloats * sizeof(float));
    *observer_ready = 0;
    result_out[0] = 0;
    result_out[1] = 0;
  }
  __syncthreads();

  if (tid < 32) {
    const uint64_t tmap_addr = __cvta_generic_to_shared(tmap_smem);
    const uint64_t global_tmap = reinterpret_cast<uint64_t>(global_scratch);
    tensormap_cp_fenceproxy(global_tmap, tmap_addr);
    fence_proxy_tensormap_acquire(global_tmap);
  }
  __syncthreads();

  if (tid == 0) {
    mbarrier_init(false_barrier, 1);
    mbarrier_arrive_expect_tx(false_barrier, TileFloats * sizeof(float));
  }
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  if (warp == 1 && lane == 0) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t iterations = 0;
    volatile const uint32_t* tile_words =
        reinterpret_cast<volatile const uint32_t*>(tile);
    *observer_ready = 1;
    asm volatile("membar.cta;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    do {
      ++iterations;
    } while (!tile_signature_matches<TileFloats>(tile_words));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    cycle_start[0] = start;
    cycle_end[0] = end;
    result_out[0] = iterations;
    result_out[1] = tile_words[0];
  } else if (warp == 0 && lane == 0) {
    while (*observer_ready == 0) {
    }
    const uint32_t smem_addr = smem_u32_addr(tile);
    const uint32_t mbar_addr = smem_u32_addr(false_barrier);
    cp_async_bulk_tensor_1d_load(
        smem_addr, reinterpret_cast<uint64_t>(global_scratch), 0, mbar_addr);
  }
}

template <int TileFloats>
__global__ void tma_data_visibility_control_kernel(uint32_t expected_iterations,
                                                   uint64_t* cycle_start,
                                                   uint64_t* cycle_end,
                                                   uint32_t* result_out) {
  extern __shared__ uint8_t smem[];

  uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_smem =
      reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~static_cast<uintptr_t>(127));
  float* tile = reinterpret_cast<float*>(tmap_smem + kTmaTensorMapBytes);
  uint32_t* control_tile_words =
      reinterpret_cast<uint32_t*>(tile + TileFloats);
  volatile uint32_t* observer_ready =
      reinterpret_cast<volatile uint32_t*>(control_tile_words + TileFloats);
  volatile uint32_t* observer_iterations = observer_ready + 1;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  initialize_tensormap_shared(tmap_smem, tid);
  for (int i = tid; i < TileFloats; i += blockDim.x) {
    reinterpret_cast<uint32_t*>(tile)[i] = tma_signature_word(i);
  }
  if (tid == 0) {
    reinterpret_cast<uint32_t*>(tile)[TileFloats - 1] = 0;
    *observer_ready = 0;
    *observer_iterations = 0;
    result_out[0] = 0;
    result_out[1] = 0;
  }
  __syncthreads();

  if (warp == 1 && lane == 0) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t iterations = 0;
    volatile const uint32_t* tile_words =
        reinterpret_cast<volatile const uint32_t*>(tile);
    *observer_ready = 1;
    asm volatile("membar.cta;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    do {
      ++iterations;
      *observer_iterations = iterations;
      asm volatile("membar.cta;" ::: "memory");
    } while (!tile_sentinel_matches<TileFloats>(tile_words));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    cycle_start[0] = start;
    cycle_end[0] = end;
    result_out[0] = iterations;
    result_out[1] = tile_words[TileFloats - 1];
  } else if (warp == 0 && lane == 0) {
    while (*observer_ready == 0) {
    }

    const uint32_t trigger_iteration =
        expected_iterations > 0 ? expected_iterations - 1 : 0;
    while (*observer_iterations < trigger_iteration) {
    }

    uint32_t* tile_words = reinterpret_cast<uint32_t*>(tile);
    tile_words[TileFloats - 1] =
        tma_signature_word(static_cast<uint32_t>(TileFloats - 1));
    asm volatile("membar.cta;" ::: "memory");
  }
}

template <int TileFloats>
__global__ void tma_barrier_visibility_kernel(const float* source,
                                              uint8_t* global_scratch,
                                              uint64_t* cycle_start,
                                              uint64_t* cycle_end,
                                              uint32_t* result_out) {
  extern __shared__ uint8_t smem[];

  uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_smem =
      reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~static_cast<uintptr_t>(127));
  float* tile = reinterpret_cast<float*>(tmap_smem + kTmaTensorMapBytes);
  uint32_t* control_tile_words =
      reinterpret_cast<uint32_t*>(tile + TileFloats);
  uint64_t* barrier = reinterpret_cast<uint64_t*>(control_tile_words + TileFloats);
  uint64_t* ready_barrier = barrier + 1;
  volatile uint32_t* observer_ready =
      reinterpret_cast<volatile uint32_t*>(ready_barrier + 1);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  initialize_tensormap_shared(tmap_smem, tid);
  for (int i = tid; i < TileFloats; i += blockDim.x) {
    tile[i] = 0.0f;
  }
  initialize_signature_tile<TileFloats>(control_tile_words, tid, blockDim.x);
  if (tid == 0) {
    configure_tensormap_1d<TileFloats>(tmap_smem, source);
    mbarrier_init(barrier, 1);
    mbarrier_arrive_expect_tx(barrier, TileFloats * sizeof(float));
    mbarrier_init(ready_barrier, 1);
    mbarrier_arrive_expect_tx(ready_barrier, 0);
    while (!mbarrier_try_wait_parity(ready_barrier, 0)) {
    }
    *observer_ready = 0;
    result_out[0] = 0;
    result_out[1] = 0;
  }
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  if (tid < 32) {
    const uint64_t tmap_addr = __cvta_generic_to_shared(tmap_smem);
    const uint64_t global_tmap = reinterpret_cast<uint64_t>(global_scratch);
    tensormap_cp_fenceproxy(global_tmap, tmap_addr);
    fence_proxy_tensormap_acquire(global_tmap);
  }
  __syncthreads();

  if (warp == 1 && lane == 0) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t iterations = 0;
    *observer_ready = 1;
    asm volatile("membar.cta;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    do {
      ++iterations;
    } while (!mbarrier_try_wait_parity(barrier, 0));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    cycle_start[0] = start;
    cycle_end[0] = end;
    result_out[0] = iterations;
    result_out[1] = mbarrier_try_wait_parity(ready_barrier, 0) ? 1u : 0u;
  } else if (warp == 0 && lane == 0) {
    while (*observer_ready == 0) {
    }
    const uint32_t smem_addr = smem_u32_addr(tile);
    const uint32_t mbar_addr = smem_u32_addr(barrier);
    cp_async_bulk_tensor_1d_load(
        smem_addr, reinterpret_cast<uint64_t>(global_scratch), 0, mbar_addr);
  }
}

template <int TileFloats>
__global__ void tma_barrier_visibility_control_kernel(
    uint32_t expected_iterations, uint64_t* cycle_start, uint64_t* cycle_end,
    uint32_t* result_out) {
  extern __shared__ uint8_t smem[];

  uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_smem =
      reinterpret_cast<uint8_t*>((smem_ptr + 127) & ~static_cast<uintptr_t>(127));
  float* tile = reinterpret_cast<float*>(tmap_smem + kTmaTensorMapBytes);
  uint32_t* control_tile_words =
      reinterpret_cast<uint32_t*>(tile + TileFloats);
  uint64_t* false_barrier =
      reinterpret_cast<uint64_t*>(control_tile_words + TileFloats);
  uint64_t* true_barrier = false_barrier + 1;

  const int tid = static_cast<int>(threadIdx.x);

  initialize_tensormap_shared(tmap_smem, tid);
  if (tid == 0) {
    mbarrier_init(false_barrier, 2);
    mbarrier_arrive(false_barrier);
    mbarrier_init(true_barrier, 1);
    mbarrier_arrive_expect_tx(true_barrier, 0);
    while (!mbarrier_try_wait_parity(true_barrier, 0)) {
    }
  }
  __syncthreads();

  if (tid == 32) {
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t iterations = 0;
    uint32_t sink = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    do {
      ++iterations;
      const bool ready =
          iterations >= expected_iterations
              ? mbarrier_try_wait_parity(true_barrier, 0)
              : mbarrier_try_wait_parity(false_barrier, 0);
      sink ^= ready ? 1u : 0u;
      if (ready && iterations >= expected_iterations) {
        break;
      }
    } while (true);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    cycle_start[0] = start;
    cycle_end[0] = end;
    result_out[0] = iterations;
    result_out[1] = sink;
  }
}

void expect_valid_sweep_summary(const MBarrierSweepSummary& summary,
                                const std::vector<uint32_t>& sweep_values,
                                const MBarrierBenchOptions& options) {
  ASSERT_EQ(summary.points.size(), sweep_values.size());
  for (size_t i = 0; i < sweep_values.size(); ++i) {
    EXPECT_EQ(summary.points[i].instruction_count, sweep_values[i]);
    EXPECT_EQ(summary.points[i].cycles.sample_count,
              options.measured_iterations);
  }
}

inline uint64_t adjusted_elapsed(uint64_t start, uint64_t end,
                                 uint64_t clock_overhead) {
  uint64_t elapsed = end - start;
  if (elapsed > clock_overhead) {
    elapsed -= clock_overhead;
  }
  return elapsed;
}

inline double median_signed(std::vector<int64_t> samples) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  return static_cast<double>(samples[samples.size() / 2]);
}

inline void print_tma_data_visibility_summary(int tile_bytes,
                                              const TmaVisibilitySummary& summary,
                                              int sample_count) {
  printf("│ %8d │ %7d │ %10.2f │ %10.2f │ %10.2f │ %10.2f │\n",
         tile_bytes, sample_count, summary.data_raw.min,
         summary.data_raw.median, summary.data_raw.max,
         summary.data_iters.median);
}

inline void print_tma_barrier_visibility_summary(
    int tile_bytes, const TmaVisibilitySummary& summary, int sample_count) {
  printf("│ %8d │ %7d │ %10.2f │ %10.2f │ %10.2f │ %10.2f │ %10.2f │ %10.2f │ %10.2f │\n",
         tile_bytes, sample_count, summary.barrier_raw.min,
         summary.barrier_raw.median, summary.barrier_raw.max,
         summary.barrier_control.median, summary.barrier_iters.median,
         summary.barrier_extra_median,
         static_cast<double>(summary.clock_overhead));
}

inline void export_tma_visibility_csv_row(std::ofstream& out,
                                          uint32_t tile_bytes,
                                          const TmaVisibilitySummary& summary,
                                          const MBarrierBenchOptions& options) {
  out << tile_bytes << ',' << options.warmup_iterations << ','
      << options.measured_iterations << ',' << summary.clock_overhead << ','
      << summary.data_raw.sample_count << ',' << summary.data_raw.min << ','
      << summary.data_raw.p10 << ',' << summary.data_raw.median << ','
      << summary.data_raw.p90 << ',' << summary.data_raw.max << ','
      << summary.data_iters.min << ',' << summary.data_iters.p10 << ','
      << summary.data_iters.median << ',' << summary.data_iters.p90 << ','
      << summary.data_iters.max << ',' << summary.barrier_raw.sample_count
      << ',' << summary.barrier_raw.min << ',' << summary.barrier_raw.p10
      << ',' << summary.barrier_raw.median << ','
      << summary.barrier_raw.p90 << ',' << summary.barrier_raw.max << ','
      << summary.barrier_control.sample_count << ','
      << summary.barrier_control.min << ',' << summary.barrier_control.p10
      << ',' << summary.barrier_control.median << ','
      << summary.barrier_control.p90 << ',' << summary.barrier_control.max
      << ',' << summary.barrier_iters.min << ',' << summary.barrier_iters.p10
      << ',' << summary.barrier_iters.median << ','
      << summary.barrier_iters.p90 << ',' << summary.barrier_iters.max << ','
      << summary.barrier_extra_median << '\n';
}

}  // namespace

class MBarrierVisibilityTest : public MBarrierBenchmarkFixture {
 protected:
  void SetUp() override {
    MBarrierBenchmarkFixture::SetUp();

    cudaError_t err = cudaMalloc(&d_result_, 8 * sizeof(uint32_t));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate visibility result buffer";

    err = cudaMalloc(&d_source_, kMaxTmaTileFloats * sizeof(float));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate TMA source buffer";

    err = cudaMalloc(&d_global_scratch_, kTmaGlobalScratchBytes);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate TMA scratch buffer";

    std::vector<uint32_t> signature_words(kMaxTmaTileFloats);
    for (int i = 0; i < kMaxTmaTileFloats; ++i) {
      signature_words[i] = tma_signature_word(static_cast<uint32_t>(i));
    }
    err = cudaMemcpy(d_source_, signature_words.data(),
                     signature_words.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to initialize TMA source buffer";

    err = cudaMemset(d_global_scratch_, 0, kTmaGlobalScratchBytes);
    ASSERT_EQ(err, cudaSuccess) << "Failed to clear TMA scratch buffer";
  }

  void TearDown() override {
    if (d_global_scratch_ != nullptr) {
      cudaFree(d_global_scratch_);
      d_global_scratch_ = nullptr;
    }
    if (d_source_ != nullptr) {
      cudaFree(d_source_);
      d_source_ = nullptr;
    }
    if (d_result_ != nullptr) {
      cudaFree(d_result_);
      d_result_ = nullptr;
    }
    MBarrierBenchmarkFixture::TearDown();
  }

  uint32_t* result_ptr() { return d_result_; }
  float* source_ptr() { return d_source_; }
  uint8_t* global_scratch_ptr() { return d_global_scratch_; }

  template <typename LaunchFn>
  MBarrierThresholdSummary measure_threshold_hits(
      LaunchFn&& launch, const std::vector<uint32_t>& sweep_values,
      const MBarrierBenchOptions& options, double filler_step_cycles,
      uint64_t clock_overhead, const char* parameter, const char* method) {
    MBarrierThresholdSummary summary;
    summary.filler_step_cycles = filler_step_cycles;
    summary.clock_overhead = clock_overhead;
    summary.points.reserve(sweep_values.size());

    for (const uint32_t sweep_value : sweep_values) {
      for (int i = 0; i < options.warmup_iterations; ++i) {
        launch(sweep_value);
        const cudaError_t warmup_err = cudaDeviceSynchronize();
        EXPECT_EQ(warmup_err, cudaSuccess)
            << "Warmup kernel failed: " << cudaGetErrorString(warmup_err);
        if (warmup_err != cudaSuccess) {
          print_mbarrier_threshold(parameter, method, summary);
          return summary;
        }
      }

      int hits = 0;
      for (int i = 0; i < options.measured_iterations; ++i) {
        launch(sweep_value);
        const cudaError_t sync_err = cudaDeviceSynchronize();
        EXPECT_EQ(sync_err, cudaSuccess)
            << "Measured kernel failed: " << cudaGetErrorString(sync_err);
        if (sync_err != cudaSuccess) {
          print_mbarrier_threshold(parameter, method, summary);
          return summary;
        }

        const cudaError_t copy_err =
            cudaMemcpy(h_result_, d_result_, 2 * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
        EXPECT_EQ(copy_err, cudaSuccess)
            << "Failed to copy visibility result from device";
        if (copy_err != cudaSuccess) {
          print_mbarrier_threshold(parameter, method, summary);
          return summary;
        }

        hits += h_result_[0] != 0 ? 1 : 0;
      }

      const double hit_rate =
          static_cast<double>(hits) / options.measured_iterations;
      summary.points.push_back(
          {sweep_value, hits, options.measured_iterations, hit_rate});

      if (summary.first_any_hit < 0 && hits > 0) {
        summary.first_any_hit = static_cast<int>(sweep_value);
        summary.approx_first_any_cycles =
            filler_step_cycles * static_cast<double>(sweep_value);
      }
      if (summary.first_all_hit < 0 && hits == options.measured_iterations) {
        summary.first_all_hit = static_cast<int>(sweep_value);
        summary.approx_first_all_cycles =
            filler_step_cycles * static_cast<double>(sweep_value);
      }
    }

    print_mbarrier_threshold(parameter, method, summary);
    return summary;
  }

  template <typename DataRawLaunchFn, typename BarrierRawLaunchFn,
            typename BarrierControlLaunchFn>
  TmaVisibilitySummary measure_tma_visibility(
      DataRawLaunchFn&& data_raw_launch,
      BarrierRawLaunchFn&& barrier_raw_launch,
      BarrierControlLaunchFn&& barrier_control_launch,
      const MBarrierBenchOptions& options, uint64_t clock_overhead) {
    auto run_and_read = [&](auto&& launch, uint32_t* iteration_count,
                            uint64_t* elapsed_cycles) -> bool {
      launch();
      const cudaError_t sync_err = cudaDeviceSynchronize();
      EXPECT_EQ(sync_err, cudaSuccess)
          << "Measured kernel failed: " << cudaGetErrorString(sync_err);
      if (sync_err != cudaSuccess) {
        return false;
      }

      uint64_t start = 0;
      uint64_t end = 0;
      cudaError_t copy_err =
          cudaMemcpy(&start, cycle_start_ptr(), sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);
      EXPECT_EQ(copy_err, cudaSuccess)
          << "Failed to copy cycle_start from device";
      if (copy_err != cudaSuccess) {
        return false;
      }

      copy_err = cudaMemcpy(&end, cycle_end_ptr(), sizeof(uint64_t),
                            cudaMemcpyDeviceToHost);
      EXPECT_EQ(copy_err, cudaSuccess) << "Failed to copy cycle_end from device";
      if (copy_err != cudaSuccess) {
        return false;
      }

      copy_err = cudaMemcpy(h_result_, result_ptr(), 8 * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost);
      EXPECT_EQ(copy_err, cudaSuccess)
          << "Failed to copy result buffer from device";
      if (copy_err != cudaSuccess) {
        return false;
      }

      *iteration_count = h_result_[0];
      *elapsed_cycles = adjusted_elapsed(start, end, clock_overhead);
      return true;
    };

    for (int i = 0; i < options.warmup_iterations; ++i) {
      uint32_t data_iters = 0;
      uint32_t barrier_iters = 0;
      uint64_t elapsed = 0;
      if (!run_and_read([&] { data_raw_launch(); }, &data_iters, &elapsed)) {
        return {};
      }
      if (!run_and_read([&] { barrier_raw_launch(); }, &barrier_iters,
                        &elapsed)) {
        return {};
      }
      if (!run_and_read([&] { barrier_control_launch(barrier_iters); },
                        &barrier_iters, &elapsed)) {
        return {};
      }
    }

    std::vector<uint64_t> data_raw_samples;
    std::vector<uint64_t> barrier_raw_samples;
    std::vector<uint64_t> barrier_control_samples;
    std::vector<uint64_t> data_iter_samples;
    std::vector<uint64_t> barrier_iter_samples;
    std::vector<int64_t> barrier_extra_samples;

    data_raw_samples.reserve(options.measured_iterations);
    barrier_raw_samples.reserve(options.measured_iterations);
    barrier_control_samples.reserve(options.measured_iterations);
    data_iter_samples.reserve(options.measured_iterations);
    barrier_iter_samples.reserve(options.measured_iterations);
    barrier_extra_samples.reserve(options.measured_iterations);

    for (int i = 0; i < options.measured_iterations; ++i) {
      uint32_t data_iters = 0;
      uint32_t barrier_iters = 0;
      uint32_t dummy_iters = 0;
      uint64_t data_raw_elapsed = 0;
      uint64_t barrier_raw_elapsed = 0;
      uint64_t barrier_control_elapsed = 0;

      if (!run_and_read([&] { data_raw_launch(); }, &data_iters,
                        &data_raw_elapsed)) {
        break;
      }
      if (!run_and_read([&] { barrier_raw_launch(); }, &barrier_iters,
                        &barrier_raw_elapsed)) {
        break;
      }
      if (!run_and_read([&] { barrier_control_launch(barrier_iters); },
                        &dummy_iters, &barrier_control_elapsed)) {
        break;
      }

      const int64_t barrier_extra =
          static_cast<int64_t>(barrier_raw_elapsed) -
          static_cast<int64_t>(barrier_control_elapsed);

      data_raw_samples.push_back(data_raw_elapsed);
      barrier_raw_samples.push_back(barrier_raw_elapsed);
      barrier_control_samples.push_back(barrier_control_elapsed);
      data_iter_samples.push_back(data_iters);
      barrier_iter_samples.push_back(barrier_iters);
      barrier_extra_samples.push_back(barrier_extra);
    }

    TmaVisibilitySummary summary;
    summary.clock_overhead = clock_overhead;
    summary.data_raw = summarize_mbarrier_cycles(data_raw_samples);
    summary.barrier_raw = summarize_mbarrier_cycles(barrier_raw_samples);
    summary.barrier_control = summarize_mbarrier_cycles(barrier_control_samples);
    summary.data_iters = summarize_mbarrier_cycles(data_iter_samples);
    summary.barrier_iters = summarize_mbarrier_cycles(barrier_iter_samples);
    summary.barrier_extra_median = median_signed(barrier_extra_samples);
    return summary;
  }

 private:
  uint32_t* d_result_ = nullptr;
  float* d_source_ = nullptr;
  uint8_t* d_global_scratch_ = nullptr;
  uint32_t h_result_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

void launch_arrive_visibility_filler_calibration(uint32_t step_count,
                                                 uint64_t* cycle_start,
                                                 uint64_t* cycle_end,
                                                 uint32_t* result_out) {
  arrive_visibility_filler_calibration_kernel<<<1, 32>>>(
      step_count, cycle_start, cycle_end, result_out);
}

void launch_arrive_visibility_probe(uint32_t step_count, uint32_t* result_out) {
  arrive_visibility_probe_kernel<<<1, 32>>>(step_count, result_out);
}

void launch_tma_data_visibility(uint32_t tile_bytes, size_t shared_memory_bytes,
                                float* source, uint8_t* global_scratch,
                                uint64_t* cycle_start, uint64_t* cycle_end,
                                uint32_t* result_out) {
  switch (tile_bytes) {
    case 128:
      tma_data_visibility_kernel<32><<<1, kTmaThreads, shared_memory_bytes>>>(
          source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 256:
      tma_data_visibility_kernel<64><<<1, kTmaThreads, shared_memory_bytes>>>(
          source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 512:
      tma_data_visibility_kernel<128><<<1, kTmaThreads, shared_memory_bytes>>>(
          source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 1024:
      tma_data_visibility_kernel<256><<<1, kTmaThreads, shared_memory_bytes>>>(
          source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported TMA tile size: " << tile_bytes;
      return;
  }
}

void launch_tma_data_visibility_control(uint32_t tile_bytes,
                                        size_t shared_memory_bytes,
                                        uint32_t expected_iterations,
                                        uint64_t* cycle_start,
                                        uint64_t* cycle_end,
                                        uint32_t* result_out) {
  switch (tile_bytes) {
    case 128:
      tma_data_visibility_control_kernel<32>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 256:
      tma_data_visibility_control_kernel<64>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 512:
      tma_data_visibility_control_kernel<128>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 1024:
      tma_data_visibility_control_kernel<256>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported TMA tile size: " << tile_bytes;
      return;
  }
}

void launch_tma_barrier_visibility(uint32_t tile_bytes,
                                   size_t shared_memory_bytes, float* source,
                                   uint8_t* global_scratch,
                                   uint64_t* cycle_start, uint64_t* cycle_end,
                                   uint32_t* result_out) {
  switch (tile_bytes) {
    case 128:
      tma_barrier_visibility_kernel<32>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 256:
      tma_barrier_visibility_kernel<64>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 512:
      tma_barrier_visibility_kernel<128>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    case 1024:
      tma_barrier_visibility_kernel<256>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              source, global_scratch, cycle_start, cycle_end, result_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported TMA tile size: " << tile_bytes;
      return;
  }
}

void launch_tma_barrier_visibility_control(uint32_t tile_bytes,
                                           size_t shared_memory_bytes,
                                           uint32_t expected_iterations,
                                           uint64_t* cycle_start,
                                           uint64_t* cycle_end,
                                           uint32_t* result_out) {
  switch (tile_bytes) {
    case 128:
      tma_barrier_visibility_control_kernel<32>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 256:
      tma_barrier_visibility_control_kernel<64>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 512:
      tma_barrier_visibility_control_kernel<128>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    case 1024:
      tma_barrier_visibility_control_kernel<256>
          <<<1, kTmaThreads, shared_memory_bytes>>>(
              expected_iterations, cycle_start, cycle_end, result_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported TMA tile size: " << tile_bytes;
      return;
  }
}

TEST_F(MBarrierVisibilityTest, P3Arrive) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierVisibilityTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> sweep_values = {0, 1, 2, 3, 4, 5, 6, 7, 8,
                                              10, 12, 16, 20, 24, 32};

  const auto filler_summary = measure_with_sweep(
      [&](uint32_t step_count) {
        launch_arrive_visibility_filler_calibration(step_count, cycle_start_ptr(),
                                                    cycle_end_ptr(),
                                                    result_ptr());
      },
      sweep_values, options, clock_overhead, "P3_arrive_step",
      "T2b filler calibration");

  expect_valid_sweep_summary(filler_summary, sweep_values, options);

  const auto threshold_summary = measure_threshold_hits(
      [&](uint32_t step_count) {
        launch_arrive_visibility_probe(step_count, result_ptr());
      },
      sweep_values, options, filler_summary.fit.slope, clock_overhead,
      "P3_arrive", "T2b arrive visibility threshold");

  ASSERT_EQ(threshold_summary.points.size(), sweep_values.size());
  constexpr const char* kFillerCsvFile =
      "MBarrierVisibilityTest.P3Arrive.FillerCalibration.csv";
  constexpr const char* kThresholdCsvFile =
      "MBarrierVisibilityTest.P3Arrive.Threshold.csv";
  export_mbarrier_sweep_csv(kFillerCsvFile, "P3_arrive_step",
                            "T2b filler calibration", filler_summary, options);
  export_mbarrier_threshold_csv(kThresholdCsvFile, "P3_arrive",
                                "T2b arrive visibility threshold",
                                threshold_summary, options);
  printf("Results exported to: %s\n", kFillerCsvFile);
  printf("Results exported to: %s\n", kThresholdCsvFile);
}

TEST_F(MBarrierVisibilityTest, P3Tma) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierVisibilityTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 3;
  options.measured_iterations = 31;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> tile_bytes = {128, 256, 512, 1024};
  std::vector<TmaVisibilitySummary> summaries;
  summaries.reserve(tile_bytes.size());
  constexpr const char* kCsvFile = "MBarrierVisibilityTest.P3Tma.csv";
  std::ofstream csv_out(kCsvFile);
  ASSERT_TRUE(csv_out.is_open()) << "Failed to open CSV output file: "
                                 << kCsvFile;
  csv_out
      << "tile_bytes,warmup_iterations,measured_iterations,clock_overhead,"
         "data_sample_count,data_min,data_p10,data_median,data_p90,data_max,"
         "data_iters_min,data_iters_p10,data_iters_median,data_iters_p90,"
         "data_iters_max,barrier_sample_count,barrier_min,barrier_p10,"
         "barrier_median,barrier_p90,barrier_max,barrier_control_sample_count,"
         "barrier_control_min,barrier_control_p10,barrier_control_median,"
         "barrier_control_p90,barrier_control_max,barrier_iters_min,"
         "barrier_iters_p10,barrier_iters_median,barrier_iters_p90,"
         "barrier_iters_max,barrier_extra_median\n";

  printf("\n=== P3_tma Data Visibility (T2c) ===\n\n");
  printf("┌──────────┬─────────┬────────────┬────────────┬────────────┬────────────┐\n");
  printf("│ Tile B   │ Samples │ Data Min   │ Data Med   │ Data Max   │ Iter Med   │\n");
  printf("├──────────┼─────────┼────────────┼────────────┼────────────┼────────────┤\n");

  for (const uint32_t bytes : tile_bytes) {
    const size_t shared_memory_bytes =
        kTmaTensorMapBytes + 2 * static_cast<size_t>(bytes) + 64;

    const auto summary = measure_tma_visibility(
        [&] {
          launch_tma_data_visibility(bytes, shared_memory_bytes, source_ptr(),
                                     global_scratch_ptr(), cycle_start_ptr(),
                                     cycle_end_ptr(), result_ptr());
        },
        [&] {
          launch_tma_barrier_visibility(bytes, shared_memory_bytes,
                                        source_ptr(), global_scratch_ptr(),
                                        cycle_start_ptr(), cycle_end_ptr(),
                                        result_ptr());
        },
        [&](uint32_t expected_iterations) {
          launch_tma_barrier_visibility_control(bytes, shared_memory_bytes,
                                                expected_iterations,
                                                cycle_start_ptr(),
                                                cycle_end_ptr(), result_ptr());
        },
        options, clock_overhead);

    EXPECT_EQ(summary.data_raw.sample_count, options.measured_iterations);
    EXPECT_EQ(summary.barrier_raw.sample_count, options.measured_iterations);
    EXPECT_EQ(summary.barrier_control.sample_count, options.measured_iterations);
    print_tma_data_visibility_summary(bytes, summary,
                                      options.measured_iterations);
    export_tma_visibility_csv_row(csv_out, bytes, summary, options);
    summaries.push_back(summary);
  }
  printf("└──────────┴─────────┴────────────┴────────────┴────────────┴────────────┘\n");

  printf("\n=== P3_tma Barrier Visibility (T2c) ===\n\n");
  printf("┌──────────┬─────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐\n");
  printf("│ Tile B   │ Samples │ Bar Min    │ Bar Med    │ Bar Max    │ Ctrl Med   │ Iter Med   │ Extra Med  │ Clock OH   │\n");
  printf("├──────────┼─────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
  for (size_t i = 0; i < tile_bytes.size(); ++i) {
    print_tma_barrier_visibility_summary(tile_bytes[i], summaries[i],
                                         options.measured_iterations);
  }
  printf("└──────────┴─────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n");
  csv_out.close();
  printf("Results exported to: %s\n", kCsvFile);
}
