/**
 * MMA ILP Issue Gap Benchmark
 *
 * Measures MMA instruction execution cycles under different ILP levels to
 * observe tensor core pipeline behavior.
 *
 * Principle:
 * - By varying mma_count, measure total_cycles = slope * mma_count + intercept
 * - Slope = cycles/iteration. For ILP=N, effective_cycles_per_mma = slope / N
 * - Intercept = fixed overhead + completion latency of the last MMA
 *
 * Design points:
 * - Uses templates and #pragma unroll for compiler-driven ILP unrolling
 * - Each ILP chain uses independent accumulator arrays, no dependencies between
 * chains
 * - Dependencies exist within a chain (D is both output and accumulator input)
 * - All MMA variants are tested via TYPED_TEST_SUITE (one build, all variants)
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <vector>

#include "common/microbench_utils.cuh"

// ============================================================================
// MMA Instruction Traits (Strategy Pattern)
// ============================================================================

// Strategy 1: FP16 Accumulate to FP32 (M16N8K16) - Default
// Registers: A=4, B=2, C=4
struct MmaOp_F16_M16N8K16 {
  static const char* name() { return "F16_M16N8K16"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 16;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 209.5;
  static const char* throughput_unit() { return "TFLOPS"; }
  static __device__ __forceinline__ void exec(float C[4], const unsigned A[],
                                              const unsigned B[]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
  }
};

// Strategy 2: TF32 Accumulate to FP32 (M16N8K8)
// Registers: A=4, B=2, C=4
struct MmaOp_TF32_M16N8K8 {
  static const char* name() { return "TF32_M16N8K8"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 8;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 104.8;
  static const char* throughput_unit() { return "TFLOPS"; }
  static __device__ __forceinline__ void exec(float C[4], const unsigned A[],
                                              const unsigned B[]) {
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#endif
  }
};

// Strategy 3: Int8 Accumulate to Int32 (M16N8K32)
// Registers: A=4 (16B), B=2 (8B), C=4 (16B)
// Note: C array in kernel is float, so we punish type safety slightly here
// using reinterpretation for simplified benchmarking code structure.
struct MmaOp_S8_M16N8K32 {
  static const char* name() { return "S8_M16N8K32"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 32;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 838.0;
  static const char* throughput_unit() { return "TOPS"; }
  static __device__ __forceinline__ void exec(float C_as_float[4],
                                              const unsigned A[],
                                              const unsigned B[]) {
#if __CUDA_ARCH__ >= 750 || !defined(__CUDA_ARCH__)
    int* C = reinterpret_cast<int*>(C_as_float);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#endif
  }
};

// Strategy 4: FP16 Accumulate to FP32 (M16N8K8)
// Registers: A=2, B=1, C=4
struct MmaOp_F16_M16N8K8 {
  static const char* name() { return "F16_M16N8K8"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 8;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 209.5;
  static const char* throughput_unit() { return "TFLOPS"; }
  static __device__ __forceinline__ void exec(float C[4], const unsigned A[],
                                              const unsigned B[]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
  }
};

// Strategy 5: BF16 Accumulate to FP32 (M16N8K8)
// Registers: A=2, B=1, C=4
struct MmaOp_BF16_M16N8K8 {
  static const char* name() { return "BF16_M16N8K8"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 8;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 209.5;
  static const char* throughput_unit() { return "TFLOPS"; }
  static __device__ __forceinline__ void exec(float C[4], const unsigned A[],
                                              const unsigned B[]) {
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
#endif
  }
};

// Strategy 6: TF32 Accumulate to FP32 (M16N8K4)
// Registers: A=2, B=1, C=4
struct MmaOp_TF32_M16N8K4 {
  static const char* name() { return "TF32_M16N8K4"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 4;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 104.8;
  static const char* throughput_unit() { return "TFLOPS"; }
  static __device__ __forceinline__ void exec(float C[4], const unsigned A[],
                                              const unsigned B[]) {
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
#endif
  }
};

// Strategy 7: Int8 Accumulate to Int32 (M16N8K16)
// Registers: A=2, B=1, C=4
struct MmaOp_S8_M16N8K16 {
  static const char* name() { return "S8_M16N8K16"; }
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 16;
  static constexpr double kOpsPerMma = 2.0 * kM * kN * kK;
  static constexpr double kWhitepaperBoostClockMHz = 2407.0;
  static constexpr double kWhitepaperDensePeakAtBoost = 838.0;
  static const char* throughput_unit() { return "TOPS"; }
  static __device__ __forceinline__ void exec(float C_as_float[4],
                                              const unsigned A[],
                                              const unsigned B[]) {
#if __CUDA_ARCH__ >= 750 || !defined(__CUDA_ARCH__)
    int* C = reinterpret_cast<int*>(C_as_float);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
#endif
  }
};

// ============================================================================
// All MMA Variants Type List
// ============================================================================

using AllMmaOps = ::testing::Types<MmaOp_F16_M16N8K16, MmaOp_F16_M16N8K8,
                                   MmaOp_BF16_M16N8K8, MmaOp_TF32_M16N8K8,
                                   MmaOp_TF32_M16N8K4, MmaOp_S8_M16N8K32,
                                   MmaOp_S8_M16N8K16>;

// ============================================================================
// Compiler-Driven Loop Unrolling for MMA Instructions
// ============================================================================

template <typename MmaOp, int ILP>
__device__ __forceinline__ void execute_parallel_mmas(const unsigned* A_frag,
                                                      const unsigned* B_frag,
                                                      float C_frag[][4]) {
#pragma unroll
  for (int i = 0; i < ILP; i++) {
    MmaOp::exec(C_frag[i], A_frag, B_frag);
  }
}

// ============================================================================
// MMA ILP Measurement Kernel Template
// ============================================================================

template <typename MmaOp, int ILP>
__global__ void mma_ilp_kernel(uint64_t* cycle_start, uint64_t* cycle_end,
                               float* D_out, int mma_count) {
  // Only first warp participates
  if (threadIdx.x >= 32) return;
  int lane = threadIdx.x;

  unsigned A_frag[4] = {0x3C003C00, 0x3C003C00, 0x3C003C00, 0x3C003C00};
  unsigned B_frag[2] = {0x3C003C00, 0x3C003C00};
  float C_frag[ILP][4];

// Initialize all ILP chains to zero
#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    C_frag[chain][0] = 0.0f;
    C_frag[chain][1] = 0.0f;
    C_frag[chain][2] = 0.0f;
    C_frag[chain][3] = 0.0f;
  }

  __syncwarp();

  uint64_t start;
  if (lane == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    *cycle_start = start;
  }
  __syncwarp();

#pragma unroll 1
  for (int iter = 0; iter < mma_count; iter++) {
    execute_parallel_mmas<MmaOp, ILP>(A_frag, B_frag, C_frag);
  }

  __syncwarp();
  uint64_t end;
  if (lane == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_end = end;
  }

  float final_D[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    final_D[0] += C_frag[chain][0];
    final_D[1] += C_frag[chain][1];
    final_D[2] += C_frag[chain][2];
    final_D[3] += C_frag[chain][3];
  }

  if (lane == 0) {
    D_out[0] = final_D[0];
    D_out[1] = final_D[1];
    D_out[2] = final_D[2];
    D_out[3] = final_D[3];
  }
}

// ============================================================================
// Kernel Dispatcher
// ============================================================================

template <typename MmaOp>
inline bool launch_ilp_kernel(int ilp, uint64_t* cycle_start,
                              uint64_t* cycle_end, float* D_out,
                              int mma_count) {
  switch (ilp) {
#define DISPATCH_ILP(value)                                          \
  case value:                                                        \
    mma_ilp_kernel<MmaOp, value>                                     \
        <<<1, 32>>>(cycle_start, cycle_end, D_out, mma_count);       \
    return true;
    DISPATCH_ILP(1)
    DISPATCH_ILP(2)
    DISPATCH_ILP(4)
    DISPATCH_ILP(8)
    DISPATCH_ILP(16)
    DISPATCH_ILP(32)
    DISPATCH_ILP(64)
#undef DISPATCH_ILP
    default:
      return false;
  }
}

// ============================================================================
// Multi-Warp MMA Kernel
// ============================================================================

template <typename MmaOp, int ILP>
__global__ void mma_multi_warp_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, float* D_out,
                                      int mma_count, int num_warps) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  if (warp_id >= num_warps) return;

  unsigned A_frag[4] = {0x3C003C00, 0x3C003C00, 0x3C003C00, 0x3C003C00};
  unsigned B_frag[2] = {0x3C003C00, 0x3C003C00};
  float C_frag[ILP][4];

#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    C_frag[chain][0] = 0.0f;
    C_frag[chain][1] = 0.0f;
    C_frag[chain][2] = 0.0f;
    C_frag[chain][3] = 0.0f;
  }

  __syncthreads();

  uint64_t start;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    *cycle_start = start;
  }
  __syncthreads();

#pragma unroll 1
  for (int iter = 0; iter < mma_count; iter++) {
    execute_parallel_mmas<MmaOp, ILP>(A_frag, B_frag, C_frag);
  }

  __syncthreads();

  uint64_t end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_end = end;
  }

  float final_D[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    final_D[0] += C_frag[chain][0];
    final_D[1] += C_frag[chain][1];
    final_D[2] += C_frag[chain][2];
    final_D[3] += C_frag[chain][3];
  }

  if (lane == 0 && warp_id < num_warps) {
    D_out[warp_id * 4 + 0] = final_D[0];
    D_out[warp_id * 4 + 1] = final_D[1];
    D_out[warp_id * 4 + 2] = final_D[2];
    D_out[warp_id * 4 + 3] = final_D[3];
  }
}

// Multi-warp kernel dispatcher
template <typename MmaOp>
inline bool launch_multi_warp_kernel(int ilp, int num_warps,
                                     uint64_t* cycle_start, uint64_t* cycle_end,
                                     float* D_out, int mma_count) {
  int threads = num_warps * 32;
  switch (ilp) {
#define DISPATCH_MW(value)                                                     \
  case value:                                                                  \
    mma_multi_warp_kernel<MmaOp, value>                                        \
        <<<1, threads>>>(cycle_start, cycle_end, D_out, mma_count, num_warps); \
    return true;
    DISPATCH_MW(1)
    DISPATCH_MW(2)
    DISPATCH_MW(4)
    DISPATCH_MW(8)
    DISPATCH_MW(16)
    DISPATCH_MW(32)
    DISPATCH_MW(64)
#undef DISPATCH_MW
    default:
      return false;
  }
}

// ============================================================================
// Peak Throughput Kernel
// ============================================================================

template <typename MmaOp, int ILP>
__global__ void mma_peak_throughput_kernel(float* D_out, uint64_t* block_cycles,
                                           int mma_count) {
  int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  int global_warp = global_thread / 32;
  int lane = threadIdx.x % 32;
  uint64_t start_cycles = 0;

  unsigned A_frag[4] = {0x3C003C00, 0x3C003C00, 0x3C003C00, 0x3C003C00};
  unsigned B_frag[2] = {0x3C003C00, 0x3C003C00};
  float C_frag[ILP][4];

#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    C_frag[chain][0] = 0.0f;
    C_frag[chain][1] = 0.0f;
    C_frag[chain][2] = 0.0f;
    C_frag[chain][3] = 0.0f;
  }

  __syncthreads();
  if (threadIdx.x == 0 && block_cycles != nullptr) {
    start_cycles = clock64();
  }
  __syncthreads();

#pragma unroll 1
  for (int iter = 0; iter < mma_count; iter++) {
    execute_parallel_mmas<MmaOp, ILP>(A_frag, B_frag, C_frag);
  }

  float final_D[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int chain = 0; chain < ILP; chain++) {
    final_D[0] += C_frag[chain][0];
    final_D[1] += C_frag[chain][1];
    final_D[2] += C_frag[chain][2];
    final_D[3] += C_frag[chain][3];
  }

  if (lane == 0) {
    int out_idx = global_warp * 4;
    D_out[out_idx + 0] = final_D[0];
    D_out[out_idx + 1] = final_D[1];
    D_out[out_idx + 2] = final_D[2];
    D_out[out_idx + 3] = final_D[3];
  }

  __syncthreads();
  if (threadIdx.x == 0 && block_cycles != nullptr) {
    block_cycles[blockIdx.x] = clock64() - start_cycles;
  }
}

template <typename MmaOp>
inline bool launch_peak_throughput_kernel(int ilp, int blocks,
                                          int threads_per_block, float* D_out,
                                          uint64_t* block_cycles,
                                          int mma_count) {
  switch (ilp) {
#define DISPATCH_PEAK(value)                                                     \
  case value:                                                                    \
    mma_peak_throughput_kernel<MmaOp, value>                                     \
        <<<blocks, threads_per_block>>>(D_out, block_cycles, mma_count);         \
    return true;
    DISPATCH_PEAK(1)
    DISPATCH_PEAK(2)
    DISPATCH_PEAK(4)
    DISPATCH_PEAK(8)
    DISPATCH_PEAK(16)
    DISPATCH_PEAK(32)
#undef DISPATCH_PEAK
    default:
      return false;
  }
}

template <typename MmaOp>
inline int query_peak_blocks_per_sm(int ilp, int threads_per_block) {
  int blocks_per_sm = 0;
  cudaError_t err = cudaSuccess;

  switch (ilp) {
#define QUERY_OCCUPANCY(value)                                                   \
  case value:                                                                    \
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(                         \
        &blocks_per_sm, mma_peak_throughput_kernel<MmaOp, value>,                \
        threads_per_block, 0);                                                   \
    break;
    QUERY_OCCUPANCY(1)
    QUERY_OCCUPANCY(2)
    QUERY_OCCUPANCY(4)
    QUERY_OCCUPANCY(8)
    QUERY_OCCUPANCY(16)
    QUERY_OCCUPANCY(32)
#undef QUERY_OCCUPANCY
    default:
      return 0;
  }

  return (err == cudaSuccess) ? blocks_per_sm : 0;
}

struct PeakThroughputResult {
  const char* name;
  const char* unit;
  int best_ilp;
  int blocks_per_sm;
  int total_warps;
  double median_us;
  double measured_clock_ghz;
  double whitepaper_boost_clock_mhz;
  double whitepaper_dense_peak_at_boost;
  double current_theoretical_peak;
  double efficiency_pct;
  double throughput;
  bool valid;

  PeakThroughputResult()
      : name(nullptr),
        unit("TFLOPS"),
        best_ilp(0),
        blocks_per_sm(0),
        total_warps(0),
        median_us(0.0),
        measured_clock_ghz(0.0),
        whitepaper_boost_clock_mhz(0.0),
        whitepaper_dense_peak_at_boost(0.0),
        current_theoretical_peak(0.0),
        efficiency_pct(0.0),
        throughput(0.0),
        valid(false) {}
};

static double get_reference_sm_clock_mhz_override() {
  const char* env = std::getenv("FLASHGPU_MMA_SM_CLOCK_MHZ");
  if (!env || env[0] == '\0') return 0.0;

  double mhz = std::atof(env);
  return (mhz > 0.0) ? mhz : 0.0;
}

static double resolve_reference_sm_clock_mhz(const cudaDeviceProp& prop) {
  double override_mhz = get_reference_sm_clock_mhz_override();
  if (override_mhz > 0.0) return override_mhz;

  // CUDA reports clockRate in kHz.
  return static_cast<double>(prop.clockRate) / 1000.0;
}

static const char* resolve_reference_sm_clock_source() {
  return (get_reference_sm_clock_mhz_override() > 0.0)
             ? "env FLASHGPU_MMA_SM_CLOCK_MHZ"
             : "cudaDeviceProp.clockRate";
}

static double median_of(std::vector<double> values) {
  if (values.empty()) return 0.0;

  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

template <typename MmaOp>
static bool run_peak_config(int ilp, int grid_blocks, int threads_per_block,
                            int total_warps, int mma_count, int warmup,
                            int iterations, bool collect_measured_clock,
                            PeakThroughputResult* out) {
  size_t out_elems = static_cast<size_t>(total_warps) * 4;
  float* d_out = nullptr;
  if (cudaMalloc(&d_out, out_elems * sizeof(float)) != cudaSuccess) return false;

  uint64_t* d_block_cycles = nullptr;
  if (collect_measured_clock &&
      cudaMalloc(&d_block_cycles,
                 static_cast<size_t>(grid_blocks) * sizeof(uint64_t)) !=
          cudaSuccess) {
    cudaFree(d_out);
    return false;
  }

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  if (cudaEventCreate(&start_event) != cudaSuccess) {
    cudaFree(d_out);
    if (d_block_cycles) cudaFree(d_block_cycles);
    return false;
  }
  if (cudaEventCreate(&stop_event) != cudaSuccess) {
    cudaEventDestroy(start_event);
    cudaFree(d_out);
    if (d_block_cycles) cudaFree(d_block_cycles);
    return false;
  }

  bool launch_failed = false;
  for (int i = 0; i < warmup; i++) {
    if (!launch_peak_throughput_kernel<MmaOp>(ilp, grid_blocks,
                                              threads_per_block, d_out,
                                              d_block_cycles, mma_count)) {
      launch_failed = true;
      break;
    }
  }
  if (!launch_failed && cudaDeviceSynchronize() != cudaSuccess) {
    launch_failed = true;
  }

  std::vector<uint64_t> elapsed_ns;
  std::vector<double> measured_clock_ghz;
  std::vector<uint64_t> block_cycles(
      collect_measured_clock ? static_cast<size_t>(grid_blocks) : 0);

  for (int i = 0; !launch_failed && i < iterations; i++) {
    if (cudaEventRecord(start_event) != cudaSuccess) {
      launch_failed = true;
      break;
    }
    if (!launch_peak_throughput_kernel<MmaOp>(ilp, grid_blocks,
                                              threads_per_block, d_out,
                                              d_block_cycles, mma_count)) {
      launch_failed = true;
      break;
    }
    if (cudaEventRecord(stop_event) != cudaSuccess ||
        cudaEventSynchronize(stop_event) != cudaSuccess) {
      launch_failed = true;
      break;
    }

    if (cudaPeekAtLastError() != cudaSuccess) {
      launch_failed = true;
      break;
    }

    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start_event, stop_event) !=
        cudaSuccess) {
      launch_failed = true;
      break;
    }

    uint64_t elapsed_ns_i = static_cast<uint64_t>(std::llround(elapsed_ms * 1e6));
    elapsed_ns.push_back(elapsed_ns_i);

    if (collect_measured_clock) {
      if (cudaMemcpy(block_cycles.data(), d_block_cycles,
                     block_cycles.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        launch_failed = true;
        break;
      }

      uint64_t max_cycles = *std::max_element(block_cycles.begin(),
                                              block_cycles.end());
      measured_clock_ghz.push_back(
          (elapsed_ns_i > 0)
              ? (static_cast<double>(max_cycles) /
                 static_cast<double>(elapsed_ns_i))
              : 0.0);
    }
  }

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFree(d_out);
  if (d_block_cycles) cudaFree(d_block_cycles);

  if (launch_failed || elapsed_ns.empty()) return false;

  BenchmarkStats stats = calculate_stats(elapsed_ns, true);
  if (stats.median <= 0.0) return false;

  double elapsed_s = stats.median * 1e-9;
  double total_ops = MmaOp::kOpsPerMma * static_cast<double>(mma_count) *
                     static_cast<double>(ilp) *
                     static_cast<double>(total_warps);
  out->median_us = stats.median * 1e-3;
  out->throughput = total_ops / elapsed_s / 1e12;
  out->measured_clock_ghz =
      collect_measured_clock ? median_of(measured_clock_ghz) : 0.0;
  return true;
}

template <typename MmaOp>
static PeakThroughputResult measure_variant_peak_throughput() {
  constexpr int kThreadsPerBlock = 256;
  constexpr int kSweepWarmup = 3;
  constexpr int kSweepIterations = 7;
  constexpr int kRerunWarmup = 5;
  constexpr int kRerunIterations = 31;
  constexpr int kMmaCount = 2048;
  const int ilp_values[] = {1, 2, 4, 8, 16, 32};

  PeakThroughputResult result;
  result.name = MmaOp::name();
  result.unit = MmaOp::throughput_unit();
  result.whitepaper_boost_clock_mhz = MmaOp::kWhitepaperBoostClockMHz;
  result.whitepaper_dense_peak_at_boost = MmaOp::kWhitepaperDensePeakAtBoost;

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return result;

  int best_ilp = 0;
  int best_blocks_per_sm = 0;
  int best_grid_blocks = 0;
  int best_total_warps = 0;
  double best_throughput = 0.0;
  bool found_best = false;

  for (int ilp : ilp_values) {
    int blocks_per_sm = query_peak_blocks_per_sm<MmaOp>(ilp, kThreadsPerBlock);
    if (blocks_per_sm <= 0) continue;

    int grid_blocks = std::max(1, prop.multiProcessorCount * blocks_per_sm);
    int total_warps = grid_blocks * (kThreadsPerBlock / 32);
    PeakThroughputResult sweep_result;
    if (!run_peak_config<MmaOp>(ilp, grid_blocks, kThreadsPerBlock, total_warps,
                                kMmaCount, kSweepWarmup, kSweepIterations,
                                false, &sweep_result)) {
      continue;
    }

    if (!found_best || sweep_result.throughput > best_throughput) {
      best_ilp = ilp;
      best_blocks_per_sm = blocks_per_sm;
      best_grid_blocks = grid_blocks;
      best_total_warps = total_warps;
      best_throughput = sweep_result.throughput;
      found_best = true;
    }
  }

  if (!found_best) return result;

  result.best_ilp = best_ilp;
  result.blocks_per_sm = best_blocks_per_sm;
  result.total_warps = best_total_warps;

  if (!run_peak_config<MmaOp>(best_ilp, best_grid_blocks, kThreadsPerBlock,
                              best_total_warps, kMmaCount, kRerunWarmup,
                              kRerunIterations, true, &result)) {
    return result;
  }

  result.valid = true;
  return result;
}

// ============================================================================
// Typed Test Fixture (parameterized on MmaOp)
// ============================================================================

template <typename MmaOp>
class MMAIssueTest : public ::testing::Test {
 protected:
  uint64_t* d_cycle_start;
  uint64_t* d_cycle_end;
  float* d_out;

  void SetUp() override {
    cudaSetDevice(0);

    cudaMalloc(&d_cycle_start, sizeof(uint64_t));
    cudaMalloc(&d_cycle_end, sizeof(uint64_t));
    // Allocate space for up to 32 warps
    cudaMalloc(&d_out, 32 * 4 * sizeof(float));
  }

  void TearDown() override {
    cudaFree(d_cycle_start);
    cudaFree(d_cycle_end);
    cudaFree(d_out);
  }

  // Run measurement for a specific ILP and mma_count
  uint64_t run_measurement(int ilp, int mma_count, uint64_t clock_overhead) {
    launch_ilp_kernel<MmaOp>(ilp, d_cycle_start, d_cycle_end, d_out,
                             mma_count);
    cudaDeviceSynchronize();

    uint64_t start, end;
    cudaMemcpy(&start, d_cycle_start, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, d_cycle_end, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t raw_cycles = end - start;
    return (raw_cycles > clock_overhead) ? (raw_cycles - clock_overhead)
                                         : raw_cycles;
  }
};

TYPED_TEST_SUITE(MMAIssueTest, AllMmaOps);

// ============================================================================
// ILP Test - runs for every MMA variant automatically
// ============================================================================

TYPED_TEST(MMAIssueTest, ILPMinimal) {
  using MmaOp = TypeParam;
  const char* op_name = MmaOp::name();

  printf("\n=== MMA ILP Issue Gap Minimal Test [%s] ===\n\n", op_name);

  uint64_t clock_overhead = 0;

  const int mma_count = 16;
  const int warmup = 3;
  const int iterations = 10;

  std::vector<int> ilp_values = {1, 2, 4, 8};

  // Export file named per variant
  char filename[256];
  snprintf(filename, sizeof(filename), "MMAIssueTest.ILPMinimal.%s.txt",
           op_name);
  std::ofstream out(filename);
  out << "MMA Variant: " << op_name << "\n";
  out << "┌─────────┬────────────┬──────────────┬────────────────┬─────────────"
         "────┐\n";
  out << "│   ILP   │ Iterations │ Total MMAs   │  Total Cycles  │  Cycles/MMA "
         "    │\n";
  out << "├─────────┼────────────┼──────────────┼────────────────┼─────────────"
         "────┤\n";

  printf(
      "┌─────────┬────────────┬──────────────┬────────────────┬────────────────"
      "─┐\n");
  printf(
      "│   ILP   │ Iterations │ Total MMAs   │  Total Cycles  │  Cycles/MMA    "
      " │\n");
  printf(
      "├─────────┼────────────┼──────────────┼────────────────┼────────────────"
      "─┤\n");

  for (int ilp : ilp_values) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
      launch_ilp_kernel<MmaOp>(ilp, this->d_cycle_start, this->d_cycle_end,
                               this->d_out, mma_count);
    }
    cudaDeviceSynchronize();

    // Measure
    std::vector<uint64_t> measurements;
    for (int i = 0; i < iterations; i++) {
      uint64_t cycles = this->run_measurement(ilp, mma_count, clock_overhead);
      measurements.push_back(cycles);
    }

    // Take median
    std::sort(measurements.begin(), measurements.end());
    uint64_t median_cycles = measurements[iterations / 2];

    int total_mmas = mma_count * ilp;
    double cycles_per_mma = (double)median_cycles / total_mmas;

    printf(
        "│  %3d    │    %4d    │    %6d    │     %6lu     │     %7.2f     │\n",
        ilp, mma_count, total_mmas, median_cycles, cycles_per_mma);

    char buf[256];
    snprintf(
        buf, sizeof(buf),
        "│  %3d    │    %4d    │    %6d    │     %6lu     │     %7.2f     │\n",
        ilp, mma_count, total_mmas, median_cycles, cycles_per_mma);
    out << buf;
  }
  printf(
      "└─────────┴────────────┴──────────────┴────────────────┴────────────────"
      "─┘\n");

  out << "└─────────┴────────────┴──────────────┴────────────────┴─────────────"
         "────┘\n";
  out.close();
  printf("\nResults exported to: %s\n", filename);
}

// ============================================================================
// Multi-Warp Test - runs for every MMA variant automatically
// ============================================================================

TYPED_TEST(MMAIssueTest, MultiWarpMinimal) {
  using MmaOp = TypeParam;
  const char* op_name = MmaOp::name();

  printf("\n=== Multi-Warp MMA Test [%s] ===\n\n", op_name);

  uint64_t clock_overhead = 0;

  const int ilp = 1;
  const int mma_count = 1000;
  std::vector<int> warp_counts = {1, 2, 4, 8, 16, 32};

  char filename[256];
  snprintf(filename, sizeof(filename), "MMAIssueTest.MultiWarpMinimal.%s.txt",
           op_name);
  std::ofstream out(filename);
  out << "MMA Variant: " << op_name << "\n";
  out << "┌─────────┬────────────┬────────────┬────────────┬───────────┐\n";
  out << "│  Warps  │ Total MMAs │   Cycles   │ Cycles/MMA │  Speedup  │\n";
  out << "├─────────┼────────────┼────────────┼────────────┼───────────┤\n";

  printf("┌─────────┬────────────┬────────────┬────────────┬───────────┐\n");
  printf("│  Warps  │ Total MMAs │   Cycles   │ Cycles/MMA │  Speedup  │\n");
  printf("├─────────┼────────────┼────────────┼────────────┼───────────┤\n");

  double baseline = 0;

  for (int num_warps : warp_counts) {
    launch_multi_warp_kernel<MmaOp>(ilp, num_warps, this->d_cycle_start,
                                    this->d_cycle_end, this->d_out, mma_count);
    cudaDeviceSynchronize();

    uint64_t start, end;
    cudaMemcpy(&start, this->d_cycle_start, sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, this->d_cycle_end, sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    uint64_t cycles = (end > start + clock_overhead)
                          ? (end - start - clock_overhead)
                          : (end - start);
    int total_mmas = mma_count * num_warps;
    double cyc_per_mma = (double)cycles / total_mmas;

    if (num_warps == 1) baseline = cyc_per_mma;
    double speedup = (cyc_per_mma > 0) ? (baseline / cyc_per_mma) : 0.0;

    printf("│   %3d   │   %6d   │  %8lu  │   %7.2f  │   %5.2fx  │\n",
           num_warps, total_mmas, cycles, cyc_per_mma, speedup);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "│   %3d   │   %6d   │  %8lu  │   %7.2f  │   %5.2fx  │\n",
             num_warps, total_mmas, cycles, cyc_per_mma, speedup);
    out << buf;
  }
  printf("└─────────┴────────────┴────────────┴────────────┴───────────┘\n");

  out << "└─────────┴────────────┴────────────┴────────────┴───────────┘\n";
  out.close();
  printf(
      "\nExpected: 1->2 warps speedup ~= 2x (if 2 independent Tensor Cores)\n");
  printf("Results exported to: %s\n", filename);
}

// ============================================================================
// Summary Table - all variants side-by-side (ILP=8, single test)
// ============================================================================

// Measure median cycles for a given MmaOp, ILP, and mma_count
template <typename MmaOp>
static uint64_t measure_median_cycles(int ilp, int mma_count, int iterations,
                                      uint64_t* d_cycle_start,
                                      uint64_t* d_cycle_end, float* d_out) {
  std::vector<uint64_t> measurements;
  for (int i = 0; i < iterations; i++) {
    launch_ilp_kernel<MmaOp>(ilp, d_cycle_start, d_cycle_end, d_out, mma_count);
    cudaDeviceSynchronize();

    uint64_t start, end;
    cudaMemcpy(&start, d_cycle_start, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, d_cycle_end, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    measurements.push_back(end - start);
  }
  std::sort(measurements.begin(), measurements.end());
  return measurements[iterations / 2];
}

// Use linear regression across multiple mma_counts to extract slope
// (= cycles per iteration = ILP * cycles_per_mma), removing fixed overhead.
//
// Model: total_cycles = slope * mma_count + intercept
//   slope = ILP * cycles_per_mma
//   intercept = fixed overhead (clock read, syncwarp, last MMA completion)
template <typename MmaOp>
static double measure_variant_cycles_per_mma(uint64_t* d_cycle_start,
                                             uint64_t* d_cycle_end,
                                             float* d_out) {
  const int ilp = 8;
  const int warmup = 5;
  const int iterations = 20;
  // Multiple mma_counts for linear regression
  const int counts[] = {64, 128, 256, 512, 1024};
  const int num_points = sizeof(counts) / sizeof(counts[0]);

  // Warmup with longest chain
  for (int i = 0; i < warmup; i++) {
    launch_ilp_kernel<MmaOp>(ilp, d_cycle_start, d_cycle_end, d_out,
                             counts[num_points - 1]);
  }
  cudaDeviceSynchronize();

  // Collect (mma_count, median_cycles) data points
  double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
  for (int p = 0; p < num_points; p++) {
    int mc = counts[p];
    uint64_t median = measure_median_cycles<MmaOp>(ilp, mc, iterations,
                                                   d_cycle_start, d_cycle_end,
                                                   d_out);
    double x = (double)mc;
    double y = (double)median;
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
  }

  // Linear regression: slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
  double n = (double)num_points;
  double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

  // slope = cycles per iteration = ILP * cycles_per_mma
  return slope / ilp;
}

TEST(MMAIssueSummary, AllVariants) {
  cudaSetDevice(0);

  uint64_t* d_cycle_start;
  uint64_t* d_cycle_end;
  float* d_out;
  cudaMalloc(&d_cycle_start, sizeof(uint64_t));
  cudaMalloc(&d_cycle_end, sizeof(uint64_t));
  cudaMalloc(&d_out, 32 * 4 * sizeof(float));

  struct Result {
    const char* name;
    double cycles_per_mma;
  };

  Result results[] = {
      {MmaOp_F16_M16N8K16::name(),
       measure_variant_cycles_per_mma<MmaOp_F16_M16N8K16>(d_cycle_start,
                                                           d_cycle_end, d_out)},
      {MmaOp_F16_M16N8K8::name(),
       measure_variant_cycles_per_mma<MmaOp_F16_M16N8K8>(d_cycle_start,
                                                          d_cycle_end, d_out)},
      {MmaOp_BF16_M16N8K8::name(),
       measure_variant_cycles_per_mma<MmaOp_BF16_M16N8K8>(d_cycle_start,
                                                           d_cycle_end, d_out)},
      {MmaOp_TF32_M16N8K8::name(),
       measure_variant_cycles_per_mma<MmaOp_TF32_M16N8K8>(d_cycle_start,
                                                           d_cycle_end, d_out)},
      {MmaOp_TF32_M16N8K4::name(),
       measure_variant_cycles_per_mma<MmaOp_TF32_M16N8K4>(d_cycle_start,
                                                           d_cycle_end, d_out)},
      {MmaOp_S8_M16N8K32::name(),
       measure_variant_cycles_per_mma<MmaOp_S8_M16N8K32>(d_cycle_start,
                                                          d_cycle_end, d_out)},
      {MmaOp_S8_M16N8K16::name(),
       measure_variant_cycles_per_mma<MmaOp_S8_M16N8K16>(d_cycle_start,
                                                          d_cycle_end, d_out)},
  };

  int num_variants = sizeof(results) / sizeof(results[0]);

  printf("\n=== MMA Variant Summary (ILP=8, linear regression over mma_count=64..1024) ===\n\n");
  printf("┌──────────────────┬─────────────┐\n");
  printf("│  MMA Variant     │  Cycles/MMA │\n");
  printf("├──────────────────┼─────────────┤\n");

  std::ofstream out("MMAIssueTest.Summary.txt");
  out << "MMA Variant Summary (ILP=8, linear regression over mma_count=64..1024)\n";
  out << "┌──────────────────┬─────────────┐\n";
  out << "│  MMA Variant     │  Cycles/MMA │\n";
  out << "├──────────────────┼─────────────┤\n";

  for (int i = 0; i < num_variants; i++) {
    printf("│  %-15s │    %7.2f  │\n", results[i].name,
           results[i].cycles_per_mma);

    char buf[128];
    snprintf(buf, sizeof(buf), "│  %-15s │    %7.2f  │\n", results[i].name,
             results[i].cycles_per_mma);
    out << buf;
  }

  printf("└──────────────────┴─────────────┘\n");
  out << "└──────────────────┴─────────────┘\n";
  out.close();
  printf("\nResults exported to: MMAIssueTest.Summary.txt\n");

  cudaFree(d_cycle_start);
  cudaFree(d_cycle_end);
  cudaFree(d_out);
}

// ============================================================================
// Peak Throughput Summary - all variants side-by-side
// ============================================================================

TEST(MMAPeak, AllVariants) {
  constexpr const char* kOutputFile = "MMAPeak.Summary.txt";

  cudaSetDevice(0);

  cudaDeviceProp prop;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&prop, 0));

  PeakThroughputResult results[] = {
      measure_variant_peak_throughput<MmaOp_F16_M16N8K16>(),
      measure_variant_peak_throughput<MmaOp_F16_M16N8K8>(),
      measure_variant_peak_throughput<MmaOp_BF16_M16N8K8>(),
      measure_variant_peak_throughput<MmaOp_TF32_M16N8K8>(),
      measure_variant_peak_throughput<MmaOp_TF32_M16N8K4>(),
      measure_variant_peak_throughput<MmaOp_S8_M16N8K32>(),
      measure_variant_peak_throughput<MmaOp_S8_M16N8K16>(),
  };

  double reference_sm_mhz = resolve_reference_sm_clock_mhz(prop);
  const char* reference_clock_source = resolve_reference_sm_clock_source();
  int num_variants = sizeof(results) / sizeof(results[0]);
  for (int i = 0; i < num_variants; i++) {
    ASSERT_TRUE(results[i].valid)
        << "Failed to measure peak throughput for " << results[i].name;
    double measured_sm_mhz = results[i].measured_clock_ghz * 1000.0;
    results[i].current_theoretical_peak =
        results[i].whitepaper_dense_peak_at_boost * measured_sm_mhz /
        results[i].whitepaper_boost_clock_mhz;
    results[i].efficiency_pct =
        (results[i].current_theoretical_peak > 0.0)
            ? (results[i].throughput / results[i].current_theoretical_peak *
               100.0)
            : 0.0;
  }

  printf("\n=== MMA Peak Summary ===\n");
  printf("Device: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
  printf("Reference SM clock: %.2f MHz (%s)\n", reference_sm_mhz,
         reference_clock_source);
  printf("1 MMA = 2 * M * N * K ops; Meas GHz comes from block clock64 / "
         "cudaEvent time.\n");
  printf("Theo Now scales the RTX 5090 dense peak at 2407 MHz using Meas GHz.\n\n");

  printf("┌──────────────────┬─────────┬───────────┬────────────┬────────────┬──────────┬────────────┬────────────┬──────────┬────────┐\n");
  printf("│  MMA Variant     │ BestILP │ Blocks/SM │ TotalWarps │ Median us  │ Meas GHz │ Throughput │ Theo Now   │ %% Peak   │ Unit   │\n");
  printf("├──────────────────┼─────────┼───────────┼────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼────────┤\n");

  std::ofstream out(kOutputFile);
  out << "MMA Peak Summary\n";
  out << "Device: " << prop.name << " (SMs=" << prop.multiProcessorCount << ")\n";
  out << "Reference SM clock: " << reference_sm_mhz << " MHz ("
      << reference_clock_source << ")\n";
  out << "1 MMA = 2 * M * N * K ops; Meas GHz comes from block clock64 / "
         "cudaEvent time.\n";
  out << "Theo Now scales the RTX 5090 dense peak at 2407 MHz using Meas GHz.\n";
  out << "┌──────────────────┬─────────┬───────────┬────────────┬────────────┬──────────┬────────────┬────────────┬──────────┬────────┐\n";
  out << "│  MMA Variant     │ BestILP │ Blocks/SM │ TotalWarps │ Median us  │ Meas GHz │ Throughput │ Theo Now   │ % Peak   │ Unit   │\n";
  out << "├──────────────────┼─────────┼───────────┼────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼────────┤\n";

  for (int i = 0; i < num_variants; i++) {
    printf("│  %-15s │  %5d  │    %3d    │   %6d   │   %7.2f  │  %6.3f  │   %8.2f │   %8.2f │  %6.2f  │ %-6s │\n",
           results[i].name, results[i].best_ilp, results[i].blocks_per_sm,
           results[i].total_warps, results[i].median_us,
           results[i].measured_clock_ghz, results[i].throughput,
           results[i].current_theoretical_peak, results[i].efficiency_pct,
           results[i].unit);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "│  %-15s │  %5d  │    %3d    │   %6d   │   %7.2f  │  %6.3f  │   %8.2f │   %8.2f │  %6.2f  │ %-6s │\n",
             results[i].name, results[i].best_ilp, results[i].blocks_per_sm,
             results[i].total_warps, results[i].median_us,
             results[i].measured_clock_ghz, results[i].throughput,
             results[i].current_theoretical_peak, results[i].efficiency_pct,
             results[i].unit);
    out << buf;
  }

  printf("└──────────────────┴─────────┴───────────┴────────────┴────────────┴──────────┴────────────┴────────────┴──────────┴────────┘\n");
  printf("\nResults exported to: %s\n", kOutputFile);

  out << "└──────────────────┴─────────┴───────────┴────────────┴────────────┴──────────┴────────────┴────────────┴──────────┴────────┘\n";
  out.close();
}
