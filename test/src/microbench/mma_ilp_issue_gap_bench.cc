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
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <vector>

#include "microbench_utils.cuh"

// ============================================================================
// MMA Instruction Traits (Strategy Pattern)
// ============================================================================

// Strategy 1: FP16 Accumulate to FP32 (M16N8K16) - Default
// Registers: A=4, B=2, C=4
struct MmaOp_F16_M16N8K16 {
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

// SELECT ACTIVE MMA TYPE HERE
// using CurrentMmaOp = MmaOp_F16_M16N8K16; // 32cyles
// using CurrentMmaOp = MmaOp_F16_M16N8K8; // 16cycles
// using CurrentMmaOp = MmaOp_BF16_M16N8K8; // 16cycles
using CurrentMmaOp = MmaOp_TF32_M16N8K8;  // 32cycles
// using CurrentMmaOp = MmaOp_TF32_M16N8K4; // 16cycles
// using CurrentMmaOp = MmaOp_S8_M16N8K32; // 16cycles(?)
// using CurrentMmaOp = MmaOp_S8_M16N8K16; // 8cycles(?)

// ============================================================================
// Compiler-Driven Loop Unrolling for MMA Instructions
// ============================================================================

template <int ILP>
__device__ __forceinline__ void execute_parallel_mmas(const unsigned* A_frag,
                                                      const unsigned* B_frag,
                                                      float C_frag[][4]) {
#pragma unroll
  for (int i = 0; i < ILP; i++) {
    CurrentMmaOp::exec(C_frag[i], A_frag, B_frag);
  }
}

// ============================================================================
// MMA ILP Measurement Kernel Template
// ============================================================================

template <int ILP>
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
    execute_parallel_mmas<ILP>(A_frag, B_frag, C_frag);
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

inline bool launch_ilp_kernel(int ilp, uint64_t* cycle_start,
                              uint64_t* cycle_end, float* D_out,
                              int mma_count) {
  switch (ilp) {
#define DISPATCH_ILP(value)                                    \
  case value:                                                  \
    mma_ilp_kernel<value>                                      \
        <<<1, 32>>>(cycle_start, cycle_end, D_out, mma_count); \
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

template <int ILP>
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
    execute_parallel_mmas<ILP>(A_frag, B_frag, C_frag);
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
inline bool launch_multi_warp_kernel(int ilp, int num_warps,
                                     uint64_t* cycle_start, uint64_t* cycle_end,
                                     float* D_out, int mma_count) {
  int threads = num_warps * 32;
  switch (ilp) {
#define DISPATCH_MW(value)                                                     \
  case value:                                                                  \
    mma_multi_warp_kernel<value>                                               \
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
// Test Fixture
// ============================================================================

class MMAILPIssueGapTest : public ::testing::Test {
 protected:
  uint64_t* d_cycle_start;
  uint64_t* d_cycle_end;
  float* d_out;

  void SetUp() override {
    // Minimal setup
    cudaSetDevice(0);

    cudaMalloc(&d_cycle_start, sizeof(uint64_t));
    cudaMalloc(&d_cycle_end, sizeof(uint64_t));
    // Allocate space for up to 4 warps
    cudaMalloc(&d_out, 4 * 4 * sizeof(float));
  }

  void TearDown() override {
    cudaFree(d_cycle_start);
    cudaFree(d_cycle_end);
    cudaFree(d_out);
  }

  // Run measurement for a specific ILP and mma_count
  uint64_t run_measurement(int ilp, int mma_count, uint64_t clock_overhead) {
    launch_ilp_kernel(ilp, d_cycle_start, d_cycle_end, d_out, mma_count);
    cudaDeviceSynchronize();

    uint64_t start, end;
    cudaMemcpy(&start, d_cycle_start, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, d_cycle_end, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t raw_cycles = end - start;
    return (raw_cycles > clock_overhead) ? (raw_cycles - clock_overhead)
                                         : raw_cycles;
  }
};

// ============================================================================
// Quick Test - Basic validation
// ============================================================================

TEST_F(MMAILPIssueGapTest, ILPMinimal) {
  printf("\n=== MMA ILP Issue Gap Minimal Test ===\n\n");

  // Clock overhead measurement disabled for simplicity
  uint64_t clock_overhead = 0;
  printf("Clock64 overhead: %lu cycles (disabled)\n\n", clock_overhead);

  // Test different ILP levels with fixed mma_count
  const int mma_count = 16;  // iterations per kernel
  const int warmup = 3;
  const int iterations = 10;

  std::vector<int> ilp_values = {1, 2, 4, 8};

  // Open file for export
  std::ofstream out("MMAILPIssueGapTest.ILPMinimal.txt");
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
      launch_ilp_kernel(ilp, d_cycle_start, d_cycle_end, d_out, mma_count);
    }
    cudaDeviceSynchronize();

    // Measure
    std::vector<uint64_t> measurements;
    for (int i = 0; i < iterations; i++) {
      uint64_t cycles = run_measurement(ilp, mma_count, clock_overhead);
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

    // Write to file
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
  printf("\nResults exported to: MMAILPIssueGapTest.ILPMinimal.txt\n");
}

// ============================================================================
// Minimal Multi-Warp Test for GPGPU-Sim
// ============================================================================

TEST_F(MMAILPIssueGapTest, MultiWarpMinimal) {
  printf("\n=== Minimal Multi-Warp MMA Test (for GPGPU-Sim) ===\n\n");

  // Clock overhead measurement disabled for simplicity
  uint64_t clock_overhead = 0;
  printf("Clock overhead: %lu cycles (disabled)\n\n", clock_overhead);

  // Minimal parameters for fast simulation
  const int ilp = 1;
  const int mma_count =
      1000;  // Increased from 16 to 1000 to minimize overhead impact
  std::vector<int> warp_counts = {
      1,  2, 4, 8,
      16, 32};  // Removed 64, 128 as max threads per block is 1024 (32 warps)

  // Open file for export
  std::ofstream out("MMAILPIssueGapTest.MultiWarpMinimal.txt");
  out << "┌─────────┬────────────┬────────────┬────────────┬───────────┐\n";
  out << "│  Warps  │ Total MMAs │   Cycles   │ Cycles/MMA │  Speedup  │\n";
  out << "├─────────┼────────────┼────────────┼────────────┼───────────┤\n";

  printf("┌─────────┬────────────┬────────────┬────────────┬───────────┐\n");
  printf("│  Warps  │ Total MMAs │   Cycles   │ Cycles/MMA │  Speedup  │\n");
  printf("├─────────┼────────────┼────────────┼────────────┼───────────┤\n");

  double baseline = 0;

  for (int num_warps : warp_counts) {
    launch_multi_warp_kernel(ilp, num_warps, d_cycle_start, d_cycle_end, d_out,
                             mma_count);
    cudaDeviceSynchronize();

    uint64_t start, end;
    cudaMemcpy(&start, d_cycle_start, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, d_cycle_end, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t cycles = (end > start + clock_overhead)
                          ? (end - start - clock_overhead)
                          : (end - start);
    int total_mmas = mma_count * num_warps;
    double cyc_per_mma = (double)cycles / total_mmas;

    if (num_warps == 1) baseline = cyc_per_mma;
    double speedup = (cyc_per_mma > 0) ? (baseline / cyc_per_mma) : 0.0;

    printf("│   %3d   │   %6d   │  %8lu  │   %7.2f  │   %5.2fx  │\n", num_warps,
           total_mmas, cycles, cyc_per_mma, speedup);

    // Write to file
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
  printf("Results exported to: MMAILPIssueGapTest.MultiWarpMinimal.txt\n");
}
