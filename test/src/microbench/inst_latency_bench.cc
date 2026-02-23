#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

constexpr int MEASURE_ITERS = 20;

// ============================================================================
// Macro-based instruction unrolling
// ============================================================================

// Integer ADD (dependent chain)
#define INT_ADD_X1(a, b) asm volatile("add.s32 %0, %0, %1;" : "+r"(a) : "r"(b));
#define INT_ADD_X2(a, b) \
  INT_ADD_X1(a, b);      \
  INT_ADD_X1(a, b)
#define INT_ADD_X4(a, b) \
  INT_ADD_X2(a, b);      \
  INT_ADD_X2(a, b)
#define INT_ADD_X8(a, b) \
  INT_ADD_X4(a, b);      \
  INT_ADD_X4(a, b)
#define INT_ADD_X16(a, b) \
  INT_ADD_X8(a, b);       \
  INT_ADD_X8(a, b)
#define INT_ADD_X32(a, b) \
  INT_ADD_X16(a, b);      \
  INT_ADD_X16(a, b)
#define INT_ADD_X64(a, b) \
  INT_ADD_X32(a, b);      \
  INT_ADD_X32(a, b)

// Integer MUL (dependent chain)
#define INT_MUL_X1(a, b) \
  asm volatile("mul.lo.s32 %0, %0, %1;" : "+r"(a) : "r"(b));
#define INT_MUL_X2(a, b) \
  INT_MUL_X1(a, b);      \
  INT_MUL_X1(a, b)
#define INT_MUL_X4(a, b) \
  INT_MUL_X2(a, b);      \
  INT_MUL_X2(a, b)
#define INT_MUL_X8(a, b) \
  INT_MUL_X4(a, b);      \
  INT_MUL_X4(a, b)
#define INT_MUL_X16(a, b) \
  INT_MUL_X8(a, b);       \
  INT_MUL_X8(a, b)
#define INT_MUL_X32(a, b) \
  INT_MUL_X16(a, b);      \
  INT_MUL_X16(a, b)
#define INT_MUL_X64(a, b) \
  INT_MUL_X32(a, b);      \
  INT_MUL_X32(a, b)

// Integer MAD (dependent chain)
#define INT_MAD_X1(a, b, c) \
  asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(a) : "r"(b), "r"(c));
#define INT_MAD_X2(a, b, c) \
  INT_MAD_X1(a, b, c);      \
  INT_MAD_X1(a, b, c)
#define INT_MAD_X4(a, b, c) \
  INT_MAD_X2(a, b, c);      \
  INT_MAD_X2(a, b, c)
#define INT_MAD_X8(a, b, c) \
  INT_MAD_X4(a, b, c);      \
  INT_MAD_X4(a, b, c)
#define INT_MAD_X16(a, b, c) \
  INT_MAD_X8(a, b, c);       \
  INT_MAD_X8(a, b, c)
#define INT_MAD_X32(a, b, c) \
  INT_MAD_X16(a, b, c);      \
  INT_MAD_X16(a, b, c)
#define INT_MAD_X64(a, b, c) \
  INT_MAD_X32(a, b, c);      \
  INT_MAD_X32(a, b, c)

// FP32 ADD (dependent chain)
#define FP_ADD_X1(a, b) asm volatile("add.f32 %0, %0, %1;" : "+f"(a) : "f"(b));
#define FP_ADD_X2(a, b) \
  FP_ADD_X1(a, b);      \
  FP_ADD_X1(a, b)
#define FP_ADD_X4(a, b) \
  FP_ADD_X2(a, b);      \
  FP_ADD_X2(a, b)
#define FP_ADD_X8(a, b) \
  FP_ADD_X4(a, b);      \
  FP_ADD_X4(a, b)
#define FP_ADD_X16(a, b) \
  FP_ADD_X8(a, b);       \
  FP_ADD_X8(a, b)
#define FP_ADD_X32(a, b) \
  FP_ADD_X16(a, b);      \
  FP_ADD_X16(a, b)
#define FP_ADD_X64(a, b) \
  FP_ADD_X32(a, b);      \
  FP_ADD_X32(a, b)

// FP32 MUL (dependent chain)
#define FP_MUL_X1(a, b) asm volatile("mul.f32 %0, %0, %1;" : "+f"(a) : "f"(b));
#define FP_MUL_X2(a, b) \
  FP_MUL_X1(a, b);      \
  FP_MUL_X1(a, b)
#define FP_MUL_X4(a, b) \
  FP_MUL_X2(a, b);      \
  FP_MUL_X2(a, b)
#define FP_MUL_X8(a, b) \
  FP_MUL_X4(a, b);      \
  FP_MUL_X4(a, b)
#define FP_MUL_X16(a, b) \
  FP_MUL_X8(a, b);       \
  FP_MUL_X8(a, b)
#define FP_MUL_X32(a, b) \
  FP_MUL_X16(a, b);      \
  FP_MUL_X16(a, b)
#define FP_MUL_X64(a, b) \
  FP_MUL_X32(a, b);      \
  FP_MUL_X32(a, b)

// FP32 FMA (dependent chain)
#define FP_FMA_X1(a, b, c) \
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#define FP_FMA_X2(a, b, c) \
  FP_FMA_X1(a, b, c);      \
  FP_FMA_X1(a, b, c)
#define FP_FMA_X4(a, b, c) \
  FP_FMA_X2(a, b, c);      \
  FP_FMA_X2(a, b, c)
#define FP_FMA_X8(a, b, c) \
  FP_FMA_X4(a, b, c);      \
  FP_FMA_X4(a, b, c)
#define FP_FMA_X16(a, b, c) \
  FP_FMA_X8(a, b, c);       \
  FP_FMA_X8(a, b, c)
#define FP_FMA_X32(a, b, c) \
  FP_FMA_X16(a, b, c);      \
  FP_FMA_X16(a, b, c)
#define FP_FMA_X64(a, b, c) \
  FP_FMA_X32(a, b, c);      \
  FP_FMA_X32(a, b, c)

// FP64 ADD (dependent chain)
#define DP_ADD_X1(a, b) asm volatile("add.f64 %0, %0, %1;" : "+d"(a) : "d"(b));
#define DP_ADD_X2(a, b) \
  DP_ADD_X1(a, b);      \
  DP_ADD_X1(a, b)
#define DP_ADD_X4(a, b) \
  DP_ADD_X2(a, b);      \
  DP_ADD_X2(a, b)
#define DP_ADD_X8(a, b) \
  DP_ADD_X4(a, b);      \
  DP_ADD_X4(a, b)
#define DP_ADD_X16(a, b) \
  DP_ADD_X8(a, b);       \
  DP_ADD_X8(a, b)
#define DP_ADD_X32(a, b) \
  DP_ADD_X16(a, b);      \
  DP_ADD_X16(a, b)
#define DP_ADD_X64(a, b) \
  DP_ADD_X32(a, b);      \
  DP_ADD_X32(a, b)

// FP64 FMA (dependent chain)
#define DP_FMA_X1(a, b, c) \
  asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(a) : "d"(b), "d"(c));
#define DP_FMA_X2(a, b, c) \
  DP_FMA_X1(a, b, c);      \
  DP_FMA_X1(a, b, c)
#define DP_FMA_X4(a, b, c) \
  DP_FMA_X2(a, b, c);      \
  DP_FMA_X2(a, b, c)
#define DP_FMA_X8(a, b, c) \
  DP_FMA_X4(a, b, c);      \
  DP_FMA_X4(a, b, c)
#define DP_FMA_X16(a, b, c) \
  DP_FMA_X8(a, b, c);       \
  DP_FMA_X8(a, b, c)
#define DP_FMA_X32(a, b, c) \
  DP_FMA_X16(a, b, c);      \
  DP_FMA_X16(a, b, c)
#define DP_FMA_X64(a, b, c) \
  DP_FMA_X32(a, b, c);      \
  DP_FMA_X32(a, b, c)

// SFU operations (sin, cos, exp, etc.)
#define SFU_SIN_X1(a) asm volatile("sin.approx.f32 %0, %0;" : "+f"(a));
#define SFU_SIN_X2(a) \
  SFU_SIN_X1(a);      \
  SFU_SIN_X1(a)
#define SFU_SIN_X4(a) \
  SFU_SIN_X2(a);      \
  SFU_SIN_X2(a)
#define SFU_SIN_X8(a) \
  SFU_SIN_X4(a);      \
  SFU_SIN_X4(a)
#define SFU_SIN_X16(a) \
  SFU_SIN_X8(a);       \
  SFU_SIN_X8(a)
#define SFU_SIN_X32(a) \
  SFU_SIN_X16(a);      \
  SFU_SIN_X16(a)
#define SFU_SIN_X64(a) \
  SFU_SIN_X32(a);      \
  SFU_SIN_X32(a)

#define SFU_EXP_X1(a) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(a));
#define SFU_EXP_X2(a) \
  SFU_EXP_X1(a);      \
  SFU_EXP_X1(a)
#define SFU_EXP_X4(a) \
  SFU_EXP_X2(a);      \
  SFU_EXP_X2(a)
#define SFU_EXP_X8(a) \
  SFU_EXP_X4(a);      \
  SFU_EXP_X4(a)
#define SFU_EXP_X16(a) \
  SFU_EXP_X8(a);       \
  SFU_EXP_X8(a)
#define SFU_EXP_X32(a) \
  SFU_EXP_X16(a);      \
  SFU_EXP_X16(a)
#define SFU_EXP_X64(a) \
  SFU_EXP_X32(a);      \
  SFU_EXP_X32(a)

#define SFU_RCP_X1(a) asm volatile("rcp.approx.f32 %0, %0;" : "+f"(a));
#define SFU_RCP_X2(a) \
  SFU_RCP_X1(a);      \
  SFU_RCP_X1(a)
#define SFU_RCP_X4(a) \
  SFU_RCP_X2(a);      \
  SFU_RCP_X2(a)
#define SFU_RCP_X8(a) \
  SFU_RCP_X4(a);      \
  SFU_RCP_X4(a)
#define SFU_RCP_X16(a) \
  SFU_RCP_X8(a);       \
  SFU_RCP_X8(a)
#define SFU_RCP_X32(a) \
  SFU_RCP_X16(a);      \
  SFU_RCP_X16(a)
#define SFU_RCP_X64(a) \
  SFU_RCP_X32(a);      \
  SFU_RCP_X32(a)

// Tensor Core MMA
#define MMA_INST(D, A, B)                                               \
  asm volatile(                                                         \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "              \
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                  \
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#define MMA_X1(D, A, B) MMA_INST(D, A, B)
#define MMA_X2(D, A, B) \
  MMA_X1(D, A, B);      \
  MMA_X1(D, A, B)
#define MMA_X4(D, A, B) \
  MMA_X2(D, A, B);      \
  MMA_X2(D, A, B)
#define MMA_X8(D, A, B) \
  MMA_X4(D, A, B);      \
  MMA_X4(D, A, B)
#define MMA_X16(D, A, B) \
  MMA_X8(D, A, B);       \
  MMA_X8(D, A, B)
#define MMA_X32(D, A, B) \
  MMA_X16(D, A, B);      \
  MMA_X16(D, A, B)
#define MMA_X64(D, A, B) \
  MMA_X32(D, A, B);      \
  MMA_X32(D, A, B)

// ============================================================================
// Latency measurement kernels
// ============================================================================

__global__ void measure_clock_overhead(uint64_t* overhead) {
  uint64_t start, end;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
  *overhead = end - start;
}

// Integer ADD latency
// NOTE: 'inc' is passed as parameter to prevent compile-time constant folding
template <int COUNT>
__global__ void int_add_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, int* result,
                                       int inc) {
  int a = threadIdx.x + 1;
  int b = inc;  // Runtime value prevents PTXAS optimization

  // Warmup
  INT_ADD_X8(a, b);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    INT_ADD_X8(a, b);
  }
  if constexpr (COUNT == 16) {
    INT_ADD_X16(a, b);
  }
  if constexpr (COUNT == 32) {
    INT_ADD_X32(a, b);
  }
  if constexpr (COUNT == 64) {
    INT_ADD_X64(a, b);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// Integer MUL latency
// NOTE: 'mul' is passed as parameter to prevent compile-time constant folding
// Using multiply by 1 would be optimized away entirely
template <int COUNT>
__global__ void int_mul_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, int* result,
                                       int mul) {
  int a = threadIdx.x + 2;
  int b = mul;  // Runtime value prevents PTXAS optimization

  INT_MUL_X8(a, b);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    INT_MUL_X8(a, b);
  }
  if constexpr (COUNT == 16) {
    INT_MUL_X16(a, b);
  }
  if constexpr (COUNT == 32) {
    INT_MUL_X32(a, b);
  }
  if constexpr (COUNT == 64) {
    INT_MUL_X64(a, b);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// Integer MAD latency
// NOTE: 'mul' and 'add' are passed as parameters to prevent compile-time constant folding
template <int COUNT>
__global__ void int_mad_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, int* result,
                                       int mul, int add) {
  int a = threadIdx.x + 1;
  int b = mul;  // Runtime value prevents PTXAS optimization
  int c = add;

  INT_MAD_X8(a, b, c);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    INT_MAD_X8(a, b, c);
  }
  if constexpr (COUNT == 16) {
    INT_MAD_X16(a, b, c);
  }
  if constexpr (COUNT == 32) {
    INT_MAD_X32(a, b, c);
  }
  if constexpr (COUNT == 64) {
    INT_MAD_X64(a, b, c);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// FP32 ADD latency
template <int COUNT>
__global__ void fp_add_latency_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, float* result,
                                      float inc) {
  float a = (float)(threadIdx.x + 1);
  float b = inc;  // Runtime value prevents PTXAS optimization

  FP_ADD_X8(a, b);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    FP_ADD_X8(a, b);
  }
  if constexpr (COUNT == 16) {
    FP_ADD_X16(a, b);
  }
  if constexpr (COUNT == 32) {
    FP_ADD_X32(a, b);
  }
  if constexpr (COUNT == 64) {
    FP_ADD_X64(a, b);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// FP32 MUL latency
template <int COUNT>
__global__ void fp_mul_latency_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, float* result,
                                      float mul) {
  float a = (float)(threadIdx.x + 1);
  float b = mul;  // Runtime value prevents PTXAS optimization

  FP_MUL_X8(a, b);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    FP_MUL_X8(a, b);
  }
  if constexpr (COUNT == 16) {
    FP_MUL_X16(a, b);
  }
  if constexpr (COUNT == 32) {
    FP_MUL_X32(a, b);
  }
  if constexpr (COUNT == 64) {
    FP_MUL_X64(a, b);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// FP32 FMA latency
template <int COUNT>
__global__ void fp_fma_latency_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, float* result,
                                      float mul, float add) {
  float a = (float)(threadIdx.x + 1);
  float b = mul;  // Runtime value prevents PTXAS optimization
  float c = add;

  FP_FMA_X8(a, b, c);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    FP_FMA_X8(a, b, c);
  }
  if constexpr (COUNT == 16) {
    FP_FMA_X16(a, b, c);
  }
  if constexpr (COUNT == 32) {
    FP_FMA_X32(a, b, c);
  }
  if constexpr (COUNT == 64) {
    FP_FMA_X64(a, b, c);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// FP64 ADD latency
template <int COUNT>
__global__ void dp_add_latency_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, double* result,
                                      double inc) {
  double a = (double)(threadIdx.x + 1);
  double b = inc;  // Runtime value prevents PTXAS optimization

  DP_ADD_X8(a, b);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    DP_ADD_X8(a, b);
  }
  if constexpr (COUNT == 16) {
    DP_ADD_X16(a, b);
  }
  if constexpr (COUNT == 32) {
    DP_ADD_X32(a, b);
  }
  if constexpr (COUNT == 64) {
    DP_ADD_X64(a, b);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// FP64 FMA latency
template <int COUNT>
__global__ void dp_fma_latency_kernel(uint64_t* cycle_start,
                                      uint64_t* cycle_end, double* result,
                                      double mul, double add) {
  double a = (double)(threadIdx.x + 1);
  double b = mul;  // Runtime value prevents PTXAS optimization
  double c = add;

  DP_FMA_X8(a, b, c);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    DP_FMA_X8(a, b, c);
  }
  if constexpr (COUNT == 16) {
    DP_FMA_X16(a, b, c);
  }
  if constexpr (COUNT == 32) {
    DP_FMA_X32(a, b, c);
  }
  if constexpr (COUNT == 64) {
    DP_FMA_X64(a, b, c);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// SFU SIN latency
template <int COUNT>
__global__ void sfu_sin_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, float* result,
                                       float scale) {
  float a = (float)(threadIdx.x + 1) * scale;  // Runtime value

  SFU_SIN_X8(a);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    SFU_SIN_X8(a);
  }
  if constexpr (COUNT == 16) {
    SFU_SIN_X16(a);
  }
  if constexpr (COUNT == 32) {
    SFU_SIN_X32(a);
  }
  if constexpr (COUNT == 64) {
    SFU_SIN_X64(a);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// SFU EXP2 latency
template <int COUNT>
__global__ void sfu_exp_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, float* result,
                                       float scale) {
  float a = (float)(threadIdx.x + 1) * scale;  // Runtime value

  SFU_EXP_X8(a);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    SFU_EXP_X8(a);
  }
  if constexpr (COUNT == 16) {
    SFU_EXP_X16(a);
  }
  if constexpr (COUNT == 32) {
    SFU_EXP_X32(a);
  }
  if constexpr (COUNT == 64) {
    SFU_EXP_X64(a);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// SFU RCP latency
template <int COUNT>
__global__ void sfu_rcp_latency_kernel(uint64_t* cycle_start,
                                       uint64_t* cycle_end, float* result,
                                       float scale) {
  float a = (float)(threadIdx.x + 1) * scale;  // Runtime value

  SFU_RCP_X8(a);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    SFU_RCP_X8(a);
  }
  if constexpr (COUNT == 16) {
    SFU_RCP_X16(a);
  }
  if constexpr (COUNT == 32) {
    SFU_RCP_X32(a);
  }
  if constexpr (COUNT == 64) {
    SFU_RCP_X64(a);
  }

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = a;
  }
}

// Tensor Core MMA latency
template <int COUNT>
__global__ void mma_latency_kernel(uint64_t* cycle_start, uint64_t* cycle_end,
                                   float* result) {
  if (threadIdx.x >= 32) return;

  unsigned A_frag[4] = {0x3C003C00, 0x3C003C00, 0x3C003C00, 0x3C003C00};
  unsigned B_frag[2] = {0x3C003C00, 0x3C003C00};
  float D_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  MMA_X8(D_frag, A_frag, B_frag);
  __syncwarp();

  uint64_t start, end;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  if constexpr (COUNT == 8) {
    MMA_X8(D_frag, A_frag, B_frag);
  }
  if constexpr (COUNT == 16) {
    MMA_X16(D_frag, A_frag, B_frag);
  }
  if constexpr (COUNT == 32) {
    MMA_X32(D_frag, A_frag, B_frag);
  }
  if constexpr (COUNT == 64) {
    MMA_X64(D_frag, A_frag, B_frag);
  }

  // Force compiler to preserve computation result
  asm volatile("" : : "f"(D_frag[0]), "f"(D_frag[1]), "f"(D_frag[2]), "f"(D_frag[3]) : "memory");

  __syncwarp();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
    *result = D_frag[0];  // Write result after timing
  }
}

// Shared memory latency kernel
__global__ void smem_latency_kernel(uint64_t* cycle_start, uint64_t* cycle_end,
                                    int iterations) {
  __shared__ float smem[1024];

  int tid = threadIdx.x;
  smem[tid] = (float)tid;
  __syncthreads();

  float val = 0.0f;
  int idx = tid;

  // Warmup
  for (int i = 0; i < 8; i++) {
    val = smem[idx];
    idx = (int)val % 1024;
  }
  __syncwarp();

  uint64_t start, end;
  if (tid == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  }
  __syncwarp();

  // Pointer chasing to measure latency
  for (int i = 0; i < iterations; i++) {
    val = smem[idx];
    idx = ((int)val * 37 + 17) %
          1024;  // Pseudo-random access to avoid prediction
  }

  __syncwarp();
  if (tid == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    *cycle_start = start;
    *cycle_end = end;
  }

  // Prevent optimization
  if (val < -1e30f) smem[0] = val;
}

// ============================================================================
// Test fixture and helper functions
// ============================================================================

class InstLatencyTest : public ::testing::Test {
 protected:
  uint64_t* d_cycle_start;
  uint64_t* d_cycle_end;
  uint64_t* d_overhead;
  void* d_result;

  uint64_t h_cycle_start;
  uint64_t h_cycle_end;
  uint64_t h_overhead;

  void SetUp() override {
    cudaMalloc(&d_cycle_start, sizeof(uint64_t));
    cudaMalloc(&d_cycle_end, sizeof(uint64_t));
    cudaMalloc(&d_overhead, sizeof(uint64_t));
    cudaMalloc(&d_result, 256);
  }

  void TearDown() override {
    cudaFree(d_cycle_start);
    cudaFree(d_cycle_end);
    cudaFree(d_overhead);
    cudaFree(d_result);
  }

  uint64_t measureOverhead() {
    measure_clock_overhead<<<1, 1>>>(d_overhead);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_overhead, d_overhead, sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    return h_overhead;
  }

  struct LinearFit {
    double slope;
    double intercept;
    double r_squared;
  };

  LinearFit linearRegression(const std::vector<int>& x,
                             const std::vector<double>& y) {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; i++) {
      sum_x += x[i];
      sum_y += y[i];
      sum_xy += x[i] * y[i];
      sum_xx += x[i] * x[i];
    }

    LinearFit fit;
    fit.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    fit.intercept = (sum_y - fit.slope * sum_x) / n;

    double mean_y = sum_y / n;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
      double pred = fit.slope * x[i] + fit.intercept;
      ss_res += (y[i] - pred) * (y[i] - pred);
      ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
    }
    fit.r_squared = 1.0 - ss_res / ss_tot;

    return fit;
  }

  template <typename KernelFunc>
  double measureLatency(KernelFunc kernel, int count, uint64_t overhead) {
    double total = 0;
    for (int iter = 0; iter < MEASURE_ITERS; iter++) {
      kernel();
      cudaDeviceSynchronize();

      cudaMemcpy(&h_cycle_start, d_cycle_start, sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_cycle_end, d_cycle_end, sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);

      uint64_t elapsed = h_cycle_end - h_cycle_start;
      if (elapsed > overhead) elapsed -= overhead;
      total += elapsed;
    }
    return total / MEASURE_ITERS;
  }
};

// ============================================================================
// Main calibration test - measures all instruction types
// ============================================================================

TEST_F(InstLatencyTest, FullCalibrationSuite) {
  printf("\n");
  printf(
      "========================================================================"
      "========\n");
  printf(
      "  GPGPU-Sim Configuration Calibration - Instruction Latency "
      "Measurements\n");
  printf(
      "========================================================================"
      "========\n\n");

  // Device info retrieval removed (causes abort in simulator mode)
  printf("Device: GPGPU-Sim Device\n");
  printf(
      "Note: cudaGetDeviceProperties removed to avoid simulator "
      "incompatibility\n");
  printf("\n");

  uint64_t overhead = measureOverhead();
  printf("Clock64 overhead: %lu cycles\n\n", overhead);

  std::vector<int> counts = {8, 16, 32, 64};

  // Helper lambda for regression
  auto runAndReport = [&](const char* name, auto& results) {
    auto fit = linearRegression(counts, results);
    printf("  Slope (cycles/op): %.2f\n", fit.slope);
    printf("  Intercept: %.2f\n", fit.intercept);
    printf("  R²: %.4f\n\n", fit.r_squared);
    return fit.slope;
  };

  // Storage for final results
  double int_add_lat = 0, int_mul_lat = 0, int_mad_lat = 0;
  double fp_add_lat = 0, fp_mul_lat = 0, fp_fma_lat = 0;
  double dp_add_lat = 0, dp_fma_lat = 0;
  double sfu_sin_lat = 0, sfu_exp_lat = 0, sfu_rcp_lat = 0;
  double mma_lat = 0;

  // ========== Integer Operations ==========
  printf("=== Integer Operations ===\n");
  printf("(Config: -ptx_opcode_latency_int ADD,MAX,MUL,MAD,DIV,SHFL)\n\n");

  // INT ADD
  {
    printf("INT ADD:\n");
    std::vector<double> results;
    int inc = 1;  // Runtime value to prevent optimization
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                int_add_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, inc);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                int_add_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, inc);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                int_add_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, inc);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                int_add_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, inc);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    int_add_lat = runAndReport("INT ADD", results);
  }

  // INT MUL
  {
    printf("INT MUL:\n");
    std::vector<double> results;
    int mul = 1;  // Runtime value to prevent optimization
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                int_mul_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                int_mul_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                int_mul_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                int_mul_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    int_mul_lat = runAndReport("INT MUL", results);
  }

  // INT MAD
  {
    printf("INT MAD:\n");
    std::vector<double> results;
    int mul = 1, add = 1;  // Runtime values to prevent optimization
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                int_mad_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul, add);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                int_mad_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul, add);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                int_mad_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul, add);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                int_mad_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (int*)d_result, mul, add);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    int_mad_lat = runAndReport("INT MAD", results);
  }

  // ========== FP32 Operations ==========
  printf("=== FP32 Operations ===\n");
  printf("(Config: -ptx_opcode_latency_fp ADD,MAX,MUL,MAD,DIV)\n\n");

  // FP ADD
  {
    printf("FP32 ADD:\n");
    std::vector<double> results;
    float inc = 0.1f;  // Runtime value to prevent optimization
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                fp_add_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, inc);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                fp_add_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, inc);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                fp_add_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, inc);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                fp_add_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, inc);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    fp_add_lat = runAndReport("FP32 ADD", results);
  }

  // FP MUL
  {
    printf("FP32 MUL:\n");
    std::vector<double> results;
    float mul = 1.0001f;  // Runtime value to prevent optimization (not exactly 1.0)
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                fp_mul_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, mul);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                fp_mul_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, mul);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                fp_mul_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, mul);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                fp_mul_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, mul);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    fp_mul_lat = runAndReport("FP32 MUL", results);
  }

  // FP FMA
  {
    printf("FP32 FMA:\n");
    std::vector<double> results;
    float fma_mul = 1.0001f, fma_add = 0.1f;  // Runtime values
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                fp_fma_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, fma_mul, fma_add);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                fp_fma_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, fma_mul, fma_add);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                fp_fma_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, fma_mul, fma_add);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                fp_fma_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, fma_mul, fma_add);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    fp_fma_lat = runAndReport("FP32 FMA", results);
  }

  // ========== FP64 Operations ==========
  printf("=== FP64 Operations ===\n");
  printf("(Config: -ptx_opcode_latency_dp ADD,MAX,MUL,MAD,DIV)\n\n");

  // DP ADD
  {
    printf("FP64 ADD:\n");
    std::vector<double> results;
    double dp_inc = 0.1;  // Runtime value
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                dp_add_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_inc);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                dp_add_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_inc);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                dp_add_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_inc);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                dp_add_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_inc);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    dp_add_lat = runAndReport("FP64 ADD", results);
  }

  // DP FMA
  {
    printf("FP64 FMA:\n");
    std::vector<double> results;
    double dp_fma_mul = 1.0001, dp_fma_add = 0.1;  // Runtime values
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                dp_fma_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_fma_mul, dp_fma_add);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                dp_fma_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_fma_mul, dp_fma_add);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                dp_fma_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_fma_mul, dp_fma_add);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                dp_fma_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (double*)d_result, dp_fma_mul, dp_fma_add);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    dp_fma_lat = runAndReport("FP64 FMA", results);
  }

  // ========== SFU Operations ==========
  printf("=== SFU Operations ===\n");
  printf("(Config: -ptx_opcode_latency_sfu, -ptx_opcode_initiation_sfu)\n\n");

  // SFU SIN
  {
    printf("SFU SIN:\n");
    std::vector<double> results;
    float sfu_scale = 0.01f;  // Runtime value
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                sfu_sin_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, sfu_scale);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                sfu_sin_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, sfu_scale);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                sfu_sin_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, sfu_scale);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                sfu_sin_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, sfu_scale);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    sfu_sin_lat = runAndReport("SFU SIN", results);
  }

  // SFU EXP2
  {
    printf("SFU EXP2:\n");
    std::vector<double> results;
    float exp_scale = 0.001f;  // Runtime value (small to avoid overflow)
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                sfu_exp_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, exp_scale);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                sfu_exp_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, exp_scale);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                sfu_exp_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, exp_scale);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                sfu_exp_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, exp_scale);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    sfu_exp_lat = runAndReport("SFU EXP2", results);
  }

  // SFU RCP
  {
    printf("SFU RCP:\n");
    std::vector<double> results;
    float rcp_scale = 1.0f;  // Runtime value
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                sfu_rcp_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, rcp_scale);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                sfu_rcp_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, rcp_scale);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                sfu_rcp_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, rcp_scale);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                sfu_rcp_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result, rcp_scale);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    sfu_rcp_lat = runAndReport("SFU RCP", results);
  }

  // ========== Tensor Core Operations ==========
  printf("=== Tensor Core Operations ===\n");
  printf(
      "(Config: -ptx_opcode_latency_tensor, "
      "-ptx_opcode_initiation_tensor)\n\n");

  {
    printf("MMA m16n8k16:\n");
    std::vector<double> results;
    for (int count : counts) {
      double lat;
      switch (count) {
        case 8:
          lat = measureLatency(
              [&] {
                mma_latency_kernel<8>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result);
              },
              count, overhead);
          break;
        case 16:
          lat = measureLatency(
              [&] {
                mma_latency_kernel<16>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result);
              },
              count, overhead);
          break;
        case 32:
          lat = measureLatency(
              [&] {
                mma_latency_kernel<32>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result);
              },
              count, overhead);
          break;
        case 64:
          lat = measureLatency(
              [&] {
                mma_latency_kernel<64>
                    <<<1, 32>>>(d_cycle_start, d_cycle_end, (float*)d_result);
              },
              count, overhead);
          break;
      }
      results.push_back(lat);
      printf("  %d ops: %.1f cycles (%.2f cycles/op)\n", count, lat,
             lat / count);
    }
    mma_lat = runAndReport("MMA", results);
  }

  // ========== Shared Memory Latency ==========
  printf("=== Shared Memory Latency ===\n");
  printf("(Config: -gpgpu_smem_latency)\n\n");

  {
    std::vector<int> smem_counts = {16, 32, 64, 128};
    std::vector<double> results;
    for (int count : smem_counts) {
      double total = 0;
      for (int iter = 0; iter < MEASURE_ITERS; iter++) {
        smem_latency_kernel<<<1, 32>>>(d_cycle_start, d_cycle_end, count);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_cycle_start, d_cycle_start, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_cycle_end, d_cycle_end, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint64_t elapsed = h_cycle_end - h_cycle_start;
        if (elapsed > overhead) elapsed -= overhead;
        total += elapsed;
      }
      double lat = total / MEASURE_ITERS;
      results.push_back(lat);
      printf("  %d accesses: %.1f cycles (%.2f cycles/access)\n", count, lat,
             lat / count);
    }
    auto fit = linearRegression(smem_counts, results);
    printf("  Slope (cycles/access): %.2f\n", fit.slope);
    printf("  Intercept: %.2f\n", fit.intercept);
    printf("  R²: %.4f\n\n", fit.r_squared);
  }

  // ========== Summary and Recommendations ==========
  printf(
      "========================================================================"
      "========\n");
  printf("  RECOMMENDED GPGPU-Sim CONFIGURATION VALUES\n");
  printf(
      "========================================================================"
      "========\n\n");

  printf("# Instruction latencies (measured on GPGPU-Sim Device)\n");
  printf("# Format: ADD,MAX,MUL,MAD,DIV,SHFL\n\n");

  int int_lat =
      (int)std::round((int_add_lat + int_mul_lat + int_mad_lat) / 3.0);
  int fp_lat = (int)std::round((fp_add_lat + fp_mul_lat + fp_fma_lat) / 3.0);
  int dp_lat = (int)std::round((dp_add_lat + dp_fma_lat) / 2.0);
  int sfu_lat =
      (int)std::round((sfu_sin_lat + sfu_exp_lat + sfu_rcp_lat) / 3.0);
  int tensor_lat = (int)std::round(mma_lat);

  printf("-ptx_opcode_latency_int %d,%d,%d,%d,21,14\n", int_lat, int_lat,
         (int)std::round(int_mul_lat), (int)std::round(int_mad_lat));
  printf("-ptx_opcode_latency_fp %d,%d,%d,%d,39\n", fp_lat, fp_lat,
         (int)std::round(fp_mul_lat), (int)std::round(fp_fma_lat));
  printf("-ptx_opcode_latency_dp %d,%d,%d,%d,330\n", dp_lat, dp_lat, dp_lat,
         dp_lat);
  printf("-ptx_opcode_latency_sfu %d\n", sfu_lat);
  printf("-ptx_opcode_latency_tensor %d\n", tensor_lat);
  printf("-ptx_opcode_initiation_tensor %d\n", tensor_lat);

  printf("\n# Individual measurements:\n");
  printf("#   INT ADD: %.1f cycles\n", int_add_lat);
  printf("#   INT MUL: %.1f cycles\n", int_mul_lat);
  printf("#   INT MAD: %.1f cycles\n", int_mad_lat);
  printf("#   FP32 ADD: %.1f cycles\n", fp_add_lat);
  printf("#   FP32 MUL: %.1f cycles\n", fp_mul_lat);
  printf("#   FP32 FMA: %.1f cycles\n", fp_fma_lat);
  printf("#   FP64 ADD: %.1f cycles\n", dp_add_lat);
  printf("#   FP64 FMA: %.1f cycles\n", dp_fma_lat);
  printf("#   SFU SIN: %.1f cycles\n", sfu_sin_lat);
  printf("#   SFU EXP: %.1f cycles\n", sfu_exp_lat);
  printf("#   SFU RCP: %.1f cycles\n", sfu_rcp_lat);
  printf("#   MMA: %.1f cycles\n", mma_lat);

  printf(
      "\n======================================================================"
      "==========\n");
}
