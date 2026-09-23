#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kSharedAWords = 2048;
constexpr int kSharedBWords = 8192;
constexpr int kMaxAccumulatorRegisters = 128;
constexpr int kSoftmaxRegs = 32;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;

struct MixSample {
  uint64_t qk_issue_cycles;
  uint64_t qk_wait_cycles;
  uint64_t pv_issue_cycles;
  uint64_t pv_wait_cycles;
  uint64_t math_cycles;
  uint64_t total_cycles;
  uint32_t qk_wgmma_count;
  uint32_t pv_wgmma_count;
};

struct MixResult {
  std::string mode;
  int math_iters;
  int blocks;
  int rounds;
  int qk_ops_per_round;
  int pv_ops_per_round;
  uint64_t median_qk_issue_cycles;
  uint64_t median_qk_wait_cycles;
  uint64_t median_pv_issue_cycles;
  uint64_t median_pv_wait_cycles;
  uint64_t median_math_cycles;
  uint64_t median_total_cycles;
  double qk_issue_cycles_per_wgmma;
  double pv_issue_cycles_per_wgmma;
  double total_cycles_per_round;
};

enum class MixMode {
  QkOnly,
  PvOnly,
  QkPvOnly,
  MathOnly,
  QkWaitMath,
  QkMathWait,
  QkWaitMathPv,
  QkMathWaitPv,
};

enum class MathKind : int {
  Softmax32 = 0,
  AddChain,
  MulChain,
  FmaChain,
  MaxChain,
  Ex2Chain,
  Ex2Indep2,
  Ex2Indep4,
  FmaIndep8,
  IntAddChain,
  IntAddIndep8,
  Lop3Indep8,
  Mix4,
};

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void force_u64_ready(uint64_t &value) {
  asm volatile("add.u64 %0, %0, 0;" : "+l"(value)::"memory");
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const *ptr) {
  uint32_t smem_ptr = 0;
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
               "cvt.u32.u64 %0, smem_ptr; }\n"
               : "=r"(smem_ptr)
               : "l"(ptr));
  return smem_ptr;
}

__device__ __forceinline__ uint64_t make_gmma_swizzle_desc(
    void const *ptr, uint32_t leading_byte_offset, uint32_t stride_byte_offset,
    uint32_t swizzle_mode) {
  uint32_t smem_addr = smem_ptr_to_uint(ptr);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3fff);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3fff) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3fff) << 32;
  desc |= static_cast<uint64_t>(swizzle_mode & 0x3) << 62;
  return desc;
}

#define WGMMA_REG_LIST_D64                                                   \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, " \
  "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, " \
  "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, " \
  "%58, %59, %60, %61, %62, %63"

#define WGMMA_REG_LIST_D88                                                   \
  WGMMA_REG_LIST_D64                                                         \
  ", %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, "     \
  "%77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87"

#define WGMMA_FLOAT_D32                                                       \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),              \
      "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),          \
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),    \
      "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),    \
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),    \
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),    \
      "+f"(d[30]), "+f"(d[31])

#define WGMMA_FLOAT_D64                                                       \
  WGMMA_FLOAT_D32, "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),     \
      "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),    \
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),    \
      "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),    \
      "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),    \
      "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),    \
      "+f"(d[61]), "+f"(d[62]), "+f"(d[63])

#define WGMMA_FLOAT_D88                                                       \
  WGMMA_FLOAT_D64, "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),     \
      "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]),    \
      "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]),    \
      "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]),    \
      "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87])

__device__ __forceinline__ void issue_qk_m64n176k16_ss(
    uint64_t desc_a, uint64_t desc_b,
    float (&d)[kMaxAccumulatorRegisters]) {
  int32_t scale_d = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %88, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n176k16.f32.f16.f16 "
      "{" WGMMA_REG_LIST_D88 "}, %89, %90, p, 1, 1, 0, 0;\n"
      "}\n"
      : WGMMA_FLOAT_D88
      : "r"(scale_d), "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void issue_pv_m64n128k16_rs(
    const uint32_t (&a_regs)[4], uint64_t desc_b,
    float (&d)[kMaxAccumulatorRegisters]) {
  int32_t scale_d = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{" WGMMA_REG_LIST_D64 "}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : WGMMA_FLOAT_D64
      : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
        "l"(desc_b), "r"(scale_d));
}

__device__ __forceinline__ void touch_a_registers(uint32_t (&a_regs)[4]) {
  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3])::"memory");
}

__device__ __forceinline__ void touch_softmax_regs(
    float (&x)[kSoftmaxRegs]) {
  asm volatile(""
               : "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]),
                 "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]),
                 "+f"(x[8]), "+f"(x[9]), "+f"(x[10]), "+f"(x[11]),
                 "+f"(x[12]), "+f"(x[13]), "+f"(x[14]), "+f"(x[15]),
                 "+f"(x[16]), "+f"(x[17]), "+f"(x[18]), "+f"(x[19]),
                 "+f"(x[20]), "+f"(x[21]), "+f"(x[22]), "+f"(x[23]),
                 "+f"(x[24]), "+f"(x[25]), "+f"(x[26]), "+f"(x[27]),
                 "+f"(x[28]), "+f"(x[29]), "+f"(x[30]), "+f"(x[31])
               :
               : "memory");
}

__device__ __forceinline__ uint64_t issue_qk_group(
    uint64_t desc_a, uint64_t desc_b, float (&d)[kMaxAccumulatorRegisters],
    int qk_ops) {
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < qk_ops; ++op) {
    issue_qk_m64n176k16_ss(desc_a, desc_b, d);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t issue_pv_group(
    uint64_t desc_b, const uint32_t (&a_regs)[4],
    float (&d)[kMaxAccumulatorRegisters], int pv_ops) {
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < pv_ops; ++op) {
    issue_pv_m64n128k16_rs(a_regs, desc_b, d);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t wait_wgmma_group_0() {
  const uint64_t start = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_softmax_math(
    float (&x)[kSoftmaxRegs], int math_iters) {
  if (math_iters <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  const float scale = 0.9375f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int iter = 0; iter < math_iters; ++iter) {
    float row_max = x[0];
#pragma unroll
    for (int i = 1; i < kSoftmaxRegs; ++i) {
      float next_max;
      asm volatile("max.ftz.f32 %0, %1, %2;\n"
                   : "=f"(next_max)
                   : "f"(row_max), "f"(x[i]));
      row_max = next_max;
    }

    float row_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kSoftmaxRegs; ++i) {
      float shifted;
      float expv;
      float next_sum;
      float updated;
      asm volatile("sub.rn.ftz.f32 %0, %1, %2;\n"
                   : "=f"(shifted)
                   : "f"(x[i]), "f"(row_max));
      asm volatile("ex2.approx.ftz.f32 %0, %1;\n"
                   : "=f"(expv)
                   : "f"(shifted));
      asm volatile("add.rn.ftz.f32 %0, %1, %2;\n"
                   : "=f"(next_sum)
                   : "f"(row_sum), "f"(expv));
      asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;\n"
                   : "=f"(updated)
                   : "f"(expv), "f"(scale),
                     "f"(x[(i + 7) & (kSoftmaxRegs - 1)]));
      row_sum = next_sum;
      x[i] = updated;
    }

    const float norm = row_sum + 1.0f;
#pragma unroll
    for (int i = 0; i < kSoftmaxRegs; ++i) {
      float updated;
      asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;\n"
                   : "=f"(updated)
                   : "f"(x[i]), "f"(0.5f), "f"(norm));
      x[i] = updated;
    }
  }
  const uint64_t end = read_clock64();
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_add_chain_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0];
  const float rhs = x[1] + 0.03125f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("add.rn.ftz.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_mul_chain_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0] + 1.0f;
  const float rhs = 0.9990234375f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("mul.rn.ftz.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_fma_chain_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0];
  const float scale = 0.9375f;
  const float bias = x[1] + 0.015625f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;\n"
                 : "+f"(acc)
                 : "f"(scale), "f"(bias));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_max_chain_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0];
  const float rhs = x[1] + 0.5f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("max.ftz.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_ex2_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0] - 0.5f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;\n" : "+f"(acc));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_fma_indep8_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3];
  float a4 = x[4], a5 = x[5], a6 = x[6], a7 = x[7];
  const float scale = 0.9375f;
  const float bias = 0.015625f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile(
        "fma.rn.ftz.f32 %0, %0, %8, %9;\n"
        "fma.rn.ftz.f32 %1, %1, %8, %9;\n"
        "fma.rn.ftz.f32 %2, %2, %8, %9;\n"
        "fma.rn.ftz.f32 %3, %3, %8, %9;\n"
        "fma.rn.ftz.f32 %4, %4, %8, %9;\n"
        "fma.rn.ftz.f32 %5, %5, %8, %9;\n"
        "fma.rn.ftz.f32 %6, %6, %8, %9;\n"
        "fma.rn.ftz.f32 %7, %7, %8, %9;\n"
        : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3), "+f"(a4),
          "+f"(a5), "+f"(a6), "+f"(a7)
        : "f"(scale), "f"(bias));
  }
  const uint64_t end = read_clock64();
  x[0] = a0;
  x[1] = a1;
  x[2] = a2;
  x[3] = a3;
  x[4] = a4;
  x[5] = a5;
  x[6] = a6;
  x[7] = a7;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_mix4_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float acc = x[0];
  const float rhs = x[1] + 0.25f;
  const float scale = 0.9375f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    float shifted;
    float expv;
    asm volatile("max.ftz.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(rhs));
    asm volatile("sub.rn.ftz.f32 %0, %1, %2;\n"
                 : "=f"(shifted)
                 : "f"(acc), "f"(rhs));
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n"
                 : "=f"(expv)
                 : "f"(shifted));
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %0;\n"
                 : "+f"(acc)
                 : "f"(expv), "f"(scale));
  }
  const uint64_t end = read_clock64();
  x[0] = acc;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_ex2_indep2_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float a0 = x[0] - 0.5f;
  float a1 = x[1] - 0.25f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile(
        "ex2.approx.ftz.f32 %0, %0;\n"
        "ex2.approx.ftz.f32 %1, %1;\n"
        : "+f"(a0), "+f"(a1));
  }
  const uint64_t end = read_clock64();
  x[0] = a0;
  x[1] = a1;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_ex2_indep4_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  if (math_ops <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  float a0 = x[0] - 0.5f;
  float a1 = x[1] - 0.25f;
  float a2 = x[2] - 0.125f;
  float a3 = x[3] - 0.0625f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile(
        "ex2.approx.ftz.f32 %0, %0;\n"
        "ex2.approx.ftz.f32 %1, %1;\n"
        "ex2.approx.ftz.f32 %2, %2;\n"
        "ex2.approx.ftz.f32 %3, %3;\n"
        : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3));
  }
  const uint64_t end = read_clock64();
  x[0] = a0;
  x[1] = a1;
  x[2] = a2;
  x[3] = a3;
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_int_add_chain_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  uint32_t acc = static_cast<uint32_t>(threadIdx.x) * 1103515245u + 12345u;
  uint32_t rhs = 0x9e3779b9u ^ static_cast<uint32_t>(threadIdx.x);
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile("add.u32 %0, %0, %1;\n" : "+r"(acc) : "r"(rhs));
  }
  const uint64_t end = read_clock64();
  x[0] = static_cast<float>(acc & 0xffffu);
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_int_add_indep8_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  uint32_t a0 = 0x12345678u ^ static_cast<uint32_t>(threadIdx.x);
  uint32_t a1 = 0x9abcdef0u ^ static_cast<uint32_t>(threadIdx.x << 1);
  uint32_t a2 = 0x0fedcba9u ^ static_cast<uint32_t>(threadIdx.x << 2);
  uint32_t a3 = 0x87654321u ^ static_cast<uint32_t>(threadIdx.x << 3);
  uint32_t a4 = 0x13579bdfu ^ static_cast<uint32_t>(threadIdx.x << 4);
  uint32_t a5 = 0x2468ace0u ^ static_cast<uint32_t>(threadIdx.x << 5);
  uint32_t a6 = 0xfdb97531u ^ static_cast<uint32_t>(threadIdx.x << 6);
  uint32_t a7 = 0xeca86420u ^ static_cast<uint32_t>(threadIdx.x << 7);
  const uint32_t rhs = 0x9e3779b9u;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile(
        "add.u32 %0, %0, %8;\n"
        "add.u32 %1, %1, %8;\n"
        "add.u32 %2, %2, %8;\n"
        "add.u32 %3, %3, %8;\n"
        "add.u32 %4, %4, %8;\n"
        "add.u32 %5, %5, %8;\n"
        "add.u32 %6, %6, %8;\n"
        "add.u32 %7, %7, %8;\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4),
          "+r"(a5), "+r"(a6), "+r"(a7)
        : "r"(rhs));
  }
  const uint64_t end = read_clock64();
  x[0] = static_cast<float>((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) &
                            0xffffu);
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_lop3_indep8_math(
    float (&x)[kSoftmaxRegs], int math_ops) {
  uint32_t a0 = 0x12345678u ^ static_cast<uint32_t>(threadIdx.x);
  uint32_t a1 = 0x9abcdef0u ^ static_cast<uint32_t>(threadIdx.x << 1);
  uint32_t a2 = 0x0fedcba9u ^ static_cast<uint32_t>(threadIdx.x << 2);
  uint32_t a3 = 0x87654321u ^ static_cast<uint32_t>(threadIdx.x << 3);
  uint32_t a4 = 0x13579bdfu ^ static_cast<uint32_t>(threadIdx.x << 4);
  uint32_t a5 = 0x2468ace0u ^ static_cast<uint32_t>(threadIdx.x << 5);
  uint32_t a6 = 0xfdb97531u ^ static_cast<uint32_t>(threadIdx.x << 6);
  uint32_t a7 = 0xeca86420u ^ static_cast<uint32_t>(threadIdx.x << 7);
  const uint32_t mask = 0xa5a5a5a5u;
  const uint32_t salt = 0x3c6ef372u;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
    asm volatile(
        "lop3.b32 %0, %0, %8, %9, 0xca;\n"
        "lop3.b32 %1, %1, %8, %9, 0xca;\n"
        "lop3.b32 %2, %2, %8, %9, 0xca;\n"
        "lop3.b32 %3, %3, %8, %9, 0xca;\n"
        "lop3.b32 %4, %4, %8, %9, 0xca;\n"
        "lop3.b32 %5, %5, %8, %9, 0xca;\n"
        "lop3.b32 %6, %6, %8, %9, 0xca;\n"
        "lop3.b32 %7, %7, %8, %9, 0xca;\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4),
          "+r"(a5), "+r"(a6), "+r"(a7)
        : "r"(mask), "r"(salt));
  }
  const uint64_t end = read_clock64();
  x[0] = static_cast<float>((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) &
                            0xffffu);
  touch_softmax_regs(x);
  return end - start;
}

__device__ __forceinline__ uint64_t run_math_block(
    float (&x)[kSoftmaxRegs], int math_kind, int math_iters) {
#ifdef FLASHGPU_SIM_REPRESENTATIVE
  return run_softmax_math(x, math_iters);
#else
  switch (static_cast<MathKind>(math_kind)) {
    case MathKind::Softmax32:
      return run_softmax_math(x, math_iters);
    case MathKind::AddChain:
      return run_add_chain_math(x, math_iters);
    case MathKind::MulChain:
      return run_mul_chain_math(x, math_iters);
    case MathKind::FmaChain:
      return run_fma_chain_math(x, math_iters);
    case MathKind::MaxChain:
      return run_max_chain_math(x, math_iters);
    case MathKind::Ex2Chain:
      return run_ex2_math(x, math_iters);
    case MathKind::Ex2Indep2:
      return run_ex2_indep2_math(x, math_iters);
    case MathKind::Ex2Indep4:
      return run_ex2_indep4_math(x, math_iters);
    case MathKind::FmaIndep8:
      return run_fma_indep8_math(x, math_iters);
    case MathKind::IntAddChain:
      return run_int_add_chain_math(x, math_iters);
    case MathKind::IntAddIndep8:
      return run_int_add_indep8_math(x, math_iters);
    case MathKind::Lop3Indep8:
      return run_lop3_indep8_math(x, math_iters);
    case MathKind::Mix4:
      return run_mix4_math(x, math_iters);
  }
  return run_softmax_math(x, math_iters);
#endif
}

template <MixMode Mode>
__global__ void mix_kernel(MixSample *samples, float *sink_out, int rounds,
                           int qk_ops, int pv_ops, int math_iters,
                           int math_kind) {
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedAWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedBWords];
  for (int i = threadIdx.x; i < kSharedAWords; i += blockDim.x) {
    smem_a[i] = 0x3c003c00u;
  }
  for (int i = threadIdx.x; i < kSharedBWords; i += blockDim.x) {
    smem_b[i] = 0x3c003c00u;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);

  uint32_t a_regs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_regs[i] = 0x3c003c00u;
  }
  touch_a_registers(a_regs);

  float qk_acc[kMaxAccumulatorRegisters];
  float pv_acc[kMaxAccumulatorRegisters];
  float softmax_regs[kSoftmaxRegs];
#pragma unroll
  for (int i = 0; i < kMaxAccumulatorRegisters; ++i) {
    qk_acc[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
    pv_acc[i] = static_cast<float>((threadIdx.x & 7) + i + 17) * 0.001f;
  }
#pragma unroll
  for (int i = 0; i < kSoftmaxRegs; ++i) {
    softmax_regs[i] =
        static_cast<float>((threadIdx.x + 3 * i) & 0x1f) * 0.03125f;
  }
  touch_softmax_regs(softmax_regs);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t qk_issue_cycles = 0;
  uint64_t qk_wait_cycles = 0;
  uint64_t pv_issue_cycles = 0;
  uint64_t pv_wait_cycles = 0;
  uint64_t math_cycles = 0;
  const uint64_t total_start = read_clock64();

#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    if constexpr (Mode == MixMode::QkOnly) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::PvOnly) {
      pv_issue_cycles += issue_pv_group(desc_b, a_regs, pv_acc, pv_ops);
      pv_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::QkPvOnly) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
      pv_issue_cycles += issue_pv_group(desc_b, a_regs, pv_acc, pv_ops);
      pv_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::MathOnly) {
      math_cycles += run_math_block(softmax_regs, math_kind, math_iters);
    } else if constexpr (Mode == MixMode::QkWaitMath) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
      math_cycles += run_math_block(softmax_regs, math_kind, math_iters);
    } else if constexpr (Mode == MixMode::QkMathWait) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      math_cycles += run_math_block(softmax_regs, math_kind, math_iters);
      qk_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::QkWaitMathPv) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
      math_cycles += run_math_block(softmax_regs, math_kind, math_iters);
      pv_issue_cycles += issue_pv_group(desc_b, a_regs, pv_acc, pv_ops);
      pv_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::QkMathWaitPv) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      math_cycles += run_math_block(softmax_regs, math_kind, math_iters);
      qk_wait_cycles += wait_wgmma_group_0();
      pv_issue_cycles += issue_pv_group(desc_b, a_regs, pv_acc, pv_ops);
      pv_wait_cycles += wait_wgmma_group_0();
    }
  }

  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  float sink = 0.0f;
#pragma unroll
  for (int i = 0; i < 88; ++i) {
    sink += qk_acc[i];
  }
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    sink += pv_acc[i];
  }
#pragma unroll
  for (int i = 0; i < kSoftmaxRegs; ++i) {
    sink += softmax_regs[i];
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].qk_issue_cycles = qk_issue_cycles;
    samples[blockIdx.x].qk_wait_cycles = qk_wait_cycles;
    samples[blockIdx.x].pv_issue_cycles = pv_issue_cycles;
    samples[blockIdx.x].pv_wait_cycles = pv_wait_cycles;
    samples[blockIdx.x].math_cycles = math_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].qk_wgmma_count =
        (Mode == MixMode::QkOnly || Mode == MixMode::QkPvOnly ||
         Mode == MixMode::QkWaitMath || Mode == MixMode::QkMathWait ||
         Mode == MixMode::QkWaitMathPv || Mode == MixMode::QkMathWaitPv)
            ? static_cast<uint32_t>(rounds * qk_ops)
            : 0;
    samples[blockIdx.x].pv_wgmma_count =
        (Mode == MixMode::PvOnly || Mode == MixMode::QkPvOnly ||
         Mode == MixMode::QkWaitMathPv || Mode == MixMode::QkMathWaitPv)
            ? static_cast<uint32_t>(rounds * pv_ops)
            : 0;
    sink_out[blockIdx.x] = sink;
  }
#else
  if (threadIdx.x == 0) {
    samples[blockIdx.x] = {};
    sink_out[blockIdx.x] = 0.0f;
  }
#endif
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_add_chain_math(float &a0,
                                                             float &a1) {
  const float rhs = a1 + 0.03125f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("add.rn.ftz.f32 %0, %0, %1;\n" : "+f"(a0) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_mul_chain_math(float &a0) {
  const float rhs = 0.9990234375f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("mul.rn.ftz.f32 %0, %0, %1;\n" : "+f"(a0) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_fma_chain_math(float &a0,
                                                             float &a1) {
  const float scale = 0.9375f;
  const float bias = a1 + 0.015625f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;\n"
                 : "+f"(a0)
                 : "f"(scale), "f"(bias));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_max_chain_math(float &a0,
                                                             float &a1) {
  const float rhs = a1 + 0.5f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("max.ftz.f32 %0, %0, %1;\n" : "+f"(a0) : "f"(rhs));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_ex2_math(float &a0) {
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;\n" : "+f"(a0));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_ex2_indep2_math(float &a0,
                                                              float &a1) {
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile(
        "ex2.approx.ftz.f32 %0, %0;\n"
        "ex2.approx.ftz.f32 %1, %1;\n"
        : "+f"(a0), "+f"(a1));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_ex2_indep4_math(
    float &a0, float &a1, float &a2, float &a3) {
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile(
        "ex2.approx.ftz.f32 %0, %0;\n"
        "ex2.approx.ftz.f32 %1, %1;\n"
        "ex2.approx.ftz.f32 %2, %2;\n"
        "ex2.approx.ftz.f32 %3, %3;\n"
        : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_fma_indep8_math(
    float &a0, float &a1, float &a2, float &a3, float &a4, float &a5,
    float &a6, float &a7) {
  const float scale = 0.9375f;
  const float bias = 0.015625f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile(
        "fma.rn.ftz.f32 %0, %0, %8, %9;\n"
        "fma.rn.ftz.f32 %1, %1, %8, %9;\n"
        "fma.rn.ftz.f32 %2, %2, %8, %9;\n"
        "fma.rn.ftz.f32 %3, %3, %8, %9;\n"
        "fma.rn.ftz.f32 %4, %4, %8, %9;\n"
        "fma.rn.ftz.f32 %5, %5, %8, %9;\n"
        "fma.rn.ftz.f32 %6, %6, %8, %9;\n"
        "fma.rn.ftz.f32 %7, %7, %8, %9;\n"
        : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3), "+f"(a4),
          "+f"(a5), "+f"(a6), "+f"(a7)
        : "f"(scale), "f"(bias));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_int_add_chain_math(
    uint32_t &i0, uint32_t &i1) {
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile("add.u32 %0, %0, %1;\n" : "+r"(i0) : "r"(i1));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_int_add_indep8_math(
    uint32_t &i0, uint32_t &i1, uint32_t &i2, uint32_t &i3, uint32_t &i4,
    uint32_t &i5, uint32_t &i6, uint32_t &i7) {
  const uint32_t rhs = 0x9e3779b9u;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile(
        "add.u32 %0, %0, %8;\n"
        "add.u32 %1, %1, %8;\n"
        "add.u32 %2, %2, %8;\n"
        "add.u32 %3, %3, %8;\n"
        "add.u32 %4, %4, %8;\n"
        "add.u32 %5, %5, %8;\n"
        "add.u32 %6, %6, %8;\n"
        "add.u32 %7, %7, %8;\n"
        : "+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3), "+r"(i4),
          "+r"(i5), "+r"(i6), "+r"(i7)
        : "r"(rhs));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_lop3_indep8_math(
    uint32_t &i0, uint32_t &i1, uint32_t &i2, uint32_t &i3, uint32_t &i4,
    uint32_t &i5, uint32_t &i6, uint32_t &i7) {
  const uint32_t mask = 0xa5a5a5a5u;
  const uint32_t salt = 0x3c6ef372u;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    asm volatile(
        "lop3.b32 %0, %0, %8, %9, 0xca;\n"
        "lop3.b32 %1, %1, %8, %9, 0xca;\n"
        "lop3.b32 %2, %2, %8, %9, 0xca;\n"
        "lop3.b32 %3, %3, %8, %9, 0xca;\n"
        "lop3.b32 %4, %4, %8, %9, 0xca;\n"
        "lop3.b32 %5, %5, %8, %9, 0xca;\n"
        "lop3.b32 %6, %6, %8, %9, 0xca;\n"
        "lop3.b32 %7, %7, %8, %9, 0xca;\n"
        : "+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3), "+r"(i4),
          "+r"(i5), "+r"(i6), "+r"(i7)
        : "r"(mask), "r"(salt));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <int Ops>
__device__ __forceinline__ uint64_t run_clean_mix4_math(float &a0,
                                                        float &a1) {
  const float rhs = a1 + 0.25f;
  const float scale = 0.9375f;
  const uint64_t start = read_clock64();
#pragma unroll
  for (int op = 0; op < Ops; ++op) {
    float shifted;
    float expv;
    asm volatile("max.ftz.f32 %0, %0, %1;\n" : "+f"(a0) : "f"(rhs));
    asm volatile("sub.rn.ftz.f32 %0, %1, %2;\n"
                 : "=f"(shifted)
                 : "f"(a0), "f"(rhs));
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n"
                 : "=f"(expv)
                 : "f"(shifted));
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %0;\n"
                 : "+f"(a0)
                 : "f"(expv), "f"(scale));
  }
  const uint64_t end = read_clock64();
  return end - start;
}

template <MathKind Kind, int Ops>
__device__ __forceinline__ uint64_t run_clean_static_math(
    float &a0, float &a1, float &a2, float &a3, float &a4, float &a5,
    float &a6, float &a7, uint32_t &i0, uint32_t &i1, uint32_t &i2,
    uint32_t &i3, uint32_t &i4, uint32_t &i5, uint32_t &i6, uint32_t &i7) {
  if constexpr (Kind == MathKind::AddChain) {
    return run_clean_add_chain_math<Ops>(a0, a1);
  } else if constexpr (Kind == MathKind::MulChain) {
    return run_clean_mul_chain_math<Ops>(a0);
  } else if constexpr (Kind == MathKind::FmaChain) {
    return run_clean_fma_chain_math<Ops>(a0, a1);
  } else if constexpr (Kind == MathKind::MaxChain) {
    return run_clean_max_chain_math<Ops>(a0, a1);
  } else if constexpr (Kind == MathKind::Ex2Chain) {
    return run_clean_ex2_math<Ops>(a0);
  } else if constexpr (Kind == MathKind::Ex2Indep2) {
    return run_clean_ex2_indep2_math<Ops>(a0, a1);
  } else if constexpr (Kind == MathKind::Ex2Indep4) {
    return run_clean_ex2_indep4_math<Ops>(a0, a1, a2, a3);
  } else if constexpr (Kind == MathKind::FmaIndep8) {
    return run_clean_fma_indep8_math<Ops>(a0, a1, a2, a3, a4, a5, a6, a7);
  } else if constexpr (Kind == MathKind::IntAddChain) {
    return run_clean_int_add_chain_math<Ops>(i0, i1);
  } else if constexpr (Kind == MathKind::IntAddIndep8) {
    return run_clean_int_add_indep8_math<Ops>(i0, i1, i2, i3, i4, i5, i6, i7);
  } else if constexpr (Kind == MathKind::Lop3Indep8) {
    return run_clean_lop3_indep8_math<Ops>(i0, i1, i2, i3, i4, i5, i6, i7);
  } else if constexpr (Kind == MathKind::Mix4) {
    return run_clean_mix4_math<Ops>(a0, a1);
  } else {
    return 0;
  }
}

template <MixMode Mode, MathKind Kind, int Ops>
__global__ void fine_mix_kernel(MixSample *samples, float *sink_out,
                                int rounds, int qk_ops) {
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedAWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedBWords];
  for (int i = threadIdx.x; i < kSharedAWords; i += blockDim.x) {
    smem_a[i] = 0x3c003c00u;
  }
  for (int i = threadIdx.x; i < kSharedBWords; i += blockDim.x) {
    smem_b[i] = 0x3c003c00u;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);

  float qk_acc[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < kMaxAccumulatorRegisters; ++i) {
    qk_acc[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
  }

  float a0 = static_cast<float>((threadIdx.x + 1) & 0x1f) * 0.03125f;
  float a1 = static_cast<float>((threadIdx.x + 3) & 0x1f) * 0.03125f;
  float a2 = static_cast<float>((threadIdx.x + 5) & 0x1f) * 0.03125f;
  float a3 = static_cast<float>((threadIdx.x + 7) & 0x1f) * 0.03125f;
  float a4 = static_cast<float>((threadIdx.x + 11) & 0x1f) * 0.03125f;
  float a5 = static_cast<float>((threadIdx.x + 13) & 0x1f) * 0.03125f;
  float a6 = static_cast<float>((threadIdx.x + 17) & 0x1f) * 0.03125f;
  float a7 = static_cast<float>((threadIdx.x + 19) & 0x1f) * 0.03125f;
  uint32_t i0 = 0x12345678u ^ static_cast<uint32_t>(threadIdx.x);
  uint32_t i1 = 0x9abcdef0u ^ static_cast<uint32_t>(threadIdx.x << 1);
  uint32_t i2 = 0x0fedcba9u ^ static_cast<uint32_t>(threadIdx.x << 2);
  uint32_t i3 = 0x87654321u ^ static_cast<uint32_t>(threadIdx.x << 3);
  uint32_t i4 = 0x13579bdfu ^ static_cast<uint32_t>(threadIdx.x << 4);
  uint32_t i5 = 0x2468ace0u ^ static_cast<uint32_t>(threadIdx.x << 5);
  uint32_t i6 = 0xfdb97531u ^ static_cast<uint32_t>(threadIdx.x << 6);
  uint32_t i7 = 0xeca86420u ^ static_cast<uint32_t>(threadIdx.x << 7);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t qk_issue_cycles = 0;
  uint64_t qk_wait_cycles = 0;
  uint64_t math_cycles = 0;
  const uint64_t total_start = read_clock64();

#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    if constexpr (Mode == MixMode::QkOnly) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::MathOnly) {
      math_cycles += run_clean_static_math<Kind, Ops>(
          a0, a1, a2, a3, a4, a5, a6, a7, i0, i1, i2, i3, i4, i5, i6, i7);
    } else if constexpr (Mode == MixMode::QkWaitMath) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      qk_wait_cycles += wait_wgmma_group_0();
      force_u64_ready(qk_issue_cycles);
      force_u64_ready(qk_wait_cycles);
      math_cycles += run_clean_static_math<Kind, Ops>(
          a0, a1, a2, a3, a4, a5, a6, a7, i0, i1, i2, i3, i4, i5, i6, i7);
    } else if constexpr (Mode == MixMode::QkMathWait) {
      qk_issue_cycles += issue_qk_group(desc_a, desc_b, qk_acc, qk_ops);
      force_u64_ready(qk_issue_cycles);
      math_cycles += run_clean_static_math<Kind, Ops>(
          a0, a1, a2, a3, a4, a5, a6, a7, i0, i1, i2, i3, i4, i5, i6, i7);
      qk_wait_cycles += wait_wgmma_group_0();
    }
  }

  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  float sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  sink += static_cast<float>((i0 ^ i1 ^ i2 ^ i3 ^ i4 ^ i5 ^ i6 ^ i7) &
                             0xffu);
#pragma unroll
  for (int i = 0; i < 88; ++i) {
    sink += qk_acc[i];
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].qk_issue_cycles = qk_issue_cycles;
    samples[blockIdx.x].qk_wait_cycles = qk_wait_cycles;
    samples[blockIdx.x].pv_issue_cycles = 0;
    samples[blockIdx.x].pv_wait_cycles = 0;
    samples[blockIdx.x].math_cycles = math_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].qk_wgmma_count =
        (Mode == MixMode::QkOnly || Mode == MixMode::QkWaitMath ||
         Mode == MixMode::QkMathWait)
            ? static_cast<uint32_t>(rounds * qk_ops)
            : 0;
    samples[blockIdx.x].pv_wgmma_count = 0;
    sink_out[blockIdx.x] = sink;
  }
#else
  if (threadIdx.x == 0) {
    samples[blockIdx.x] = {};
    sink_out[blockIdx.x] = 0.0f;
  }
#endif
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

int get_env_int(const char *name, int default_value) {
  const char *text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') return default_value;
  char *end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
    throw std::runtime_error(std::string(name) + " must be a non-negative int");
  }
  return static_cast<int>(parsed);
}

std::string get_env_string(const char *name, const char *default_value) {
  const char *text = std::getenv(name);
  return text == nullptr || text[0] == '\0' ? std::string(default_value)
                                            : std::string(text);
}

uint64_t median_u64(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0 : values[values.size() / 2];
}

const char *mode_name(MixMode mode) {
  switch (mode) {
    case MixMode::QkOnly:
      return "qk_only";
    case MixMode::PvOnly:
      return "pv_only";
    case MixMode::QkPvOnly:
      return "qk_pv_only";
    case MixMode::MathOnly:
      return "math_only";
    case MixMode::QkWaitMath:
      return "qk_wait_math";
    case MixMode::QkMathWait:
      return "qk_math_wait";
    case MixMode::QkWaitMathPv:
      return "qk_wait_math_pv";
    case MixMode::QkMathWaitPv:
      return "qk_math_wait_pv";
  }
  return "unknown";
}

const char *math_kind_name(int math_kind) {
  switch (static_cast<MathKind>(math_kind)) {
    case MathKind::Softmax32:
      return "softmax32";
    case MathKind::AddChain:
      return "add_chain";
    case MathKind::MulChain:
      return "mul_chain";
    case MathKind::FmaChain:
      return "fma_chain";
    case MathKind::MaxChain:
      return "max_chain";
    case MathKind::Ex2Chain:
      return "ex2";
    case MathKind::Ex2Indep2:
      return "ex2_indep2";
    case MathKind::Ex2Indep4:
      return "ex2_indep4";
    case MathKind::FmaIndep8:
      return "fma_indep8";
    case MathKind::IntAddChain:
      return "int_add_chain";
    case MathKind::IntAddIndep8:
      return "int_add_indep8";
    case MathKind::Lop3Indep8:
      return "lop3_indep8";
    case MathKind::Mix4:
      return "mix4";
  }
  return "unknown";
}

int parse_math_kind(const std::string &name) {
  if (name == "softmax32") return static_cast<int>(MathKind::Softmax32);
  if (name == "add_chain") return static_cast<int>(MathKind::AddChain);
  if (name == "mul_chain") return static_cast<int>(MathKind::MulChain);
  if (name == "fma_chain") return static_cast<int>(MathKind::FmaChain);
  if (name == "max_chain") return static_cast<int>(MathKind::MaxChain);
  if (name == "ex2") return static_cast<int>(MathKind::Ex2Chain);
  if (name == "ex2_indep2") return static_cast<int>(MathKind::Ex2Indep2);
  if (name == "ex2_indep4") return static_cast<int>(MathKind::Ex2Indep4);
  if (name == "fma_indep8") return static_cast<int>(MathKind::FmaIndep8);
  if (name == "int_add_chain") return static_cast<int>(MathKind::IntAddChain);
  if (name == "int_add_indep8") {
    return static_cast<int>(MathKind::IntAddIndep8);
  }
  if (name == "lop3_indep8") return static_cast<int>(MathKind::Lop3Indep8);
  if (name == "mix4") return static_cast<int>(MathKind::Mix4);
  throw std::runtime_error(
      "unsupported WGMMA_SOFTMAX_MIX_MATH_KIND=" + name +
      " (expected softmax32, add_chain, mul_chain, fma_chain, max_chain, "
      "ex2, ex2_indep2, ex2_indep4, fma_indep8, int_add_chain, "
      "int_add_indep8, lop3_indep8, or mix4)");
}

int math_inst_per_thread(int math_kind, int math_ops) {
  switch (static_cast<MathKind>(math_kind)) {
    case MathKind::Softmax32:
      return 191 * math_ops;
    case MathKind::Ex2Indep2:
      return 2 * math_ops;
    case MathKind::Ex2Indep4:
      return 4 * math_ops;
    case MathKind::FmaIndep8:
      return 8 * math_ops;
    case MathKind::IntAddIndep8:
    case MathKind::Lop3Indep8:
      return 8 * math_ops;
    case MathKind::Mix4:
      return 4 * math_ops;
    case MathKind::AddChain:
    case MathKind::MulChain:
    case MathKind::FmaChain:
    case MathKind::MaxChain:
    case MathKind::Ex2Chain:
    case MathKind::IntAddChain:
      return math_ops;
  }
  return 0;
}

template <MixMode Mode>
MixResult run_case(cudaDeviceProp prop, int requested_blocks, int rounds,
                   int warmup_rounds, int qk_ops, int pv_ops,
                   int math_iters,
                   int math_kind = static_cast<int>(MathKind::Softmax32)) {
  const int blocks =
      requested_blocks > 0 ? requested_blocks
                           : std::max(1, prop.multiProcessorCount);
  const int effective_rounds = std::max(1, rounds);
  const char *name = mode_name(Mode);

  MixSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, blocks * sizeof(MixSample)),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, blocks * sizeof(float)), "cudaMalloc sink");

  for (int i = 0; i < warmup_rounds; ++i) {
    mix_kernel<Mode><<<blocks, kWarpgroupThreads>>>(
        d_samples, d_sink, 1, qk_ops, pv_ops, math_iters, math_kind);
    check_cuda(cudaGetLastError(), "warmup launch");
  }
  if (warmup_rounds > 0) {
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  mix_kernel<Mode><<<blocks, kWarpgroupThreads>>>(
      d_samples, d_sink, effective_rounds, qk_ops, pv_ops, math_iters,
      math_kind);
  check_cuda(cudaGetLastError(), "timed launch");
  check_cuda(cudaDeviceSynchronize(), "timed synchronize");

  std::vector<MixSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples, blocks * sizeof(MixSample),
                        cudaMemcpyDeviceToHost),
             "copy samples");
  check_cuda(cudaFree(d_samples), "cudaFree samples");
  check_cuda(cudaFree(d_sink), "cudaFree sink");

  std::vector<uint64_t> qk_issue_values;
  std::vector<uint64_t> qk_wait_values;
  std::vector<uint64_t> pv_issue_values;
  std::vector<uint64_t> pv_wait_values;
  std::vector<uint64_t> math_values;
  std::vector<uint64_t> total_values;
  for (const MixSample &sample : samples) {
    qk_issue_values.push_back(sample.qk_issue_cycles);
    qk_wait_values.push_back(sample.qk_wait_cycles);
    pv_issue_values.push_back(sample.pv_issue_cycles);
    pv_wait_values.push_back(sample.pv_wait_cycles);
    math_values.push_back(sample.math_cycles);
    total_values.push_back(sample.total_cycles);
  }
  if (total_values.empty()) {
    throw std::runtime_error(std::string(name) + ": no timing samples");
  }

  MixResult result{};
  result.mode = name;
  result.math_iters = math_iters;
  result.blocks = blocks;
  result.rounds = effective_rounds;
  result.qk_ops_per_round = qk_ops;
  result.pv_ops_per_round = pv_ops;
  result.median_qk_issue_cycles = median_u64(qk_issue_values);
  result.median_qk_wait_cycles = median_u64(qk_wait_values);
  result.median_pv_issue_cycles = median_u64(pv_issue_values);
  result.median_pv_wait_cycles = median_u64(pv_wait_values);
  result.median_math_cycles = median_u64(math_values);
  result.median_total_cycles = median_u64(total_values);
  const double qk_count = static_cast<double>(effective_rounds * qk_ops);
  const double pv_count = static_cast<double>(effective_rounds * pv_ops);
  result.qk_issue_cycles_per_wgmma =
      result.median_qk_issue_cycles == 0
          ? 0.0
          : static_cast<double>(result.median_qk_issue_cycles) / qk_count;
  result.pv_issue_cycles_per_wgmma =
      result.median_pv_issue_cycles == 0
          ? 0.0
          : static_cast<double>(result.median_pv_issue_cycles) / pv_count;
  result.total_cycles_per_round =
      static_cast<double>(result.median_total_cycles) /
      static_cast<double>(effective_rounds);
  return result;
}

void print_results(const std::vector<MixResult> &results) {
  printf("mode,math_iters,blocks,rounds,qk_ops,pv_ops,qk_issue,qk_wait,"
         "pv_issue,pv_wait,math,total,qk_issue_per_wgmma,"
         "pv_issue_per_wgmma,total_per_round\n");
  for (const MixResult &r : results) {
    printf("%s,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%.3f,%.3f,%.3f\n",
           r.mode.c_str(), r.math_iters, r.blocks, r.rounds,
           r.qk_ops_per_round, r.pv_ops_per_round,
           static_cast<unsigned long long>(r.median_qk_issue_cycles),
           static_cast<unsigned long long>(r.median_qk_wait_cycles),
           static_cast<unsigned long long>(r.median_pv_issue_cycles),
           static_cast<unsigned long long>(r.median_pv_wait_cycles),
           static_cast<unsigned long long>(r.median_math_cycles),
           static_cast<unsigned long long>(r.median_total_cycles),
           r.qk_issue_cycles_per_wgmma, r.pv_issue_cycles_per_wgmma,
           r.total_cycles_per_round);
  }
}

void write_csv(const std::string &path, const std::vector<MixResult> &results) {
  std::ofstream out(path);
  out << "mode,math_iters,blocks,rounds,qk_ops_per_round,pv_ops_per_round,"
         "median_qk_issue_cycles,median_qk_wait_cycles,"
         "median_pv_issue_cycles,median_pv_wait_cycles,median_math_cycles,"
         "median_total_cycles,qk_issue_cycles_per_wgmma,"
         "pv_issue_cycles_per_wgmma,total_cycles_per_round\n";
  for (const MixResult &r : results) {
    out << r.mode << "," << r.math_iters << "," << r.blocks << ","
        << r.rounds << "," << r.qk_ops_per_round << ","
        << r.pv_ops_per_round << "," << r.median_qk_issue_cycles << ","
        << r.median_qk_wait_cycles << "," << r.median_pv_issue_cycles << ","
        << r.median_pv_wait_cycles << "," << r.median_math_cycles << ","
        << r.median_total_cycles << "," << r.qk_issue_cycles_per_wgmma << ","
        << r.pv_issue_cycles_per_wgmma << "," << r.total_cycles_per_round
        << "\n";
  }
}

struct FineMixResult {
  std::string math_kind;
  int math_ops;
  int math_inst_per_thread;
  MixResult mix;
};

template <MixMode Mode, MathKind Kind, int Ops>
FineMixResult run_static_fine_case(cudaDeviceProp prop, int requested_blocks,
                                   int rounds, int warmup_rounds,
                                   int qk_ops) {
  const int blocks =
      requested_blocks > 0 ? requested_blocks
                           : std::max(1, prop.multiProcessorCount);
  const int effective_rounds = std::max(1, rounds);

  MixSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, blocks * sizeof(MixSample)),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, blocks * sizeof(float)), "cudaMalloc sink");

  for (int i = 0; i < warmup_rounds; ++i) {
    fine_mix_kernel<Mode, Kind, Ops>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, 1, qk_ops);
    check_cuda(cudaGetLastError(), "warmup launch");
  }
  if (warmup_rounds > 0) {
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  fine_mix_kernel<Mode, Kind, Ops><<<blocks, kWarpgroupThreads>>>(
      d_samples, d_sink, effective_rounds, qk_ops);
  check_cuda(cudaGetLastError(), "timed launch");
  check_cuda(cudaDeviceSynchronize(), "timed synchronize");

  std::vector<MixSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples, blocks * sizeof(MixSample),
                        cudaMemcpyDeviceToHost),
             "copy samples");
  check_cuda(cudaFree(d_samples), "cudaFree samples");
  check_cuda(cudaFree(d_sink), "cudaFree sink");

  std::vector<uint64_t> qk_issue_values;
  std::vector<uint64_t> qk_wait_values;
  std::vector<uint64_t> math_values;
  std::vector<uint64_t> total_values;
  for (const MixSample &sample : samples) {
    qk_issue_values.push_back(sample.qk_issue_cycles);
    qk_wait_values.push_back(sample.qk_wait_cycles);
    math_values.push_back(sample.math_cycles);
    total_values.push_back(sample.total_cycles);
  }
  if (total_values.empty()) {
    throw std::runtime_error(std::string(mode_name(Mode)) +
                             ": no timing samples");
  }

  FineMixResult result{};
  result.math_kind = math_kind_name(static_cast<int>(Kind));
  result.math_ops = Ops;
  result.math_inst_per_thread =
      math_inst_per_thread(static_cast<int>(Kind), Ops);
  result.mix.mode = mode_name(Mode);
  result.mix.math_iters = Ops;
  result.mix.blocks = blocks;
  result.mix.rounds = effective_rounds;
  result.mix.qk_ops_per_round = qk_ops;
  result.mix.pv_ops_per_round = 0;
  result.mix.median_qk_issue_cycles = median_u64(qk_issue_values);
  result.mix.median_qk_wait_cycles = median_u64(qk_wait_values);
  result.mix.median_pv_issue_cycles = 0;
  result.mix.median_pv_wait_cycles = 0;
  result.mix.median_math_cycles = median_u64(math_values);
  result.mix.median_total_cycles = median_u64(total_values);
  const double qk_count = static_cast<double>(effective_rounds * qk_ops);
  result.mix.qk_issue_cycles_per_wgmma =
      result.mix.median_qk_issue_cycles == 0
          ? 0.0
          : static_cast<double>(result.mix.median_qk_issue_cycles) / qk_count;
  result.mix.pv_issue_cycles_per_wgmma = 0.0;
  result.mix.total_cycles_per_round =
      static_cast<double>(result.mix.median_total_cycles) /
      static_cast<double>(effective_rounds);
  return result;
}

#define RUN_STATIC_FINE_OPS(MODE, KIND)                                      \
  switch (math_ops) {                                                        \
    case 0:                                                                  \
      return run_static_fine_case<MODE, KIND, 0>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 1:                                                                  \
      return run_static_fine_case<MODE, KIND, 1>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 2:                                                                  \
      return run_static_fine_case<MODE, KIND, 2>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 4:                                                                  \
      return run_static_fine_case<MODE, KIND, 4>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 8:                                                                  \
      return run_static_fine_case<MODE, KIND, 8>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 16:                                                                 \
      return run_static_fine_case<MODE, KIND, 16>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 24:                                                                 \
      return run_static_fine_case<MODE, KIND, 24>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 32:                                                                 \
      return run_static_fine_case<MODE, KIND, 32>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 64:                                                                 \
      return run_static_fine_case<MODE, KIND, 64>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    default:                                                                 \
      throw std::runtime_error("FineMathSweep math_ops must be one of "      \
                               "0,1,2,4,8,16,24,32,64");                    \
  }

template <MixMode Mode, MathKind Kind>
FineMixResult run_fine_ops(cudaDeviceProp prop, int blocks, int rounds,
                           int warmup_rounds, int qk_ops, int math_ops) {
  RUN_STATIC_FINE_OPS(Mode, Kind);
}

template <MixMode Mode>
FineMixResult run_fine_case(cudaDeviceProp prop, int blocks, int rounds,
                            int warmup_rounds, int qk_ops, int math_kind,
                            int math_ops) {
  switch (static_cast<MathKind>(math_kind)) {
    case MathKind::AddChain:
      return run_fine_ops<Mode, MathKind::AddChain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::MulChain:
      return run_fine_ops<Mode, MathKind::MulChain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::FmaChain:
      return run_fine_ops<Mode, MathKind::FmaChain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::MaxChain:
      return run_fine_ops<Mode, MathKind::MaxChain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Ex2Chain:
      return run_fine_ops<Mode, MathKind::Ex2Chain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Ex2Indep2:
      return run_fine_ops<Mode, MathKind::Ex2Indep2>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Ex2Indep4:
      return run_fine_ops<Mode, MathKind::Ex2Indep4>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::FmaIndep8:
      return run_fine_ops<Mode, MathKind::FmaIndep8>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::IntAddChain:
      return run_fine_ops<Mode, MathKind::IntAddChain>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::IntAddIndep8:
      return run_fine_ops<Mode, MathKind::IntAddIndep8>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Lop3Indep8:
      return run_fine_ops<Mode, MathKind::Lop3Indep8>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Mix4:
      return run_fine_ops<Mode, MathKind::Mix4>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_ops);
    case MathKind::Softmax32:
      break;
  }
  throw std::runtime_error("FineMathSweep does not support softmax32");
}

#define RUN_STATIC_FINE_DENSE_OPS(MODE, KIND)                                \
  switch (math_ops) {                                                        \
    case 0:                                                                  \
      return run_static_fine_case<MODE, KIND, 0>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 1:                                                                  \
      return run_static_fine_case<MODE, KIND, 1>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 2:                                                                  \
      return run_static_fine_case<MODE, KIND, 2>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 3:                                                                  \
      return run_static_fine_case<MODE, KIND, 3>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 4:                                                                  \
      return run_static_fine_case<MODE, KIND, 4>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 5:                                                                  \
      return run_static_fine_case<MODE, KIND, 5>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 6:                                                                  \
      return run_static_fine_case<MODE, KIND, 6>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 7:                                                                  \
      return run_static_fine_case<MODE, KIND, 7>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 8:                                                                  \
      return run_static_fine_case<MODE, KIND, 8>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 9:                                                                  \
      return run_static_fine_case<MODE, KIND, 9>(prop, blocks, rounds,       \
                                                 warmup_rounds, qk_ops);     \
    case 10:                                                                 \
      return run_static_fine_case<MODE, KIND, 10>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 11:                                                                 \
      return run_static_fine_case<MODE, KIND, 11>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 12:                                                                 \
      return run_static_fine_case<MODE, KIND, 12>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 13:                                                                 \
      return run_static_fine_case<MODE, KIND, 13>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 14:                                                                 \
      return run_static_fine_case<MODE, KIND, 14>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 15:                                                                 \
      return run_static_fine_case<MODE, KIND, 15>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 16:                                                                 \
      return run_static_fine_case<MODE, KIND, 16>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 17:                                                                 \
      return run_static_fine_case<MODE, KIND, 17>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 18:                                                                 \
      return run_static_fine_case<MODE, KIND, 18>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 19:                                                                 \
      return run_static_fine_case<MODE, KIND, 19>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 20:                                                                 \
      return run_static_fine_case<MODE, KIND, 20>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 21:                                                                 \
      return run_static_fine_case<MODE, KIND, 21>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 22:                                                                 \
      return run_static_fine_case<MODE, KIND, 22>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 23:                                                                 \
      return run_static_fine_case<MODE, KIND, 23>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    case 24:                                                                 \
      return run_static_fine_case<MODE, KIND, 24>(prop, blocks, rounds,      \
                                                  warmup_rounds, qk_ops);    \
    default:                                                                 \
      throw std::runtime_error("FineEx2FmaDenseSweep math_ops must be "      \
                               "in [0, 24]");                               \
  }

template <MixMode Mode, MathKind Kind>
FineMixResult run_fine_dense_ops(cudaDeviceProp prop, int blocks, int rounds,
                                 int warmup_rounds, int qk_ops,
                                 int math_ops) {
  RUN_STATIC_FINE_DENSE_OPS(Mode, Kind);
}

template <MathKind Kind>
void append_fine_dense_kind(cudaDeviceProp prop, int blocks, int rounds,
                            int warmup_rounds, int qk_ops,
                            const int *math_ops_values, int math_ops_count,
                            std::vector<FineMixResult> *results) {
  for (int i = 0; i < math_ops_count; ++i) {
    const int math_ops = math_ops_values[i];
    results->push_back(run_fine_dense_ops<MixMode::MathOnly, Kind>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops));
    results->push_back(run_fine_dense_ops<MixMode::QkWaitMath, Kind>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops));
    results->push_back(run_fine_dense_ops<MixMode::QkMathWait, Kind>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops));
  }
}

#undef RUN_STATIC_FINE_DENSE_OPS

#undef RUN_STATIC_FINE_OPS

#ifndef FLASHGPU_SIM_REPRESENTATIVE
FineMixResult run_fine_qk_only(cudaDeviceProp prop, int blocks, int rounds,
                               int warmup_rounds, int qk_ops) {
  FineMixResult result = run_static_fine_case<MixMode::QkOnly,
                                              MathKind::AddChain, 0>(
      prop, blocks, rounds, warmup_rounds, qk_ops);
  result.math_kind = "none";
  result.math_ops = 0;
  result.math_inst_per_thread = 0;
  return result;
}
#endif

void print_fine_results(const std::vector<FineMixResult> &results) {
  printf("mode,math_kind,math_ops,math_inst_per_thread,blocks,rounds,qk_ops,"
         "qk_issue,qk_wait,math,total,total_per_round,math_per_round\n");
  for (const FineMixResult &r : results) {
    const MixResult &m = r.mix;
    const double math_per_round =
        static_cast<double>(m.median_math_cycles) /
        static_cast<double>(m.rounds);
    printf("%s,%s,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%.3f,%.3f\n",
           m.mode.c_str(), r.math_kind.c_str(), r.math_ops,
           r.math_inst_per_thread, m.blocks, m.rounds,
           m.qk_ops_per_round,
           static_cast<unsigned long long>(m.median_qk_issue_cycles),
           static_cast<unsigned long long>(m.median_qk_wait_cycles),
           static_cast<unsigned long long>(m.median_math_cycles),
           static_cast<unsigned long long>(m.median_total_cycles),
           m.total_cycles_per_round, math_per_round);
  }
}

void write_fine_csv(const std::string &path,
                    const std::vector<FineMixResult> &results) {
  std::ofstream out(path);
  out << "mode,math_kind,math_ops,math_inst_per_thread,blocks,rounds,"
         "qk_ops_per_round,median_qk_issue_cycles,median_qk_wait_cycles,"
         "median_math_cycles,median_total_cycles,total_cycles_per_round,"
         "math_cycles_per_round\n";
  for (const FineMixResult &r : results) {
    const MixResult &m = r.mix;
    const double math_per_round =
        static_cast<double>(m.median_math_cycles) /
        static_cast<double>(m.rounds);
    out << m.mode << "," << r.math_kind << "," << r.math_ops << ","
        << r.math_inst_per_thread << "," << m.blocks << "," << m.rounds
        << "," << m.qk_ops_per_round << "," << m.median_qk_issue_cycles
        << "," << m.median_qk_wait_cycles << "," << m.median_math_cycles
        << "," << m.median_total_cycles << "," << m.total_cycles_per_round
        << "," << math_per_round << "\n";
  }
}

void append_selected(const std::string &selected, cudaDeviceProp prop,
                     int blocks, int rounds, int warmup_rounds, int qk_ops,
                     int pv_ops, int math_iters,
                     int math_kind, std::vector<MixResult> *results) {
#ifdef FLASHGPU_SIM_REPRESENTATIVE
  if (selected != "qk_wait_math_pv") {
    throw std::runtime_error(
        "simulator representative supports qk_wait_math_pv only");
  }
  results->push_back(run_case<MixMode::QkWaitMathPv>(
      prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
      math_kind));
#else
  if (selected == "qk_only") {
    results->push_back(run_case<MixMode::QkOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "pv_only") {
    results->push_back(run_case<MixMode::PvOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "qk_pv_only") {
    results->push_back(run_case<MixMode::QkPvOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "math_only") {
    results->push_back(run_case<MixMode::MathOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "qk_wait_math") {
    results->push_back(run_case<MixMode::QkWaitMath>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "qk_math_wait") {
    results->push_back(run_case<MixMode::QkMathWait>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "qk_wait_math_pv") {
    results->push_back(run_case<MixMode::QkWaitMathPv>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else if (selected == "qk_math_wait_pv") {
    results->push_back(run_case<MixMode::QkMathWaitPv>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters,
        math_kind));
  } else {
    throw std::runtime_error(
        "unsupported WGMMA_SOFTMAX_MIX_SELECTED=" + selected +
        " (expected qk_only, pv_only, qk_pv_only, math_only, "
        "qk_wait_math, qk_math_wait, qk_wait_math_pv, or "
        "qk_math_wait_pv)");
  }
#endif
}

void require_hopper_or_newer() {
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  if (prop.major < 9) {
    GTEST_SKIP() << "WGMMA requires SM90 or newer hardware.";
  }
}

}  // namespace

TEST(WgmmaSoftmaxMixBench, Selected) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 4096);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 8);
  const int pv_ops = get_env_int("WGMMA_SOFTMAX_MIX_PV_OPS", 11);
  const int math_iters = get_env_int("WGMMA_SOFTMAX_MIX_MATH_ITERS", 1);
  const int math_kind = parse_math_kind(
      get_env_string("WGMMA_SOFTMAX_MIX_MATH_KIND", "softmax32"));
  const std::string selected =
      get_env_string("WGMMA_SOFTMAX_MIX_SELECTED", "qk_wait_math_pv");
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.Selected");

  std::vector<MixResult> results;
  append_selected(selected, prop, blocks, rounds, warmup_rounds, qk_ops,
                  pv_ops, math_iters, math_kind, &results);
  print_results(results);
  write_csv(prefix + ".csv", results);
}

#ifndef FLASHGPU_SIM_REPRESENTATIVE
TEST(WgmmaSoftmaxMixBench, Sweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 4096);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 8);
  const int pv_ops = get_env_int("WGMMA_SOFTMAX_MIX_PV_OPS", 11);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.Sweep");

  std::vector<MixResult> results;
  for (int math_iters : {1, 2, 4, 8}) {
    results.push_back(run_case<MixMode::QkPvOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::MathOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::QkWaitMathPv>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::QkMathWaitPv>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
  }

  print_results(results);
  write_csv(prefix + ".csv", results);
}

TEST(WgmmaSoftmaxMixBench, FineMathSweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 8192);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 8);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.FineMathSweep");

  const int math_kinds[] = {
      static_cast<int>(MathKind::AddChain),
      static_cast<int>(MathKind::MulChain),
      static_cast<int>(MathKind::FmaChain),
      static_cast<int>(MathKind::MaxChain),
      static_cast<int>(MathKind::Ex2Chain),
      static_cast<int>(MathKind::FmaIndep8),
      static_cast<int>(MathKind::Mix4),
  };
  const int math_ops_values[] = {0, 1, 2, 4, 8, 16, 24, 32, 64};

  std::vector<FineMixResult> results;
  results.push_back(
      run_fine_qk_only(prop, blocks, rounds, warmup_rounds, qk_ops));
  for (int math_kind : math_kinds) {
    for (int math_ops : math_ops_values) {
      results.push_back(run_fine_case<MixMode::MathOnly>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_kind, math_ops));
      results.push_back(run_fine_case<MixMode::QkWaitMath>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_kind, math_ops));
      results.push_back(run_fine_case<MixMode::QkMathWait>(
          prop, blocks, rounds, warmup_rounds, qk_ops, math_kind, math_ops));
    }
  }

  print_fine_results(results);
  write_fine_csv(prefix + ".csv", results);
}

TEST(WgmmaSoftmaxMixBench, FineEx2FmaDenseSweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 8192);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 8);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.FineEx2FmaDenseSweep");

  const int math_ops_values[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

  std::vector<FineMixResult> results;
  results.push_back(
      run_fine_qk_only(prop, blocks, rounds, warmup_rounds, qk_ops));
  append_fine_dense_kind<MathKind::Ex2Chain>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::FmaIndep8>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);

  print_fine_results(results);
  write_fine_csv(prefix + ".csv", results);
}

TEST(WgmmaSoftmaxMixBench, FineOverlapModelDenseSweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 8192);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 8);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.FineOverlapModelDenseSweep");

  const int math_ops_values[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  std::vector<FineMixResult> results;
  results.push_back(
      run_fine_qk_only(prop, blocks, rounds, warmup_rounds, qk_ops));
  append_fine_dense_kind<MathKind::Ex2Chain>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::Ex2Indep2>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::Ex2Indep4>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::FmaIndep8>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::IntAddChain>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::IntAddIndep8>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::Lop3Indep8>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);
  append_fine_dense_kind<MathKind::Mix4>(
      prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
      static_cast<int>(std::size(math_ops_values)), &results);

  print_fine_results(results);
  write_fine_csv(prefix + ".csv", results);
}

TEST(WgmmaSoftmaxMixBench, FineQkOpsModelSweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 8192);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.FineQkOpsModelSweep");

  const int qk_ops_values[] = {1, 2, 4, 8, 16};
  const int math_ops_values[] = {0, 4, 8, 12, 16};

  std::vector<FineMixResult> results;
  for (int qk_ops : qk_ops_values) {
    results.push_back(
        run_fine_qk_only(prop, blocks, rounds, warmup_rounds, qk_ops));
    append_fine_dense_kind<MathKind::AddChain>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::FmaChain>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::Ex2Chain>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::Ex2Indep4>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::FmaIndep8>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::IntAddChain>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::IntAddIndep8>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::Lop3Indep8>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
    append_fine_dense_kind<MathKind::Mix4>(
        prop, blocks, rounds, warmup_rounds, qk_ops, math_ops_values,
        static_cast<int>(std::size(math_ops_values)), &results);
  }

  print_fine_results(results);
  write_fine_csv(prefix + ".csv", results);
}

TEST(WgmmaSoftmaxMixBench, LocalOverlapSweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_SOFTMAX_MIX_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_SOFTMAX_MIX_ROUNDS", 4096);
  const int warmup_rounds = get_env_int("WGMMA_SOFTMAX_MIX_WARMUP", 2);
  const int qk_ops = get_env_int("WGMMA_SOFTMAX_MIX_QK_OPS", 1);
  const int pv_ops = get_env_int("WGMMA_SOFTMAX_MIX_PV_OPS", 0);
  const std::string prefix = get_env_string(
      "WGMMA_SOFTMAX_MIX_OUT_PREFIX",
      "WgmmaSoftmaxMixBench.LocalOverlapSweep");

  std::vector<MixResult> results;
  for (int math_iters : {0, 1, 2, 4, 8, 16}) {
    results.push_back(run_case<MixMode::QkOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::MathOnly>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::QkWaitMath>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
    results.push_back(run_case<MixMode::QkMathWait>(
        prop, blocks, rounds, warmup_rounds, qk_ops, pv_ops, math_iters));
  }

  print_results(results);
  write_csv(prefix + ".csv", results);
}
#endif
