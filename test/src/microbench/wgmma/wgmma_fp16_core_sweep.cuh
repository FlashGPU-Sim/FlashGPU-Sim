#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wgmma_fp16_core {

constexpr int kWarpgroupThreads = 128;
constexpr int kSharedAWords = 2048;
constexpr int kSharedBWords = 8192;
constexpr int kMaxAccumulatorRegisters = 128;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;

struct TimingSample {
  uint64_t issue_cycles;
  uint64_t wait_cycles;
  uint64_t total_cycles;
  uint32_t wgmma_count;
};

struct SweepResult {
  std::string case_name;
  std::string op_name;
  std::string operand;
  std::string accumulator;
  int n;
  int k;
  int groups_before_wait;
  int ops_per_group;
  int blocks;
  int rounds;
  uint64_t median_issue_cycles;
  uint64_t median_wait_cycles;
  uint64_t median_total_cycles;
  double issue_cycles_per_wgmma;
  double wait_cycles_per_round;
  double total_cycles_per_wgmma;
};

enum class AccumulatorMode { Same, Rot2, Rot4 };

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
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

template <typename Accumulator>
__device__ __forceinline__ Accumulator initial_accumulator(int thread_id,
                                                           int index) {
  return static_cast<Accumulator>((thread_id & 0x7) + index) * 0.001f;
}

template <typename Op>
__device__ __forceinline__ void make_rs_a_regs(uint32_t (&a_regs)[4]) {
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    a_regs[reg] = Op::kARegPattern;
  }
}

__device__ __forceinline__ void touch_a_registers(uint32_t (&a_regs)[4]) {
  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3])::"memory");
}

__device__ __forceinline__ void touch_accumulators(
    float (&d)[kMaxAccumulatorRegisters]) {
  asm volatile(""
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
                 "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
                 "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
                 "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                 "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
                 "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
                 "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                 "+f"(d[30]), "+f"(d[31])::"memory");
}

template <int N> struct WgmmaF16Ss;
template <int N> struct WgmmaF16Rs;

#define WGMMA_CAT_IMPL(a, b) a##b
#define WGMMA_CAT(a, b) WGMMA_CAT_IMPL(a, b)
#define WGMMA_REG_LIST(D) WGMMA_CAT(WGMMA_REG_LIST_D, D)
#define WGMMA_FLOAT_D(D) WGMMA_CAT(WGMMA_FLOAT_D, D)

#define WGMMA_REG_LIST_D32 \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
#define WGMMA_REG_LIST_D64 \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, " \
  "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, " \
  "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
#define WGMMA_REG_LIST_D88 \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, " \
  "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, " \
  "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, " \
  "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, " \
  "%80, %81, %82, %83, %84, %85, %86, %87"
#define WGMMA_REG_LIST_D96 \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, " \
  "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, " \
  "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, " \
  "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, " \
  "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"

#define WGMMA_FLOAT_D32 \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
      "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define WGMMA_FLOAT_D64 \
  WGMMA_FLOAT_D32, "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
      "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), \
      "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), \
      "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), \
      "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), \
      "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), \
      "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), \
      "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
#define WGMMA_FLOAT_D88 \
  WGMMA_FLOAT_D64, "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), \
      "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), \
      "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), \
      "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), \
      "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), \
      "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87])
#define WGMMA_FLOAT_D96 \
  WGMMA_FLOAT_D88, "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), \
      "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])

#define DEFINE_F16_SS_N(N, D, P_DESC_A, P_DESC_B, P_SCALE_D, P_SCALE_A,      \
                        P_SCALE_B, P_TNSP_A, P_TNSP_B)                      \
  template <>                                                                \
  struct WgmmaF16Ss<N> {                                                     \
    using Accumulator = float;                                               \
    static constexpr int kK = 16;                                            \
    static constexpr int kD = D;                                             \
    static constexpr uint32_t kARegPattern = 0x3c003c00u;                    \
    static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;              \
    static const char *name() { return "m64n" #N "k16.f32.f16.f16.ss"; }   \
    __device__ __forceinline__ static void exec(                             \
        uint64_t desc_a, uint64_t desc_b, const uint32_t (&)[4],             \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 1;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST(D) "}, " P_DESC_A ", " P_DESC_B               \
          ", p, " P_SCALE_A ", " P_SCALE_B ", " P_TNSP_A ", " P_TNSP_B    \
          ";\n"                                                             \
          "}\n"                                                             \
          : WGMMA_FLOAT_D(D)                                                 \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),       \
            "n"(0), "n"(0));                                                 \
    }                                                                        \
  }

#define DEFINE_F16_RS_N(N, D, P_A0, P_A1, P_A2, P_A3, P_DESC_B, P_SCALE_D,  \
                        P_SCALE_A, P_SCALE_B, P_TNSP_B)                     \
  template <>                                                                \
  struct WgmmaF16Rs<N> {                                                     \
    using Accumulator = float;                                               \
    static constexpr int kK = 16;                                            \
    static constexpr int kD = D;                                             \
    static constexpr uint32_t kARegPattern = 0x3c003c00u;                    \
    static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;              \
    static const char *name() { return "m64n" #N "k16.f32.f16.f16.rs"; }   \
    __device__ __forceinline__ static void exec(                             \
        uint64_t, uint64_t desc_b, const uint32_t (&a_regs)[4],              \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 1;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST(D) "}, {" P_A0 ", " P_A1 ", " P_A2 ", "      \
          P_A3 "}, " P_DESC_B ", p, " P_SCALE_A ", " P_SCALE_B ", "        \
          P_TNSP_B ";\n"                                                    \
          "}\n"                                                             \
          : WGMMA_FLOAT_D(D)                                                 \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),    \
            "n"(0));                                                         \
    }                                                                        \
  }

DEFINE_F16_SS_N(64, 32, "%32", "%33", "%34", "%35", "%36", "%37", "%38");
DEFINE_F16_SS_N(128, 64, "%64", "%65", "%66", "%67", "%68", "%69", "%70");
DEFINE_F16_SS_N(176, 88, "%88", "%89", "%90", "%91", "%92", "%93", "%94");
DEFINE_F16_SS_N(192, 96, "%96", "%97", "%98", "%99", "%100", "%101", "%102");

DEFINE_F16_RS_N(64, 32, "%32", "%33", "%34", "%35", "%36", "%37", "%38",
                "%39", "%40");
DEFINE_F16_RS_N(128, 64, "%64", "%65", "%66", "%67", "%68", "%69", "%70",
                "%71", "%72");
DEFINE_F16_RS_N(176, 88, "%88", "%89", "%90", "%91", "%92", "%93", "%94",
                "%95", "%96");
DEFINE_F16_RS_N(192, 96, "%96", "%97", "%98", "%99", "%100", "%101", "%102",
                "%103", "%104");

template <template <int> class WgmmaOp, int N, int I, int Count>
__device__ __forceinline__ void issue_same_acc_ops(
    uint64_t desc_a, uint64_t desc_b, const uint32_t (&a_regs)[4],
    float (&d)[kMaxAccumulatorRegisters]) {
  if constexpr (I < Count) {
    WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d);
    issue_same_acc_ops<WgmmaOp, N, I + 1, Count>(desc_a, desc_b, a_regs, d);
  }
}

template <template <int> class WgmmaOp, int N, int I, int Count>
__device__ __forceinline__ void issue_rot2_acc_ops(
    uint64_t desc_a, uint64_t desc_b, const uint32_t (&a_regs)[4],
    float (&d0)[kMaxAccumulatorRegisters],
    float (&d1)[kMaxAccumulatorRegisters]) {
  if constexpr (I < Count) {
    if constexpr ((I & 1) == 0) {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d0);
    } else {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d1);
    }
    issue_rot2_acc_ops<WgmmaOp, N, I + 1, Count>(desc_a, desc_b, a_regs, d0,
                                                 d1);
  }
}

template <template <int> class WgmmaOp, int N, int I, int Count>
__device__ __forceinline__ void issue_rot4_acc_ops(
    uint64_t desc_a, uint64_t desc_b, const uint32_t (&a_regs)[4],
    float (&d0)[kMaxAccumulatorRegisters],
    float (&d1)[kMaxAccumulatorRegisters],
    float (&d2)[kMaxAccumulatorRegisters],
    float (&d3)[kMaxAccumulatorRegisters]) {
  if constexpr (I < Count) {
    if constexpr ((I & 3) == 0) {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d0);
    } else if constexpr ((I & 3) == 1) {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d1);
    } else if constexpr ((I & 3) == 2) {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d2);
    } else {
      WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d3);
    }
    issue_rot4_acc_ops<WgmmaOp, N, I + 1, Count>(desc_a, desc_b, a_regs, d0,
                                                 d1, d2, d3);
  }
}

template <template <int> class WgmmaOp, int N, int GroupsBeforeWait,
          int OpsPerGroup>
__global__ void same_acc_kernel(TimingSample *samples, float *sink_out,
                                int rounds) {
#if __CUDA_ARCH__ >= 900
  using Op = WgmmaOp<N>;
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedAWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedBWords];
  for (int i = threadIdx.x; i < kSharedAWords; i += blockDim.x) {
    smem_a[i] = Op::kSharedWordPattern;
  }
  for (int i = threadIdx.x; i < kSharedBWords; i += blockDim.x) {
    smem_b[i] = Op::kSharedWordPattern;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  uint32_t a_regs[4];
  make_rs_a_regs<Op>(a_regs);
  touch_a_registers(a_regs);

  float d[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < Op::kD; ++i) {
    d[i] = initial_accumulator<float>(threadIdx.x, i);
  }
  touch_accumulators(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
#pragma unroll
    for (int group = 0; group < GroupsBeforeWait; ++group) {
      issue_same_acc_ops<WgmmaOp, N, 0, OpsPerGroup>(desc_a, desc_b, a_regs,
                                                     d);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  touch_accumulators(d);
  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * GroupsBeforeWait * OpsPerGroup);
    sink_out[blockIdx.x] = d[0];
  }
#else
  if (threadIdx.x == 0) samples[blockIdx.x] = {};
#endif
}

template <template <int> class WgmmaOp, int N, int GroupsBeforeWait,
          int OpsPerGroup>
__global__ void rot2_acc_kernel(TimingSample *samples, float *sink_out,
                                int rounds) {
#if __CUDA_ARCH__ >= 900
  using Op = WgmmaOp<N>;
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedAWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedBWords];
  for (int i = threadIdx.x; i < kSharedAWords; i += blockDim.x) {
    smem_a[i] = Op::kSharedWordPattern;
  }
  for (int i = threadIdx.x; i < kSharedBWords; i += blockDim.x) {
    smem_b[i] = Op::kSharedWordPattern;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  uint32_t a_regs[4];
  make_rs_a_regs<Op>(a_regs);
  touch_a_registers(a_regs);

  float d0[kMaxAccumulatorRegisters];
  float d1[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < Op::kD; ++i) {
    d0[i] = initial_accumulator<float>(threadIdx.x, i);
    d1[i] = initial_accumulator<float>(threadIdx.x + 17, i);
  }
  touch_accumulators(d0);
  touch_accumulators(d1);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
#pragma unroll
    for (int group = 0; group < GroupsBeforeWait; ++group) {
      issue_rot2_acc_ops<WgmmaOp, N, 0, OpsPerGroup>(desc_a, desc_b, a_regs,
                                                     d0, d1);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  touch_accumulators(d0);
  touch_accumulators(d1);
  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * GroupsBeforeWait * OpsPerGroup);
    sink_out[blockIdx.x] = d0[0] + d1[0];
  }
#else
  if (threadIdx.x == 0) samples[blockIdx.x] = {};
#endif
}

template <template <int> class WgmmaOp, int N, int GroupsBeforeWait,
          int OpsPerGroup>
__global__ void rot4_acc_kernel(TimingSample *samples, float *sink_out,
                                int rounds) {
#if __CUDA_ARCH__ >= 900
  using Op = WgmmaOp<N>;
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedAWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedBWords];
  for (int i = threadIdx.x; i < kSharedAWords; i += blockDim.x) {
    smem_a[i] = Op::kSharedWordPattern;
  }
  for (int i = threadIdx.x; i < kSharedBWords; i += blockDim.x) {
    smem_b[i] = Op::kSharedWordPattern;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  uint32_t a_regs[4];
  make_rs_a_regs<Op>(a_regs);
  touch_a_registers(a_regs);

  float d0[kMaxAccumulatorRegisters];
  float d1[kMaxAccumulatorRegisters];
  float d2[kMaxAccumulatorRegisters];
  float d3[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < Op::kD; ++i) {
    d0[i] = initial_accumulator<float>(threadIdx.x, i);
    d1[i] = initial_accumulator<float>(threadIdx.x + 17, i);
    d2[i] = initial_accumulator<float>(threadIdx.x + 29, i);
    d3[i] = initial_accumulator<float>(threadIdx.x + 43, i);
  }
  touch_accumulators(d0);
  touch_accumulators(d1);
  touch_accumulators(d2);
  touch_accumulators(d3);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
#pragma unroll
    for (int group = 0; group < GroupsBeforeWait; ++group) {
      issue_rot4_acc_ops<WgmmaOp, N, 0, OpsPerGroup>(desc_a, desc_b, a_regs,
                                                     d0, d1, d2, d3);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  touch_accumulators(d0);
  touch_accumulators(d1);
  touch_accumulators(d2);
  touch_accumulators(d3);
  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * GroupsBeforeWait * OpsPerGroup);
    sink_out[blockIdx.x] = d0[0] + d1[0] + d2[0] + d3[0];
  }
#else
  if (threadIdx.x == 0) samples[blockIdx.x] = {};
#endif
}

std::string make_case_name(const char *operand, const char *accumulator, int n,
                           int groups_before_wait, int ops_per_group) {
  return std::string("f16_") + operand + "_n" + std::to_string(n) + "_g" +
         std::to_string(groups_before_wait) + "_o" +
         std::to_string(ops_per_group) + "_" + accumulator;
}

template <AccumulatorMode Mode, template <int> class WgmmaOp, int N,
          int GroupsBeforeWait, int OpsPerGroup>
SweepResult run_case(const char *operand, const char *accumulator,
                     cudaDeviceProp prop, int blocks, int rounds,
                     int warmup_rounds) {
  using Op = WgmmaOp<N>;
  const std::string case_name =
      make_case_name(operand, accumulator, N, GroupsBeforeWait, OpsPerGroup);

  TimingSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, blocks * sizeof(TimingSample)),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, blocks * sizeof(float)), "cudaMalloc sink");

  for (int i = 0; i < 3; ++i) {
    if constexpr (Mode == AccumulatorMode::Same) {
      same_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
          <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, warmup_rounds);
    } else if constexpr (Mode == AccumulatorMode::Rot2) {
      rot2_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
          <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, warmup_rounds);
    } else {
      rot4_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
          <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, warmup_rounds);
    }
    check_cuda(cudaGetLastError(), "warmup launch");
  }
  check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

  if constexpr (Mode == AccumulatorMode::Same) {
    same_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, rounds);
  } else if constexpr (Mode == AccumulatorMode::Rot2) {
    rot2_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, rounds);
  } else {
    rot4_acc_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, rounds);
  }
  check_cuda(cudaGetLastError(), "timed launch");
  check_cuda(cudaDeviceSynchronize(), "timed synchronize");

  std::vector<TimingSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples,
                        blocks * sizeof(TimingSample),
                        cudaMemcpyDeviceToHost),
             "copy timing samples");
  check_cuda(cudaFree(d_samples), "cudaFree samples");
  check_cuda(cudaFree(d_sink), "cudaFree sink");

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> total_values;
  for (const TimingSample &sample : samples) {
    if (sample.wgmma_count == 0) continue;
    issue_values.push_back(sample.issue_cycles);
    wait_values.push_back(sample.wait_cycles);
    total_values.push_back(sample.total_cycles);
  }
  if (issue_values.empty()) {
    throw std::runtime_error(case_name + ": no timing samples were produced");
  }

  const double wgmma_count =
      static_cast<double>(rounds * GroupsBeforeWait * OpsPerGroup);
  SweepResult result{};
  result.case_name = case_name;
  result.op_name = Op::name();
  result.operand = operand;
  result.accumulator = accumulator;
  result.n = N;
  result.k = Op::kK;
  result.groups_before_wait = GroupsBeforeWait;
  result.ops_per_group = OpsPerGroup;
  result.blocks = blocks;
  result.rounds = rounds;
  result.median_issue_cycles = median_u64(issue_values);
  result.median_wait_cycles = median_u64(wait_values);
  result.median_total_cycles = median_u64(total_values);
  result.issue_cycles_per_wgmma = result.median_issue_cycles / wgmma_count;
  result.wait_cycles_per_round =
      static_cast<double>(result.median_wait_cycles) / rounds;
  result.total_cycles_per_wgmma = result.median_total_cycles / wgmma_count;
  (void)prop;
  return result;
}

bool selected_by_filter(const std::string &filter, const std::string &name) {
  return filter.empty() || filter == "all" ||
         name.find(filter) != std::string::npos;
}

template <AccumulatorMode Mode, template <int> class WgmmaOp, int N,
          int GroupsBeforeWait, int OpsPerGroup>
void append_case(const char *operand, const char *accumulator,
                 cudaDeviceProp prop, int blocks, int rounds,
                 int warmup_rounds, const std::string &filter,
                 std::vector<SweepResult> *results) {
  const std::string case_name =
      make_case_name(operand, accumulator, N, GroupsBeforeWait, OpsPerGroup);
  if (!selected_by_filter(filter, case_name)) return;
  results->push_back(
      run_case<Mode, WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>(
          operand, accumulator, prop, blocks, rounds, warmup_rounds));
}

void write_csv(const std::string &path, const std::vector<SweepResult> &results) {
  std::ofstream out(path);
  out << "case,op,operand,accumulator,n,k,groups_before_wait,ops_per_group,"
         "blocks,rounds,median_issue_cycles,median_wait_cycles,"
         "median_total_cycles,issue_cycles_per_wgmma,wait_cycles_per_round,"
         "total_cycles_per_wgmma\n";
  for (const SweepResult &r : results) {
    out << r.case_name << "," << r.op_name << "," << r.operand << ","
        << r.accumulator << "," << r.n << "," << r.k << ","
        << r.groups_before_wait << "," << r.ops_per_group << ","
        << r.blocks << "," << r.rounds << "," << r.median_issue_cycles
        << "," << r.median_wait_cycles << "," << r.median_total_cycles
        << "," << r.issue_cycles_per_wgmma << "," << r.wait_cycles_per_round
        << "," << r.total_cycles_per_wgmma << "\n";
  }
}

void print_results(const std::vector<SweepResult> &results) {
  printf("case,op,operand,accumulator,n,groups,ops_per_group,blocks,rounds,"
         "issue_per_wgmma,wait_per_round,total_per_wgmma\n");
  for (const SweepResult &r : results) {
    printf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n",
           r.case_name.c_str(), r.op_name.c_str(), r.operand.c_str(),
           r.accumulator.c_str(), r.n, r.groups_before_wait, r.ops_per_group,
           r.blocks, r.rounds, r.issue_cycles_per_wgmma,
           r.wait_cycles_per_round, r.total_cycles_per_wgmma);
  }
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

#define WGMMA_FP16_APPEND_OPS(MODE, OP, OPERAND, ACCUM, N, G)               \
  append_case<MODE, OP, N, G, 1>(OPERAND, ACCUM, prop, blocks, rounds,       \
                                warmup_rounds, filter, &results);           \
  append_case<MODE, OP, N, G, 2>(OPERAND, ACCUM, prop, blocks, rounds,       \
                                warmup_rounds, filter, &results);           \
  append_case<MODE, OP, N, G, 4>(OPERAND, ACCUM, prop, blocks, rounds,       \
                                warmup_rounds, filter, &results);           \
  append_case<MODE, OP, N, G, 8>(OPERAND, ACCUM, prop, blocks, rounds,       \
                                warmup_rounds, filter, &results);           \
  append_case<MODE, OP, N, G, 16>(OPERAND, ACCUM, prop, blocks, rounds,      \
                                 warmup_rounds, filter, &results);          \
  append_case<MODE, OP, N, G, 32>(OPERAND, ACCUM, prop, blocks, rounds,      \
                                 warmup_rounds, filter, &results)

template <template <int> class Op, int Group>
void run_group_sweep(const char *operand, cudaDeviceProp prop, int blocks,
                     int rounds, int warmup_rounds, const std::string &filter,
                     std::vector<SweepResult> *results_ptr) {
  std::vector<SweepResult> &results = *results_ptr;
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Same, Op, operand, "same", 64,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Same, Op, operand, "same", 128,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Same, Op, operand, "same", 176,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Same, Op, operand, "same", 192,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Rot2, Op, operand, "rot2", 64,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Rot2, Op, operand, "rot2", 128,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Rot2, Op, operand, "rot2", 176,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Rot2, Op, operand, "rot2", 192,
                        Group);
  WGMMA_FP16_APPEND_OPS(AccumulatorMode::Rot4, Op, operand, "rot4", 64,
                        Group);
}

#undef WGMMA_FP16_APPEND_OPS

}  // namespace wgmma_fp16_core

TEST(WgmmaFp16CoreSweep, Selected) {
  using namespace wgmma_fp16_core;
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_FP16_SWEEP_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_FP16_SWEEP_ROUNDS", 256);
  const int warmup_rounds = get_env_int("WGMMA_FP16_SWEEP_WARMUP_ROUNDS", 16);
  const std::string filter =
      get_env_string("WGMMA_FP16_SWEEP_FILTER", "all");
  const std::string prefix = get_env_string(
      "WGMMA_FP16_SWEEP_OUT_PREFIX",
      WGMMA_FP16_SWEEP_DEFAULT_PREFIX);

  std::vector<SweepResult> results;
#if WGMMA_FP16_SWEEP_OPERAND_SS
  run_group_sweep<WgmmaF16Ss, WGMMA_FP16_SWEEP_GROUP>(
      "ss", prop, blocks, rounds, warmup_rounds, filter, &results);
#else
  run_group_sweep<WgmmaF16Rs, WGMMA_FP16_SWEEP_GROUP>(
      "rs", prop, blocks, rounds, warmup_rounds, filter, &results);
#endif

  ASSERT_FALSE(results.empty()) << "filter matched no cases: " << filter;
  print_results(results);
  write_csv(prefix + ".csv", results);
}
