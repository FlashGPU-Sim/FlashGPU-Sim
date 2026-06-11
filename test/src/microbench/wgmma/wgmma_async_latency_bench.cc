#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kSharedWords = 4096;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;
constexpr int kRounds = 256;

struct TimingSample {
  uint64_t issue_cycles;
  uint64_t wait_cycles;
  uint64_t total_cycles;
  uint32_t wgmma_count;
};

struct TimingResult {
  const char *name;
  int n;
  int k;
  int groups;
  int ops_per_group;
  int rounds;
  uint64_t median_issue_cycles;
  uint64_t median_wait_cycles;
  uint64_t median_total_cycles;
  double issue_cycles_per_wgmma;
  double wait_cycles_per_round;
  double total_cycles_per_wgmma;
};

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
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  desc |= static_cast<uint64_t>(swizzle_mode & 0x3) << 62;
  return desc;
}

template <typename WgmmaOp>
__device__ __forceinline__ void make_rs_a_regs(uint32_t (&a_regs)[4]) {
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    a_regs[reg] = WgmmaOp::kARegPattern;
  }
}

__device__ __forceinline__ void touch_a_registers(uint32_t (&a_regs)[4]) {
  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3])::"memory");
}

__device__ __forceinline__ void touch_accumulators(float (&d)[32]) {
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

__device__ __forceinline__ void touch_accumulators(int32_t (&d)[32]) {
  asm volatile(""
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]),
                 "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]),
                 "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
                 "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]),
                 "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]),
                 "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]),
                 "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
                 "+r"(d[30]), "+r"(d[31])::"memory");
}

template <typename Accumulator>
__device__ __forceinline__ Accumulator initial_accumulator(int thread_id,
                                                           int index) {
  if constexpr (std::is_same<Accumulator, float>::value) {
    return static_cast<float>((thread_id & 0x7) + index) * 0.001f;
  } else {
    return static_cast<Accumulator>((thread_id & 0x7) + index);
  }
}

template <int N>
struct WgmmaF16Ss;

template <>
struct WgmmaF16Ss<8> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n8k16.f32.f16.f16.ss"; }
  __device__ __forceinline__ static void exec(uint64_t desc_a, uint64_t desc_b,
                                              const uint32_t (&)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0),
          "n"(0));
  }
};

template <>
struct WgmmaF16Ss<16> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n16k16.f32.f16.f16.ss"; }
  __device__ __forceinline__ static void exec(uint64_t desc_a, uint64_t desc_b,
                                              const uint32_t (&)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0),
          "n"(0));
  }
};

template <>
struct WgmmaF16Ss<32> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n32k16.f32.f16.f16.ss"; }
  __device__ __forceinline__ static void exec(uint64_t desc_a, uint64_t desc_b,
                                              const uint32_t (&)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, p, %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0),
          "n"(0));
  }
};

template <>
struct WgmmaF16Ss<64> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n64k16.f32.f16.f16.ss"; }
  __device__ __forceinline__ static void exec(uint64_t desc_a, uint64_t desc_b,
                                              const uint32_t (&)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
          "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0),
          "n"(0));
  }
};

template <int N>
struct WgmmaF16Rs;

template <>
struct WgmmaF16Rs<8> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n8k16.f32.f16.f16.rs"; }
  __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,
                                              const uint32_t (&a_regs)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }
};

template <>
struct WgmmaF16Rs<16> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n16k16.f32.f16.f16.rs"; }
  __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,
                                              const uint32_t (&a_regs)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "%12, p, %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }
};

template <>
struct WgmmaF16Rs<32> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n32k16.f32.f16.f16.rs"; }
  __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,
                                              const uint32_t (&a_regs)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, "
        "%20, p, %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15])
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }
};

template <>
struct WgmmaF16Rs<64> {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n64k16.f32.f16.f16.rs"; }
  __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,
                                              const uint32_t (&a_regs)[4],
                                              float (&d)[32]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, "
        "%36, p, %38, %39, %40;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
          "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }
};

#define WGMMA_FLOAT_D4 "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
#define WGMMA_FLOAT_D8                                                        \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),              \
      "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
#define WGMMA_FLOAT_D32                                                       \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),              \
      "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),          \
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),     \
      "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),     \
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),     \
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),     \
      "+f"(d[30]), "+f"(d[31])
#define WGMMA_INT_D4 "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
#define WGMMA_INT_D32                                                         \
  "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]),              \
      "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]),          \
      "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),     \
      "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),     \
      "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),     \
      "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),     \
      "+r"(d[30]), "+r"(d[31])

#define DEFINE_FLOAT_RS_3IMM(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)         \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".f32." TYPES ".rs";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %9, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".f32." TYPES " "    \
          "{%0, %1, %2, %3}, "                                                \
          "{%4, %5, %6, %7}, "                                                \
          "%8, p, %10, %11, %12;\n"                                          \
          "}\n"                                                               \
          : WGMMA_FLOAT_D4                                                    \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),      \
            "n"(0));                                                          \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".f32." TYPES ".rs";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %37, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".f32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "{%32, %33, %34, %35}, "                                            \
          "%36, p, %38, %39, %40;\n"                                         \
          "}\n"                                                               \
          : WGMMA_FLOAT_D32                                                   \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),      \
            "n"(0));                                                          \
    }                                                                         \
  }

#define DEFINE_FLOAT_RS_2IMM(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)         \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".f32." TYPES ".rs";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %9, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".f32." TYPES " "    \
          "{%0, %1, %2, %3}, "                                                \
          "{%4, %5, %6, %7}, "                                                \
          "%8, p, %10, %11;\n"                                               \
          "}\n"                                                               \
          : WGMMA_FLOAT_D4                                                    \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1));      \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".f32." TYPES ".rs";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %37, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".f32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "{%32, %33, %34, %35}, "                                            \
          "%36, p, %38, %39;\n"                                              \
          "}\n"                                                               \
          : WGMMA_FLOAT_D32                                                   \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1));      \
    }                                                                         \
  }

#define DEFINE_FLOAT_SS_4IMM(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)         \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".f32." TYPES ".ss";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %6, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".f32." TYPES " "    \
          "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"                  \
          "}\n"                                                               \
          : WGMMA_FLOAT_D4                                                    \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),         \
            "n"(0), "n"(0));                                                  \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".f32." TYPES ".ss";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %34, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".f32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "%32, %33, p, %35, %36, %37, %38;\n"                               \
          "}\n"                                                               \
          : WGMMA_FLOAT_D32                                                   \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),         \
            "n"(0), "n"(0));                                                  \
    }                                                                         \
  }

#define DEFINE_FLOAT_SS_2IMM(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)         \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".f32." TYPES ".ss";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %6, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".f32." TYPES " "    \
          "{%0, %1, %2, %3}, %4, %5, p, %7, %8;\n"                           \
          "}\n"                                                               \
          : WGMMA_FLOAT_D4                                                    \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1));         \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = float;                                                \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".f32." TYPES ".ss";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                float (&d)[32]) {             \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %34, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".f32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "%32, %33, p, %35, %36;\n"                                         \
          "}\n"                                                               \
          : WGMMA_FLOAT_D32                                                   \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1));         \
    }                                                                         \
  }

#define DEFINE_INT_RS(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)                \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = int32_t;                                              \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".s32." TYPES ".rs";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                int32_t (&d)[32]) {           \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %9, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".s32." TYPES " "    \
          "{%0, %1, %2, %3}, "                                                \
          "{%4, %5, %6, %7}, "                                                \
          "%8, p;\n"                                                         \
          "}\n"                                                               \
          : WGMMA_INT_D4                                                      \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d));                      \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = int32_t;                                              \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".s32." TYPES ".rs";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t, uint64_t desc_b,    \
                                                const uint32_t (&a_regs)[4],  \
                                                int32_t (&d)[32]) {           \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %37, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".s32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "{%32, %33, %34, %35}, "                                            \
          "%36, p;\n"                                                        \
          "}\n"                                                               \
          : WGMMA_INT_D32                                                     \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                 \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d));                      \
    }                                                                         \
  }

#define DEFINE_INT_SS(OP, TYPES, KVALUE, A_PATTERN, B_PATTERN)                \
  template <int N>                                                            \
  struct OP;                                                                  \
  template <>                                                                 \
  struct OP<8> {                                                              \
    using Accumulator = int32_t;                                              \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n8k" #KVALUE ".s32." TYPES ".ss";                          \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                int32_t (&d)[32]) {           \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %6, 0;\n"                                           \
          "wgmma.mma_async.sync.aligned.m64n8k" #KVALUE ".s32." TYPES " "    \
          "{%0, %1, %2, %3}, %4, %5, p;\n"                                   \
          "}\n"                                                               \
          : WGMMA_INT_D4                                                      \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d));                         \
    }                                                                         \
  };                                                                          \
  template <>                                                                 \
  struct OP<64> {                                                             \
    using Accumulator = int32_t;                                              \
    static constexpr int kK = KVALUE;                                         \
    static constexpr uint32_t kARegPattern = A_PATTERN;                       \
    static constexpr uint32_t kSharedWordPattern = B_PATTERN;                 \
    static const char *name() {                                               \
      return "m64n64k" #KVALUE ".s32." TYPES ".ss";                         \
    }                                                                         \
    __device__ __forceinline__ static void exec(uint64_t desc_a,              \
                                                uint64_t desc_b,              \
                                                const uint32_t (&)[4],        \
                                                int32_t (&d)[32]) {           \
      int32_t scale_d = 1;                                                    \
      asm volatile(                                                           \
          "{\n"                                                               \
          ".reg .pred p;\n"                                                   \
          "setp.ne.b32 p, %34, 0;\n"                                          \
          "wgmma.mma_async.sync.aligned.m64n64k" #KVALUE ".s32." TYPES " "   \
          "{%0, %1, %2, %3, %4, %5, %6, %7, "                                 \
          "%8, %9, %10, %11, %12, %13, %14, %15, "                           \
          "%16, %17, %18, %19, %20, %21, %22, %23, "                         \
          "%24, %25, %26, %27, %28, %29, %30, %31}, "                        \
          "%32, %33, p;\n"                                                   \
          "}\n"                                                               \
          : WGMMA_INT_D32                                                     \
          : "l"(desc_a), "l"(desc_b), "r"(scale_d));                         \
    }                                                                         \
  }

DEFINE_FLOAT_RS_3IMM(WgmmaBF16Rs, "bf16.bf16", 16, 0x3f803f80u,
                     0x3f803f80u);
DEFINE_FLOAT_SS_4IMM(WgmmaBF16Ss, "bf16.bf16", 16, 0x3f803f80u,
                     0x3f803f80u);
DEFINE_FLOAT_RS_2IMM(WgmmaTF32Rs, "tf32.tf32", 8, 0x3f800000u,
                     0x3f800000u);
DEFINE_FLOAT_SS_2IMM(WgmmaTF32Ss, "tf32.tf32", 8, 0x3f800000u,
                     0x3f800000u);
DEFINE_FLOAT_RS_2IMM(WgmmaFP8E4M3Rs, "e4m3.e4m3", 32, 0x38383838u,
                     0x38383838u);
DEFINE_FLOAT_SS_2IMM(WgmmaFP8E4M3Ss, "e4m3.e4m3", 32, 0x38383838u,
                     0x38383838u);
DEFINE_FLOAT_RS_2IMM(WgmmaFP8E5M2Rs, "e5m2.e5m2", 32, 0x3c3c3c3cu,
                     0x3c3c3c3cu);
DEFINE_FLOAT_SS_2IMM(WgmmaFP8E5M2Ss, "e5m2.e5m2", 32, 0x3c3c3c3cu,
                     0x3c3c3c3cu);
DEFINE_FLOAT_RS_2IMM(WgmmaFP8E4M3E5M2Rs, "e4m3.e5m2", 32, 0x38383838u,
                     0x3c3c3c3cu);
DEFINE_FLOAT_SS_2IMM(WgmmaFP8E4M3E5M2Ss, "e4m3.e5m2", 32, 0x38383838u,
                     0x3c3c3c3cu);
DEFINE_FLOAT_RS_2IMM(WgmmaFP8E5M2E4M3Rs, "e5m2.e4m3", 32, 0x3c3c3c3cu,
                     0x38383838u);
DEFINE_FLOAT_SS_2IMM(WgmmaFP8E5M2E4M3Ss, "e5m2.e4m3", 32, 0x3c3c3c3cu,
                     0x38383838u);
DEFINE_INT_RS(WgmmaS8S8Rs, "s8.s8.satfinite", 32, 0x01010101u,
              0x01010101u);
DEFINE_INT_SS(WgmmaS8S8Ss, "s8.s8.satfinite", 32, 0x01010101u,
              0x01010101u);
DEFINE_INT_RS(WgmmaU8U8Rs, "u8.u8", 32, 0x01010101u, 0x01010101u);
DEFINE_INT_SS(WgmmaU8U8Ss, "u8.u8", 32, 0x01010101u, 0x01010101u);
DEFINE_INT_RS(WgmmaS8U8Rs, "s8.u8.satfinite", 32, 0x01010101u,
              0x01010101u);
DEFINE_INT_SS(WgmmaS8U8Ss, "s8.u8.satfinite", 32, 0x01010101u,
              0x01010101u);
DEFINE_INT_RS(WgmmaU8S8Rs, "u8.s8", 32, 0x01010101u, 0x01010101u);
DEFINE_INT_SS(WgmmaU8S8Ss, "u8.s8", 32, 0x01010101u, 0x01010101u);
DEFINE_INT_RS(WgmmaB1Rs, "b1.b1.and.popc", 256, 0xffffffffu, 0xffffffffu);
DEFINE_INT_SS(WgmmaB1Ss, "b1.b1.and.popc", 256, 0xffffffffu, 0xffffffffu);

template <template <int> class WgmmaOp, int N, int GroupsBeforeWait,
          int OpsPerGroup>
__global__ void wgmma_timing_kernel(TimingSample *samples, float *sink_out,
                                    int rounds) {
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  using Op = WgmmaOp<N>;
  using Accumulator = typename Op::Accumulator;

  __shared__ __align__(128) uint32_t smem_a[kSharedWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedWords];

  for (int i = threadIdx.x; i < kSharedWords; i += blockDim.x) {
    smem_a[i] = Op::kSharedWordPattern;
    smem_b[i] = Op::kSharedWordPattern;
  }
  __syncthreads();

  const uint64_t desc_a = make_gmma_swizzle_desc(
      smem_a, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);
  const uint64_t desc_b = make_gmma_swizzle_desc(
      smem_b, kLeadingByteOffset, kStrideByteOffset, kSwizzleMode128B);

  uint32_t a_regs[4];
  make_rs_a_regs<Op>(a_regs);
  touch_a_registers(a_regs);

  Accumulator d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    d[i] = initial_accumulator<Accumulator>(threadIdx.x, i);
  }
  touch_accumulators(d);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  uint64_t total_start = read_clock64();
  for (int round = 0; round < rounds; ++round) {
    uint64_t issue_start = read_clock64();
#pragma unroll
    for (int group = 0; group < GroupsBeforeWait; ++group) {
#pragma unroll
      for (int op = 0; op < OpsPerGroup; ++op) {
        WgmmaOp<N>::exec(desc_a, desc_b, a_regs, d);
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }
    uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  uint64_t total_end = read_clock64();

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  touch_accumulators(d);

  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * GroupsBeforeWait * OpsPerGroup);
    sink_out[blockIdx.x] = static_cast<float>(d[0]);
  }
#else
  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = 0;
    samples[blockIdx.x].wait_cycles = 0;
    samples[blockIdx.x].total_cycles = 0;
    samples[blockIdx.x].wgmma_count = 0;
    sink_out[blockIdx.x] = 0.0f;
  }
#endif
}

uint64_t median_u64(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0 : values[values.size() / 2];
}

template <template <int> class WgmmaOp, int N, int GroupsBeforeWait,
          int OpsPerGroup>
TimingResult run_case(cudaDeviceProp prop) {
  const int blocks = std::max(1, prop.multiProcessorCount);
  TimingSample *d_samples = nullptr;
  float *d_sink = nullptr;
  cudaMalloc(&d_samples, blocks * sizeof(TimingSample));
  cudaMalloc(&d_sink, blocks * sizeof(float));

  for (int i = 0; i < 3; ++i) {
    wgmma_timing_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, 16);
  }
  cudaDeviceSynchronize();

  wgmma_timing_kernel<WgmmaOp, N, GroupsBeforeWait, OpsPerGroup>
      <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, kRounds);
  cudaDeviceSynchronize();

  std::vector<TimingSample> samples(blocks);
  cudaMemcpy(samples.data(), d_samples, blocks * sizeof(TimingSample),
             cudaMemcpyDeviceToHost);
  cudaFree(d_samples);
  cudaFree(d_sink);

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> total_values;
  for (const TimingSample &sample : samples) {
    if (sample.wgmma_count == 0) continue;
    issue_values.push_back(sample.issue_cycles);
    wait_values.push_back(sample.wait_cycles);
    total_values.push_back(sample.total_cycles);
  }

  TimingResult result{};
  result.name = WgmmaOp<N>::name();
  result.n = N;
  result.k = WgmmaOp<N>::kK;
  result.groups = GroupsBeforeWait;
  result.ops_per_group = OpsPerGroup;
  result.rounds = kRounds;
  result.median_issue_cycles = median_u64(issue_values);
  result.median_wait_cycles = median_u64(wait_values);
  result.median_total_cycles = median_u64(total_values);
  const double wgmma_count =
      static_cast<double>(kRounds * GroupsBeforeWait * OpsPerGroup);
  result.issue_cycles_per_wgmma = result.median_issue_cycles / wgmma_count;
  result.wait_cycles_per_round =
      static_cast<double>(result.median_wait_cycles) / kRounds;
  result.total_cycles_per_wgmma = result.median_total_cycles / wgmma_count;
  return result;
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

void write_results_csv(const char *filename,
                       const std::vector<TimingResult> &results) {
  std::ofstream out(filename);
  out << "name,n,k,groups_before_wait,ops_per_group,rounds,"
         "median_issue_cycles,median_wait_cycles,median_total_cycles,"
         "issue_cycles_per_wgmma,wait_cycles_per_round,total_cycles_per_wgmma\n";
  for (const TimingResult &r : results) {
    out << r.name << "," << r.n << "," << r.k << "," << r.groups << ","
        << r.ops_per_group << "," << r.rounds << ","
        << r.median_issue_cycles << "," << r.median_wait_cycles << ","
        << r.median_total_cycles << "," << r.issue_cycles_per_wgmma << ","
        << r.wait_cycles_per_round << "," << r.total_cycles_per_wgmma << "\n";
  }
}

void print_results(const std::vector<TimingResult> &results) {
  printf("name,n,k,groups,ops_per_group,issue_cycles_per_wgmma,"
         "wait_cycles_per_round,total_cycles_per_wgmma\n");
  for (const TimingResult &r : results) {
    printf("%s,%d,%d,%d,%d,%.3f,%.3f,%.3f\n", r.name, r.n, r.k, r.groups,
           r.ops_per_group, r.issue_cycles_per_wgmma,
           r.wait_cycles_per_round, r.total_cycles_per_wgmma);
  }
}

}  // namespace

TEST(WgmmaAsyncLatencyBench, F16SsShapeSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 16, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 32, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 1, 1>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.F16SsShapeSweep.csv", results);
}

TEST(WgmmaAsyncLatencyBench, F16SsM64N64GroupSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 2, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 4, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 1, 2>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 2, 2>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 4, 2>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.F16SsM64N64GroupSweep.csv",
                    results);
}

TEST(WgmmaAsyncLatencyBench, F16RsShapeSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 16, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 32, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 1, 1>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.F16RsShapeSweep.csv", results);
}

TEST(WgmmaAsyncLatencyBench, F16RsM64N64GroupSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 2, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 4, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 1, 2>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 2, 2>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 4, 2>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.F16RsM64N64GroupSweep.csv",
                    results);
}

TEST(WgmmaAsyncLatencyBench, CommonDataTypeRsSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaBF16Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaBF16Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaTF32Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaTF32Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3E5M2Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3E5M2Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2E4M3Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2E4M3Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8S8Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8S8Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8U8Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8U8Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8U8Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8U8Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8S8Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8S8Rs, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaB1Rs, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaB1Rs, 64, 1, 1>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.CommonDataTypeRsSweep.csv",
                    results);
}

TEST(WgmmaAsyncLatencyBench, CommonDataTypeSsSweep) {
  require_hopper_or_newer();
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  std::vector<TimingResult> results;
  results.push_back(run_case<WgmmaF16Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaF16Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaBF16Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaBF16Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaTF32Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaTF32Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3E5M2Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E4M3E5M2Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2E4M3Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaFP8E5M2E4M3Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8S8Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8S8Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8U8Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8U8Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8U8Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaS8U8Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8S8Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaU8S8Ss, 64, 1, 1>(prop));
  results.push_back(run_case<WgmmaB1Ss, 8, 1, 1>(prop));
  results.push_back(run_case<WgmmaB1Ss, 64, 1, 1>(prop));

  print_results(results);
  write_results_csv("WgmmaAsyncLatencyBench.CommonDataTypeSsSweep.csv",
                    results);
}
