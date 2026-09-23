#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kSharedAWords = 2048;
constexpr int kSharedBWords = 8192;
constexpr int kMaxAccumulatorRegisters = 128;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;

enum class MixMode { WgmmaOnly, MathOnly, WgmmaWaitMath, WgmmaMathWait };
enum class AccMode { Accumulate, Overwrite };
enum class MathKind {
  FmaIndep8,
  Ex2Indep4,
  IntAddIndep8,
  Lop3Indep8,
  Mix4,
};

struct TimingSample {
  uint64_t issue_cycles;
  uint64_t wait_cycles;
  uint64_t math_cycles;
  uint64_t total_cycles;
  uint32_t wgmma_count;
};

struct TimingResult {
  std::string case_name;
  std::string mode;
  std::string operand;
  std::string acc_mode;
  std::string math_kind;
  int n;
  int math_ops;
  int math_inst_per_thread;
  int wgmma_ops_per_round;
  int blocks;
  int rounds;
  uint64_t median_issue_cycles;
  uint64_t median_wait_cycles;
  uint64_t median_math_cycles;
  uint64_t median_total_cycles;
  double issue_cycles_per_wgmma;
  double wait_cycles_per_round;
  double math_cycles_per_round;
  double total_cycles_per_round;
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

#define WGMMA_REG_LIST_D128                                                  \
  WGMMA_REG_LIST_D88                                                         \
  ", %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "    \
  "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "      \
  "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "      \
  "%123, %124, %125, %126, %127"

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

#define WGMMA_FLOAT_D128                                                      \
  WGMMA_FLOAT_D88, "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),     \
      "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]),    \
      "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]),  \
      "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]),             \
      "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]),             \
      "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]),             \
      "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]),             \
      "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]),             \
      "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]),             \
      "+f"(d[126]), "+f"(d[127])

#define WGMMA_OUT_FLOAT_D32                                                   \
  "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),              \
      "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),          \
      "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]),    \
      "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),    \
      "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]),    \
      "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),    \
      "=f"(d[30]), "=f"(d[31])

#define WGMMA_OUT_FLOAT_D64                                                   \
  WGMMA_OUT_FLOAT_D32, "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), \
      "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]),    \
      "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]),    \
      "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]),    \
      "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),    \
      "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]),    \
      "=f"(d[61]), "=f"(d[62]), "=f"(d[63])

#define WGMMA_OUT_FLOAT_D88                                                   \
  WGMMA_OUT_FLOAT_D64, "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), \
      "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]),    \
      "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]),    \
      "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]),    \
      "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87])

#define WGMMA_OUT_FLOAT_D128                                                  \
  WGMMA_OUT_FLOAT_D88, "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), \
      "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]), "=f"(d[96]),    \
      "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]), "=f"(d[101]),  \
      "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]),             \
      "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]),             \
      "=f"(d[110]), "=f"(d[111]), "=f"(d[112]), "=f"(d[113]),             \
      "=f"(d[114]), "=f"(d[115]), "=f"(d[116]), "=f"(d[117]),             \
      "=f"(d[118]), "=f"(d[119]), "=f"(d[120]), "=f"(d[121]),             \
      "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]),             \
      "=f"(d[126]), "=f"(d[127])

template <int N>
struct WgmmaF16Ss;

template <int N>
struct WgmmaF16Rs;

#define DEFINE_WGMMA_SS_N(N, D, P_SCALE_D, P_DESC_A, P_DESC_B)              \
  template <>                                                                \
  struct WgmmaF16Ss<N> {                                                     \
    static constexpr int kD = D;                                             \
    static const char *name() { return "ss"; }                              \
    __device__ __forceinline__ static void exec_accumulate(                  \
        uint64_t desc_a, uint64_t desc_b, const uint32_t (&)[4],             \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 1;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST_D##D "}, " P_DESC_A ", " P_DESC_B             \
          ", p, 1, 1, 0, 0;\n"                                              \
          "}\n"                                                             \
          : WGMMA_FLOAT_D##D                                                 \
          : "r"(scale_d), "l"(desc_a), "l"(desc_b));                      \
    }                                                                        \
    __device__ __forceinline__ static void exec_overwrite(                   \
        uint64_t desc_a, uint64_t desc_b, const uint32_t (&)[4],             \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 0;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST_D##D "}, " P_DESC_A ", " P_DESC_B             \
          ", p, 1, 1, 0, 0;\n"                                              \
          "}\n"                                                             \
          : WGMMA_OUT_FLOAT_D##D                                             \
          : "r"(scale_d), "l"(desc_a), "l"(desc_b));                      \
    }                                                                        \
  }

#define DEFINE_WGMMA_RS_N(N, D, P_A0, P_A1, P_A2, P_A3, P_DESC_B,           \
                          P_SCALE_D)                                        \
  template <>                                                                \
  struct WgmmaF16Rs<N> {                                                     \
    static constexpr int kD = D;                                             \
    static const char *name() { return "rs"; }                              \
    __device__ __forceinline__ static void exec_accumulate(                  \
        uint64_t, uint64_t desc_b, const uint32_t (&a_regs)[4],              \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 1;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST_D##D "}, {" P_A0 ", " P_A1 ", " P_A2 ", "   \
          P_A3 "}, " P_DESC_B ", p, 1, 1, 0;\n"                            \
          "}\n"                                                             \
          : WGMMA_FLOAT_D##D                                                 \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d));                   \
    }                                                                        \
    __device__ __forceinline__ static void exec_overwrite(                   \
        uint64_t, uint64_t desc_b, const uint32_t (&a_regs)[4],              \
        float (&d)[kMaxAccumulatorRegisters]) {                              \
      int32_t scale_d = 0;                                                   \
      asm volatile(                                                          \
          "{\n"                                                             \
          ".reg .pred p;\n"                                                 \
          "setp.ne.b32 p, " P_SCALE_D ", 0;\n"                             \
          "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.f16.f16 "        \
          "{" WGMMA_REG_LIST_D##D "}, {" P_A0 ", " P_A1 ", " P_A2 ", "   \
          P_A3 "}, " P_DESC_B ", p, 1, 1, 0;\n"                            \
          "}\n"                                                             \
          : WGMMA_OUT_FLOAT_D##D                                             \
          : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]),                \
            "r"(a_regs[3]), "l"(desc_b), "r"(scale_d));                   \
    }                                                                        \
  }

DEFINE_WGMMA_SS_N(128, 64, "%64", "%65", "%66");
DEFINE_WGMMA_SS_N(176, 88, "%88", "%89", "%90");
DEFINE_WGMMA_SS_N(256, 128, "%128", "%129", "%130");

DEFINE_WGMMA_RS_N(128, 64, "%64", "%65", "%66", "%67", "%68", "%69");
DEFINE_WGMMA_RS_N(176, 88, "%88", "%89", "%90", "%91", "%92", "%93");
DEFINE_WGMMA_RS_N(256, 128, "%128", "%129", "%130", "%131", "%132",
                  "%133");

template <typename Op, AccMode Mode>
__device__ __forceinline__ void issue_wgmma(uint64_t desc_a, uint64_t desc_b,
                                            const uint32_t (&a_regs)[4],
                                            float (&d)[kMaxAccumulatorRegisters]) {
  if constexpr (Mode == AccMode::Accumulate) {
    Op::exec_accumulate(desc_a, desc_b, a_regs, d);
  } else {
    Op::exec_overwrite(desc_a, desc_b, a_regs, d);
  }
}

template <typename Op, AccMode Mode>
__device__ __forceinline__ uint64_t issue_wgmma_group(
    uint64_t desc_a, uint64_t desc_b, const uint32_t (&a_regs)[4],
    float (&d)[kMaxAccumulatorRegisters], int wgmma_ops) {
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < wgmma_ops; ++op) {
    issue_wgmma<Op, Mode>(desc_a, desc_b, a_regs, d);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t wait_wgmma_group_0() {
  const uint64_t start = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_fma_indep8(int math_ops, float &a0,
                                                   float &a1, float &a2,
                                                   float &a3, float &a4,
                                                   float &a5, float &a6,
                                                   float &a7) {
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
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_ex2_indep4(int math_ops, float &a0,
                                                   float &a1, float &a2,
                                                   float &a3) {
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
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_int_add_indep8(
    int math_ops, uint32_t &i0, uint32_t &i1, uint32_t &i2, uint32_t &i3,
    uint32_t &i4, uint32_t &i5, uint32_t &i6, uint32_t &i7) {
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
        : "+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3), "+r"(i4),
          "+r"(i5), "+r"(i6), "+r"(i7)
        : "r"(rhs));
  }
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_lop3_indep8(
    int math_ops, uint32_t &i0, uint32_t &i1, uint32_t &i2, uint32_t &i3,
    uint32_t &i4, uint32_t &i5, uint32_t &i6, uint32_t &i7) {
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
        : "+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3), "+r"(i4),
          "+r"(i5), "+r"(i6), "+r"(i7)
        : "r"(mask), "r"(salt));
  }
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_mix4(int math_ops, float &a0,
                                             float &a1) {
  const float rhs = a1 + 0.25f;
  const float scale = 0.9375f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int op = 0; op < math_ops; ++op) {
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
  return read_clock64() - start;
}

__device__ __forceinline__ uint64_t run_math_block(
    int math_kind, int math_ops, float &a0, float &a1, float &a2, float &a3,
    float &a4, float &a5, float &a6, float &a7, uint32_t &i0, uint32_t &i1,
    uint32_t &i2, uint32_t &i3, uint32_t &i4, uint32_t &i5, uint32_t &i6,
    uint32_t &i7) {
  switch (static_cast<MathKind>(math_kind)) {
    case MathKind::FmaIndep8:
      return run_fma_indep8(math_ops, a0, a1, a2, a3, a4, a5, a6, a7);
    case MathKind::Ex2Indep4:
      return run_ex2_indep4(math_ops, a0, a1, a2, a3);
    case MathKind::IntAddIndep8:
      return run_int_add_indep8(math_ops, i0, i1, i2, i3, i4, i5, i6, i7);
    case MathKind::Lop3Indep8:
      return run_lop3_indep8(math_ops, i0, i1, i2, i3, i4, i5, i6, i7);
    case MathKind::Mix4:
      return run_mix4(math_ops, a0, a1);
  }
  return 0;
}

template <MixMode Mode, template <int> class WgmmaOp, int N, AccMode Acc>
__global__ void rf_bandwidth_kernel(TimingSample *samples, float *sink_out,
                                    int rounds, int wgmma_ops, int math_kind,
                                    int math_ops) {
#if __CUDA_ARCH__ >= 900
  using Op = WgmmaOp<N>;
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
  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3])::"memory");

  float d[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < kMaxAccumulatorRegisters; ++i) {
    d[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
  }

  float f0 = static_cast<float>((threadIdx.x + 1) & 0x1f) * 0.03125f;
  float f1 = static_cast<float>((threadIdx.x + 3) & 0x1f) * 0.03125f;
  float f2 = static_cast<float>((threadIdx.x + 5) & 0x1f) * 0.03125f;
  float f3 = static_cast<float>((threadIdx.x + 7) & 0x1f) * 0.03125f;
  float f4 = static_cast<float>((threadIdx.x + 11) & 0x1f) * 0.03125f;
  float f5 = static_cast<float>((threadIdx.x + 13) & 0x1f) * 0.03125f;
  float f6 = static_cast<float>((threadIdx.x + 17) & 0x1f) * 0.03125f;
  float f7 = static_cast<float>((threadIdx.x + 19) & 0x1f) * 0.03125f;
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

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  uint64_t math_cycles = 0;
  const uint64_t total_start = read_clock64();

#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    if constexpr (Mode == MixMode::WgmmaOnly) {
      issue_cycles +=
          issue_wgmma_group<Op, Acc>(desc_a, desc_b, a_regs, d, wgmma_ops);
      wait_cycles += wait_wgmma_group_0();
    } else if constexpr (Mode == MixMode::MathOnly) {
      math_cycles += run_math_block(math_kind, math_ops, f0, f1, f2, f3, f4,
                                    f5, f6, f7, i0, i1, i2, i3, i4, i5, i6,
                                    i7);
    } else if constexpr (Mode == MixMode::WgmmaWaitMath) {
      issue_cycles +=
          issue_wgmma_group<Op, Acc>(desc_a, desc_b, a_regs, d, wgmma_ops);
      wait_cycles += wait_wgmma_group_0();
      force_u64_ready(issue_cycles);
      force_u64_ready(wait_cycles);
      math_cycles += run_math_block(math_kind, math_ops, f0, f1, f2, f3, f4,
                                    f5, f6, f7, i0, i1, i2, i3, i4, i5, i6,
                                    i7);
    } else if constexpr (Mode == MixMode::WgmmaMathWait) {
      issue_cycles +=
          issue_wgmma_group<Op, Acc>(desc_a, desc_b, a_regs, d, wgmma_ops);
      force_u64_ready(issue_cycles);
      math_cycles += run_math_block(math_kind, math_ops, f0, f1, f2, f3, f4,
                                    f5, f6, f7, i0, i1, i2, i3, i4, i5, i6,
                                    i7);
      wait_cycles += wait_wgmma_group_0();
    }
  }

  const uint64_t total_end = read_clock64();
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  float sink = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
  sink += static_cast<float>((i0 ^ i1 ^ i2 ^ i3 ^ i4 ^ i5 ^ i6 ^ i7) &
                             0xffu);
#pragma unroll
  for (int i = 0; i < Op::kD; ++i) {
    sink += d[i];
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].math_cycles = math_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        (Mode == MixMode::MathOnly)
            ? 0
            : static_cast<uint32_t>(rounds * wgmma_ops);
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

std::vector<std::string> split_csv(std::string text) {
  std::vector<std::string> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), ::isspace),
               item.end());
    if (!item.empty()) values.push_back(item);
  }
  return values;
}

bool list_contains(const std::vector<std::string> &values,
                   const std::string &needle) {
  return std::find(values.begin(), values.end(), needle) != values.end() ||
         std::find(values.begin(), values.end(), "all") != values.end();
}

uint64_t median_u64(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0 : values[values.size() / 2];
}

const char *mode_name(MixMode mode) {
  switch (mode) {
    case MixMode::WgmmaOnly:
      return "wgmma_only";
    case MixMode::MathOnly:
      return "math_only";
    case MixMode::WgmmaWaitMath:
      return "wgmma_wait_math";
    case MixMode::WgmmaMathWait:
      return "wgmma_math_wait";
  }
  return "unknown";
}

const char *math_kind_name(MathKind kind) {
  switch (kind) {
    case MathKind::FmaIndep8:
      return "fma_indep8";
    case MathKind::Ex2Indep4:
      return "ex2_indep4";
    case MathKind::IntAddIndep8:
      return "int_add_indep8";
    case MathKind::Lop3Indep8:
      return "lop3_indep8";
    case MathKind::Mix4:
      return "mix4";
  }
  return "unknown";
}

MathKind parse_math_kind(const std::string &name) {
  if (name == "fma_indep8") return MathKind::FmaIndep8;
  if (name == "ex2_indep4") return MathKind::Ex2Indep4;
  if (name == "int_add_indep8") return MathKind::IntAddIndep8;
  if (name == "lop3_indep8") return MathKind::Lop3Indep8;
  if (name == "mix4") return MathKind::Mix4;
  throw std::runtime_error(
      "unsupported WGMMA_RF_BW_MATH_KINDS entry: " + name);
}

int math_inst_per_thread(MathKind kind, int math_ops) {
  switch (kind) {
    case MathKind::FmaIndep8:
    case MathKind::IntAddIndep8:
    case MathKind::Lop3Indep8:
      return 8 * math_ops;
    case MathKind::Ex2Indep4:
    case MathKind::Mix4:
      return 4 * math_ops;
  }
  return 0;
}

std::string make_case_name(const std::string &mode, const std::string &operand,
                           const std::string &acc_mode, int n,
                           const std::string &math_kind, int math_ops,
                           int wgmma_ops) {
  std::string name = mode;
  if (mode != "math_only") {
    name += "_" + operand + "_n" + std::to_string(n) + "_" + acc_mode +
            "_w" + std::to_string(wgmma_ops);
  }
  if (mode != "wgmma_only") {
    name += "_" + math_kind + "_m" + std::to_string(math_ops);
  }
  return name;
}

bool selected_by_filter(const std::string &filter, const std::string &name) {
  return filter.empty() || filter == "all" || name.find(filter) != std::string::npos;
}

template <MixMode Mode, template <int> class WgmmaOp, int N, AccMode Acc>
TimingResult run_case(cudaDeviceProp prop, int requested_blocks, int rounds,
                      int warmup_rounds, int wgmma_ops, MathKind math_kind,
                      int math_ops, const std::string &operand,
                      const std::string &acc_mode) {
  const int blocks =
      requested_blocks > 0 ? requested_blocks
                           : std::max(1, prop.multiProcessorCount);
  const int effective_rounds = std::max(1, rounds);

  TimingSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, blocks * sizeof(TimingSample)),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, blocks * sizeof(float)), "cudaMalloc sink");

  if (warmup_rounds > 0) {
    rf_bandwidth_kernel<Mode, WgmmaOp, N, Acc>
        <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, warmup_rounds,
                                        wgmma_ops, static_cast<int>(math_kind),
                                        math_ops);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  rf_bandwidth_kernel<Mode, WgmmaOp, N, Acc>
      <<<blocks, kWarpgroupThreads>>>(d_samples, d_sink, effective_rounds,
                                      wgmma_ops, static_cast<int>(math_kind),
                                      math_ops);
  check_cuda(cudaGetLastError(), "timed launch");
  check_cuda(cudaDeviceSynchronize(), "timed synchronize");

  std::vector<TimingSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples,
                        blocks * sizeof(TimingSample),
                        cudaMemcpyDeviceToHost),
             "copy samples");
  check_cuda(cudaFree(d_samples), "cudaFree samples");
  check_cuda(cudaFree(d_sink), "cudaFree sink");

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> math_values;
  std::vector<uint64_t> total_values;
  for (const TimingSample &sample : samples) {
    issue_values.push_back(sample.issue_cycles);
    wait_values.push_back(sample.wait_cycles);
    math_values.push_back(sample.math_cycles);
    total_values.push_back(sample.total_cycles);
  }
  if (total_values.empty()) {
    throw std::runtime_error(std::string(mode_name(Mode)) +
                             ": no timing samples");
  }

  TimingResult result{};
  result.mode = mode_name(Mode);
  result.operand = Mode == MixMode::MathOnly ? "none" : operand;
  result.acc_mode = Mode == MixMode::MathOnly ? "none" : acc_mode;
  result.n = Mode == MixMode::MathOnly ? 0 : N;
  result.math_kind =
      Mode == MixMode::WgmmaOnly ? "none" : math_kind_name(math_kind);
  result.math_ops = Mode == MixMode::WgmmaOnly ? 0 : math_ops;
  result.math_inst_per_thread =
      Mode == MixMode::WgmmaOnly ? 0 : math_inst_per_thread(math_kind, math_ops);
  result.wgmma_ops_per_round = Mode == MixMode::MathOnly ? 0 : wgmma_ops;
  result.blocks = blocks;
  result.rounds = effective_rounds;
  result.median_issue_cycles = median_u64(issue_values);
  result.median_wait_cycles = median_u64(wait_values);
  result.median_math_cycles = median_u64(math_values);
  result.median_total_cycles = median_u64(total_values);
  const double wgmma_count =
      static_cast<double>(effective_rounds * std::max(1, wgmma_ops));
  result.issue_cycles_per_wgmma =
      result.median_issue_cycles == 0
          ? 0.0
          : static_cast<double>(result.median_issue_cycles) / wgmma_count;
  result.wait_cycles_per_round =
      static_cast<double>(result.median_wait_cycles) / effective_rounds;
  result.math_cycles_per_round =
      static_cast<double>(result.median_math_cycles) / effective_rounds;
  result.total_cycles_per_round =
      static_cast<double>(result.median_total_cycles) / effective_rounds;
  result.case_name = make_case_name(
      result.mode, result.operand, result.acc_mode, result.n, result.math_kind,
      result.math_ops, result.wgmma_ops_per_round);
  return result;
}

template <MixMode Mode, template <int> class WgmmaOp, int N, AccMode Acc>
void append_wgmma_math_case(cudaDeviceProp prop, int blocks, int rounds,
                            int warmup_rounds, int wgmma_ops,
                            MathKind math_kind, int math_ops,
                            const std::string &operand,
                            const std::string &acc_mode,
                            const std::string &filter,
                            std::vector<TimingResult> *results) {
  const std::string candidate = make_case_name(
      mode_name(Mode), operand, acc_mode, N,
      Mode == MixMode::WgmmaOnly ? "none" : math_kind_name(math_kind),
      Mode == MixMode::WgmmaOnly ? 0 : math_ops, wgmma_ops);
  if (!selected_by_filter(filter, candidate)) return;
  TimingResult result =
      run_case<Mode, WgmmaOp, N, Acc>(prop, blocks, rounds, warmup_rounds,
                                      wgmma_ops, math_kind, math_ops, operand,
                                      acc_mode);
  results->push_back(std::move(result));
}

template <template <int> class WgmmaOp, int N, AccMode Acc>
void append_wgmma_sweep(cudaDeviceProp prop, int blocks, int rounds,
                        int warmup_rounds, int wgmma_ops,
                        const std::vector<MathKind> &math_kinds,
                        const std::vector<int> &math_ops_values,
                        const std::vector<std::string> &modes,
                        const std::string &operand,
                        const std::string &acc_mode,
                        const std::string &filter,
                        std::vector<TimingResult> *results) {
  if (list_contains(modes, "wgmma_only")) {
    append_wgmma_math_case<MixMode::WgmmaOnly, WgmmaOp, N, Acc>(
        prop, blocks, rounds, warmup_rounds, wgmma_ops, MathKind::FmaIndep8, 0,
        operand, acc_mode, filter, results);
  }
  for (MathKind math_kind : math_kinds) {
    for (int math_ops : math_ops_values) {
      if (list_contains(modes, "wgmma_wait_math")) {
        append_wgmma_math_case<MixMode::WgmmaWaitMath, WgmmaOp, N, Acc>(
            prop, blocks, rounds, warmup_rounds, wgmma_ops, math_kind,
            math_ops, operand, acc_mode, filter, results);
      }
      if (list_contains(modes, "wgmma_math_wait")) {
        append_wgmma_math_case<MixMode::WgmmaMathWait, WgmmaOp, N, Acc>(
            prop, blocks, rounds, warmup_rounds, wgmma_ops, math_kind,
            math_ops, operand, acc_mode, filter, results);
      }
    }
  }
}

template <template <int> class WgmmaOp, int N>
void append_operand_shape(cudaDeviceProp prop, int blocks, int rounds,
                          int warmup_rounds, int wgmma_ops,
                          const std::vector<MathKind> &math_kinds,
                          const std::vector<int> &math_ops_values,
                          const std::vector<std::string> &modes,
                          const std::vector<std::string> &acc_modes,
                          const std::string &operand,
                          const std::string &filter,
                          std::vector<TimingResult> *results) {
  if (list_contains(acc_modes, "accumulate")) {
    append_wgmma_sweep<WgmmaOp, N, AccMode::Accumulate>(
        prop, blocks, rounds, warmup_rounds, wgmma_ops, math_kinds,
        math_ops_values, modes, operand, "accumulate", filter, results);
  }
  if (list_contains(acc_modes, "overwrite")) {
    append_wgmma_sweep<WgmmaOp, N, AccMode::Overwrite>(
        prop, blocks, rounds, warmup_rounds, wgmma_ops, math_kinds,
        math_ops_values, modes, operand, "overwrite", filter, results);
  }
}

template <template <int> class WgmmaOp>
void append_operand(cudaDeviceProp prop, int blocks, int rounds,
                    int warmup_rounds, int wgmma_ops,
                    const std::vector<MathKind> &math_kinds,
                    const std::vector<int> &math_ops_values,
                    const std::vector<std::string> &modes,
                    const std::vector<std::string> &acc_modes,
                    const std::vector<std::string> &shapes,
                    const std::string &operand,
                    const std::string &filter,
                    std::vector<TimingResult> *results) {
  if (list_contains(shapes, "128")) {
    append_operand_shape<WgmmaOp, 128>(prop, blocks, rounds, warmup_rounds,
                                       wgmma_ops, math_kinds, math_ops_values,
                                       modes, acc_modes, operand, filter,
                                       results);
  }
  if (list_contains(shapes, "176")) {
    append_operand_shape<WgmmaOp, 176>(prop, blocks, rounds, warmup_rounds,
                                       wgmma_ops, math_kinds, math_ops_values,
                                       modes, acc_modes, operand, filter,
                                       results);
  }
  if (list_contains(shapes, "256")) {
    append_operand_shape<WgmmaOp, 256>(prop, blocks, rounds, warmup_rounds,
                                       wgmma_ops, math_kinds, math_ops_values,
                                       modes, acc_modes, operand, filter,
                                       results);
  }
}

void append_math_only(cudaDeviceProp prop, int blocks, int rounds,
                      int warmup_rounds,
                      const std::vector<MathKind> &math_kinds,
                      const std::vector<int> &math_ops_values,
                      const std::vector<std::string> &modes,
                      const std::string &filter,
                      std::vector<TimingResult> *results) {
#ifndef FLASHGPU_SIM_REPRESENTATIVE
  if (!list_contains(modes, "math_only")) return;
  for (MathKind math_kind : math_kinds) {
    for (int math_ops : math_ops_values) {
      const std::string candidate =
          make_case_name("math_only", "none", "none", 0,
                         math_kind_name(math_kind), math_ops, 0);
      if (!selected_by_filter(filter, candidate)) continue;
      TimingResult result =
          run_case<MixMode::MathOnly, WgmmaF16Ss, 128, AccMode::Accumulate>(
              prop, blocks, rounds, warmup_rounds, 1, math_kind, math_ops,
              "none", "none");
      results->push_back(std::move(result));
    }
  }
#endif
}

void write_csv(const std::string &path, const std::vector<TimingResult> &results) {
  std::ofstream out(path);
  out << "case,mode,operand,acc_mode,n,math_kind,math_ops,"
         "math_inst_per_thread,wgmma_ops_per_round,blocks,rounds,"
         "median_issue_cycles,median_wait_cycles,median_math_cycles,"
         "median_total_cycles,issue_cycles_per_wgmma,wait_cycles_per_round,"
         "math_cycles_per_round,total_cycles_per_round\n";
  for (const TimingResult &r : results) {
    out << r.case_name << "," << r.mode << "," << r.operand << ","
        << r.acc_mode << "," << r.n << "," << r.math_kind << ","
        << r.math_ops << "," << r.math_inst_per_thread << ","
        << r.wgmma_ops_per_round << "," << r.blocks << "," << r.rounds
        << "," << r.median_issue_cycles << "," << r.median_wait_cycles
        << "," << r.median_math_cycles << "," << r.median_total_cycles
        << "," << r.issue_cycles_per_wgmma << ","
        << r.wait_cycles_per_round << "," << r.math_cycles_per_round << ","
        << r.total_cycles_per_round << "\n";
  }
}

void print_results(const std::vector<TimingResult> &results) {
  printf("case,mode,operand,acc_mode,n,math_kind,math_ops,wgmma_ops,"
         "issue_per_wgmma,wait_per_round,math_per_round,total_per_round\n");
  for (const TimingResult &r : results) {
    printf("%s,%s,%s,%s,%d,%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
           r.case_name.c_str(), r.mode.c_str(), r.operand.c_str(),
           r.acc_mode.c_str(), r.n, r.math_kind.c_str(), r.math_ops,
           r.wgmma_ops_per_round, r.issue_cycles_per_wgmma,
           r.wait_cycles_per_round, r.math_cycles_per_round,
           r.total_cycles_per_round);
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

std::vector<MathKind> parse_math_kinds(const std::string &text) {
  std::vector<std::string> names = split_csv(text);
  std::vector<MathKind> result;
  if (list_contains(names, "all")) {
    result = {MathKind::FmaIndep8, MathKind::IntAddIndep8,
              MathKind::Lop3Indep8, MathKind::Ex2Indep4, MathKind::Mix4};
    return result;
  }
  for (const std::string &name : names) {
    result.push_back(parse_math_kind(name));
  }
  return result;
}

std::vector<int> parse_int_list(const std::string &text) {
  std::vector<int> result;
  for (const std::string &item : split_csv(text)) {
    char *end = nullptr;
    long value = std::strtol(item.c_str(), &end, 10);
    if (end == item.c_str() || *end != '\0' || value < 0 || value > INT_MAX) {
      throw std::runtime_error("invalid integer list entry: " + item);
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

}  // namespace

TEST(WgmmaRfBandwidth, Sweep) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_RF_BW_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_RF_BW_ROUNDS", 4096);
  const int warmup_rounds = get_env_int("WGMMA_RF_BW_WARMUP_ROUNDS", 32);
  const int wgmma_ops = get_env_int("WGMMA_RF_BW_WGMMA_OPS", 8);
  const std::string filter = get_env_string("WGMMA_RF_BW_FILTER", "all");
  const std::string prefix =
      get_env_string("WGMMA_RF_BW_OUT_PREFIX", "wgmma_rf_bandwidth");

  std::vector<std::string> operands =
      split_csv(get_env_string("WGMMA_RF_BW_OPERANDS", "ss,rs"));
  std::vector<std::string> acc_modes =
      split_csv(get_env_string("WGMMA_RF_BW_ACC_MODES",
                               "accumulate,overwrite"));
  std::vector<std::string> shapes =
      split_csv(get_env_string("WGMMA_RF_BW_SHAPES", "256"));
  std::vector<std::string> modes = split_csv(get_env_string(
      "WGMMA_RF_BW_MODES",
      "math_only,wgmma_only,wgmma_wait_math,wgmma_math_wait"));
  std::vector<MathKind> math_kinds = parse_math_kinds(get_env_string(
      "WGMMA_RF_BW_MATH_KINDS",
      "fma_indep8,int_add_indep8,lop3_indep8,ex2_indep4,mix4"));
  std::vector<int> math_ops_values =
      parse_int_list(get_env_string("WGMMA_RF_BW_MATH_OPS",
                                    "0,1,2,4,8,16,24,32,64"));

  std::vector<TimingResult> results;
#ifdef FLASHGPU_SIM_REPRESENTATIVE
  results.push_back(run_case<MixMode::WgmmaOnly, WgmmaF16Ss, 256,
                             AccMode::Accumulate>(
      prop, blocks, rounds, warmup_rounds, wgmma_ops, MathKind::FmaIndep8, 0,
      "ss", "accumulate"));
#else
  append_math_only(prop, blocks, rounds, warmup_rounds, math_kinds,
                   math_ops_values, modes, filter, &results);
  if (list_contains(operands, "ss")) {
    append_operand<WgmmaF16Ss>(prop, blocks, rounds, warmup_rounds, wgmma_ops,
                               math_kinds, math_ops_values, modes, acc_modes,
                               shapes, "ss", filter, &results);
  }
  if (list_contains(operands, "rs")) {
    append_operand<WgmmaF16Rs>(prop, blocks, rounds, warmup_rounds, wgmma_ops,
                               math_kinds, math_ops_values, modes, acc_modes,
                               shapes, "rs", filter, &results);
  }
#endif

  ASSERT_FALSE(results.empty()) << "WGMMA_RF_BW_FILTER matched no cases";
  print_results(results);
  write_csv(prefix + ".csv", results);
}
