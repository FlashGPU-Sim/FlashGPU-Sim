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

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kMixedBlockThreads = 2 * kWarpgroupThreads;
constexpr int kSharedAWords = 2048;
constexpr int kSharedBWords = 8192;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;
constexpr int kMaxAccumulatorRegisters = 64;
constexpr int kMixedSoftmaxRegs = 32;
constexpr int kMixedWgmmaOpsPerRound = 16;
constexpr int kMixedWgmmaN = 128;

struct MixedSoftmaxSample {
  uint64_t wgmma_issue_cycles;
  uint64_t wgmma_wait_cycles;
  uint64_t softmax_cycles;
  uint64_t total_cycles;
  uint32_t wgmma_count;
  uint32_t block_id;
};

struct MixedSoftmaxResult {
  std::string case_name;
  std::string op_name;
  int math_iters;
  int blocks;
  int rounds;
  int wgmma_ops_per_round;
  uint64_t flops_per_block;
  uint64_t median_wgmma_issue_cycles;
  uint64_t median_wgmma_wait_cycles;
  uint64_t median_softmax_cycles;
  uint64_t median_total_cycles;
  double issue_cycles_per_wgmma;
  double total_cycles_per_wgmma;
  double flops_per_total_cycle;
};

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ int make_warp_uniform_i32(int value) {
#if __CUDA_ARCH__ >= 700
  return __shfl_sync(0xffffffffu, value, 0);
#else
  return value;
#endif
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cuda(cudaError_t status, const std::string &what) {
  check_cuda(status, what.c_str());
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

void require_hopper_or_newer() {
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  if (prop.major < 9) {
    GTEST_SKIP() << "WGMMA requires SM90 or newer hardware.";
  }
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

__device__ __forceinline__ void touch_a_registers(uint32_t (&a_regs)[4]) {
  asm volatile(""
               : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3])::"memory");
}

#define WGMMA_REG_LIST_D64                                                   \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "  \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, " \
  "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, " \
  "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, " \
  "%58, %59, %60, %61, %62, %63"

#define WGMMA_FLOAT_D64                                                       \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),              \
      "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),         \
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),                \
      "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),                \
      "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),                \
      "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),                \
      "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),                \
      "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]),                \
      "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]),                \
      "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),                \
      "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),                \
      "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]),                \
      "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]),                \
      "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]),                \
      "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]),                \
      "+f"(d[62]), "+f"(d[63])

__device__ __forceinline__ void touch_accumulators(
    float (&d)[kMaxAccumulatorRegisters]) {
  asm volatile("" : WGMMA_FLOAT_D64 : : "memory");
}

__device__ __forceinline__ float initial_accumulator(int thread_id, int index) {
  return static_cast<float>((thread_id & 0x7) + index) * 0.001f;
}

struct WgmmaF16Ss128 {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr int kD = 64;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n128k16.f32.f16.f16.ss"; }

  __device__ __forceinline__ static void exec(
      uint64_t desc_a, uint64_t desc_b, const uint32_t (&)[4],
      float (&d)[kMaxAccumulatorRegisters]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{" WGMMA_REG_LIST_D64 "}, %64, %65, p, %67, %68, %69, %70;\n"
        "}\n"
        : WGMMA_FLOAT_D64
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1),
          "n"(0), "n"(0));
  }
};

struct WgmmaF16Rs128 {
  using Accumulator = float;
  static constexpr int kK = 16;
  static constexpr int kD = 64;
  static constexpr uint32_t kARegPattern = 0x3c003c00u;
  static constexpr uint32_t kSharedWordPattern = 0x3c003c00u;
  static const char *name() { return "m64n128k16.f32.f16.f16.rs"; }

  __device__ __forceinline__ static void exec(
      uint64_t, uint64_t desc_b, const uint32_t (&a_regs)[4],
      float (&d)[kMaxAccumulatorRegisters]) {
    int32_t scale_d = 1;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{" WGMMA_REG_LIST_D64 "}, {%64, %65, %66, %67}, %68, p, %70, %71, "
        "%72;\n"
        "}\n"
        : WGMMA_FLOAT_D64
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }
};

template <typename Op>
__device__ __forceinline__ void make_rs_a_regs(uint32_t (&a_regs)[4]) {
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    a_regs[reg] = Op::kARegPattern;
  }
}

__device__ __forceinline__ void touch_softmax_regs(
    float (&x)[kMixedSoftmaxRegs]) {
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

__device__ __forceinline__ float sum_softmax_regs(
    const float (&x)[kMixedSoftmaxRegs]) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kMixedSoftmaxRegs; ++i) {
    sum += x[i];
  }
  return sum;
}

template <int MathIters>
__device__ __forceinline__ uint64_t run_softmax_math_stress(
    float (&x)[kMixedSoftmaxRegs]) {
  if constexpr (MathIters <= 0) {
    touch_softmax_regs(x);
    return 0;
  }

  const float scale = 0.9375f;
  const uint64_t start = read_clock64();
#pragma unroll 1
  for (int iter = 0; iter < MathIters; ++iter) {
    float row_max = x[0];
#pragma unroll
    for (int i = 1; i < kMixedSoftmaxRegs; ++i) {
      float next_max;
      asm volatile("max.ftz.f32 %0, %1, %2;\n"
                   : "=f"(next_max)
                   : "f"(row_max), "f"(x[i]));
      row_max = next_max;
    }

    float row_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kMixedSoftmaxRegs; ++i) {
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
                     "f"(x[(i + 7) & (kMixedSoftmaxRegs - 1)]));
      row_sum = next_sum;
      x[i] = updated;
    }

    const float norm = row_sum + 1.0f;
#pragma unroll
    for (int i = 0; i < kMixedSoftmaxRegs; ++i) {
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

template <typename Op, int MathIters>
__global__ void mixed_wgmma_softmax_kernel(MixedSoftmaxSample *samples,
                                           float *sink_out, int rounds) {
#if __CUDA_ARCH__ >= 900
  using Accumulator = typename Op::Accumulator;

  const int warpgroup_idx =
      make_warp_uniform_i32(static_cast<int>(threadIdx.x) / kWarpgroupThreads);
  const bool mma_warpgroup = warpgroup_idx == 0;
  const int local_thread =
      static_cast<int>(threadIdx.x) - warpgroup_idx * kWarpgroupThreads;

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

  Accumulator d[kMaxAccumulatorRegisters];
#pragma unroll
  for (int i = 0; i < Op::kD; ++i) {
    d[i] = initial_accumulator(local_thread, i);
  }

  float softmax_regs[kMixedSoftmaxRegs];
#pragma unroll
  for (int i = 0; i < kMixedSoftmaxRegs; ++i) {
    softmax_regs[i] =
        static_cast<float>((local_thread + 3 * i) & 0x1f) * 0.03125f;
  }

  if (mma_warpgroup) {
    touch_a_registers(a_regs);
    touch_accumulators(d);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  } else {
    touch_softmax_regs(softmax_regs);
  }
  __syncthreads();

  uint64_t wgmma_issue_cycles = 0;
  uint64_t wgmma_wait_cycles = 0;
  uint64_t softmax_cycles = 0;
  const uint64_t total_start = read_clock64();

  for (int round = 0; round < rounds; ++round) {
    __syncthreads();
    if (mma_warpgroup) {
      const uint64_t issue_start = read_clock64();
#pragma unroll
      for (int op = 0; op < kMixedWgmmaOpsPerRound; ++op) {
        Op::exec(desc_a, desc_b, a_regs, d);
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      const uint64_t wait_start = read_clock64();
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      const uint64_t wait_end = read_clock64();
      wgmma_issue_cycles += wait_start - issue_start;
      wgmma_wait_cycles += wait_end - wait_start;
    } else {
      softmax_cycles += run_softmax_math_stress<MathIters>(softmax_regs);
    }
    __syncthreads();
  }

  const uint64_t total_end = read_clock64();
  if (mma_warpgroup) {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    touch_accumulators(d);
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].wgmma_issue_cycles = wgmma_issue_cycles;
    samples[blockIdx.x].wgmma_wait_cycles = wgmma_wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * kMixedWgmmaOpsPerRound);
    samples[blockIdx.x].block_id = blockIdx.x;
    sink_out[blockIdx.x] = d[0];
  } else if (threadIdx.x == kWarpgroupThreads) {
    samples[blockIdx.x].softmax_cycles = softmax_cycles;
    sink_out[gridDim.x + blockIdx.x] = sum_softmax_regs(softmax_regs);
  }
#else
  if (threadIdx.x == 0) {
    samples[blockIdx.x].wgmma_issue_cycles = 0;
    samples[blockIdx.x].wgmma_wait_cycles = 0;
    samples[blockIdx.x].softmax_cycles = 0;
    samples[blockIdx.x].total_cycles = 0;
    samples[blockIdx.x].wgmma_count = 0;
    samples[blockIdx.x].block_id = blockIdx.x;
    sink_out[blockIdx.x] = 0.0f;
  }
#endif
}

template <typename Op, int MathIters>
MixedSoftmaxResult run_mixed_softmax_case(const cudaDeviceProp &prop,
                                          const char *op_kind,
                                          int requested_blocks, int rounds,
                                          int warmup_rounds) {
  const int blocks =
      requested_blocks > 0 ? requested_blocks : std::max(1, prop.multiProcessorCount);
  const int effective_rounds = std::max(1, rounds);
  const std::string case_name =
      std::string(op_kind) + "_m" + std::to_string(MathIters);

  auto check_case = [&](cudaError_t status, const std::string &what) {
    check_cuda(status, case_name + ": " + what);
  };

  MixedSoftmaxSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_case(cudaMalloc(&d_samples, blocks * sizeof(MixedSoftmaxSample)),
             "cudaMalloc(samples)");
  check_case(cudaMalloc(&d_sink, 2 * blocks * sizeof(float)),
             "cudaMalloc(sink)");
  check_case(cudaGetLastError(), "clear pre-launch CUDA error");

  auto launch_once = [&](int timed_rounds, const char *tag) {
    check_case(cudaMemset(d_samples, 0, blocks * sizeof(MixedSoftmaxSample)),
               std::string(tag) + " clear samples");
    mixed_wgmma_softmax_kernel<Op, MathIters>
        <<<blocks, kMixedBlockThreads>>>(d_samples, d_sink, timed_rounds);
    check_case(cudaGetLastError(), std::string(tag) + " launch");
    check_case(cudaDeviceSynchronize(), std::string(tag) + " synchronize");
  };

  for (int i = 0; i < warmup_rounds; ++i) {
    launch_once(1, "warmup");
  }

  launch_once(effective_rounds, "timed");

  std::vector<MixedSoftmaxSample> samples(blocks);
  check_case(cudaMemcpy(samples.data(), d_samples,
                        blocks * sizeof(MixedSoftmaxSample),
                        cudaMemcpyDeviceToHost),
             "copy timing samples");
  check_case(cudaFree(d_samples), "cudaFree(samples)");
  check_case(cudaFree(d_sink), "cudaFree(sink)");

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> softmax_values;
  std::vector<uint64_t> total_values;
  for (const MixedSoftmaxSample &sample : samples) {
    if (sample.wgmma_count == 0) continue;
    issue_values.push_back(sample.wgmma_issue_cycles);
    wait_values.push_back(sample.wgmma_wait_cycles);
    softmax_values.push_back(sample.softmax_cycles);
    total_values.push_back(sample.total_cycles);
  }
  if (softmax_values.empty()) {
    softmax_values.push_back(0);
  }
  if (total_values.empty()) {
    throw std::runtime_error(case_name + ": no timing samples were produced");
  }

  constexpr uint64_t flops_per_wgmma =
      2ull * 64ull * static_cast<uint64_t>(kMixedWgmmaN) *
      static_cast<uint64_t>(Op::kK);
  const uint64_t flops =
      flops_per_wgmma *
      static_cast<uint64_t>(effective_rounds * kMixedWgmmaOpsPerRound);

  MixedSoftmaxResult result{};
  result.case_name = case_name;
  result.op_name = Op::name();
  result.math_iters = MathIters;
  result.blocks = blocks;
  result.rounds = effective_rounds;
  result.wgmma_ops_per_round = kMixedWgmmaOpsPerRound;
  result.flops_per_block = flops;
  result.median_wgmma_issue_cycles = median_u64(issue_values);
  result.median_wgmma_wait_cycles = median_u64(wait_values);
  result.median_softmax_cycles = median_u64(softmax_values);
  result.median_total_cycles = median_u64(total_values);
  const double wgmma_count =
      static_cast<double>(effective_rounds * kMixedWgmmaOpsPerRound);
  result.issue_cycles_per_wgmma =
      static_cast<double>(result.median_wgmma_issue_cycles) / wgmma_count;
  result.total_cycles_per_wgmma =
      static_cast<double>(result.median_total_cycles) / wgmma_count;
  result.flops_per_total_cycle =
      result.median_total_cycles == 0
          ? 0.0
          : static_cast<double>(result.flops_per_block) /
                static_cast<double>(result.median_total_cycles);
  return result;
}

void write_mixed_softmax_csv(const char *filename,
                             const std::vector<MixedSoftmaxResult> &results) {
  std::ofstream out(filename);
  out << "case,op,math_iters,blocks,rounds,wgmma_ops_per_round,"
         "flops_per_block,median_wgmma_issue_cycles,"
         "median_wgmma_wait_cycles,median_softmax_cycles,median_total_cycles,"
         "issue_cycles_per_wgmma,total_cycles_per_wgmma,"
         "flops_per_total_cycle\n";
  for (const MixedSoftmaxResult &r : results) {
    out << r.case_name << "," << r.op_name << "," << r.math_iters << ","
        << r.blocks << "," << r.rounds << "," << r.wgmma_ops_per_round << ","
        << r.flops_per_block << "," << r.median_wgmma_issue_cycles << ","
        << r.median_wgmma_wait_cycles << "," << r.median_softmax_cycles << ","
        << r.median_total_cycles << "," << r.issue_cycles_per_wgmma << ","
        << r.total_cycles_per_wgmma << "," << r.flops_per_total_cycle << "\n";
  }
}

void print_mixed_softmax_results(
    const std::vector<MixedSoftmaxResult> &results) {
  printf("case,op,math_iters,blocks,rounds,wgmma_issue,wgmma_wait,softmax,"
         "total,issue_per_wgmma,total_per_wgmma\n");
  for (const MixedSoftmaxResult &r : results) {
    printf("%s,%s,%d,%d,%d,%llu,%llu,%llu,%llu,%.3f,%.3f\n",
           r.case_name.c_str(), r.op_name.c_str(), r.math_iters, r.blocks,
           r.rounds,
           static_cast<unsigned long long>(r.median_wgmma_issue_cycles),
           static_cast<unsigned long long>(r.median_wgmma_wait_cycles),
           static_cast<unsigned long long>(r.median_softmax_cycles),
           static_cast<unsigned long long>(r.median_total_cycles),
           r.issue_cycles_per_wgmma, r.total_cycles_per_wgmma);
  }
}

template <typename Op, int MathIters>
void run_and_append_mixed_softmax_case(
    const cudaDeviceProp &prop, const char *op_kind, int blocks, int rounds,
    int warmup_rounds, std::vector<MixedSoftmaxResult> *results) {
  results->push_back(run_mixed_softmax_case<Op, MathIters>(
      prop, op_kind, blocks, rounds, warmup_rounds));
}

void run_selected_mixed_softmax_case(
    const std::string &selected, const cudaDeviceProp &prop, int blocks,
    int rounds, int warmup_rounds, std::vector<MixedSoftmaxResult> *results) {
  if (selected == "ss_m0") {
    run_and_append_mixed_softmax_case<WgmmaF16Ss128, 0>(
        prop, "ss", blocks, rounds, warmup_rounds, results);
  } else if (selected == "ss_m4") {
    run_and_append_mixed_softmax_case<WgmmaF16Ss128, 4>(
        prop, "ss", blocks, rounds, warmup_rounds, results);
  } else if (selected == "ss_m16") {
    run_and_append_mixed_softmax_case<WgmmaF16Ss128, 16>(
        prop, "ss", blocks, rounds, warmup_rounds, results);
  } else if (selected == "ss_m64") {
    run_and_append_mixed_softmax_case<WgmmaF16Ss128, 64>(
        prop, "ss", blocks, rounds, warmup_rounds, results);
  } else if (selected == "rs_m0") {
    run_and_append_mixed_softmax_case<WgmmaF16Rs128, 0>(
        prop, "rs", blocks, rounds, warmup_rounds, results);
  } else if (selected == "rs_m4") {
    run_and_append_mixed_softmax_case<WgmmaF16Rs128, 4>(
        prop, "rs", blocks, rounds, warmup_rounds, results);
  } else if (selected == "rs_m16") {
    run_and_append_mixed_softmax_case<WgmmaF16Rs128, 16>(
        prop, "rs", blocks, rounds, warmup_rounds, results);
  } else if (selected == "rs_m64") {
    run_and_append_mixed_softmax_case<WgmmaF16Rs128, 64>(
        prop, "rs", blocks, rounds, warmup_rounds, results);
  } else {
    throw std::runtime_error(
        "unsupported WGMMA_MIXED_SELECTED=" + selected +
        " (expected {ss,rs}_m{0,4,16,64})");
  }
}

TEST(WgmmaMixedSoftmaxBench, RsSsProbe) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_MIXED_BLOCKS", prop.multiProcessorCount);
  const int rounds = get_env_int("WGMMA_MIXED_ROUNDS", 256);
  const int warmup_rounds = get_env_int("WGMMA_MIXED_WARMUP", 3);

  std::vector<MixedSoftmaxResult> results;
  run_and_append_mixed_softmax_case<WgmmaF16Ss128, 0>(
      prop, "ss", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Ss128, 4>(
      prop, "ss", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Ss128, 16>(
      prop, "ss", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Ss128, 64>(
      prop, "ss", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Rs128, 0>(
      prop, "rs", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Rs128, 4>(
      prop, "rs", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Rs128, 16>(
      prop, "rs", blocks, rounds, warmup_rounds, &results);
  run_and_append_mixed_softmax_case<WgmmaF16Rs128, 64>(
      prop, "rs", blocks, rounds, warmup_rounds, &results);

  print_mixed_softmax_results(results);
  const std::string out_prefix =
      get_env_string("WGMMA_MIXED_OUT_PREFIX", "wgmma_mixed_softmax");
  write_mixed_softmax_csv((out_prefix + ".csv").c_str(), results);
}

TEST(WgmmaMixedSoftmaxBench, SelectedKernel) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_MIXED_BLOCKS", 1);
  const int rounds = get_env_int("WGMMA_MIXED_ROUNDS", 64);
  const int warmup_rounds = get_env_int("WGMMA_MIXED_WARMUP", 1);
  const std::string selected =
      get_env_string("WGMMA_MIXED_SELECTED", "rs_m16");

  std::vector<MixedSoftmaxResult> results;
  run_selected_mixed_softmax_case(selected, prop, blocks, rounds, warmup_rounds,
                                  &results);

  print_mixed_softmax_results(results);
  const std::string out_prefix = get_env_string(
      "WGMMA_MIXED_OUT_PREFIX", "wgmma_mixed_softmax_selected");
  write_mixed_softmax_csv((out_prefix + ".csv").c_str(), results);
}

}  // namespace
