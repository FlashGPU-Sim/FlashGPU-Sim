#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/mbarrier/bench_utils.cuh"
#include "common/mbarrier/device_kernels.cuh"

namespace {

constexpr uint32_t kChainLength = 256;

#define MBARRIER_FALSE_PAIR                                                   \
  "@!p0 mbarrier.try_wait.parity.shared::cta.b64 p1, [%1], %2;\n"            \
  "@!p1 mbarrier.try_wait.parity.shared::cta.b64 p0, [%1], %2;\n"
#define MBARRIER_FALSE_X2 MBARRIER_FALSE_PAIR MBARRIER_FALSE_PAIR
#define MBARRIER_FALSE_X4 MBARRIER_FALSE_X2 MBARRIER_FALSE_X2
#define MBARRIER_FALSE_X8 MBARRIER_FALSE_X4 MBARRIER_FALSE_X4
#define MBARRIER_FALSE_X16 MBARRIER_FALSE_X8 MBARRIER_FALSE_X8
#define MBARRIER_FALSE_X32 MBARRIER_FALSE_X16 MBARRIER_FALSE_X16
#define MBARRIER_FALSE_X64 MBARRIER_FALSE_X32 MBARRIER_FALSE_X32
#define MBARRIER_FALSE_X128 MBARRIER_FALSE_X64 MBARRIER_FALSE_X64

#define MBARRIER_TRUE_PAIR                                                    \
  "@p0 mbarrier.try_wait.parity.shared::cta.b64 p1, [%1], %2;\n"             \
  "@p1 mbarrier.try_wait.parity.shared::cta.b64 p0, [%1], %2;\n"
#define MBARRIER_TRUE_X2 MBARRIER_TRUE_PAIR MBARRIER_TRUE_PAIR
#define MBARRIER_TRUE_X4 MBARRIER_TRUE_X2 MBARRIER_TRUE_X2
#define MBARRIER_TRUE_X8 MBARRIER_TRUE_X4 MBARRIER_TRUE_X4
#define MBARRIER_TRUE_X16 MBARRIER_TRUE_X8 MBARRIER_TRUE_X8
#define MBARRIER_TRUE_X32 MBARRIER_TRUE_X16 MBARRIER_TRUE_X16
#define MBARRIER_TRUE_X64 MBARRIER_TRUE_X32 MBARRIER_TRUE_X32
#define MBARRIER_TRUE_X128 MBARRIER_TRUE_X64 MBARRIER_TRUE_X64

template <int InstructionCount>
__device__ __forceinline__ uint32_t run_try_wait_false_chain(
    uint32_t bar_addr, uint32_t parity) {
  uint32_t sink = 0;
  if constexpr (InstructionCount == 64) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 0;\n"
                 MBARRIER_FALSE_X32
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 128) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 0;\n"
                 MBARRIER_FALSE_X64
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 192) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 0;\n"
                 MBARRIER_FALSE_X64
                 MBARRIER_FALSE_X32
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 256) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 0;\n"
                 MBARRIER_FALSE_X128
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  }
  return sink;
}

template <int InstructionCount>
__device__ __forceinline__ uint32_t run_try_wait_true_chain(uint32_t bar_addr,
                                                            uint32_t parity) {
  uint32_t sink = 0;
  if constexpr (InstructionCount == 64) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 1;\n"
                 MBARRIER_TRUE_X32
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 128) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 1;\n"
                 MBARRIER_TRUE_X64
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 192) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 1;\n"
                 MBARRIER_TRUE_X64
                 MBARRIER_TRUE_X32
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  } else if constexpr (InstructionCount == 256) {
    asm volatile("{\n"
                 ".reg .pred p0;\n"
                 ".reg .pred p1;\n"
                 "mov.pred p0, 1;\n"
                 MBARRIER_TRUE_X128
                 "selp.u32 %0, 1, 0, p0;\n"
                 "}\n"
                 : "=r"(sink)
                 : "r"(bar_addr), "r"(parity));
  }
  return sink;
}

template <int InstructionCount>
__global__ void try_wait_false_latency_kernel(uint64_t* cycle_start,
                                              uint64_t* cycle_end,
                                              uint32_t* sink_out) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 2);
    mbarrier_arrive(&barrier);
    sink_out[0] = run_try_wait_false_chain<64>(smem_u32_addr(&barrier), 0);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    const uint32_t bar_addr = smem_u32_addr(&barrier);
    uint32_t sink = run_try_wait_false_chain<64>(bar_addr, 0);

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    sink ^= run_try_wait_false_chain<InstructionCount>(bar_addr, 0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = sink;
  }
}

template <int InstructionCount>
__global__ void try_wait_true_latency_kernel(uint64_t* cycle_start,
                                             uint64_t* cycle_end,
                                             uint32_t* sink_out) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(&barrier, 0);
    while (!mbarrier_try_wait_parity(&barrier, 0)) {
    }
    sink_out[0] = 1;
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    const uint32_t bar_addr = smem_u32_addr(&barrier);
    uint32_t sink = run_try_wait_true_chain<64>(bar_addr, 0);

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    sink ^= run_try_wait_true_chain<InstructionCount>(bar_addr, 0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = sink;
  }
}

__device__ __forceinline__ uint32_t run_smem_load_step(uint32_t idx,
                                                       uint32_t base_addr) {
  asm volatile("{\n"
               "shl.b32 %0, %0, 2;\n"
               "add.u32 %0, %0, %1;\n"
               "ld.shared.u32 %0, [%0];\n"
               "}\n"
               : "+r"(idx)
               : "r"(base_addr));
  return idx;
}

template <int InstructionCount>
__device__ __forceinline__ uint32_t run_smem_load_chain(uint32_t idx,
                                                        uint32_t base_addr) {
  if constexpr (InstructionCount == 64) {
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
    idx = run_smem_load_step(idx, base_addr);
  } else if constexpr (InstructionCount == 128) {
    idx = run_smem_load_chain<64>(idx, base_addr);
    idx = run_smem_load_chain<64>(idx, base_addr);
  } else if constexpr (InstructionCount == 192) {
    idx = run_smem_load_chain<128>(idx, base_addr);
    idx = run_smem_load_chain<64>(idx, base_addr);
  } else if constexpr (InstructionCount == 256) {
    idx = run_smem_load_chain<128>(idx, base_addr);
    idx = run_smem_load_chain<128>(idx, base_addr);
  }
  return idx;
}

template <int InstructionCount>
__global__ void smem_load_latency_kernel(uint64_t* cycle_start,
                                         uint64_t* cycle_end,
                                         uint32_t* sink_out) {
  __shared__ uint32_t chain[kChainLength];

  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < kChainLength; ++i) {
      chain[i] = (i + 1) & (kChainLength - 1);
    }
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    const uint32_t base_addr = smem_u32_addr(&chain[0]);
    uint32_t idx = static_cast<uint32_t>(clock64()) & (kChainLength - 1);
    for (int i = 0; i < 8; ++i) {
      idx = run_smem_load_step(idx, base_addr);
    }

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    idx = run_smem_load_chain<InstructionCount>(idx, base_addr);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = idx ^ (kChainLength - 1);
  }
}

void expect_valid_sweep_summary(const MBarrierSweepSummary& summary,
                                const std::vector<uint32_t>& instruction_counts,
                                const MBarrierBenchOptions& options) {
  ASSERT_EQ(summary.points.size(), instruction_counts.size());
  for (size_t i = 0; i < instruction_counts.size(); ++i) {
    EXPECT_EQ(summary.points[i].instruction_count, instruction_counts[i]);
    EXPECT_EQ(summary.points[i].cycles.sample_count,
              options.measured_iterations);
  }
}

void launch_try_wait_false_latency(uint32_t instruction_count,
                                   uint64_t* cycle_start, uint64_t* cycle_end,
                                   uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      try_wait_false_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end,
                                                   sink_out);
      return;
    case 128:
      try_wait_false_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end,
                                                    sink_out);
      return;
    case 192:
      try_wait_false_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end,
                                                    sink_out);
      return;
    case 256:
      try_wait_false_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end,
                                                    sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported false-path try_wait instruction count: "
                    << instruction_count;
      return;
  }
}

void launch_try_wait_true_latency(uint32_t instruction_count,
                                  uint64_t* cycle_start, uint64_t* cycle_end,
                                  uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      try_wait_true_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end,
                                                  sink_out);
      return;
    case 128:
      try_wait_true_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end,
                                                   sink_out);
      return;
    case 192:
      try_wait_true_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end,
                                                   sink_out);
      return;
    case 256:
      try_wait_true_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end,
                                                   sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported true-path try_wait instruction count: "
                    << instruction_count;
      return;
  }
}

void launch_smem_load_latency(uint32_t instruction_count, uint64_t* cycle_start,
                              uint64_t* cycle_end, uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      smem_load_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end, sink_out);
      return;
    case 128:
      smem_load_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end, sink_out);
      return;
    case 192:
      smem_load_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end, sink_out);
      return;
    case 256:
      smem_load_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end, sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported shared-load instruction count: "
                    << instruction_count;
      return;
  }
}

}  // namespace

class MBarrierLatencyTest : public MBarrierBenchmarkFixture {
 protected:
  void SetUp() override {
    MBarrierBenchmarkFixture::SetUp();
    const cudaError_t err = cudaMalloc(&d_sink_, sizeof(uint32_t));
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate sink buffer";
  }

  void TearDown() override {
    if (d_sink_ != nullptr) {
      cudaFree(d_sink_);
      d_sink_ = nullptr;
    }
    MBarrierBenchmarkFixture::TearDown();
  }

  uint32_t* sink_ptr() { return d_sink_; }

 private:
  uint32_t* d_sink_ = nullptr;
};

TEST_F(MBarrierLatencyTest, P1) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> instruction_counts = {64, 128, 192, 256};

  const auto summary = measure_with_sweep(
      [&](uint32_t instruction_count) {
        launch_try_wait_false_latency(instruction_count, cycle_start_ptr(),
                                      cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P1",
      "T1 predicate-dependent sweep");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.P1.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P1", "T1 predicate-dependent sweep",
                            summary, options);
  printf("Results exported to: %s\n", kCsvFile);
}

TEST_F(MBarrierLatencyTest, P2) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> instruction_counts = {64, 128, 192, 256};

  const auto summary = measure_with_sweep(
      [&](uint32_t instruction_count) {
        launch_try_wait_true_latency(instruction_count, cycle_start_ptr(),
                                     cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P2",
      "T2a predicate-dependent sweep");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.P2.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P2", "T2a predicate-dependent sweep",
                            summary, options);
  printf("Results exported to: %s\n", kCsvFile);
}

TEST_F(MBarrierLatencyTest, PLd) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> instruction_counts = {64, 128, 192, 256};

  const auto summary = measure_with_sweep(
      [&](uint32_t instruction_count) {
        launch_smem_load_latency(instruction_count, cycle_start_ptr(),
                                 cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P_ld",
      "T3 dependent ld.shared sweep");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.PLd.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P_ld", "T3 dependent ld.shared sweep",
                            summary, options);
  printf("Results exported to: %s\n", kCsvFile);
}
