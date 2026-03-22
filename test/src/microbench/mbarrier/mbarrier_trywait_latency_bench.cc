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

#define SMEM_CHASE_STEP "ld.shared.u32 %0, [%0];\n"
#define SMEM_CHASE_X2 SMEM_CHASE_STEP SMEM_CHASE_STEP
#define SMEM_CHASE_X4 SMEM_CHASE_X2 SMEM_CHASE_X2
#define SMEM_CHASE_X8 SMEM_CHASE_X4 SMEM_CHASE_X4
#define SMEM_CHASE_X16 SMEM_CHASE_X8 SMEM_CHASE_X8
#define SMEM_CHASE_X32 SMEM_CHASE_X16 SMEM_CHASE_X16
#define SMEM_CHASE_X64 SMEM_CHASE_X32 SMEM_CHASE_X32
#define SMEM_CHASE_X128 SMEM_CHASE_X64 SMEM_CHASE_X64

#define PRED_CHASE_STEP "@!p0 ld.shared.u32 %0, [%0];\n"
#define PRED_CHASE_X2 PRED_CHASE_STEP PRED_CHASE_STEP
#define PRED_CHASE_X4 PRED_CHASE_X2 PRED_CHASE_X2
#define PRED_CHASE_X8 PRED_CHASE_X4 PRED_CHASE_X4
#define PRED_CHASE_X16 PRED_CHASE_X8 PRED_CHASE_X8
#define PRED_CHASE_X32 PRED_CHASE_X16 PRED_CHASE_X16
#define PRED_CHASE_X64 PRED_CHASE_X32 PRED_CHASE_X32
#define PRED_CHASE_X128 PRED_CHASE_X64 PRED_CHASE_X64

// Manual predicate chain: pointer-chasing ld.shared.u32 + setp to predicate.
// Dependency: addr → ld.shared.u32 → data → setp → predicate → selp → addr.
// Each step = 3 SASS instructions. InstructionCount counts ld operations.
#define MANUAL_PRED_STEP                                                       \
  "ld.shared.u32 %0, [%0];\n"                                                 \
  "setp.eq.u32 p0, %0, 0xFFFFFFFF;\n"                                         \
  "selp.u32 %0, %1, %0, p0;\n"
#define MANUAL_PRED_X2 MANUAL_PRED_STEP MANUAL_PRED_STEP
#define MANUAL_PRED_X4 MANUAL_PRED_X2 MANUAL_PRED_X2
#define MANUAL_PRED_X8 MANUAL_PRED_X4 MANUAL_PRED_X4
#define MANUAL_PRED_X16 MANUAL_PRED_X8 MANUAL_PRED_X8
#define MANUAL_PRED_X32 MANUAL_PRED_X16 MANUAL_PRED_X16
#define MANUAL_PRED_X64 MANUAL_PRED_X32 MANUAL_PRED_X32
#define MANUAL_PRED_X128 MANUAL_PRED_X64 MANUAL_PRED_X64

template <int InstructionCount>
__device__ __forceinline__ uint32_t run_smem_chase_chain(uint32_t addr) {
  if constexpr (InstructionCount == 64) {
    asm volatile(SMEM_CHASE_X64 : "+r"(addr));
  } else if constexpr (InstructionCount == 128) {
    asm volatile(SMEM_CHASE_X128 : "+r"(addr));
  } else if constexpr (InstructionCount == 192) {
    asm volatile(SMEM_CHASE_X128 SMEM_CHASE_X64 : "+r"(addr));
  } else if constexpr (InstructionCount == 256) {
    asm volatile(SMEM_CHASE_X128 SMEM_CHASE_X128 : "+r"(addr));
  }
  return addr;
}

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

template <int InstructionCount>
__global__ void smem_chase_latency_kernel(uint64_t* cycle_start,
                                          uint64_t* cycle_end,
                                          uint32_t* sink_out) {
  __shared__ uint32_t chain[kChainLength];

  if (threadIdx.x == 0) {
    const uint32_t base = smem_u32_addr(&chain[0]);
    for (uint32_t i = 0; i < kChainLength; ++i) {
      chain[i] = base + (((i + 1) & (kChainLength - 1)) * 4);
    }
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    uint32_t addr = smem_u32_addr(&chain[0]);
    // warmup
    asm volatile(SMEM_CHASE_X8 : "+r"(addr));

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    addr = run_smem_chase_chain<InstructionCount>(addr);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = addr;
  }
}

void launch_smem_chase_latency(uint32_t instruction_count,
                               uint64_t* cycle_start, uint64_t* cycle_end,
                               uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      smem_chase_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end,
                                               sink_out);
      return;
    case 128:
      smem_chase_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    case 192:
      smem_chase_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    case 256:
      smem_chase_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported chase instruction count: "
                    << instruction_count;
      return;
  }
}

template <int InstructionCount>
__global__ void manual_pred_latency_kernel(uint64_t* cycle_start,
                                           uint64_t* cycle_end,
                                           uint32_t* sink_out) {
  __shared__ uint32_t chain[kChainLength];

  if (threadIdx.x == 0) {
    const uint32_t base = smem_u32_addr(&chain[0]);
    for (uint32_t i = 0; i < kChainLength; ++i) {
      chain[i] = base + (((i + 1) & (kChainLength - 1)) * 4);
    }
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    uint32_t addr = smem_u32_addr(&chain[0]);
    const uint32_t reset_addr = addr;
    // warmup: 8 chase steps
    asm volatile(SMEM_CHASE_X8 : "+r"(addr));

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    // Each step: ld.shared.u32 [addr]→data, setp data→pred, selp pred→addr
    if constexpr (InstructionCount == 64) {
      asm volatile("{ .reg .pred p0;\n" MANUAL_PRED_X64 "}\n"
                   : "+r"(addr) : "r"(reset_addr));
    } else if constexpr (InstructionCount == 128) {
      asm volatile("{ .reg .pred p0;\n" MANUAL_PRED_X128 "}\n"
                   : "+r"(addr) : "r"(reset_addr));
    } else if constexpr (InstructionCount == 192) {
      asm volatile("{ .reg .pred p0;\n" MANUAL_PRED_X128 MANUAL_PRED_X64
                   "}\n" : "+r"(addr) : "r"(reset_addr));
    } else if constexpr (InstructionCount == 256) {
      asm volatile("{ .reg .pred p0;\n" MANUAL_PRED_X128 MANUAL_PRED_X128
                   "}\n" : "+r"(addr) : "r"(reset_addr));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = addr;
  }
}

void launch_manual_pred_latency(uint32_t instruction_count,
                                uint64_t* cycle_start, uint64_t* cycle_end,
                                uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      manual_pred_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    case 128:
      manual_pred_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end,
                                                 sink_out);
      return;
    case 192:
      manual_pred_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end,
                                                 sink_out);
      return;
    case 256:
      manual_pred_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end,
                                                 sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported manual-pred instruction count: "
                    << instruction_count;
      return;
  }
}

template <int InstructionCount>
__global__ void pred_chase_latency_kernel(uint64_t* cycle_start,
                                          uint64_t* cycle_end,
                                          uint32_t* sink_out) {
  __shared__ uint32_t chain[kChainLength];

  if (threadIdx.x == 0) {
    const uint32_t base = smem_u32_addr(&chain[0]);
    for (uint32_t i = 0; i < kChainLength; ++i) {
      chain[i] = base + (((i + 1) & (kChainLength - 1)) * 4);
    }
  }
  __syncwarp();

  // Store a zero that the compiler cannot prove is zero.
  volatile __shared__ uint32_t opaque_zero;
  if (threadIdx.x == 0) {
    opaque_zero = 0;
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    uint32_t addr = smem_u32_addr(&chain[0]);
    const uint32_t pred_val = opaque_zero;
    // warmup
    asm volatile(SMEM_CHASE_X8 : "+r"(addr));

    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    if constexpr (InstructionCount == 64) {
      asm volatile("{ .reg .pred p0; setp.ne.u32 p0, %1, 0;\n"
                   PRED_CHASE_X64 "}\n" : "+r"(addr) : "r"(pred_val));
    } else if constexpr (InstructionCount == 128) {
      asm volatile("{ .reg .pred p0; setp.ne.u32 p0, %1, 0;\n"
                   PRED_CHASE_X128 "}\n" : "+r"(addr) : "r"(pred_val));
    } else if constexpr (InstructionCount == 192) {
      asm volatile("{ .reg .pred p0; setp.ne.u32 p0, %1, 0;\n"
                   PRED_CHASE_X128 PRED_CHASE_X64 "}\n"
                   : "+r"(addr) : "r"(pred_val));
    } else if constexpr (InstructionCount == 256) {
      asm volatile("{ .reg .pred p0; setp.ne.u32 p0, %1, 0;\n"
                   PRED_CHASE_X128 PRED_CHASE_X128 "}\n"
                   : "+r"(addr) : "r"(pred_val));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = addr;
  }
}

void launch_pred_chase_latency(uint32_t instruction_count,
                               uint64_t* cycle_start, uint64_t* cycle_end,
                               uint32_t* sink_out) {
  switch (instruction_count) {
    case 64:
      pred_chase_latency_kernel<64><<<1, 32>>>(cycle_start, cycle_end,
                                               sink_out);
      return;
    case 128:
      pred_chase_latency_kernel<128><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    case 192:
      pred_chase_latency_kernel<192><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    case 256:
      pred_chase_latency_kernel<256><<<1, 32>>>(cycle_start, cycle_end,
                                                sink_out);
      return;
    default:
      ADD_FAILURE() << "Unsupported pred-chase instruction count: "
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

TEST_F(MBarrierLatencyTest, PLdChase) {
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
        launch_smem_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P_ld_chase",
      "T4 pointer-chasing ld.shared sweep");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.PLdChase.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P_ld_chase",
                            "T4 pointer-chasing ld.shared sweep", summary,
                            options);
  printf("Results exported to: %s\n", kCsvFile);
}

TEST_F(MBarrierLatencyTest, PManualPred) {
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
        launch_manual_pred_latency(instruction_count, cycle_start_ptr(),
                                   cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P_manual_pred",
      "T5 ld.shared.u64 + setp predicate chain");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.PManualPred.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P_manual_pred",
                            "T5 ld.shared.u32 + setp + selp chain", summary,
                            options);
  printf("Results exported to: %s\n", kCsvFile);
}

TEST_F(MBarrierLatencyTest, PPredChase) {
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
        launch_pred_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      },
      instruction_counts, options, clock_overhead, "P_pred_chase",
      "T6 @!p0 ld.shared.u32 pointer-chase (static pred)");

  expect_valid_sweep_summary(summary, instruction_counts, options);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.PPredChase.csv";
  export_mbarrier_sweep_csv(kCsvFile, "P_pred_chase",
                            "T6 @!p0 ld.shared.u32 pointer-chase (static pred)",
                            summary, options);
  printf("Results exported to: %s\n", kCsvFile);
}

TEST_F(MBarrierLatencyTest, Summary) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  const std::vector<uint32_t> instruction_counts = {64, 128, 192, 256};

  struct Entry {
    const char* name;
    const char* sass;
    MBarrierLinearFit fit;
  };
  std::vector<Entry> entries;

  // 1. Baseline: pure pointer-chasing ld.shared
  {
    const auto s = measure_with_sweep(
        [&](uint32_t n) {
          launch_smem_chase_latency(n, cycle_start_ptr(), cycle_end_ptr(),
                                    sink_ptr());
        },
        instruction_counts, options, clock_overhead, "PLdChase",
        "pointer-chasing ld.shared");
    entries.push_back({"ld.shared (chase)", "LDS -> LDS", s.fit});
  }

  // 2. + predicate gate (static, always true)
  {
    const auto s = measure_with_sweep(
        [&](uint32_t n) {
          launch_pred_chase_latency(n, cycle_start_ptr(), cycle_end_ptr(),
                                    sink_ptr());
        },
        instruction_counts, options, clock_overhead, "PPredChase",
        "+ static predicate");
    entries.push_back({"+ predicate gate", "@P BRA + LDS", s.fit});
  }

  // 3. + predicate + compare (ld.shared + setp + selp)
  {
    const auto s = measure_with_sweep(
        [&](uint32_t n) {
          launch_manual_pred_latency(n, cycle_start_ptr(), cycle_end_ptr(),
                                     sink_ptr());
        },
        instruction_counts, options, clock_overhead, "PManualPred",
        "+ predicate + compare");
    entries.push_back({"+ pred + compare", "LDS -> ISETP -> SEL", s.fit});
  }

  // 4. try_wait false-path (fused hardware instruction)
  {
    const auto s = measure_with_sweep(
        [&](uint32_t n) {
          launch_try_wait_false_latency(n, cycle_start_ptr(), cycle_end_ptr(),
                                        sink_ptr());
        },
        instruction_counts, options, clock_overhead, "P1",
        "try_wait false-path");
    entries.push_back({"try_wait (false)", "SYNCS.TRYWAIT @!P", s.fit});
  }

  // 5. try_wait true-path (fused hardware instruction)
  {
    const auto s = measure_with_sweep(
        [&](uint32_t n) {
          launch_try_wait_true_latency(n, cycle_start_ptr(), cycle_end_ptr(),
                                       sink_ptr());
        },
        instruction_counts, options, clock_overhead, "P2",
        "try_wait true-path");
    entries.push_back({"try_wait (true)", "SYNCS.TRYWAIT @P", s.fit});
  }

  const double base_slope = entries[0].fit.slope;

  printf("\n=== MBarrier Latency Decomposition Summary ===\n");
  printf("Clock overhead: %lu cycles\n\n", clock_overhead);
  printf("┌───┬──────────────────────────┬───────────────────────┬──────────┬──────────┬────────┬──────────┐\n");
  printf("│ # │ Test                     │ SASS Pattern          │    Slope │ Intercpt │    R^2 │    Delta │\n");
  printf("├───┼──────────────────────────┼───────────────────────┼──────────┼──────────┼────────┼──────────┤\n");
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    const double delta = e.fit.slope - base_slope;
    printf("│ %zu │ %-24s │ %-21s │ %6.2f c │ %6.2f c │ %6.4f │ %+6.2f c │\n",
           i + 1, e.name, e.sass, e.fit.slope, e.fit.intercept,
           e.fit.r_squared, delta);
  }
  printf("└───┴──────────────────────────┴───────────────────────┴──────────┴──────────┴────────┴──────────┘\n");
  printf("\nDecomposition:\n");
  printf("  ld.shared base latency:      %6.2f cycles\n", base_slope);
  printf("  + predicate gate overhead:   %+6.2f cycles\n",
         entries[1].fit.slope - base_slope);
  printf("  + manual compare (setp+selp):%+6.2f cycles\n",
         entries[2].fit.slope - base_slope);
  printf("  try_wait fused overhead:     %+6.2f cycles  (vs %.2f for manual)\n",
         entries[3].fit.slope - base_slope,
         entries[2].fit.slope - base_slope);
}
