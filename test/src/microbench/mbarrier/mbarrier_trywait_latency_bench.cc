#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
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

// Measures arrive_expect_tx(0) + try_wait as a tight pair using inline asm.
// The barrier is pre-initialized with expected_arrivals=1, so arrive completes
// the phase and the immediately following try_wait should observe it.
__global__ void arrive_try_wait_pair_kernel(uint64_t* cycle_start,
                                            uint64_t* cycle_end,
                                            uint32_t* sink_out) {
  __shared__ uint64_t barrier;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    const uint32_t bar_addr = smem_u32_addr(&barrier);

    // Warmup: do one arrive+try_wait cycle to prime the barrier unit.
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 0;\n" ::"r"(
            bar_addr));
    while (!mbarrier_try_wait_parity(&barrier, 0)) {
    }

    // Measured region: arrive + single try_wait, tightly coupled.
    // Phase is now 1, so arrive flips to phase 2, try_wait checks parity 1.
    uint32_t hit = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    asm volatile(
        "{ .reg .pred p;\n"
        "mov.u64 %0, %%clock64;\n"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%3], 0;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%3], 1;\n"
        "mov.u64 %1, %%clock64;\n"
        "selp.u32 %2, 1, 0, p;\n"
        "}\n"
        : "=l"(start), "=l"(end), "=r"(hit)
        : "r"(bar_addr)
        : "memory");

    cycle_start[0] = start;
    cycle_end[0] = end;
    sink_out[0] = hit;
  }
}

// Cross-warp arrive visibility: warp 0 does arrive, warp 1 polls try_wait.
// Records arrive timestamp (warp 0) and try_wait success timestamp (warp 1).
// Difference = cross-warp arrive→try_wait visibility latency.
__global__ void arrive_crosswarp_kernel(uint64_t* arrive_time,
                                        uint64_t* observe_time,
                                        uint32_t* result_out) {
  __shared__ uint64_t barrier;
  volatile __shared__ uint32_t observer_ready;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    mbarrier_init(&barrier, 1);
    observer_ready = 0;
  }
  __syncthreads();

  if (warp == 1 && lane == 0) {
    // Observer: signal ready, then poll try_wait.
    observer_ready = 1;
    asm volatile("membar.cta;" ::: "memory");
    uint64_t end = 0;
    uint32_t iters = 0;
    bool hit = false;
    do {
      ++iters;
      hit = mbarrier_try_wait_parity(&barrier, 0);
    } while (!hit);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    observe_time[0] = end;
    result_out[0] = iters;
    result_out[1] = hit ? 1u : 0u;
  } else if (warp == 0 && lane == 0) {
    // Producer: wait for observer, then arrive.
    while (observer_ready == 0) {
    }
    uint64_t start = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
    mbarrier_arrive_expect_tx(&barrier, 0);
    arrive_time[0] = start;
  }
}

}  // namespace

namespace {

struct SweepBenchSpec {
  const char* test_name;
  const char* description;
  const char* csv_file;
};

struct ScalarBenchSpec {
  const char* test_name;
  const char* description;
  const char* csv_file;
};

struct BreakdownRow {
  const char* row_name;
  const char* notes;
  MBarrierLinearFit fit;
  double delta = 0.0;
};

inline std::vector<uint32_t> default_instruction_counts() {
  return {64, 128, 192, 256};
}

inline MBarrierBenchOptions default_bench_options() {
  MBarrierBenchOptions options;
  options.warmup_iterations = 5;
  options.measured_iterations = 51;
  options.subtract_clock_overhead = true;
  return options;
}

inline void print_csv_path(const char* filename) {
  printf("CSV: %s\n", filename);
}

inline void print_breakdown_summary(const char* test_name,
                                    const char* description,
                                    const std::vector<BreakdownRow>& rows,
                                    uint64_t clock_overhead) {
  printf("\n=== %s ===\n", test_name);
  printf("Description: %s\n\n", description);
  printf("┌─────────────┬───────────────────────┬──────────┬──────────┬────────┬──────────┐\n");
  printf("│ Test        │ Notes                 │  Slope   │ Intercpt │   R^2  │  Delta   │\n");
  printf("├─────────────┼───────────────────────┼──────────┼──────────┼────────┼──────────┤\n");
  for (const auto& row : rows) {
    printf("│ %-11s │ %-21s │ %6.2f c │ %6.2f c │ %6.4f │ %+6.2f c │\n",
           row.row_name, row.notes, row.fit.slope, row.fit.intercept,
           row.fit.r_squared, row.delta);
  }
  printf("└─────────────┴───────────────────────┴──────────┴──────────┴────────┴──────────┘\n");
  printf("Clock overhead: %lu cycles\n", static_cast<unsigned long>(clock_overhead));
}

inline void export_breakdown_csv(const char* filename, const char* test_name,
                                 const char* description,
                                 const std::vector<BreakdownRow>& rows,
                                 const MBarrierBenchOptions& options,
                                 uint64_t clock_overhead) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    ADD_FAILURE() << "Failed to open CSV output file: " << filename;
    return;
  }

  export_mbarrier_csv_header(out);
  for (const auto& row : rows) {
    out << test_name << ',' << description << ",fit,-1,0,"
        << options.warmup_iterations << ',' << options.measured_iterations
        << ',' << clock_overhead << ",,,,,," << row.fit.slope << ','
        << row.fit.intercept << ',' << row.fit.r_squared << ',' << row.delta
        << ',' << row.row_name << ' ' << row.notes << '\n';
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

  template <typename LaunchForCountFn>
  void run_sweep_case(const SweepBenchSpec& spec,
                      LaunchForCountFn&& launch_for_count) {
    const uint64_t clock_overhead = measure_clock_overhead();
    const MBarrierBenchOptions options = default_bench_options();
    const std::vector<uint32_t> instruction_counts = default_instruction_counts();

    const auto summary = measure_with_sweep(
        [&](uint32_t instruction_count) {
          launch_for_count(instruction_count);
        },
        instruction_counts, options, clock_overhead, spec.test_name,
        spec.description);

    expect_valid_sweep_summary(summary, instruction_counts, options);
    export_mbarrier_sweep_csv(spec.csv_file, spec.test_name, spec.description,
                              summary, options);
    print_csv_path(spec.csv_file);
  }

  template <typename LaunchForCountFn>
  MBarrierSweepSummary measure_sweep_silent(LaunchForCountFn&& launch_for_count,
                                            const MBarrierBenchOptions& options,
                                            uint64_t clock_overhead) {
    MBarrierSweepSummary summary;
    summary.clock_overhead = clock_overhead;

    const std::vector<uint32_t> instruction_counts = default_instruction_counts();
    std::vector<double> medians;
    medians.reserve(instruction_counts.size());
    summary.points.reserve(instruction_counts.size());

    for (const uint32_t instruction_count : instruction_counts) {
      const MBarrierCycleSummary cycles = measure_summary(
          [&] { launch_for_count(instruction_count); }, options,
          clock_overhead);
      summary.points.push_back({instruction_count, cycles});
      medians.push_back(cycles.median);
    }

    summary.fit = mbarrier_linear_regression(instruction_counts, medians);
    return summary;
  }

  void export_scalar_case(const ScalarBenchSpec& spec,
                          const MBarrierCycleSummary& summary,
                          const MBarrierBenchOptions& options,
                          uint64_t clock_overhead,
                          const char* notes = "") {
    print_mbarrier_scalar(spec.test_name, spec.description, summary,
                          clock_overhead);
    export_mbarrier_scalar_csv(spec.csv_file, spec.test_name, spec.description,
                               summary, options, clock_overhead, notes);
    print_csv_path(spec.csv_file);
  }

 private:
  uint32_t* d_sink_ = nullptr;
};

TEST_F(MBarrierLatencyTest, WaitFalse) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  run_sweep_case(
      {"WaitFalse", "false-path try_wait dependency chain",
       "MBarrierLatencyTest.WaitFalse.csv"},
      [&](uint32_t instruction_count) {
        launch_try_wait_false_latency(instruction_count, cycle_start_ptr(),
                                      cycle_end_ptr(), sink_ptr());
      });
}

TEST_F(MBarrierLatencyTest, WaitTrue) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  run_sweep_case(
      {"WaitTrue", "true-path try_wait dependency chain",
       "MBarrierLatencyTest.WaitTrue.csv"},
      [&](uint32_t instruction_count) {
        launch_try_wait_true_latency(instruction_count, cycle_start_ptr(),
                                     cycle_end_ptr(), sink_ptr());
      });
}

TEST_F(MBarrierLatencyTest, LoadChase) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  run_sweep_case(
      {"LoadChase", "pointer-chasing ld.shared baseline",
       "MBarrierLatencyTest.LoadChase.csv"},
      [&](uint32_t instruction_count) {
        launch_smem_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      });
}

TEST_F(MBarrierLatencyTest, PredGate) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  run_sweep_case(
      {"PredGate", "predicated ld.shared pointer chase",
       "MBarrierLatencyTest.PredGate.csv"},
      [&](uint32_t instruction_count) {
        launch_pred_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      });
}

TEST_F(MBarrierLatencyTest, PredChain) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  run_sweep_case(
      {"PredChain", "ld.shared plus setp plus selp chain",
       "MBarrierLatencyTest.PredChain.csv"},
      [&](uint32_t instruction_count) {
        launch_manual_pred_latency(instruction_count, cycle_start_ptr(),
                                   cycle_end_ptr(), sink_ptr());
      });
}

TEST_F(MBarrierLatencyTest, Breakdown) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  const MBarrierBenchOptions options = default_bench_options();

  const auto load_chase = measure_sweep_silent(
      [&](uint32_t instruction_count) {
        launch_smem_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);
  const auto pred_gate = measure_sweep_silent(
      [&](uint32_t instruction_count) {
        launch_pred_chase_latency(instruction_count, cycle_start_ptr(),
                                  cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);
  const auto pred_chain = measure_sweep_silent(
      [&](uint32_t instruction_count) {
        launch_manual_pred_latency(instruction_count, cycle_start_ptr(),
                                   cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);
  const auto wait_false = measure_sweep_silent(
      [&](uint32_t instruction_count) {
        launch_try_wait_false_latency(instruction_count, cycle_start_ptr(),
                                      cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);
  const auto wait_true = measure_sweep_silent(
      [&](uint32_t instruction_count) {
        launch_try_wait_true_latency(instruction_count, cycle_start_ptr(),
                                     cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);

  const double base_slope = load_chase.fit.slope;
  const std::vector<BreakdownRow> rows = {
      {"LoadChase", "LDS -> LDS", load_chase.fit,
       load_chase.fit.slope - base_slope},
      {"PredGate", "@P BRA + LDS", pred_gate.fit,
       pred_gate.fit.slope - base_slope},
      {"PredChain", "LDS -> ISETP -> SEL", pred_chain.fit,
       pred_chain.fit.slope - base_slope},
      {"WaitFalse", "SYNCS.TRYWAIT @!P", wait_false.fit,
       wait_false.fit.slope - base_slope},
      {"WaitTrue", "SYNCS.TRYWAIT @P", wait_true.fit,
       wait_true.fit.slope - base_slope},
  };

  print_breakdown_summary("Breakdown", "latency decomposition summary", rows,
                          clock_overhead);
  constexpr const char* kCsvFile = "MBarrierLatencyTest.Breakdown.csv";
  export_breakdown_csv(kCsvFile, "Breakdown",
                       "latency decomposition summary", rows, options,
                       clock_overhead);
  print_csv_path(kCsvFile);
}

TEST_F(MBarrierLatencyTest, ArriveLocal) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  const MBarrierBenchOptions options = default_bench_options();

  // Verify try_wait actually returned true.
  uint32_t h_hit = 0;
  arrive_try_wait_pair_kernel<<<1, 32>>>(cycle_start_ptr(), cycle_end_ptr(),
                                         sink_ptr());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&h_hit, sink_ptr(), sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(h_hit, 1u) << "try_wait did not observe the arrive!";

  const auto summary = measure_summary(
      [&] {
        arrive_try_wait_pair_kernel<<<1, 32>>>(cycle_start_ptr(),
                                               cycle_end_ptr(), sink_ptr());
      },
      options, clock_overhead);

  export_scalar_case(
      {"ArriveLocal", "same-thread arrive and try_wait pair",
       "MBarrierLatencyTest.ArriveLocal.csv"},
      summary, options, clock_overhead,
      "clock64 arrive_expect_tx try_wait clock64");
}

TEST_F(MBarrierLatencyTest, ArriveWarp) {
  if (!mbarrier_running_in_native_mode()) {
    GTEST_SKIP() << "MBarrierLatencyTest requires native GPU mode.";
  }

  const uint64_t clock_overhead = measure_clock_overhead();
  const MBarrierBenchOptions options = default_bench_options();

  // Allocate separate device buffers for the two timestamps and results.
  uint64_t* d_arrive_time = nullptr;
  uint64_t* d_observe_time = nullptr;
  uint32_t* d_result = nullptr;
  ASSERT_EQ(cudaMalloc(&d_arrive_time, sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_observe_time, sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_result, 2 * sizeof(uint32_t)), cudaSuccess);

  std::vector<uint64_t> samples;
  samples.reserve(options.measured_iterations);

  for (int i = 0;
       i < options.warmup_iterations + options.measured_iterations; ++i) {
    arrive_crosswarp_kernel<<<1, 64>>>(d_arrive_time, d_observe_time, d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    if (i >= options.warmup_iterations) {
      uint64_t h_arrive = 0, h_observe = 0;
      uint32_t h_result[2] = {};
      ASSERT_EQ(cudaMemcpy(&h_arrive, d_arrive_time, sizeof(uint64_t),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(&h_observe, d_observe_time, sizeof(uint64_t),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(h_result[1], 1u) << "try_wait did not observe the arrive";
      samples.push_back(h_observe - h_arrive - clock_overhead);
    }
  }

  const MBarrierCycleSummary summary = summarize_mbarrier_cycles(samples);
  export_scalar_case(
      {"ArriveWarp", "cross-warp arrive visibility",
       "MBarrierLatencyTest.ArriveWarp.csv"},
      summary, options, clock_overhead,
      "observe_time minus arrive_time minus clock_overhead");

  cudaFree(d_arrive_time);
  cudaFree(d_observe_time);
  cudaFree(d_result);
}
