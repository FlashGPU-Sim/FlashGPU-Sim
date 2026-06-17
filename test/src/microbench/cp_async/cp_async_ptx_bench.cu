#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kCpBytes = 16;
constexpr int kMaxCopies = 8;

enum Mode {
  kEmpty = 0,
  kIssue = 1,
  kComplete = 2,
  kWait = 3,
  kDelayWait = 4,
  kIssueOnly = 5,
  kCommitOnly = 6,
  kWaitEmpty = 7,
  kBandwidth = 8,
};

struct Options {
  int device = 0;
  int blocks = 170;
  int threads = 32;
  int active_warps = 1;
  int iters = 1024;
  int warmup = 128;
  int groups = 64;
  int copies = 8;
  int bytes = 64 * 1024;
  int wait_k = 0;
  int delay = 0;
  Mode mode = kIssue;
  std::string csv_path;
};

#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                           \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

__device__ __forceinline__ uint32_t smem_addr_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void *ptr) {
  uint64_t out = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(out) : "l"(ptr));
  return out;
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr,
                                               uint64_t gmem_addr) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                   smem_addr),
               "l"(gmem_addr)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

#define CP1 "cp.async.cg.shared.global.L2::128B [%2], [%3], 16;\n"
#define CP2 CP1 "cp.async.cg.shared.global.L2::128B [%4], [%5], 16;\n"
#define CP4 CP2 "cp.async.cg.shared.global.L2::128B [%6], [%7], 16;\n" \
            "cp.async.cg.shared.global.L2::128B [%8], [%9], 16;\n"
#define CP8 CP4 "cp.async.cg.shared.global.L2::128B [%10], [%11], 16;\n" \
            "cp.async.cg.shared.global.L2::128B [%12], [%13], 16;\n"     \
            "cp.async.cg.shared.global.L2::128B [%14], [%15], 16;\n"     \
            "cp.async.cg.shared.global.L2::128B [%16], [%17], 16;\n"

#define WAIT0 "cp.async.wait_group 0;\n"
#define WAIT1 "cp.async.wait_group 1;\n"
#define WAIT2 "cp.async.wait_group 2;\n"
#define WAIT4 "cp.async.wait_group 4;\n"

#define ASM_INPUTS                                                            \
  "r"(s[0]), "l"(g[0]), "r"(s[1]), "l"(g[1]), "r"(s[2]), "l"(g[2]),      \
      "r"(s[3]), "l"(g[3]), "r"(s[4]), "l"(g[4]), "r"(s[5]), "l"(g[5]), \
      "r"(s[6]), "l"(g[6]), "r"(s[7]), "l"(g[7]), "r"(groups),          \
      "r"(delay)

#define DELAY_LOOP                                                        \
  "mov.u32 d, %19;\n"                                                     \
  "setp.eq.u32 p, d, 0;\n"                                                \
  "@p bra L_delay_done_%=;\n"                                             \
  "cvt.u64.u32 d64, d;\n"                                                 \
  "mov.u64 t0, %%clock64;\n"                                              \
  "L_delay_%=:\n"                                                         \
  "mov.u64 t1, %%clock64;\n"                                              \
  "sub.u64 dt, t1, t0;\n"                                                 \
  "setp.lt.u64 p, dt, d64;\n"                                             \
  "@p bra L_delay_%=;\n"                                                  \
  "L_delay_done_%=:\n"

#define GEN_EMPTY()                                                        \
  __device__ __forceinline__ uint64_t run_empty_asm(int groups) {          \
    uint64_t start = 0, end = 0;                                           \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n;\n"                                         \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %2;\n"                                       \
                 "mov.u64 %0, %%clock64;\n"                               \
                 "L_empty_%=:\n"                                          \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_empty_%=;\n"                                   \
                 "mov.u64 %1, %%clock64;\n"                               \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : "r"(groups)                                             \
                 : "memory");                                             \
    return end - start;                                                    \
  }

#define GEN_ISSUE(N, CPSEQ)                                                \
  template <>                                                              \
  __device__ __forceinline__ uint64_t                                      \
  run_issue_asm<N>(const uint32_t *s, const uint64_t *g, int groups,        \
                   int delay) {                                            \
    uint64_t start = 0, end = 0;                                           \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n;\n"                                         \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %18;\n"                                      \
                 "mov.u64 %0, %%clock64;\n"                               \
                 "L_issue_%=:\n" CPSEQ                                    \
                 "cp.async.commit_group;\n"                                \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_issue_%=;\n"                                   \
                 "mov.u64 %1, %%clock64;\n"                               \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : ASM_INPUTS                                              \
                 : "memory");                                             \
    return end - start;                                                    \
  }

#define GEN_ISSUE_ONLY(N, CPSEQ)                                           \
  template <>                                                              \
  __device__ __forceinline__ uint64_t                                      \
  run_issue_only_asm<N>(const uint32_t *s, const uint64_t *g, int groups) { \
    uint64_t start = 0, end = 0;                                           \
    const int delay = 0;                                                   \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n;\n"                                         \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %18;\n"                                      \
                 "mov.u64 %0, %%clock64;\n"                               \
                 "L_issue_only_%=:\n" CPSEQ                               \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_issue_only_%=;\n"                              \
                 "mov.u64 %1, %%clock64;\n"                               \
                 "cp.async.commit_group;\n" WAIT0                         \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : ASM_INPUTS                                              \
                 : "memory");                                             \
    return end - start;                                                    \
  }

#define GEN_COMPLETE(N, CPSEQ)                                             \
  template <>                                                              \
  __device__ __forceinline__ uint64_t                                      \
  run_complete_asm<N>(const uint32_t *s, const uint64_t *g, int groups,     \
                      int delay) {                                         \
    uint64_t start = 0, end = 0;                                           \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n;\n"                                         \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %18;\n"                                      \
                 "mov.u64 %0, %%clock64;\n"                               \
                 "L_complete_%=:\n" CPSEQ                                 \
                 "cp.async.commit_group;\n" WAIT0                         \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_complete_%=;\n"                                \
                 "mov.u64 %1, %%clock64;\n"                               \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : ASM_INPUTS                                              \
                 : "memory");                                             \
    return end - start;                                                    \
  }

#define GEN_WAIT(N, K, CPSEQ, WAITSEQ)                                     \
  template <>                                                              \
  __device__ __forceinline__ uint64_t                                      \
  run_wait_asm<N, K>(const uint32_t *s, const uint64_t *g, int groups,      \
                     int delay) {                                          \
    uint64_t start = 0, end = 0;                                           \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n, d;\n"                                      \
                 ".reg .u64 t0, t1, dt, d64;\n"                           \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %18;\n"                                      \
                 "L_wait_issue_%=:\n" CPSEQ                               \
                 "cp.async.commit_group;\n"                                \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_wait_issue_%=;\n" DELAY_LOOP                   \
                 "mov.u64 %0, %%clock64;\n" WAITSEQ                      \
                 "mov.u64 %1, %%clock64;\n" WAIT0                        \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : ASM_INPUTS                                              \
                 : "memory");                                             \
    return end - start;                                                    \
  }

#define GEN_WAIT_EMPTY(K, WAITSEQ)                                         \
  template <>                                                              \
  __device__ __forceinline__ uint64_t run_wait_empty_asm<K>(int groups) {  \
    uint64_t start = 0, end = 0;                                           \
    asm volatile("{\n"                                                     \
                 ".reg .u32 n;\n"                                         \
                 ".reg .pred p;\n"                                        \
                 "mov.u32 n, %2;\n"                                       \
                 "mov.u64 %0, %%clock64;\n"                               \
                 "L_wait_empty_%=:\n" WAITSEQ                             \
                 "sub.u32 n, n, 1;\n"                                     \
                 "setp.ne.u32 p, n, 0;\n"                                 \
                 "@p bra L_wait_empty_%=;\n"                              \
                 "mov.u64 %1, %%clock64;\n"                               \
                 "}\n"                                                     \
                 : "=l"(start), "=l"(end)                                  \
                 : "r"(groups)                                             \
                 : "memory");                                             \
    return end - start;                                                    \
  }

template <int Copies>
__device__ __forceinline__ uint64_t run_issue_asm(const uint32_t *,
                                                  const uint64_t *, int,
                                                  int) {
  return 0;
}
template <int Copies>
__device__ __forceinline__ uint64_t run_issue_only_asm(const uint32_t *,
                                                       const uint64_t *, int) {
  return 0;
}
template <int Copies>
__device__ __forceinline__ uint64_t run_complete_asm(const uint32_t *,
                                                     const uint64_t *, int,
                                                     int) {
  return 0;
}
template <int Copies, int WaitK>
__device__ __forceinline__ uint64_t run_wait_asm(const uint32_t *,
                                                 const uint64_t *, int, int) {
  return 0;
}
template <int WaitK>
__device__ __forceinline__ uint64_t run_wait_empty_asm(int) {
  return 0;
}

__device__ __forceinline__ uint64_t run_commit_only_asm(int groups) {
  uint64_t start = 0, end = 0;
  asm volatile("{\n"
               ".reg .u32 n;\n"
               ".reg .pred p;\n"
               "mov.u32 n, %2;\n"
               "mov.u64 %0, %%clock64;\n"
               "L_commit_only_%=:\n"
               "cp.async.commit_group;\n"
               "sub.u32 n, n, 1;\n"
               "setp.ne.u32 p, n, 0;\n"
               "@p bra L_commit_only_%=;\n"
               "mov.u64 %1, %%clock64;\n"
               WAIT0
               "}\n"
               : "=l"(start), "=l"(end)
               : "r"(groups)
               : "memory");
  return end - start;
}

GEN_EMPTY()
GEN_ISSUE(1, CP1)
GEN_ISSUE(2, CP2)
GEN_ISSUE(4, CP4)
GEN_ISSUE(8, CP8)
GEN_ISSUE_ONLY(1, CP1)
GEN_ISSUE_ONLY(2, CP2)
GEN_ISSUE_ONLY(4, CP4)
GEN_ISSUE_ONLY(8, CP8)
GEN_COMPLETE(1, CP1)
GEN_COMPLETE(2, CP2)
GEN_COMPLETE(4, CP4)
GEN_COMPLETE(8, CP8)
GEN_WAIT(1, 0, CP1, WAIT0)
GEN_WAIT(2, 0, CP2, WAIT0)
GEN_WAIT(4, 0, CP4, WAIT0)
GEN_WAIT(8, 0, CP8, WAIT0)
GEN_WAIT(1, 1, CP1, WAIT1)
GEN_WAIT(2, 1, CP2, WAIT1)
GEN_WAIT(4, 1, CP4, WAIT1)
GEN_WAIT(8, 1, CP8, WAIT1)
GEN_WAIT(1, 2, CP1, WAIT2)
GEN_WAIT(2, 2, CP2, WAIT2)
GEN_WAIT(4, 2, CP4, WAIT2)
GEN_WAIT(8, 2, CP8, WAIT2)
GEN_WAIT(1, 4, CP1, WAIT4)
GEN_WAIT(2, 4, CP2, WAIT4)
GEN_WAIT(4, 4, CP4, WAIT4)
GEN_WAIT(8, 4, CP8, WAIT4)
GEN_WAIT_EMPTY(0, WAIT0)
GEN_WAIT_EMPTY(1, WAIT1)
GEN_WAIT_EMPTY(2, WAIT2)
GEN_WAIT_EMPTY(4, WAIT4)

__device__ __forceinline__ uint64_t dispatch_issue(const uint32_t *s,
                                                   const uint64_t *g,
                                                   int copies, int groups) {
  switch (copies) {
    case 1:
      return run_issue_asm<1>(s, g, groups, 0);
    case 2:
      return run_issue_asm<2>(s, g, groups, 0);
    case 4:
      return run_issue_asm<4>(s, g, groups, 0);
    default:
      return run_issue_asm<8>(s, g, groups, 0);
  }
}

__device__ __forceinline__ uint64_t dispatch_issue_only(const uint32_t *s,
                                                        const uint64_t *g,
                                                        int copies,
                                                        int groups) {
  switch (copies) {
    case 1:
      return run_issue_only_asm<1>(s, g, groups);
    case 2:
      return run_issue_only_asm<2>(s, g, groups);
    case 4:
      return run_issue_only_asm<4>(s, g, groups);
    default:
      return run_issue_only_asm<8>(s, g, groups);
  }
}

__device__ __forceinline__ uint64_t dispatch_complete(const uint32_t *s,
                                                      const uint64_t *g,
                                                      int copies, int groups) {
  switch (copies) {
    case 1:
      return run_complete_asm<1>(s, g, groups, 0);
    case 2:
      return run_complete_asm<2>(s, g, groups, 0);
    case 4:
      return run_complete_asm<4>(s, g, groups, 0);
    default:
      return run_complete_asm<8>(s, g, groups, 0);
  }
}

template <int WaitK>
__device__ __forceinline__ uint64_t dispatch_wait_k(const uint32_t *s,
                                                    const uint64_t *g,
                                                    int copies, int groups,
                                                    int delay) {
  switch (copies) {
    case 1:
      return run_wait_asm<1, WaitK>(s, g, groups, delay);
    case 2:
      return run_wait_asm<2, WaitK>(s, g, groups, delay);
    case 4:
      return run_wait_asm<4, WaitK>(s, g, groups, delay);
    default:
      return run_wait_asm<8, WaitK>(s, g, groups, delay);
  }
}

__device__ __forceinline__ uint64_t dispatch_wait(const uint32_t *s,
                                                  const uint64_t *g, int copies,
                                                  int groups, int wait_k,
                                                  int delay) {
  switch (wait_k) {
    case 1:
      return dispatch_wait_k<1>(s, g, copies, groups, delay);
    case 2:
      return dispatch_wait_k<2>(s, g, copies, groups, delay);
    case 4:
      return dispatch_wait_k<4>(s, g, copies, groups, delay);
    default:
      return dispatch_wait_k<0>(s, g, copies, groups, delay);
  }
}

__device__ __forceinline__ uint64_t dispatch_wait_empty(int groups,
                                                        int wait_k) {
  switch (wait_k) {
    case 1:
      return run_wait_empty_asm<1>(groups);
    case 2:
      return run_wait_empty_asm<2>(groups);
    case 4:
      return run_wait_empty_asm<4>(groups);
    default:
      return run_wait_empty_asm<0>(groups);
  }
}

__global__ void cp_async_ptx_kernel(const uint8_t *global_data,
                                    uint64_t *samples, uint32_t *sinks,
                                    int iters, int warmup, int groups,
                                    int copies, int wait_k, int delay,
                                    int mode_i, int tile_bytes,
                                    int active_warps) {
  extern __shared__ uint8_t smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid / kWarpSize;
  const int total_iters = warmup + iters;
  const int per_warp_tile_bytes = kMaxCopies * kWarpSize * kCpBytes;
  const int timing_bytes =
      ((active_warps * static_cast<int>(sizeof(uint64_t)) + 15) / 16) * 16;
  const int block_base = blockIdx.x * tile_bytes;
  uint64_t *warp_elapsed = reinterpret_cast<uint64_t *>(smem_raw);
  uint8_t *smem = smem_raw + timing_bytes;

  uint32_t s[kMaxCopies];
  uint64_t g[kMaxCopies];
  const uint32_t smem_base =
      smem_addr_u32(smem + warp * per_warp_tile_bytes);
  const uint8_t *gmem_base =
      global_data + block_base + warp * per_warp_tile_bytes;

#pragma unroll
  for (int i = 0; i < kMaxCopies; ++i) {
    const int offset = (i * kWarpSize + lane) * kCpBytes;
    s[i] = smem_base + offset;
    g[i] = gmem_addr_u64(gmem_base + offset);
  }

  uint64_t acc = 0;
  for (int iter = 0; iter < total_iters; ++iter) {
    __syncthreads();
    uint64_t elapsed = 0;
    if (warp < active_warps) {
      switch (mode_i) {
        case kEmpty:
          elapsed = run_empty_asm(groups);
          break;
        case kIssue:
          elapsed = dispatch_issue(s, g, copies, groups);
          asm volatile("cp.async.wait_group 0;\n" ::: "memory");
          break;
        case kComplete:
          elapsed = dispatch_complete(s, g, copies, groups);
          break;
        case kWait:
        case kDelayWait:
          elapsed = dispatch_wait(s, g, copies, groups, wait_k, delay);
          break;
        case kIssueOnly:
          elapsed = dispatch_issue_only(s, g, copies, groups);
          break;
        case kCommitOnly:
          elapsed = run_commit_only_asm(groups);
          break;
        case kWaitEmpty:
          elapsed = dispatch_wait_empty(groups, wait_k);
          break;
      }
    }
    if (warp < active_warps && lane == 0) {
      warp_elapsed[warp] = elapsed;
    }
    __syncthreads();

    if (tid == 0 && iter >= warmup) {
      uint64_t block_elapsed = 0;
      for (int w = 0; w < active_warps; ++w) {
        if (warp_elapsed[w] > block_elapsed) block_elapsed = warp_elapsed[w];
      }
      samples[blockIdx.x * iters + (iter - warmup)] = block_elapsed;
    }
    if (warp < active_warps && lane == 0) {
      acc += *reinterpret_cast<volatile uint32_t *>(
          smem + warp * per_warp_tile_bytes);
    }
  }
  if (warp < active_warps && lane == 0) {
    sinks[blockIdx.x * active_warps + warp] = static_cast<uint32_t>(acc);
  }
}

__global__ void cp_async_bw_kernel(const uint8_t *global_data,
                                   uint64_t *samples, uint32_t *sinks,
                                   int iters, int warmup, int bytes_per_iter,
                                   int rounds_per_commit, int active_warps) {
  extern __shared__ uint8_t smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid / kWarpSize;
  const int total_iters = warmup + iters;
  const int warp_bytes = bytes_per_iter / active_warps;
  const int rounds_per_warp = warp_bytes / (kWarpSize * kCpBytes);
  const int timing_bytes =
      ((active_warps * static_cast<int>(sizeof(uint64_t)) + 15) / 16) * 16;
  const size_t block_iter_base =
      static_cast<size_t>(blockIdx.x) * total_iters * bytes_per_iter;
  uint64_t *warp_elapsed = reinterpret_cast<uint64_t *>(smem_raw);
  uint8_t *smem = smem_raw + timing_bytes;

  uint64_t acc = 0;
  for (int iter = 0; iter < total_iters; ++iter) {
    __syncthreads();
    uint64_t elapsed = 0;
    if (warp < active_warps) {
      const uint8_t *gmem_base =
          global_data + block_iter_base +
          static_cast<size_t>(iter) * bytes_per_iter + warp * warp_bytes;
      uint8_t *smem_base = smem + warp * warp_bytes;
      const uint64_t start = clock64();
      int in_group = 0;
      for (int r = 0; r < rounds_per_warp; ++r) {
        const int offset = (r * kWarpSize + lane) * kCpBytes;
        cp_async_cg_16(smem_addr_u32(smem_base + offset),
                       gmem_addr_u64(gmem_base + offset));
        in_group++;
        if (in_group == rounds_per_commit) {
          cp_async_commit_group();
          in_group = 0;
        }
      }
      if (in_group != 0) cp_async_commit_group();
      cp_async_wait_all();
      const uint64_t end = clock64();
      elapsed = end - start;
    }
    if (warp < active_warps && lane == 0) warp_elapsed[warp] = elapsed;
    __syncthreads();

    if (tid == 0 && iter >= warmup) {
      uint64_t block_elapsed = 0;
      for (int w = 0; w < active_warps; ++w) {
        if (warp_elapsed[w] > block_elapsed) block_elapsed = warp_elapsed[w];
      }
      samples[blockIdx.x * iters + (iter - warmup)] = block_elapsed;
    }
    if (warp < active_warps && lane == 0) {
      const int sink_offset = warp * warp_bytes + ((iter * 128) % warp_bytes);
      acc += *reinterpret_cast<volatile uint32_t *>(smem + sink_offset);
    }
  }
  if (warp < active_warps && lane == 0) {
    sinks[blockIdx.x * active_warps + warp] = static_cast<uint32_t>(acc);
  }
}

const char *mode_name(Mode mode) {
  switch (mode) {
    case kEmpty:
      return "empty";
    case kIssue:
      return "issue";
    case kComplete:
      return "complete";
    case kWait:
      return "wait";
    case kDelayWait:
      return "delay-wait";
    case kIssueOnly:
      return "issue-only";
    case kCommitOnly:
      return "commit-only";
    case kWaitEmpty:
      return "wait-empty";
    case kBandwidth:
      return "bw";
  }
  return "unknown";
}

Mode parse_mode(const char *value) {
  if (std::strcmp(value, "empty") == 0) return kEmpty;
  if (std::strcmp(value, "issue") == 0) return kIssue;
  if (std::strcmp(value, "complete") == 0) return kComplete;
  if (std::strcmp(value, "wait") == 0) return kWait;
  if (std::strcmp(value, "delay-wait") == 0) return kDelayWait;
  if (std::strcmp(value, "issue-only") == 0) return kIssueOnly;
  if (std::strcmp(value, "commit-only") == 0) return kCommitOnly;
  if (std::strcmp(value, "wait-empty") == 0) return kWaitEmpty;
  if (std::strcmp(value, "bw") == 0 || std::strcmp(value, "bandwidth") == 0)
    return kBandwidth;
  std::fprintf(stderr, "unknown mode: %s\n", value);
  std::exit(1);
}

bool parse_int_arg(const char *arg, const char *name, int *out) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  *out = std::atoi(arg + n + 1);
  return true;
}

bool parse_string_arg(const char *arg, const char *name, std::string *out) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  *out = arg + n + 1;
  return true;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (parse_int_arg(arg, "--device", &opt.device) ||
        parse_int_arg(arg, "--blocks", &opt.blocks) ||
        parse_int_arg(arg, "--threads", &opt.threads) ||
        parse_int_arg(arg, "--active-warps", &opt.active_warps) ||
        parse_int_arg(arg, "--iters", &opt.iters) ||
        parse_int_arg(arg, "--warmup", &opt.warmup) ||
        parse_int_arg(arg, "--groups", &opt.groups) ||
        parse_int_arg(arg, "--copies", &opt.copies) ||
        parse_int_arg(arg, "--bytes", &opt.bytes) ||
        parse_int_arg(arg, "--wait-k", &opt.wait_k) ||
        parse_int_arg(arg, "--delay", &opt.delay) ||
        parse_string_arg(arg, "--csv", &opt.csv_path)) {
      continue;
    }
    if (std::strncmp(arg, "--mode=", 7) == 0) {
      opt.mode = parse_mode(arg + 7);
      continue;
    }
    std::fprintf(stderr, "unknown argument: %s\n", arg);
    std::exit(1);
  }
  if (opt.threads < 32 || (opt.threads % 32) != 0) {
    std::fprintf(stderr, "--threads must be a positive multiple of 32\n");
    std::exit(1);
  }
  if (opt.active_warps < 1 || opt.active_warps > opt.threads / kWarpSize) {
    std::fprintf(stderr,
                 "--active-warps must be in [1, --threads/32]\n");
    std::exit(1);
  }
  if (!(opt.copies == 1 || opt.copies == 2 || opt.copies == 4 ||
        opt.copies == 8)) {
    std::fprintf(stderr, "--copies must be 1, 2, 4, or 8\n");
    std::exit(1);
  }
  if (!(opt.wait_k == 0 || opt.wait_k == 1 || opt.wait_k == 2 ||
        opt.wait_k == 4)) {
    std::fprintf(stderr, "--wait-k must be 0, 1, 2, or 4\n");
    std::exit(1);
  }
  if (opt.groups < 1 || opt.iters < 1 || opt.warmup < 0 || opt.delay < 0) {
    std::fprintf(stderr, "invalid non-positive loop parameter\n");
    std::exit(1);
  }
  if (opt.mode == kBandwidth) {
    const int warp_quantum = opt.active_warps * kWarpSize * kCpBytes;
    if (opt.bytes < warp_quantum || (opt.bytes % warp_quantum) != 0) {
      std::fprintf(stderr,
                   "--bytes must be a positive multiple of "
                   "active_warps*32*16 (%d) for bw mode\n",
                   warp_quantum);
      std::exit(1);
    }
    const int rounds_per_warp =
        opt.bytes / (opt.active_warps * kWarpSize * kCpBytes);
    if ((rounds_per_warp % opt.copies) != 0) {
      std::fprintf(stderr,
                   "bw mode requires rounds_per_warp (%d) to be divisible by "
                   "--copies (%d)\n",
                   rounds_per_warp, opt.copies);
      std::exit(1);
    }
  }
  return opt;
}

double measured_op_count(const Options &opt) {
  if (opt.mode == kBandwidth) {
    return static_cast<double>(opt.bytes) / (kWarpSize * kCpBytes);
  }
  switch (opt.mode) {
    case kCommitOnly:
    case kWaitEmpty:
    case kEmpty:
      return static_cast<double>(opt.groups) * opt.active_warps;
    default:
      return static_cast<double>(opt.copies) * opt.groups * opt.active_warps;
  }
}

double measured_byte_count(const Options &opt) {
  if (opt.mode == kBandwidth) return static_cast<double>(opt.bytes);
  switch (opt.mode) {
    case kIssue:
    case kComplete:
    case kWait:
    case kDelayWait:
    case kIssueOnly:
      return measured_op_count(opt) * kWarpSize * kCpBytes;
    default:
      return 0.0;
  }
}

struct Summary {
  uint64_t min = 0;
  uint64_t p01 = 0;
  uint64_t p05 = 0;
  uint64_t p50 = 0;
  uint64_t p90 = 0;
  uint64_t p95 = 0;
  uint64_t p99 = 0;
  uint64_t max = 0;
  double mean = 0.0;
  double stdev = 0.0;
};

uint64_t percentile(const std::vector<uint64_t> &v, double p) {
  if (v.empty()) return 0;
  const double idx = p * static_cast<double>(v.size() - 1);
  return v[static_cast<size_t>(idx + 0.5)];
}

Summary summarize(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  Summary s;
  s.min = values.front();
  s.max = values.back();
  s.p01 = percentile(values, 0.01);
  s.p05 = percentile(values, 0.05);
  s.p50 = percentile(values, 0.50);
  s.p90 = percentile(values, 0.90);
  s.p95 = percentile(values, 0.95);
  s.p99 = percentile(values, 0.99);
  const long double sum =
      std::accumulate(values.begin(), values.end(), (long double)0.0);
  s.mean = static_cast<double>(sum / values.size());
  long double var = 0.0;
  for (uint64_t x : values) {
    const long double d = static_cast<long double>(x) - s.mean;
    var += d * d;
  }
  s.stdev = static_cast<double>(std::sqrt(var / values.size()));
  return s;
}

void append_csv(const Options &opt, const Summary &s, uint64_t sink) {
  if (opt.csv_path.empty()) return;
  const bool exists = static_cast<bool>(std::ifstream(opt.csv_path));
  std::ofstream out(opt.csv_path, std::ios::app);
  if (!exists) {
    out << "mode,copies,groups,wait_k,delay,blocks,threads,active_warps,"
           "iters,bytes,samples,mean,stdev,"
           "min,p01,p05,p50,p90,p95,p99,max,cycles_per_op_p50,"
           "bytes_per_cycle_p50,sink\n";
  }
  const double op_count = measured_op_count(opt);
  const double byte_count = measured_byte_count(opt);
  out << mode_name(opt.mode) << ',' << opt.copies << ',' << opt.groups << ','
      << opt.wait_k << ',' << opt.delay << ',' << opt.blocks << ','
      << opt.threads << ',' << opt.active_warps << ',' << opt.iters << ','
      << static_cast<uint64_t>(byte_count) << ','
      << static_cast<size_t>(opt.blocks) * opt.iters << ','
      << s.mean << ',' << s.stdev << ',' << s.min << ',' << s.p01 << ','
      << s.p05 << ',' << s.p50 << ',' << s.p90 << ',' << s.p95 << ','
      << s.p99 << ',' << s.max << ',' << (s.p50 / op_count) << ','
      << (s.p50 ? byte_count / s.p50 : 0.0) << ',' << sink << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));

  const int per_warp_tile_bytes = kMaxCopies * kWarpSize * kCpBytes;
  const bool bandwidth_mode = opt.mode == kBandwidth;
  const int tile_bytes =
      bandwidth_mode ? opt.bytes : per_warp_tile_bytes * opt.active_warps;
  const size_t timing_bytes =
      ((opt.active_warps * sizeof(uint64_t) + 15) / 16) * 16;
  const size_t sample_count = static_cast<size_t>(opt.blocks) * opt.iters;
  const size_t global_bytes =
      bandwidth_mode
          ? static_cast<size_t>(opt.blocks) * (opt.warmup + opt.iters) *
                opt.bytes
          : static_cast<size_t>(opt.blocks) * tile_bytes;
  const size_t smem_bytes = timing_bytes + tile_bytes;

  uint8_t *d_global = nullptr;
  uint64_t *d_samples = nullptr;
  uint32_t *d_sinks = nullptr;
  CUDA_CHECK(cudaMalloc(&d_global, global_bytes));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(
      &d_sinks,
      static_cast<size_t>(opt.blocks) * opt.active_warps * sizeof(uint32_t)));
  if (!bandwidth_mode) {
    CUDA_CHECK(cudaMemset(d_global, 0x5a, global_bytes));
  }
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(
      d_sinks, 0,
      static_cast<size_t>(opt.blocks) * opt.active_warps * sizeof(uint32_t)));

  if (bandwidth_mode) {
    CUDA_CHECK(cudaFuncSetAttribute(
        cp_async_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    cp_async_bw_kernel<<<opt.blocks, opt.threads, smem_bytes>>>(
        d_global, d_samples, d_sinks, opt.iters, opt.warmup, opt.bytes,
        opt.copies, opt.active_warps);
  } else {
    cp_async_ptx_kernel<<<opt.blocks, opt.threads, smem_bytes>>>(
        d_global, d_samples, d_sinks, opt.iters, opt.warmup, opt.groups,
        opt.copies, opt.wait_k, opt.delay, static_cast<int>(opt.mode),
        tile_bytes, opt.active_warps);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(static_cast<size_t>(opt.blocks) *
                              opt.active_warps);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples, samples.size() * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks, sinks.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  const uint64_t sink =
      std::accumulate(sinks.begin(), sinks.end(), static_cast<uint64_t>(0));
  Summary s = summarize(samples);
  const double op_count = measured_op_count(opt);
  const double byte_count = measured_byte_count(opt);

  std::printf(
      "device=%s mode=%s copies=%d groups=%d wait_k=%d delay=%d blocks=%d "
      "threads=%d active_warps=%d iters=%d warmup=%d bytes=%d samples=%zu\n",
      prop.name, mode_name(opt.mode), opt.copies, opt.groups, opt.wait_k,
      opt.delay, opt.blocks, opt.threads, opt.active_warps, opt.iters,
      opt.warmup, opt.bytes, sample_count);
  std::printf(
      "mean=%.2f stdev=%.2f min=%llu p01=%llu p05=%llu p50=%llu p90=%llu "
      "p95=%llu p99=%llu max=%llu cycles_per_op_p50=%.4f "
      "bytes_per_cycle_p50=%.4f sink=%llu\n",
      s.mean, s.stdev, (unsigned long long)s.min, (unsigned long long)s.p01,
      (unsigned long long)s.p05, (unsigned long long)s.p50,
      (unsigned long long)s.p90, (unsigned long long)s.p95,
      (unsigned long long)s.p99, (unsigned long long)s.max, s.p50 / op_count,
      s.p50 ? byte_count / s.p50 : 0.0, (unsigned long long)sink);

  append_csv(opt, s, sink);

  CUDA_CHECK(cudaFree(d_global));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
