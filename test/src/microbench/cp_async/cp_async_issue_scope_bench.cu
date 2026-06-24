#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kCpBytes = 16;
constexpr int kMaxCopies = 32;

struct Options {
  int device = 0;
  int blocks = 170;
  int threads = 128;
  int active_warps = 1;
  int iters = 4096;
  int warmup = 512;
  int chains = 2;  // number of CP8 groups in the timed asm region
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

#define CP1 "cp.async.cg.shared.global.L2::128B [dst], [src], 16;\n"
#define CP2 CP1 "cp.async.cg.shared.global.L2::128B [dst+0x200], [src], 16;\n"
#define CP4 CP2 "cp.async.cg.shared.global.L2::128B [dst+0x400], [src], 16;\n" \
            "cp.async.cg.shared.global.L2::128B [dst+0x600], [src], 16;\n"
#define CP8 CP4 "cp.async.cg.shared.global.L2::128B [dst+0x800], [src], 16;\n" \
            "cp.async.cg.shared.global.L2::128B [dst+0xa00], [src], 16;\n"     \
            "cp.async.cg.shared.global.L2::128B [dst+0xc00], [src], 16;\n"     \
            "cp.async.cg.shared.global.L2::128B [dst+0xe00], [src], 16;\n"
#define CP16 CP8 "cp.async.cg.shared.global.L2::128B [dst+0x1000], [src], 16;\n" \
             "cp.async.cg.shared.global.L2::128B [dst+0x1200], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1400], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1600], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1800], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1a00], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1c00], [src], 16;\n"    \
             "cp.async.cg.shared.global.L2::128B [dst+0x1e00], [src], 16;\n"
#define CP32 CP16 "cp.async.cg.shared.global.L2::128B [dst+0x2000], [src], 16;\n" \
             "cp.async.cg.shared.global.L2::128B [dst+0x2200], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2400], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2600], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2800], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2a00], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2c00], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x2e00], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3000], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3200], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3400], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3600], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3800], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3a00], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3c00], [src], 16;\n"     \
             "cp.async.cg.shared.global.L2::128B [dst+0x3e00], [src], 16;\n"

#define GEN_CHAIN(Name, Body)                                             \
  __device__ __forceinline__ uint64_t Name(const uint32_t *s,             \
                                           const uint64_t *g) {           \
    uint64_t start = 0;                                                   \
    uint64_t end = 0;                                                     \
    asm volatile("{\n"                                                    \
                 ".reg .u32 dst;\n"                                     \
                 ".reg .u64 src;\n"                                     \
                 "mov.u32 dst, %2;\n"                                   \
                 "mov.u64 src, %3;\n"                                   \
                 "mov.u64 %0, %%clock64;\n" Body                         \
                 "mov.u64 %1, %%clock64;\n"                              \
                 "cp.async.commit_group;\n"                              \
                 "cp.async.wait_group 0;\n"                              \
                 "}\n"                                                    \
                 : "=l"(start), "=l"(end)                                \
                 : "r"(s[0]), "l"(g[0])                                  \
                 : "memory");                                            \
    return end - start;                                                   \
  }

GEN_CHAIN(run_cp8_chain, CP8)
GEN_CHAIN(run_cp16_chain, CP16)
GEN_CHAIN(run_cp32_chain, CP32)

__device__ __forceinline__ uint64_t run_chain(const uint32_t *s,
                                              const uint64_t *g, int chains) {
  if (chains <= 1) return run_cp8_chain(s, g);
  if (chains == 2) return run_cp16_chain(s, g);
  return run_cp32_chain(s, g);
}

__global__ void cp_async_issue_scope_kernel(const uint8_t *global_data,
                                            uint64_t *samples, uint32_t *sinks,
                                            int iters, int warmup,
                                            int active_warps, int chains) {
  extern __shared__ uint8_t smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid / kWarpSize;
  const int total_iters = warmup + iters;
  const int per_warp_bytes = kMaxCopies * kWarpSize * kCpBytes;
  const int timing_bytes =
      ((active_warps * static_cast<int>(sizeof(uint64_t)) + 15) / 16) * 16;
  uint64_t *warp_elapsed = reinterpret_cast<uint64_t *>(smem_raw);
  uint8_t *smem = smem_raw + timing_bytes;

  uint32_t s[kMaxCopies];
  uint64_t g[kMaxCopies];
  const uint32_t smem_base = smem_addr_u32(smem + warp * per_warp_bytes);
  const uint8_t *gmem_base =
      global_data + blockIdx.x * active_warps * per_warp_bytes +
      warp * per_warp_bytes;

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
      elapsed = run_chain(s, g, chains);
    }
    if (warp < active_warps && lane == 0) {
      warp_elapsed[warp] = elapsed;
    }
    __syncthreads();
    if (tid == 0 && iter >= warmup) {
      uint64_t block_elapsed = 0;
      for (int w = 0; w < active_warps; ++w) {
        block_elapsed =
            block_elapsed > warp_elapsed[w] ? block_elapsed : warp_elapsed[w];
      }
      samples[blockIdx.x * iters + (iter - warmup)] = block_elapsed;
    }
    if (warp < active_warps && lane == 0) {
      acc += *reinterpret_cast<volatile uint32_t *>(
          smem + warp * per_warp_bytes);
    }
  }
  if (warp < active_warps && lane == 0) {
    sinks[blockIdx.x * active_warps + warp] = static_cast<uint32_t>(acc);
  }
}

bool parse_int_arg(const char *arg, const char *name, int *value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = std::atoi(arg + n + 1);
    return true;
  }
  return false;
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
        parse_int_arg(arg, "--chains", &opt.chains)) {
      continue;
    }
    std::fprintf(stderr, "unknown arg: %s\n", arg);
    std::exit(1);
  }
  if (opt.threads % kWarpSize != 0 || opt.active_warps < 1 ||
      opt.active_warps > opt.threads / kWarpSize ||
      (opt.chains != 1 && opt.chains != 2 && opt.chains != 4)) {
    std::fprintf(stderr,
                 "requires threads%%32==0, 1<=active_warps<=threads/32, "
                 "chains in {1,2,4}\n");
    std::exit(1);
  }
  return opt;
}

uint64_t percentile(const std::vector<uint64_t> &values, double q) {
  const double idx = q * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(idx + 0.5)];
}

}  // namespace

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));

  const int per_warp_bytes = kMaxCopies * kWarpSize * kCpBytes;
  const size_t timing_bytes =
      ((opt.active_warps * sizeof(uint64_t) + 15) / 16) * 16;
  const size_t smem_bytes =
      timing_bytes + static_cast<size_t>(opt.threads / kWarpSize) *
                         static_cast<size_t>(per_warp_bytes);
  const size_t global_bytes =
      static_cast<size_t>(opt.blocks) * opt.active_warps * per_warp_bytes;
  const size_t sample_count = static_cast<size_t>(opt.blocks) * opt.iters;

  uint8_t *d_global = nullptr;
  uint64_t *d_samples = nullptr;
  uint32_t *d_sinks = nullptr;
  CUDA_CHECK(cudaMalloc(&d_global, global_bytes));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sinks, static_cast<size_t>(opt.blocks) *
                                      opt.active_warps * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_global, 0x5a, global_bytes));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sinks, 0, static_cast<size_t>(opt.blocks) *
                                      opt.active_warps * sizeof(uint32_t)));

  CUDA_CHECK(cudaFuncSetAttribute(
      cp_async_issue_scope_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  cp_async_issue_scope_kernel<<<opt.blocks, opt.threads, smem_bytes>>>(
      d_global, d_samples, d_sinks, opt.iters, opt.warmup, opt.active_warps,
      opt.chains);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(static_cast<size_t>(opt.blocks) *
                              opt.active_warps);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        samples.size() * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks,
                        sinks.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  std::sort(samples.begin(), samples.end());
  const long double sum =
      std::accumulate(samples.begin(), samples.end(), static_cast<long double>(0));
  const double mean = static_cast<double>(sum / samples.size());
  const uint64_t sink =
      std::accumulate(sinks.begin(), sinks.end(), static_cast<uint64_t>(0));
  const int cp_per_warp = opt.chains * 8;
  const double aggregate_cp = static_cast<double>(cp_per_warp * opt.active_warps);
  const double p50 = static_cast<double>(percentile(samples, 0.50));

  std::printf(
      "device=%s blocks=%d threads=%d active_warps=%d chains=%d cp_per_warp=%d "
      "iters=%d warmup=%d samples=%zu\n",
      prop.name, opt.blocks, opt.threads, opt.active_warps, opt.chains,
      cp_per_warp, opt.iters, opt.warmup, samples.size());
  std::printf(
      "mean=%.2f min=%llu p05=%llu p50=%llu p95=%llu max=%llu "
      "agg_cp_per_cycle_p50=%.6f equiv_cycle_per_cp_sm_p50=%.3f "
      "per_warp_cycle_per_cp_p50=%.3f sink=%llu\n",
      mean, static_cast<unsigned long long>(samples.front()),
      static_cast<unsigned long long>(percentile(samples, 0.05)),
      static_cast<unsigned long long>(percentile(samples, 0.50)),
      static_cast<unsigned long long>(percentile(samples, 0.95)),
      static_cast<unsigned long long>(samples.back()), aggregate_cp / p50,
      p50 / aggregate_cp, p50 / static_cast<double>(cp_per_warp),
      static_cast<unsigned long long>(sink));

  CUDA_CHECK(cudaFree(d_global));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
