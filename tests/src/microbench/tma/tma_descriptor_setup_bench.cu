#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kTensormapSize = 128;
constexpr int kHalfBytes = 2;

enum BenchCase {
  kCaseReplace = 0,
  kCasePublish = 1,
  kCaseFull = 2,
  kCasePair = 3,
};

struct Options {
  int device = 0;
  int blocks = 128;
  int iters = 4096;
  int warmup = 256;
  int tiles = 4096;
  BenchCase bench_case = kCaseFull;
  bool hot = false;
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

__host__ __device__ inline uintptr_t align_up(uintptr_t value,
                                              uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void tensormap_cp_fenceproxy(uint8_t* global_tmap,
                                                        uint8_t* smem_tmap) {
  const uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
  const uint64_t tmap_s = __cvta_generic_to_shared(smem_tmap);
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic."
      "release.gpu.sync.aligned [%0], [%1], 0x80;\n"
      :
      : "l"(tmap_g), "l"(tmap_s)
      : "memory");
}

__device__ __forceinline__ void fence_proxy_tensormap_acquire(
    uint8_t* global_tmap) {
  const uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
  asm volatile(
      "fence.proxy.tensormap::generic.acquire.gpu [%0], 0x80;\n"
      "cp.async.bulk.commit_group;\n"
      "cp.async.bulk.wait_group.read 0;\n"
      :
      : "l"(tmap_g)
      : "memory");
}

__device__ __forceinline__ void setup_tensormap_2d_fp16(
    uint8_t* tmap_smem, const uint16_t* base, uint32_t box0, uint32_t box1,
    uint32_t dim0, uint32_t dim1) {
  const uint64_t tmap_s = __cvta_generic_to_shared(tmap_smem);
  const uint64_t base_u64 = reinterpret_cast<uint64_t>(base);
  const uint64_t stride0 = static_cast<uint64_t>(dim0) * kHalfBytes;
  const uint32_t elem_stride = 1;

  asm volatile(
      "tensormap.replace.tile.global_address.shared::cta.b1024.b64 "
      "[%0], %1;\n"
      :
      : "l"(tmap_s), "l"(base_u64)
      : "memory");
  asm volatile("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x1;\n"
               :
               : "l"(tmap_s)
               : "memory");
  asm volatile(
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x0, %1;\n"
      :
      : "l"(tmap_s), "r"(box0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 0x1, %1;\n"
      :
      : "l"(tmap_s), "r"(box1)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x0, "
      "%1;\n"
      :
      : "l"(tmap_s), "r"(dim0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0x1, "
      "%1;\n"
      :
      : "l"(tmap_s), "r"(dim1)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0x0, "
      "%1;\n"
      :
      : "l"(tmap_s), "l"(stride0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x0, "
      "%1;\n"
      :
      : "l"(tmap_s), "r"(elem_stride)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], 0x1, "
      "%1;\n"
      :
      : "l"(tmap_s), "r"(elem_stride)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x6;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], "
      "0x0;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], 0x3;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;\n"
      :
      : "l"(tmap_s)
      : "memory");
}

__device__ __forceinline__ void zero_tensormap_warp(int tid,
                                                    uint8_t* tmap_smem) {
  if (tid < 32) {
    reinterpret_cast<uint32_t*>(tmap_smem)[tid] = 0;
  }
  asm volatile("bar.warp.sync -1;" ::: "memory");
}

__device__ __forceinline__ void publish_tensormap_warp(int tid,
                                                       uint8_t* global_tmap,
                                                       uint8_t* smem_tmap) {
  if (tid < 32) {
    tensormap_cp_fenceproxy(global_tmap, smem_tmap);
    fence_proxy_tensormap_acquire(global_tmap);
  }
}

__device__ __forceinline__ void replace_one_tensormap(
    int tid, uint8_t* smem_tmap, const uint16_t* base, uint32_t box0,
    uint32_t box1, uint32_t dim0, uint32_t dim1) {
  zero_tensormap_warp(tid, smem_tmap);
  if (tid == 0) {
    setup_tensormap_2d_fp16(smem_tmap, base, box0, box1, dim0, dim1);
  }
}

__device__ __forceinline__ void full_one_tensormap(
    int tid, uint8_t* global_tmap, uint8_t* smem_tmap, const uint16_t* base,
    uint32_t box0, uint32_t box1, uint32_t dim0, uint32_t dim1) {
  replace_one_tensormap(tid, smem_tmap, base, box0, box1, dim0, dim1);
  publish_tensormap_warp(tid, global_tmap, smem_tmap);
}

__global__ void tma_descriptor_setup_kernel(
    const uint16_t* a, const uint16_t* b, uint8_t* global_tmaps,
    uint64_t* samples, uint32_t* block_sinks, int iters, int warmup, int tiles,
    int bench_case_i, int hot) {
  extern __shared__ uint8_t smem[];

  const int tid = threadIdx.x;
  const BenchCase bench_case = static_cast<BenchCase>(bench_case_i);
  uintptr_t p = reinterpret_cast<uintptr_t>(smem);
  uint8_t* tmap_a_smem = reinterpret_cast<uint8_t*>(align_up(p, 128));
  p = reinterpret_cast<uintptr_t>(tmap_a_smem) + kTensormapSize;
  uint8_t* tmap_b_smem = reinterpret_cast<uint8_t*>(align_up(p, 128));

  uint8_t* global_tmap_a = global_tmaps + static_cast<size_t>(blockIdx.x) * 256;
  uint8_t* global_tmap_b = global_tmap_a + 128;

  replace_one_tensormap(tid, tmap_a_smem, a, 64, 64, 128,
                        static_cast<uint32_t>(64 * tiles));
  replace_one_tensormap(tid, tmap_b_smem, b, 64, 128, 128,
                        static_cast<uint32_t>(128 * tiles));
  __syncthreads();

  uint32_t sink = 0;
  const int total_iters = warmup + iters;
  for (int i = 0; i < total_iters; ++i) {
    const int tile =
        hot ? 0
            : static_cast<int>(
                  (static_cast<uint64_t>(i) * gridDim.x + blockIdx.x) % tiles);
    const uint16_t* a_base = a + static_cast<size_t>(tile) * 128 * 64;
    const uint16_t* b_base = b + static_cast<size_t>(tile) * 128 * 128;

    __syncthreads();
    uint64_t start = 0;
    if (tid == 0) {
      start = clock64_now();
    }

    if (bench_case == kCaseReplace) {
      replace_one_tensormap(tid, tmap_a_smem, a_base, 64, 64, 128,
                            static_cast<uint32_t>(64 * tiles));
      __syncthreads();
    } else if (bench_case == kCasePublish) {
      publish_tensormap_warp(tid, global_tmap_a, tmap_a_smem);
      __syncthreads();
    } else if (bench_case == kCaseFull) {
      full_one_tensormap(tid, global_tmap_a, tmap_a_smem, a_base, 64, 64, 128,
                         static_cast<uint32_t>(64 * tiles));
      __syncthreads();
    } else {
      full_one_tensormap(tid, global_tmap_a, tmap_a_smem, a_base, 64, 64, 128,
                         static_cast<uint32_t>(64 * tiles));
      __syncthreads();
      full_one_tensormap(tid, global_tmap_b, tmap_b_smem, b_base, 64, 128, 128,
                         static_cast<uint32_t>(128 * tiles));
      __syncthreads();
    }

    if (tid == 0) {
      const uint64_t end = clock64_now();
      if (i >= warmup) {
        samples[static_cast<size_t>(blockIdx.x) * iters + (i - warmup)] =
            end - start;
      }
      sink += reinterpret_cast<volatile uint32_t*>(tmap_a_smem)[0];
      sink += reinterpret_cast<volatile uint32_t*>(global_tmap_a)[0];
    }
  }

  if (tid == 0) {
    block_sinks[blockIdx.x] = sink;
  }
}

const char* case_name(BenchCase c) {
  switch (c) {
    case kCaseReplace:
      return "replace";
    case kCasePublish:
      return "publish";
    case kCaseFull:
      return "full";
    case kCasePair:
      return "pair";
  }
  return "unknown";
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device N] [--blocks N] [--iters N] [--warmup N]\n"
      "          [--tiles N] [--case replace|publish|full|pair]\n"
      "          [--hot|--stride] [--csv path]\n",
      argv0);
}

bool parse_int_arg(const char* arg, const char* name, int* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = std::atoi(arg + n + 1);
    return true;
  }
  return false;
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else if (parse_int_arg(arg, "--device", &opts.device) ||
               parse_int_arg(arg, "--blocks", &opts.blocks) ||
               parse_int_arg(arg, "--iters", &opts.iters) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--tiles", &opts.tiles)) {
      continue;
    } else if (std::strncmp(arg, "--case=", 7) == 0) {
      const char* value = arg + 7;
      if (std::strcmp(value, "replace") == 0) {
        opts.bench_case = kCaseReplace;
      } else if (std::strcmp(value, "publish") == 0) {
        opts.bench_case = kCasePublish;
      } else if (std::strcmp(value, "full") == 0) {
        opts.bench_case = kCaseFull;
      } else if (std::strcmp(value, "pair") == 0) {
        opts.bench_case = kCasePair;
      } else {
        std::fprintf(stderr, "Unknown case: %s\n", value);
        std::exit(1);
      }
    } else if (std::strcmp(arg, "--hot") == 0) {
      opts.hot = true;
    } else if (std::strcmp(arg, "--stride") == 0) {
      opts.hot = false;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.blocks <= 0 || opts.iters <= 0 || opts.warmup < 0 ||
      opts.tiles <= 0) {
    std::fprintf(stderr,
                 "blocks, iters, and tiles must be positive; warmup must be "
                 "non-negative\n");
    std::exit(1);
  }
  return opts;
}

size_t dynamic_smem_bytes() {
  uintptr_t p = 0;
  p = align_up(p, 128) + kTensormapSize;
  p = align_up(p, 128) + kTensormapSize;
  return align_up(p, 128);
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double q) {
  if (sorted.empty()) {
    return 0;
  }
  const double pos = q * static_cast<double>(sorted.size() - 1);
  return sorted[static_cast<size_t>(std::llround(pos))];
}

void print_stats(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double sum =
      std::accumulate(sorted.begin(), sorted.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  const double mean = sum / static_cast<double>(sorted.size());
  double sq = 0.0;
  for (uint64_t v : sorted) {
    const double d = static_cast<double>(v) - mean;
    sq += d * d;
  }
  const double stdev = std::sqrt(sq / static_cast<double>(sorted.size()));

  std::printf(
      "samples=%zu mean=%.2f stdev=%.2f min=%llu p01=%llu p05=%llu "
      "p50=%llu p90=%llu p95=%llu p99=%llu p999=%llu max=%llu\n",
      sorted.size(), mean, stdev,
      static_cast<unsigned long long>(sorted.front()),
      static_cast<unsigned long long>(percentile(sorted, 0.01)),
      static_cast<unsigned long long>(percentile(sorted, 0.05)),
      static_cast<unsigned long long>(percentile(sorted, 0.50)),
      static_cast<unsigned long long>(percentile(sorted, 0.90)),
      static_cast<unsigned long long>(percentile(sorted, 0.95)),
      static_cast<unsigned long long>(percentile(sorted, 0.99)),
      static_cast<unsigned long long>(percentile(sorted, 0.999)),
      static_cast<unsigned long long>(sorted.back()));
}

void write_csv(const std::string& path, const std::vector<uint64_t>& values,
               int blocks, int iters) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "block,iter,cycles\n";
  for (int b = 0; b < blocks; ++b) {
    for (int i = 0; i < iters; ++i) {
      out << b << "," << i << "," << values[static_cast<size_t>(b) * iters + i]
          << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate,
                                    opts.device));
  const size_t smem_bytes = dynamic_smem_bytes();

  CUDA_CHECK(cudaFuncSetAttribute(tma_descriptor_setup_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));

  const size_t a_elems = static_cast<size_t>(128) * 64 * opts.tiles;
  const size_t b_elems = static_cast<size_t>(128) * 128 * opts.tiles;
  const size_t sample_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.iters);

  uint16_t* d_a = nullptr;
  uint16_t* d_b = nullptr;
  uint8_t* d_tmaps = nullptr;
  uint64_t* d_samples = nullptr;
  uint32_t* d_sinks = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_tmaps, static_cast<size_t>(opts.blocks) * 256));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(d_a, 0x3c, a_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemset(d_b, 0x5a, b_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemset(d_tmaps, 0, static_cast<size_t>(opts.blocks) * 256));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sinks, 0,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "case=%s locality=%s blocks=%d threads=%d iters=%d warmup=%d tiles=%d "
      "smem=%zu bytes\n",
      case_name(opts.bench_case), opts.hot ? "hot" : "stride", opts.blocks,
      kThreads, opts.iters, opts.warmup, opts.tiles, smem_bytes);

  tma_descriptor_setup_kernel<<<opts.blocks, kThreads, smem_bytes>>>(
      d_a, d_b, d_tmaps, d_samples, d_sinks, opts.iters, opts.warmup,
      opts.tiles, static_cast<int>(opts.bench_case), opts.hot ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(opts.blocks);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        sample_count * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  const uint64_t sink_sum =
      std::accumulate(sinks.begin(), sinks.end(), uint64_t{0});
  print_stats(samples);
  std::printf("sink=%llu\n", static_cast<unsigned long long>(sink_sum));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.blocks, opts.iters);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_tmaps));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
