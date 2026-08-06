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
constexpr int kASubtileBytes = 64 * 64 * kHalfBytes;
constexpr int kBSubtileBytes = 64 * 128 * kHalfBytes;
constexpr int kATotalBytes = 2 * kASubtileBytes;
constexpr int kBTotalBytes = 2 * kBSubtileBytes;
constexpr int kPairBytes = kATotalBytes + kBTotalBytes;

enum BenchCase {
  kCaseEmpty = 0,
  kCaseA16 = 1,
  kCaseB32 = 2,
  kCasePair48 = 3,
};

enum IssueMode {
  kIssueLlama = 0,
  kIssueSerial = 1,
};

enum LocalityMode {
  kLocalityStride = 0,
  kLocalityHot = 1,
  kLocalityBlockHot = 2,
};

struct Options {
  int device = 0;
  int blocks = 128;
  int iters = 4096;
  int warmup = 256;
  int tiles = 4096;
  int repeat = 1;
  BenchCase bench_case = kCasePair48;
  IssueMode issue_mode = kIssueLlama;
  LocalityMode locality = kLocalityStride;
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

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(smem_addr_u32(bar)), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(smem_addr_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar,
                                                     uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "waitLoop:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
      "@!complete bra.uni waitLoop;\n"
      "}\n"
      :
      : "r"(smem_addr_u32(bar)), "r"(parity)
      : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
               :
               : "r"(smem_addr_u32(bar))
               : "memory");
}

__device__ __forceinline__ void tensormap_cp_fenceproxy(uint8_t* global_tmap,
                                                        uint8_t* smem_tmap) {
  uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
  uint64_t tmap_s = __cvta_generic_to_shared(smem_tmap);
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic."
      "release.gpu.sync.aligned [%0], [%1], 0x80;\n"
      :
      : "l"(tmap_g), "l"(tmap_s)
      : "memory");
}

__device__ __forceinline__ void fence_proxy_tensormap_acquire(
    uint8_t* global_tmap) {
  uint64_t tmap_g = reinterpret_cast<uint64_t>(global_tmap);
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
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], "
      "0x0, %1;\n"
      :
      : "l"(tmap_s), "r"(box0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], "
      "0x1, %1;\n"
      :
      : "l"(tmap_s), "r"(box1)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], "
      "0x0, %1;\n"
      :
      : "l"(tmap_s), "r"(dim0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], "
      "0x1, %1;\n"
      :
      : "l"(tmap_s), "r"(dim1)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 "
      "[%0], 0x0, %1;\n"
      :
      : "l"(tmap_s), "l"(stride0)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 "
      "[%0], 0x0, %1;\n"
      :
      : "l"(tmap_s), "r"(elem_stride)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 "
      "[%0], 0x1, %1;\n"
      :
      : "l"(tmap_s), "r"(elem_stride)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], "
      "0x6;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 "
      "[%0], 0x0;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 "
      "[%0], 0x3;\n"
      :
      : "l"(tmap_s)
      : "memory");
  asm volatile(
      "tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], "
      "0x0;\n"
      :
      : "l"(tmap_s)
      : "memory");
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_cta(
    uint8_t* smem_dst, uint8_t* global_tmap, int32_t coord0, int32_t coord1,
    uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(smem_addr_u32(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(global_tmap)), "r"(coord0), "r"(coord1),
        "r"(smem_addr_u32(bar))
      : "memory");
}

__device__ __forceinline__ void issue_a16(int tid, IssueMode issue_mode,
                                          uint8_t* dst_a0, uint8_t* dst_a1,
                                          uint8_t* tmap_a, int coord1,
                                          uint64_t* bar) {
  if (issue_mode == kIssueSerial) {
    if (tid == 0) {
      cp_async_bulk_tensor_2d_cta(dst_a0, tmap_a, 0, coord1, bar);
      cp_async_bulk_tensor_2d_cta(dst_a1, tmap_a, 64, coord1, bar);
    }
  } else {
    if (tid == 0) {
      cp_async_bulk_tensor_2d_cta(dst_a0, tmap_a, 0, coord1, bar);
    } else if (tid == 32) {
      cp_async_bulk_tensor_2d_cta(dst_a1, tmap_a, 64, coord1, bar);
    }
  }
}

__device__ __forceinline__ void issue_b32(int tid, IssueMode issue_mode,
                                          uint8_t* dst_b0, uint8_t* dst_b1,
                                          uint8_t* tmap_b, int coord1,
                                          uint64_t* bar) {
  if (issue_mode == kIssueSerial) {
    if (tid == 0) {
      cp_async_bulk_tensor_2d_cta(dst_b0, tmap_b, 0, coord1, bar);
      cp_async_bulk_tensor_2d_cta(dst_b1, tmap_b, 64, coord1, bar);
    }
  } else {
    if (tid == 0) {
      cp_async_bulk_tensor_2d_cta(dst_b0, tmap_b, 0, coord1, bar);
    } else if (tid == 32) {
      cp_async_bulk_tensor_2d_cta(dst_b1, tmap_b, 64, coord1, bar);
    }
  }
}

__global__ void tma_completion_latency_kernel(
    const uint16_t* a, const uint16_t* b, uint8_t* global_tmaps,
    uint64_t* samples, uint32_t* block_sinks, int iters, int warmup, int tiles,
    int repeat, int bench_case_i, int issue_mode_i, int locality_i) {
  extern __shared__ uint8_t smem[];

  const int tid = threadIdx.x;
  const BenchCase bench_case = static_cast<BenchCase>(bench_case_i);
  const IssueMode issue_mode = static_cast<IssueMode>(issue_mode_i);
  const LocalityMode locality = static_cast<LocalityMode>(locality_i);
  uintptr_t p = reinterpret_cast<uintptr_t>(smem);

  uint8_t* tmap_a_smem = reinterpret_cast<uint8_t*>(align_up(p, 128));
  p = reinterpret_cast<uintptr_t>(tmap_a_smem) + kTensormapSize;
  uint8_t* tmap_b_smem = reinterpret_cast<uint8_t*>(align_up(p, 128));
  p = reinterpret_cast<uintptr_t>(tmap_b_smem) + kTensormapSize;
  uint8_t* dst_a0 = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_a0) + kASubtileBytes;
  uint8_t* dst_a1 = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_a1) + kASubtileBytes;
  uint8_t* dst_b0 = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_b0) + kBSubtileBytes;
  uint8_t* dst_b1 = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_b1) + kBSubtileBytes;
  uint64_t* bar = reinterpret_cast<uint64_t*>(align_up(p, 8));

  uint8_t* global_tmap_a = global_tmaps + blockIdx.x * 256;
  uint8_t* global_tmap_b = global_tmap_a + 128;

  if (tid < 32) {
    reinterpret_cast<uint32_t*>(tmap_a_smem)[tid] = 0;
    reinterpret_cast<uint32_t*>(tmap_b_smem)[tid] = 0;
  }
  asm volatile("bar.warp.sync -1;" ::: "memory");
  __syncthreads();

  if (tid == 0) {
    setup_tensormap_2d_fp16(tmap_a_smem, a, 64, 64, 128,
                            static_cast<uint32_t>(64 * tiles));
    setup_tensormap_2d_fp16(tmap_b_smem, b, 64, 128, 128,
                            static_cast<uint32_t>(128 * tiles));
  }
  __syncthreads();

  if (tid < 32) {
    tensormap_cp_fenceproxy(global_tmap_a, tmap_a_smem);
    fence_proxy_tensormap_acquire(global_tmap_a);
    tensormap_cp_fenceproxy(global_tmap_b, tmap_b_smem);
    fence_proxy_tensormap_acquire(global_tmap_b);
  }
  __syncthreads();

  const uint32_t bytes_per_repeat =
      bench_case == kCasePair48
          ? kPairBytes
          : (bench_case == kCaseA16
                 ? kATotalBytes
                 : (bench_case == kCaseB32 ? kBTotalBytes : 0));
  const uint32_t expected_bytes = bytes_per_repeat * repeat;
  uint32_t sink = 0;
  const int total_iters = warmup + iters;

  for (int i = 0; i < total_iters; ++i) {
    if (tid == 0) {
      mbarrier_init(bar, 1);
      mbarrier_arrive_expect_tx(bar, expected_bytes);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0) {
      start = clock64_now();
    }

    for (int r = 0; r < repeat; ++r) {
      uint64_t tile = 0;
      if (locality == kLocalityHot) {
        tile = static_cast<uint64_t>(r);
      } else if (locality == kLocalityBlockHot) {
        tile = static_cast<uint64_t>(blockIdx.x) * repeat + r;
      } else {
        tile = (static_cast<uint64_t>(i) * gridDim.x + blockIdx.x) * repeat + r;
      }
      tile %= static_cast<uint64_t>(tiles);
      const int a_coord1 = static_cast<int>(tile) * 64;
      const int b_coord1 = static_cast<int>(tile) * 128;

      if (bench_case == kCaseA16 || bench_case == kCasePair48) {
        issue_a16(tid, issue_mode, dst_a0, dst_a1, global_tmap_a, a_coord1,
                  bar);
      }

      if (bench_case == kCasePair48 && issue_mode == kIssueLlama) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
      }

      if (bench_case == kCaseB32 || bench_case == kCasePair48) {
        issue_b32(tid, issue_mode, dst_b0, dst_b1, global_tmap_b, b_coord1,
                  bar);
      }
    }

    if (tid == 0) {
      mbarrier_wait_parity(bar, 0);
      const uint64_t end = clock64_now();
      if (i >= warmup) {
        samples[blockIdx.x * iters + (i - warmup)] = end - start;
      }
      sink += *reinterpret_cast<volatile uint16_t*>(dst_a0);
      sink += *reinterpret_cast<volatile uint16_t*>(dst_b0);
      mbarrier_inval(bar);
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sinks[blockIdx.x] = sink;
  }
}

const char* case_name(BenchCase c) {
  switch (c) {
    case kCaseEmpty:
      return "empty";
    case kCaseA16:
      return "a16";
    case kCaseB32:
      return "b32";
    case kCasePair48:
      return "pair48";
  }
  return "unknown";
}

const char* issue_name(IssueMode mode) {
  return mode == kIssueSerial ? "serial" : "llama";
}

const char* locality_name(LocalityMode mode) {
  switch (mode) {
    case kLocalityStride:
      return "stride";
    case kLocalityHot:
      return "hot";
    case kLocalityBlockHot:
      return "block-hot";
  }
  return "unknown";
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device N] [--blocks N] [--iters N] [--warmup N]\n"
      "          [--tiles N] [--repeat N] [--case empty|a16|b32|pair48]\n"
      "          [--issue llama|serial] [--hot|--block-hot|--stride]\n"
      "          [--csv path]\n",
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
               parse_int_arg(arg, "--tiles", &opts.tiles) ||
               parse_int_arg(arg, "--repeat", &opts.repeat)) {
      continue;
    } else if (std::strncmp(arg, "--case=", 7) == 0) {
      const char* value = arg + 7;
      if (std::strcmp(value, "empty") == 0) {
        opts.bench_case = kCaseEmpty;
      } else if (std::strcmp(value, "a16") == 0) {
        opts.bench_case = kCaseA16;
      } else if (std::strcmp(value, "b32") == 0) {
        opts.bench_case = kCaseB32;
      } else if (std::strcmp(value, "pair48") == 0) {
        opts.bench_case = kCasePair48;
      } else {
        std::fprintf(stderr, "Unknown case: %s\n", value);
        std::exit(1);
      }
    } else if (std::strncmp(arg, "--issue=", 8) == 0) {
      const char* value = arg + 8;
      if (std::strcmp(value, "llama") == 0) {
        opts.issue_mode = kIssueLlama;
      } else if (std::strcmp(value, "serial") == 0) {
        opts.issue_mode = kIssueSerial;
      } else {
        std::fprintf(stderr, "Unknown issue mode: %s\n", value);
        std::exit(1);
      }
    } else if (std::strcmp(arg, "--hot") == 0) {
      opts.locality = kLocalityHot;
    } else if (std::strcmp(arg, "--block-hot") == 0) {
      opts.locality = kLocalityBlockHot;
    } else if (std::strcmp(arg, "--stride") == 0) {
      opts.locality = kLocalityStride;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.blocks <= 0 || opts.iters <= 0 || opts.warmup < 0 ||
      opts.repeat <= 0 || opts.tiles <= 0) {
    std::fprintf(stderr,
                 "blocks, iters, tiles, and repeat must be positive; warmup "
                 "must be non-negative\n");
    std::exit(1);
  }
  return opts;
}

size_t dynamic_smem_bytes() {
  uintptr_t p = 0;
  p = align_up(p, 128) + kTensormapSize;
  p = align_up(p, 128) + kTensormapSize;
  p = align_up(p, 1024) + kASubtileBytes;
  p = align_up(p, 1024) + kASubtileBytes;
  p = align_up(p, 1024) + kBSubtileBytes;
  p = align_up(p, 1024) + kBSubtileBytes;
  p = align_up(p, 8) + sizeof(uint64_t);
  return align_up(p, 1024);
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

double mean_cycles(const std::vector<uint64_t>& values) {
  const double sum =
      std::accumulate(values.begin(), values.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  return sum / static_cast<double>(values.size());
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
      out << b << "," << i << "," << values[b * iters + i] << "\n";
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

  CUDA_CHECK(cudaFuncSetAttribute(tma_completion_latency_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));
  CUDA_CHECK(
      cudaFuncSetAttribute(tma_completion_latency_kernel,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxShared));

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
      "case=%s issue=%s locality=%s blocks=%d threads=%d iters=%d warmup=%d "
      "tiles=%d repeat=%d smem=%zu bytes expected_tx=%d\n",
      case_name(opts.bench_case), issue_name(opts.issue_mode),
      locality_name(opts.locality), opts.blocks, kThreads, opts.iters,
      opts.warmup, opts.tiles, opts.repeat, smem_bytes,
      opts.bench_case == kCasePair48
          ? kPairBytes * opts.repeat
          : (opts.bench_case == kCaseA16
                 ? kATotalBytes * opts.repeat
                 : (opts.bench_case == kCaseB32 ? kBTotalBytes * opts.repeat
                                                : 0)));
  const uint64_t expected_tx_bytes =
      opts.bench_case == kCasePair48
          ? static_cast<uint64_t>(kPairBytes) * opts.repeat
          : (opts.bench_case == kCaseA16
                 ? static_cast<uint64_t>(kATotalBytes) * opts.repeat
                 : (opts.bench_case == kCaseB32
                        ? static_cast<uint64_t>(kBTotalBytes) * opts.repeat
                        : 0));
  const uint64_t measured_data_bytes =
      expected_tx_bytes * static_cast<uint64_t>(opts.blocks) *
      static_cast<uint64_t>(opts.iters);
  const double working_set_mib =
      (static_cast<double>(kPairBytes) * static_cast<double>(opts.tiles)) /
      (1024.0 * 1024.0);
  std::printf("working_set_pair48_mib=%.2f measured_data_bytes=%llu\n",
              working_set_mib,
              static_cast<unsigned long long>(measured_data_bytes));

  tma_completion_latency_kernel<<<opts.blocks, kThreads, smem_bytes>>>(
      d_a, d_b, d_tmaps, d_samples, d_sinks, opts.iters, opts.warmup,
      opts.tiles, opts.repeat, static_cast<int>(opts.bench_case),
      static_cast<int>(opts.issue_mode), static_cast<int>(opts.locality));
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
  const double mean = mean_cycles(samples);
  const double aggregate_bytes_per_cycle =
      mean > 0.0 ? static_cast<double>(expected_tx_bytes) *
                       static_cast<double>(opts.blocks) / mean
                 : 0.0;
  const double aggregate_gib_per_sec =
      aggregate_bytes_per_cycle * static_cast<double>(clock_rate_khz) * 1000.0 /
      (1024.0 * 1024.0 * 1024.0);
  std::printf("mean_aggregate_tma_bw_gib_s=%.2f bytes_per_cycle=%.2f\n",
              aggregate_gib_per_sec, aggregate_bytes_per_cycle);
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
