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

constexpr int kDefaultThreads = 256;
constexpr int kLoadBytes = 16;
constexpr int kLoadUnroll = 4;
constexpr size_t kDefaultTileBytes = 48 * 1024;

enum LocalityMode {
  kLocalityStride = 0,
  kLocalityHot = 1,
  kLocalityBlockHot = 2,
};

struct Options {
  int device = 0;
  int blocks = 128;
  int threads = kDefaultThreads;
  int iters = 4096;
  int warmup = 256;
  int tiles = 4096;
  int repeat = 1;
  size_t tile_bytes = kDefaultTileBytes;
  LocalityMode locality = kLocalityStride;
  bool event_timing = false;
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

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint4 ld_global_cg_u4(const void* ptr) {
  uint4 value;
  const uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_global(ptr));
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(addr)
               : "memory");
  return value;
}

struct U4x4 {
  uint4 v0;
  uint4 v1;
  uint4 v2;
  uint4 v3;
};

__device__ __forceinline__ U4x4 ld_global_cg_u4x4(const void* ptr0,
                                                  const void* ptr1,
                                                  const void* ptr2,
                                                  const void* ptr3) {
  U4x4 value;
  const uint64_t addr0 = static_cast<uint64_t>(__cvta_generic_to_global(ptr0));
  const uint64_t addr1 = static_cast<uint64_t>(__cvta_generic_to_global(ptr1));
  const uint64_t addr2 = static_cast<uint64_t>(__cvta_generic_to_global(ptr2));
  const uint64_t addr3 = static_cast<uint64_t>(__cvta_generic_to_global(ptr3));
  asm volatile(
      "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%16];\n"
      "ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%17];\n"
      "ld.global.cg.v4.u32 {%8, %9, %10, %11}, [%18];\n"
      "ld.global.cg.v4.u32 {%12, %13, %14, %15}, [%19];\n"
      : "=r"(value.v0.x), "=r"(value.v0.y), "=r"(value.v0.z), "=r"(value.v0.w),
        "=r"(value.v1.x), "=r"(value.v1.y), "=r"(value.v1.z), "=r"(value.v1.w),
        "=r"(value.v2.x), "=r"(value.v2.y), "=r"(value.v2.z), "=r"(value.v2.w),
        "=r"(value.v3.x), "=r"(value.v3.y), "=r"(value.v3.z), "=r"(value.v3.w)
      : "l"(addr0), "l"(addr1), "l"(addr2), "l"(addr3)
      : "memory");
  return value;
}

__device__ __forceinline__ uint32_t fold_u4(uint4 value) {
  uint32_t sink = value.x;
  sink ^= value.y;
  sink += value.z;
  sink ^= value.w;
  return sink;
}

__global__ void global_load_bw_kernel(const uint8_t* data, uint64_t* samples,
                                      uint32_t* thread_sinks, int iters,
                                      int warmup, int tiles, int repeat,
                                      uint64_t tile_bytes, int locality_i) {
  const int tid = threadIdx.x;
  const LocalityMode locality = static_cast<LocalityMode>(locality_i);
  const uint64_t block_stride = static_cast<uint64_t>(blockDim.x) * kLoadBytes;
  uint32_t sink = static_cast<uint32_t>(blockIdx.x * blockDim.x + tid);
  const int total_iters = warmup + iters;

  for (int i = 0; i < total_iters; ++i) {
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

      const uint8_t* base = data + tile * tile_bytes;
      for (uint64_t offset = static_cast<uint64_t>(tid) * kLoadBytes;
           offset < tile_bytes; offset += block_stride * kLoadUnroll) {
        const uint64_t offset0 = offset;
        const uint64_t offset1 = offset + block_stride;
        const uint64_t offset2 = offset + block_stride * 2;
        const uint64_t offset3 = offset + block_stride * 3;

        uint32_t local = 0;
        if (offset3 < tile_bytes) {
          const U4x4 values = ld_global_cg_u4x4(base + offset0, base + offset1,
                                                base + offset2, base + offset3);
          local ^= fold_u4(values.v0);
          local += fold_u4(values.v1);
          local ^= fold_u4(values.v2);
          local += fold_u4(values.v3);
        } else {
          if (offset0 < tile_bytes) {
            local ^= fold_u4(ld_global_cg_u4(base + offset0));
          }
          if (offset1 < tile_bytes) {
            local += fold_u4(ld_global_cg_u4(base + offset1));
          }
          if (offset2 < tile_bytes) {
            local ^= fold_u4(ld_global_cg_u4(base + offset2));
          }
          if (offset3 < tile_bytes) {
            local += fold_u4(ld_global_cg_u4(base + offset3));
          }
        }
        sink ^= local;
      }
    }

    __syncthreads();
    if (tid == 0) {
      const uint64_t end = clock64_now();
      if (i >= warmup) {
        samples[blockIdx.x * iters + (i - warmup)] = end - start;
      }
    }
  }

  thread_sinks[blockIdx.x * blockDim.x + tid] = sink;
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
      "Usage: %s [--device N] [--blocks N] [--threads N]\n"
      "          [--iters N] [--warmup N]\n"
      "          [--tiles N] [--repeat N] [--tile-bytes N]\n"
      "          [--hot|--block-hot|--stride] [--event] [--csv path]\n",
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

bool parse_size_arg(const char* arg, const char* name, size_t* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(arg + n + 1, &end, 0);
    if (end == arg + n + 1 || *end != '\0') {
      std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
      std::exit(1);
    }
    *value = static_cast<size_t>(parsed);
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
               parse_int_arg(arg, "--threads", &opts.threads) ||
               parse_int_arg(arg, "--iters", &opts.iters) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--tiles", &opts.tiles) ||
               parse_int_arg(arg, "--repeat", &opts.repeat)) {
      continue;
    } else if (parse_size_arg(arg, "--tile-bytes", &opts.tile_bytes)) {
      continue;
    } else if (std::strcmp(arg, "--hot") == 0) {
      opts.locality = kLocalityHot;
    } else if (std::strcmp(arg, "--block-hot") == 0) {
      opts.locality = kLocalityBlockHot;
    } else if (std::strcmp(arg, "--stride") == 0) {
      opts.locality = kLocalityStride;
    } else if (std::strcmp(arg, "--event") == 0) {
      opts.event_timing = true;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.blocks <= 0 || opts.threads <= 0 || opts.iters <= 0 ||
      opts.warmup < 0 || opts.repeat <= 0 || opts.tiles <= 0 ||
      opts.tile_bytes == 0) {
    std::fprintf(stderr,
                 "blocks, threads, iters, tiles, repeat, and tile-bytes must "
                 "be positive; warmup must be non-negative\n");
    std::exit(1);
  }
  if (opts.threads % 32 != 0 || opts.threads > 1024) {
    std::fprintf(stderr,
                 "threads must be a positive multiple of 32 and <= 1024\n");
    std::exit(1);
  }
  if (opts.tile_bytes % (static_cast<size_t>(opts.threads) * kLoadBytes) != 0) {
    std::fprintf(stderr,
                 "tile-bytes must be a multiple of %zu so measured bytes are "
                 "exact\n",
                 static_cast<size_t>(opts.threads) * kLoadBytes);
    std::exit(1);
  }
  if (opts.locality != kLocalityStride &&
      opts.tiles < opts.blocks * opts.repeat) {
    std::fprintf(stderr,
                 "hot/block-hot modes need tiles >= blocks * repeat to avoid "
                 "aliasing\n");
    std::exit(1);
  }
  return opts;
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

  const size_t data_bytes = opts.tile_bytes * static_cast<size_t>(opts.tiles);
  const size_t sample_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.iters);
  const size_t sink_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.threads);
  const size_t bytes_per_block_iter =
      opts.tile_bytes * static_cast<size_t>(opts.repeat);
  const size_t measured_bytes = bytes_per_block_iter *
                                static_cast<size_t>(opts.blocks) *
                                static_cast<size_t>(opts.iters);
  const size_t kernel_bytes = bytes_per_block_iter *
                              static_cast<size_t>(opts.blocks) *
                              static_cast<size_t>(opts.iters + opts.warmup);

  uint8_t* d_data = nullptr;
  uint64_t* d_samples = nullptr;
  uint32_t* d_sinks = nullptr;

  CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sinks, sink_count * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(d_data, 0x5a, data_bytes));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sinks, 0, sink_count * sizeof(uint32_t)));

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "case=global-load-cg locality=%s blocks=%d threads=%d iters=%d "
      "warmup=%d tiles=%d repeat=%d tile_bytes=%zu data_bytes=%zu "
      "bytes_per_block_iter=%zu measured_bytes=%zu kernel_bytes=%zu\n",
      locality_name(opts.locality), opts.blocks, opts.threads, opts.iters,
      opts.warmup, opts.tiles, opts.repeat, opts.tile_bytes, data_bytes,
      bytes_per_block_iter, measured_bytes, kernel_bytes);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (opts.event_timing) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
  }
  global_load_bw_kernel<<<opts.blocks, opts.threads>>>(
      d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.tiles,
      opts.repeat, static_cast<uint64_t>(opts.tile_bytes),
      static_cast<int>(opts.locality));
  CUDA_CHECK(cudaGetLastError());
  if (opts.event_timing) {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms > 0.0f) {
      const double measured_gbs =
          static_cast<double>(measured_bytes) / (elapsed_ms * 1.0e6);
      const double kernel_gbs =
          static_cast<double>(kernel_bytes) / (elapsed_ms * 1.0e6);
      std::printf("event_elapsed_ms=%.6f measured_GBps=%.3f kernel_GBps=%.3f\n",
                  elapsed_ms, measured_gbs, kernel_gbs);
    }
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(sink_count);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        sample_count * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks, sink_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  const uint64_t sink_sum =
      std::accumulate(sinks.begin(), sinks.end(), uint64_t{0});
  print_stats(samples);
  std::printf("sink=%llu\n", static_cast<unsigned long long>(sink_sum));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.blocks, opts.iters);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  if (opts.event_timing) {
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
