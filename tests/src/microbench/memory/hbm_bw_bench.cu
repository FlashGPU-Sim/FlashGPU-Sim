#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

void check_cuda(cudaError_t status, const char *where) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", where,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

struct Options {
  std::size_t mib = 1024;
  int blocks_per_sm = 4;
  int threads = 256;
  int chains = 8;
  int passes = 8;
  int warmup = 1;
  int iterations = 5;
};

void print_usage(const char *program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --mib N             Streaming working set in MiB (default: 1024)\n"
      "  --blocks-per-sm N   Grid size multiplier (default: 4)\n"
      "  --threads N         Threads per block, multiple of 32 (default: 256)\n"
      "  --chains N          Independent loads per loop: 1, 2, 4, or 8 "
      "(default: 8)\n"
      "  --passes N          Full working-set traversals per launch "
      "(default: 8)\n"
      "  --warmup N          Untimed launches (default: 1)\n"
      "  --iterations N      Timed launches (default: 5)\n",
      program);
}

std::uint64_t parse_integer(const char *name, const char *text,
                            bool allow_zero) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || *text == '\0' || *end != '\0' ||
      (!allow_zero && value == 0)) {
    std::fprintf(stderr, "Invalid value for %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<std::uint64_t>(value);
}

int parse_int(const char *name, const char *text, bool allow_zero) {
  const std::uint64_t value = parse_integer(name, text, allow_zero);
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr, "Value for %s is too large: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value for %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }

    const char *name = argv[i];
    const char *value = argv[++i];
    if (std::strcmp(name, "--mib") == 0) {
      const std::uint64_t mib = parse_integer(name, value, false);
      if (mib > std::numeric_limits<std::size_t>::max() / kMiB) {
        std::fprintf(stderr, "Working set is too large: %s MiB\n", value);
        std::exit(EXIT_FAILURE);
      }
      options.mib = static_cast<std::size_t>(mib);
    } else if (std::strcmp(name, "--blocks-per-sm") == 0) {
      options.blocks_per_sm = parse_int(name, value, false);
    } else if (std::strcmp(name, "--threads") == 0) {
      options.threads = parse_int(name, value, false);
    } else if (std::strcmp(name, "--chains") == 0) {
      options.chains = parse_int(name, value, false);
    } else if (std::strcmp(name, "--passes") == 0) {
      options.passes = parse_int(name, value, false);
    } else if (std::strcmp(name, "--warmup") == 0) {
      options.warmup = parse_int(name, value, true);
    } else if (std::strcmp(name, "--iterations") == 0) {
      options.iterations = parse_int(name, value, false);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", name);
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.threads < 32 || options.threads > 1024 ||
      options.threads % 32 != 0) {
    std::fprintf(stderr, "--threads must be a multiple of 32 in [32, 1024]\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.chains != 1 && options.chains != 2 && options.chains != 4 &&
      options.chains != 8) {
    std::fprintf(stderr, "--chains must be 1, 2, 4, or 8\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ uint4 load_global_cg(const uint4 *pointer) {
  uint4 value;
  const auto address = reinterpret_cast<unsigned long long>(pointer);
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
  return value;
}

template <int Chains>
__global__ void hbm_bw_kernel(const uint4 *__restrict__ source,
                              std::size_t vector_count,
                              std::uint64_t *__restrict__ sink, int passes) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t thread_count =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t tile_stride = thread_count * Chains;

  std::uint32_t accumulators[Chains] = {};

  for (int pass = 0; pass < passes; ++pass) {
    for (std::size_t base = thread; base < vector_count; base += tile_stride) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        const std::size_t index =
            base + static_cast<std::size_t>(chain) * thread_count;
        if (index < vector_count) {
          const uint4 value = load_global_cg(source + index);
          // cudaMemset initializes every byte to one.  Using all four words
          // keeps the complete 16-byte load live, while the low bit makes
          // each executed vector load contribute exactly one to the checksum.
          const std::uint32_t contribution =
              (value.x | value.y | value.z | value.w) & 1u;
          accumulators[chain] += contribution;
        }
      }
    }
  }

  std::uint64_t checksum = 0;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
    checksum += accumulators[chain];
  }
  sink[thread] = checksum;
}

template <int Chains>
void launch_kernel(dim3 grid, dim3 block, const uint4 *source,
                   std::size_t vector_count, std::uint64_t *sink, int passes) {
  hbm_bw_kernel<Chains><<<grid, block>>>(source, vector_count, sink, passes);
}

void launch_selected(int chains, dim3 grid, dim3 block, const uint4 *source,
                     std::size_t vector_count, std::uint64_t *sink,
                     int passes) {
  switch (chains) {
    case 1:
      launch_kernel<1>(grid, block, source, vector_count, sink, passes);
      break;
    case 2:
      launch_kernel<2>(grid, block, source, vector_count, sink, passes);
      break;
    case 4:
      launch_kernel<4>(grid, block, source, vector_count, sink, passes);
      break;
    case 8:
      launch_kernel<8>(grid, block, source, vector_count, sink, passes);
      break;
    default:
      std::abort();
  }
}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

}  // namespace

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;
  check_cuda(cudaDeviceGetAttribute(&memory_clock_khz,
                                    cudaDevAttrMemoryClockRate, device),
             "query memory clock");
  check_cuda(cudaDeviceGetAttribute(&memory_bus_width_bits,
                                    cudaDevAttrGlobalMemoryBusWidth, device),
             "query memory bus width");

  const std::size_t workset_bytes = options.mib * kMiB;
  const std::size_t vector_count = workset_bytes / sizeof(uint4);
  const int blocks = properties.multiProcessorCount * options.blocks_per_sm;
  const std::size_t sink_count =
      static_cast<std::size_t>(blocks) * options.threads;

  if (blocks <= 0 || static_cast<std::uint64_t>(vector_count) >
                         std::numeric_limits<std::uint64_t>::max() /
                             static_cast<std::uint64_t>(options.passes)) {
    std::fprintf(stderr, "Requested launch or byte count is too large\n");
    return EXIT_FAILURE;
  }

  const std::size_t max_vectors_per_thread =
      (vector_count + sink_count - 1) / sink_count;
  if (static_cast<std::uint64_t>(max_vectors_per_thread) * options.passes >
      std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(
        stderr,
        "Per-thread checksum counter would overflow; reduce --passes\n");
    return EXIT_FAILURE;
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const std::size_t allocation_bytes =
      workset_bytes + sink_count * sizeof(std::uint64_t);
  if (allocation_bytes > free_bytes) {
    std::fprintf(stderr,
                 "Insufficient device memory: need=%zu free=%zu bytes\n",
                 allocation_bytes, free_bytes);
    return EXIT_FAILURE;
  }

  uint4 *device_source = nullptr;
  std::uint64_t *device_sink = nullptr;
  check_cuda(cudaMalloc(&device_source, workset_bytes), "allocate source");
  check_cuda(cudaMalloc(&device_sink, sink_count * sizeof(std::uint64_t)),
             "allocate sink");
  check_cuda(cudaMemset(device_source, 1, workset_bytes), "initialize source");

  const dim3 grid(blocks);
  const dim3 block(options.threads);

  std::printf(
      "DEVICE name=%s compute_capability=%d.%d sm_count=%d l2_bytes=%d "
      "memory_total_bytes=%zu memory_free_bytes=%zu memory_clock_khz=%d "
      "memory_bus_width_bits=%d\n",
      properties.name, properties.major, properties.minor,
      properties.multiProcessorCount, properties.l2CacheSize, total_bytes,
      free_bytes, memory_clock_khz, memory_bus_width_bits);
  std::printf(
      "CONFIG workset_bytes=%zu blocks=%d blocks_per_sm=%d threads=%d "
      "chains=%d passes=%d warmup=%d iterations=%d\n",
      workset_bytes, blocks, options.blocks_per_sm, options.threads,
      options.chains, options.passes, options.warmup, options.iterations);

  for (int i = 0; i < options.warmup; ++i) {
    launch_selected(options.chains, grid, block, device_source, vector_count,
                    device_sink, options.passes);
    check_cuda(cudaGetLastError(), "warmup launch");
  }
  check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");

  const std::uint64_t logical_bytes =
      static_cast<std::uint64_t>(workset_bytes) * options.passes;
  std::vector<float> timings;
  timings.reserve(options.iterations);
  for (int i = 0; i < options.iterations; ++i) {
    check_cuda(cudaEventRecord(start), "record start event");
    launch_selected(options.chains, grid, block, device_source, vector_count,
                    device_sink, options.passes);
    check_cuda(cudaGetLastError(), "timed launch");
    check_cuda(cudaEventRecord(stop), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "synchronize stop event");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    timings.push_back(elapsed_ms);
    const double gbps = static_cast<double>(logical_bytes) /
                        (static_cast<double>(elapsed_ms) * 1.0e6);
    std::printf("SAMPLE iteration=%d time_ms=%.6f logical_GBps=%.3f\n", i,
                elapsed_ms, gbps);
  }

  std::vector<std::uint64_t> host_sink(sink_count);
  check_cuda(
      cudaMemcpy(host_sink.data(), device_sink,
                 sink_count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
      "copy checksum");
  const std::uint64_t checksum =
      std::accumulate(host_sink.begin(), host_sink.end(), std::uint64_t{0});
  const std::uint64_t expected =
      static_cast<std::uint64_t>(vector_count) * options.passes;
  if (checksum != expected) {
    std::fprintf(stderr, "Checksum mismatch: expected=%llu actual=%llu\n",
                 static_cast<unsigned long long>(expected),
                 static_cast<unsigned long long>(checksum));
    return EXIT_FAILURE;
  }

  const double mean_ms =
      std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
  const double median_ms = median(timings);
  const double best_ms = *std::min_element(timings.begin(), timings.end());
  const double mean_gbps = logical_bytes / (mean_ms * 1.0e6);
  const double median_gbps = logical_bytes / (median_ms * 1.0e6);
  const double best_gbps = logical_bytes / (best_ms * 1.0e6);
  const std::uint64_t coalesced_sectors = logical_bytes / 32;

  std::printf(
      "RESULT checksum=PASS checksum_value=%llu logical_read_bytes=%llu "
      "coalesced_32B_sectors=%llu mean_time_ms=%.6f median_time_ms=%.6f "
      "best_time_ms=%.6f mean_logical_GBps=%.3f median_logical_GBps=%.3f "
      "best_logical_GBps=%.3f\n",
      static_cast<unsigned long long>(checksum),
      static_cast<unsigned long long>(logical_bytes),
      static_cast<unsigned long long>(coalesced_sectors), mean_ms, median_ms,
      best_ms, mean_gbps, median_gbps, best_gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_source);
  cudaFree(device_sink);
  return EXIT_SUCCESS;
}
