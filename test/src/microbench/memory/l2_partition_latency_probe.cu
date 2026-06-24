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

struct Options {
  int device = 0;
  int blocks = 0;
  int threads = 32;
  int repeats = 256;
  int warmup = 64;
  int bit_first = 7;
  int bit_last = 30;
  int dense_offsets = 256;
  size_t dense_stride = 128;
  size_t base_offset = 0;
  size_t data_bytes = 2ULL * 1024 * 1024 * 1024;
  size_t smem_bytes = 0;
  bool powers_only = false;
  std::string csv_path;
};

struct Sample {
  uint32_t block;
  uint32_t smid;
  uint32_t offset_idx;
  uint64_t offset;
  uint64_t cycles;
  uint64_t sink;
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

__device__ __forceinline__ uint32_t smid_now() {
  uint32_t value = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void* ptr) {
  uint64_t out = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(out) : "l"(ptr));
  return out;
}

__device__ __forceinline__ uint64_t ld_global_cg_u64(uint64_t addr) {
  uint64_t value = 0;
  asm volatile("ld.global.cg.u64 %0, [%1];\n"
               : "=l"(value)
               : "l"(addr)
               : "memory");
  return value;
}

__global__ void init_self_pointer_kernel(uint8_t* data,
                                         const uint64_t* offsets,
                                         int num_offsets) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_offsets) return;
  const uint64_t addr = gmem_addr_u64(data + offsets[idx]);
  *reinterpret_cast<uint64_t*>(data + offsets[idx]) = addr;
}

__global__ void l2_partition_latency_kernel(const uint8_t* data,
                                            const uint64_t* offsets,
                                            int num_offsets, int repeats,
                                            int warmup, Sample* samples) {
  const int tid = threadIdx.x;
  const uint32_t smid = smid_now();

  for (int logical = 0; logical < num_offsets; ++logical) {
    const int offset_idx = (logical + blockIdx.x) % num_offsets;
    const uint64_t offset = offsets[offset_idx];
    uint64_t ptr = gmem_addr_u64(data + offset);
    uint64_t sink = ptr;

    if (tid == 0) {
      for (int i = 0; i < warmup; ++i) {
        ptr = ld_global_cg_u64(ptr);
      }
      const uint64_t start = clock64_now();
      for (int i = 0; i < repeats; ++i) {
        ptr = ld_global_cg_u64(ptr);
      }
      const uint64_t end = clock64_now();
      sink ^= ptr;

      const size_t out_idx =
          static_cast<size_t>(blockIdx.x) * static_cast<size_t>(num_offsets) +
          static_cast<size_t>(offset_idx);
      samples[out_idx] = {static_cast<uint32_t>(blockIdx.x), smid,
                          static_cast<uint32_t>(offset_idx), offset,
                          end - start, sink};
    }
  }
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device=N] [--blocks=N] [--threads=N]\n"
      "          [--repeats=N] [--warmup=N]\n"
      "          [--bit-first=N] [--bit-last=N] [--dense-offsets=N]\n"
      "          [--dense-stride=N] [--base-offset=N] [--data-bytes=N]\n"
      "          [--smem-bytes=N] [--powers-only] [--csv=path]\n\n"
      "The kernel records a latency vector indexed by (smid, address offset).\n"
      "Offsets include 0, powers of two in [bit-first, bit-last], and a dense\n"
      "stream unless --powers-only is used.\n",
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
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(arg + n + 1, &end, 0);
  if (end == arg + n + 1) {
    std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
    std::exit(1);
  }
  if (*end == 'k' || *end == 'K') {
    parsed *= 1024ULL;
    ++end;
  } else if (*end == 'm' || *end == 'M') {
    parsed *= 1024ULL * 1024ULL;
    ++end;
  } else if (*end == 'g' || *end == 'G') {
    parsed *= 1024ULL * 1024ULL * 1024ULL;
    ++end;
  }
  if (*end != '\0') {
    std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
    std::exit(1);
  }
  *value = static_cast<size_t>(parsed);
  return true;
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
               parse_int_arg(arg, "--repeats", &opts.repeats) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--bit-first", &opts.bit_first) ||
               parse_int_arg(arg, "--bit-last", &opts.bit_last) ||
               parse_int_arg(arg, "--dense-offsets", &opts.dense_offsets)) {
      continue;
    } else if (parse_size_arg(arg, "--dense-stride", &opts.dense_stride) ||
               parse_size_arg(arg, "--base-offset", &opts.base_offset) ||
               parse_size_arg(arg, "--data-bytes", &opts.data_bytes) ||
               parse_size_arg(arg, "--smem-bytes", &opts.smem_bytes)) {
      continue;
    } else if (std::strcmp(arg, "--powers-only") == 0) {
      opts.powers_only = true;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (opts.threads <= 0 || opts.threads > 1024 || opts.threads % 32 != 0 ||
      opts.repeats <= 0 || opts.warmup < 0 || opts.bit_first < 3 ||
      opts.bit_last < opts.bit_first || opts.bit_last >= 62 ||
      opts.dense_offsets < 0 || opts.dense_stride < 8 ||
      opts.data_bytes < 4096) {
    std::fprintf(stderr, "Invalid options\n");
    std::exit(1);
  }
  opts.base_offset -= opts.base_offset % 8;
  opts.dense_stride -= opts.dense_stride % 8;
  opts.smem_bytes -= opts.smem_bytes % 256;
  return opts;
}

std::vector<uint64_t> build_offsets(const Options& opts) {
  std::vector<uint64_t> offsets;
  offsets.push_back(opts.base_offset);
  for (int bit = opts.bit_first; bit <= opts.bit_last; ++bit) {
    offsets.push_back(opts.base_offset + (1ULL << bit));
  }
  if (!opts.powers_only) {
    for (int i = 0; i < opts.dense_offsets; ++i) {
      offsets.push_back(opts.base_offset +
                        static_cast<uint64_t>(i + 1) * opts.dense_stride);
    }
  }
  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
  for (uint64_t offset : offsets) {
    if (offset + sizeof(uint64_t) > opts.data_bytes) {
      std::fprintf(stderr,
                   "Offset %llu exceeds data allocation %zu; increase "
                   "--data-bytes or lower bit range\n",
                   static_cast<unsigned long long>(offset), opts.data_bytes);
      std::exit(1);
    }
  }
  return offsets;
}

void write_csv(const std::string& path, const std::vector<Sample>& samples,
               int repeats) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "block,smid,offset_idx,offset,cycles,repeats,cycles_per_load,sink\n";
  for (const Sample& s : samples) {
    out << s.block << "," << s.smid << "," << s.offset_idx << ","
        << s.offset << "," << s.cycles << "," << repeats << ","
        << (static_cast<double>(s.cycles) / static_cast<double>(repeats))
        << "," << s.sink << "\n";
  }
}

void print_summary(const std::vector<Sample>& samples, int repeats,
                   int num_offsets) {
  std::vector<uint32_t> smids;
  for (const Sample& s : samples) smids.push_back(s.smid);
  std::sort(smids.begin(), smids.end());
  smids.erase(std::unique(smids.begin(), smids.end()), smids.end());

  double total = 0.0;
  uint64_t min_cycles = UINT64_MAX;
  uint64_t max_cycles = 0;
  for (const Sample& s : samples) {
    total += static_cast<double>(s.cycles);
    min_cycles = std::min(min_cycles, s.cycles);
    max_cycles = std::max(max_cycles, s.cycles);
  }
  const double mean = total / static_cast<double>(samples.size());
  std::printf(
      "samples=%zu offsets=%d unique_smids=%zu repeats=%d "
      "cycles_mean=%.2f cycles_per_load_mean=%.3f min=%llu max=%llu\n",
      samples.size(), num_offsets, smids.size(), repeats, mean,
      mean / static_cast<double>(repeats),
      static_cast<unsigned long long>(min_cycles),
      static_cast<unsigned long long>(max_cycles));
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, opts.device));
  const int blocks = opts.blocks > 0 ? opts.blocks : prop.multiProcessorCount;
  const std::vector<uint64_t> offsets = build_offsets(opts);

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "case=l2-partition-latency blocks=%d threads=%d repeats=%d warmup=%d "
      "offsets=%zu bit_first=%d bit_last=%d dense_offsets=%d "
      "dense_stride=%zu base_offset=%zu data_bytes=%zu smem_bytes=%zu\n",
      blocks, opts.threads, opts.repeats, opts.warmup, offsets.size(),
      opts.bit_first, opts.bit_last, opts.dense_offsets, opts.dense_stride,
      opts.base_offset, opts.data_bytes, opts.smem_bytes);

  uint8_t* d_data = nullptr;
  uint64_t* d_offsets = nullptr;
  Sample* d_samples = nullptr;
  const size_t num_samples =
      static_cast<size_t>(blocks) * static_cast<size_t>(offsets.size());
  CUDA_CHECK(cudaMalloc(&d_data, opts.data_bytes));
  CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_samples, num_samples * sizeof(Sample)));
  CUDA_CHECK(cudaMemset(d_data, 0, opts.data_bytes));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                        offsets.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  init_self_pointer_kernel<<<(offsets.size() + 255) / 256, 256>>>(
      d_data, d_offsets, static_cast<int>(offsets.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (opts.smem_bytes > 0) {
    CUDA_CHECK(cudaFuncSetAttribute(
        l2_partition_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(opts.smem_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(l2_partition_latency_kernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    100));
  }

  l2_partition_latency_kernel<<<blocks, opts.threads, opts.smem_bytes>>>(
      d_data, d_offsets, static_cast<int>(offsets.size()), opts.repeats,
      opts.warmup, d_samples);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Sample> samples(num_samples);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples, num_samples * sizeof(Sample),
                        cudaMemcpyDeviceToHost));
  print_summary(samples, opts.repeats, static_cast<int>(offsets.size()));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.repeats);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
