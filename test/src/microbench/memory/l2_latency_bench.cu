#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

enum class CacheOp {
  kCg,
  kCa,
  kCv,
};

struct Options {
  int device = 0;
  size_t bytes = 64 * 1024;
  int steps = 4096;
  int samples = 2048;
  int warmup_steps = 65536;
  uint32_t seed = 1;
  CacheOp cache_op = CacheOp::kCg;
  std::string case_name = "l2-hit";
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

template <CacheOp Op>
__device__ __forceinline__ uint32_t ld_global_u32(const uint32_t* ptr) {
  uint32_t value = 0;
  const uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_global(ptr));
  if constexpr (Op == CacheOp::kCg) {
    asm volatile("ld.global.cg.u32 %0, [%1];\n"
                 : "=r"(value)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (Op == CacheOp::kCa) {
    asm volatile("ld.global.ca.u32 %0, [%1];\n"
                 : "=r"(value)
                 : "l"(addr)
                 : "memory");
  } else {
    asm volatile("ld.global.cv.u32 %0, [%1];\n"
                 : "=r"(value)
                 : "l"(addr)
                 : "memory");
  }
  return value;
}

template <CacheOp Op>
__global__ void pointer_chase_latency_kernel(const uint32_t* next,
                                             uint64_t* samples,
                                             uint32_t* sink, int sample_count,
                                             int steps, int warmup_steps,
                                             uint32_t start_idx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint32_t idx = start_idx;
  for (int i = 0; i < warmup_steps; ++i) {
    idx = ld_global_u32<Op>(next + idx);
  }

  for (int sample = 0; sample < sample_count; ++sample) {
    const uint64_t start = clock64_now();
#pragma unroll 1
    for (int i = 0; i < steps; ++i) {
      idx = ld_global_u32<Op>(next + idx);
    }
    const uint64_t end = clock64_now();
    samples[sample] = end - start;
  }

  sink[0] = idx;
}

__global__ void empty_chain_latency_kernel(uint64_t* samples, uint32_t* sink,
                                           int sample_count, int steps,
                                           uint32_t start_idx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint32_t idx = start_idx;
  for (int sample = 0; sample < sample_count; ++sample) {
    const uint64_t start = clock64_now();
#pragma unroll 1
    for (int i = 0; i < steps; ++i) {
      idx = idx * 1664525u + 1013904223u;
    }
    const uint64_t end = clock64_now();
    samples[sample] = end - start;
  }

  sink[0] = idx;
}

const char* cache_op_name(CacheOp op) {
  switch (op) {
    case CacheOp::kCg:
      return "cg";
    case CacheOp::kCa:
      return "ca";
    case CacheOp::kCv:
      return "cv";
  }
  return "unknown";
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device=N] [--case=name] [--bytes=N]\n"
      "          [--steps=N] [--samples=N] [--warmup-steps=N]\n"
      "          [--cache-op=cg|ca|cv] [--seed=N] [--csv=path]\n\n"
      "Recommended:\n"
      "  L2 hit:  --case=l2-hit --bytes=65536 --warmup-steps=65536 --cache-op=cg\n"
      "  miss:    --case=miss --bytes=268435456 --warmup-steps=0 --cache-op=cg\n",
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

bool parse_u32_arg(const char* arg, const char* name, uint32_t* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = static_cast<uint32_t>(std::strtoul(arg + n + 1, nullptr, 0));
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
               parse_int_arg(arg, "--steps", &opts.steps) ||
               parse_int_arg(arg, "--samples", &opts.samples) ||
               parse_int_arg(arg, "--warmup-steps", &opts.warmup_steps)) {
      continue;
    } else if (parse_u32_arg(arg, "--seed", &opts.seed)) {
      continue;
    } else if (parse_size_arg(arg, "--bytes", &opts.bytes)) {
      continue;
    } else if (std::strncmp(arg, "--case=", 7) == 0) {
      opts.case_name = arg + 7;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else if (std::strcmp(arg, "--cache-op=cg") == 0) {
      opts.cache_op = CacheOp::kCg;
    } else if (std::strcmp(arg, "--cache-op=ca") == 0) {
      opts.cache_op = CacheOp::kCa;
    } else if (std::strcmp(arg, "--cache-op=cv") == 0) {
      opts.cache_op = CacheOp::kCv;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.steps <= 0 || opts.samples <= 0 || opts.warmup_steps < 0 ||
      opts.bytes < sizeof(uint32_t) * 2) {
    std::fprintf(stderr,
                 "steps/samples must be positive, warmup non-negative, and "
                 "bytes must hold at least two uint32 nodes\n");
    std::exit(1);
  }
  opts.bytes -= opts.bytes % sizeof(uint32_t);
  return opts;
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double q) {
  if (sorted.empty()) return 0;
  const double pos = q * static_cast<double>(sorted.size() - 1);
  return sorted[static_cast<size_t>(std::llround(pos))];
}

struct Summary {
  double mean_cycles = 0.0;
  double stdev_cycles = 0.0;
  uint64_t min_cycles = 0;
  uint64_t p50_cycles = 0;
  uint64_t p90_cycles = 0;
  uint64_t p95_cycles = 0;
  uint64_t p99_cycles = 0;
  uint64_t max_cycles = 0;
};

Summary summarize(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  Summary s;
  const double sum =
      std::accumulate(sorted.begin(), sorted.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  s.mean_cycles = sum / static_cast<double>(sorted.size());
  double sq = 0.0;
  for (uint64_t v : sorted) {
    const double d = static_cast<double>(v) - s.mean_cycles;
    sq += d * d;
  }
  s.stdev_cycles = std::sqrt(sq / static_cast<double>(sorted.size()));
  s.min_cycles = sorted.front();
  s.p50_cycles = percentile(sorted, 0.50);
  s.p90_cycles = percentile(sorted, 0.90);
  s.p95_cycles = percentile(sorted, 0.95);
  s.p99_cycles = percentile(sorted, 0.99);
  s.max_cycles = sorted.back();
  return s;
}

void print_summary(const char* label, const Summary& s, int steps) {
  std::printf(
      "%s cycles: mean=%.2f stdev=%.2f min=%llu p50=%llu p90=%llu p95=%llu "
      "p99=%llu max=%llu\n",
      label, s.mean_cycles, s.stdev_cycles,
      static_cast<unsigned long long>(s.min_cycles),
      static_cast<unsigned long long>(s.p50_cycles),
      static_cast<unsigned long long>(s.p90_cycles),
      static_cast<unsigned long long>(s.p95_cycles),
      static_cast<unsigned long long>(s.p99_cycles),
      static_cast<unsigned long long>(s.max_cycles));
  std::printf(
      "%s cycles_per_load: mean=%.3f p50=%.3f p90=%.3f p95=%.3f p99=%.3f\n",
      label, s.mean_cycles / steps,
      static_cast<double>(s.p50_cycles) / steps,
      static_cast<double>(s.p90_cycles) / steps,
      static_cast<double>(s.p95_cycles) / steps,
      static_cast<double>(s.p99_cycles) / steps);
}

std::vector<uint32_t> make_pointer_chase(size_t nodes, uint32_t seed) {
  std::vector<uint32_t> order(nodes);
  for (size_t i = 0; i < nodes; ++i) order[i] = static_cast<uint32_t>(i);
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);

  std::vector<uint32_t> next(nodes);
  for (size_t i = 0; i < nodes; ++i) {
    next[order[i]] = order[(i + 1) % nodes];
  }
  return next;
}

void write_csv(const std::string& path, const Options& opts, size_t nodes,
               const std::vector<uint64_t>& samples,
               const std::vector<uint64_t>& empty_samples) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "case,cache_op,nodes,bytes,steps,warmup_steps,sample,kind,cycles,"
         "cycles_per_load\n";
  for (int i = 0; i < opts.samples; ++i) {
    out << opts.case_name << "," << cache_op_name(opts.cache_op) << ","
        << nodes << "," << opts.bytes << "," << opts.steps << ","
        << opts.warmup_steps << "," << i << ",load," << samples[i] << ","
        << static_cast<double>(samples[i]) / opts.steps << "\n";
    out << opts.case_name << "," << cache_op_name(opts.cache_op) << ","
        << nodes << "," << opts.bytes << "," << opts.steps << ","
        << opts.warmup_steps << "," << i << ",empty," << empty_samples[i]
        << "," << static_cast<double>(empty_samples[i]) / opts.steps << "\n";
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

  const size_t nodes = opts.bytes / sizeof(uint32_t);
  if (nodes > static_cast<size_t>(UINT32_MAX)) {
    std::fprintf(stderr, "nodes exceed uint32 index range\n");
    return 1;
  }

  std::printf("building_pointer_chase nodes=%zu bytes=%zu seed=%u\n", nodes,
              opts.bytes, opts.seed);
  std::vector<uint32_t> h_next = make_pointer_chase(nodes, opts.seed);

  uint32_t* d_next = nullptr;
  uint64_t* d_samples = nullptr;
  uint64_t* d_empty_samples = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_next, opts.bytes));
  CUDA_CHECK(cudaMalloc(&d_samples, opts.samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_empty_samples, opts.samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_next, h_next.data(), opts.bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_samples, 0, opts.samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_empty_samples, 0, opts.samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));

  std::printf(
      "device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n", opts.device,
      prop.name, prop.major, prop.minor, prop.multiProcessorCount,
      clock_rate_khz);
  std::printf(
      "case=%s cache_op=%s bytes=%zu nodes=%zu steps=%d samples=%d "
      "warmup_steps=%d\n",
      opts.case_name.c_str(), cache_op_name(opts.cache_op), opts.bytes, nodes,
      opts.steps, opts.samples, opts.warmup_steps);

  if (opts.cache_op == CacheOp::kCg) {
    pointer_chase_latency_kernel<CacheOp::kCg><<<1, 1>>>(
        d_next, d_samples, d_sink, opts.samples, opts.steps,
        opts.warmup_steps, 0);
  } else if (opts.cache_op == CacheOp::kCa) {
    pointer_chase_latency_kernel<CacheOp::kCa><<<1, 1>>>(
        d_next, d_samples, d_sink, opts.samples, opts.steps,
        opts.warmup_steps, 0);
  } else {
    pointer_chase_latency_kernel<CacheOp::kCv><<<1, 1>>>(
        d_next, d_samples, d_sink, opts.samples, opts.steps,
        opts.warmup_steps, 0);
  }
  CUDA_CHECK(cudaGetLastError());
  empty_chain_latency_kernel<<<1, 1>>>(d_empty_samples, d_sink, opts.samples,
                                       opts.steps, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint64_t> samples(opts.samples);
  std::vector<uint64_t> empty_samples(opts.samples);
  uint32_t sink = 0;
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        opts.samples * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(empty_samples.data(), d_empty_samples,
                        opts.samples * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sink, d_sink, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  const Summary load_summary = summarize(samples);
  const Summary empty_summary = summarize(empty_samples);
  print_summary("load", load_summary, opts.steps);
  print_summary("empty", empty_summary, opts.steps);
  std::printf("net_cycles_per_load mean=%.3f p50=%.3f\n",
              (load_summary.mean_cycles - empty_summary.mean_cycles) /
                  opts.steps,
              static_cast<double>(load_summary.p50_cycles -
                                  empty_summary.p50_cycles) /
                  opts.steps);
  std::printf("sink=%u\n", sink);

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, opts, nodes, samples, empty_samples);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_next));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_empty_samples));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
