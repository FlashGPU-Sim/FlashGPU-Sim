#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

enum class BenchMode : int {
  kMma = 0,
  kFma = 1,
  kEmpty = 2,
};

struct KernelResult {
  uint64_t t0;
  uint64_t t1;
  uint64_t t2;
  float sink;
};

constexpr int kMaxSaturationWarps = 16;
constexpr double kFlopPerMma16816 = 16.0 * 8.0 * 16.0 * 2.0;

struct SaturationResult {
  uint64_t t0_min;
  uint64_t t0_max;
  uint64_t t1_min;
  uint64_t t1_max;
  uint64_t t2_min;
  uint64_t t2_max;
  uint64_t warp_t0[kMaxSaturationWarps];
  uint64_t warp_t1[kMaxSaturationWarps];
  uint64_t warp_t2[kMaxSaturationWarps];
  float sink;
};

struct KernelAttrs {
  int regs;
  size_t local_bytes;
  size_t shared_bytes;
  int max_threads_per_block;
};

struct Stats {
  double min;
  double median;
  double mean;
  double p95;
  double max;
};

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint64_t read_clock64_after_mma_operands(
    const unsigned* a, const unsigned* b) {
  uint64_t value;
  asm volatile(
      "{\n"
      "  .reg .u32 dep;\n"
      "  xor.b32 dep, %1, %2;\n"
      "  xor.b32 dep, dep, %3;\n"
      "  xor.b32 dep, dep, %4;\n"
      "  xor.b32 dep, dep, %5;\n"
      "  xor.b32 dep, dep, %6;\n"
      "  mov.u64 %0, %%clock64;\n"
      "}\n"
      : "=l"(value)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1])
      : "memory");
  return value;
}

__device__ __forceinline__ void issue_order_anchor() {
  asm volatile(
      "{\n"
      "  bra.uni MMA_ACCEPT_ANCHOR;\n"
      "MMA_ACCEPT_ANCHOR:\n"
      "}\n" ::: "memory");
}

__device__ __forceinline__ unsigned make_half_pair_one() {
  unsigned value;
  asm volatile("mov.u32 %0, 0x3c003c00;" : "=r"(value)::"memory");
  return value;
}

__device__ __forceinline__ void emit_mma_group(float* d, const unsigned* a,
                                               const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]));
}

__device__ __forceinline__ void emit_fma_group(float* d, float x, float y) {
  asm volatile(
      "fma.rn.f32 %0, %4, %5, %0;\n"
      "fma.rn.f32 %1, %4, %5, %1;\n"
      "fma.rn.f32 %2, %4, %5, %2;\n"
      "fma.rn.f32 %3, %4, %5, %3;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "f"(x), "f"(y));
}

template <int N, BenchMode Mode>
__device__ __forceinline__ void init_accumulators(float d[][4], int lane) {
  if constexpr (Mode != BenchMode::kEmpty) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const float base =
          0.001953125f * static_cast<float>((i + 1) * ((lane & 7) + 1));
      d[i][0] = base + 0.125f;
      d[i][1] = base + 0.250f;
      d[i][2] = base + 0.375f;
      d[i][3] = base + 0.500f;
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      asm volatile("" ::"f"(d[i][0]), "f"(d[i][1]), "f"(d[i][2]),
                   "f"(d[i][3])
                   : "memory");
    }
  }
}

template <int N, BenchMode Mode>
__device__ __forceinline__ void emit_issue_sequence(float d[][4],
                                                    const unsigned* a,
                                                    const unsigned* b,
                                                    int lane) {
  const float x = 1.0f + static_cast<float>(lane & 3) * 0.125f;
  const float y = 2.0f;

#pragma unroll
  for (int i = 0; i < N; ++i) {
    if constexpr (Mode == BenchMode::kMma) {
      emit_mma_group(d[i], a, b);
    } else if constexpr (Mode == BenchMode::kFma) {
      emit_fma_group(d[i], x, y);
    } else {
      asm volatile("");
    }
  }
}

template <int N, BenchMode Mode>
__device__ __forceinline__ float consume_outputs(float d[][4], int lane) {
  if constexpr (Mode == BenchMode::kEmpty) {
    return static_cast<float>(lane);
  } else {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      sum += d[i][0];
      sum += d[i][1];
      sum += d[i][2];
      sum += d[i][3];
    }
    return sum;
  }
}

template <int N, BenchMode Mode>
__global__ __launch_bounds__(32, 1) void mma_accept_queue_kernel(
    KernelResult* out) {
  const int lane = threadIdx.x & 31;
  constexpr int kSlots = (N > 0) ? N : 1;
  float d[kSlots][4];

  unsigned a[4] = {
      0x3c003c00u ^ static_cast<unsigned>(lane),
      0x3c003c00u ^ static_cast<unsigned>(lane << 1),
      0x3c003c00u ^ static_cast<unsigned>(lane << 2),
      0x3c003c00u ^ static_cast<unsigned>(lane << 3),
  };
  unsigned b[2] = {
      0x3c003c00u ^ static_cast<unsigned>(lane << 4),
      0x3c003c00u ^ static_cast<unsigned>(lane << 5),
  };

  init_accumulators<N, Mode>(d, lane);
  asm volatile("bar.warp.sync -1;" ::: "memory");
  const uint64_t t0 = read_clock64();
  asm volatile("" ::: "memory");

  emit_issue_sequence<N, Mode>(d, a, b, lane);

  issue_order_anchor();
  asm volatile("" ::: "memory");
  const uint64_t t1 = read_clock64();

  const float sink = consume_outputs<N, Mode>(d, lane);
  asm volatile("" ::"f"(sink) : "memory");

  const uint64_t t2 = read_clock64();
  if (lane == 0) {
    out[0].t0 = t0;
    out[0].t1 = t1;
    out[0].t2 = t2;
    out[0].sink = sink;
  }
}

template <int N, int Warps>
__global__ __launch_bounds__(Warps * 32, 1) void mma_saturation_kernel(
    SaturationResult* out) {
  static_assert(Warps > 0 && Warps <= kMaxSaturationWarps,
                "unsupported warp count");
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int kSlots = (N > 0) ? N : 1;
  float d[kSlots][4];

  unsigned a[4] = {
      make_half_pair_one(),
      make_half_pair_one(),
      make_half_pair_one(),
      make_half_pair_one(),
  };
  unsigned b[2] = {
      make_half_pair_one(),
      make_half_pair_one(),
  };

  init_accumulators<N, BenchMode::kMma>(d, tid);

  __shared__ uint64_t s_t0[kMaxSaturationWarps];
  __shared__ uint64_t s_t1[kMaxSaturationWarps];
  __shared__ uint64_t s_t2[kMaxSaturationWarps];
  __shared__ float s_sink[kMaxSaturationWarps];

  __syncthreads();
  asm volatile("" ::: "memory");
  const uint64_t t0 = read_clock64_after_mma_operands(a, b);

  emit_issue_sequence<N, BenchMode::kMma>(d, a, b, lane);

  issue_order_anchor();
  asm volatile("" ::: "memory");
  const uint64_t t1 = read_clock64();

  const float sink = consume_outputs<N, BenchMode::kMma>(d, lane);
  asm volatile("" ::"f"(sink) : "memory");

  const uint64_t t2 = read_clock64();
  if (lane == 0) {
    s_t0[warp] = t0;
    s_t1[warp] = t1;
    s_t2[warp] = t2;
    s_sink[warp] = sink;
  }

  __syncthreads();
  if (tid == 0) {
    uint64_t t0_min = s_t0[0];
    uint64_t t0_max = s_t0[0];
    uint64_t t1_min = s_t1[0];
    uint64_t t1_max = s_t1[0];
    uint64_t t2_min = s_t2[0];
    uint64_t t2_max = s_t2[0];
    float sink_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Warps; ++i) {
      t0_min = min(t0_min, s_t0[i]);
      t0_max = max(t0_max, s_t0[i]);
      t1_min = min(t1_min, s_t1[i]);
      t1_max = max(t1_max, s_t1[i]);
      t2_min = min(t2_min, s_t2[i]);
      t2_max = max(t2_max, s_t2[i]);
      out[0].warp_t0[i] = s_t0[i];
      out[0].warp_t1[i] = s_t1[i];
      out[0].warp_t2[i] = s_t2[i];
      sink_sum += s_sink[i];
    }
    out[0].t0_min = t0_min;
    out[0].t0_max = t0_max;
    out[0].t1_min = t1_min;
    out[0].t1_max = t1_max;
    out[0].t2_min = t2_min;
    out[0].t2_max = t2_max;
    out[0].sink = sink_sum;
  }
}

const char* mode_name(BenchMode mode) {
  switch (mode) {
    case BenchMode::kMma:
      return "mma";
    case BenchMode::kFma:
      return "fma";
    case BenchMode::kEmpty:
      return "empty";
  }
  return "unknown";
}

BenchMode parse_mode(const std::string& text) {
  if (text == "mma") return BenchMode::kMma;
  if (text == "fma") return BenchMode::kFma;
  if (text == "empty") return BenchMode::kEmpty;
  std::fprintf(stderr, "unknown mode: %s\n", text.c_str());
  std::exit(2);
}

Stats summarize(std::vector<uint64_t> values) {
  if (values.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0};
  std::sort(values.begin(), values.end());
  double sum = 0.0;
  for (uint64_t value : values) sum += static_cast<double>(value);
  const size_t p95_idx =
      std::min(values.size() - 1, static_cast<size_t>(values.size() * 0.95));
  return {
      static_cast<double>(values.front()),
      static_cast<double>(values[values.size() / 2]),
      sum / static_cast<double>(values.size()),
      static_cast<double>(values[p95_idx]),
      static_cast<double>(values.back()),
  };
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::atoi(item.c_str()));
  }
  return values;
}

std::vector<BenchMode> parse_mode_list(const std::string& text) {
  std::vector<BenchMode> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(parse_mode(item));
  }
  return values;
}

bool has_token(const std::string& text, const char* token) {
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == token) return true;
  }
  return false;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

template <int N, BenchMode Mode>
bool query_attrs(KernelAttrs* attrs) {
  cudaFuncAttributes cuda_attrs;
  const cudaError_t err =
      cudaFuncGetAttributes(&cuda_attrs, mma_accept_queue_kernel<N, Mode>);
  if (err != cudaSuccess) return false;
  attrs->regs = cuda_attrs.numRegs;
  attrs->local_bytes = cuda_attrs.localSizeBytes;
  attrs->shared_bytes = cuda_attrs.sharedSizeBytes;
  attrs->max_threads_per_block = cuda_attrs.maxThreadsPerBlock;
  return true;
}

template <int N, BenchMode Mode>
bool launch_once(KernelResult* d_result, KernelAttrs* attrs) {
  if (!query_attrs<N, Mode>(attrs)) return false;
  mma_accept_queue_kernel<N, Mode><<<1, 32>>>(d_result);
  return cudaPeekAtLastError() == cudaSuccess;
}

template <int N>
bool launch_for_mode(BenchMode mode, KernelResult* d_result,
                     KernelAttrs* attrs) {
  switch (mode) {
    case BenchMode::kMma:
      return launch_once<N, BenchMode::kMma>(d_result, attrs);
    case BenchMode::kFma:
      return launch_once<N, BenchMode::kFma>(d_result, attrs);
    case BenchMode::kEmpty:
      return launch_once<N, BenchMode::kEmpty>(d_result, attrs);
  }
  return false;
}

bool launch_for_n(int n, BenchMode mode, KernelResult* d_result,
                  KernelAttrs* attrs) {
  switch (n) {
#define DISPATCH_N(value) \
  case value:             \
    return launch_for_mode<value>(mode, d_result, attrs);
    DISPATCH_N(0)
    DISPATCH_N(1)
    DISPATCH_N(2)
    DISPATCH_N(3)
    DISPATCH_N(4)
    DISPATCH_N(5)
    DISPATCH_N(6)
    DISPATCH_N(8)
    DISPATCH_N(10)
    DISPATCH_N(12)
    DISPATCH_N(16)
    DISPATCH_N(20)
    DISPATCH_N(24)
    DISPATCH_N(32)
    DISPATCH_N(40)
    DISPATCH_N(48)
    DISPATCH_N(56)
    DISPATCH_N(64)
#undef DISPATCH_N
    default:
      return false;
  }
}

template <int N, int Warps>
bool query_saturation_attrs(KernelAttrs* attrs) {
  cudaFuncAttributes cuda_attrs;
  const cudaError_t err =
      cudaFuncGetAttributes(&cuda_attrs, mma_saturation_kernel<N, Warps>);
  if (err != cudaSuccess) return false;
  attrs->regs = cuda_attrs.numRegs;
  attrs->local_bytes = cuda_attrs.localSizeBytes;
  attrs->shared_bytes = cuda_attrs.sharedSizeBytes;
  attrs->max_threads_per_block = cuda_attrs.maxThreadsPerBlock;
  return true;
}

template <int N, int Warps>
bool launch_saturation_once(SaturationResult* d_result, KernelAttrs* attrs) {
  if (!query_saturation_attrs<N, Warps>(attrs)) return false;
  mma_saturation_kernel<N, Warps><<<1, Warps * 32>>>(d_result);
  return cudaPeekAtLastError() == cudaSuccess;
}

template <int N>
bool launch_saturation_for_warps(int warps, SaturationResult* d_result,
                                 KernelAttrs* attrs) {
  switch (warps) {
#define DISPATCH_W(value) \
  case value:             \
    return launch_saturation_once<N, value>(d_result, attrs);
    DISPATCH_W(1)
    DISPATCH_W(2)
    DISPATCH_W(4)
    DISPATCH_W(8)
    DISPATCH_W(16)
#undef DISPATCH_W
    default:
      return false;
  }
}

bool launch_saturation_for_n_warps(int n, int warps,
                                   SaturationResult* d_result,
                                   KernelAttrs* attrs) {
  switch (n) {
#define DISPATCH_N(value) \
  case value:             \
    return launch_saturation_for_warps<value>(warps, d_result, attrs);
    DISPATCH_N(0)
    DISPATCH_N(1)
    DISPATCH_N(2)
    DISPATCH_N(3)
    DISPATCH_N(4)
    DISPATCH_N(5)
    DISPATCH_N(6)
    DISPATCH_N(8)
    DISPATCH_N(10)
    DISPATCH_N(12)
    DISPATCH_N(16)
    DISPATCH_N(20)
    DISPATCH_N(24)
    DISPATCH_N(32)
    DISPATCH_N(40)
    DISPATCH_N(48)
    DISPATCH_N(56)
    DISPATCH_N(64)
#undef DISPATCH_N
    default:
      return false;
  }
}

int main(int argc, char** argv) {
  int samples = 101;
  int warmup = 10;
  bool raw = false;
  double clock_ghz = 0.0;
  std::string bench_text = "accept";
  std::string n_text = "0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,56,64";
  std::string mode_text = "mma,fma,empty";
  std::string warp_text = "1,2,4,8,16";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--samples") {
      samples = std::atoi(need_value("--samples"));
    } else if (arg == "--warmup") {
      warmup = std::atoi(need_value("--warmup"));
    } else if (arg == "--bench") {
      bench_text = need_value("--bench");
    } else if (arg == "--n") {
      n_text = need_value("--n");
    } else if (arg == "--modes") {
      mode_text = need_value("--modes");
    } else if (arg == "--warps") {
      warp_text = need_value("--warps");
    } else if (arg == "--clock-ghz") {
      clock_ghz = std::atof(need_value("--clock-ghz"));
    } else if (arg == "--raw") {
      raw = true;
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: %s [--samples N] [--warmup N] "
          "[--bench accept,saturation] [--n 0,1,2,4,...] "
          "[--modes mma,fma,empty] [--warps 1,2,4,8,16] "
          "[--clock-ghz GHz] [--raw]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  std::vector<int> n_values = parse_int_list(n_text);
  std::vector<BenchMode> modes = parse_mode_list(mode_text);
  std::vector<int> warp_values = parse_int_list(warp_text);
  const bool run_accept = has_token(bench_text, "accept");
  const bool run_saturation = has_token(bench_text, "saturation");
  if (!run_accept && !run_saturation) {
    std::fprintf(stderr, "unknown --bench value: %s\n", bench_text.c_str());
    return 2;
  }

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  KernelResult* d_result = nullptr;
  check_cuda(cudaMalloc(&d_result, sizeof(KernelResult)), "cudaMalloc");
  SaturationResult* d_saturation = nullptr;
  check_cuda(cudaMalloc(&d_saturation, sizeof(SaturationResult)),
             "cudaMalloc saturation");

  std::printf("# mma_accept_queue_bench\n");
  std::printf("# device=%s cc=%d.%d sms=%d runtime=%d driver=%d\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              runtime_version, driver_version);
  std::printf("# samples=%d warmup=%d bench=%s clock_ghz=%.6f\n", samples,
              warmup, bench_text.c_str(), clock_ghz);

  if (run_accept) {
    std::printf(
        "bench,mode,n,regs,local_bytes,accept_min,accept_median,accept_mean,"
        "accept_p95,accept_max,complete_min,complete_median,complete_mean,"
        "complete_p95,complete_max,consume_median,sink\n");

    for (BenchMode mode : modes) {
      for (int n : n_values) {
        KernelAttrs attrs = {};
        bool supported = true;
        for (int i = 0; i < warmup; ++i) {
          supported = launch_for_n(n, mode, d_result, &attrs);
          if (!supported) break;
        }
        if (!supported) {
          std::printf("accept,%s,%d,unsupported,unsupported\n",
                      mode_name(mode), n);
          continue;
        }
        check_cuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");

        std::vector<uint64_t> accept;
        std::vector<uint64_t> complete;
        std::vector<uint64_t> consume;
        accept.reserve(samples);
        complete.reserve(samples);
        consume.reserve(samples);
        KernelResult last = {};

        for (int sample = 0; sample < samples; ++sample) {
          KernelAttrs sample_attrs = {};
          supported = launch_for_n(n, mode, d_result, &sample_attrs);
          if (!supported) break;
          check_cuda(cudaDeviceSynchronize(), "sample cudaDeviceSynchronize");
          check_cuda(cudaMemcpy(&last, d_result, sizeof(KernelResult),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy result");
          accept.push_back(last.t1 - last.t0);
          complete.push_back(last.t2 - last.t0);
          consume.push_back(last.t2 - last.t1);
          attrs = sample_attrs;
          if (raw) {
            std::printf("raw_accept,%s,%d,%d,%llu,%llu,%llu,%f\n",
                        mode_name(mode), n, sample,
                        static_cast<unsigned long long>(last.t1 - last.t0),
                        static_cast<unsigned long long>(last.t2 - last.t0),
                        static_cast<unsigned long long>(last.t2 - last.t1),
                        last.sink);
          }
        }
        if (!supported) {
          std::printf("accept,%s,%d,launch_failed,launch_failed\n",
                      mode_name(mode), n);
          continue;
        }

        const Stats a = summarize(accept);
        const Stats c = summarize(complete);
        const Stats u = summarize(consume);
        std::printf(
            "accept,%s,%d,%d,%zu,%.0f,%.0f,%.2f,%.0f,%.0f,%.0f,%.0f,%.2f,"
            "%.0f,%.0f,%.0f,%f\n",
            mode_name(mode), n, attrs.regs, attrs.local_bytes, a.min, a.median,
            a.mean, a.p95, a.max, c.min, c.median, c.mean, c.p95, c.max,
            u.median, last.sink);
      }
    }
  }

  if (run_saturation) {
    std::printf(
        "bench,warps,n,regs,local_bytes,shared_bytes,total_mma,"
        "accept_min,accept_median,accept_mean,accept_p95,accept_max,"
        "complete_median,consume_median,start_span_median,"
        "issue_end_span_median,complete_end_span_median,"
        "mma_per_cycle_sm,flop_per_cycle_sm,tflops_at_clock,sink\n");

    for (int warps : warp_values) {
      for (int n : n_values) {
        KernelAttrs attrs = {};
        bool supported = true;
        for (int i = 0; i < warmup; ++i) {
          supported =
              launch_saturation_for_n_warps(n, warps, d_saturation, &attrs);
          if (!supported) break;
        }
        if (!supported) {
          std::printf("saturation,%d,%d,unsupported,unsupported\n", warps, n);
          continue;
        }
        check_cuda(cudaDeviceSynchronize(), "saturation warmup sync");

        std::vector<uint64_t> accept;
        std::vector<uint64_t> complete;
        std::vector<uint64_t> consume;
        std::vector<uint64_t> start_span;
        std::vector<uint64_t> issue_end_span;
        std::vector<uint64_t> complete_end_span;
        accept.reserve(samples);
        complete.reserve(samples);
        consume.reserve(samples);
        start_span.reserve(samples);
        issue_end_span.reserve(samples);
        complete_end_span.reserve(samples);
        SaturationResult last = {};

        for (int sample = 0; sample < samples; ++sample) {
          KernelAttrs sample_attrs = {};
          supported = launch_saturation_for_n_warps(n, warps, d_saturation,
                                                    &sample_attrs);
          if (!supported) break;
          check_cuda(cudaDeviceSynchronize(), "saturation sample sync");
          check_cuda(cudaMemcpy(&last, d_saturation, sizeof(SaturationResult),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy saturation result");
          const uint64_t accept_cycles = last.t1_max - last.t0_min;
          const uint64_t complete_cycles = last.t2_max - last.t0_min;
          accept.push_back(accept_cycles);
          complete.push_back(complete_cycles);
          consume.push_back(complete_cycles - accept_cycles);
          start_span.push_back(last.t0_max - last.t0_min);
          issue_end_span.push_back(last.t1_max - last.t1_min);
          complete_end_span.push_back(last.t2_max - last.t2_min);
          attrs = sample_attrs;
          if (raw) {
            std::printf(
                "raw_saturation,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%f\n",
                warps, n, sample,
                static_cast<unsigned long long>(last.t0_min),
                static_cast<unsigned long long>(last.t0_max),
                static_cast<unsigned long long>(last.t1_min),
                static_cast<unsigned long long>(last.t1_max),
                static_cast<unsigned long long>(last.t2_min),
                static_cast<unsigned long long>(last.t2_max), last.sink);
          }
        }
        if (!supported) {
          std::printf("saturation,%d,%d,launch_failed,launch_failed\n", warps,
                      n);
          continue;
        }

        const Stats a = summarize(accept);
        const Stats c = summarize(complete);
        const Stats u = summarize(consume);
        const Stats s = summarize(start_span);
        const Stats e = summarize(issue_end_span);
        const Stats x = summarize(complete_end_span);
        const int total_mma = warps * n;
        const double mma_per_cycle =
            (a.median > 0.0) ? static_cast<double>(total_mma) / a.median : 0.0;
        const double flop_per_cycle_sm = mma_per_cycle * kFlopPerMma16816;
        const double tflops_at_clock =
            flop_per_cycle_sm * static_cast<double>(prop.multiProcessorCount) *
            clock_ghz / 1000.0;
        std::printf(
            "saturation,%d,%d,%d,%zu,%zu,%d,%.0f,%.0f,%.2f,%.0f,%.0f,%.0f,"
            "%.0f,%.0f,%.0f,%.0f,%.6f,%.2f,%.2f,%f\n",
            warps, n, attrs.regs, attrs.local_bytes, attrs.shared_bytes,
            total_mma, a.min, a.median, a.mean, a.p95, a.max, c.median,
            u.median, s.median, e.median, x.median, mma_per_cycle,
            flop_per_cycle_sm, tflops_at_clock, last.sink);
      }
    }
  }

  check_cuda(cudaFree(d_result), "cudaFree");
  check_cuda(cudaFree(d_saturation), "cudaFree saturation");
  return 0;
}
