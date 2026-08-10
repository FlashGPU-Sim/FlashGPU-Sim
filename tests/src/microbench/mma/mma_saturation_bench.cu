#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct KernelAttrs {
  int regs;
  size_t local_bytes;
  size_t shared_bytes;
  int max_threads_per_block;
};

struct RunResult {
  float sink;
  float elapsed_ms;
  uint64_t mma_inst;
  double tflops;
};

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
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

template <int Unroll>
__device__ __forceinline__ void init_accumulators(float d[][4], int lane,
                                                  int warp_id) {
#pragma unroll
  for (int i = 0; i < Unroll; ++i) {
    const float base = 0.001953125f *
                       static_cast<float>((i + 1) * ((lane & 7) + 1) +
                                          (warp_id & 31));
    d[i][0] = base + 0.125f;
    d[i][1] = base + 0.250f;
    d[i][2] = base + 0.375f;
    d[i][3] = base + 0.500f;
  }

#pragma unroll
  for (int i = 0; i < Unroll; ++i) {
    asm volatile("" ::"f"(d[i][0]), "f"(d[i][1]), "f"(d[i][2]),
                 "f"(d[i][3])
                 : "memory");
  }
}

template <int Unroll>
__device__ __forceinline__ void emit_mma_unroll(float d[][4],
                                                const unsigned* a,
                                                const unsigned* b) {
#pragma unroll
  for (int i = 0; i < Unroll; ++i) {
    emit_mma_group(d[i], a, b);
  }
}

template <int Unroll>
__device__ __forceinline__ float consume_outputs(float d[][4]) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < Unroll; ++i) {
    sum += d[i][0];
    sum += d[i][1];
    sum += d[i][2];
    sum += d[i][3];
  }
  return sum;
}

template <int WarpsPerCta, int Unroll>
__global__ __launch_bounds__(WarpsPerCta * 32, 1) void mma_saturation_kernel(
    float* out, int repeat) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int global_warp = static_cast<int>(blockIdx.x) * WarpsPerCta + warp;

  float d[Unroll][4];
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

  init_accumulators<Unroll>(d, lane, global_warp);
  asm volatile("bar.sync 0;" ::: "memory");

  for (int iter = 0; iter < repeat; ++iter) {
    emit_mma_unroll<Unroll>(d, a, b);
  }

  const float sink = consume_outputs<Unroll>(d);
  if (lane == 0) {
    out[static_cast<size_t>(blockIdx.x) * WarpsPerCta + warp] = sink;
  }
}

template <int WarpsPerCta, int Unroll>
bool query_attrs(KernelAttrs* attrs) {
  cudaFuncAttributes cuda_attrs;
  const cudaError_t err =
      cudaFuncGetAttributes(&cuda_attrs,
                            mma_saturation_kernel<WarpsPerCta, Unroll>);
  if (err != cudaSuccess) return false;
  attrs->regs = cuda_attrs.numRegs;
  attrs->local_bytes = cuda_attrs.localSizeBytes;
  attrs->shared_bytes = cuda_attrs.sharedSizeBytes;
  attrs->max_threads_per_block = cuda_attrs.maxThreadsPerBlock;
  return true;
}

template <int WarpsPerCta, int Unroll>
RunResult run_once(int blocks, int repeat, KernelAttrs* attrs) {
  if (!query_attrs<WarpsPerCta, Unroll>(attrs)) {
    std::fprintf(stderr, "unsupported kernel variant warps=%d unroll=%d\n",
                 WarpsPerCta, Unroll);
    std::exit(2);
  }

  float* d_out = nullptr;
  const size_t out_count = static_cast<size_t>(blocks) * WarpsPerCta;
  check_cuda(cudaMalloc(&d_out, out_count * sizeof(float)), "cudaMalloc out");

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  mma_saturation_kernel<WarpsPerCta, Unroll>
      <<<blocks, WarpsPerCta * 32>>>(d_out, repeat);
  check_cuda(cudaPeekAtLastError(), "kernel launch");
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");

  std::vector<float> host(out_count);
  check_cuda(cudaMemcpy(host.data(), d_out, out_count * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy out");
  float sink = 0.0f;
  for (float value : host) sink += value;

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  check_cuda(cudaFree(d_out), "cudaFree out");

  const uint64_t mma_inst = static_cast<uint64_t>(blocks) * WarpsPerCta *
                            static_cast<uint64_t>(Unroll) *
                            static_cast<uint64_t>(repeat);
  constexpr double kFlopPerMma = 16.0 * 8.0 * 16.0 * 2.0;
  const double seconds = static_cast<double>(elapsed_ms) * 1.0e-3;
  const double tflops =
      seconds > 0.0 ? (static_cast<double>(mma_inst) * kFlopPerMma / seconds) *
                           1.0e-12
                    : 0.0;
  return {sink, elapsed_ms, mma_inst, tflops};
}

template <int WarpsPerCta>
RunResult run_for_unroll(int unroll, int blocks, int repeat,
                         KernelAttrs* attrs) {
  switch (unroll) {
    case 4:
      return run_once<WarpsPerCta, 4>(blocks, repeat, attrs);
    case 8:
      return run_once<WarpsPerCta, 8>(blocks, repeat, attrs);
    case 16:
      return run_once<WarpsPerCta, 16>(blocks, repeat, attrs);
    case 24:
      return run_once<WarpsPerCta, 24>(blocks, repeat, attrs);
    case 32:
      return run_once<WarpsPerCta, 32>(blocks, repeat, attrs);
    default:
      std::fprintf(stderr, "unsupported unroll=%d\n", unroll);
      std::exit(2);
  }
}

RunResult run_for_shape(int warps, int unroll, int blocks, int repeat,
                        KernelAttrs* attrs) {
  switch (warps) {
    case 1:
      return run_for_unroll<1>(unroll, blocks, repeat, attrs);
    case 2:
      return run_for_unroll<2>(unroll, blocks, repeat, attrs);
    case 4:
      return run_for_unroll<4>(unroll, blocks, repeat, attrs);
    case 8:
      return run_for_unroll<8>(unroll, blocks, repeat, attrs);
    case 16:
      return run_for_unroll<16>(unroll, blocks, repeat, attrs);
    default:
      std::fprintf(stderr, "unsupported warps=%d\n", warps);
      std::exit(2);
  }
}

int main(int argc, char** argv) {
  int warps = 8;
  int unroll = 16;
  int repeat = 4096;
  int blocks_per_sm = 4;
  int blocks = 0;
  int warmup = 2;
  int samples = 5;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--warps") {
      warps = std::atoi(need_value("--warps"));
    } else if (arg == "--unroll") {
      unroll = std::atoi(need_value("--unroll"));
    } else if (arg == "--repeat") {
      repeat = std::atoi(need_value("--repeat"));
    } else if (arg == "--blocks-per-sm") {
      blocks_per_sm = std::atoi(need_value("--blocks-per-sm"));
    } else if (arg == "--blocks") {
      blocks = std::atoi(need_value("--blocks"));
    } else if (arg == "--warmup") {
      warmup = std::atoi(need_value("--warmup"));
    } else if (arg == "--samples") {
      samples = std::atoi(need_value("--samples"));
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: %s [--warps 1|2|4|8|16] [--unroll 4|8|16|24|32] "
          "[--repeat N] [--blocks-per-sm N] [--blocks N] "
          "[--warmup N] [--samples N]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  if (blocks == 0) {
    blocks = prop.multiProcessorCount * blocks_per_sm;
  }

  KernelAttrs attrs = {};
  for (int i = 0; i < warmup; ++i) {
    (void)run_for_shape(warps, unroll, blocks, repeat, &attrs);
  }
  check_cuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");

  std::vector<RunResult> runs;
  runs.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    runs.push_back(run_for_shape(warps, unroll, blocks, repeat, &attrs));
  }
  check_cuda(cudaDeviceSynchronize(), "sample cudaDeviceSynchronize");

  auto by_ms = runs;
  std::sort(by_ms.begin(), by_ms.end(),
            [](const RunResult& a, const RunResult& b) {
              return a.elapsed_ms < b.elapsed_ms;
            });
  const RunResult best = by_ms.front();
  const RunResult median = by_ms[by_ms.size() / 2];

  std::printf("# mma_saturation_bench\n");
  std::printf("# device=%s cc=%d.%d sms=%d runtime=%d driver=%d\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              runtime_version, driver_version);
  std::printf(
      "warps,unroll,repeat,blocks,blocks_per_sm,regs,local_bytes,"
      "mma_inst,best_ms,best_tflops,median_ms,median_tflops,sink\n");
  std::printf("%d,%d,%d,%d,%.3f,%d,%zu,%llu,%.6f,%.3f,%.6f,%.3f,%f\n",
              warps, unroll, repeat, blocks,
              static_cast<double>(blocks) /
                  static_cast<double>(prop.multiProcessorCount),
              attrs.regs, attrs.local_bytes,
              static_cast<unsigned long long>(median.mma_inst), best.elapsed_ms,
              best.tflops, median.elapsed_ms, median.tflops, median.sink);
  return 0;
}
