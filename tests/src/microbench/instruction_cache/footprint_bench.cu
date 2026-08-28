#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    const cudaError_t error__ = (call);                                  \
    if (error__ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error__));                         \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#ifdef ICACHE_TIMING
#define ICACHE_STEP_0()                                                   \
  asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value0));
#define ICACHE_STEP_1()                                                   \
  asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value1));
#define ICACHE_STEP_2()                                                   \
  asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value2));
#define ICACHE_STEP_3()                                                   \
  asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value3));
#define BENCHMARK_MODE "timing-4-chain"
#define BENCHMARK_CHAINS 4
#else
#define ICACHE_STEP_0() \
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
#define ICACHE_STEP_1() ICACHE_STEP_0()
#define ICACHE_STEP_2() ICACHE_STEP_0()
#define ICACHE_STEP_3() ICACHE_STEP_0()
#define BENCHMARK_MODE "footprint"
#define BENCHMARK_CHAINS 0
#endif
#define REP_1() ICACHE_STEP_0()
#define REP_2() ICACHE_STEP_0() ICACHE_STEP_1()
#define REP_4() REP_2() ICACHE_STEP_2() ICACHE_STEP_3()
#define REP_8() REP_4() REP_4()
#define REP_16() REP_8() REP_8()
#define REP_32() REP_16() REP_16()
#define REP_64() REP_32() REP_32()
#define REP_128() REP_64() REP_64()
#define REP_256() REP_128() REP_128()
#define REP_512() REP_256() REP_256()
#define REP_1024() REP_512() REP_512()
#define REP_2048() REP_1024() REP_1024()
#define REP_4096() REP_2048() REP_2048()
#define REP_8192() REP_4096() REP_4096()
#define REP_16384() REP_8192() REP_8192()

#ifndef FOOTPRINT_STEPS
#define FOOTPRINT_STEPS 512
#endif

#if FOOTPRINT_STEPS <= 0 || FOOTPRINT_STEPS >= 32768
#error FOOTPRINT_STEPS must be in [1, 32767]
#endif

#if FOOTPRINT_STEPS & 16384
#define BODY_16384() REP_16384()
#else
#define BODY_16384()
#endif
#if FOOTPRINT_STEPS & 8192
#define BODY_8192() REP_8192()
#else
#define BODY_8192()
#endif
#if FOOTPRINT_STEPS & 4096
#define BODY_4096() REP_4096()
#else
#define BODY_4096()
#endif
#if FOOTPRINT_STEPS & 2048
#define BODY_2048() REP_2048()
#else
#define BODY_2048()
#endif
#if FOOTPRINT_STEPS & 1024
#define BODY_1024() REP_1024()
#else
#define BODY_1024()
#endif
#if FOOTPRINT_STEPS & 512
#define BODY_512() REP_512()
#else
#define BODY_512()
#endif
#if FOOTPRINT_STEPS & 256
#define BODY_256() REP_256()
#else
#define BODY_256()
#endif
#if FOOTPRINT_STEPS & 128
#define BODY_128() REP_128()
#else
#define BODY_128()
#endif
#if FOOTPRINT_STEPS & 64
#define BODY_64() REP_64()
#else
#define BODY_64()
#endif
#if FOOTPRINT_STEPS & 32
#define BODY_32() REP_32()
#else
#define BODY_32()
#endif
#if FOOTPRINT_STEPS & 16
#define BODY_16() REP_16()
#else
#define BODY_16()
#endif
#if FOOTPRINT_STEPS & 8
#define BODY_8() REP_8()
#else
#define BODY_8()
#endif
#if FOOTPRINT_STEPS & 4
#define BODY_4() REP_4()
#else
#define BODY_4()
#endif
#if FOOTPRINT_STEPS & 2
#define BODY_2() REP_2()
#else
#define BODY_2()
#endif
#if FOOTPRINT_STEPS & 1
#define BODY_1() REP_1()
#else
#define BODY_1()
#endif

// clang-format off
#define FOOTPRINT_BODY()                                                  \
  BODY_16384()                                                            \
  BODY_8192() BODY_4096() BODY_2048() BODY_1024() BODY_512() BODY_256()   \
  BODY_128() BODY_64() BODY_32() BODY_16() BODY_8() BODY_4() BODY_2()      \
  BODY_1()
// clang-format on

struct Options {
  int device = 0;
  int blocks = 1;
  int repetitions = 2048;
  int warmup_launches = 0;
};

bool parse_int(const char* arg, const char* name, int* value) {
  const size_t length = std::strlen(name);
  if (std::strncmp(arg, name, length) != 0 || arg[length] != '=') return false;
  char* end = nullptr;
  const long parsed = std::strtol(arg + length + 1, &end, 0);
  if (end == arg + length + 1 || *end != '\0') {
    std::fprintf(stderr, "Invalid integer option: %s\n", arg);
    std::exit(1);
  }
  *value = static_cast<int>(parsed);
  return true;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 ||
        std::strcmp(argv[i], "-h") == 0) {
      std::printf(
          "Usage: %s [--device=N] [--blocks=N] [--repetitions=N] "
          "[--warmup-launches=N]\n",
          argv[0]);
      std::exit(0);
    }
    if (parse_int(argv[i], "--device", &options.device) ||
        parse_int(argv[i], "--blocks", &options.blocks) ||
        parse_int(argv[i], "--repetitions", &options.repetitions) ||
        parse_int(argv[i], "--warmup-launches", &options.warmup_launches)) {
      continue;
    }
    std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
    std::exit(1);
  }
  if (options.device < 0 || options.blocks <= 0 || options.repetitions < 0 ||
      options.warmup_launches < 0) {
    std::fprintf(stderr, "Option values are outside their valid range\n");
    std::exit(1);
  }
  return options;
}

__global__ __launch_bounds__(32, 1) void footprint_kernel(
    unsigned int* output, unsigned long long* cycles, int repetitions) {
  const unsigned int seed = threadIdx.x + blockIdx.x + 1;
#ifdef ICACHE_TIMING
  unsigned int value0 = seed;
  unsigned int value1 = seed + 0x9e3779b9u;
  unsigned int value2 = seed + 2u * 0x9e3779b9u;
  unsigned int value3 = seed + 3u * 0x9e3779b9u;
#else
  unsigned int value0 = seed;
#endif
  const unsigned long long begin = clock64();
#pragma unroll 1
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    FOOTPRINT_BODY()
  }
  const unsigned long long end = clock64();
#ifdef ICACHE_TIMING
  const unsigned int sink = value0 ^ value1 ^ value2 ^ value3;
#else
  const unsigned int sink = value0;
#endif
  if (threadIdx.x == 0) {
    output[blockIdx.x] = sink;
    cycles[blockIdx.x] = end - begin;
  }
}

void launch(const Options& options, unsigned int* output,
            unsigned long long* cycles) {
  footprint_kernel<<<options.blocks, 32>>>(output, cycles, options.repetitions);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(options.device));

  unsigned int* output = nullptr;
  unsigned long long* cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&output, options.blocks * sizeof(*output)));
  CUDA_CHECK(cudaMalloc(&cycles, options.blocks * sizeof(*cycles)));

  for (int i = 0; i < options.warmup_launches; ++i) {
    launch(options, output, cycles);
  }
  launch(options, output, cycles);

  unsigned int host_output = 0;
  unsigned long long host_cycles = 0;
  CUDA_CHECK(cudaMemcpy(&host_output, output, sizeof(host_output),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_cycles, cycles, sizeof(host_cycles),
                        cudaMemcpyDeviceToHost));

  std::printf(
      "mode=%s footprint_steps=%d dependency_chains=%d blocks=%d "
      "repetitions=%d warmup_launches=%d cycles=%llu sink=%u\n",
      BENCHMARK_MODE, FOOTPRINT_STEPS, BENCHMARK_CHAINS, options.blocks,
      options.repetitions, options.warmup_launches, host_cycles, host_output);

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(cycles));
  return 0;
}
