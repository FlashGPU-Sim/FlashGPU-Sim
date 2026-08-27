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

#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)
#define PC_STEP() asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
#define PC_REP_1() PC_STEP()
#define PC_REP_2() PC_REP_1() PC_REP_1()
#define PC_REP_4() PC_REP_2() PC_REP_2()
#define PC_REP_8() PC_REP_4() PC_REP_4()
#define PC_REP_16() PC_REP_8() PC_REP_8()
#define PC_REP_32() PC_REP_16() PC_REP_16()
#define PC_REP_64() PC_REP_32() PC_REP_32()
#define PC_REP_128() PC_REP_64() PC_REP_64()
#define PC_REP_256() PC_REP_128() PC_REP_128()
#define PC_REP_512() PC_REP_256() PC_REP_256()
#define PC_REP_1024() PC_REP_512() PC_REP_512()

#ifndef STREAM_BODY_STEPS
#define STREAM_BODY_STEPS 1024
#endif

#if STREAM_BODY_STEPS == 16
#define STREAM_BODY() PC_REP_16()
#elif STREAM_BODY_STEPS == 32
#define STREAM_BODY() PC_REP_32()
#elif STREAM_BODY_STEPS == 64
#define STREAM_BODY() PC_REP_64()
#elif STREAM_BODY_STEPS == 128
#define STREAM_BODY() PC_REP_128()
#elif STREAM_BODY_STEPS == 256
#define STREAM_BODY() PC_REP_256()
#elif STREAM_BODY_STEPS == 512
#define STREAM_BODY() PC_REP_512()
#elif STREAM_BODY_STEPS == 1024
#define STREAM_BODY() PC_REP_1024()
#else
#error STREAM_BODY_STEPS must be 16, 32, 64, 128, 256, 512, or 1024
#endif

#define DEFINE_PC_STREAM(id)                                             \
  __device__ __noinline__ unsigned int pc_stream_##id(int repetitions) { \
    unsigned int value = 0;                                              \
    asm volatile("mov.u32 %0, " STRINGIFY(id) ";" : "=r"(value));        \
    for (int iteration = 0; iteration < repetitions; ++iteration) {      \
      STREAM_BODY()                                                      \
    }                                                                    \
    return value;                                                        \
  }

DEFINE_PC_STREAM(0)
DEFINE_PC_STREAM(1)
DEFINE_PC_STREAM(2)
DEFINE_PC_STREAM(3)
DEFINE_PC_STREAM(4)
DEFINE_PC_STREAM(5)
DEFINE_PC_STREAM(6)
DEFINE_PC_STREAM(7)
DEFINE_PC_STREAM(8)
DEFINE_PC_STREAM(9)
DEFINE_PC_STREAM(10)
DEFINE_PC_STREAM(11)
DEFINE_PC_STREAM(12)
DEFINE_PC_STREAM(13)
DEFINE_PC_STREAM(14)
DEFINE_PC_STREAM(15)
DEFINE_PC_STREAM(16)
DEFINE_PC_STREAM(17)
DEFINE_PC_STREAM(18)
DEFINE_PC_STREAM(19)
DEFINE_PC_STREAM(20)
DEFINE_PC_STREAM(21)
DEFINE_PC_STREAM(22)
DEFINE_PC_STREAM(23)
DEFINE_PC_STREAM(24)
DEFINE_PC_STREAM(25)
DEFINE_PC_STREAM(26)
DEFINE_PC_STREAM(27)
DEFINE_PC_STREAM(28)
DEFINE_PC_STREAM(29)
DEFINE_PC_STREAM(30)
DEFINE_PC_STREAM(31)

#define PC_STREAM_CASE(id)               \
  case id:                               \
    value = pc_stream_##id(repetitions); \
    break;

__global__ __launch_bounds__(1024, 1) void parallel_pc_kernel(
    unsigned int* output, unsigned long long* cycles, int repetitions) {
  const unsigned int warp_id = threadIdx.x / warpSize;
  const unsigned int lane_id = threadIdx.x % warpSize;
  const unsigned long long begin = clock64();
  unsigned int value = 0;
  switch (warp_id) {
    PC_STREAM_CASE(0)
    PC_STREAM_CASE(1)
    PC_STREAM_CASE(2)
    PC_STREAM_CASE(3)
    PC_STREAM_CASE(4)
    PC_STREAM_CASE(5)
    PC_STREAM_CASE(6)
    PC_STREAM_CASE(7)
    PC_STREAM_CASE(8)
    PC_STREAM_CASE(9)
    PC_STREAM_CASE(10)
    PC_STREAM_CASE(11)
    PC_STREAM_CASE(12)
    PC_STREAM_CASE(13)
    PC_STREAM_CASE(14)
    PC_STREAM_CASE(15)
    PC_STREAM_CASE(16)
    PC_STREAM_CASE(17)
    PC_STREAM_CASE(18)
    PC_STREAM_CASE(19)
    PC_STREAM_CASE(20)
    PC_STREAM_CASE(21)
    PC_STREAM_CASE(22)
    PC_STREAM_CASE(23)
    PC_STREAM_CASE(24)
    PC_STREAM_CASE(25)
    PC_STREAM_CASE(26)
    PC_STREAM_CASE(27)
    PC_STREAM_CASE(28)
    PC_STREAM_CASE(29)
    PC_STREAM_CASE(30)
    PC_STREAM_CASE(31)
  }
  const unsigned long long end = clock64();
  if (lane_id == 0) {
    output[warp_id] = value;
    cycles[warp_id] = end - begin;
  }
}

struct Options {
  int device = 0;
  int streams = 1;
  int repetitions = 1;
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
          "Usage: %s [--device=N] [--streams=1..32] [--repetitions=N] "
          "[--warmup-launches=N]\n",
          argv[0]);
      std::exit(0);
    }
    if (parse_int(argv[i], "--device", &options.device) ||
        parse_int(argv[i], "--streams", &options.streams) ||
        parse_int(argv[i], "--repetitions", &options.repetitions) ||
        parse_int(argv[i], "--warmup-launches", &options.warmup_launches)) {
      continue;
    }
    std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
    std::exit(1);
  }
  if (options.device < 0 || options.streams < 1 || options.streams > 32 ||
      options.repetitions < 0 || options.warmup_launches < 0) {
    std::fprintf(stderr, "Option values are outside their valid range\n");
    std::exit(1);
  }
  return options;
}

void launch(const Options& options, unsigned int* output,
            unsigned long long* cycles) {
  parallel_pc_kernel<<<1, options.streams * 32>>>(output, cycles,
                                                  options.repetitions);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(options.device));
  unsigned int* output = nullptr;
  unsigned long long* cycles = nullptr;
  CUDA_CHECK(cudaMallocManaged(&output, options.streams * sizeof(*output)));
  CUDA_CHECK(cudaMallocManaged(&cycles, options.streams * sizeof(*cycles)));

  for (int i = 0; i < options.warmup_launches; ++i) {
    launch(options, output, cycles);
  }
  launch(options, output, cycles);

  unsigned long long max_cycles = 0;
  unsigned int checksum = 0;
  for (int i = 0; i < options.streams; ++i) {
    if (cycles[i] > max_cycles) max_cycles = cycles[i];
    checksum += output[i];
  }
  std::printf(
      "streams=%d body_steps=%d repetitions=%d warmup_launches=%d "
      "max_cycles=%llu checksum=%u\n",
      options.streams, STREAM_BODY_STEPS, options.repetitions,
      options.warmup_launches, max_cycles, checksum);

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(cycles));
  return 0;
}
