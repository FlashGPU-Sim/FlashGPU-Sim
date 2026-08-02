#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kCopyBytes = 16;

enum class Mode { Register, Full, Immediate, Odd, Zero, Ignore };

struct alignas(kCopyBytes) Block16 {
  uint8_t bytes[kCopyBytes];
};

#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t error__ = (expr);                                        \
    if (error__ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error__));                         \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t global_address(const void *ptr) {
  uint64_t address = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(address) : "l"(ptr));
  return address;
}

__host__ __device__ constexpr uint32_t source_size_for_lane(int lane) {
  return lane < 8 ? 0 : lane < 16 ? 4 : lane < 24 ? 8 : 16;
}

__host__ __device__ constexpr uint32_t odd_source_size_for_lane(int lane) {
  return lane < 16 ? 1 : 15;
}

template <Mode CopyMode>
__global__ void cp_async_src_size_kernel(const Block16 *input,
                                         Block16 *output) {
  __shared__ __align__(kCopyBytes) Block16 shared[kWarpSize];

  int lane = threadIdx.x;
  uint32_t dst = shared_address(&shared[lane]);
  uint64_t src = global_address(&input[lane]);
  if constexpr (CopyMode == Mode::Full) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(dst), "l"(src)
                 : "memory");
  } else if constexpr (CopyMode == Mode::Immediate) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 4;\n"
                 :
                 : "r"(dst), "l"(src)
                 : "memory");
  } else if constexpr (CopyMode == Mode::Ignore) {
    uint32_t ignore = lane < 16;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "cp.async.cg.shared.global [%0], [%1], 16, p;\n\t"
        "}\n"
        :
        : "r"(dst), "l"(src), "r"(ignore)
        : "memory");
  } else {
    uint32_t src_size = CopyMode == Mode::Odd ? odd_source_size_for_lane(lane)
                        : CopyMode == Mode::Zero ? 0
                                                 : source_size_for_lane(lane);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(dst), "l"(src), "r"(src_size)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  output[lane] = shared[lane];
}

}  // namespace

int main(int argc, char **argv) {
  Mode mode = Mode::Register;
  if (argc == 2 && std::strcmp(argv[1], "--full") == 0) {
    mode = Mode::Full;
  } else if (argc == 2 && std::strcmp(argv[1], "--immediate") == 0) {
    mode = Mode::Immediate;
  } else if (argc == 2 && std::strcmp(argv[1], "--odd") == 0) {
    mode = Mode::Odd;
  } else if (argc == 2 && std::strcmp(argv[1], "--zero") == 0) {
    mode = Mode::Zero;
  } else if (argc == 2 && std::strcmp(argv[1], "--ignore") == 0) {
    mode = Mode::Ignore;
  } else if (argc != 1) {
    std::fprintf(stderr,
                 "usage: %s [--full|--immediate|--odd|--zero|--ignore]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<Block16> input(kWarpSize);
  std::vector<Block16> output(kWarpSize);
  for (int lane = 0; lane < kWarpSize; ++lane) {
    for (int byte = 0; byte < kCopyBytes; ++byte) {
      input[lane].bytes[byte] =
          static_cast<uint8_t>(1 + (lane * kCopyBytes + byte) % 251);
      output[lane].bytes[byte] = 0xa5;
    }
  }

  Block16 *device_input = nullptr;
  Block16 *device_output = nullptr;
  size_t bytes = input.size() * sizeof(Block16);
  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(
      cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(device_output, output.data(), bytes, cudaMemcpyHostToDevice));

  switch (mode) {
    case Mode::Register:
      cp_async_src_size_kernel<Mode::Register>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
    case Mode::Full:
      cp_async_src_size_kernel<Mode::Full>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
    case Mode::Immediate:
      cp_async_src_size_kernel<Mode::Immediate>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
    case Mode::Odd:
      cp_async_src_size_kernel<Mode::Odd>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
    case Mode::Zero:
      cp_async_src_size_kernel<Mode::Zero>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
    case Mode::Ignore:
      cp_async_src_size_kernel<Mode::Ignore>
          <<<1, kWarpSize>>>(device_input, device_output);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(
      cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int lane = 0; lane < kWarpSize; ++lane) {
    int src_size = mode == Mode::Full        ? kCopyBytes
                   : mode == Mode::Immediate ? 4
                   : mode == Mode::Odd       ? odd_source_size_for_lane(lane)
                   : mode == Mode::Zero      ? 0
                   : mode == Mode::Ignore    ? (lane < 16 ? 0 : kCopyBytes)
                                             : source_size_for_lane(lane);
    for (int byte = 0; byte < kCopyBytes; ++byte) {
      uint8_t expected = byte < src_size ? input[lane].bytes[byte] : 0;
      if (output[lane].bytes[byte] != expected) {
        if (mismatches < 8) {
          std::fprintf(stderr,
                       "lane=%d byte=%d src_size=%d expected=%u actual=%u\n",
                       lane, byte, src_size, unsigned(expected),
                       unsigned(output[lane].bytes[byte]));
        }
        ++mismatches;
      }
    }
  }

  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_input));
  const char *mode_name = mode == Mode::Full        ? "full-control"
                          : mode == Mode::Immediate ? "immediate-4"
                          : mode == Mode::Odd       ? "register-1-15"
                          : mode == Mode::Zero      ? "register-all-zero"
                          : mode == Mode::Ignore    ? "ignore-src"
                                                    : "register-0-4-8-16";
  std::printf("cp.async src_size mode=%s mismatches=%d\n", mode_name,
              mismatches);
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
