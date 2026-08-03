#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kCopyBytes = 16;

enum class CopyMode {
  Full,
  Immediate,
  Register,
  Odd,
  Zero,
  Ignore,
  CacheHint,
  SourceSizeAndCacheHint,
};

struct alignas(kCopyBytes) Block16 {
  uint8_t bytes[kCopyBytes];
};

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t global_address(const void *ptr) {
  uint64_t address = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(address) : "l"(ptr));
  return address;
}

__host__ __device__ constexpr uint32_t source_size_for_lane(int lane) {
  return lane < 8 ? 0 : lane < 16 ? 4 : lane < 24 ? 8 : 12;
}

__host__ __device__ constexpr uint32_t odd_source_size_for_lane(int lane) {
  return lane < 16 ? 1 : 15;
}

__device__ __forceinline__ uint64_t l2_cache_policy() {
  uint64_t policy = 0;
  // Keep the policy as a b64 register operand. The simulator currently treats
  // cache policy as a performance hint and does not decode its value.
  asm volatile("mov.b64 %0, 0;" : "=l"(policy));
  return policy;
}

template <CopyMode Mode>
__global__ void cp_async_src_size_kernel(const Block16 *input,
                                         Block16 *output) {
  __shared__ __align__(kCopyBytes) Block16 shared[kWarpSize];

  const int lane = threadIdx.x;
  const uint32_t dst = shared_address(&shared[lane]);
  const uint64_t src = global_address(&input[lane]);

  if constexpr (Mode == CopyMode::Full) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(dst), "l"(src)
                 : "memory");
  } else if constexpr (Mode == CopyMode::Immediate) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 4;"
                 :
                 : "r"(dst), "l"(src)
                 : "memory");
  } else if constexpr (Mode == CopyMode::Ignore) {
    const uint32_t ignore = lane < 16;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "cp.async.cg.shared.global [%0], [%1], 16, p;\n\t"
        "}"
        :
        : "r"(dst), "l"(src), "r"(ignore)
        : "memory");
  } else if constexpr (Mode == CopyMode::CacheHint) {
    const uint64_t policy = l2_cache_policy();
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                 :
                 : "r"(dst), "l"(src), "l"(policy)
                 : "memory");
  } else if constexpr (Mode == CopyMode::SourceSizeAndCacheHint) {
    const uint32_t source_size = source_size_for_lane(lane);
    const uint64_t policy = l2_cache_policy();
    asm volatile(
        "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
        :
        : "r"(dst), "l"(src), "r"(source_size), "l"(policy)
        : "memory");
  } else {
    const uint32_t source_size =
        Mode == CopyMode::Odd    ? odd_source_size_for_lane(lane)
        : Mode == CopyMode::Zero ? 0
                                 : source_size_for_lane(lane);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(dst), "l"(src), "r"(source_size)
                 : "memory");
  }

  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  output[lane] = shared[lane];
}

uint32_t expected_source_size(CopyMode mode, int lane) {
  switch (mode) {
    case CopyMode::Full:
    case CopyMode::CacheHint:
      return kCopyBytes;
    case CopyMode::Immediate:
      return 4;
    case CopyMode::Register:
    case CopyMode::SourceSizeAndCacheHint:
      return source_size_for_lane(lane);
    case CopyMode::Odd:
      return odd_source_size_for_lane(lane);
    case CopyMode::Zero:
      return 0;
    case CopyMode::Ignore:
      return lane < 16 ? 0 : kCopyBytes;
  }
  return 0;
}

template <CopyMode Mode>
void launch_kernel(const Block16 *input, Block16 *output) {
  cp_async_src_size_kernel<Mode><<<1, kWarpSize>>>(input, output);
}

void run_and_check(CopyMode mode) {
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
  const size_t bytes = input.size() * sizeof(Block16);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_input, bytes));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_output, bytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_input, input.data(), bytes,
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_output, output.data(), bytes,
                                    cudaMemcpyHostToDevice));

  switch (mode) {
    case CopyMode::Full:
      launch_kernel<CopyMode::Full>(device_input, device_output);
      break;
    case CopyMode::Immediate:
      launch_kernel<CopyMode::Immediate>(device_input, device_output);
      break;
    case CopyMode::Register:
      launch_kernel<CopyMode::Register>(device_input, device_output);
      break;
    case CopyMode::Odd:
      launch_kernel<CopyMode::Odd>(device_input, device_output);
      break;
    case CopyMode::Zero:
      launch_kernel<CopyMode::Zero>(device_input, device_output);
      break;
    case CopyMode::Ignore:
      launch_kernel<CopyMode::Ignore>(device_input, device_output);
      break;
    case CopyMode::CacheHint:
      launch_kernel<CopyMode::CacheHint>(device_input, device_output);
      break;
    case CopyMode::SourceSizeAndCacheHint:
      launch_kernel<CopyMode::SourceSizeAndCacheHint>(device_input,
                                                      device_output);
      break;
  }

  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), device_output, bytes,
                                    cudaMemcpyDeviceToHost));

  for (int lane = 0; lane < kWarpSize; ++lane) {
    const uint32_t source_size = expected_source_size(mode, lane);
    for (int byte = 0; byte < kCopyBytes; ++byte) {
      const uint8_t expected = byte < source_size ? input[lane].bytes[byte] : 0;
      EXPECT_EQ(expected, output[lane].bytes[byte])
          << "lane=" << lane << " byte=" << byte;
    }
  }

  EXPECT_EQ(cudaSuccess, cudaFree(device_output));
  EXPECT_EQ(cudaSuccess, cudaFree(device_input));
}

TEST(CpAsyncSrcSizeTest, FullCopy) { run_and_check(CopyMode::Full); }

TEST(CpAsyncSrcSizeTest, ImmediateSourceSize) {
  run_and_check(CopyMode::Immediate);
}

TEST(CpAsyncSrcSizeTest, PerLaneRegisterSourceSize) {
  run_and_check(CopyMode::Register);
}

TEST(CpAsyncSrcSizeTest, OddSourceSizes) { run_and_check(CopyMode::Odd); }

TEST(CpAsyncSrcSizeTest, AllZeroSourceSize) { run_and_check(CopyMode::Zero); }

TEST(CpAsyncSrcSizeTest, IgnoreSourcePredicate) {
  run_and_check(CopyMode::Ignore);
}

TEST(CpAsyncSrcSizeTest, CachePolicyWithoutSourceSize) {
  run_and_check(CopyMode::CacheHint);
}

TEST(CpAsyncSrcSizeTest, SourceSizeWithCachePolicy) {
  run_and_check(CopyMode::SourceSizeAndCacheHint);
}

}  // namespace
