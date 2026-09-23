#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include "common/cluster_launch.h"

namespace {
__global__ void event_clock_span(unsigned long long *span) {
  const unsigned long long begin = clock64();
  unsigned long long end;
  do {
    end = clock64();
  } while (end - begin < 10000);
  *span = end - begin;
}
__global__ void event_empty_kernel(unsigned long long *span) {
  span[blockIdx.x] = clock64();
}
}  // namespace

TEST(CudaEventTiming, RepeatedSmallLaunchRetainsConfiguredDelay) {
  const char *expected = std::getenv("FLASHGPU_TEST_LAUNCH_LATENCY");
  if (!expected) GTEST_SKIP() << "Set the simulator launch-delay expectation";
  const unsigned latency = std::strtoul(expected, nullptr, 10);
  ASSERT_GT(latency, 0u);
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  unsigned long long *span = nullptr;
  ASSERT_EQ(cudaMalloc(&span, 4 * sizeof(*span)), cudaSuccess);
  cudaEvent_t begin, end;
  ASSERT_EQ(cudaEventCreate(&begin), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
  // A one-CTA kernel leaves other SMs idle. Repeated launches must not reuse
  // a retired kernel binding and bypass the configured dispatch delay.
  for (int repeat = 0; repeat < 5; ++repeat) {
    void *args[] = {&span};
    ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                  (const void *)event_empty_kernel, dim3(4), dim3(32),
                  dim3(2, 1, 1), args), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(begin), cudaSuccess);
    event_empty_kernel<<<1, 1>>>(span);
    ASSERT_EQ(cudaEventRecord(end), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
    float ms = 0;
    ASSERT_EQ(cudaEventElapsedTime(&ms, begin, end), cudaSuccess);
    EXPECT_GE(ms * prop.clockRate, latency) << "launch " << repeat;
  }
  EXPECT_EQ(cudaEventDestroy(begin), cudaSuccess);
  EXPECT_EQ(cudaEventDestroy(end), cudaSuccess);
  EXPECT_EQ(cudaFree(span), cudaSuccess);
}

TEST(CudaEventTiming, DeviceCyclesAndRerecord) {
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  ASSERT_GT(prop.clockRate, 0);
  unsigned long long *device_span = nullptr;
  ASSERT_EQ(cudaMalloc(&device_span, sizeof(*device_span)), cudaSuccess);
  cudaEvent_t begin, end;
  ASSERT_EQ(cudaEventCreate(&begin), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_EQ(cudaEventRecord(begin), cudaSuccess);
    event_clock_span<<<1, 1>>>(device_span);
    ASSERT_EQ(cudaEventRecord(end), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
    float ms = 0;
    ASSERT_EQ(cudaEventElapsedTime(&ms, begin, end), cudaSuccess);
    unsigned long long span = 0;
    ASSERT_EQ(cudaMemcpy(&span, device_span, sizeof(span),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    const double event_cycles = ms * prop.clockRate;
    // Events include kernel setup/drain; host simulation time must not enter.
    EXPECT_GE(event_cycles, static_cast<double>(span));
    EXPECT_LT(event_cycles, static_cast<double>(span) + 10000);
  }
  EXPECT_EQ(cudaEventDestroy(begin), cudaSuccess);
  EXPECT_EQ(cudaEventDestroy(end), cudaSuccess);
  EXPECT_EQ(cudaFree(device_span), cudaSuccess);
}

TEST(CudaEventTiming, DirectRuntimeLaunchWithoutCompilerPush) {
  unsigned long long *span = nullptr;
  ASSERT_EQ(cudaMalloc(&span, 4 * sizeof(*span)), cudaSuccess);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void *args[] = {&span};
  ASSERT_EQ(cudaLaunchKernel((const void *)event_empty_kernel, dim3(4),
                            dim3(1), args, 0, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  unsigned long long values[4]{};
  ASSERT_EQ(cudaMemcpy(values, span, sizeof(values), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (auto value : values) EXPECT_GT(value, 0u);
  // A normal compiler-generated launch must still work after the direct API.
  event_empty_kernel<<<1, 1>>>(span);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  EXPECT_EQ(cudaFree(span), cudaSuccess);
}
