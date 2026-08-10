#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kDevice = 0;
constexpr uint8_t kFirstSentinel = 0x5Au;
constexpr uint8_t kLastSentinel = 0xA5u;
constexpr uint32_t kExpectedSentinels =
    (static_cast<uint32_t>(kFirstSentinel) << 8) | kLastSentinel;

__device__ void touch_dynamic_shared_memory(uint8_t *dynamic_shared,
                                            int dynamic_bytes,
                                            uint32_t *output) {
  if (threadIdx.x == 0) {
    dynamic_shared[0] = kFirstSentinel;
    dynamic_shared[dynamic_bytes - 1] = kLastSentinel;
    output[0] = (static_cast<uint32_t>(dynamic_shared[0]) << 8) |
                dynamic_shared[dynamic_bytes - 1];
  }
}

// Keep separate kernel symbols because cudaFuncSetAttribute changes persistent
// per-function state in the current CUDA context.
__global__ void no_optin_shared_memory_kernel(int dynamic_bytes,
                                              uint32_t *output) {
  extern __shared__ uint8_t dynamic_shared[];
  touch_dynamic_shared_memory(dynamic_shared, dynamic_bytes, output);
}

__global__ void optin_shared_memory_kernel(int dynamic_bytes,
                                           uint32_t *output) {
  extern __shared__ uint8_t dynamic_shared[];
  touch_dynamic_shared_memory(dynamic_shared, dynamic_bytes, output);
}

class SharedMemoryOptinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaSetDevice(kDevice), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(
                  &default_limit_, cudaDevAttrMaxSharedMemoryPerBlock, kDevice),
              cudaSuccess);
    ASSERT_EQ(
        cudaDeviceGetAttribute(
            &optin_limit_, cudaDevAttrMaxSharedMemoryPerBlockOptin, kDevice),
        cudaSuccess);
  }

  void TearDown() override {
    if (device_output_ != nullptr) {
      EXPECT_EQ(cudaFree(device_output_), cudaSuccess);
    }
  }

  int default_limit_ = 0;
  int optin_limit_ = 0;
  uint32_t *device_output_ = nullptr;
};

TEST_F(SharedMemoryOptinTest, ReportsConsistentDeviceLimits) {
  int per_sm_limit = 0;
  ASSERT_EQ(
      cudaDeviceGetAttribute(
          &per_sm_limit, cudaDevAttrMaxSharedMemoryPerMultiprocessor, kDevice),
      cudaSuccess);

  ASSERT_GT(default_limit_, 0);
  ASSERT_GT(optin_limit_, default_limit_);
  ASSERT_GE(per_sm_limit, optin_limit_);

  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, kDevice), cudaSuccess);
  EXPECT_EQ(properties.sharedMemPerBlock, static_cast<size_t>(default_limit_));
  EXPECT_EQ(properties.sharedMemPerBlockOptin,
            static_cast<size_t>(optin_limit_));
  EXPECT_EQ(properties.sharedMemPerMultiprocessor,
            static_cast<size_t>(per_sm_limit));

  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  CUdevice driver_device = -1;
  ASSERT_EQ(cuDeviceGet(&driver_device, kDevice), CUDA_SUCCESS);

  int driver_default_limit = 0;
  int driver_optin_limit = 0;
  int driver_per_sm_limit = 0;
  ASSERT_EQ(cuDeviceGetAttribute(
                &driver_default_limit,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, driver_device),
            CUDA_SUCCESS);
  ASSERT_EQ(
      cuDeviceGetAttribute(
          &driver_optin_limit,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, driver_device),
      CUDA_SUCCESS);
  ASSERT_EQ(cuDeviceGetAttribute(
                &driver_per_sm_limit,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                driver_device),
            CUDA_SUCCESS);
  EXPECT_EQ(driver_default_limit, default_limit_);
  EXPECT_EQ(driver_optin_limit, optin_limit_);
  EXPECT_EQ(driver_per_sm_limit, per_sm_limit);
}

TEST_F(SharedMemoryOptinTest, RequiresOptInAboveDefaultLimit) {
  ASSERT_GT(optin_limit_, default_limit_);
  const int difference = optin_limit_ - default_limit_;
  const int requested_dynamic_bytes =
      default_limit_ + difference / 2 + difference % 2;
  ASSERT_GT(requested_dynamic_bytes, default_limit_);
  ASSERT_LE(requested_dynamic_bytes, optin_limit_);

  ASSERT_EQ(cudaMalloc(&device_output_, sizeof(*device_output_)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output_, 0, sizeof(*device_output_)),
            cudaSuccess);

  no_optin_shared_memory_kernel<<<1, 1, requested_dynamic_bytes>>>(
      requested_dynamic_bytes, device_output_);
  ASSERT_NE(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaFuncSetAttribute(optin_shared_memory_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 requested_dynamic_bytes),
            cudaSuccess);

  optin_shared_memory_kernel<<<1, 1, requested_dynamic_bytes>>>(
      requested_dynamic_bytes, device_output_);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, device_output_, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpectedSentinels);
}

TEST_F(SharedMemoryOptinTest, RejectsAttributeAboveOptInLimit) {
  ASSERT_LT(optin_limit_, std::numeric_limits<int>::max());
  EXPECT_EQ(cudaFuncSetAttribute(optin_shared_memory_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 optin_limit_ + 1),
            cudaErrorInvalidValue);
  EXPECT_EQ(cudaGetLastError(), cudaErrorInvalidValue);
  EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

}  // namespace
