#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kDevice = 0;
constexpr int kDefaultLimit = 49152;
constexpr int kOptinLimit = 101376;
constexpr int kPerSmLimit = 102400;
constexpr int kRequestedDynamicBytes = 64 * 1024;
constexpr uint32_t kSentinel = 0x5A17C0DEu;

__global__ void optin_shared_memory_kernel(uint32_t *output) {
  extern __shared__ uint32_t dynamic_shared[];
  if (threadIdx.x == 0) {
    dynamic_shared[0] = kSentinel;
    output[0] = dynamic_shared[0];
  }
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

TEST_F(SharedMemoryOptinTest, ReportsDistinctDeviceLimits) {
  int per_sm_limit = 0;
  ASSERT_EQ(
      cudaDeviceGetAttribute(
          &per_sm_limit, cudaDevAttrMaxSharedMemoryPerMultiprocessor, kDevice),
      cudaSuccess);

  EXPECT_EQ(default_limit_, kDefaultLimit);
  EXPECT_EQ(optin_limit_, kOptinLimit);
  EXPECT_EQ(per_sm_limit, kPerSmLimit);

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

TEST_F(SharedMemoryOptinTest, ExecutesAboveDefaultDynamicSharedMemoryLimit) {
  ASSERT_GT(kRequestedDynamicBytes, default_limit_);
  ASSERT_LE(kRequestedDynamicBytes, optin_limit_);

  ASSERT_EQ(cudaFuncSetAttribute(optin_shared_memory_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kRequestedDynamicBytes),
            cudaSuccess);

  ASSERT_EQ(cudaMalloc(&device_output_, sizeof(*device_output_)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output_, 0, sizeof(*device_output_)),
            cudaSuccess);

  optin_shared_memory_kernel<<<1, 1, kRequestedDynamicBytes>>>(device_output_);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, device_output_, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kSentinel);

  ASSERT_LT(optin_limit_, std::numeric_limits<int>::max());
  EXPECT_EQ(cudaFuncSetAttribute(optin_shared_memory_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 optin_limit_ + 1),
            cudaErrorInvalidValue);
}

}  // namespace
