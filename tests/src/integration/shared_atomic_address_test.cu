#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

__global__ void wrapped_shared_atomic_address_kernel(uint32_t *output) {
  __shared__ volatile uint32_t value;
  value = 0;

  uint32_t old = 0;
  asm volatile(
      "{\n\t"
      ".reg .u32 wrapped_address;\n\t"
      "add.u32 wrapped_address, 0xffffffff, 1;\n\t"
      "atom.shared.add.u32 %0, [wrapped_address], 7;\n\t"
      "}"
      : "=r"(old)
      :
      : "memory");
  output[0] = value;
  output[1] = old;
}

TEST(SharedAtomicAddressIntegrationTest,
     WrappedU32AddressMatchesCanonicalSharedOffset) {
  uint32_t *device_output = nullptr;
  ASSERT_EQ(cudaMalloc(&device_output, 2 * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output, 0, 2 * sizeof(uint32_t)), cudaSuccess);

  wrapped_shared_atomic_address_kernel<<<1, 1>>>(device_output);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t output[2] = {};
  ASSERT_EQ(
      cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(output[0], 7U);
  EXPECT_EQ(output[1], 0U);

  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}

}  // namespace
