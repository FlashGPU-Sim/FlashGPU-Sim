#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace dynamic_shared_test {
__global__ void aligned_dynamic(unsigned *out, unsigned bytes) {
  extern __shared__ __align__(1024) unsigned char arena[];
  __shared__ volatile unsigned prefix;
  if (threadIdx.x == 0) prefix = 0x12345678u;
  if (threadIdx.x < bytes) arena[threadIdx.x] = (threadIdx.x * 13 + 7) & 255;
  __syncthreads();
  if (threadIdx.x == 1) {
    out[0] = __cvta_generic_to_shared(arena);
    out[1] = prefix;
    out[2] = bytes ? arena[0] : 0;
    out[3] = bytes ? arena[bytes - 1] : 0;
  }
}
__global__ void ordinary_dynamic(unsigned *out) {
  extern __shared__ __align__(16) unsigned char ordinary[];
  __shared__ volatile unsigned prefix;
  if (threadIdx.x == 0) prefix = 99;
  __syncthreads();
  if (threadIdx.x == 1) {
    out[0] = __cvta_generic_to_shared(ordinary);
    out[1] = prefix;
  }
}
TEST(DynamicSharedAlignment, DeclaredAlignmentAcrossLaunchSizes) {
  unsigned *device;
  ASSERT_EQ(cudaMalloc(&device, 4 * sizeof(unsigned)), cudaSuccess);
  // A static prefix forces nonzero padding. Repeat a launch after a larger
  // allocation to catch stale address updates; zero size must still align.
  for (unsigned bytes : {0u, 32u, 256u, 32u, 0u}) {
    SCOPED_TRACE(bytes);
    aligned_dynamic<<<1, 256, bytes>>>(device, bytes);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    unsigned host[4];
    ASSERT_EQ(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host[0] % 1024, 0u);
    EXPECT_EQ(host[1], 0x12345678u);
    EXPECT_EQ(host[2], bytes ? 7u : 0u);
    EXPECT_EQ(host[3], bytes ? ((bytes - 1) * 13 + 7) & 255 : 0u);
  }
  ordinary_dynamic<<<1, 32, 32>>>(device);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned host[4];
  ASSERT_EQ(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(host[0] % 16, 0u);
  EXPECT_EQ(host[1], 99u);
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
}  // namespace dynamic_shared_test
