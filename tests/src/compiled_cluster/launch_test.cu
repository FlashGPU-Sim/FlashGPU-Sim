#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ __cluster_dims__(2, 1, 1) void compiled(unsigned *out) {
  if (threadIdx.x == 0) {
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    out[blockIdx.x * 2] = rank;
    out[blockIdx.x * 2 + 1] = blockIdx.x;
  }
}
void check(bool explicit_launch) {
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&out, 8 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemset(out, 0xa5, 8 * sizeof(unsigned)), cudaSuccess);
  if (explicit_launch) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(4);
    config.blockDim = dim3(32);
    config.attrs = &attribute;
    config.numAttrs = 1;
    void *args[] = {&out};
    ASSERT_EQ(cudaLaunchKernelExC(&config, (const void *)compiled, args),
              cudaSuccess);
  } else
    compiled<<<4, 32>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::array<unsigned, 8> values;
  ASSERT_EQ(
      cudaMemcpy(values.data(), out, sizeof(values), cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned cta = 0; cta < 4; ++cta) {
    EXPECT_EQ(values[cta * 2], cta % 2);
    EXPECT_EQ(values[cta * 2 + 1], cta);
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(CompiledCluster, DefaultLaunchUsesCubinRequirement) { check(false); }
TEST(CompiledCluster, MatchingExplicitLaunch) { check(true); }
TEST(CompiledCluster, RejectsGridNotMultipleOfRequirement) {
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&out, 8 * sizeof(unsigned)), cudaSuccess);
  compiled<<<3, 32>>>(out);
  EXPECT_NE(cudaGetLastError(), cudaSuccess);
  (void)cudaDeviceSynchronize();
  (void)cudaGetLastError();
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
