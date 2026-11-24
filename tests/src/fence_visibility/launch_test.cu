#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace {
// A timing regression, not a hardware latency measurement. The test compares
// the observed fence span to the visibility delay from its actual config.
template <bool Sequential>
__global__ void store_fence(uint64_t *elapsed, unsigned *values) {
  __shared__ unsigned data[32];
  const unsigned lane = threadIdx.x;
  const unsigned addr = unsigned(__cvta_generic_to_shared(data + lane));
  uint64_t begin, end;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) :: "memory");
  asm volatile("st.shared.u32 [%0],%1;" :: "r"(addr), "r"(lane + 42) : "memory");
  if constexpr (Sequential)
    asm volatile("fence.sc.cta;" ::: "memory");
  else
    asm volatile("fence.cta;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");
  elapsed[lane] = end - begin;
  values[lane] = data[lane];
}

template <bool Sequential>
void check() {
  unsigned delay = 0;
  std::ifstream config("gpgpusim.config");
  for (std::string line; std::getline(config, line);) {
    std::istringstream fields(line);
    std::string option;
    fields >> option;
    if (option == "-gpgpu_smem_store_visibility_latency") fields >> delay;
  }
  if (!delay) GTEST_SKIP() << "Requires simulator shared-store visibility config";
  uint64_t *elapsed;
  unsigned *values;
  ASSERT_EQ(cudaMalloc(&elapsed, 32 * sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&values, 32 * sizeof(unsigned)), cudaSuccess);
  store_fence<Sequential><<<1,32>>>(elapsed, values);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint64_t clocks[32]; unsigned output[32];
  ASSERT_EQ(cudaMemcpy(clocks, elapsed, sizeof(clocks), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output, values, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(output[lane], lane + 42);
    EXPECT_GE(clocks[lane], delay) << "fence passed before shared-store visibility";
  }
  EXPECT_EQ(cudaFree(elapsed), cudaSuccess);
  EXPECT_EQ(cudaFree(values), cudaSuccess);
}
TEST(SharedFenceVisibility, Default) { check<false>(); }
TEST(SharedFenceVisibility, Sequential) { check<true>(); }
} // namespace
