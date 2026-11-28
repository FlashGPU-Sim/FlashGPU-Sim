#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>

namespace {
// Two clusters, two phases, and both ranks. Exercise state-space and generic
// addresses through real memory operations, not just pointer-bit equality.
__global__ void shared_address_roundtrip(const unsigned *input, unsigned *out) {
  __shared__ __align__(16) unsigned tile[128];
  unsigned rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned lane = threadIdx.x;
  for (unsigned phase = 0; phase < 2; ++phase) {
    unsigned local = __cvta_generic_to_shared(tile + lane * 4);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(local),
                 "l"(input + (blockIdx.x * 2 + phase) * 128 + lane * 4)
                 : "memory");
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    asm volatile(
        "barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::
            : "memory");
    unsigned mapped32, from32, from64, from_generic, roundtrip;
    unsigned long long mapped64, generic, converted;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped32)
                 : "r"(local), "r"(rank ^ 1));
    asm volatile("mapa.shared::cluster.u64 %0, %1, %2;"
                 : "=l"(mapped64)
                 : "l"((unsigned long long)local), "r"(rank ^ 1));
    asm volatile("mapa.u64 %0, %1, %2;"
                 : "=l"(generic)
                 : "l"(tile + lane * 4), "r"(rank ^ 1));
    asm volatile("cvta.to.shared::cluster.u64 %0, %1;"
                 : "=l"(converted)
                 : "l"(generic));
    roundtrip = converted == mapped64;
    asm volatile("ld.shared::cluster.u32 %0, [%1];"
                 : "=r"(from32)
                 : "r"(mapped32)
                 : "memory");
    asm volatile("ld.shared::cluster.u32 %0, [%1];"
                 : "=r"(from64)
                 : "l"(mapped64)
                 : "memory");
    asm volatile("ld.u32 %0, [%1];"
                 : "=r"(from_generic)
                 : "l"(generic)
                 : "memory");
    const unsigned base = ((blockIdx.x * 2 + phase) * 32 + lane) * 6;
    out[base] = tile[lane * 4];  // local CTA read after cp.async
    out[base + 1] = from32;
    out[base + 2] = from64;
    out[base + 3] = from_generic;
    out[base + 4] = roundtrip;
    out[base + 5] = local >> 24;
    asm volatile(
        "barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::
            : "memory");
  }
}

TEST(SharedAddress, RankBitsMapWidthsGenericRoundtripAndLocalAsyncCopy) {
  unsigned *input, *out;
  std::array<unsigned, 1024> data;
  for (unsigned i = 0; i < data.size(); ++i) data[i] = 0x12340000 + i;
  ASSERT_EQ(cudaMalloc(&input, sizeof(data)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 4 * 2 * 32 * 6 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(input, data.data(), sizeof(data), cudaMemcpyHostToDevice),
      cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(4);
  config.blockDim = dim3(32);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2;
  attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  ASSERT_EQ(cudaLaunchKernelEx(&config, shared_address_roundtrip, input, out),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::array<unsigned, 1536> result{};
  ASSERT_EQ(
      cudaMemcpy(result.data(), out, sizeof(result), cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned cta = 0; cta < 4; ++cta)
    for (unsigned phase = 0; phase < 2; ++phase)
      for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned base = ((cta * 2 + phase) * 32 + lane) * 6;
        EXPECT_EQ(result[base], data[(cta * 2 + phase) * 128 + lane * 4]);
        for (unsigned form = 1; form <= 3; ++form)
          EXPECT_EQ(result[base + form],
                    data[((cta ^ 1) * 2 + phase) * 128 + lane * 4])
              << cta << "/" << phase << "/" << lane << "/" << form;
        EXPECT_EQ(result[base + 4], 1u);
        EXPECT_EQ(result[base + 5], cta % 2);
      }
  EXPECT_EQ(cudaFree(input), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
