#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

namespace {
template <unsigned Scope, bool Sequential = false>
__device__ __forceinline__ void fence() {
  if constexpr (Sequential) {
    if constexpr (Scope == 0) asm volatile("fence.sc.cta;" ::: "memory");
    if constexpr (Scope == 1) asm volatile("fence.sc.cluster;" ::: "memory");
    if constexpr (Scope == 2) asm volatile("fence.sc.gpu;" ::: "memory");
    if constexpr (Scope == 3) asm volatile("fence.sc.sys;" ::: "memory");
  } else {
    if constexpr (Scope == 0) asm volatile("fence.acq_rel.cta;" ::: "memory");
    if constexpr (Scope == 1) asm volatile("fence.acq_rel.cluster;" ::: "memory");
    if constexpr (Scope == 2) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if constexpr (Scope == 3) asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
}

// CTA scope communicates across two warps. Wider scopes communicate across
// CTA pairs; the cluster variant explicitly launches two-CTA clusters.
template <unsigned Scope, bool Sequential = false>
__global__ void publish(const unsigned *input, unsigned *data,
                        volatile unsigned *flags, unsigned *output) {
  const unsigned group = Scope == 0 ? blockIdx.x : blockIdx.x / 2;
  const unsigned i = group * 32 + (threadIdx.x & 31);
  const bool producer = Scope == 0 ? threadIdx.x < 32 : (blockIdx.x & 1) == 0;
  if (producer) {
    data[i] = input[i] * 3 + 7;
    fence<Scope, Sequential>();
    flags[i] = 1;
  } else {
    unsigned ready = 0;
    // Bound failed progress instead of leaving a regression spinning forever.
    for (unsigned retry = 0; retry < 4096 && !ready; ++retry) ready = flags[i];
    fence<Scope, Sequential>();
    output[i] = ready ? data[i] : 0xbad00000u;
  }
}

template <unsigned Scope, bool Sequential = false>
void check() {
  unsigned input[64], result[64];
  for (unsigned i = 0; i < 64; ++i) input[i] = 12345u + 1664525u * i;
  unsigned *device;
  ASSERT_EQ(cudaMalloc(&device, 4 * sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(device + 64, 0, 3 * sizeof(input)), cudaSuccess);
  if constexpr (Scope == 1) {
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(4); launch.blockDim = dim3(32);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2; attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
    launch.attrs = &attr; launch.numAttrs = 1;
    ASSERT_EQ(cudaLaunchKernelEx(&launch, publish<Scope, Sequential>, device, device + 64,
                                static_cast<volatile unsigned *>(device + 128), device + 192), cudaSuccess);
  } else {
    publish<Scope, Sequential><<<Scope == 0 ? 2 : 4, Scope == 0 ? 64 : 32>>>(
        device, device + 64, device + 128, device + 192);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result, device + 192, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < 64; ++i) EXPECT_EQ(result[i], input[i] * 3 + 7) << i;
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
TEST(AcqRelFence, CtaPublication) { check<0>(); }
TEST(AcqRelFence, ClusterPublication) { check<1>(); }
TEST(AcqRelFence, GpuPublication) { check<2>(); }
TEST(AcqRelFence, SysPublication) { check<3>(); }

TEST(AcqRelFence, SequentialCtaPublication) { check<0, true>(); }
TEST(AcqRelFence, SequentialClusterPublication) { check<1, true>(); }
TEST(AcqRelFence, SequentialGpuPublication) { check<2, true>(); }
TEST(AcqRelFence, SequentialSysPublication) { check<3, true>(); }

__global__ void predicated(unsigned *out) {
  const unsigned tid = threadIdx.x;
  volatile unsigned *ptr = out + tid;
  *ptr = tid + 42;
  asm volatile("{ .reg .pred p; setp.ne.u32 p,%0,0; @p fence.acq_rel.cta; }"
               :: "r"(tid & 1) : "memory");
  *ptr += 1;
}
TEST(AcqRelFence, Predicated) {
  unsigned *device, output[32];
  ASSERT_EQ(cudaMalloc(&device, sizeof(output)), cudaSuccess);
  predicated<<<1,32>>>(device);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < 32; ++i) EXPECT_EQ(output[i], i + 43);
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
template <bool Wide>
__global__ void volatile_copy(const uint64_t *in, uint64_t *out) {
  const unsigned i = threadIdx.x;
  if constexpr (Wide) {
    uint64_t value;
    asm volatile("ld.volatile.global.u64 %0,[%1];" : "=l"(value) : "l"(in+i) : "memory");
    asm volatile("st.volatile.global.u64 [%0],%1;" :: "l"(out+i), "l"(value) : "memory");
  } else {
    unsigned value;
    asm volatile("ld.volatile.global.u32 %0,[%1];" : "=r"(value) : "l"(in+i) : "memory");
    asm volatile("st.volatile.global.u32 [%0],%1;" :: "l"(out+i), "r"(value) : "memory");
  }
}
template <bool Wide> void check_volatile() {
  uint64_t input[32], output[32], *device;
  for (unsigned i=0;i<32;++i) input[i]=0xfedcba9812345600ull+i;
  ASSERT_EQ(cudaMalloc(&device, 2*sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(device+32, 0xab, sizeof(input)), cudaSuccess);
  volatile_copy<Wide><<<1,32>>>(device, device+32);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output, device+32, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
  for(unsigned i=0;i<32;++i)
    EXPECT_EQ(output[i], Wide ? input[i] : 0xabababab00000000ull | uint32_t(input[i]));
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
TEST(AcqRelFence, VolatileU32) { check_volatile<false>(); }
TEST(AcqRelFence, VolatileU64) { check_volatile<true>(); }
} // namespace
