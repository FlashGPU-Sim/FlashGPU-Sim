#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
template <bool Mixed, unsigned Compact = 0, unsigned Issuer = 0>
__global__ void mma_commit(float *out) {
  __shared__ __align__(16) unsigned short a[128 * 16], b[8 * 16];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned tmem;
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < 128 * 16; i += 128)
    a[i] = Mixed ? 0x2222 : 0x4000;                                     // 2
  if (t < 8 * 16) b[t] = Compact ? 0x4444 : (Mixed ? 0x3838 : 0xbc00);  // -1
  const unsigned ba = (unsigned)__cvta_generic_to_shared(&barrier);
  if (t == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(ba) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&tmem))
        : "memory");
  __syncthreads();
  if constexpr (Mixed) {
    const unsigned address = tmem + ((t / 32) << 21);
    const unsigned scale = Compact ? 0x7f807d7f : 0x7f7f7f7f;
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%1,%1,%1};" ::"r"(
            address + 16),
        "r"(scale)
        : "memory");
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%1,%1,%1};" ::"r"(
            address + 20),
        "r"(scale)
        : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (t == Issuer) {
    // Canonical interleaved K-major, 16-byte atoms, eight rows per atom.
    const unsigned long long ad = ((unsigned)__cvta_generic_to_shared(a) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    const unsigned long long bd = ((unsigned)__cvta_generic_to_shared(b) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    if constexpr (Compact) {
      const unsigned id = Compact == 2 ? 2 : 0;
      const unsigned desc = (1u << 27) | (1u << 23) | (1u << 17) | (1u << 7) |
                            (1u << 10) | (id << 29) | (id << 4);
      asm volatile(
          "{ .reg .pred p; mov.pred p,0; "
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
          "[%0],%1,%2,%3,[%4],[%5],p; "
          "mov.pred p,1; "
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
          "[%0],%1,%2,%3,[%4],[%5],p; }" ::"r"(tmem),
          "l"(ad), "l"(bd), "r"(desc), "r"(tmem + 16), "r"(tmem + 20)
          : "memory");
    } else if constexpr (Mixed) {
      const unsigned desc = (1u << 27) | (1u << 23) | (1u << 17) | (5u << 7);
      asm volatile(
          "{ .reg .pred p; mov.pred p,0; "
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
          "[%0],%1,%2,%3,[%4],[%5],p; }" ::"r"(tmem),
          "l"(ad), "l"(bd), "r"(desc), "r"(tmem + 16), "r"(tmem + 20)
          : "memory");
    } else {
      const unsigned desc = (8u << 24) | (1u << 17) | 0x10;
      asm volatile(
          "{ .reg .pred p; mov.pred p,0; tcgen05.mma.cta_group::1.kind::f16 "
          "[%0],%1,%2,%3,p; }" ::"r"(tmem),
          "l"(ad), "l"(bd), "r"(desc)
          : "memory");
    }
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(ba)
        : "memory");
  }
  asm volatile(
      "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
      "p,[%0],0; @!p bra wait; }" ::"r"(ba)
      : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  unsigned v[8];
  const unsigned address = tmem + ((t / 32) << 21);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]),
        "=r"(v[6]), "=r"(v[7])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (unsigned i = 0; i < 8; ++i) out[t * 8 + i] = __uint_as_float(v[i]);
  __syncthreads();
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,32;" ::"r"(tmem)
        : "memory");
  }
}
}  // namespace
TEST(Tcgen05Mma, SharedOperandsCommitAndWarpPartitionLoad) {
  float *device = nullptr;
  std::vector<float> result(128 * 8);
  ASSERT_EQ(cudaMalloc(&device, result.size() * sizeof(float)), cudaSuccess);
  mma_commit<false><<<1, 128>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < result.size(); ++i)
    ASSERT_EQ(result[i], -32.0f) << i;
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}
TEST(Tcgen05Mma, NonzeroIssuingLane) {
  float *device = nullptr;
  std::vector<float> result(128 * 8);
  ASSERT_EQ(cudaMalloc(&device, result.size() * sizeof(float)), cudaSuccess);
  mma_commit<false, 0, 7><<<1, 128>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < result.size(); ++i)
    ASSERT_EQ(result[i], -32.0f) << i;
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}
TEST(Tcgen05Mma, BlockScaledFp4Fp8ThroughSharedBackend) {
  float *device = nullptr;
  std::vector<float> result(128 * 8);
  ASSERT_EQ(cudaMalloc(&device, result.size() * sizeof(float)), cudaSuccess);
  mma_commit<true><<<1, 128>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < result.size(); ++i) ASSERT_EQ(result[i], 32.0f) << i;
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}

TEST(Tcgen05Mma, CompactFp4Block32ScaleSelectorsAndAccumulate) {
  float *device = nullptr;
  std::vector<float> result(128 * 8);
  ASSERT_EQ(cudaMalloc(&device, result.size() * sizeof(float)), cudaSuccess);
  for (unsigned id : {0u, 2u}) {
    if (id == 0)
      mma_commit<true, 1><<<1, 128>>>(device);
    else
      mma_commit<true, 2><<<1, 128>>>(device);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    // A=1, B=2. Per-32 scales are {1, 1/4} or {2, 1}; two accumulated MMAs.
    for (unsigned i = 0; i < result.size(); ++i)
      ASSERT_EQ(result[i], id ? 640.0f : 136.0f) << id << ':' << i;
  }
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}

namespace {
template <unsigned M>
__global__ void tf32_tmem_mma(float *out) {
  __shared__ __align__(16) float b[64];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned tmem;
  const unsigned t = threadIdx.x, warp = t / 32, lane = t % 32;
  for (unsigned i = t; i < 64; i += 128) {
    unsigned col = i / 8, k = i % 8;
    b[col * 4 + (k / 4) * 32 + k % 4] = float(int(col) - int(k));
  }
  unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (!t)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&tmem))
        : "memory");
  __syncthreads();
  // All four warps deliberately use the same TMEM base, as real mHC does.
  unsigned address = tmem, row = M == 64 ? warp * 16 + lane : t;
  unsigned values[8];
  for (unsigned k = 0; k < 8; ++k)
    values[k] = __float_as_uint(float(row + k + 1)) | 0x123u;
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" ::
          "r"(address + 16),
      "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]),
      "r"(values[4]), "r"(values[5]), "r"(values[6]), "r"(values[7])
      : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  for (unsigned round = 0; round < 2; ++round) {
    if (!t) {
      const unsigned long long bd =
          ((unsigned)__cvta_generic_to_shared(b) >> 4) | (8ull << 16) |
          (16ull << 32) | (1ull << 46);
      const unsigned idesc =
          ((M / 16) << 24) | (1u << 17) | 0x10 | (2u << 7) | (2u << 10);
      asm volatile(
          "{ .reg .pred p; setp.ne.u32 p,%4,0; "
          "tcgen05.mma.cta_group::1.kind::tf32 [%0],[%1],%2,%3,p; }" ::"r"(
              tmem),
          "r"(tmem + 16), "l"(bd), "r"(idesc), "r"(round)
          : "memory");
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(
              bar)
          : "memory");
    }
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],%1; @!p bra wait; }" ::"r"(bar),
        "r"(round)
        : "memory");
    __syncthreads();
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
        "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
      : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  if (M == 128 || lane < 16)
    for (unsigned col = 0; col < 8; ++col)
      out[row * 8 + col] = __uint_as_float(values[col]);
  __syncthreads();
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,32;" ::"r"(tmem)
        : "memory");
  }
}
TEST(Tcgen05Mma, Tf32TmemOperandsAndPartitionLayout) {
  float *dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, 128 * 8 * 4), cudaSuccess);
  for (unsigned m : {64u, 128u}) {
    if (m == 64)
      tf32_tmem_mma<64><<<1, 128>>>(dst);
    else
      tf32_tmem_mma<128><<<1, 128>>>(dst);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> out(m * 8);
    ASSERT_EQ(
        cudaMemcpy(out.data(), dst, out.size() * 4, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (unsigned row = 0; row < m; ++row)
      for (unsigned col = 0; col < 8; ++col) {
        float expected = 0;
        for (unsigned k = 0; k < 8; ++k)
          expected += float(row + k + 1) * float(int(col) - int(k));
        EXPECT_EQ(out[row * 8 + col], 2 * expected)
            << m << "," << row << "," << col;
      }
  }
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}

}  // namespace
