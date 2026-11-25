#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

template <unsigned M, bool Bf16, bool Masked = false>
__global__ __cluster_dims__(2, 1, 1) void paired_dense_mma(float *out, unsigned lane_mask = 0) {
  __shared__ __align__(16) unsigned short a[M / 2 * 16], b[8 * 16];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned tmem;
  const unsigned t = threadIdx.x, rank = blockIdx.x % 2;
  for (unsigned i = t; i < M / 2 * 16; i += 128) {
    const unsigned row = i / 16, k = i % 16;
    const float value = float((rank + 1) * (row % 4 + 1));
    a[(row / 8) * 128 + (k / 8) * 64 + (row % 8) * 8 + k % 8] =
        Bf16 ? __float_as_uint(value) >> 16
             : __half_as_ushort(__float2half_rn(value));
  }
  if (t < 128) {
    const unsigned row = t / 16, k = t % 16;
    const float value = float((rank + 1) * (k % 4 + 1));
    b[(k / 8) * 64 + row * 8 + k % 8] =
        Bf16 ? __float_as_uint(value) >> 16
             : __half_as_ushort(__float2half_rn(value));
  }
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (!t)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&tmem))
        : "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile(
      "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
          : "memory");
  for (unsigned round = 0; round < 2; ++round) {
    if (!t && !rank) {
      const unsigned long long ad =
          ((unsigned)__cvta_generic_to_shared(a) >> 4) | (8ull << 16) |
          (16ull << 32) | (1ull << 46);
      const unsigned long long bd =
          ((unsigned)__cvta_generic_to_shared(b) >> 4) | (8ull << 16) |
          (16ull << 32) | (1ull << 46);
      const unsigned desc = ((M / 16) << 24) | (2u << 17) | 0x10 |
                            (Bf16 ? ((1u << 7) | (1u << 10)) : 0);
      if (round == 0) {
        asm volatile(
            "{ .reg .pred p; mov.pred p,0; tcgen05.mma.cta_group::2.kind::f16 "
            "[%0],%1,%2,%3,p; }" ::"r"(tmem),
            "l"(ad), "l"(bd), "r"(desc)
            : "memory");
      } else if constexpr (Masked) {
        // Different words in the two CTA halves catch accidental rank aliasing.
        asm volatile(
            "{ .reg .pred p; mov.pred p,1; tcgen05.mma.cta_group::2.kind::f16 "
            "[%0],%1,%2,%3,{%4,%5,%6,%7,%7,%6,%5,%4},p; }"
            :: "r"(tmem), "l"(ad), "l"(bd), "r"(desc),
               "r"(lane_mask), "r"(lane_mask << 1),
               "r"(lane_mask << 2), "r"(lane_mask << 3) : "memory");
      } else {
        // DeepInfer attention spells the optional pair mask as eight words.
        asm volatile(
            "{ .reg .pred p; .reg .b32 z; mov.pred p,1; mov.b32 z,0; "
            "tcgen05.mma.cta_group::2.kind::f16 "
            "[%0],%1,%2,%3,{z,z,z,z,z,z,z,z},p; }" ::"r"(tmem),
            "l"(ad), "l"(bd), "r"(desc)
            : "memory");
      }
      asm volatile(
          "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
          "multicast::cluster.b64 [%0],%1;" ::"r"(bar),
          "h"((unsigned short)3)
          : "memory");
    }
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],%1; @!p bra wait; }" ::"r"(bar),
        "r"(round)
        : "memory");
    __syncthreads();
    asm volatile(
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
            : "memory");
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned half = 0; half < (M == 128 ? 1 : 2); ++half) {
    unsigned v[8];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]),
          "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
        : "r"(tmem + half * 8));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    const unsigned row = blockIdx.x * (M / 2) + (M == 128 ? t % 64 : t),
                   col = M == 128 ? (t / 64) * 8 : half * 8;
    for (unsigned i = 0; i < 8; ++i)
      out[row * 16 + col + i] = __uint_as_float(v[i]);
  }
  __syncthreads();
  asm volatile(
      "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
          : "memory");
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,32;" ::"r"(tmem)
        : "memory");
  }
}
TEST(Tcgen05Pair, DenseHalfAndBfloatFoldedAndFullLayouts) {
  float *dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, 512 * 16 * 4), cudaSuccess);
  for (unsigned m : {128u, 256u})
    for (bool bf : {false, true}) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(4, 1, 1);
      cfg.blockDim = dim3(128, 1, 1);
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim = {2, 1, 1};
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      if (m == 128 && bf)
        ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<128, true>, dst, 0u),
                  cudaSuccess);
      if (m == 128 && !bf)
        ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<128, false>, dst, 0u),
                  cudaSuccess);
      if (m == 256 && bf)
        ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<256, true>, dst, 0u),
                  cudaSuccess);
      if (m == 256 && !bf)
        ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<256, false>, dst, 0u),
                  cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<float> values(2 * m * 16);
      ASSERT_EQ(cudaMemcpy(values.data(), dst, values.size() * 4,
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (unsigned row = 0; row < 2 * m; ++row)
        for (unsigned col = 0; col < 16; ++col)
          EXPECT_EQ(values[row * 16 + col], 80.0f * (row / (m / 2) % 2 + 1) *
                                                (row % 4 + 1) * (col / 8 + 1))
              << m << "," << bf << "," << row << "," << col;
    }
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}

TEST(Tcgen05Pair, OutputMasksUseBothCtasPhysicalLaneGroups) {
  float *dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, 256 * 16 * 4), cudaSuccess);
  for (unsigned m : {128u, 256u}) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    if (m == 128)
      ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<128, true, true>, dst, 1u), cudaSuccess);
    else
      ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_dense_mma<256, false, true>, dst, 1u), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> values(m * 16);
    ASSERT_EQ(cudaMemcpy(values.data(), dst, values.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    for (unsigned row = 0; row < m; ++row)
      for (unsigned col = 0; col < 16; ++col) {
        const unsigned rank = row / (m / 2);
        const unsigned lane = m == 128 ? row % 64 + (col / 8) * 64 : row % 128;
        const unsigned mask_bit = rank ? 3 - lane / 32 : lane / 32;
        const bool masked = lane % 32 == mask_bit;
        const float dot = 40.0f * (rank + 1) * (row % 4 + 1) * (col / 8 + 1);
        EXPECT_EQ(values[row * 16 + col], (masked ? 1 : 2) * dot) << m << ':' << row << ':' << col;
      }
  }
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}

// Two-CTA TMEM-A regression: nonzero A base, distinct CTA/cluster/half data.
template <unsigned M, unsigned N, bool Bf16>
__global__ __cluster_dims__(2, 1, 1) void paired_tmem_a_mma(float *out) {
  __shared__ __align__(16) unsigned short b[N / 2 * 16];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned tmem;
  const unsigned t = threadIdx.x, rank = blockIdx.x % 2;
  for (unsigned i = t; i < N / 2 * 16; i += 128) {
    const unsigned row = i / 16, k = i % 16;
    const float value = float((rank + 1) * (k % 4 + 1));
    b[(row / 8) * 128 + (k / 8) * 64 + (row % 8) * 8 + k % 8] =
        Bf16 ? __float_as_uint(value) >> 16
             : __half_as_ushort(__float2half_rn(value));
  }
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (!t)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],128;" ::
            "r"((unsigned)__cvta_generic_to_shared(&tmem))
        : "memory");
  __syncthreads();
  // Preserve distinct upper/lower A images for Layout B (FlashMLA dual GEMM).
  const float av = float((rank + 1) * (t % 4 + 1) * (blockIdx.x / 2 + 1) *
                         (M == 128 ? t / 64 + 1 : 1));
  const unsigned short bits =
      Bf16 ? __float_as_uint(av) >> 16 : __half_as_ushort(__float2half_rn(av));
  const unsigned packed = unsigned(bits) | (unsigned(bits) << 16);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%1,%1,%1,%1,%1,%1,%1};" ::
          "r"(tmem + 64),
      "r"(packed)
      : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile(
      "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
          : "memory");
  for (unsigned round = 0; round < 2; ++round) {
    if (!t && !rank) {
      const unsigned long long bd =
          ((unsigned)__cvta_generic_to_shared(b) >> 4) | (8ull << 16) |
          (16ull << 32) | (1ull << 46);
      const unsigned desc = ((M / 16) << 24) | ((N / 8) << 17) | 0x10 |
                            (Bf16 ? ((1u << 7) | (1u << 10)) : 0);
      if (round == 0) {
        asm volatile(
            "{ .reg .pred p; mov.pred p,0; tcgen05.mma.cta_group::2.kind::f16 "
            "[%0],[%1],%2,%3,p; }" ::"r"(tmem),
            "r"(tmem + 64), "l"(bd), "r"(desc)
            : "memory");
      } else {
        // DeepInfer attention spells the optional pair mask as eight words.
        asm volatile(
            "{ .reg .pred p; .reg .b32 z; mov.pred p,1; mov.b32 z,0; "
            "tcgen05.mma.cta_group::2.kind::f16 "
            "[%0],[%1],%2,%3,{z,z,z,z,z,z,z,z},p; }" ::"r"(tmem),
            "r"(tmem + 64), "l"(bd), "r"(desc)
            : "memory");
      }
      asm volatile(
          "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
          "multicast::cluster.b64 [%0],%1;" ::"r"(bar),
          "h"((unsigned short)3)
          : "memory");
    }
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],%1; @!p bra wait; }" ::"r"(bar),
        "r"(round)
        : "memory");
    __syncthreads();
    asm volatile(
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
            : "memory");
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned half = 0; half < (M == 128 ? N / 16 : N / 8); ++half) {
    unsigned v[8];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]),
          "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
        : "r"(tmem + half * 8));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    const unsigned row = blockIdx.x * (M / 2) + (M == 128 ? t % 64 : t),
                   col = M == 128 ? (t / 64) * (N / 2) + half * 8 : half * 8;
    for (unsigned i = 0; i < 8; ++i)
      out[row * N + col + i] = __uint_as_float(v[i]);
  }
  __syncthreads();
  asm volatile(
      "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
          : "memory");
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,128;" ::"r"(tmem)
        : "memory");
  }
}
template <unsigned M, unsigned N, bool Bf16>
void check_paired_tmem_a() {
  float *dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, 2 * M * N * 4), cudaSuccess);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(4, 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  ASSERT_EQ(cudaLaunchKernelEx(&cfg, paired_tmem_a_mma<M, N, Bf16>, dst),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> values(2 * M * N);
  ASSERT_EQ(
      cudaMemcpy(values.data(), dst, values.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned row = 0; row < 2 * M; ++row)
    for (unsigned col = 0; col < N; ++col) {
      const float expected = 80.0f * (row / (M / 2) % 2 + 1) * (row % 4 + 1) *
                             (row / M + 1) * (col / (N / 2) + 1) *
                             (M == 128 ? col / (N / 2) + 1 : 1);
      EXPECT_EQ(values[row * N + col], expected)
          << M << "," << N << "," << Bf16 << "," << row << "," << col;
    }
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}
TEST(Tcgen05Pair, DenseTmemAHalfAndBfloatPhysicalImages) {
  check_paired_tmem_a<128, 32, false>();
  check_paired_tmem_a<256, 32, false>();
  check_paired_tmem_a<128, 32, true>();
  check_paired_tmem_a<256, 32, true>();
  check_paired_tmem_a<128, 64, false>();
  check_paired_tmem_a<256, 64, false>();
  check_paired_tmem_a<128, 64, true>();
  check_paired_tmem_a<256, 64, true>();
}

template <bool ExplicitStride = false>
__global__ void __cluster_dims__(2, 1, 1) paired_mma_commit(float *out) {
  __shared__ __align__(16) unsigned short a[128 * 16], b[8 * 16];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned tmem;
  __shared__ __align__(16) unsigned scales_a[128], scales_b[128];
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < 128 * 16; i += 128)
    a[i] = blockIdx.x / 2
               ? (blockIdx.x % 2 ? 0x7777 : 0x6666)
               : (blockIdx.x % 2 ? 0x4444
                                 : 0x2222);  // FP4 1/2 in pair 0, 4/6 in pair 1
  if (t < 8 * 16) b[t] = (blockIdx.x % 2) ? 0x4040 : 0x3838;  // FP8 1/2
  const unsigned ba = (unsigned)__cvta_generic_to_shared(&barrier);
  if (t == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(ba) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&tmem))
        : "memory");
  __syncthreads();
  scales_a[t] = blockIdx.x % 2 ? 0x80808080 : 0x7f7f7f7f;
  // Each CTA replicates the full SFB image, including both N halves.
  scales_b[t] = (t / 4 >= 8 && t / 4 < 16) ? 0x80808080 : 0x7f7f7f7f;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (t == 0 && blockIdx.x % 2 == 0) {
    // Canonical interleaved K-major, 16-byte atoms, eight rows per atom.
    const unsigned long long ad = ((unsigned)__cvta_generic_to_shared(a) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    const unsigned long long bd = ((unsigned)__cvta_generic_to_shared(b) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    const unsigned long long sd =
        ((unsigned)__cvta_generic_to_shared(scales_a) >> 4) | (1ull << 46) |
        (ExplicitStride ? 8ull << 32 : 0);
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0],%1;" ::"r"(tmem + 16),
        "l"(sd)
        : "memory");
    const unsigned long long sbd =
        ((unsigned)__cvta_generic_to_shared(scales_b) >> 4) | (1ull << 46) |
        (ExplicitStride ? 8ull << 32 : 0);
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0],%1;" ::"r"(tmem + 20),
        "l"(sbd)
        : "memory");
    const unsigned desc = (1u << 28) | (1u << 23) | (2u << 17) | (5u << 7);
    asm volatile(
        "{ .reg .pred p; mov.pred p,0; "
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale "
        "[%0],%1,%2,%3,[%4],[%5],p; }" ::"r"(tmem),
        "l"(ad), "l"(bd), "r"(desc), "r"(tmem + 16), "r"(tmem + 20)
        : "memory");
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
        "multicast::cluster.b64 [%0],%1;" ::"r"(ba),
        "h"((unsigned short)3)
        : "memory");
  }
  asm volatile(
      "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
      "p,[%0],0; @!p bra wait; }" ::"r"(ba)
      : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned half = 0; half < 2; ++half) {
    unsigned v[8];
    const unsigned address = tmem + ((t / 32) << 21) + half * 8;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]),
          "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
        : "r"(address));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    for (unsigned i = 0; i < 8; ++i)
      out[(blockIdx.x * 128 + t) * 16 + half * 8 + i] = __uint_as_float(v[i]);
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,32;" ::"r"(tmem)
        : "memory");
  }
}
template <bool ExplicitStride = false>
void check_paired_mma(unsigned blocks) {
  float *device = nullptr;
  std::vector<float> result(blocks * 128 * 16);
  ASSERT_EQ(cudaMalloc(&device, result.size() * sizeof(float)), cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks, 1, 1);
  config.blockDim = dim3(128, 1, 1);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;
  ASSERT_EQ(cudaLaunchKernelEx(&config, paired_mma_commit<ExplicitStride>, device),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned row = 0; row < blocks * 128; ++row)
    for (unsigned col = 0; col < 16; ++col)
      ASSERT_EQ(
          result[row * 16 + col],
          32.0f *
              (row / 256 ? (row / 128 % 2 ? 12 : 4) : (row / 128 % 2 ? 4 : 1)) *
              (col < 8 ? 1 : 4))
          << row << "," << col;
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}

TEST(Tcgen05Pair, BlockScaledFp4Fp8MulticastCommit) { check_paired_mma(2); }
TEST(Tcgen05Pair, MulticastAndTmemStayWithinEachCluster) {
  check_paired_mma(4);
}

TEST(Tcgen05Pair, ExplicitContiguousScaleStride) { check_paired_mma<true>(4); }
