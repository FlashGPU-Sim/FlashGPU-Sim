#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <vector>
#include "../tma/cp_kernels.cuh"
#include "common/gpgpusim_config_topology.h"
namespace {
// Two clusters and two barrier phases. Modes select multicast, paired
// completion, remote destination, and shared::cta with local completion.
template <int Mode, int Rank = 2, bool LongSuspend = false>
__global__ void tensor_cluster(const CUtensorMap *map, unsigned *out) {
  __shared__ __align__(128) unsigned tile[128];
  __shared__ __align__(8) unsigned long long barrier;
  constexpr bool Paired = Mode == 1 || Mode == 3 || Mode >= 6;
  constexpr bool Multicast = Mode < 2 || Mode == 5;
  constexpr unsigned Mask = Mode == 5 ? 2 : 3;
  constexpr unsigned Destination = Mode == 4 || Mode >= 6 ? 1 : 0;
  constexpr unsigned Owner =
      Mode == 8 ? 0 : (Paired || Mode == 4 || Mode >= 6 ? 1 : 0);
  constexpr unsigned Issuer = Mode >= 2 ? 1 : 0;
  unsigned rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (threadIdx.x == 0) mbarrier_init(&barrier, 1);
  __syncthreads();
  for (unsigned phase = 0; phase < 2; ++phase) {
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) tile[threadIdx.x + i * 32] = 0xccccccccu;

    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned copies =
          Multicast ? (Paired ? (rank == Owner ? 2 : 0) : ((Mask >> rank) & 1))
                    : (rank == Owner ? 1 : 0);
      mbarrier_arrive_expect_tx(&barrier, copies * 512);
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (rank == Issuer && threadIdx.x == 0) {
      unsigned dst = __cvta_generic_to_shared(tile),
               bar = __cvta_generic_to_shared(&barrier);
      if constexpr (!Multicast)
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(dst)
                     : "r"(dst), "r"(Destination));
      if constexpr (Mode == 8)
        bar &= ~(1u << 24);  // Compiler idiom: peer rank 1 completes at rank 0.
      else if constexpr (Paired || !Multicast)
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(bar)
                     : "r"(bar), "r"(Owner));
#define COPY(RANK, COORDS, MODS, MASK)                                   \
  asm volatile("cp.async.bulk.tensor." RANK                              \
               ".shared::cluster.global."                                \
               "mbarrier::complete_tx::bytes" MODS " [%0], [%1, " COORDS \
               "], [%2]" MASK ";" ::"r"(dst),                            \
               "l"(map), "r"(bar), "r"(0), "r"(phase * 4),               \
               "h"((unsigned short)Mask)                                 \
               : "memory")
#define SHAPE(MODS, MASK)                       \
  if constexpr (Rank == 1) {                    \
    COPY("1d", "{%3}", MODS, MASK);             \
  } else if constexpr (Rank == 2) {             \
    COPY("2d", "{%3,%4}", MODS, MASK);          \
  } else if constexpr (Rank == 3) {             \
    COPY("3d", "{%3,%4,%3}", MODS, MASK);       \
  } else if constexpr (Rank == 4) {             \
    COPY("4d", "{%3,%4,%3,%3}", MODS, MASK);    \
  } else {                                      \
    COPY("5d", "{%3,%4,%3,%3,%3}", MODS, MASK); \
  }
      if constexpr (Mode == 7) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global."
            "mbarrier::complete_tx::bytes.cta_group::2 "
            "[%0],[%1,{%3,%4}],[%2];" ::"r"(dst),
            "l"(map), "r"(bar), "r"(0), "r"(phase * 4)
            : "memory");
      } else if constexpr (Multicast && Paired) {
        SHAPE(".multicast::cluster.cta_group::2", ", %5");
      } else if constexpr (Multicast) {
        SHAPE(".multicast::cluster", ", %5");
      } else if constexpr (Paired) {
        SHAPE(".cta_group::2", "");
      } else {
        SHAPE("", "");
      }
#undef SHAPE
#undef COPY
    }
    if constexpr (LongSuspend) {
      const unsigned bar = __cvta_generic_to_shared(&barrier);
      // A completion must notify a sleeping waiter, not wait for this hint
      // to expire. Short polling intervals otherwise conceal a missing wake.
      asm volatile("{ .reg .pred done; retry: "
                   "mbarrier.try_wait.parity.shared::cta.b64 "
                   "done, [%0], %1, 10000000; "
                   "@!done bra retry; }" :: "r"(bar), "r"(phase) : "memory");
    } else {
      wait(&barrier, phase);
    }
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      out[phase * 512 + blockIdx.x * 128 + threadIdx.x + i * 32] =
          tile[(threadIdx.x + i * 32) ^
               (Rank >= 3 ? (((threadIdx.x + i * 32) >> 5) & 7) * 4 : 0)];
  }
}
template <int Rank, int Mode, bool LongSuspend = false>
void check_cluster() {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  SKIP_IF_CLUSTER_NOC_OFF();
  unsigned *input, *out;
  CUtensorMap *dm;
  std::vector<unsigned> data(128 * 8 * 8);
  for (unsigned i = 0; i < data.size(); ++i) data[i] = i + 17;
  ASSERT_EQ(cudaMalloc(&input, data.size() * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 4096), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dm, sizeof(CUtensorMap)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(input, data.data(), data.size() * 4, cudaMemcpyHostToDevice),
      cudaSuccess);
  alignas(128) CUtensorMap map{};
  cuuint64_t dims[] = {128, 8, 2, 2, 2}, strides[] = {512, 4096, 8192, 16384};
  cuuint32_t box[] = {Rank == 1 ? 128u : 32u, 4, 1, 1, 1},
             steps[] = {1, 1, 1, 1, 1};
  ASSERT_EQ(cuTensorMapEncodeTiled(&map,
                                   Rank == 1 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                                             : CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                   Rank, input, dims, strides, box, steps,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
            CUDA_SUCCESS);
  if constexpr (Rank >= 3) {
    // Retained Driver controls, as in launch_test.cu. The compatibility
    // cuTensorMapEncodeTiled host format has no native 3D/5D decoder.
    std::array<uint64_t, 16> raw{};
    constexpr unsigned elem = Rank == 4 ? 1 : 4;
    raw[0] = reinterpret_cast<uint64_t>(input);
    raw[1] = Rank == 3 ? 0x2663a0 : Rank == 4 ? 0x266030 : 0x2663c0;
    for (unsigned i = 1; i < Rank; ++i)
      raw[(i + 2) / 2] |= uint64_t(strides[i - 1] / 16) << ((i % 2) * 32);
    dims[0] = 512 / elem;
    for (unsigned i = 0; i < Rank; ++i)
      raw[4 + i / 2] |= (dims[i] - 1) << ((i % 2) * 32);
    raw[6] |= uint64_t(128 / elem - 1) << 56;
    raw[7] = 3;
    raw[8] = 512;
    raw[9] = 1024;
    std::memcpy(&map, raw.data(), sizeof(map));
  }
  ASSERT_EQ(cudaMemcpy(dm, &map, sizeof(map), cudaMemcpyHostToDevice),
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
  constexpr unsigned mode = Mode;
  ASSERT_EQ(cudaLaunchKernelEx(&config, tensor_cluster<Mode, Rank, LongSuspend>, dm, out),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::array<unsigned, 1024> result{};
  ASSERT_EQ(cudaMemcpy(result.data(), out, 4096, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 1024; ++i) {
    const unsigned rank = (i / 128) % 2;
    const bool copied = mode < 2 || (mode >= 4 ? rank == 1 : rank == 0);
    EXPECT_EQ(
        result[i],
        copied
            ? data[Rank == 1 ? i % 128
                             : ((i / 512) * 4 + (i % 128) / 32) * 128 + i % 32]
            : 0xccccccccu)
        << i;
  }
  EXPECT_EQ(cudaFree(dm), cudaSuccess);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(TmaTensorCluster, MulticastAndPairedMulticast) {
  check_cluster<2, 0>();
  check_cluster<2, 1>();
  check_cluster<2, 2>();
  check_cluster<2, 3>();
  check_cluster<2, 4>();
  check_cluster<2, 5>();
  check_cluster<2, 6>();
  check_cluster<2, 7>();
}
TEST(TmaTensorCluster, PairedCompletionWithRankBitMask) {
  check_cluster<2, 8>();
}
TEST(TmaTensorCluster, CompletionWakesLongSuspendedWaiter) {
  check_cluster<2, 2, true>();  // Ordinary remote TMA, no TCGen05 dependency.
  check_cluster<2, 8, true>();  // Paired completion into the masked rank owner.
}
TEST(TmaTensorCluster, PairedRanks) {
  check_cluster<1, 3>();
  check_cluster<3, 3>();
  check_cluster<4, 3>();
  check_cluster<5, 3>();
}
TEST(TmaTensorCluster, MulticastRanks) {
  check_cluster<1, 0>();
  check_cluster<1, 1>();
  check_cluster<3, 0>();
  check_cluster<3, 1>();
  check_cluster<4, 0>();
  check_cluster<4, 1>();
  check_cluster<5, 0>();
  check_cluster<5, 1>();
}
}  // namespace
