#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "common/gpgpusim_config_topology.h"
#include "cp_kernels.cuh"
namespace {
template <int Mode>
__global__ void gather_cluster(const CUtensorMap *map, unsigned *out) {
  __shared__ __align__(128) unsigned tile[128];
  __shared__ __align__(8) unsigned long long barrier;
  constexpr bool Paired = Mode == 1 || Mode == 3;
  constexpr bool Multicast = Mode < 2 || Mode == 5;
  constexpr unsigned Mask = Mode == 5 ? 2 : 3;
  constexpr unsigned Destination = Mode == 4 ? 1 : 0;
  constexpr unsigned Owner = Paired || Mode == 4 ? 1 : 0;
  constexpr unsigned Issuer = Mode >= 2 ? 1 : 0;
  unsigned rank = blockIdx.x;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) tile[threadIdx.x + i * 32] = 0xccccccccu;
  if (threadIdx.x == 0) mbarrier_init(&barrier, 1);
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
    if constexpr (Paired || !Multicast)
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                   : "=r"(bar)
                   : "r"(bar), "r"(Owner));
    if constexpr (Multicast && Paired)
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global."
          "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::"
          "cache_hint [%0], [%1, {%2,%3,%4,%5,%6}], [%7], %8, %9;" ::"r"(dst),
          "l"(map), "r"(0), "r"(3), "r"(2), "r"(1), "r"(0), "r"(bar),
          "h"((unsigned short)Mask), "l"(0x1000000000000000ull)
          : "memory");
    else if constexpr (Multicast)
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global."
          "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::"
          "cache_hint [%0], [%1, {%2,%3,%4,%5,%6}], [%7], %8, %9;" ::"r"(dst),
          "l"(map), "r"(0), "r"(3), "r"(2), "r"(1), "r"(0), "r"(bar),
          "h"((unsigned short)Mask), "l"(0x1000000000000000ull)
          : "memory");
    else if constexpr (Paired)
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global."
          "mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, "
          "{%2,%3,%4,%5,%6}], [%7], %9;" ::"r"(dst),
          "l"(map), "r"(0), "r"(3), "r"(2), "r"(1), "r"(0), "r"(bar),
          "h"((unsigned short)Mask), "l"(0x1000000000000000ull)
          : "memory");
    else if constexpr (true)
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global."
          "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, "
          "{%2,%3,%4,%5,%6}], [%7], %9;" ::"r"(dst),
          "l"(map), "r"(0), "r"(3), "r"(2), "r"(1), "r"(0), "r"(bar),
          "h"((unsigned short)Mask), "l"(0x1000000000000000ull)
          : "memory");
  }
  wait(&barrier, 0);
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#pragma unroll
  for (unsigned i = 0; i < 4; ++i)
    out[rank * 128 + threadIdx.x + i * 32] = tile[threadIdx.x + i * 32];
}
TEST(TmaGather4Cluster, MulticastAndPairedMulticast) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  SKIP_IF_CLUSTER_NOC_OFF();
  unsigned *input, *out;
  CUtensorMap *dm;
  std::vector<unsigned> data(128 * 8);
  for (unsigned i = 0; i < data.size(); ++i) data[i] = i + 17;
  ASSERT_EQ(cudaMalloc(&input, 4096), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 1024), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dm, sizeof(CUtensorMap)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(input, data.data(), 4096, cudaMemcpyHostToDevice),
            cudaSuccess);
  alignas(128) CUtensorMap map{};
  cuuint64_t dims[] = {128, 8}, strides[] = {512};
  cuuint32_t box[] = {32, 1}, steps[] = {1, 1};
  ASSERT_EQ(
      cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, input, dims, strides, box,
          steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(dm, &map, sizeof(map), cudaMemcpyHostToDevice),
            cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(2);
  config.blockDim = dim3(32);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2;
  attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  for (unsigned mode = 0; mode < 6; ++mode) {
    SCOPED_TRACE(mode);
    switch (mode) {
      case 0:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<0>, dm, out),
                  cudaSuccess);
        break;
      case 1:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<1>, dm, out),
                  cudaSuccess);
        break;
      case 2:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<2>, dm, out),
                  cudaSuccess);
        break;
      case 3:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<3>, dm, out),
                  cudaSuccess);
        break;
      case 4:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<4>, dm, out),
                  cudaSuccess);
        break;
      case 5:
        ASSERT_EQ(cudaLaunchKernelEx(&config, gather_cluster<5>, dm, out),
                  cudaSuccess);
        break;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::array<unsigned, 256> result{};
    ASSERT_EQ(cudaMemcpy(result.data(), out, 1024, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (unsigned i = 0; i < 256; ++i) {
      const unsigned rank = i / 128;
      const bool copied =
          mode < 2 || (mode == 4 || mode == 5 ? rank == 1 : rank == 0);
      EXPECT_EQ(result[i], copied ? data[(3 - (i % 128) / 32) * 128 + i % 32]
                                  : 0xccccccccu)
          << i;
    }
  }
  EXPECT_EQ(cudaFree(dm), cudaSuccess);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
