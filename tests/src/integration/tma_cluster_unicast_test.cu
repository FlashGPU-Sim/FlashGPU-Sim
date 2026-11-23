#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"
#include "ptx/mbarrier.cuh"

namespace {
using namespace flashgpu::test::ptx;
constexpr int kTile = 32;
constexpr int kIterations = 32;
constexpr int kBlocks = 4;  // Two independent clusters when clusterDim.x == 2.

// Issue #44: shared::cluster is also the destination space of *unicast*
// loads. Every CTA owns a distinct tile and a barrier at the same local offset.
// Reuse the barrier so stray peer completions cannot hide after one phase.
template <bool Tensor>
__global__ void cluster_unicast_kernel(
    const __grid_constant__ CUtensorMap map, const uint32_t *input,
    uint32_t *output) {
  __shared__ __align__(128) uint32_t tile[kTile];
  __shared__ uint64_t bar;
  if (threadIdx.x == 0) mbarrier_init(&bar, blockDim.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    int offset = (blockIdx.x * kIterations + iteration) * kTile;
    mbarrier_arrive_expect_tx(&bar, threadIdx.x == 0 ? sizeof(tile) : 0);
    if (threadIdx.x == 0) {
      if constexpr (Tensor) {
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global."
            "mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
            :: "r"(smem_u32_addr(tile)), "l"(&map), "r"(offset),
               "r"(smem_u32_addr(&bar)) : "memory");
      } else {
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1], %2, [%3];"
            :: "r"(smem_u32_addr(tile)), "l"(input + offset),
               "n"(sizeof(tile)), "r"(smem_u32_addr(&bar)) : "memory");
      }
    }
    mbarrier_wait_parity(&bar, iteration & 1);
    output[offset + threadIdx.x] = tile[threadIdx.x];
    __syncthreads();  // All consumers finish before reusing this tile.
  }
  if (threadIdx.x == 0) mbarrier_inval(&bar);
}

template <bool Tensor>
void run_unicast(int cluster_size) {
  if (cluster_size > 1) {
    SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  }
  constexpr size_t count = kBlocks * kIterations * kTile;
  std::vector<uint32_t> input(count), output(count, 0);
  for (size_t i = 0; i < count; ++i) input[i] = 0x10000 + i;
  uint32_t *device_input = nullptr, *device_output = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, count * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, count * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output, 0, count * sizeof(uint32_t)), cudaSuccess);

  CUtensorMap map{};
  if constexpr (Tensor) {
    cuuint64_t dims[] = {count};
    cuuint64_t strides[] = {count * sizeof(uint32_t)};
    cuuint32_t box[] = {kTile}, element_strides[] = {1};
    ASSERT_EQ(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 1, device_input, dims, strides,
        box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), CUDA_SUCCESS);
  }
  if (cluster_size == 0) {
    cluster_unicast_kernel<Tensor><<<kBlocks, kTile>>>(
        map, device_input, device_output);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  } else {
    void *args[] = {&map, &device_input, &device_output};
    ASSERT_EQ(flash_test::launch_kernel_with_cluster(
        (const void *)cluster_unicast_kernel<Tensor>, dim3(kBlocks), dim3(kTile),
        dim3(cluster_size, 1, 1), args), cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, count * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(output, input) << "Each CTA must receive only its own tiles";
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}

TEST(TMAClusterUnicastTest, BulkOrdinaryLaunch) { run_unicast<false>(0); }
TEST(TMAClusterUnicastTest, BulkClusterSizeOne) { run_unicast<false>(1); }
TEST(TMAClusterUnicastTest, BulkClusterSizeTwo) { run_unicast<false>(2); }
TEST(TMAClusterUnicastTest, TensorOrdinaryLaunch) { run_unicast<true>(0); }
TEST(TMAClusterUnicastTest, TensorClusterSizeOne) { run_unicast<true>(1); }
TEST(TMAClusterUnicastTest, TensorClusterSizeTwo) { run_unicast<true>(2); }

// Extended address-space and cluster-size coverage.
constexpr int kRounds = 8;

enum Mode { CtaLocal, ClusterLocal, Remote, Multicast };

__device__ void cluster_sync() {
  asm volatile(
      "barrier.cluster.arrive.aligned;\n"
      "barrier.cluster.wait.aligned;" ::
          : "memory");
}

__device__ uint32_t map_shared(uint32_t address, unsigned rank) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(address), "r"(rank));
  return mapped;
}

// Rotate between all ranks, highest rank only, all except the issuer, and
// alternating ranks. In particular, size 16 exercises mask bit 15.
__host__ __device__ uint16_t destinations(int size, int round) {
  unsigned all = (1u << size) - 1;
  switch (round % 4) {
    case 0:
      return all;
    case 1:
      return 1u << (size - 1);
    case 2:
      return size == 1 ? 1 : all & ~1u;
    default:
      return all & 0xAAAAu ? all & 0xAAAAu : 1;
  }
}

template <bool Tensor, Mode AddressMode, int Tile>
__global__ void address_kernel(const __grid_constant__ CUtensorMap map,
                               const uint32_t *input, uint32_t *output,
                               unsigned long long *elapsed, int size) {
  // Both sides of the transfer have sentinels to detect bad address offsets.
  __shared__ __align__(128) uint32_t storage[Tile + 64];
  __shared__ uint64_t bar;
  uint32_t *tile = storage + 32;
  int rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int cluster_x;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_x));
  int producer_bid = (blockIdx.x / cluster_x) * cluster_x;
  if (threadIdx.x == 0) mbarrier_init(&bar, 1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  for (int round = 0; round < kRounds; ++round) {
    for (int i = threadIdx.x; i < Tile + 64; i += blockDim.x)
      storage[i] = 0xDEADBEEF;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint16_t mask = destinations(size, round);
    bool selected = AddressMode != Multicast || (mask & (1u << rank));
    if (threadIdx.x == 0 && selected)
      mbarrier_arrive_expect_tx(&bar, Tile * sizeof(uint32_t));
    cluster_sync();

    unsigned long long begin = clock64();
    if (threadIdx.x == 0 && (AddressMode != Multicast || rank == 0)) {
      uint32_t dst = smem_u32_addr(tile), mb = smem_u32_addr(&bar);
      if constexpr (AddressMode == Remote) {
        unsigned peer = (rank + 1) % size;
        dst = map_shared(dst, peer);
        mb = map_shared(mb, peer);
      }
      int source = AddressMode == Multicast ? producer_bid : bid;
      int offset = (source * kRounds + round) * Tile;
      if constexpr (Tensor) {
        if constexpr (AddressMode == CtaLocal) {
          asm volatile(
              "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_"
              "tx::bytes [%0], [%1, {%2}], [%3];" ::"r"(dst),
              "l"(&map), "r"(offset), "r"(mb)
              : "memory");
        } else if constexpr (AddressMode == Multicast) {
          asm volatile(
              "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
              "complete_tx::bytes.multicast::cluster [%0], [%1, {%2}], [%3], "
              "%4;" ::"r"(dst),
              "l"(&map), "r"(offset), "r"(mb), "h"(mask)
              : "memory");
        } else {
          asm volatile(
              "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
              "complete_tx::bytes [%0], [%1, {%2}], [%3];" ::"r"(dst),
              "l"(&map), "r"(offset), "r"(mb)
              : "memory");
        }
      } else {
        if constexpr (AddressMode == CtaLocal) {
          asm volatile(
              "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1], %2, [%3];" ::"r"(dst),
              "l"(input + offset), "n"(Tile * 4), "r"(mb)
              : "memory");
        } else if constexpr (AddressMode == Multicast) {
          asm volatile(
              "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
              "bytes.multicast::cluster [%0], [%1], %2, [%3], %4;" ::"r"(dst),
              "l"(input + offset), "n"(Tile * 4), "r"(mb), "h"(mask)
              : "memory");
        } else {
          asm volatile(
              "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
              "bytes [%0], [%1], %2, [%3];" ::"r"(dst),
              "l"(input + offset), "n"(Tile * 4), "r"(mb)
              : "memory");
        }
      }
    }
    // The phase advances only in selected multicast rounds.
    if (threadIdx.x == 0 && selected) {
      unsigned phase = 0;
      for (int r = 0; r < round; ++r)
        phase += AddressMode != Multicast ||
                 (destinations(size, r) & (1u << rank)) != 0;
      mbarrier_wait_parity(&bar, phase & 1);
      elapsed[bid * kRounds + round] = clock64() - begin;
    }
    __syncthreads();
    cluster_sync();  // Keep every destination alive, including non-receivers.
    for (int i = threadIdx.x; i < Tile + 64; i += blockDim.x)
      output[(bid * kRounds + round) * (Tile + 64) + i] = storage[i];
    __syncthreads();
    cluster_sync();  // Consume this round before any issuer reuses the tile.
  }
  if (threadIdx.x == 0) mbarrier_inval(&bar);
}

struct Shape {
  int size;
  dim3 dim;
};

// Native hardware may support fewer ranks than the simulator preset. Query
// the actual device/kernel limit, and report unsupported native sizes as skips.
template <bool Tensor, Mode AddressMode, int Tile = 32>
void run_address(Shape shape) {
  auto topology = flash_test::read_gpgpusim_topology();
  if (topology.found_config &&
      topology.n_cores_per_cluster < unsigned(shape.size))
    GTEST_SKIP() << "Config has fewer SMs per GPC than cluster size "
                 << shape.size;
  const int blocks = 2 * shape.size;
  const size_t count = blocks * kRounds * Tile;
  std::vector<uint32_t> input(count);
  for (size_t i = 0; i < count; ++i) input[i] = 0x10000u + i;
  uint32_t *src = nullptr, *dst = nullptr;
  unsigned long long *elapsed = nullptr;
  ASSERT_EQ(cudaMalloc(&src, count * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dst, blocks * kRounds * (Tile + 64) * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&elapsed, blocks * kRounds * 8), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(src, input.data(), count * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(dst, 0, blocks * kRounds * (Tile + 64) * 4),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(elapsed, 0, blocks * kRounds * 8), cudaSuccess);
  CUtensorMap map{};
  cuuint64_t dims[] = {count}, strides[] = {count * 4};
  cuuint32_t box[] = {Tile}, steps[] = {1};
  ASSERT_EQ(
      cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 1, src, dims, strides, box,
          steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      CUDA_SUCCESS);
  auto kernel = address_kernel<Tensor, AddressMode, Tile>;
  ASSERT_EQ(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
            cudaSuccess);
  dim3 grid(shape.dim.x * 2, shape.dim.y, shape.dim.z);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(32);
  int limit = 0;
  ASSERT_EQ(cudaOccupancyMaxPotentialClusterSize(&limit, kernel, &cfg),
            cudaSuccess);
  if (shape.size > limit) {
    cudaFree(src);
    cudaFree(dst);
    cudaFree(elapsed);
    GTEST_SKIP() << "Device/kernel cluster limit " << limit;
  }
  void *args[] = {&map, &src, &dst, &elapsed, &shape.size};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster((const void *)kernel, grid,
                                                   dim3(32), shape.dim, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> output(blocks * kRounds * (Tile + 64));
  std::vector<unsigned long long> cycles(blocks * kRounds);
  ASSERT_EQ(
      cudaMemcpy(output.data(), dst, output.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(cycles.data(), elapsed, cycles.size() * 8,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int bid = 0; bid < blocks; ++bid) {
    // CUDA cluster ranks are local coordinates, not flattened grid indices.
    int x = bid % int(grid.x), y = bid / grid.x % int(grid.y),
        z = bid / (grid.x * grid.y);
    int rank = x % shape.dim.x + shape.dim.x * (y + shape.dim.y * z);
    int group = x / shape.dim.x;
    for (int r = 0; r < kRounds; ++r) {
      bool selected = AddressMode != Multicast ||
                      (destinations(shape.size, r) & (1u << rank));
      int source_rank =
          AddressMode == Remote ? (rank + shape.size - 1) % shape.size : rank;
      if (AddressMode == Multicast) source_rank = 0;
      int sx = source_rank % shape.dim.x + group * shape.dim.x;
      int sy = source_rank / shape.dim.x % shape.dim.y,
          sz = source_rank / (shape.dim.x * shape.dim.y);
      int source = sx + grid.x * (sy + grid.y * sz);
      for (int i = 0; i < Tile + 64; ++i) {
        uint32_t want = selected && i >= 32 && i < Tile + 32
                            ? input[(source * kRounds + r) * Tile + i - 32]
                            : 0xDEADBEEF;
        ASSERT_EQ(output[(bid * kRounds + r) * (Tile + 64) + i], want)
            << "block=" << bid << " rank=" << rank << " round=" << r
            << " element=" << i;
      }
      if (selected) EXPECT_GT(cycles[bid * kRounds + r], 0u);
    }
  }
  std::vector<unsigned long long> nonzero;
  for (auto value : cycles)
    if (value) nonzero.push_back(value);
  std::sort(nonzero.begin(), nonzero.end());
  ASSERT_FALSE(nonzero.empty());
  printf(
      "TMA_TIMING tensor=%d mode=%d size=%d shape=%ux%ux%u bytes=%d min=%llu "
      "median=%llu max=%llu\n",
      Tensor, AddressMode, shape.size, shape.dim.x, shape.dim.y, shape.dim.z,
      Tile * 4, nonzero.front(), nonzero[nonzero.size() / 2], nonzero.back());
  EXPECT_EQ(cudaFree(src), cudaSuccess);
  EXPECT_EQ(cudaFree(dst), cudaSuccess);
  EXPECT_EQ(cudaFree(elapsed), cudaSuccess);
}

class TMAClusterAddressTest : public ::testing::TestWithParam<Shape> {};
TEST_P(TMAClusterAddressTest, BulkCtaLocal) {
  run_address<false, CtaLocal>(GetParam());
}
TEST_P(TMAClusterAddressTest, BulkClusterLocal) {
  run_address<false, ClusterLocal>(GetParam());
}
TEST_P(TMAClusterAddressTest, BulkRemote) {
  run_address<false, Remote>(GetParam());
}
TEST_P(TMAClusterAddressTest, BulkMulticast) {
  run_address<false, Multicast>(GetParam());
}
TEST_P(TMAClusterAddressTest, TensorCtaLocal) {
  run_address<true, CtaLocal>(GetParam());
}
TEST_P(TMAClusterAddressTest, TensorClusterLocal) {
  run_address<true, ClusterLocal>(GetParam());
}
TEST_P(TMAClusterAddressTest, TensorRemote) {
  run_address<true, Remote>(GetParam());
}
TEST_P(TMAClusterAddressTest, TensorMulticast) {
  run_address<true, Multicast>(GetParam());
}
std::vector<Shape> shapes() {
  // Self-only, first peer, portable limit, opt-in limit, and 3D rank mapping.
  // Keep all transfer modes at these boundaries without sweeping every size.
  return {{1, dim3(1)},
          {2, dim3(2)},
          {8, dim3(8)},
          {16, dim3(16)},
          {16, dim3(4, 2, 2)}};
}
INSTANTIATE_TEST_SUITE_P(RepresentativeShapes, TMAClusterAddressTest,
                         ::testing::ValuesIn(shapes()),
                         [](const ::testing::TestParamInfo<Shape> &info) {
                           return "Size" + std::to_string(info.param.size) +
                                  "_" + std::to_string(info.param.dim.x) + "x" +
                                  std::to_string(info.param.dim.y) + "x" +
                                  std::to_string(info.param.dim.z);
                         });
TEST(TMAClusterAddressBoundaryTest, BulkRemote1KiBSize16) {
  run_address<false, Remote, 256>({16, dim3(16)});
}
TEST(TMAClusterAddressBoundaryTest, TensorRemote1KiBSize16) {
  run_address<true, Remote, 256>({16, dim3(16)});
}
TEST(TMAClusterAddressBoundaryTest, BulkMulticast1KiBSize16) {
  run_address<false, Multicast, 256>({16, dim3(16)});
}
TEST(TMAClusterAddressBoundaryTest, TensorMulticast1KiBSize16) {
  run_address<true, Multicast, 256>({16, dim3(16)});
}
__global__ void limit_probe() {}
TEST(TMAClusterAddressBoundaryTest, RejectsBeyondPortableAndOptInLimits) {
  const void *kernel = (const void *)limit_probe;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(32);
  cfg.blockDim = dim3(32);
  int portable = 0, extended = 0;
  ASSERT_EQ(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 0),
            cudaSuccess);
  ASSERT_EQ(cudaOccupancyMaxPotentialClusterSize(&portable, kernel, &cfg),
            cudaSuccess);
  EXPECT_LE(portable, 8);
  EXPECT_NE(flash_test::launch_kernel_with_cluster(kernel, dim3(18), dim3(32),
                                                   dim3(9), nullptr),
            cudaSuccess);
  cudaGetLastError();
  ASSERT_EQ(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
            cudaSuccess);
  ASSERT_EQ(cudaOccupancyMaxPotentialClusterSize(&extended, kernel, &cfg),
            cudaSuccess);
  EXPECT_LE(extended, 16);
  EXPECT_GE(extended, portable);
  auto topology = flash_test::read_gpgpusim_topology();
  if (topology.found_config) {
    EXPECT_EQ(portable, std::min(8u, topology.n_cores_per_cluster));
    EXPECT_EQ(extended, std::min(16u, topology.n_cores_per_cluster));
  }
  EXPECT_NE(flash_test::launch_kernel_with_cluster(kernel, dim3(34), dim3(32),
                                                   dim3(17), nullptr),
            cudaSuccess);
  cudaGetLastError();
  // The product wraps to zero in 32 bits. Reject before dispatching any CTA.
  const dim3 overflowing_cluster(1u << 30, 4, 1);
  EXPECT_NE(flash_test::launch_kernel_with_cluster(
                kernel, overflowing_cluster, dim3(32), overflowing_cluster,
                nullptr),
            cudaSuccess);
  cudaGetLastError();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
}  // namespace
