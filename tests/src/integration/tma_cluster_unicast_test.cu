#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
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
}  // namespace
