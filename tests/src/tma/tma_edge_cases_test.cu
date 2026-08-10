#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "ptx/mbarrier.cuh"
#include "ptx/tma.cuh"

namespace {

using flashgpu::test::ptx::cp_async_bulk_tensor_1d_load;
using flashgpu::test::ptx::fence_proxy_tensormap_acquire;
using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;
using flashgpu::test::ptx::smem_u32_addr;
using flashgpu::test::ptx::smem_u64_addr;
using flashgpu::test::ptx::tensormap_cp_fenceproxy;
using flashgpu::test::ptx::tensormap_set_global_address;

constexpr unsigned kTensorMapBytes = 128;
constexpr unsigned kTileElements = 4;
constexpr unsigned kTileBytes = kTileElements * sizeof(uint32_t);

__device__ void initialize_1d_tensor_map(uint8_t *tensor_map,
                                         const uint32_t *input,
                                         uint32_t global_elements) {
  const uint64_t tensor_map_addr = smem_u64_addr(tensor_map);
  tensormap_set_global_address(tensor_map_addr,
                               reinterpret_cast<uint64_t>(input));
  asm volatile(
      "tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], 0x0;\n"
      : : "l"(tensor_map_addr));
  asm volatile(
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 "
      "[%0], 0x0, %1;\n"
      : : "l"(tensor_map_addr), "r"(kTileElements));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 "
      "[%0], 0x0, %1;\n"
      : : "l"(tensor_map_addr), "r"(global_elements));
  asm volatile(
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32 "
      "[%0], 0x0, 0x1;\n"
      : : "l"(tensor_map_addr));
  asm volatile(
      "tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], 0x2;\n"
      : : "l"(tensor_map_addr));
  asm volatile(
      "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 "
      "[%0], 0x0;\n"
      : : "l"(tensor_map_addr));
  asm volatile(
      "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 "
      "[%0], 0x0;\n"
      : : "l"(tensor_map_addr));
  asm volatile(
      "tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], 0x0;\n"
      : : "l"(tensor_map_addr));
}

__global__ void tma_negative_coordinate_kernel(const uint32_t *input,
                                                uint32_t *output,
                                                uint8_t *global_tensor_map) {
  extern __shared__ __align__(128) uint8_t shared[];
  uint8_t *tensor_map = shared;
  uint32_t *tile =
      reinterpret_cast<uint32_t *>(tensor_map + kTensorMapBytes);
  uint64_t *barrier = reinterpret_cast<uint64_t *>(tile + kTileElements);

  if (threadIdx.x < kTensorMapBytes / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(tensor_map)[threadIdx.x] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    initialize_1d_tensor_map(tensor_map, input, 8);
    tensormap_cp_fenceproxy(reinterpret_cast<uint64_t>(global_tensor_map),
                            smem_u64_addr(tensor_map));
    fence_proxy_tensormap_acquire(
        reinterpret_cast<uint64_t>(global_tensor_map));
    mbarrier_init(barrier, 1);
    mbarrier_arrive_expect_tx(barrier, kTileBytes);
  }
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  if (threadIdx.x == 0) {
    cp_async_bulk_tensor_1d_load(
        smem_u32_addr(tile), reinterpret_cast<uint64_t>(global_tensor_map), -2,
        smem_u32_addr(barrier));
    mbarrier_wait_parity(barrier, 0);
    mbarrier_inval(barrier);
  }
  __syncthreads();

  if (threadIdx.x < kTileElements) {
    output[threadIdx.x] = tile[threadIdx.x];
  }
}

TEST(TmaEdgeCasesTest, NegativeCoordinateZeroFillsLeadingElements) {
  constexpr std::array<uint32_t, 8> input = {11, 22, 33, 44,
                                              55, 66, 77, 88};
  constexpr std::array<uint32_t, kTileElements> expected = {0, 0, 11, 22};
  std::array<uint32_t, kTileElements> output = {};

  uint32_t *device_input = nullptr;
  uint32_t *device_output = nullptr;
  uint8_t *device_tensor_map = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, sizeof(output)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tensor_map, kTensorMapBytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), sizeof(input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  constexpr unsigned shared_bytes =
      kTensorMapBytes + kTileBytes + sizeof(uint64_t);
  tma_negative_coordinate_kernel<<<1, 32, shared_bytes>>>(
      device_input, device_output, device_tensor_map);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, sizeof(output),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(output, expected);

  EXPECT_EQ(cudaFree(device_tensor_map), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

}  // namespace
