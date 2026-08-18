#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

#include "ptx/mbarrier.cuh"
#include "ptx/tma.cuh"

namespace {

using flashgpu::test::ptx::cp_async_bulk_tensor_1d_load;
using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;
using flashgpu::test::ptx::smem_u32_addr;

constexpr uint32_t kLogicalElements = 128;
constexpr uint32_t kSharedBytes = 128;

__global__ void load_subbyte_layout_kernel(const CUtensorMap* tensor_map,
                                           uint32_t global_bytes,
                                           uint8_t* output) {
  __shared__ __align__(128) uint8_t tile[kSharedBytes];
  __shared__ __align__(8) uint64_t barrier;

  if (threadIdx.x != 0) return;
  mbarrier_init(&barrier, 1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  mbarrier_arrive_expect_tx(&barrier, global_bytes);
  cp_async_bulk_tensor_1d_load(smem_u32_addr(tile),
                               reinterpret_cast<uint64_t>(tensor_map), 0,
                               smem_u32_addr(&barrier));
  mbarrier_wait_parity(&barrier, 0);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

  const auto* tile_words = reinterpret_cast<const uint32_t*>(tile);
  auto* output_words = reinterpret_cast<uint32_t*>(output);
  for (uint32_t i = 0; i < kSharedBytes / sizeof(uint32_t); ++i)
    output_words[i] = tile_words[i];
  mbarrier_inval(&barrier);
}

CUresult encode_subbyte_map(
    CUtensorMap* map, CUtensorMapDataType type, void* base,
    uint64_t global_dim = kLogicalElements, uint32_t box_dim = kLogicalElements,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  const cuuint64_t global_dims[1] = {global_dim};
  const cuuint64_t global_strides[1] = {0};
  const cuuint32_t box_dims[1] = {box_dim};
  const cuuint32_t element_strides[1] = {1};
  return cuTensorMapEncodeTiled(map, type, 1, base, global_dims, global_strides,
                                box_dims, element_strides,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, oob_fill);
}

void expect_padded_load(CUtensorMapDataType type, uint32_t packed_group_bytes) {
  constexpr uint32_t kGroups = kLogicalElements / 16;
  const uint32_t global_bytes = kGroups * packed_group_bytes;

  // Allocate and initialize 128 bytes so the legacy bug, which incorrectly
  // fetched 16 global bytes per group, produces deterministic wrong data
  // instead of merely reading beyond the fixture.
  std::array<uint8_t, kSharedBytes> input{};
  for (uint32_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i + 1);

  uint8_t* device_input = nullptr;
  uint8_t* device_output = nullptr;
  CUtensorMap* device_map = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, kSharedBytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_map, sizeof(CUtensorMap)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), input.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output, 0xcd, kSharedBytes), cudaSuccess);

  CUtensorMap map{};
  ASSERT_EQ(encode_subbyte_map(&map, type, device_input), CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(device_map, &map, sizeof(map), cudaMemcpyHostToDevice),
            cudaSuccess);

  load_subbyte_layout_kernel<<<1, 1>>>(device_map, global_bytes, device_output);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::array<uint8_t, kSharedBytes> output{};
  ASSERT_EQ(cudaMemcpy(output.data(), device_output, output.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (uint32_t group = 0; group < kGroups; ++group) {
    for (uint32_t byte = 0; byte < packed_group_bytes; ++byte) {
      EXPECT_EQ(output[group * 16 + byte],
                input[group * packed_group_bytes + byte])
          << "group=" << group << " payload_byte=" << byte;
    }
  }

  EXPECT_EQ(cudaFree(device_map), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

TEST(TmaSubbyteLayoutIntegrationTest, U4LoadUsesEightPayloadBytesPerContainer) {
  expect_padded_load(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 8);
}

TEST(TmaSubbyteLayoutIntegrationTest,
     U6LoadUsesTwelvePayloadBytesPerContainer) {
  expect_padded_load(CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, 12);
}

TEST(TmaSubbyteLayoutIntegrationTest, EnforcesAlign16TensorMapGeometry) {
  CUtensorMap map{};
  void* aligned_base = reinterpret_cast<void*>(0x100000);

  EXPECT_EQ(encode_subbyte_map(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                               aligned_base),
            CUDA_SUCCESS);
  EXPECT_EQ(encode_subbyte_map(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                               reinterpret_cast<void*>(0x100010)),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(encode_subbyte_map(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                               aligned_base, 192),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(encode_subbyte_map(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                               aligned_base, kLogicalElements, 64),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(encode_subbyte_map(&map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                               aligned_base, kLogicalElements, kLogicalElements,
                               CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(encode_subbyte_map(
                &map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, aligned_base,
                kLogicalElements, kLogicalElements, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA),
            CUDA_ERROR_INVALID_VALUE);
}

}  // namespace
