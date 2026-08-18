#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

constexpr uint32_t kColumns = 16;
constexpr uint32_t kRows = 2;
constexpr uint32_t kElements = kColumns * kRows;

__global__ void tensor_reduce_add_kernel(const CUtensorMap *tensor_map,
                                         const float *contribution) {
  __shared__ __align__(128) float tile[kElements];
  if (threadIdx.x < kElements) tile[threadIdx.x] = contribution[threadIdx.x];
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint64_t map_address = reinterpret_cast<uint64_t>(tensor_map);
    const uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(tile));
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
        "[%0, {%2, %3}], [%1];\n"
        :
        : "l"(map_address), "r"(shared_address), "r"(0), "r"(0)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
  }
}

TEST(TmaTensorReduceTest, AddsSharedTileToExistingGlobalTensor) {
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);

  std::array<float, kElements> initial;
  std::array<float, kElements> contribution;
  std::array<float, kElements> result{};
  for (uint32_t i = 0; i < kElements; ++i) {
    initial[i] = static_cast<float>(i) + 0.25f;
    contribution[i] = static_cast<float>(i % 7) - 2.0f;
  }

  float *device_output = nullptr;
  float *device_contribution = nullptr;
  CUtensorMap *device_tensor_map = nullptr;
  ASSERT_EQ(cudaMalloc(&device_output, sizeof(initial)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_contribution, sizeof(contribution)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tensor_map, sizeof(CUtensorMap)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_output, initial.data(), sizeof(initial),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_contribution, contribution.data(),
                       sizeof(contribution), cudaMemcpyHostToDevice),
            cudaSuccess);

  CUtensorMap tensor_map{};
  const uint64_t global_dimensions[2] = {kColumns, kRows};
  const uint64_t global_strides[1] = {kColumns * sizeof(float)};
  const uint32_t box_dimensions[2] = {kColumns, kRows};
  const uint32_t element_strides[2] = {1, 1};
  ASSERT_EQ(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, device_output,
          global_dimensions, global_strides, box_dimensions, element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(device_tensor_map, &tensor_map, sizeof(tensor_map),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  tensor_reduce_add_kernel<<<1, 32>>>(device_tensor_map, device_contribution);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device_output, sizeof(result),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_FLOAT_EQ(result[i], initial[i] + contribution[i]) << "index " << i;

  EXPECT_EQ(cudaFree(device_tensor_map), cudaSuccess);
  EXPECT_EQ(cudaFree(device_contribution), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}

TEST(TmaTensorReduceTest, DistinguishesFloat32AndFloat32FtzTensorMapTypes) {
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);

  std::array<uint32_t, kElements> initial_bits{};
  std::array<uint32_t, kElements> contribution_bits{};
  std::array<uint32_t, kElements> result_bits{};
  contribution_bits[0] = 1u;

  float *device_output = nullptr;
  float *device_contribution = nullptr;
  CUtensorMap *device_tensor_map = nullptr;
  ASSERT_EQ(cudaMalloc(&device_output, sizeof(initial_bits)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_contribution, sizeof(contribution_bits)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_tensor_map, sizeof(CUtensorMap)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_output, initial_bits.data(), sizeof(initial_bits),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_contribution, contribution_bits.data(),
                       sizeof(contribution_bits), cudaMemcpyHostToDevice),
            cudaSuccess);

  CUtensorMap tensor_map{};
  const uint64_t global_dimensions[2] = {kColumns, kRows};
  const uint64_t global_strides[1] = {kColumns * sizeof(float)};
  const uint32_t box_dimensions[2] = {kColumns, kRows};
  const uint32_t element_strides[2] = {1, 1};
  ASSERT_EQ(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, device_output,
          global_dimensions, global_strides, box_dimensions, element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(device_tensor_map, &tensor_map, sizeof(tensor_map),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  tensor_reduce_add_kernel<<<1, 32>>>(device_tensor_map, device_contribution);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result_bits.data(), device_output, sizeof(result_bits),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(result_bits[0], 1u);

  ASSERT_EQ(cudaMemcpy(device_output, initial_bits.data(), sizeof(initial_bits),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ, 2, device_output,
          global_dimensions, global_strides, box_dimensions, element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(device_tensor_map, &tensor_map, sizeof(tensor_map),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  tensor_reduce_add_kernel<<<1, 32>>>(device_tensor_map, device_contribution);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result_bits.data(), device_output, sizeof(result_bits),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(result_bits[0], 0u);

  EXPECT_EQ(cudaFree(device_tensor_map), cudaSuccess);
  EXPECT_EQ(cudaFree(device_contribution), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}

}  // namespace
