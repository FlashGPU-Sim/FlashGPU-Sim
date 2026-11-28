#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>

namespace {
__global__ void literal_coordinate_load(const __grid_constant__ CUtensorMap map,
                                       unsigned *output) {
  __shared__ __align__(128) unsigned tile[32];
  __shared__ __align__(8) unsigned long long bar;
  const unsigned bp = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bp) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(tile));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 128;"
                 :: "r"(bp) : "memory");
    // Keep the literal coordinates in PTX, rather than loading two registers.
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes [%0], [%1, {0, 1}], [%2];"
                 :: "r"(dst), "l"(&map), "r"(bp) : "memory");
  }
  asm volatile("{.reg .pred ready; retry: "
               "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0; "
               "@!ready bra retry;}" :: "r"(bp) : "memory");
  output[threadIdx.x] = tile[threadIdx.x];
}
TEST(TensorCoordinateLiterals, LoadSecondRow) {
  unsigned *input = nullptr, *output = nullptr;
  std::array<unsigned, 64> values{};
  for (unsigned i = 0; i < values.size(); ++i) values[i] = 1000 + 17 * i;
  ASSERT_EQ(cudaMalloc(&input, sizeof(values)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output, 32 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(input, values.data(), sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  CUtensorMap map{};
  cuuint64_t dims[] = {32, 2}, strides[] = {128};
  cuuint32_t box[] = {32, 1}, steps[] = {1, 1};
  ASSERT_EQ(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
              input, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
              CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), CUDA_SUCCESS);
  literal_coordinate_load<<<1, 32>>>(map, output);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::array<unsigned, 32> result{};
  ASSERT_EQ(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < result.size(); ++i) EXPECT_EQ(result[i], values[i + 32]);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
  EXPECT_EQ(cudaFree(output), cudaSuccess);
}
}  // namespace
