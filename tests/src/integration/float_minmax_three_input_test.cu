#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// Each architecture manifest compiles one target. SM90 cannot assemble this
// PTX instruction even with a recent toolkit.
#if (__CUDACC_VER_MAJOR__ >= 13 || \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)) && \
    defined(__CUDA_ARCH_LIST__) && __CUDA_ARCH_LIST__ >= 1000
#define TEST_THREE_INPUT_MINMAX 1
#endif

#if defined(TEST_THREE_INPUT_MINMAX)
__global__ void three_input_minmax(const float *input, float *output) {
  const unsigned i = threadIdx.x;
  const float a = input[3 * i], b = input[3 * i + 1], c = input[3 * i + 2];
  float lo, hi;
  asm volatile("min.f32 %0, %1, %2, %3;" : "=f"(lo)
               : "f"(a), "f"(b), "f"(c));
  asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(hi)
               : "f"(a), "f"(b), "f"(c));
  output[2 * i] = lo;
  output[2 * i + 1] = hi;
}
#endif

TEST(FloatMinMaxThreeInput, EveryOperandParticipates) {
#if defined(TEST_THREE_INPUT_MINMAX)
  // Include every position of the extrema, NaNs, infinities and signed zero.
  const float input[][3] = {
      {1, 2, 9}, {1, 9, 2}, {9, 1, 2},
      {-1, -2, -9}, {-1, -9, -2}, {-9, -1, -2},
      {NAN, NAN, 7}, {NAN, 7, NAN}, {7, NAN, NAN},
      {1, 2, INFINITY}, {1, 2, -INFINITY},
      {-0.0f, -0.0f, 0.0f}, {0.0f, 0.0f, -0.0f}};
  const float expected[][2] = {
      {1, 9}, {1, 9}, {1, 9}, {-9, -1}, {-9, -1}, {-9, -1},
      {7, 7}, {7, 7}, {7, 7}, {1, INFINITY}, {-INFINITY, 2},
      {-0.0f, 0.0f}, {-0.0f, 0.0f}};
  constexpr unsigned count = sizeof(input) / sizeof(input[0]);
  float *device_input = nullptr, *device_output = nullptr;
  float output[count][2] = {};
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_input, sizeof(input)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_output, sizeof(output)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_input, input, sizeof(input),
                                    cudaMemcpyHostToDevice));
  three_input_minmax<<<1, count>>>(device_input, device_output);
  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output, device_output, sizeof(output),
                                    cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaFree(device_output));
  ASSERT_EQ(cudaSuccess, cudaFree(device_input));
  for (unsigned i = 0; i < count; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      uint32_t actual_bits, expected_bits;
      std::memcpy(&actual_bits, &output[i][j], sizeof(actual_bits));
      std::memcpy(&expected_bits, &expected[i][j], sizeof(expected_bits));
      EXPECT_EQ(expected_bits, actual_bits) << "input " << i << ", min/max " << j;
    }
  }
#else
  GTEST_SKIP() << "Three-input FP32 min/max requires CUDA 12.9+ and SM100+.";
#endif
}

}  // namespace
