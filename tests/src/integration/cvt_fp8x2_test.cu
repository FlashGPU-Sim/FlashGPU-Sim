#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ uint16_t cvt_e4m3x2(float high, float low) {
  uint16_t result;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(result)
      : "f"(high), "f"(low));
  return result;
}

__device__ __forceinline__ uint16_t cvt_e5m2x2(float high, float low) {
  uint16_t result;
  asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
      : "=h"(result)
      : "f"(high), "f"(low));
  return result;
}

__global__ void cvt_fp8x2_kernel(uint16_t* output, const float* input) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  output[0] = cvt_e4m3x2(input[0], input[1]);
  output[1] = cvt_e5m2x2(input[0], input[1]);
  output[2] = cvt_e4m3x2(input[2], input[3]);
  output[3] = cvt_e4m3x2(input[4], input[5]);
  output[4] = cvt_e4m3x2(input[6], input[7]);
}

int main() {
  // 1/-2, finite saturation, two round-to-nearest-even ties, NaN/-0.
  const float input[] = {1.0f,    -2.0f,   500.0f, -500.0f,
                         1.0625f, 1.1875f, NAN,    -0.0f};
  const uint16_t expected[] = {0x38c0, 0x3cc0, 0x7efe, 0x383a, 0x7f80};
  float* device_input = nullptr;
  uint16_t* device_output = nullptr;
  if (cudaMalloc(&device_input, sizeof(input)) != cudaSuccess ||
      cudaMalloc(&device_output, sizeof(expected)) != cudaSuccess)
    return 1;
  if (cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;
  cvt_fp8x2_kernel<<<1, 1>>>(device_output, device_input);
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;

  uint16_t actual[sizeof(expected) / sizeof(expected[0])]{};
  if (cudaMemcpy(actual, device_output, sizeof(actual),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  bool pass = true;
  for (unsigned i = 0; i < sizeof(actual) / sizeof(actual[0]); ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "cvt fp8x2 case %u: got 0x%04x expected 0x%04x\n", i,
                   actual[i], expected[i]);
      pass = false;
    }
  }
  std::printf("CVT_FP8X2 verification=%s\n", pass ? "PASS" : "FAIL");
  cudaFree(device_input);
  cudaFree(device_output);
  return pass ? 0 : 1;
}
