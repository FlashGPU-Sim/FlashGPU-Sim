// CUDA integration tests for Hopper warp-group MMA.
//
// These tests intentionally use input patterns whose full M64N8 output tile has
// one expected value. That keeps the first WGMMA bring-up focused on instruction
// execution, shared-memory descriptors, and async commit/wait behavior without
// depending on the full accumulator-to-matrix layout mapping.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace wgmma_test {

constexpr int kM = 64;
constexpr int kN = 8;
constexpr int kK = 16;
constexpr int kWarpgroupThreads = 128;
constexpr int kAccumRegsPerThread = 4;
constexpr int kOutputRegs = kWarpgroupThreads * kAccumRegsPerThread;
constexpr int kSharedBElements = 256;
static_assert(kM * kN == kOutputRegs,
              "m64n8 WGMMA must produce one accumulator per output element.");
static_assert(kSharedBElements >= kN * kK,
              "shared-memory B tile must cover at least N*K half values.");

uint16_t f32_to_f16(float f32) {
  uint32_t bits;
  std::memcpy(&bits, &f32, sizeof(float));

  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
  uint32_t frac = bits & 0x7FFFFF;

  if (exp > 15) return static_cast<uint16_t>((sign << 15) | 0x7C00);
  if (exp < -14) {
    if (exp < -24) return static_cast<uint16_t>(sign << 15);
    uint32_t denorm_frac = (0x800000 | frac) >> (-14 - exp);
    return static_cast<uint16_t>((sign << 15) | (denorm_frac >> 13));
  }

  uint32_t f16_exp = static_cast<uint32_t>(exp + 15);
  uint32_t f16_frac = frac >> 13;
  return static_cast<uint16_t>((sign << 15) | (f16_exp << 10) | f16_frac);
}

float f16_to_f32(uint16_t f16) {
  uint32_t sign = (f16 >> 15) & 0x1;
  uint32_t exp = (f16 >> 10) & 0x1F;
  uint32_t frac = f16 & 0x3FF;

  if (exp == 0) {
    if (frac == 0) return sign ? -0.0f : 0.0f;
    float result = frac / 1024.0f / 16384.0f;
    return sign ? -result : result;
  }
  if (exp == 31) return frac ? NAN : (sign ? -INFINITY : INFINITY);

  uint32_t f32_exp = exp - 15 + 127;
  uint32_t f32_frac = frac << 13;
  uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;

  float result;
  std::memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

struct RunResult {
  cudaError_t setup_error = cudaSuccess;
  cudaError_t launch_error = cudaSuccess;
  cudaError_t sync_error = cudaSuccess;
  cudaError_t copy_error = cudaSuccess;
  std::vector<float> output;
};

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const *ptr) {
  uint32_t smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
      "cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr)
      : "l"(ptr));
  return smem_ptr;
}

__device__ __forceinline__ uint64_t make_gmma_k_major_desc(
    void const *ptr, uint32_t leading_byte_offset, uint32_t stride_byte_offset) {
  uint32_t smem_addr = smem_ptr_to_uint(ptr);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  return desc;
}

__device__ __forceinline__ void wgmma_m64n8k16_f32_f16_f16_rs(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
    float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %9, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "%8, p, %10, %11, %12;\n"
      "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
        "r"(scale_d), "n"(1), "n"(1), "n"(0));
}

__global__ void wgmma_m64n8k16_f16_uniform_kernel(float *out,
                                                  uint16_t a_bits,
                                                  uint16_t b_bits,
                                                  float initial_c,
                                                  int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(16) uint16_t smem_b[kSharedBElements];

  for (int i = threadIdx.x; i < kSharedBElements; i += blockDim.x) {
    smem_b[i] = b_bits;
  }
  __syncthreads();

  uint64_t desc_b = make_gmma_k_major_desc(
      smem_b, /*leading_byte_offset=*/256, /*stride_byte_offset=*/16);

  uint32_t packed_a = static_cast<uint32_t>(a_bits) |
                      (static_cast<uint32_t>(a_bits) << 16);
  uint32_t a0 = packed_a;
  uint32_t a1 = packed_a;
  uint32_t a2 = packed_a;
  uint32_t a3 = packed_a;

  float d0 = initial_c;
  float d1 = initial_c;
  float d2 = initial_c;
  float d3 = initial_c;

  asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)::"memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  wgmma_m64n8k16_f32_f16_f16_rs(a0, a1, a2, a3, desc_b, d0, d1, d2, d3,
                                scale_d);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)::"memory");

  int base = threadIdx.x * kAccumRegsPerThread;
  out[base + 0] = d0;
  out[base + 1] = d1;
  out[base + 2] = d2;
  out[base + 3] = d3;
}

RunResult run_wgmma_kernel(uint16_t a_bits, uint16_t b_bits, float initial_c,
                           int32_t scale_d) {
  RunResult result;
  result.output.assign(kOutputRegs, 0.0f);

  float *d_out = nullptr;

  result.setup_error = cudaMalloc(&d_out, kOutputRegs * sizeof(float));
  if (result.setup_error != cudaSuccess) return result;

  cudaGetLastError();
  wgmma_m64n8k16_f16_uniform_kernel<<<1, kWarpgroupThreads>>>(
      d_out, a_bits, b_bits, initial_c, scale_d);

  result.launch_error = cudaGetLastError();
  if (result.launch_error == cudaSuccess) {
    result.sync_error = cudaDeviceSynchronize();
  }

  if (result.launch_error == cudaSuccess && result.sync_error == cudaSuccess) {
    result.copy_error =
        cudaMemcpy(result.output.data(), d_out, kOutputRegs * sizeof(float),
                   cudaMemcpyDeviceToHost);
  }

  cudaFree(d_out);
  return result;
}

class WgmmaF16M64N8K16IntegrationTest : public ::testing::Test {
 protected:
  void validate_uniform_wgmma(uint16_t a_bits, uint16_t b_bits, float initial_c,
                              int32_t scale_d, float tolerance = 1e-3f) {
    RunResult result = run_wgmma_kernel(a_bits, b_bits, initial_c, scale_d);

    ASSERT_EQ(result.setup_error, cudaSuccess)
        << "CUDA setup error: " << cudaGetErrorString(result.setup_error);
    ASSERT_EQ(result.launch_error, cudaSuccess)
        << "CUDA launch error: " << cudaGetErrorString(result.launch_error);
    ASSERT_EQ(result.sync_error, cudaSuccess)
        << "CUDA sync error: " << cudaGetErrorString(result.sync_error);
    ASSERT_EQ(result.copy_error, cudaSuccess)
        << "CUDA copy error: " << cudaGetErrorString(result.copy_error);

    ASSERT_EQ(result.output.size(), static_cast<size_t>(kOutputRegs));

    float a = f16_to_f32(a_bits);
    float b = f16_to_f32(b_bits);
    float expected = static_cast<float>(kK) * a * b +
                     (scale_d != 0 ? initial_c : 0.0f);

    for (int i = 0; i < kOutputRegs; ++i) {
      EXPECT_NEAR(result.output[i], expected, tolerance)
          << "Mismatch at accumulator register " << i
          << " (got: " << result.output[i] << ", expected: " << expected
          << ")";
    }
  }
};

TEST_F(WgmmaF16M64N8K16IntegrationTest, AllOnesTest) {
  validate_uniform_wgmma(f32_to_f16(1.0f), f32_to_f16(1.0f), 0.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, ZeroAWithAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(0.0f), f32_to_f16(1.0f), 3.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, MixedSignTest) {
  validate_uniform_wgmma(f32_to_f16(-1.0f), f32_to_f16(2.0f), 0.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, NonZeroAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(2.0f), f32_to_f16(0.5f), 7.0f,
                         /*scale_d=*/1);
}

TEST_F(WgmmaF16M64N8K16IntegrationTest, ScaleDZeroIgnoresAccumulatorTest) {
  validate_uniform_wgmma(f32_to_f16(1.0f), f32_to_f16(1.0f), 9.0f,
                         /*scale_d=*/0);
}

}  // namespace wgmma_test
