// Shared helpers for Hopper WGMMA integration tests.
//
// Data-type-specific tests live in cuda_wgmma_<type>_test.cc files. This header
// keeps common WGMMA shape/layout helpers in one place so new data types can
// reuse the same test infrastructure without duplicating descriptor and
// accumulator mapping logic.

#ifndef TEST_SRC_HOPPER_WGMMA_TENSOR_WGMMA_TEST_CUH
#define TEST_SRC_HOPPER_WGMMA_TENSOR_WGMMA_TEST_CUH

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
constexpr int kOutputRegs = kM * kN;
constexpr int kParallelWgmmaGroups = 3;
constexpr int kParallelWarpgroupThreads =
    kParallelWgmmaGroups * kWarpgroupThreads;
constexpr int kParallelOutputRegs = kParallelWgmmaGroups * kOutputRegs;
constexpr int kSharedBElements = 256;
constexpr int kBLeadingByteOffset = 256;
constexpr int kBStrideByteOffset = 16;
constexpr int kBContiguousK =
    kBStrideByteOffset / static_cast<int>(sizeof(uint16_t));
constexpr int kBLeadingElements =
    kBLeadingByteOffset / static_cast<int>(sizeof(uint16_t));
static_assert(kM * kN == kOutputRegs,
              "m64n8 WGMMA must produce one accumulator per output element.");
static_assert(kSharedBElements >= kN * kK,
              "shared-memory B tile must cover at least N*K half values.");
static_assert(kBContiguousK == 8,
              "m64n8k16 K-major B descriptor expects 8 contiguous K values.");
static_assert(kBLeadingElements + (kN - 1) * kBContiguousK +
                      (kK - 1) % kBContiguousK <
                  kSharedBElements,
              "shared-memory B tile must cover the descriptor-padded layout.");

struct RunResult {
  cudaError_t setup_error = cudaSuccess;
  cudaError_t input_copy_error = cudaSuccess;
  cudaError_t launch_error = cudaSuccess;
  cudaError_t sync_error = cudaSuccess;
  cudaError_t copy_error = cudaSuccess;
  std::vector<float> output;
};

__host__ __device__ __forceinline__ int c_matrix_index(int row, int col) {
  return row * kN + col;
}

__host__ __device__ __forceinline__ int a_matrix_index(int row, int k) {
  return row * kK + k;
}

__host__ __device__ __forceinline__ int b_matrix_index(int col, int k) {
  return col * kK + k;
}

__host__ __device__ __forceinline__ int b_smem_index(int col, int k) {
  return (k / kBContiguousK) * kBLeadingElements + col * kBContiguousK +
         (k % kBContiguousK);
}

__host__ __device__ __forceinline__ void accumulator_matrix_coord(
    int thread_id, int reg_id, int &row, int &col) {
  int tid_mma_col = thread_id % 4;
  int tid_row = (thread_id / 4) % 8;
  int tid_m_block = thread_id / 32;

  int reg_col = reg_id & 0x1;
  int reg_row_block = (reg_id >> 1) & 0x1;

  row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  col = 2 * tid_mma_col + reg_col;
}

__device__ __forceinline__ uint32_t pack_f16_pair(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

inline uint16_t f32_to_f16(float f32) {
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

inline float f16_to_f32(uint16_t f16) {
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

}  // namespace wgmma_test

#endif  // TEST_SRC_HOPPER_WGMMA_TENSOR_WGMMA_TEST_CUH
