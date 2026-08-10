// Shared helpers for Hopper WGMMA integration tests.
//
// Data-type-specific tests live in wgmma_<type>_test.cu files. This header
// keeps common WGMMA shape/layout helpers, type dispatch, CUDA runners, random
// input generation, and CPU references in one place so new data types exercise
// the same path.

#ifndef TEST_SRC_WGMMA_TENSOR_WGMMA_TEST_CUH
#define TEST_SRC_WGMMA_TENSOR_WGMMA_TEST_CUH

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
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
constexpr int kSharedBElements = 512;
constexpr int kBLeadingByteOffset = 256;
constexpr int kBStrideByteOffset = 16;
constexpr int kBContiguousK =
    kBStrideByteOffset / static_cast<int>(sizeof(uint16_t));
constexpr int kBLeadingElements =
    kBLeadingByteOffset / static_cast<int>(sizeof(uint16_t));
constexpr int kSharedBBytes = 1024;
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

template <typename AccumulatorT>
struct WgmmaRunResult {
  cudaError_t setup_error = cudaSuccess;
  cudaError_t input_copy_error = cudaSuccess;
  cudaError_t launch_error = cudaSuccess;
  cudaError_t sync_error = cudaSuccess;
  cudaError_t copy_error = cudaSuccess;
  std::vector<AccumulatorT> output;
};

using RunResult = WgmmaRunResult<float>;

enum class WgmmaKind {
  BF16,
  TF32,
  FP8E4M3E4M3,
  FP8E5M2E5M2,
  FP8E4M3E5M2,
  FP8E5M2E4M3,
  S8S8,
  U8U8,
  S8U8,
  U8S8,
  B1
};

template <WgmmaKind Kind>
struct WgmmaTraits;

__host__ __device__ __forceinline__ int c_matrix_index(int row, int col) {
  return row * kN + col;
}

__host__ __device__ __forceinline__ int a_matrix_index(int row, int k) {
  return row * kK + k;
}

__host__ __device__ __forceinline__ int b_matrix_index(int col, int k) {
  return col * kK + k;
}

template <int K>
__host__ __device__ __forceinline__ int a_matrix_index_k(int row, int k) {
  return row * K + k;
}

template <int K>
__host__ __device__ __forceinline__ int b_matrix_index_k(int col, int k) {
  return col * K + k;
}

template <typename ElementT>
__host__ __device__ __forceinline__ int b_contiguous_elements() {
  return kBStrideByteOffset / static_cast<int>(sizeof(ElementT));
}

template <typename ElementT>
__host__ __device__ __forceinline__ int b_leading_elements() {
  return kBLeadingByteOffset / static_cast<int>(sizeof(ElementT));
}

template <typename ElementT>
__host__ __device__ __forceinline__ int b_smem_index_typed(int col, int k) {
  const int contiguous_k = b_contiguous_elements<ElementT>();
  return (k / contiguous_k) * b_leading_elements<ElementT>() +
         col * contiguous_k + (k % contiguous_k);
}

__host__ __device__ __forceinline__ int b_smem_index(int col, int k) {
  return b_smem_index_typed<uint16_t>(col, k);
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

__host__ __device__ __forceinline__ uint32_t pack_u8x4(uint8_t x0, uint8_t x1,
                                                       uint8_t x2,
                                                       uint8_t x3) {
  return static_cast<uint32_t>(x0) | (static_cast<uint32_t>(x1) << 8) |
         (static_cast<uint32_t>(x2) << 16) |
         (static_cast<uint32_t>(x3) << 24);
}

inline uint32_t f32_bits(float f32) {
  uint32_t bits;
  std::memcpy(&bits, &f32, sizeof(float));
  return bits;
}

inline float bits_to_f32(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

inline uint16_t f32_to_f16(float f32) {
  uint32_t bits = f32_bits(f32);

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
  uint32_t f32_bits_value = (sign << 31) | (f32_exp << 23) | f32_frac;
  return bits_to_f32(f32_bits_value);
}

inline uint16_t f32_to_bf16(float f32) {
  uint32_t bits = f32_bits(f32);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7FFFu + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_to_f32(uint16_t bf16) {
  return bits_to_f32(static_cast<uint32_t>(bf16) << 16);
}

inline float tf32_round(float f32) {
  return bits_to_f32(f32_bits(f32) & 0xFFFFE000u);
}

inline uint8_t f32_to_e4m3(float f32) {
  __nv_fp8_e4m3 value(f32);
  return value.__x;
}

inline uint8_t f32_to_e5m2(float f32) {
  __nv_fp8_e5m2 value(f32);
  return value.__x;
}

inline float e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

inline float e5m2_to_f32(uint8_t bits) {
  __nv_fp8_e5m2 value;
  value.__x = bits;
  return static_cast<float>(value);
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

template <typename AccumulatorT>
__device__ __forceinline__ void operand_barrier(uint32_t (&a_regs)[4],
                                                AccumulatorT (&d)[4]) {
  if constexpr (std::is_same<AccumulatorT, float>::value) {
    asm volatile("" : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3]), "+f"(d[0]), "+f"(d[1]), "+f"(d[2]),
                 "+f"(d[3])::"memory");
  } else {
    asm volatile("" : "+r"(a_regs[0]), "+r"(a_regs[1]), "+r"(a_regs[2]),
                 "+r"(a_regs[3]), "+r"(d[0]), "+r"(d[1]), "+r"(d[2]),
                 "+r"(d[3])::"memory");
  }
}

template <typename AccumulatorT>
__device__ __forceinline__ void result_barrier(AccumulatorT (&d)[4]) {
  if constexpr (std::is_same<AccumulatorT, float>::value) {
    asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])::
                     "memory");
  } else {
    asm volatile("" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])::
                     "memory");
  }
}

template <typename StorageT>
__device__ __forceinline__ uint32_t pack_k16_pair(const StorageT *A, int row,
                                                  int reg, int lane) {
  int tid_mma_col = lane % 4;
  int reg_row_block = reg & 0x1;
  int reg_k_block = (reg >> 1) & 0x1;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;
  int a_row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  int k0 = 2 * tid_mma_col + 8 * reg_k_block;
  (void)row;
  return pack_f16_pair(A[a_matrix_index_k<16>(a_row, k0)],
                       A[a_matrix_index_k<16>(a_row, k0 + 1)]);
}

__device__ __forceinline__ uint32_t pack_k8_tf32(const uint32_t *A, int reg,
                                                 int lane) {
  int tid_mma_col = lane % 4;
  int reg_row_block = reg & 0x1;
  int reg_k_block = (reg >> 1) & 0x1;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;
  int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  int k = tid_mma_col + 4 * reg_k_block;
  return A[a_matrix_index_k<8>(row, k)];
}

template <typename StorageT>
__device__ __forceinline__ uint32_t pack_k32_x4(const StorageT *A, int reg,
                                                int lane) {
  int tid_mma_col = lane % 4;
  int reg_row_block = reg & 0x1;
  int reg_k_block = (reg >> 1) & 0x1;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;
  int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  int k0 = 4 * tid_mma_col + 16 * reg_k_block;
  return pack_u8x4(A[a_matrix_index_k<32>(row, k0)],
                   A[a_matrix_index_k<32>(row, k0 + 1)],
                   A[a_matrix_index_k<32>(row, k0 + 2)],
                   A[a_matrix_index_k<32>(row, k0 + 3)]);
}

__device__ __forceinline__ uint32_t pack_k256_b1(const uint32_t *A, int reg,
                                                 int lane) {
  int tid_mma_col = lane % 4;
  int reg_row_block = reg & 0x1;
  int reg_k_block = (reg >> 1) & 0x1;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;
  int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  int word = tid_mma_col + 4 * reg_k_block;
  return A[row * 8 + word];
}

struct WgmmaFloatAccumulator {
  using Accumulator = float;

  static __device__ __forceinline__ void store_result(float *out, int index,
                                                      float value) {
    out[index] = value;
  }
};

struct WgmmaS32Accumulator {
  using Accumulator = int32_t;

  static __device__ __forceinline__ void store_result(int32_t *out, int index,
                                                      int32_t value) {
    out[index] = value;
  }
};

template <>
struct WgmmaTraits<WgmmaKind::BF16> : WgmmaFloatAccumulator {
  using AStorage = uint16_t;
  using BStorage = uint16_t;
  static constexpr int kKValue = 16;
  static constexpr int kAStorageElements = kM * kKValue;
  static constexpr int kBStorageElements = kN * kKValue;
  static constexpr int kBSharedElements = kSharedBBytes / sizeof(BStorage);
  static constexpr const char *kName = "BF16";

  static __device__ __forceinline__ uint32_t load_a_reg(const AStorage *A,
                                                        int reg, int lane) {
    return pack_k16_pair(A, 0, reg, lane);
  }

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11, %12;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1), "n"(0));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return bf16_to_f32(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return bf16_to_f32(B[b_matrix_index_k<kKValue>(col, k)]);
  }

  static AStorage random_a(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return f32_to_bf16(dist(gen));
  }

  static BStorage random_b(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return f32_to_bf16(dist(gen));
  }

  static float random_c(std::mt19937 &gen, float min_value, float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return dist(gen);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::TF32> : WgmmaFloatAccumulator {
  using AStorage = uint32_t;
  using BStorage = uint32_t;
  static constexpr int kKValue = 8;
  static constexpr int kAStorageElements = kM * kKValue;
  static constexpr int kBStorageElements = kN * kKValue;
  static constexpr int kBSharedElements = kSharedBBytes / sizeof(BStorage);
  static constexpr const char *kName = "TF32";

  static __device__ __forceinline__ uint32_t load_a_reg(const AStorage *A,
                                                        int reg, int lane) {
    return pack_k8_tf32(A, reg, lane);
  }

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return tf32_round(bits_to_f32(A[a_matrix_index_k<kKValue>(row, k)]));
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return tf32_round(bits_to_f32(B[b_matrix_index_k<kKValue>(col, k)]));
  }

  static AStorage random_a(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return f32_bits(dist(gen));
  }

  static BStorage random_b(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return f32_bits(dist(gen));
  }

  static float random_c(std::mt19937 &gen, float min_value, float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return dist(gen);
  }
};

template <WgmmaKind Kind>
struct WgmmaFp8TraitsBase : WgmmaFloatAccumulator {
  using AStorage = uint8_t;
  using BStorage = uint8_t;
  static constexpr int kKValue = 32;
  static constexpr int kAStorageElements = kM * kKValue;
  static constexpr int kBStorageElements = kN * kKValue;
  static constexpr int kBSharedElements = kSharedBBytes / sizeof(BStorage);

  static __device__ __forceinline__ uint32_t load_a_reg(const AStorage *A,
                                                        int reg, int lane) {
    return pack_k32_x4(A, reg, lane);
  }

  static AStorage random_a(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    if constexpr (Kind == WgmmaKind::FP8E4M3E4M3 ||
                  Kind == WgmmaKind::FP8E4M3E5M2) {
      return f32_to_e4m3(dist(gen));
    } else {
      return f32_to_e5m2(dist(gen));
    }
  }

  static BStorage random_b(std::mt19937 &gen, float min_value,
                           float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    if constexpr (Kind == WgmmaKind::FP8E4M3E4M3 ||
                  Kind == WgmmaKind::FP8E5M2E4M3) {
      return f32_to_e4m3(dist(gen));
    } else {
      return f32_to_e5m2(dist(gen));
    }
  }

  static float random_c(std::mt19937 &gen, float min_value, float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return dist(gen);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::FP8E4M3E4M3>
    : WgmmaFp8TraitsBase<WgmmaKind::FP8E4M3E4M3> {
  static constexpr const char *kName = "FP8E4M3E4M3";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return e4m3_to_f32(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return e4m3_to_f32(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::FP8E5M2E5M2>
    : WgmmaFp8TraitsBase<WgmmaKind::FP8E5M2E5M2> {
  static constexpr const char *kName = "FP8E5M2E5M2";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return e5m2_to_f32(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return e5m2_to_f32(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::FP8E4M3E5M2>
    : WgmmaFp8TraitsBase<WgmmaKind::FP8E4M3E5M2> {
  static constexpr const char *kName = "FP8E4M3E5M2";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e5m2 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return e4m3_to_f32(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return e5m2_to_f32(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::FP8E5M2E4M3>
    : WgmmaFp8TraitsBase<WgmmaKind::FP8E5M2E4M3> {
  static constexpr const char *kName = "FP8E5M2E4M3";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      float &d0, float &d1, float &d2, float &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p, %10, %11;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d), "n"(1), "n"(1));
  }

  static float a_value(const std::vector<AStorage> &A, int row, int k) {
    return e5m2_to_f32(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static float b_value(const std::vector<BStorage> &B, int col, int k) {
    return e4m3_to_f32(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <WgmmaKind Kind>
struct WgmmaInt8TraitsBase : WgmmaS32Accumulator {
  using AStorage = uint8_t;
  using BStorage = uint8_t;
  static constexpr int kKValue = 32;
  static constexpr int kAStorageElements = kM * kKValue;
  static constexpr int kBStorageElements = kN * kKValue;
  static constexpr int kBSharedElements = kSharedBBytes / sizeof(BStorage);

  static __device__ __forceinline__ uint32_t load_a_reg(const AStorage *A,
                                                        int reg, int lane) {
    return pack_k32_x4(A, reg, lane);
  }

  static AStorage random_a(std::mt19937 &gen, float, float) {
    if constexpr (Kind == WgmmaKind::S8S8 || Kind == WgmmaKind::S8U8) {
      std::uniform_int_distribution<int> dist(-4, 4);
      return static_cast<uint8_t>(static_cast<int8_t>(dist(gen)));
    } else {
      std::uniform_int_distribution<int> dist(0, 8);
      return static_cast<uint8_t>(dist(gen));
    }
  }

  static BStorage random_b(std::mt19937 &gen, float, float) {
    if constexpr (Kind == WgmmaKind::S8S8 || Kind == WgmmaKind::U8S8) {
      std::uniform_int_distribution<int> dist(-4, 4);
      return static_cast<uint8_t>(static_cast<int8_t>(dist(gen)));
    } else {
      std::uniform_int_distribution<int> dist(0, 8);
      return static_cast<uint8_t>(dist(gen));
    }
  }

  static int32_t random_c(std::mt19937 &gen, float, float) {
    std::uniform_int_distribution<int32_t> dist(-32, 32);
    return dist(gen);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::S8S8>
    : WgmmaInt8TraitsBase<WgmmaKind::S8S8> {
  static constexpr const char *kName = "S8S8";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p;\n"
        "}\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d));
  }

  static int32_t a_value(const std::vector<AStorage> &A, int row, int k) {
    return static_cast<int8_t>(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static int32_t b_value(const std::vector<BStorage> &B, int col, int k) {
    return static_cast<int8_t>(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::U8U8>
    : WgmmaInt8TraitsBase<WgmmaKind::U8U8> {
  static constexpr const char *kName = "U8U8";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p;\n"
        "}\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d));
  }

  static int32_t a_value(const std::vector<AStorage> &A, int row, int k) {
    return A[a_matrix_index_k<kKValue>(row, k)];
  }

  static int32_t b_value(const std::vector<BStorage> &B, int col, int k) {
    return B[b_matrix_index_k<kKValue>(col, k)];
  }
};

template <>
struct WgmmaTraits<WgmmaKind::S8U8>
    : WgmmaInt8TraitsBase<WgmmaKind::S8U8> {
  static constexpr const char *kName = "S8U8";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p;\n"
        "}\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d));
  }

  static int32_t a_value(const std::vector<AStorage> &A, int row, int k) {
    return static_cast<int8_t>(A[a_matrix_index_k<kKValue>(row, k)]);
  }

  static int32_t b_value(const std::vector<BStorage> &B, int col, int k) {
    return B[b_matrix_index_k<kKValue>(col, k)];
  }
};

template <>
struct WgmmaTraits<WgmmaKind::U8S8>
    : WgmmaInt8TraitsBase<WgmmaKind::U8S8> {
  static constexpr const char *kName = "U8S8";

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p;\n"
        "}\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d));
  }

  static int32_t a_value(const std::vector<AStorage> &A, int row, int k) {
    return A[a_matrix_index_k<kKValue>(row, k)];
  }

  static int32_t b_value(const std::vector<BStorage> &B, int col, int k) {
    return static_cast<int8_t>(B[b_matrix_index_k<kKValue>(col, k)]);
  }
};

template <>
struct WgmmaTraits<WgmmaKind::B1> : WgmmaS32Accumulator {
  using AStorage = uint32_t;
  using BStorage = uint32_t;
  static constexpr int kKValue = 256;
  static constexpr int kAStorageElements = kM * (kKValue / 32);
  static constexpr int kBStorageElements = kN * (kKValue / 32);
  static constexpr int kBStorageK = kKValue / 32;
  static constexpr int kBSharedElements = kSharedBBytes / sizeof(BStorage);
  static constexpr const char *kName = "B1";

  static __device__ __forceinline__ uint32_t load_a_reg(const AStorage *A,
                                                        int reg, int lane) {
    return pack_k256_b1(A, reg, lane);
  }

  static __device__ __forceinline__ void invoke(
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b,
      int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, int32_t scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "%8, p;\n"
        "}\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(scale_d));
  }

  static int32_t a_value(const std::vector<AStorage> &A, int row, int k) {
    uint32_t word = A[row * kBStorageK + k / 32];
    return static_cast<int32_t>((word >> (k % 32)) & 1u);
  }

  static int32_t b_value(const std::vector<BStorage> &B, int col, int k) {
    uint32_t word = B[col * kBStorageK + k / 32];
    return static_cast<int32_t>((word >> (k % 32)) & 1u);
  }

  static AStorage random_a(std::mt19937 &gen, float, float) {
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);
    return dist(gen);
  }

  static BStorage random_b(std::mt19937 &gen, float, float) {
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);
    return dist(gen);
  }

  static int32_t random_c(std::mt19937 &gen, float, float) {
    std::uniform_int_distribution<int32_t> dist(-32, 32);
    return dist(gen);
  }
};

template <WgmmaKind Kind>
__global__ void wgmma_m64n8_matrix_kernel(
    typename WgmmaTraits<Kind>::Accumulator *out,
    const typename WgmmaTraits<Kind>::AStorage *A,
    const typename WgmmaTraits<Kind>::BStorage *B,
    const typename WgmmaTraits<Kind>::Accumulator *C, int32_t scale_d) {
  using Traits = WgmmaTraits<Kind>;
  using BStorage = typename Traits::BStorage;
  using Accumulator = typename Traits::Accumulator;

  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(16) BStorage smem_b[Traits::kBSharedElements];

  for (int i = threadIdx.x; i < Traits::kBSharedElements; i += blockDim.x) {
    smem_b[i] = 0;
  }

  constexpr int kBStorageK =
      Kind == WgmmaKind::B1 ? WgmmaTraits<WgmmaKind::B1>::kBStorageK
                            : Traits::kKValue;
  for (int i = threadIdx.x; i < kN * kBStorageK; i += blockDim.x) {
    int col = i / kBStorageK;
    int k = i % kBStorageK;
    smem_b[b_smem_index_typed<BStorage>(col, k)] = B[col * kBStorageK + k];
  }
  __syncthreads();

  uint64_t desc_b =
      make_gmma_k_major_desc(smem_b, kBLeadingByteOffset, kBStrideByteOffset);

  uint32_t a_regs[4];
  for (int reg = 0; reg < 4; ++reg) {
    a_regs[reg] = Traits::load_a_reg(A, reg, threadIdx.x);
  }

  Accumulator d[4];
  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(threadIdx.x, reg, row, col);
    d[reg] = C[c_matrix_index(row, col)];
  }

  operand_barrier(a_regs, d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  Traits::invoke(a_regs[0], a_regs[1], a_regs[2], a_regs[3], desc_b, d[0],
                 d[1], d[2], d[3], scale_d);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  result_barrier(d);

  for (int reg = 0; reg < 4; ++reg) {
    int row = 0;
    int col = 0;
    accumulator_matrix_coord(threadIdx.x, reg, row, col);
    Traits::store_result(out, c_matrix_index(row, col), d[reg]);
  }
}

template <WgmmaKind Kind>
WgmmaRunResult<typename WgmmaTraits<Kind>::Accumulator>
run_wgmma_matrix_kernel(const std::vector<typename WgmmaTraits<Kind>::AStorage>
                            &A,
                        const std::vector<typename WgmmaTraits<Kind>::BStorage>
                            &B,
                        const std::vector<typename WgmmaTraits<Kind>::Accumulator>
                            &C,
                        int32_t scale_d) {
  using Traits = WgmmaTraits<Kind>;
  using AStorage = typename Traits::AStorage;
  using BStorage = typename Traits::BStorage;
  using Accumulator = typename Traits::Accumulator;

  WgmmaRunResult<Accumulator> result;
  result.output.assign(kOutputRegs, Accumulator{});

  if (A.size() != static_cast<size_t>(Traits::kAStorageElements) ||
      B.size() != static_cast<size_t>(Traits::kBStorageElements) ||
      C.size() != static_cast<size_t>(kOutputRegs)) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }

  AStorage *d_A = nullptr;
  BStorage *d_B = nullptr;
  Accumulator *d_C = nullptr;
  Accumulator *d_out = nullptr;

  result.setup_error = cudaMalloc(&d_A, A.size() * sizeof(AStorage));
  if (result.setup_error != cudaSuccess) return result;
  result.setup_error = cudaMalloc(&d_B, B.size() * sizeof(BStorage));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    return result;
  }
  result.setup_error = cudaMalloc(&d_C, C.size() * sizeof(Accumulator));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    return result;
  }
  result.setup_error = cudaMalloc(&d_out, kOutputRegs * sizeof(Accumulator));
  if (result.setup_error != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return result;
  }

  result.input_copy_error =
      cudaMemcpy(d_A, A.data(), A.size() * sizeof(AStorage),
                 cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error =
        cudaMemcpy(d_B, B.data(), B.size() * sizeof(BStorage),
                   cudaMemcpyHostToDevice);
  }
  if (result.input_copy_error == cudaSuccess) {
    result.input_copy_error =
        cudaMemcpy(d_C, C.data(), C.size() * sizeof(Accumulator),
                   cudaMemcpyHostToDevice);
  }

  if (result.input_copy_error == cudaSuccess) {
    cudaGetLastError();
    wgmma_m64n8_matrix_kernel<Kind><<<1, kWarpgroupThreads>>>(
        d_out, d_A, d_B, d_C, scale_d);

    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) {
      result.sync_error = cudaDeviceSynchronize();
    }
  }

  if (result.input_copy_error == cudaSuccess &&
      result.launch_error == cudaSuccess && result.sync_error == cudaSuccess) {
    result.copy_error =
        cudaMemcpy(result.output.data(), d_out,
                   kOutputRegs * sizeof(Accumulator), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_out);
  return result;
}

template <WgmmaKind Kind>
void fill_random_wgmma_inputs(
    std::vector<typename WgmmaTraits<Kind>::AStorage> &A,
    std::vector<typename WgmmaTraits<Kind>::BStorage> &B,
    std::vector<typename WgmmaTraits<Kind>::Accumulator> &C, uint32_t seed,
    float min_value, float max_value) {
  using Traits = WgmmaTraits<Kind>;
  A.resize(Traits::kAStorageElements);
  B.resize(Traits::kBStorageElements);
  C.resize(kOutputRegs);

  std::mt19937 gen(seed);
  for (typename Traits::AStorage &value : A) {
    value = Traits::random_a(gen, min_value, max_value);
  }
  for (typename Traits::BStorage &value : B) {
    value = Traits::random_b(gen, min_value, max_value);
  }
  for (typename Traits::Accumulator &value : C) {
    value = Traits::random_c(gen, min_value, max_value);
  }
}

template <WgmmaKind Kind>
void compute_wgmma_reference(
    const std::vector<typename WgmmaTraits<Kind>::AStorage> &A,
    const std::vector<typename WgmmaTraits<Kind>::BStorage> &B,
    const std::vector<typename WgmmaTraits<Kind>::Accumulator> &C,
    std::vector<typename WgmmaTraits<Kind>::Accumulator> &D_ref,
    int32_t scale_d) {
  using Traits = WgmmaTraits<Kind>;
  using Accumulator = typename Traits::Accumulator;
  D_ref.assign(kOutputRegs, Accumulator{});

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      if constexpr (std::is_same<Accumulator, float>::value) {
        float sum = 0.0f;
        for (int k = 0; k < Traits::kKValue; ++k) {
          sum += Traits::a_value(A, row, k) * Traits::b_value(B, col, k);
        }
        D_ref[c_matrix_index(row, col)] =
            sum + (scale_d != 0 ? C[c_matrix_index(row, col)] : 0.0f);
      } else {
        int64_t sum = 0;
        for (int k = 0; k < Traits::kKValue; ++k) {
          sum += static_cast<int64_t>(Traits::a_value(A, row, k)) *
                 static_cast<int64_t>(Traits::b_value(B, col, k));
        }
        if (scale_d != 0) {
          sum += C[c_matrix_index(row, col)];
        }
        sum = std::max<int64_t>(std::numeric_limits<int32_t>::min(),
                                std::min<int64_t>(
                                    std::numeric_limits<int32_t>::max(), sum));
        D_ref[c_matrix_index(row, col)] = static_cast<int32_t>(sum);
      }
    }
  }
}

template <typename AccumulatorT>
void assert_run_success(const WgmmaRunResult<AccumulatorT> &result,
                        size_t expected_output_size = kOutputRegs) {
  ASSERT_EQ(result.setup_error, cudaSuccess)
      << "CUDA setup error: " << cudaGetErrorString(result.setup_error);
  ASSERT_EQ(result.input_copy_error, cudaSuccess)
      << "CUDA input copy error: "
      << cudaGetErrorString(result.input_copy_error);
  ASSERT_EQ(result.launch_error, cudaSuccess)
      << "CUDA launch error: " << cudaGetErrorString(result.launch_error);
  ASSERT_EQ(result.sync_error, cudaSuccess)
      << "CUDA sync error: " << cudaGetErrorString(result.sync_error);
  ASSERT_EQ(result.copy_error, cudaSuccess)
      << "CUDA copy error: " << cudaGetErrorString(result.copy_error);

  ASSERT_EQ(result.output.size(), expected_output_size);
}

template <WgmmaKind Kind>
void validate_wgmma_matrix_result(
    const std::vector<typename WgmmaTraits<Kind>::Accumulator> &actual,
    const std::vector<typename WgmmaTraits<Kind>::Accumulator> &expected,
    float absolute_tolerance, float relative_tolerance) {
  using Accumulator = typename WgmmaTraits<Kind>::Accumulator;
  ASSERT_EQ(actual.size(), static_cast<size_t>(kOutputRegs));
  ASSERT_EQ(expected.size(), static_cast<size_t>(kOutputRegs));

  for (int row = 0; row < kM; ++row) {
    for (int col = 0; col < kN; ++col) {
      int index = c_matrix_index(row, col);
      if constexpr (std::is_same<Accumulator, float>::value) {
        float tolerance =
            std::max(absolute_tolerance,
                     std::abs(expected[index]) * relative_tolerance);
        EXPECT_NEAR(actual[index], expected[index], tolerance)
            << "Mismatch at D[" << row << "][" << col << "]"
            << " index " << index << " (got: " << actual[index]
            << ", expected: " << expected[index] << ")";
      } else {
        EXPECT_EQ(actual[index], expected[index])
            << "Mismatch at D[" << row << "][" << col << "]"
            << " index " << index << " (got: " << actual[index]
            << ", expected: " << expected[index] << ")";
      }
    }
  }
}

template <WgmmaKind Kind>
void run_random_wgmma_case(uint32_t seed, int32_t scale_d, float min_value,
                           float max_value, float absolute_tolerance,
                           float relative_tolerance) {
  using Traits = WgmmaTraits<Kind>;
  std::vector<typename Traits::AStorage> A;
  std::vector<typename Traits::BStorage> B;
  std::vector<typename Traits::Accumulator> C;
  std::vector<typename Traits::Accumulator> D_ref;

  fill_random_wgmma_inputs<Kind>(A, B, C, seed, min_value, max_value);
  compute_wgmma_reference<Kind>(A, B, C, D_ref, scale_d);

  WgmmaRunResult<typename Traits::Accumulator> result =
      run_wgmma_matrix_kernel<Kind>(A, B, C, scale_d);
  ASSERT_NO_FATAL_FAILURE(assert_run_success(result));
  validate_wgmma_matrix_result<Kind>(result.output, D_ref, absolute_tolerance,
                                     relative_tolerance);
}

}  // namespace wgmma_test

#endif  // TEST_SRC_WGMMA_TENSOR_WGMMA_TEST_CUH
