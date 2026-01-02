// Copyright (c) 2025 University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "tensor_mma.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx_ir.h"
// Forward declarations needed by ptx.tab.h
typedef void *yyscan_t;
class ptx_recognizer;
#include "../cuda-sim/ptx.tab.h"
#include "../gpu-sim.h"
#include "../../libcuda/gpgpu_context.h"

namespace flash_gpgpu_sim {

// ============================================================================
// IEEE 754 Floating-Point Format Constants
// ============================================================================

// F16 (IEEE 754 half-precision) format constants
// F16: 1 sign bit, 5 exponent bits, 10 mantissa bits
namespace F16Constants {
  constexpr uint32_t SIGN_MASK = 0x1;
  constexpr uint32_t EXPONENT_MASK = 0x1F;
  constexpr uint32_t MANTISSA_MASK = 0x3FF;

  constexpr int SIGN_SHIFT = 15;
  constexpr int EXPONENT_SHIFT = 10;
  constexpr int MANTISSA_SHIFT = 13;  // F16 to F32 mantissa shift

  constexpr int EXPONENT_BIAS = 15;
  constexpr int MAX_EXPONENT = 31;
  constexpr uint16_t INFINITY_BITS = 0x7C00;

  // Denormal number scaling: mantissa / 1024.0 / 16384.0
  constexpr float DENORMAL_SCALE_1 = 1024.0f;   // 2^10 (mantissa bits)
  constexpr float DENORMAL_SCALE_2 = 16384.0f;  // 2^14 (exponent adjustment)
}

// F32 (IEEE 754 single-precision) format constants
namespace F32Constants {
  constexpr int SIGN_SHIFT = 31;
  constexpr int EXPONENT_SHIFT = 23;
  constexpr uint32_t EXPONENT_MASK = 0xFF;
  constexpr uint32_t MANTISSA_MASK_F16 = 0x3FF;  // When converting from F16

  constexpr int EXPONENT_BIAS = 127;
  constexpr int MANTISSA_SHIFT_FROM_F16 = 13;  // F32 mantissa shift from F16
}

// BF16 (bfloat16) format constants
// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
namespace BF16Constants {
  constexpr int SHIFT = 16;  // BF16 occupies upper 16 bits of F32
}

// TF32 (TensorFloat-32) format constants
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (uses F32 container)
namespace TF32Constants {
  constexpr uint32_t MANTISSA_MASK = 0x1FFF;  // Lower 13 bits to zero out
  constexpr uint32_t ROUND_BIT = 0x1000;      // Bit 12 (rounding decision bit)
  // TF32 has 10 mantissa bits, F32 has 23, so zero out lower 13 bits (23-10=13)
}

// ============================================================================
// Integer Saturation Constants
// ============================================================================

namespace SaturationLimits {
  // Signed 8-bit integer range [-128, 127]
  constexpr int32_t S8_MAX = 127;
  constexpr int32_t S8_MIN = -128;

  // Unsigned 8-bit integer range [0, 255]
  constexpr int32_t U8_MAX = 255;
  constexpr int32_t U8_MIN = 0;

  // Signed 4-bit integer range [-8, 7]
  constexpr int32_t S4_MAX = 7;
  constexpr int32_t S4_MIN = -8;

  // Unsigned 4-bit integer range [0, 15]
  constexpr int32_t U4_MAX = 15;
  constexpr int32_t U4_MIN = 0;
}

// ============================================================================
// Type Conversion Helper Functions
// ============================================================================

// Helper: Convert F16 to F32
float mma_f16_to_f32(uint16_t f16) {
  using namespace F16Constants;
  using namespace F32Constants;

  uint32_t sign = (f16 >> F16Constants::SIGN_SHIFT) & F16Constants::SIGN_MASK;
  uint32_t exp = (f16 >> F16Constants::EXPONENT_SHIFT) & F16Constants::EXPONENT_MASK;
  uint32_t frac = f16 & F16Constants::MANTISSA_MASK;

  if (exp == 0) {
    if (frac == 0) return sign ? -0.0f : 0.0f;
    // Denormal: mantissa / 2^10 / 2^14
    float result = frac / DENORMAL_SCALE_1 / DENORMAL_SCALE_2;
    return sign ? -result : result;
  } else if (exp == MAX_EXPONENT) {
    // Inf or NaN
    return frac ? NAN : (sign ? -INFINITY : INFINITY);
  }

  // Normalized: convert exponent from F16 bias to F32 bias
  uint32_t f32_exp = exp - F16Constants::EXPONENT_BIAS + F32Constants::EXPONENT_BIAS;
  uint32_t f32_frac = frac << MANTISSA_SHIFT;
  uint32_t f32_bits = (sign << F32Constants::SIGN_SHIFT) | (f32_exp << F32Constants::EXPONENT_SHIFT) | f32_frac;

  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Convert F32 to F16
uint16_t mma_f32_to_f16(float f32) {
  using namespace F16Constants;
  using namespace F32Constants;

  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));

  uint32_t sign = (f32_bits >> F32Constants::SIGN_SHIFT) & F16Constants::SIGN_MASK;
  int32_t exp = ((f32_bits >> F32Constants::EXPONENT_SHIFT) & F32Constants::EXPONENT_MASK)
                - F32Constants::EXPONENT_BIAS + F16Constants::EXPONENT_BIAS;
  uint32_t frac = (f32_bits >> F32Constants::MANTISSA_SHIFT_FROM_F16) & F32Constants::MANTISSA_MASK_F16;

  if (exp <= 0) return sign << F16Constants::SIGN_SHIFT;  // Flush to zero
  if (exp >= F16Constants::MAX_EXPONENT) return (sign << F16Constants::SIGN_SHIFT) | F16Constants::INFINITY_BITS;  // Inf

  return (sign << F16Constants::SIGN_SHIFT) | (exp << F16Constants::EXPONENT_SHIFT) | frac;
}

// Helper: Convert BF16 to F32
float mma_bf16_to_f32(uint16_t bf16) {
  using namespace BF16Constants;
  // BF16 occupies the upper 16 bits of F32
  uint32_t f32_bits = static_cast<uint32_t>(bf16) << SHIFT;
  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Convert F32 to BF16
uint16_t mma_f32_to_bf16(float f32) {
  using namespace BF16Constants;
  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));
  // BF16 is the upper 16 bits of F32 (truncation)
  return static_cast<uint16_t>(f32_bits >> SHIFT);
}

// Helper: Round float32 to TensorFloat-32 precision
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (19 bits total)
// Per NVIDIA spec: "rounding and handling of subnormal inputs are unspecified"
// Using simple truncation (RTZ) as the most hardware-efficient approach
float mma_tf32_round(float f32) {
  using namespace TF32Constants;

  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));

  // TF32 has 10 mantissa bits, F32 has 23, so zero out lower 13 bits (23-10=13)
  // Simple truncation: just clear the lower 13 bits
  f32_bits &= ~MANTISSA_MASK;

  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Saturate S8 value
int8_t mma_saturate_s8(int32_t val) {
  using namespace SaturationLimits;
  if (val > S8_MAX) return S8_MAX;
  if (val < S8_MIN) return S8_MIN;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U8 value
uint8_t mma_saturate_u8(int32_t val) {
  using namespace SaturationLimits;
  if (val > U8_MAX) return U8_MAX;
  if (val < U8_MIN) return U8_MIN;
  return static_cast<uint8_t>(val);
}

// Helper: Saturate S4 value
int8_t mma_saturate_s4(int32_t val) {
  using namespace SaturationLimits;
  if (val > S4_MAX) return S4_MAX;
  if (val < S4_MIN) return S4_MIN;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U4 value
uint8_t mma_saturate_u4(int32_t val) {
  using namespace SaturationLimits;
  if (val > U4_MAX) return U4_MAX;
  if (val < U4_MIN) return U4_MIN;
  return static_cast<uint8_t>(val);
}

// Helper: Thread-to-element offset for MMA shapes
// Maps thread ID to matrix element offset based on shape and layout
unsigned mma_thread_to_element_offset(unsigned thread_id, mma_shape_type shape,
                                      mma_layout_mode layout,
                                      unsigned char type_size, unsigned stride) {
  unsigned row, col;

  switch (shape) {
    case MMA_M16N8K8:
      // M16N8K8: 16x8 output matrix, K=8
      // 32 threads, each thread maps to specific elements
      row = thread_id / 4;
      col = (thread_id % 4) * 2;
      break;

    case MMA_M16N8K16:
      // M16N8K16: 16x8 output matrix, K=16
      // Different thread mapping due to larger K dimension
      row = thread_id / 4;
      col = (thread_id % 4) * 2;
      // K=16 affects how many elements per thread, not the row/col mapping
      break;

    case MMA_M16N8K32:
      // M16N8K32: 16x8 output matrix, K=32
      // Even larger K dimension for integer types
      row = thread_id / 4;
      col = (thread_id % 4) * 2;
      break;

    default:
      // Unknown shape, fall back to simple linear offset
      return thread_id * type_size;
  }

  // Apply layout mode
  if (layout == MMA_ROW_COL || layout == MMA_ROW_ROW) {
    // Row-major layout
    return row * stride + col * type_size;
  } else {
    // Column-major layout (COL_ROW, COL_COL)
    return col * stride + row * type_size;
  }
}

// S8/U8 integer MMA implementation for M16N8K16
// Implements functional simulation of integer tensor core operations
void tensor_mma_s8_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t inst, int M, int N, int K,
                        bool saturate, unsigned tid,
                        const operand_info &dst) {
  // M16N8K16 implementation for S8 integers
  // Based on PTX ISA spec for integer tensor cores
  //
  // Fragment distribution pattern for 32 threads:
  //   groupID = lane_id >> 2  (lane_id / 4, range 0-7)
  //   threadID_in_group = lane_id % 4  (range 0-3)
  //
  // Each thread holds:
  //   A: 8 S8 values (2 U32 registers, packed 4 values each) - from 16×16 matrix A
  //   B: 4 S8 values (1 U32 register, packed) - from 16×8 matrix B
  //   C/D: 4 S32 values (4 S32 registers) - from 16×8 output matrix D
  //
  // Matrix A[16×16] fragment distribution (row-major):
  //   row = groupID       for a[i] where i < 4
  //       = groupID + 8   for a[i] where i >= 4
  //   col = (threadID_in_group * 4) + (i & 0x3)   for a[i] where i = {0,..,7}
  //
  // Matrix B[16×8] fragment distribution (column-major):
  //   row = (threadID_in_group * 4) + i    for b[i] where i = {0,..,3}
  //   col = groupID
  //
  // Output matrix D[16×8] mapping:
  //   row = groupID       for d[i] where i < 2
  //       = groupID + 8   for d[i] where i >= 2
  //   col = (threadID_in_group * 2) + (i & 0x1)   for d[i] where i = {0,..,3}

  // Step 1: Collect all fragments from all 32 threads into full matrices
  int32_t A_mat[M * K];  // 16×16, stored as int32 for computation
  int32_t B_mat[K * N];  // 16×8
  int32_t C_mat[M * N];  // 16×8
  int64_t D_mat[M * N];  // 16×8, use int64 to detect overflow

  // Initialize matrices
  for (int i = 0; i < M * K; i++) A_mat[i] = 0;
  for (int i = 0; i < K * N; i++) B_mat[i] = 0;
  for (int i = 0; i < M * N; i++) C_mat[i] = 0;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id >> 2;  // lane_id / 4, range 0-7
    unsigned threadID_in_group = lane_id % 4;  // range 0-3

    // A: 2 registers (8 S8 values total)
    // B: 1 register (4 S8 values)
    // C: 4 registers (4 S32 values)
    ptx_reg_t a_regs[2], b_regs[1], c_regs[4];
    thread->get_vector_operand_values(src_a, a_regs, 2);
    thread->get_vector_operand_values(src_b, b_regs, 1);
    thread->get_vector_operand_values(src_c, c_regs, 4);

    // Unpack A fragments (8 S8 values packed in 2 U32 registers)
    // a_regs[0] contains a[0..3], a_regs[1] contains a[4..7]
    int8_t a_vals[8];
    memcpy(&a_vals[0], &a_regs[0].u32, 4);  // a[0..3]
    memcpy(&a_vals[4], &a_regs[1].u32, 4);  // a[4..7]

    // A fragment distribution for M16N8K16 (row-major)
    // a[0..3] are in row=groupID, a[4..7] are in row=groupID+8
    int a_row0 = groupID;
    int a_row1 = groupID + 8;

    // Column mapping: col = (threadID_in_group * 4) + (i & 0x3)
    int col_base = threadID_in_group * 4;

    // A is row-major: A[row][col] = A[row * K + col]
    // Place a[0..3] in row=groupID
    for (int i = 0; i < 4; i++) {
      int col = col_base + (i & 0x3);
      A_mat[a_row0 * K + col] = a_vals[i];
    }
    // Place a[4..7] in row=groupID+8
    for (int i = 4; i < 8; i++) {
      int col = col_base + (i & 0x3);
      A_mat[a_row1 * K + col] = a_vals[i];
    }

    // Unpack B fragments (4 S8 values packed in 1 U32 register)
    int8_t b_vals[4];
    memcpy(&b_vals[0], &b_regs[0].u32, 4);

    // B fragment distribution for M16N8K16 (column-major)
    // row = (threadID_in_group * 4) + i for b[i] where i = {0,..,3}
    // col = groupID
    int b_col = groupID;
    int row_base = threadID_in_group * 4;

    // B is column-major: B[n][k] stored as B[n*K + k]
    for (int i = 0; i < 4; i++) {
      int row = row_base + i;
      B_mat[b_col * K + row] = b_vals[i];
    }

    // Place C fragments (4 S32 accumulator values)
    // row = groupID for c[0..1], groupID+8 for c[2..3]
    // col = (threadID_in_group * 2) + (i & 0x1)
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;

    C_mat[c_row0 * N + c_col0] = c_regs[0].s32;
    C_mat[c_row0 * N + c_col1] = c_regs[1].s32;
    C_mat[c_row1 * N + c_col0] = c_regs[2].s32;
    C_mat[c_row1 * N + c_col1] = c_regs[3].s32;
  }

  // Step 2: Perform matrix multiplication D = A * B + C
  // Use int64 to detect overflow, then saturate to S32 if enabled
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int64_t sum = 0;
      for (int k = 0; k < K; k++) {
        // A is row-major: A[m][k], B is column-major: B[n][k]
        sum += (int64_t)A_mat[m * K + k] * (int64_t)B_mat[n * K + k];
      }
      // Add accumulator
      D_mat[m * N + n] = sum + (int64_t)C_mat[m * N + n];

      // Apply saturation if enabled (.satfinite modifier)
      if (saturate) {
        if (D_mat[m * N + n] > INT32_MAX) D_mat[m * N + n] = INT32_MAX;
        if (D_mat[m * N + n] < INT32_MIN) D_mat[m * N + n] = INT32_MIN;
      }
    }
  }

  // Step 3: Distribute results back to threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id >> 2;
    unsigned threadID_in_group = lane_id % 4;

    ptx_reg_t d_regs[4];
    int d_row0 = groupID;
    int d_row1 = groupID + 8;
    int d_col0 = threadID_in_group * 2;
    int d_col1 = threadID_in_group * 2 + 1;

    d_regs[0].s32 = (int32_t)D_mat[d_row0 * N + d_col0];
    d_regs[1].s32 = (int32_t)D_mat[d_row0 * N + d_col1];
    d_regs[2].s32 = (int32_t)D_mat[d_row1 * N + d_col0];
    d_regs[3].s32 = (int32_t)D_mat[d_row1 * N + d_col1];

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2], d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_s8_impl completed for M%dN%dK%d\n", M, N, K);
  }
}

// NEW: Tensor MMA instruction implementations (separate from WMMA)
void tensor_mma_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t inst) {
  // Implementation for MMA instructions (separate from WMMA)
  // Supports: F16/F32 (M16N8K8), S8/S32 (M16N8K16)
  //
  // PTX MMA instruction format:
  // mma.sync.aligned.shape.row.col{.satfinite}.type.type.type.type d, a, b, c;
  //   shape: m16n8k8, m16n8k16, etc.
  //   layout: row/col for matrices A and B
  //   types: data types for D, A, B, C

  // Get MMA instruction properties
  mma_shape_type shape = pI->get_mma_shape();
  mma_layout_mode layout_a, layout_b;
  pI->get_mma_layout(layout_a, layout_b);
  bool saturate = pI->get_mma_saturate();

  // Determine data type from PTX instruction scalar types
  // For MMA: .m16n8k8.row.col.f32.bf16.bf16.f32
  //   First type = accumulator output type (f32)
  //   Second type = A matrix input type (bf16, f16, s8, etc.)
  // Get the input data type (first type in scalar type list, which is the accumulator for MMA)
  // For MMA, the operand types are: D (output), A (input), B (input), C (accumulator)
  // Scalar types list order: output type, then input types
  std::list<int> scalar_types = pI->get_scalar_type();
  // For MMA with .f32.bf16.bf16.f32, get_type() returns first type (f32 output),
  // get_type2() would return second if it exists
  // We need to examine the full scalar type list to find input types
  // Typically: scalar_types = {F32_TYPE, BF16_TYPE, BF16_TYPE, F32_TYPE} for bf16 MMA

  int input_type = F16_TYPE;  // Default to F16
  if (scalar_types.size() >= 2) {
    auto it = scalar_types.begin();
    ++it;  // Skip first type (accumulator output type)
    input_type = *it;  // Second type is A input type
  }

  bool is_f16_type = (input_type == F16_TYPE);
  bool is_bf16_type = (input_type == BF16_TYPE);
  bool is_tf32_type = (input_type == TF32_TYPE);
  bool is_s8_type = (input_type == S8_TYPE || input_type == U8_TYPE);

  // Determine matrix dimensions based on shape
  int M, N, K;
  switch (shape) {
    case MMA_M16N8K4:
      M = 16; N = 8; K = 4;
      break;
    case MMA_M16N8K8:
      M = 16; N = 8; K = 8;
      break;
    case MMA_M16N8K16:
      M = 16; N = 8; K = 16;
      break;
    case MMA_M16N8K32:
      M = 16; N = 8; K = 32;
      break;
    default:
      fprintf(stderr, "GPGPU-Sim: ERROR - tensor_mma_impl unsupported shape=%d\n", (int)shape);
      exit(1);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    const char *type_str = is_f16_type ? "F16" :
                           is_bf16_type ? "BF16" :
                           is_tf32_type ? "TF32" :
                           is_s8_type ? "S8" : "UNKNOWN";
    printf("GPGPU-Sim: tensor_mma_impl called for shape M%dN%dK%d, type=%s\n",
           M, N, K, type_str);
  }

  // Validate shape/type combination (from PTX ISA)
  // Shape/Type compatibility:
  // - FP16: M16N8K8, M16N8K16, M16N8K32 (K=8, 16, 32)
  // - BF16: M16N8K8, M16N8K16 (K=8, 16) - use mma_bf16_to_f32/mma_f32_to_bf16
  // - TF32: M16N8K4, M16N8K8 (K=4, 8 ONLY) - use mma_tf32_round
  // - S8/U8: M16N8K16, M16N8K32, M8N8K16 (K=16, 32) - use int64 intermediate + saturate
  // - S4/U4: M8N8K32 and larger (K=32+) - use mma_saturate_s4/u4

  // Get thread context
  unsigned tid;
  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  const operand_info &dst = pI->operand_lookup(0);

  // Dispatch to type-specific implementation based on data type
  if (is_s8_type) {
    // S8/U8 integer MMA implementation for M16N8K16/K32
    tensor_mma_s8_impl(pI, core, inst, M, N, K, saturate, tid, dst);
    return;
  } else if (is_tf32_type) {
    // TF32 floating-point MMA implementation for M16N8K8
    tensor_mma_tf32_impl(pI, core, inst, M, N, K, saturate, tid, dst);
    return;
  } else if (is_f16_type || is_bf16_type) {
    // F16/BF16 floating-point MMA implementation for M16N8K8
    tensor_mma_f16_impl(pI, core, inst, M, N, K, is_bf16_type, tid, dst);
    return;
  } else {
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      printf("GPGPU-Sim: tensor_mma_impl - unsupported data type for shape\n");
    }
    return;
  }
}

// F16/BF16 floating-point MMA implementation
void tensor_mma_f16_impl(const ptx_instruction *pI, core_t *core,
                         warp_inst_t inst, int M, int N, int K,
                         bool is_bf16, unsigned tid,
                         const operand_info &dst) {
  // F16/BF16 implementation: 2 registers for A (packed 16-bit), 1 for B
  // M16N8K8 implementation with proper NVIDIA tensor core fragment distribution
  // Based on PTX ISA and CUTLASS reference implementation
  //
  // Fragment distribution pattern for 32 threads:
  //   groupID = lane_id / 4  (0-7)
  //   threadID_in_group = lane_id % 4  (0-3)
  //
  // Each thread holds:
  //   A: 4 F16/BF16 values (2 U32 registers) - elements from 16×8 matrix A
  //   B: 2 F16/BF16 values (1 U32 register) - elements from 8×8 matrix B
  //   D: 4 F32 values (4 F32 registers) - elements from 16×8 output matrix D
  //
  // Output matrix D[16×8] mapping:
  //   Thread (groupID, threadID_in_group) produces 4 elements:
  //     d[0] at row=groupID,     col=threadID_in_group*2
  //     d[1] at row=groupID,     col=threadID_in_group*2+1
  //     d[2] at row=groupID+8,   col=threadID_in_group*2
  //     d[3] at row=groupID+8,   col=threadID_in_group*2+1

  // Step 1: Collect all fragments from all 32 threads
  float A_mat[M * K];  // 16×8
  float B_mat[K * N];  // 8×8
  float C_mat[M * N];  // 16×8
  float D_mat[M * N];  // 16×8

  // Initialize matrices
  for (int i = 0; i < M * K; i++) A_mat[i] = 0.0f;
  for (int i = 0; i < K * N; i++) B_mat[i] = 0.0f;
  for (int i = 0; i < M * N; i++) C_mat[i] = 0.0f;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;  // 0-7
    unsigned threadID_in_group = lane_id % 4;  // 0-3

    ptx_reg_t a_regs[2], b_regs[1], c_regs[4];
    thread->get_vector_operand_values(src_a, a_regs, 2);
    thread->get_vector_operand_values(src_b, b_regs, 1);
    thread->get_vector_operand_values(src_c, c_regs, 4);

    // Unpack and place A fragments into full matrix
    // Official formula: row = groupID for a0,a1; groupID+8 for a2,a3
    //                   col = threadID_in_group * 2 + (i & 0x1) where i = {0,1,2,3}
    uint16_t a_u16[4];
    a_u16[0] = a_regs[0].u16;
    a_u16[1] = (a_regs[0].u32 >> 16) & 0xFFFF;
    a_u16[2] = a_regs[1].u16;
    a_u16[3] = (a_regs[1].u32 >> 16) & 0xFFFF;

    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col0 = threadID_in_group * 2;
    int a_col1 = threadID_in_group * 2 + 1;
    // A is row-major: A[row][col] = A[row * K + col]
    // Convert from F16 or BF16 to F32 based on is_bf16 parameter
    if (is_bf16) {
      A_mat[a_row0 * K + a_col0] = mma_bf16_to_f32(a_u16[0]);  // a0
      A_mat[a_row0 * K + a_col1] = mma_bf16_to_f32(a_u16[1]);  // a1
      A_mat[a_row1 * K + a_col0] = mma_bf16_to_f32(a_u16[2]);  // a2
      A_mat[a_row1 * K + a_col1] = mma_bf16_to_f32(a_u16[3]);  // a3
    } else {  // F16
      A_mat[a_row0 * K + a_col0] = mma_f16_to_f32(a_u16[0]);  // a0
      A_mat[a_row0 * K + a_col1] = mma_f16_to_f32(a_u16[1]);  // a1
      A_mat[a_row1 * K + a_col0] = mma_f16_to_f32(a_u16[2]);  // a2
      A_mat[a_row1 * K + a_col1] = mma_f16_to_f32(a_u16[3]);  // a3
    }

    // Unpack and place B fragments into full matrix (B is column-major: B[n][k])
    // Official formula: row = (threadID_in_group * 2) + i, col = groupID, i = {0, 1}
    // B fragments hold: b_vals[0] = B[col][row0], b_vals[1] = B[col][row1]
    uint16_t b_u16[2];
    b_u16[0] = b_regs[0].u16;
    b_u16[1] = (b_regs[0].u32 >> 16) & 0xFFFF;

    int b_row0 = threadID_in_group * 2;
    int b_row1 = threadID_in_group * 2 + 1;
    int b_col = groupID;
    // B is column-major: B[n][k] stored as B[n*K + k]
    // Convert from F16 or BF16 to F32 based on is_bf16 parameter
    if (is_bf16) {
      B_mat[b_col * K + b_row0] = mma_bf16_to_f32(b_u16[0]);
      B_mat[b_col * K + b_row1] = mma_bf16_to_f32(b_u16[1]);
    } else {  // F16
      B_mat[b_col * K + b_row0] = mma_f16_to_f32(b_u16[0]);
      B_mat[b_col * K + b_row1] = mma_f16_to_f32(b_u16[1]);
    }

    // Place C fragments
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;
    C_mat[c_row0 * N + c_col0] = c_regs[0].f32;
    C_mat[c_row0 * N + c_col1] = c_regs[1].f32;
    C_mat[c_row1 * N + c_col0] = c_regs[2].f32;
    C_mat[c_row1 * N + c_col1] = c_regs[3].f32;
  }

  // Step 2: Perform matrix multiplication D = A * B + C
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        // A is row-major: A[m][k], B is column-major: B[n][k]
        sum += A_mat[m * K + k] * B_mat[n * K + k];
      }
      D_mat[m * N + n] = sum + C_mat[m * N + n];
    }
  }

  // Step 3: Distribute results back to threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;
    unsigned threadID_in_group = lane_id % 4;

    ptx_reg_t d_regs[4];
    int d_row0 = groupID;
    int d_row1 = groupID + 8;
    int d_col0 = threadID_in_group * 2;
    int d_col1 = threadID_in_group * 2 + 1;

    d_regs[0].f32 = D_mat[d_row0 * N + d_col0];
    d_regs[1].f32 = D_mat[d_row0 * N + d_col1];
    d_regs[2].f32 = D_mat[d_row1 * N + d_col0];
    d_regs[3].f32 = D_mat[d_row1 * N + d_col1];

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2], d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_impl completed for M16N8K8\n");
  }
}

// TF32 floating-point MMA implementation
void tensor_mma_tf32_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t inst, int M, int N, int K,
                          bool saturate, unsigned tid,
                          const operand_info &dst) {
  // TF32 implementation for M16N8K4: 2 F32 registers for A, 1 F32 for B
  // Each TF32 value is stored as F32 but with reduced precision (10-bit mantissa)

  // Step 1: Collect all fragments from all 32 threads
  float A_mat[M * K];  // 16×4
  float B_mat[K * N];  // 4×8
  float C_mat[M * N];  // 16×8
  float D_mat[M * N];  // 16×8

  // Initialize matrices
  for (int i = 0; i < M * K; i++) A_mat[i] = 0.0f;
  for (int i = 0; i < K * N; i++) B_mat[i] = 0.0f;
  for (int i = 0; i < M * N; i++) C_mat[i] = 0.0f;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;  // 0-7
    unsigned threadID_in_group = lane_id % 4;  // 0-3

    // TF32 M16N8K4 uses 2 F32 registers for A, 1 F32 for B
    ptx_reg_t a_regs[2], b_regs[1], c_regs[4];
    thread->get_vector_operand_values(src_a, a_regs, 2);
    thread->get_vector_operand_values(src_b, b_regs, 1);
    thread->get_vector_operand_values(src_c, c_regs, 4);

    // Unpack and place A fragments with TF32 rounding (16×4 matrix)
    // Each thread holds 2 TF32 values in one column
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col = threadID_in_group;

    A_mat[a_row0 * K + a_col] = mma_tf32_round(a_regs[0].f32);
    A_mat[a_row1 * K + a_col] = mma_tf32_round(a_regs[1].f32);

    // Unpack and place B fragments with TF32 rounding (4×8 matrix)
    // Each thread holds 1 TF32 value
    int b_row = threadID_in_group;
    int b_col = groupID;
    B_mat[b_col * K + b_row] = mma_tf32_round(b_regs[0].f32);

    // Place C fragments (no rounding needed for accumulator)
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;
    C_mat[c_row0 * N + c_col0] = c_regs[0].f32;
    C_mat[c_row0 * N + c_col1] = c_regs[1].f32;
    C_mat[c_row1 * N + c_col0] = c_regs[2].f32;
    C_mat[c_row1 * N + c_col1] = c_regs[3].f32;
  }

  // Step 2: Perform matrix multiplication D = A * B + C
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A_mat[m * K + k] * B_mat[n * K + k];
      }
      D_mat[m * N + n] = sum + C_mat[m * N + n];
    }
  }

  // Step 3: Distribute results back to threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;
    unsigned threadID_in_group = lane_id % 4;

    ptx_reg_t d_regs[4];
    int d_row0 = groupID;
    int d_row1 = groupID + 8;
    int d_col0 = threadID_in_group * 2;
    int d_col1 = threadID_in_group * 2 + 1;

    d_regs[0].f32 = D_mat[d_row0 * N + d_col0];
    d_regs[1].f32 = D_mat[d_row0 * N + d_col1];
    d_regs[2].f32 = D_mat[d_row1 * N + d_col0];
    d_regs[3].f32 = D_mat[d_row1 * N + d_col1];

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2], d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_tf32_impl completed for M16N8K4\n");
  }
}

void tensor_mma_ld_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst) {
  // Load matrix fragments from memory for MMA instruction
  // Loads matrix elements into registers for subsequent MMA compute

  // Get MMA properties
  mma_shape_type shape = pI->get_mma_shape();
  mma_layout_mode layout_a, layout_b;
  pI->get_mma_layout(layout_a, layout_b);

  // Get operands: destination register and memory address
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_addr = pI->operand_lookup(1);

  // Get thread context
  unsigned tid;
  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  // Load matrix fragments for each thread in warp
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    // Get base memory address for this thread
    ptx_reg_t base_addr = thread->get_operand_value(src_addr, src_addr, U32_TYPE, thread, 1);

    // Calculate element offsets based on shape and layout
    // For simplified implementation, load sequential elements
    unsigned nelem = dst.get_vect_nelem();
    ptx_reg_t data[8];

    for (unsigned i = 0; i < nelem && i < 8; i++) {
      // Calculate memory offset for this element
      unsigned offset = mma_thread_to_element_offset(
          thrd, shape, layout_a, sizeof(uint16_t), 0);
      offset += i * sizeof(uint16_t);

      // In full implementation, would load from memory at base_addr + offset
      // For now, initialize to zero (memory load would happen here)
      data[i].u64 = 0;
    }

    // Store loaded data to destination register
    ptx_reg_t zero;
    zero.u64 = 0;
    if (nelem == 2) {
      thread->set_vector_operand_values(dst, data[0], data[1], zero, zero);
    } else if (nelem >= 4) {
      thread->set_vector_operand_values(dst, data[0], data[1], data[2], data[3]);
    }
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_ld_impl completed for shape\n");
  }
}

void tensor_mma_st_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst) {
  // Store matrix fragments to memory from MMA computation results
  // Takes computed matrix elements from registers and writes to memory

  // Get MMA properties
  mma_shape_type shape = pI->get_mma_shape();
  mma_layout_mode layout_a, layout_b;
  pI->get_mma_layout(layout_a, layout_b);

  // Get operands: memory address and source register
  const operand_info &dst_addr = pI->operand_lookup(0);
  const operand_info &src = pI->operand_lookup(1);

  // Get thread context
  unsigned tid;
  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  // Store matrix fragments from each thread in warp
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    // Get base memory address for this thread
    ptx_reg_t base_addr = thread->get_operand_value(dst_addr, dst_addr, U32_TYPE, thread, 1);

    // Get data to store from source register
    unsigned nelem = src.get_vect_nelem();
    ptx_reg_t data[8];
    thread->get_vector_operand_values(src, data, nelem);

    // Store each element to memory
    for (unsigned i = 0; i < nelem && i < 8; i++) {
      // Calculate memory offset for this element
      unsigned offset = mma_thread_to_element_offset(
          thrd, shape, layout_a, sizeof(uint16_t), 0);
      offset += i * sizeof(uint16_t);

      // In full implementation, would store to memory at base_addr + offset
      // For now, data is ready but actual memory write would happen here
      // (requires integration with memory system)
    }
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_st_impl completed for shape\n");
  }
}

} // namespace flash_gpgpu_sim
