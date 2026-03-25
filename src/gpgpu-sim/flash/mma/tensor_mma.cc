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

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
// Forward declarations needed by ptx.tab.h
typedef void *yyscan_t;
class ptx_recognizer;
#include "../../../../libcuda/gpgpu_context.h"
#include "../../gpu-sim.h"
#include "ptx.tab.h"

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
constexpr int MANTISSA_SHIFT = 13; // F16 to F32 mantissa shift

constexpr int EXPONENT_BIAS = 15;
constexpr int MAX_EXPONENT = 31;
constexpr uint16_t INFINITY_BITS = 0x7C00;

// Denormal number scaling: mantissa / 1024.0 / 16384.0
constexpr float DENORMAL_SCALE_1 = 1024.0f;  // 2^10 (mantissa bits)
constexpr float DENORMAL_SCALE_2 = 16384.0f; // 2^14 (exponent adjustment)
} // namespace F16Constants

// F32 (IEEE 754 single-precision) format constants
namespace F32Constants {
constexpr int SIGN_SHIFT = 31;
constexpr int EXPONENT_SHIFT = 23;
constexpr uint32_t EXPONENT_MASK = 0xFF;
constexpr uint32_t MANTISSA_MASK_F16 = 0x3FF; // When converting from F16

constexpr int EXPONENT_BIAS = 127;
constexpr int MANTISSA_SHIFT_FROM_F16 = 13; // F32 mantissa shift from F16
} // namespace F32Constants

// BF16 (bfloat16) format constants
// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
namespace BF16Constants {
constexpr int SHIFT = 16; // BF16 occupies upper 16 bits of F32
}

// TF32 (TensorFloat-32) format constants
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (uses F32 container)
namespace TF32Constants {
constexpr uint32_t MANTISSA_MASK = 0x1FFF; // Lower 13 bits to zero out
constexpr uint32_t ROUND_BIT = 0x1000;     // Bit 12 (rounding decision bit)
// TF32 has 10 mantissa bits, F32 has 23, so zero out lower 13 bits (23-10=13)
} // namespace TF32Constants

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
} // namespace SaturationLimits

// ============================================================================
// Type Conversion Helper Functions
// ============================================================================

// Helper: Convert F16 to F32
float mma_f16_to_f32(uint16_t f16) {
  using namespace F16Constants;
  using namespace F32Constants;

  uint32_t sign = (f16 >> F16Constants::SIGN_SHIFT) & F16Constants::SIGN_MASK;
  uint32_t exp =
      (f16 >> F16Constants::EXPONENT_SHIFT) & F16Constants::EXPONENT_MASK;
  uint32_t frac = f16 & F16Constants::MANTISSA_MASK;

  if (exp == 0) {
    if (frac == 0)
      return sign ? -0.0f : 0.0f;
    // Denormal: mantissa / 2^10 / 2^14
    float result = frac / DENORMAL_SCALE_1 / DENORMAL_SCALE_2;
    return sign ? -result : result;
  } else if (exp == MAX_EXPONENT) {
    // Inf or NaN
    return frac ? NAN : (sign ? -INFINITY : INFINITY);
  }

  // Normalized: convert exponent from F16 bias to F32 bias
  uint32_t f32_exp =
      exp - F16Constants::EXPONENT_BIAS + F32Constants::EXPONENT_BIAS;
  uint32_t f32_frac = frac << MANTISSA_SHIFT;
  uint32_t f32_bits = (sign << F32Constants::SIGN_SHIFT) |
                      (f32_exp << F32Constants::EXPONENT_SHIFT) | f32_frac;

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

  uint32_t sign =
      (f32_bits >> F32Constants::SIGN_SHIFT) & F16Constants::SIGN_MASK;
  int32_t exp = ((f32_bits >> F32Constants::EXPONENT_SHIFT) &
                 F32Constants::EXPONENT_MASK) -
                F32Constants::EXPONENT_BIAS + F16Constants::EXPONENT_BIAS;
  uint32_t frac = (f32_bits >> F32Constants::MANTISSA_SHIFT_FROM_F16) &
                  F32Constants::MANTISSA_MASK_F16;

  if (exp <= 0)
    return sign << F16Constants::SIGN_SHIFT; // Flush to zero
  if (exp >= F16Constants::MAX_EXPONENT)
    return (sign << F16Constants::SIGN_SHIFT) |
           F16Constants::INFINITY_BITS; // Inf

  return (sign << F16Constants::SIGN_SHIFT) |
         (exp << F16Constants::EXPONENT_SHIFT) | frac;
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
  if (val > S8_MAX)
    return S8_MAX;
  if (val < S8_MIN)
    return S8_MIN;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U8 value
uint8_t mma_saturate_u8(int32_t val) {
  using namespace SaturationLimits;
  if (val > U8_MAX)
    return U8_MAX;
  if (val < U8_MIN)
    return U8_MIN;
  return static_cast<uint8_t>(val);
}

// Helper: Saturate S4 value
int8_t mma_saturate_s4(int32_t val) {
  using namespace SaturationLimits;
  if (val > S4_MAX)
    return S4_MAX;
  if (val < S4_MIN)
    return S4_MIN;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U4 value
uint8_t mma_saturate_u4(int32_t val) {
  using namespace SaturationLimits;
  if (val > U4_MAX)
    return U4_MAX;
  if (val < U4_MIN)
    return U4_MIN;
  return static_cast<uint8_t>(val);
}

// Helper: Thread-to-element offset for MMA shapes
// Maps thread ID to matrix element offset based on shape and layout
unsigned mma_thread_to_element_offset(unsigned thread_id, mma_shape_type shape,
                                      mma_layout_mode layout,
                                      unsigned char type_size,
                                      unsigned stride) {
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
  // Get the input data type (first type in scalar type list, which is the
  // accumulator for MMA) For MMA, the operand types are: D (output), A (input),
  // B (input), C (accumulator) Scalar types list order: output type, then input
  // types
  std::list<int> scalar_types = pI->get_scalar_type();
  // For MMA with .f32.bf16.bf16.f32, get_type() returns first type (f32
  // output), get_type2() would return second if it exists We need to examine
  // the full scalar type list to find input types Typically: scalar_types =
  // {F32_TYPE, BF16_TYPE, BF16_TYPE, F32_TYPE} for bf16 MMA

  int input_type = F16_TYPE; // Default to F16
  if (scalar_types.size() >= 2) {
    auto it = scalar_types.begin();
    ++it;             // Skip first type (accumulator output type)
    input_type = *it; // Second type is A input type
  }

  bool is_f16_type = (input_type == F16_TYPE);
  bool is_bf16_type = (input_type == BF16_TYPE);
  bool is_tf32_type = (input_type == TF32_TYPE);
  bool is_s8_type = (input_type == S8_TYPE || input_type == U8_TYPE);

  // Determine matrix dimensions based on shape
  int M, N, K;
  switch (shape) {
  case MMA_M16N8K4:
    M = 16;
    N = 8;
    K = 4;
    break;
  case MMA_M16N8K8:
    M = 16;
    N = 8;
    K = 8;
    break;
  case MMA_M16N8K16:
    M = 16;
    N = 8;
    K = 16;
    break;
  case MMA_M16N8K32:
    M = 16;
    N = 8;
    K = 32;
    break;
  case MMA_M8N8K4:
    M = 8;
    N = 8;
    K = 4;
    break;
  case MMA_M8N8K16:
    M = 8;
    N = 8;
    K = 16;
    break;
  default:
    fprintf(stderr, "GPGPU-Sim: ERROR - tensor_mma_impl unsupported shape=%d\n",
            (int)shape);
    exit(1);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    const char *type_str = is_f16_type    ? "F16"
                           : is_bf16_type ? "BF16"
                           : is_tf32_type ? "TF32"
                           : is_s8_type   ? "S8"
                                          : "UNKNOWN";
    printf("GPGPU-Sim: tensor_mma_impl called for shape M%dN%dK%d, type=%s\n",
           M, N, K, type_str);
  }

  // Validate shape/type combination (from PTX ISA)
  // Shape/Type compatibility:
  // - FP16: M16N8K8, M16N8K16, M16N8K32 (K=8, 16, 32)
  // - BF16: M16N8K8, M16N8K16 (K=8, 16) - use mma_bf16_to_f32/mma_f32_to_bf16
  // - TF32: M16N8K4, M16N8K8 (K=4, 8 ONLY) - use mma_tf32_round
  // - S8/U8: M16N8K16, M16N8K32, M8N8K16 (K=16, 32) - use int64 intermediate +
  // saturate
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
    // S8/U8 M8N8K16 requires specialized implementation due to smaller M
    // dimension
    if (shape == MMA_M8N8K16) {
      tensor_mma_s8_m8n8k16_impl(pI, core, inst, saturate, tid, dst);
      return;
    }
    // S8/U8 integer MMA implementation for M16N8K16/K32
    tensor_mma_s8_impl(pI, core, inst, M, N, K, saturate, tid, dst);
    return;
  } else if (is_tf32_type) {
    // TF32 floating-point MMA implementation for M16N8K8
    tensor_mma_tf32_impl(pI, core, inst, M, N, K, saturate, tid, dst);
    return;
  } else if (is_f16_type || is_bf16_type) {
    // F16/BF16 M8N8K4 requires specialized 4-computation implementation
    if (shape == MMA_M8N8K4) {
      tensor_mma_f16_m8n8k4_impl(pI, core, inst, is_bf16_type, tid, dst);
      return;
    }
    // F16/BF16 floating-point MMA implementation for other shapes
    tensor_mma_f16_impl(pI, core, inst, M, N, K, is_bf16_type, tid, dst);
    return;
  } else {
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      printf("GPGPU-Sim: tensor_mma_impl - unsupported data type for shape\n");
    }
    return;
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

  // Get operands: destination register
  const operand_info &dst = pI->operand_lookup(0);

  // Get thread context
  unsigned tid;
  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  // Load matrix fragments for each thread in warp
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    // Calculate element offsets based on shape and layout
    // For simplified implementation, load sequential elements
    unsigned nelem = dst.get_vect_nelem();
    ptx_reg_t data[8];

    for (unsigned i = 0; i < nelem && i < 8; i++) {
      // Calculate memory offset for this element
      unsigned offset = mma_thread_to_element_offset(thrd, shape, layout_a,
                                                     sizeof(uint16_t), 0);
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
      thread->set_vector_operand_values(dst, data[0], data[1], data[2],
                                        data[3]);
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

  // Get operands: source register
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

    // Get data to store from source register
    unsigned nelem = src.get_vect_nelem();
    ptx_reg_t data[8];
    thread->get_vector_operand_values(src, data, nelem);

    // Store each element to memory
    for (unsigned i = 0; i < nelem && i < 8; i++) {
      // Calculate memory offset for this element
      unsigned offset = mma_thread_to_element_offset(thrd, shape, layout_a,
                                                     sizeof(uint16_t), 0);
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
