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
#include "ptx_ir.h"

// Helper: Convert F16 to F32
float mma_f16_to_f32(uint16_t f16) {
  uint32_t sign = (f16 >> 15) & 0x1;
  uint32_t exp = (f16 >> 10) & 0x1F;
  uint32_t frac = f16 & 0x3FF;

  if (exp == 0) {
    if (frac == 0) return sign ? -0.0f : 0.0f;
    // Denormal
    float result = frac / 1024.0f / 16384.0f;
    return sign ? -result : result;
  } else if (exp == 31) {
    // Inf or NaN
    return frac ? NAN : (sign ? -INFINITY : INFINITY);
  }

  // Normalized
  uint32_t f32_exp = exp - 15 + 127;
  uint32_t f32_frac = frac << 13;
  uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;

  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Convert F32 to F16
uint16_t mma_f32_to_f16(float f32) {
  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));

  uint32_t sign = (f32_bits >> 31) & 0x1;
  int32_t exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
  uint32_t frac = (f32_bits >> 13) & 0x3FF;

  if (exp <= 0) return sign << 15;          // Flush to zero
  if (exp >= 31) return (sign << 15) | 0x7C00;  // Inf

  return (sign << 15) | (exp << 10) | frac;
}

// Helper: Convert BF16 to F32
float mma_bf16_to_f32(uint16_t bf16) {
  uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Convert F32 to BF16
uint16_t mma_f32_to_bf16(float f32) {
  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));
  return static_cast<uint16_t>(f32_bits >> 16);
}

// Helper: Round float32 to TensorFloat-32 precision
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (19 bits total)
float mma_tf32_round(float f32) {
  uint32_t f32_bits;
  memcpy(&f32_bits, &f32, sizeof(float));

  // TF32 has 10 mantissa bits, so we need to round to nearest even
  // and zero out the lower 13 bits (23 - 10 = 13)
  uint32_t mantissa_mask = 0x1FFF;  // Lower 13 bits
  uint32_t round_bit = 0x1000;      // Bit 12 (rounding bit)

  // Round to nearest even
  if ((f32_bits & round_bit) && ((f32_bits & mantissa_mask) != round_bit)) {
    f32_bits += round_bit;
  }

  // Zero out lower 13 bits
  f32_bits &= ~mantissa_mask;

  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

// Helper: Saturate S8 value
int8_t mma_saturate_s8(int32_t val) {
  if (val > 127) return 127;
  if (val < -128) return -128;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U8 value
uint8_t mma_saturate_u8(int32_t val) {
  if (val > 255) return 255;
  if (val < 0) return 0;
  return static_cast<uint8_t>(val);
}

// Helper: Saturate S4 value
int8_t mma_saturate_s4(int32_t val) {
  if (val > 7) return 7;
  if (val < -8) return -8;
  return static_cast<int8_t>(val);
}

// Helper: Saturate U4 value
uint8_t mma_saturate_u4(int32_t val) {
  if (val > 15) return 15;
  if (val < 0) return 0;
  return static_cast<uint8_t>(val);
}

// Helper: Thread-to-element offset for MMA shapes
// Maps thread ID to matrix element offset based on shape and layout
unsigned mma_thread_to_element_offset(unsigned thread_id, mma_shape_type shape,
                                      mma_layout_mode layout,
                                      unsigned char type_size, unsigned stride) {
  // Simplified mapping for M16N8K8 shape
  // TODO: Add mappings for other shapes (M16N8K16, M16N8K32, M16N16K8,
  // M16N16K16)

  if (shape == MMA_M16N8K8) {
    // For M16N8K8: 16x8 matrix, 32 threads
    // Each thread handles specific elements based on layout
    unsigned row = thread_id / 4;
    unsigned col = (thread_id % 4) * 2;

    if (layout == MMA_ROW_COL || layout == MMA_ROW_ROW) {
      // Row-major A
      return row * stride + col * type_size;
    } else {
      // Column-major A
      return col * stride + row * type_size;
    }
  }

  // Default: simple linear offset
  return thread_id * type_size;
}

// NEW: Tensor MMA instruction implementations (separate from WMMA)
void tensor_mma_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t inst) {
  // Basic implementation for F16 M16N8K8 shape
  // TODO: Extend to support all shapes (M16N8K16, M16N8K32, M16N16K8,
  // M16N16K16) TODO: Extend to support all data types (TF32, BF16, S8, U8, S4,
  // U4, B1)

  // This is a simplified stub implementation
  // Real implementation would need to:
  // 1. Extract MMA shape, layout, and data types from instruction
  // 2. Load matrices A, B, C from register file
  // 3. Perform matrix multiply-accumulate: D = A × B + C
  // 4. Store result matrix D back to register file

  unsigned warp_id = inst.warp_id();

  // For now, just log that the instruction was called
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_impl called for warp %u\n", warp_id);
  }

  // Simplified implementation: just pass through without aborting
  // In a full implementation, this would compute D = A × B + C
  // using the helper functions above for type conversions
}

void tensor_mma_ld_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst) {
  // Simplified stub for tensor MMA load operation
  // TODO: Implement full load logic for all shapes and data types
  // Should load matrix elements from memory to registers based on:
  // - MMA shape (M16N8K8, M16N8K16, etc.)
  // - Layout mode (ROW-COL, ROW-ROW, etc.)
  // - Data type (F16, BF16, TF32, S8, U8, etc.)

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_ld_impl called\n");
  }

  // Simplified: just mark as memory operation without aborting
  // Real implementation would set memory addresses and data size
}

void tensor_mma_st_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst) {
  // Simplified stub for tensor MMA store operation
  // TODO: Implement full store logic for all shapes and data types
  // Should store result matrix from registers to memory

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_st_impl called\n");
  }

  // Simplified: just mark as memory operation without aborting
  // Real implementation would set memory addresses and data size
}
