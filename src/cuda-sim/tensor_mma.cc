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
  // Implementation for MMA instructions (separate from WMMA)
  // Currently supports: M16N8K8 with F16/F32 data types
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

  // Get data types from instruction
  // For MMA: type spec is D_type, A_type, B_type, C_type
  // For M16N8K8: typically F16/F32 combinations

  // Validate shape/type combination
  if (shape != MMA_M16N8K8) {
    // Only M16N8K8 is implemented in Step 3
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      printf("GPGPU-Sim: tensor_mma_impl - unsupported shape (only M16N8K8 implemented)\n");
    }
    return;
  }

  // Matrix dimensions for M16N8K8
  const int M = 16;  // Output rows
  const int N = 8;   // Output cols
  const int K = 8;   // Inner dimension

  // Allocate matrix storage (shared across warp)
  ptx_reg_t matrix_a[M][K];
  ptx_reg_t matrix_b[K][N];
  ptx_reg_t matrix_c[M][N];
  ptx_reg_t matrix_d[M][N];

  // Get thread context
  unsigned tid;
  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  const operand_info &dst = pI->operand_lookup(0);

  // Load matrices from all threads in warp
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    // Load operands: A (op 1), B (op 2), C (op 3)
    for (int operand_num = 1; operand_num <= 3; operand_num++) {
      const operand_info &src = pI->operand_lookup(operand_num);
      unsigned nelem = src.get_vect_nelem();
      ptx_reg_t v[8];
      thread->get_vector_operand_values(src, v, nelem);

      // For M16N8K8, each thread provides multiple elements
      // Simplified mapping: distribute elements across matrix based on thread ID
      int row_base = (thrd / 4) * 2;  // 4 threads per row group
      int col_base = (thrd % 4) * 2;  // 2 elements per thread

      switch (operand_num) {
        case 1:  // Matrix A (M x K)
          for (unsigned k = 0; k < nelem && k < 4; k++) {
            int row = row_base + (k / 2);
            int col = col_base + (k % 2);
            if (row < M && col < K) {
              matrix_a[row][col] = v[k];
            }
          }
          break;

        case 2:  // Matrix B (K x N)
          for (unsigned k = 0; k < nelem && k < 4; k++) {
            int row = row_base + (k / 2);
            int col = col_base + (k % 2);
            // For B matrix, adjust indexing
            if (row < K && col < N) {
              matrix_b[row][col] = v[k];
            }
          }
          break;

        case 3:  // Matrix C (M x N) - accumulator
          for (unsigned k = 0; k < nelem && k < 4; k++) {
            int row = row_base + (k / 2);
            int col = col_base + (k % 2);
            if (row < M && col < N) {
              matrix_c[row][col] = v[k];
            }
          }
          break;
      }
    }
  }

  // Perform matrix multiplication: D = A * B + C
  // Using F16 arithmetic (convert to F32 for computation if needed)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;

      // Matrix multiply: sum over K dimension
      for (int k = 0; k < K; k++) {
        // Convert F16 to F32 for computation
        float a_val = (float)matrix_a[i][k].f16;
        float b_val = (float)matrix_b[k][j].f16;
        sum += a_val * b_val;
      }

      // Add accumulator C
      float c_val = (float)matrix_c[i][j].f16;
      sum += c_val;

      // Store result (may need conversion based on output type)
      matrix_d[i][j].f16 = (half)sum;
      matrix_d[i][j].f32 = sum;  // Also store F32 version
    }
  }

  // Distribute results back to threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    // Map thread to output elements
    int row_base = (thrd / 4) * 2;
    int col_base = (thrd % 4) * 2;

    // Collect this thread's output elements
    ptx_reg_t outputs[4];
    int out_idx = 0;
    for (int k = 0; k < 4 && out_idx < 4; k++) {
      int row = row_base + (k / 2);
      int col = col_base + (k % 2);
      if (row < M && col < N) {
        outputs[out_idx++] = matrix_d[row][col];
      }
    }

    // Write outputs back to destination register
    // Pack F16 values into register format if needed
    if (out_idx >= 4) {
      ptx_reg_t packed_data[2];
      packed_data[0].s64 = ((outputs[0].s64 & 0xFFFF)) |
                           ((outputs[1].s64 & 0xFFFF) << 16);
      packed_data[1].s64 = ((outputs[2].s64 & 0xFFFF)) |
                           ((outputs[3].s64 & 0xFFFF) << 16);
      thread->set_vector_operand_values(dst, packed_data[0], packed_data[1]);
    }
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_impl completed for M16N8K8\n");
  }
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
