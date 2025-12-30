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

namespace flash_gpgpu_sim {

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

    case MMA_M16N16K8:
      // M16N16K8: 16x16 output matrix, K=8
      // Wider output matrix (N=16 instead of N=8)
      row = thread_id / 8;  // 8 threads per row group
      col = (thread_id % 8) * 2;
      break;

    case MMA_M16N16K16_MMA:
      // M16N16K16: 16x16 output matrix, K=16
      // Standard WMMA size but with MMA instruction
      row = thread_id / 8;
      col = (thread_id % 8) * 2;
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

  // Determine matrix dimensions based on shape
  int M, N, K;
  switch (shape) {
    case MMA_M16N8K8:
      M = 16; N = 8; K = 8;
      break;
    case MMA_M16N8K16:
      M = 16; N = 8; K = 16;
      break;
    case MMA_M16N8K32:
      M = 16; N = 8; K = 32;
      break;
    case MMA_M16N16K8:
      M = 16; N = 16; K = 8;
      break;
    case MMA_M16N16K16_MMA:
      M = 16; N = 16; K = 16;
      break;
    default:
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("GPGPU-Sim: tensor_mma_impl - unsupported shape\n");
      }
      return;
  }

  // Validate shape/type combination (from PTX ISA)
  // Shape/Type compatibility:
  // - FP16: M16N8K8, M16N8K16, M16N8K32 (K=8, 16, 32)
  // - BF16: M16N8K8, M16N8K16 (K=8, 16) - use mma_bf16_to_f32/mma_f32_to_bf16
  // - TF32: M16N8K4, M16N8K8 (K=4, 8 ONLY) - use mma_tf32_round
  // - S8/U8: M16N8K16, M16N8K32, M8N8K16 (K=16, 32) - use mma_saturate_s8/u8
  // - S4/U4: M8N8K32 and larger (K=32+) - use mma_saturate_s4/u4
  //
  // Current implementation: F16/F32 primarily
  // TODO: Add full type-specific handling for BF16, TF32, INT types

  // Allocate matrix storage (use maximum dimensions to avoid VLA issues)
  const int MAX_M = 16, MAX_N = 16, MAX_K = 32;
  ptx_reg_t matrix_a[MAX_M][MAX_K];
  ptx_reg_t matrix_b[MAX_K][MAX_N];
  ptx_reg_t matrix_c[MAX_M][MAX_N];
  ptx_reg_t matrix_d[MAX_M][MAX_N];

  // Initialize matrices to zero
  memset(matrix_a, 0, sizeof(matrix_a));
  memset(matrix_b, 0, sizeof(matrix_b));
  memset(matrix_c, 0, sizeof(matrix_c));
  memset(matrix_d, 0, sizeof(matrix_d));

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
    ptx_reg_t base_addr;
    thread->get_operand_value(src_addr, base_addr, B32_TYPE, thread, 1);

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
    if (nelem == 2) {
      thread->set_vector_operand_values(dst, data[0], data[1]);
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
    ptx_reg_t base_addr;
    thread->get_operand_value(dst_addr, base_addr, B32_TYPE, thread, 1);

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
