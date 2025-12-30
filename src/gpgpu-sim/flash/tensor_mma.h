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

#ifndef FLASH_GPGPU_SIM_TENSOR_MMA_H
#define FLASH_GPGPU_SIM_TENSOR_MMA_H

#include <cstdint>

// Forward declarations
class ptx_instruction;
class core_t;
struct warp_inst_t;

namespace flash_gpgpu_sim {

// MMA shape types (separate from WMMA)
// Corresponds to the MxNxK dimensions in PTX MMA instructions
// These are the REAL shapes supported by NVIDIA hardware (SM75/SM80)
enum mma_shape_type {
  // SM75 (Turing) and SM80 (Ampere)
  MMA_M16N8K8,   // F16, BF16, TF32

  // SM80 (Ampere) only
  MMA_M16N8K4,   // TF32
  MMA_M16N8K16,  // F16, BF16, S8/U8
  MMA_M16N8K32,  // S8/U8
  MMA_M16N8K64,  // S4/U4
  MMA_M8N8K4     // F64
};

// MMA layout modes for matrix operands
// Specifies whether operands are in row-major or column-major layout
enum mma_layout_mode {
  MMA_ROW_COL,  // A: row-major, B: column-major
  MMA_ROW_ROW,  // A: row-major, B: row-major
  MMA_COL_ROW,  // A: column-major, B: row-major
  MMA_COL_COL   // A: column-major, B: column-major
};

// Main MMA instruction implementations
// These functions implement the PTX MMA (Matrix Multiply-Accumulate) instructions
// Separate from existing WMMA implementations to maintain clear separation

// tensor_mma_impl: Main compute function for MMA instructions
// Implements D = A * B + C matrix multiplication
void tensor_mma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst);

// tensor_mma_ld_impl: Load matrix fragments from memory
// Loads matrix fragments into registers for MMA computation
void tensor_mma_ld_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst);

// tensor_mma_st_impl: Store matrix fragments to memory
// Stores computed matrix fragments from registers to memory
void tensor_mma_st_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t &inst);

// Helper functions (exposed for testing and internal use)
// Type conversion functions

// Convert IEEE 754 float16 to float32
float mma_f16_to_f32(uint16_t f16);

// Convert float32 to IEEE 754 float16 with rounding
uint16_t mma_f32_to_f16(float f32);

// Convert bfloat16 to float32
// bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
float mma_bf16_to_f32(uint16_t bf16);

// Convert float32 to bfloat16 with truncation
uint16_t mma_f32_to_bf16(float f32);

// Round float32 to TensorFloat-32 precision
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (19 bits total)
float mma_tf32_round(float f32);

// Saturation functions for integer types

// Saturate int32 value to signed 8-bit range [-128, 127]
int8_t mma_saturate_s8(int32_t val);

// Saturate int32 value to unsigned 8-bit range [0, 255]
uint8_t mma_saturate_u8(int32_t val);

// Saturate int32 value to signed 4-bit range [-8, 7]
int8_t mma_saturate_s4(int32_t val);

// Saturate int32 value to unsigned 4-bit range [0, 15]
uint8_t mma_saturate_u4(int32_t val);

// Thread-to-element mapping function
// Maps warp thread ID to matrix element offset based on MMA shape and layout
// Returns byte offset into matrix fragment for given thread
// type_size: Element size in bytes (2 for F16/BF16, 4 for F32, 1 for S8/U8)
// stride: Row/column stride in elements (NOT bytes)
unsigned mma_thread_to_element_offset(unsigned thread_id, mma_shape_type shape,
                                      mma_layout_mode layout,
                                      unsigned char type_size, unsigned stride);

} // namespace flash_gpgpu_sim

#endif  // FLASH_GPGPU_SIM_TENSOR_MMA_H
