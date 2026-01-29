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

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../../../libcuda/gpgpu_context.h"
#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"

namespace flash_gpgpu_sim {

// S8/U8 integer MMA implementation for M16N8K16
// Implements functional simulation of integer tensor core operations
void tensor_mma_s8_impl(const ptx_instruction *pI, core_t *core,
                        warp_inst_t inst, int M, int N, int K, bool saturate,
                        unsigned tid, const operand_info &dst) {
  // S8/U8 integer MMA implementation for M16N8K16 and M16N8K32
  // Based on PTX ISA spec for integer tensor cores
  //
  // Fragment distribution pattern for 32 threads:
  //   groupID = lane_id >> 2  (lane_id / 4, range 0-7)
  //   threadID_in_group = lane_id % 4  (range 0-3)
  //
  // M16N8K16: Each thread holds:
  //   A: 8 S8 values (2 U32 registers, packed 4 values each) - from 16×16
  //   matrix A B: 4 S8 values (1 U32 register, packed) - from 16×8 matrix B
  //   C/D: 4 S32 values (4 S32 registers) - from 16×8 output matrix D
  //
  // M16N8K32: Each thread holds:
  //   A: 16 S8 values (4 U32 registers, packed 4 values each) - from 16×32
  //   matrix A B: 8 S8 values (2 U32 registers, packed) - from 32×8 matrix B
  //   C/D: 4 S32 values (4 S32 registers) - from 16×8 output matrix D
  //
  // Matrix A fragment distribution (row-major):
  //   K=16: row = groupID for a[0..3], groupID+8 for a[4..7]
  //         col = (threadID_in_group * 4) + (i & 0x3)
  //   K=32: row = groupID for a[0..7], groupID+8 for a[8..15]
  //         col = (threadID_in_group * 4) + (i & 0x3) for i<8
  //               (threadID_in_group * 4) + (i & 0x3) + 16 for i>=8
  //
  // Matrix B fragment distribution (column-major):
  //   K=16: row = (threadID_in_group * 4) + i for b[0..3]
  //   K=32: row = (threadID_in_group * 4) + i for b[0..3]
  //               (threadID_in_group * 4) + i + 16 for b[4..7]
  //   col = groupID
  //
  // Output matrix D[16×8] mapping (same for both K values):
  //   row = groupID for d[0..1], groupID+8 for d[2..3]
  //   col = (threadID_in_group * 2) + (i & 0x1)

  // Step 1: Collect all fragments from all 32 threads into full matrices
  int32_t A_mat[M * K]; // 16×16, stored as int32 for computation
  int32_t B_mat[K * N]; // 16×8
  int32_t C_mat[M * N]; // 16×8
  int64_t D_mat[M * N]; // 16×8, use int64 to detect overflow

  // Initialize matrices
  for (int i = 0; i < M * K; i++)
    A_mat[i] = 0;
  for (int i = 0; i < K * N; i++)
    B_mat[i] = 0;
  for (int i = 0; i < M * N; i++)
    C_mat[i] = 0;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id >> 2;          // lane_id / 4, range 0-7
    unsigned threadID_in_group = lane_id % 4; // range 0-3

    // Fragment distribution differs based on K dimension
    if (K == 16) {
      // M16N8K16: A needs 2 registers, B needs 1 register
      ptx_reg_t a_regs[2], b_regs[1], c_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 2);
      thread->get_vector_operand_values(src_b, b_regs, 1);
      thread->get_vector_operand_values(src_c, c_regs, 4);

      // Unpack A fragments (8 S8 values packed in 2 U32 registers)
      int8_t a_vals[8];
      memcpy(&a_vals[0], &a_regs[0].u32, 4); // a[0..3]
      memcpy(&a_vals[4], &a_regs[1].u32, 4); // a[4..7]

      // A fragment distribution for M16N8K16 (row-major)
      int a_row0 = groupID, a_row1 = groupID + 8;
      int col_base = threadID_in_group * 4;

      // A is row-major: A[row][col] = A[row * K + col]
      for (int i = 0; i < 4; i++) {
        int col = col_base + (i & 0x3);
        A_mat[a_row0 * K + col] = a_vals[i];
      }
      for (int i = 4; i < 8; i++) {
        int col = col_base + (i & 0x3);
        A_mat[a_row1 * K + col] = a_vals[i];
      }

      // Unpack B fragments (4 S8 values packed in 1 U32 register)
      int8_t b_vals[4];
      memcpy(&b_vals[0], &b_regs[0].u32, 4);

      // B fragment distribution for M16N8K16 (column-major)
      int b_col = groupID;
      int row_base = threadID_in_group * 4;

      // B is column-major: B[n][k] stored as B[n*K + k]
      for (int i = 0; i < 4; i++) {
        int row = row_base + i;
        B_mat[b_col * K + row] = b_vals[i];
      }

      // Place C fragments (4 S32 accumulator values)
      int c_row0 = groupID, c_row1 = groupID + 8;
      int c_col0 = threadID_in_group * 2, c_col1 = threadID_in_group * 2 + 1;

      C_mat[c_row0 * N + c_col0] = c_regs[0].s32;
      C_mat[c_row0 * N + c_col1] = c_regs[1].s32;
      C_mat[c_row1 * N + c_col0] = c_regs[2].s32;
      C_mat[c_row1 * N + c_col1] = c_regs[3].s32;

    } else if (K == 32) {
      // M16N8K32: A needs 4 registers, B needs 2 registers
      ptx_reg_t a_regs[4], b_regs[2], c_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 4);
      thread->get_vector_operand_values(src_b, b_regs, 2);
      thread->get_vector_operand_values(src_c, c_regs, 4);

      // Unpack A fragments (16 S8 values packed in 4 U32 registers)
      int8_t a_vals[16];
      for (int i = 0; i < 4; i++) {
        memcpy(&a_vals[i * 4], &a_regs[i].u32, 4);
      }

      // A fragment distribution for M16N8K32 (row-major)
      // a[0..7] in row=groupID, a[8..15] in row=groupID+8
      int a_row0 = groupID, a_row1 = groupID + 8;
      int col_base = threadID_in_group * 4;

      // A is row-major: A[row][col] = A[row * K + col]
      // Place a[0..7] in row=groupID
      for (int i = 0; i < 8; i++) {
        int col =
            (i < 4) ? (col_base + (i & 0x3)) : (col_base + (i & 0x3) + 16);
        A_mat[a_row0 * K + col] = a_vals[i];
      }
      // Place a[8..15] in row=groupID+8
      for (int i = 8; i < 16; i++) {
        int col =
            (i < 12) ? (col_base + (i & 0x3)) : (col_base + (i & 0x3) + 16);
        A_mat[a_row1 * K + col] = a_vals[i];
      }

      // Unpack B fragments (8 S8 values packed in 2 U32 registers)
      int8_t b_vals[8];
      for (int i = 0; i < 2; i++) {
        memcpy(&b_vals[i * 4], &b_regs[i].u32, 4);
      }

      // B fragment distribution for M16N8K32 (column-major)
      int b_col = groupID;
      int row_base = threadID_in_group * 4;

      // B is column-major: B[n][k] stored as B[n*K + k]
      for (int i = 0; i < 4; i++) {
        int row = row_base + i;
        B_mat[b_col * K + row] = b_vals[i];
      }
      for (int i = 4; i < 8; i++) {
        int row = row_base + (i & 0x3) + 16;
        B_mat[b_col * K + row] = b_vals[i];
      }

      // Place C fragments (4 S32 accumulator values - same for both K values)
      int c_row0 = groupID, c_row1 = groupID + 8;
      int c_col0 = threadID_in_group * 2, c_col1 = threadID_in_group * 2 + 1;

      C_mat[c_row0 * N + c_col0] = c_regs[0].s32;
      C_mat[c_row0 * N + c_col1] = c_regs[1].s32;
      C_mat[c_row1 * N + c_col0] = c_regs[2].s32;
      C_mat[c_row1 * N + c_col1] = c_regs[3].s32;
    }
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
        if (D_mat[m * N + n] > INT32_MAX)
          D_mat[m * N + n] = INT32_MAX;
        if (D_mat[m * N + n] < INT32_MIN)
          D_mat[m * N + n] = INT32_MIN;
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

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_s8_impl completed for M%dN%dK%d\n", M, N, K);
  }
}

// S8/U8 M8N8K16 specialized implementation for smaller M dimension
// Per PTX ISA: M8N8K16 with S8/U8 uses 8×8 output instead of 16×8
// Fragment distribution adapted for M=8 matrix dimension
void tensor_mma_s8_m8n8k16_impl(const ptx_instruction *pI, core_t *core,
                                warp_inst_t inst, bool saturate, unsigned tid,
                                const operand_info &dst) {
  // Constants for M8N8K16
  const int M = 8, N = 8, K = 16;

  // Fragment distribution pattern for 32 threads with M=8:
  //   groupID = lane_id / 4  (0-7)
  //   threadID_in_group = lane_id % 4  (0-3)
  //
  // Each thread holds:
  //   A: 4 S8 values (1 U32 register, packed 4 values) - from 8×16 matrix A
  //   B: 4 S8 values (1 U32 register, packed) - from 16×8 matrix B
  //   C/D: 2 S32 values (2 S32 registers) - from 8×8 output matrix D
  //
  // Matrix A[8×16] fragment distribution (row-major):
  //   row = groupID (only 0-7, no +8 offset since M=8)
  //   col = (threadID_in_group * 4) + i for a[i] where i = {0,..,3}
  //
  // Matrix B[16×8] fragment distribution (column-major):
  //   row = (threadID_in_group * 4) + i for b[i] where i = {0,..,3}
  //   col = groupID
  //
  // Output matrix D[8×8] mapping:
  //   row = groupID (only 0-7)
  //   col = (threadID_in_group * 2) + (i & 0x1) for d[i] where i = {0,1}

  // Step 1: Collect all fragments from all 32 threads into full matrices
  int32_t A_mat[M * K]; // 8×16
  int32_t B_mat[K * N]; // 16×8
  int32_t C_mat[M * N]; // 8×8
  int64_t D_mat[M * N]; // 8×8, use int64 to detect overflow

  // Initialize matrices
  for (int i = 0; i < M * K; i++)
    A_mat[i] = 0;
  for (int i = 0; i < K * N; i++)
    B_mat[i] = 0;
  for (int i = 0; i < M * N; i++)
    C_mat[i] = 0;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;           // 0-7
    unsigned threadID_in_group = lane_id % 4; // 0-3

    // M8N8K16: A needs 1 register, B needs 1 register, C needs 2 registers
    ptx_reg_t a_regs[1], b_regs[1], c_regs[2];
    thread->get_vector_operand_values(src_a, a_regs, 1);
    thread->get_vector_operand_values(src_b, b_regs, 1);
    thread->get_vector_operand_values(src_c, c_regs, 2);

    // Unpack A fragments (4 S8 values packed in 1 U32 register)
    int8_t a_vals[4];
    memcpy(&a_vals[0], &a_regs[0].u32, 4);

    // A fragment distribution for M8N8K16 (row-major)
    int a_row = groupID; // Only 0-7, no +8 offset
    int col_base = threadID_in_group * 4;

    // A is row-major: A[row][col] = A[row * K + col]
    for (int i = 0; i < 4; i++) {
      int col = col_base + i;
      A_mat[a_row * K + col] = a_vals[i];
    }

    // Unpack B fragments (4 S8 values packed in 1 U32 register)
    int8_t b_vals[4];
    memcpy(&b_vals[0], &b_regs[0].u32, 4);

    // B fragment distribution for M8N8K16 (column-major)
    int b_col = groupID;
    int row_base = threadID_in_group * 4;

    // B is column-major: B[n][k] stored as B[n*K + k]
    for (int i = 0; i < 4; i++) {
      int row = row_base + i;
      B_mat[b_col * K + row] = b_vals[i];
    }

    // Place C fragments (2 S32 accumulator values for M=8)
    int c_row = groupID; // Only 0-7
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;

    C_mat[c_row * N + c_col0] = c_regs[0].s32;
    C_mat[c_row * N + c_col1] = c_regs[1].s32;
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
        if (D_mat[m * N + n] > INT32_MAX)
          D_mat[m * N + n] = INT32_MAX;
        if (D_mat[m * N + n] < INT32_MIN)
          D_mat[m * N + n] = INT32_MIN;
      }
    }
  }

  // Step 3: Distribute results back to threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;
    unsigned threadID_in_group = lane_id % 4;

    ptx_reg_t d_regs[4]; // Function signature requires 4 args (checks
                         // num_elements internally)
    int d_row = groupID; // Only 0-7
    int d_col0 = threadID_in_group * 2;
    int d_col1 = threadID_in_group * 2 + 1;

    d_regs[0].s32 = (int32_t)D_mat[d_row * N + d_col0];
    d_regs[1].s32 = (int32_t)D_mat[d_row * N + d_col1];
    d_regs[2].s32 = 0; // Padding (function only writes first num_elements)
    d_regs[3].s32 = 0; // Padding (function only writes first num_elements)

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_s8_m8n8k16_impl completed for M8N8K16\n");
  }
}

} // namespace flash_gpgpu_sim
