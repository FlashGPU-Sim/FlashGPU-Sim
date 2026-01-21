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

#include <cstdio>

#include "../../../../libcuda/gpgpu_context.h"
#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"

namespace flash_gpgpu_sim {

// TF32 floating-point MMA implementation
void tensor_mma_tf32_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t inst, int M, int N, int K, bool saturate,
                          unsigned tid, const operand_info &dst) {
  // TF32 implementation for M16N8K4 and M16N8K8
  // K=4: A needs 2 F32 registers, B needs 1 F32 register
  // K=8: A needs 4 F32 registers, B needs 2 F32 registers
  // Each TF32 value is stored as F32 but with reduced precision (10-bit
  // mantissa)

  // Step 1: Collect all fragments from all 32 threads
  float A_mat[M * K]; // 16×K
  float B_mat[K * N]; // K×8
  float C_mat[M * N]; // 16×8
  float D_mat[M * N]; // 16×8

  // Initialize matrices
  for (int i = 0; i < M * K; i++)
    A_mat[i] = 0.0f;
  for (int i = 0; i < K * N; i++)
    B_mat[i] = 0.0f;
  for (int i = 0; i < M * N; i++)
    C_mat[i] = 0.0f;

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;           // 0-7
    unsigned threadID_in_group = lane_id % 4; // 0-3

    // Fragment distribution differs based on K dimension
    if (K == 4) {
      // M16N8K4 uses 2 F32 registers for A, 1 F32 for B
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
    } else if (K == 8) {
      // M16N8K8 uses 4 F32 registers for A, 2 F32 for B
      ptx_reg_t a_regs[4], b_regs[2], c_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 4);
      thread->get_vector_operand_values(src_b, b_regs, 2);
      thread->get_vector_operand_values(src_c, c_regs, 4);

      // Unpack and place A fragments with TF32 rounding (16×8 matrix)
      // Each thread holds 4 TF32 values
      // Fragment order: [row0,col0], [row1,col0], [row0,col1], [row1,col1]
      int a_row0 = groupID;
      int a_row1 = groupID + 8;
      int a_col0 = threadID_in_group;
      int a_col1 = threadID_in_group + 4;

      A_mat[a_row0 * K + a_col0] = mma_tf32_round(a_regs[0].f32);
      A_mat[a_row1 * K + a_col0] = mma_tf32_round(a_regs[1].f32);
      A_mat[a_row0 * K + a_col1] = mma_tf32_round(a_regs[2].f32);
      A_mat[a_row1 * K + a_col1] = mma_tf32_round(a_regs[3].f32);

      // Unpack and place B fragments with TF32 rounding (8×8 matrix)
      // Each thread holds 2 TF32 values
      int b_row0 = threadID_in_group;
      int b_row1 = threadID_in_group + 4;
      int b_col = groupID;
      B_mat[b_col * K + b_row0] = mma_tf32_round(b_regs[0].f32);
      B_mat[b_col * K + b_row1] = mma_tf32_round(b_regs[1].f32);

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

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_tf32_impl completed for M16N8K%d\n", K);
  }
}

} // namespace flash_gpgpu_sim
