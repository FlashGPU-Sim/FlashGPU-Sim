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

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../../../../libcuda/gpgpu_context.h"

namespace flash_gpgpu_sim {

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
  float A_mat[M * K];  // MxK
  float B_mat[K * N];  // KxN
  float C_mat[M * N];  // MxN
  float D_mat[M * N];  // MxN

  // Initialize matrices
  for (int i = 0; i < M * K; i++) A_mat[i] = 0.0f;
  for (int i = 0; i < K * N; i++) B_mat[i] = 0.0f;
  for (int i = 0; i < M * N; i++) C_mat[i] = 0.0f;

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_f16_impl M=%d N=%d K=%d is_bf16=%d\n", M, N, K, is_bf16);
  }

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Collect fragments from all threads
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    unsigned groupID = lane_id / 4;  // 0-7
    unsigned threadID_in_group = lane_id % 4;  // 0-3

    // Fragment distribution differs based on K dimension
    if (K == 8) {
      // M16N8K8: A needs 2 registers, B needs 1 register
      ptx_reg_t a_regs[2], b_regs[1], c_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 2);
      thread->get_vector_operand_values(src_b, b_regs, 1);
      thread->get_vector_operand_values(src_c, c_regs, 4);

      // Unpack A fragments (4 F16 values)
      uint16_t a_u16[4];
      a_u16[0] = a_regs[0].u16;
      a_u16[1] = (a_regs[0].u32 >> 16) & 0xFFFF;
      a_u16[2] = a_regs[1].u16;
      a_u16[3] = (a_regs[1].u32 >> 16) & 0xFFFF;

      int a_row0 = groupID, a_row1 = groupID + 8;
      int a_col0 = threadID_in_group * 2, a_col1 = a_col0 + 1;

      if (is_bf16) {
        A_mat[a_row0 * K + a_col0] = mma_bf16_to_f32(a_u16[0]);
        A_mat[a_row0 * K + a_col1] = mma_bf16_to_f32(a_u16[1]);
        A_mat[a_row1 * K + a_col0] = mma_bf16_to_f32(a_u16[2]);
        A_mat[a_row1 * K + a_col1] = mma_bf16_to_f32(a_u16[3]);
      } else {
        A_mat[a_row0 * K + a_col0] = mma_f16_to_f32(a_u16[0]);
        A_mat[a_row0 * K + a_col1] = mma_f16_to_f32(a_u16[1]);
        A_mat[a_row1 * K + a_col0] = mma_f16_to_f32(a_u16[2]);
        A_mat[a_row1 * K + a_col1] = mma_f16_to_f32(a_u16[3]);
      }

      // Unpack B fragments (2 F16 values)
      uint16_t b_u16[2];
      b_u16[0] = b_regs[0].u16;
      b_u16[1] = (b_regs[0].u32 >> 16) & 0xFFFF;

      int b_row0 = threadID_in_group * 2, b_row1 = b_row0 + 1;
      int b_col = groupID;

      if (is_bf16) {
        B_mat[b_col * K + b_row0] = mma_bf16_to_f32(b_u16[0]);
        B_mat[b_col * K + b_row1] = mma_bf16_to_f32(b_u16[1]);
      } else {
        B_mat[b_col * K + b_row0] = mma_f16_to_f32(b_u16[0]);
        B_mat[b_col * K + b_row1] = mma_f16_to_f32(b_u16[1]);
      }

      // C fragments (same for all K values)
      int c_row0 = groupID, c_row1 = groupID + 8;
      int c_col0 = threadID_in_group * 2, c_col1 = c_col0 + 1;
      C_mat[c_row0 * N + c_col0] = c_regs[0].f32;
      C_mat[c_row0 * N + c_col1] = c_regs[1].f32;
      C_mat[c_row1 * N + c_col0] = c_regs[2].f32;
      C_mat[c_row1 * N + c_col1] = c_regs[3].f32;
    } else if (K == 16) {
      // M16N8K16: A needs 4 registers, B needs 2 registers
      ptx_reg_t a_regs[4], b_regs[2], c_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 4);
      thread->get_vector_operand_values(src_b, b_regs, 2);
      thread->get_vector_operand_values(src_c, c_regs, 4);

      // Unpack A fragments (8 F16 values) with proper distribution per PTX ISA
      uint16_t a_u16[8];
      for (int i = 0; i < 4; i++) {
        a_u16[i*2] = a_regs[i].u16;
        a_u16[i*2+1] = (a_regs[i].u32 >> 16) & 0xFFFF;
      }

      for (int i = 0; i < 8; i++) {
        // row = groupID for i in {0,1,4,5}; groupID+8 otherwise
        int a_row = (i < 2 || (i >= 4 && i < 6)) ? groupID : (groupID + 8);
        // col = (threadID_in_group*2)+(i&1) for i<4; same+8 for i>=4
        int a_col = (i < 4) ? (threadID_in_group * 2 + (i & 1)) :
                              (threadID_in_group * 2 + (i & 1) + 8);
        A_mat[a_row * K + a_col] = is_bf16 ? mma_bf16_to_f32(a_u16[i]) :
                                              mma_f16_to_f32(a_u16[i]);
      }

      // Unpack B fragments (4 F16 values)
      uint16_t b_u16[4];
      for (int i = 0; i < 2; i++) {
        b_u16[i*2] = b_regs[i].u16;
        b_u16[i*2+1] = (b_regs[i].u32 >> 16) & 0xFFFF;
      }

      for (int i = 0; i < 4; i++) {
        // row = (threadID_in_group*2)+(i&1) for i<2; same+8 for i>=2
        int b_row = (i < 2) ? (threadID_in_group * 2 + (i & 1)) :
                              (threadID_in_group * 2 + (i & 1) + 8);
        int b_col = groupID;
        B_mat[b_col * K + b_row] = is_bf16 ? mma_bf16_to_f32(b_u16[i]) :
                                              mma_f16_to_f32(b_u16[i]);
      }

      // C fragments (same for all K values)
      int c_row0 = groupID, c_row1 = groupID + 8;
      int c_col0 = threadID_in_group * 2, c_col1 = c_col0 + 1;
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

// F16/BF16 M8N8K4 specialized implementation with 4-computation decomposition
// Per PTX ISA: "A warp executing mma.m8n8k4 with .f16 floating point type will
// compute 4 MMA operations of shape .m8n8k4"
// Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-884-f16
void tensor_mma_f16_m8n8k4_impl(const ptx_instruction *pI, core_t *core,
                                warp_inst_t inst, bool is_bf16, unsigned tid,
                                const operand_info &dst) {
  // Constants for M8N8K4
  const int M = 8, N = 8, K = 4;

  // Helper structure to identify thread groups for 4 computations
  struct ThreadGroup {
    unsigned computation_idx;  // 0-3 (which of the 4 MMA operations)
    bool is_high_group;        // false: lanes 0-15, true: lanes 16-31
    unsigned local_lane;       // 0-3 (position within 4-thread group)
  };

  // Lambda to compute thread group from lane_id
  auto get_thread_group = [](unsigned lane_id) -> ThreadGroup {
    return ThreadGroup{
      .computation_idx = (lane_id % 16) / 4,    // 0-3
      .is_high_group = (lane_id >= 16),         // lanes 16-31
      .local_lane = lane_id % 4                 // 0-3
    };
  };

  // 4 separate result matrices (one per computation)
  float A_mat[4][M * K];  // 4 separate 8×4 matrices
  float B_mat[4][K * N];  // 4 separate 4×8 matrices
  float C_mat[4][M * N];  // 4 separate 8×8 matrices
  float D_mat[4][M * N];  // 4 separate 8×8 output matrices

  // Initialize matrices
  for (int comp = 0; comp < 4; comp++) {
    for (int i = 0; i < M * K; i++) A_mat[comp][i] = 0.0f;
    for (int i = 0; i < K * N; i++) B_mat[comp][i] = 0.0f;
    for (int i = 0; i < M * N; i++) C_mat[comp][i] = 0.0f;
  }

  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b = pI->operand_lookup(2);
  const operand_info &src_c = pI->operand_lookup(3);

  // Step 1: Collect fragments from all 32 threads, distributing to their respective computations
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    ThreadGroup tg = get_thread_group(lane_id);
    unsigned comp = tg.computation_idx;  // Which of the 4 computations this thread belongs to

    // Each thread holds: 4 F16 values in A, 4 F16 values in B, 8 F32 values in C
    ptx_reg_t a_regs[2], b_regs[2], c_regs[8];
    thread->get_vector_operand_values(src_a, a_regs, 2);
    thread->get_vector_operand_values(src_b, b_regs, 2);
    thread->get_vector_operand_values(src_c, c_regs, 8);

    // Unpack A and B registers (4 F16 values each)
    uint16_t a_u16[4], b_u16[4];
    a_u16[0] = a_regs[0].u16;
    a_u16[1] = (a_regs[0].u32 >> 16) & 0xFFFF;
    a_u16[2] = a_regs[1].u16;
    a_u16[3] = (a_regs[1].u32 >> 16) & 0xFFFF;

    b_u16[0] = b_regs[0].u16;
    b_u16[1] = (b_regs[0].u32 >> 16) & 0xFFFF;
    b_u16[2] = b_regs[1].u16;
    b_u16[3] = (b_regs[1].u32 >> 16) & 0xFFFF;

    // Fragment distribution for Matrix A (row layout):
    // row = lane_id % 4          if lane_id < 16
    //       (lane_id % 4) + 4    otherwise
    // col = i                    for a[i] where i = {0,1,2,3}
    int a_row_base = (lane_id % 4) + (tg.is_high_group ? 4 : 0);
    for (int i = 0; i < 4; i++) {
      int row = a_row_base;
      int col = i;
      float val = is_bf16 ? mma_bf16_to_f32(a_u16[i]) : mma_f16_to_f32(a_u16[i]);
      A_mat[comp][row * K + col] = val;
    }

    // Fragment distribution for Matrix B (col layout):
    // row = i                    for b[i] where i = {0,1,2,3}
    // col = lane_id % 4          if lane_id < 16
    //       (lane_id % 4) + 4    otherwise
    int b_col_base = (lane_id % 4) + (tg.is_high_group ? 4 : 0);
    for (int i = 0; i < 4; i++) {
      int row = i;
      int col = b_col_base;
      float val = is_bf16 ? mma_bf16_to_f32(b_u16[i]) : mma_f16_to_f32(b_u16[i]);
      B_mat[comp][row * N + col] = val;
    }

    // Fragment distribution for Accumulator C (f32 type):
    // row = X                    if lane_id < 16
    //       X + 4                otherwise
    //       where X = (lane_id & 0b1) + (i & 0b10)  for c[i] where i = {0,...,7}
    // col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1)  for c[i]
    for (int i = 0; i < 8; i++) {
      int X = (lane_id & 0b1) + (i & 0b10);
      int row = X + (tg.is_high_group ? 4 : 0);
      int col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1);
      C_mat[comp][row * N + col] = c_regs[i].f32;
    }
  }

  // Step 2: Perform 4 separate MMA computations (REQUIRED BY PTX ISA)
  // Each computation operates on its own 8x8x4 matrices
  // Computation 0: uses threads 0-3, 16-19
  // Computation 1: uses threads 4-7, 20-23
  // Computation 2: uses threads 8-11, 24-27
  // Computation 3: uses threads 12-15, 28-31
  for (int comp = 0; comp < 4; comp++) {
    // D[comp] = A[comp] × B[comp] + C[comp]
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = C_mat[comp][i * N + j];
        for (int k = 0; k < K; k++) {
          sum += A_mat[comp][i * K + k] * B_mat[comp][k * N + j];
        }
        D_mat[comp][i * N + j] = sum;
      }
    }
  }

  // Step 3: Distribute results back to threads from their respective computations
  for (unsigned thrd = 0; thrd < core->get_warp_size(); thrd++) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];
    unsigned lane_id = thrd;
    ThreadGroup tg = get_thread_group(lane_id);
    unsigned comp = tg.computation_idx;  // Which computation this thread belongs to

    ptx_reg_t d_regs[8];

    // Use same accumulator distribution formula as for C
    for (int i = 0; i < 8; i++) {
      int X = (lane_id & 0b1) + (i & 0b10);
      int row = X + (tg.is_high_group ? 4 : 0);
      int col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1);
      d_regs[i].f32 = D_mat[comp][row * N + col];  // Read from this thread's computation result
    }

    // M8N8K4 uses 8 F32 output registers
    thread->set_wmma_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2], d_regs[3],
                                            d_regs[4], d_regs[5], d_regs[6], d_regs[7]);
  }

  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("GPGPU-Sim: tensor_mma_f16_m8n8k4_impl completed (4-computation decomposition)\n");
  }
}

} // namespace flash_gpgpu_sim
