#include "tensor_wgmma.h"

#include <cstdint>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../mma/tensor_mma.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

uint16_t packed_bf16(const ptx_reg_t &reg, int half_index) {
  if (half_index == 0)
    return static_cast<uint16_t>(reg.u32 & 0xFFFFu);
  return static_cast<uint16_t>((reg.u32 >> 16) & 0xFFFFu);
}

} // namespace

void wgmma_m64n8k16_bf16_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst) {
  constexpr int M = 64;
  constexpr int N = 8;
  constexpr int K = 16;

  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b_desc = pI->operand_lookup(2);
  const operand_info &scale_d_pred = pI->operand_lookup(3);
  if (!dst.is_vector() || dst.get_vect_nelem() != 4 || !src_a.is_vector() ||
      src_a.get_vect_nelem() != 4) {
    return;
  }

  float A_mat[M * K];
  float B_mat[N * K];
  for (int i = 0; i < M * K; ++i)
    A_mat[i] = 0.0f;
  for (int i = 0; i < N * K; ++i)
    B_mat[i] = 0.0f;

  unsigned thread_count = wgmma_thread_count(core, inst);
  ptx_thread_info *representative_thread = wgmma_thread(core, inst, 0);
  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(
      src_b_desc, src_b_desc, U64_TYPE, representative_thread, 0);
  uint64_t b_desc = static_cast<uint64_t>(b_desc_reg.u64);

  for (int col = 0; col < N; ++col) {
    for (int k = 0; k < K; ++k) {
      uint16_t b_bits = 0;
      unsigned b_smem_addr =
          wgmma_gmma_k_major_smem_addr(b_desc, col, k, sizeof(uint16_t), K);
      representative_thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits),
                                                &b_bits);
      B_mat[col * K + k] = mma_bf16_to_f32(b_bits);
    }
  }

  for (unsigned thrd = 0; thrd < thread_count; ++thrd) {
    ptx_thread_info *thread = wgmma_thread(core, inst, thrd);
    unsigned lane = wgmma_lane(thread);
    int tid_mma_col = lane % 4;
    int tid_row = (lane / 4) % 8;
    int tid_m_block = lane / 32;

    ptx_reg_t a_regs[4];
    thread->get_vector_operand_values(src_a, a_regs, 4);

    for (int reg = 0; reg < 4; ++reg) {
      int reg_row_block = reg & 0x1;
      int reg_k_block = (reg >> 1) & 0x1;
      int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
      int k0 = 2 * tid_mma_col + 8 * reg_k_block;

      A_mat[row * K + k0] = mma_bf16_to_f32(packed_bf16(a_regs[reg], 0));
      A_mat[row * K + k0 + 1] = mma_bf16_to_f32(packed_bf16(a_regs[reg], 1));
    }
  }

  for (unsigned thrd = 0; thrd < thread_count; ++thrd) {
    ptx_thread_info *thread = wgmma_thread(core, inst, thrd);
    unsigned lane = wgmma_lane(thread);

    ptx_reg_t c_regs[4];
    thread->get_vector_operand_values(dst, c_regs, 4);

    ptx_reg_t pred = thread->get_operand_value(scale_d_pred, scale_d_pred,
                                               PRED_TYPE, thread, 0);
    bool include_c = (pred.pred & 0x1u) == 0;

    ptx_reg_t d_regs[4];
    for (int reg = 0; reg < 4; ++reg) {
      int row = 0;
      int col = 0;
      wgmma_m64n8_accumulator_coord(lane, reg, row, col);

      float product = 0.0f;
      for (int k = 0; k < K; ++k)
        product += A_mat[row * K + k] * B_mat[col * K + k];

      d_regs[reg].f32 = product + (include_c ? c_regs[reg].f32 : 0.0f);
    }

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }
}

} // namespace flash_gpgpu_sim
