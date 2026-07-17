#include "tensor_wgmma.h"

#include <cassert>
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

uint16_t packed_f16(const ptx_reg_t &reg, int half_index) {
  if (half_index == 0)
    return static_cast<uint16_t>(reg.u32 & 0xFFFFu);
  return static_cast<uint16_t>((reg.u32 >> 16) & 0xFFFFu);
}

} // namespace

void wgmma_m64nXk16_f16_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  constexpr int M = 64;
  constexpr int K = 16;

  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b_desc = pI->operand_lookup(2);
  const operand_info &scale_d_pred = pI->operand_lookup(3);
  int N = pI->get_wgmma_shape_n();
  if (!dst.is_vector() ||
      dst.get_vect_nelem() != static_cast<unsigned>(N / 2)) {
    return;
  }

  float A_mat[M * K];
  float B_mat[256 * K];
  for (int i = 0; i < M * K; ++i)
    A_mat[i] = 0.0f;
  for (int i = 0; i < N * K; ++i)
    B_mat[i] = 0.0f;

  unsigned thread_count = wgmma_thread_count(core, inst);
  ptx_thread_info *representative_thread = wgmma_thread(core, inst, 0);
  bool a_from_register = src_a.is_vector();
  int operand_count = static_cast<int>(pI->get_num_operands());
  int scale_a = 1;
  int scale_b = 1;
  int major_a = 0;
  int major_b = 0;

  if (operand_count > 4) {
    ptx_reg_t scale = representative_thread->get_operand_value(
        pI->operand_lookup(4), pI->operand_lookup(4), S32_TYPE,
        representative_thread, 0);
    scale_a = scale.s32 == -1 ? -1 : 1;
  }
  if (operand_count > 5) {
    ptx_reg_t scale = representative_thread->get_operand_value(
        pI->operand_lookup(5), pI->operand_lookup(5), S32_TYPE,
        representative_thread, 0);
    scale_b = scale.s32 == -1 ? -1 : 1;
  }
  if (a_from_register) {
    if (operand_count > 6) {
      ptx_reg_t major = representative_thread->get_operand_value(
          pI->operand_lookup(6), pI->operand_lookup(6), S32_TYPE,
          representative_thread, 0);
      major_b = major.s32;
    }
  } else {
    if (operand_count > 6) {
      ptx_reg_t major = representative_thread->get_operand_value(
          pI->operand_lookup(6), pI->operand_lookup(6), S32_TYPE,
          representative_thread, 0);
      major_a = major.s32;
    }
    if (operand_count > 7) {
      ptx_reg_t major = representative_thread->get_operand_value(
          pI->operand_lookup(7), pI->operand_lookup(7), S32_TYPE,
          representative_thread, 0);
      major_b = major.s32;
    }
  }

  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(
      src_b_desc, src_b_desc, U64_TYPE, representative_thread, 0);
  uint64_t b_desc = static_cast<uint64_t>(b_desc_reg.u64);

  for (int col = 0; col < N; ++col) {
    for (int k = 0; k < K; ++k) {
      uint16_t b_bits = 0;
      unsigned b_smem_addr =
          major_b == 0
              ? wgmma_gmma_k_major_smem_addr(b_desc, col, k, sizeof(uint16_t),
                                             K)
              : wgmma_gmma_mn_major_smem_addr(b_desc, col, k, sizeof(uint16_t));
      representative_thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits),
                                                &b_bits);
      B_mat[col * K + k] = scale_b * mma_f16_to_f32(b_bits);
    }
  }

  if (a_from_register) {
    if (src_a.get_vect_nelem() != 4)
      return;
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

        A_mat[row * K + k0] =
            scale_a * mma_f16_to_f32(packed_f16(a_regs[reg], 0));
        A_mat[row * K + k0 + 1] =
            scale_a * mma_f16_to_f32(packed_f16(a_regs[reg], 1));
      }
    }
  } else {
    ptx_reg_t a_desc_reg = representative_thread->get_operand_value(
        src_a, src_a, U64_TYPE, representative_thread, 0);
    uint64_t a_desc = static_cast<uint64_t>(a_desc_reg.u64);
    for (int row = 0; row < M; ++row) {
      for (int k = 0; k < K; ++k) {
        uint16_t a_bits = 0;
        unsigned a_smem_addr =
            major_a == 0 ? wgmma_gmma_k_major_smem_addr(a_desc, row, k,
                                                        sizeof(uint16_t), K)
                         : wgmma_gmma_mn_major_smem_addr(a_desc, row, k,
                                                         sizeof(uint16_t));
        representative_thread->m_shared_mem->read(a_smem_addr, sizeof(a_bits),
                                                  &a_bits);
        A_mat[row * K + k] = scale_a * mma_f16_to_f32(a_bits);
      }
    }
  }

  for (unsigned thrd = 0; thrd < thread_count; ++thrd) {
    ptx_thread_info *thread = wgmma_thread(core, inst, thrd);
    unsigned lane = wgmma_lane(thread);
    unsigned dst_elements = dst.get_vect_nelem();

    ptx_reg_t pred = thread->get_operand_value(scale_d_pred, scale_d_pred,
                                               PRED_TYPE, thread, 0);
    bool include_c = (pred.pred & 0x1u) == 0;

    for (unsigned reg = 0; reg < dst_elements; ++reg) {
      ptx_reg_t c_reg = thread->get_reg(dst.vec_symbol(reg));
      int row = 0;
      int col_in_tile = 0;
      wgmma_m64n8_accumulator_coord(lane, reg % 4, row, col_in_tile);
      int col = static_cast<int>((reg / 4) * 8) + col_in_tile;

      float product = 0.0f;
      for (int k = 0; k < K; ++k)
        product += A_mat[row * K + k] * B_mat[col * K + k];

      ptx_reg_t d_reg;
      d_reg.f32 = product + (include_c ? c_reg.f32 : 0.0f);
      thread->set_reg(dst.vec_symbol(reg), d_reg);
    }
  }
}

} // namespace flash_gpgpu_sim
