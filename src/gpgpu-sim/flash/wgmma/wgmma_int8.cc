#include "tensor_wgmma.h"

#include <climits>
#include <cstdint>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

uint8_t packed_u8(const ptx_reg_t &reg, int byte_index) {
  return static_cast<uint8_t>((reg.u32 >> (byte_index * 8)) & 0xFFu);
}

int32_t int8_value(uint8_t bits, bool is_signed) {
  if (is_signed)
    return static_cast<int32_t>(static_cast<int8_t>(bits));
  return static_cast<int32_t>(bits);
}

int32_t saturate_s32(int64_t value) {
  if (value > INT_MAX)
    return INT_MAX;
  if (value < INT_MIN)
    return INT_MIN;
  return static_cast<int32_t>(value);
}

} // namespace

void wgmma_m64n8k32_int8_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst, bool a_is_signed,
                              bool b_is_signed) {
  constexpr int M = 64;
  constexpr int N = 8;
  constexpr int K = 32;

  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b_desc = pI->operand_lookup(2);
  const operand_info &scale_d_pred = pI->operand_lookup(3);
  if (!dst.is_vector() || dst.get_vect_nelem() != 4 || !src_a.is_vector() ||
      src_a.get_vect_nelem() != 4) {
    return;
  }

  int32_t A_mat[M * K];
  int32_t B_mat[N * K];
  for (int i = 0; i < M * K; ++i)
    A_mat[i] = 0;
  for (int i = 0; i < N * K; ++i)
    B_mat[i] = 0;

  unsigned thread_count = wgmma_thread_count(core, inst);
  ptx_thread_info *representative_thread = wgmma_thread(core, inst, 0);
  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(
      src_b_desc, src_b_desc, U64_TYPE, representative_thread, 0);
  uint64_t b_desc = static_cast<uint64_t>(b_desc_reg.u64);

  for (int col = 0; col < N; ++col) {
    for (int k = 0; k < K; ++k) {
      uint8_t b_bits = 0;
      unsigned b_smem_addr =
          wgmma_gmma_k_major_smem_addr(b_desc, col, k, sizeof(uint8_t), K);
      representative_thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits),
                                                &b_bits);
      B_mat[col * K + k] = int8_value(b_bits, b_is_signed);
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
      int k0 = 4 * tid_mma_col + 16 * reg_k_block;

      for (int byte = 0; byte < 4; ++byte)
        A_mat[row * K + k0 + byte] =
            int8_value(packed_u8(a_regs[reg], byte), a_is_signed);
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

      int64_t product = 0;
      for (int k = 0; k < K; ++k) {
        product += static_cast<int64_t>(A_mat[row * K + k]) *
                   static_cast<int64_t>(B_mat[col * K + k]);
      }

      int64_t result = product + (include_c ? c_regs[reg].s32 : 0);
      d_regs[reg].s32 = saturate_s32(result);
    }

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }
}

} // namespace flash_gpgpu_sim
