#include "tensor_wgmma.h"

#include <cmath>
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

float e4m3_to_f32(uint8_t bits) {
  const int sign = (bits >> 7) & 0x1;
  const int exp = (bits >> 3) & 0xF;
  const int mant = bits & 0x7;

  float value = 0.0f;
  if (exp == 0) {
    value = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant) / 8.0f, -6);
  } else if (exp == 0xF && mant == 0x7) {
    value = NAN;
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
  }

  return sign ? -value : value;
}

float e5m2_to_f32(uint8_t bits) {
  const int sign = (bits >> 7) & 0x1;
  const int exp = (bits >> 2) & 0x1F;
  const int mant = bits & 0x3;

  float value = 0.0f;
  if (exp == 0) {
    value = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant) / 4.0f, -14);
  } else if (exp == 0x1F) {
    value = mant == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mant) / 4.0f, exp - 15);
  }

  return sign ? -value : value;
}

float fp8_to_f32(uint8_t bits, bool is_e4m3) {
  return is_e4m3 ? e4m3_to_f32(bits) : e5m2_to_f32(bits);
}

} // namespace

void wgmma_m64n8k32_fp8_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst, bool a_is_e4m3,
                             bool b_is_e4m3) {
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
      uint8_t b_bits = 0;
      unsigned b_smem_addr =
          wgmma_gmma_k_major_smem_addr(b_desc, col, k, sizeof(uint8_t), K);
      representative_thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits),
                                                &b_bits);
      B_mat[col * K + k] = fp8_to_f32(b_bits, b_is_e4m3);
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
            fp8_to_f32(packed_u8(a_regs[reg], byte), a_is_e4m3);
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
