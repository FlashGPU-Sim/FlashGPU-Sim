#include "tensor_wgmma.h"

#include <cassert>
#include <cstdint>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../../gpu-sim.h"
#include "../mma/tensor_mma.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

unsigned wgmma_warp_base_tid(core_t *core, const warp_inst_t &inst) {
  if (core->get_gpu()->is_functional_sim())
    return inst.warp_id_func() * core->get_warp_size();
  return inst.warp_id() * core->get_warp_size();
}

unsigned wgmma_thread_count(core_t *core, const warp_inst_t &inst) {
  if (inst.is_wgmma_warpgroup())
    return inst.wgmma_warpgroup_size() * core->get_warp_size();
  return core->get_warp_size();
}

unsigned wgmma_hw_tid(core_t *core, const warp_inst_t &inst,
                      unsigned thread_idx) {
  unsigned warp_size = core->get_warp_size();
  if (inst.is_wgmma_warpgroup()) {
    unsigned warpgroup_slot = thread_idx / warp_size;
    unsigned lane = thread_idx % warp_size;
    return inst.wgmma_warpgroup_warp_id(warpgroup_slot) * warp_size + lane;
  }
  return wgmma_warp_base_tid(core, inst) + thread_idx;
}

ptx_thread_info *wgmma_thread(core_t *core, const warp_inst_t &inst,
                              unsigned thread_idx) {
  ptx_thread_info *thread =
      core->get_thread_info()[wgmma_hw_tid(core, inst, thread_idx)];
  assert(thread != NULL);
  return thread;
}

uint16_t packed_f16(const ptx_reg_t &reg, int half_index) {
  if (half_index == 0)
    return static_cast<uint16_t>(reg.u32 & 0xFFFFu);
  return static_cast<uint16_t>((reg.u32 >> 16) & 0xFFFFu);
}

unsigned wgmma_lane(const ptx_thread_info *thread) {
  return static_cast<unsigned>(thread->get_flat_tid()) % 128;
}

uint64_t gmma_desc_base(uint64_t desc) { return (desc & 0x3FFFULL) << 4; }

uint64_t gmma_desc_leading_byte_offset(uint64_t desc) {
  return ((desc >> 16) & 0x3FFFULL) << 4;
}

uint64_t gmma_desc_stride_byte_offset(uint64_t desc) {
  return ((desc >> 32) & 0x3FFFULL) << 4;
}

unsigned gmma_k_major_b_smem_addr(uint64_t desc, int col, int k) {
  constexpr unsigned bytes_per_f16 = sizeof(uint16_t);
  uint64_t base = gmma_desc_base(desc);
  uint64_t leading = gmma_desc_leading_byte_offset(desc);
  uint64_t stride = gmma_desc_stride_byte_offset(desc);

  unsigned contiguous_k =
      stride == 0 ? 16 : static_cast<unsigned>(stride / bytes_per_f16);
  if (contiguous_k == 0)
    contiguous_k = 16;

  return static_cast<unsigned>(base + (k / contiguous_k) * leading +
                               col * stride +
                               (k % contiguous_k) * bytes_per_f16);
}

void wgmma_accumulator_coord(unsigned lane, int reg, int &row, int &col) {
  int tid_mma_col = lane % 4;
  int tid_row = (lane / 4) % 8;
  int tid_m_block = lane / 32;

  int reg_col = reg & 0x1;
  int reg_row_block = (reg >> 1) & 0x1;

  row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
  col = 2 * tid_mma_col + reg_col;
}

void wgmma_m64n8k16_f16_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  constexpr int M = 64;
  constexpr int N = 8;
  constexpr int K = 16;

  if (pI->get_wgmma_shape_n() != 8 || pI->get_wgmma_shape_k() != 16)
    return;

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
      unsigned b_smem_addr = gmma_k_major_b_smem_addr(b_desc, col, k);
      representative_thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits),
                                                &b_bits);
      B_mat[col * K + k] = mma_f16_to_f32(b_bits);
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

      A_mat[row * K + k0] = mma_f16_to_f32(packed_f16(a_regs[reg], 0));
      A_mat[row * K + k0 + 1] = mma_f16_to_f32(packed_f16(a_regs[reg], 1));
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
      wgmma_accumulator_coord(lane, reg, row, col);

      float product = 0.0f;
      for (int k = 0; k < K; ++k)
        product += A_mat[row * K + k] * B_mat[col * K + k];

      d_regs[reg].f32 = product + (include_c ? c_regs[reg].f32 : 0.0f);
    }

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }
}

} // namespace

void wgmma_mma_async_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  wgmma_m64n8k16_f16_impl(pI, core, inst);
}

void wgmma_mma_async_sp_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void wgmma_fence_impl(const ptx_instruction *pI, core_t *core,
                      warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void wgmma_commit_group_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void wgmma_wait_group_impl(const ptx_instruction *pI, core_t *core,
                           warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

void setmaxnreg_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

} // namespace flash_gpgpu_sim
