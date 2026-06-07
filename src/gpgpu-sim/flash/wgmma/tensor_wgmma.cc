#include "tensor_wgmma.h"

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

void wgmma_m64n8k16_f16_uniform_impl(const ptx_instruction *pI, core_t *core,
                                     warp_inst_t &inst) {
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

  unsigned tid = wgmma_warp_base_tid(core, inst);
  for (unsigned thrd = 0; thrd < core->get_warp_size(); ++thrd) {
    ptx_thread_info *thread = core->get_thread_info()[tid + thrd];

    ptx_reg_t a_regs[4];
    ptx_reg_t c_regs[4];
    thread->get_vector_operand_values(src_a, a_regs, 4);
    thread->get_vector_operand_values(dst, c_regs, 4);

    ptx_reg_t b_desc =
        thread->get_operand_value(src_b_desc, src_b_desc, U64_TYPE, thread, 0);
    uint64_t b_smem_addr = (static_cast<uint64_t>(b_desc.u64) & 0x3FFFULL) << 4;

    uint16_t b_bits = 0;
    thread->m_shared_mem->read(b_smem_addr, sizeof(b_bits), &b_bits);

    uint16_t a_bits = static_cast<uint16_t>(a_regs[0].u32 & 0xFFFFu);
    float product = static_cast<float>(pI->get_wgmma_shape_k()) *
                    mma_f16_to_f32(a_bits) * mma_f16_to_f32(b_bits);

    ptx_reg_t pred = thread->get_operand_value(scale_d_pred, scale_d_pred,
                                               PRED_TYPE, thread, 0);
    bool include_c = (pred.pred & 0x1u) == 0;

    ptx_reg_t d_regs[4];
    for (int i = 0; i < 4; ++i) {
      d_regs[i].f32 = product + (include_c ? c_regs[i].f32 : 0.0f);
    }

    thread->set_vector_operand_values(dst, d_regs[0], d_regs[1], d_regs[2],
                                      d_regs[3]);
  }
}

} // namespace

void wgmma_mma_async_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  wgmma_m64n8k16_f16_uniform_impl(pI, core, inst);
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
