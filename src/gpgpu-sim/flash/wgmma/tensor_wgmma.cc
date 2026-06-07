#include "tensor_wgmma.h"

#include <cstdio>
#include <cstdlib>

#include "../../../cuda-sim/ptx_ir.h"

namespace flash_gpgpu_sim {

namespace {

void report_wgmma_not_implemented(const char *func, const ptx_instruction *pI) {
  const char *opcode = pI ? pI->get_opcode_cstr() : "<unknown>";
  int n = pI ? pI->get_wgmma_shape_n() : 0;
  int k = pI ? pI->get_wgmma_shape_k() : 0;
  bool sparse = pI ? pI->is_wgmma_sparse() : false;

  std::fprintf(stderr,
               "GPGPU-Sim: ERROR - %s reached for opcode %s, but WGMMA "
               "execution is not implemented yet (shape=m64n%dk%d, "
               "sparse=%d).\n",
               func, opcode, n, k, sparse ? 1 : 0);
  std::fflush(stderr);
  std::abort();
}

} // namespace

void wgmma_mma_async_impl(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst) {
  (void)core;
  (void)inst;
  report_wgmma_not_implemented(__func__, pI);
}

void wgmma_mma_async_sp_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)core;
  (void)inst;
  report_wgmma_not_implemented(__func__, pI);
}

void wgmma_fence_impl(const ptx_instruction *pI, core_t *core,
                      warp_inst_t &inst) {
  (void)core;
  (void)inst;
  report_wgmma_not_implemented(__func__, pI);
}

void wgmma_commit_group_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  (void)core;
  (void)inst;
  report_wgmma_not_implemented(__func__, pI);
}

void wgmma_wait_group_impl(const ptx_instruction *pI, core_t *core,
                           warp_inst_t &inst) {
  (void)core;
  (void)inst;
  report_wgmma_not_implemented(__func__, pI);
}

void setmaxnreg_impl(const ptx_instruction *pI, core_t *core,
                     warp_inst_t &inst) {
  (void)pI;
  (void)core;
  (void)inst;
}

} // namespace flash_gpgpu_sim
