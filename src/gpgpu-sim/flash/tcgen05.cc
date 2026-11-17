// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Jimmy Kwa, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#include "tcgen05.h"

#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "../../trace.h"
#include "../gpu-sim.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <vector>

class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

static unsigned tcgen05_cta_group(const ptx_instruction *pI) {
  std::list<int> options = pI->get_options();
  for (std::list<int>::const_iterator i = options.begin(); i != options.end();
       ++i) {
    if (*i == TCGEN05_CTA_GROUP_1_OPTION)
      return 1;
    if (*i == TCGEN05_CTA_GROUP_2_OPTION)
      return 2;
  }
  return 1;
}

static void tcgen05_assert_cta_group1(const ptx_instruction *pI) {
  assert(tcgen05_cta_group(pI) == 1 &&
         "TCGen05 cta_group::2 is parsed but not implemented");
}

bool tcgen05_has_option(const ptx_instruction *pI, int option) {
  std::list<int> options = pI->get_options();
  for (std::list<int>::const_iterator i = options.begin(); i != options.end();
       ++i) {
    if (*i == option)
      return true;
  }
  return false;
}

static bool tcgen05_is_warp_leader(const ptx_thread_info *thread) {
  return thread->get_laneid() == 0;
}

static flash_gpgpu_sim::tcgen05_tmem_scope_t
tcgen05_tmem_scope(const ptx_instruction *pI, const ptx_thread_info *thread) {
  unsigned sm_id = thread->get_hw_sid();
  if (sm_id == (unsigned)-1)
    sm_id = 0;

  unsigned cta_id = thread->get_hw_ctaid();
  if (cta_id == (unsigned)-1)
    cta_id = thread->get_flat_ctaid();

  return flash_gpgpu_sim::tcgen05_tmem_scope_t{sm_id, cta_id,
                                               tcgen05_cta_group(pI)};
}

static uint32_t tcgen05_apply_thread_lane(uint32_t address,
                                          const ptx_thread_info *thread) {
  flash_gpgpu_sim::tcgen05_tmem_address_t decoded =
      flash_gpgpu_sim::tcgen05_decode_tmem_address(address);
  return flash_gpgpu_sim::tcgen05_encode_tmem_address(
      decoded.lane + thread->get_laneid(), decoded.column);
}

static ptx_reg_t tcgen05_read_operand(const ptx_instruction *pI,
                                      ptx_thread_info *thread,
                                      unsigned operand_index) {
  assert(operand_index < pI->get_num_operands());
  const operand_info &op = pI->operand_lookup(operand_index);
  return thread->get_operand_value(op, op, B32_TYPE, thread, 1);
}

static ptx_reg_t tcgen05_read_u64_operand(const ptx_instruction *pI,
                                          ptx_thread_info *thread,
                                          unsigned operand_index) {
  assert(operand_index < pI->get_num_operands());
  const operand_info &op = pI->operand_lookup(operand_index);
  return thread->get_operand_value(op, op, B64_TYPE, thread, 1);
}

static mem_addr_t tcgen05_eval_address(const operand_info &op,
                                       ptx_thread_info *thread) {
  ptx_reg_t addr = thread->get_operand_value(op, op, B32_TYPE, thread, 0);
  return addr.u32;
}

static enum _memory_space_t tcgen05_effective_space(const operand_info &op) {
  if (op.get_addr_space() != undefined_space)
    return op.get_addr_space();
  if (op.is_shared())
    return shared_space;
  if (op.is_memory_operand() || op.get_type() == address_t ||
      op.get_type() == symbolic_t) {
    const symbol *sym = op.get_symbol();
    if (sym->is_global())
      return global_space;
    if (sym->is_local())
      return local_space;
  }
  return undefined_space;
}

static void tcgen05_write_u32_destination(
    const ptx_instruction *pI, ptx_thread_info *thread, const operand_info &dst,
    uint32_t value, enum _memory_space_t default_space = undefined_space) {
  ptx_reg_t data;
  data.u32 = value;

  if (dst.is_reg()) {
    thread->set_operand_value(dst, data, B32_TYPE, thread, pI);
    return;
  }

  enum _memory_space_t space = tcgen05_effective_space(dst);
  if (space == undefined_space && default_space != undefined_space) {
    space = default_space;
  }
  mem_addr_t addr = tcgen05_eval_address(dst, thread);
  switch (space) {
  case shared_space:
    thread->m_shared_mem->write(addr, sizeof(uint32_t), &data.u128, thread, pI);
    thread->m_last_effective_address = addr;
    thread->m_last_memory_space = shared_space;
    return;
  case global_space:
    thread->get_global_memory()->write(addr, sizeof(uint32_t), &data.u128,
                                       thread, pI);
    thread->m_last_effective_address = addr;
    thread->m_last_memory_space = global_space;
    return;
  case local_space:
    thread->m_local_mem->write(addr, sizeof(uint32_t), &data.u128, thread, pI);
    thread->m_last_effective_address = addr;
    thread->m_last_memory_space = local_space;
    return;
  default:
    printf("GPGPU-Sim PTX: ERROR ** tcgen05.alloc destination is not a "
           "supported b32 register or memory operand: %s\n",
           dst.name().c_str());
    abort();
  }
}

static std::vector<uint32_t>
tcgen05_read_vector_words(const operand_info &src, ptx_thread_info *thread) {
  assert(src.is_vector());
  unsigned nelem = src.get_vect_nelem();
  std::vector<uint32_t> values(nelem, 0);
  for (unsigned i = 0; i < nelem; ++i) {
    const symbol *sym = src.vec_symbol(i);
    if (sym && strcmp(sym->name().c_str(), "_") != 0) {
      values[i] = thread->get_reg(sym).u32;
    }
  }
  return values;
}

static void tcgen05_write_vector_words(const operand_info &dst,
                                       ptx_thread_info *thread,
                                       const std::vector<uint32_t> &values) {
  assert(dst.is_vector());
  assert(dst.get_vect_nelem() == values.size());
  for (unsigned i = 0; i < values.size(); ++i) {
    const symbol *sym = dst.vec_symbol(i);
    if (sym && strcmp(sym->name().c_str(), "_") != 0) {
      ptx_reg_t value;
      value.u32 = values[i];
      thread->set_reg(sym, value);
      if (i == 0)
        thread->m_last_set_operand_value = value;
    }
  }
}

static void tcgen05_set_commit_mbarrier_info(const ptx_instruction *pI,
                                             ptx_thread_info *thread,
                                             uint32_t addr) {
  inst_t::mbarrier_info_t info;
  if (tcgen05_is_warp_leader(thread)) {
    info.bar_id = addr;
    info.bar_count = 1;
  }
  const_cast<ptx_instruction *>(pI)->set_mbarrier_info(thread->get_laneid(),
                                                       info);
}

static bool tcgen05_read_enable_input_d(const operand_info &op,
                                        ptx_thread_info *thread) {
  if (op.is_reg() && op.get_symbol() &&
      op.get_symbol()->type()->get_key().scalar_type() == PRED_TYPE) {
    ptx_reg_t predicate =
        thread->get_operand_value(op, op, PRED_TYPE, thread, 1);
    return (predicate.pred & 0x1) == 0;
  }

  ptx_reg_t value = thread->get_operand_value(op, op, B32_TYPE, thread, 1);
  return value.u32 != 0;
}

static std::vector<uint16_t> tcgen05_read_shared_f16_linearized(
    const flash_gpgpu_sim::tcgen05_shared_descriptor_t &desc, uint32_t nelem,
    ptx_thread_info *thread) {
  assert(!desc.leading_dimension_absolute &&
         "TCGen05 MMA absolute leading-dimension mode is not implemented");

  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid()) && desc.swizzle_mode != 0) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "shared_desc swizzle=%u linearized start=%u "
                      "nelem=%u\n",
                      desc.swizzle_mode, desc.start_address, nelem);
  }

  // Minimal functional path for CuTeDSL-generated FA4 smoke. The detailed
  // Blackwell shared-memory swizzle is not modeled yet.
  std::vector<uint16_t> values(nelem, 0);
  for (uint32_t i = 0; i < nelem; ++i) {
    thread->m_shared_mem->read(desc.start_address + i * sizeof(uint16_t),
                               sizeof(uint16_t), &values[i]);
  }
  return values;
}

static std::vector<uint32_t> tcgen05_read_shared_words_linearized(
    const flash_gpgpu_sim::tcgen05_shared_descriptor_t &desc, uint32_t rows,
    uint32_t words_per_row, ptx_thread_info *thread) {
  assert(!desc.leading_dimension_absolute &&
         "TCGen05 CP absolute leading-dimension mode is not implemented");

  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid()) && desc.swizzle_mode != 0) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "cp shared_desc swizzle=%u linearized start=%u "
                      "rows=%u row_words=%u\n",
                      desc.swizzle_mode, desc.start_address, rows,
                      words_per_row);
  }

  std::vector<uint32_t> values(rows * words_per_row, 0);
  for (uint32_t i = 0; i < values.size(); ++i) {
    thread->m_shared_mem->read(desc.start_address + i * sizeof(uint32_t),
                               sizeof(uint32_t), &values[i]);
  }
  return values;
}

static bool tcgen05_cp_shape_words(const ptx_instruction *pI, uint32_t *rows,
                                   uint32_t *words_per_row) {
  if (tcgen05_has_option(pI, TCGEN05_128X256B_OPTION)) {
    *rows = 128;
    *words_per_row = 8;
    return true;
  }
  if (tcgen05_has_option(pI, TCGEN05_128X128B_OPTION)) {
    *rows = 128;
    *words_per_row = 4;
    return true;
  }
  if (tcgen05_has_option(pI, TCGEN05_64X128B_OPTION)) {
    *rows = 64;
    *words_per_row = 4;
    return true;
  }
  if (tcgen05_has_option(pI, TCGEN05_32X128B_OPTION)) {
    *rows = 32;
    *words_per_row = 4;
    return true;
  }
  if (tcgen05_has_option(pI, TCGEN05_4X256B_OPTION)) {
    *rows = 4;
    *words_per_row = 8;
    return true;
  }
  return false;
}

static void tcgen05_check_disable_output_lane_zero(const operand_info &op,
                                                   ptx_thread_info *thread) {
  assert(op.is_vector());
  assert(op.get_vect_nelem() == 4 &&
         "Only TCGen05 cta_group::1 disable-output-lane vectors are supported");

  std::vector<uint32_t> lanes = tcgen05_read_vector_words(op, thread);
  for (unsigned i = 0; i < lanes.size(); ++i) {
    assert(lanes[i] == 0 &&
           "TCGen05 MMA disable-output-lane masks are not implemented");
  }
}

void handle_tcgen05_alloc_inst(const ptx_instruction *pI,
                               ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  const operand_info &dst = pI->operand_lookup(0);
  uint32_t ncols = tcgen05_read_operand(pI, thread, 1).u32;
  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  uint32_t base = manager.alloc(scope, ncols);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "alloc line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "base=%u ncols=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, base, ncols);
  }
  tcgen05_write_u32_destination(pI, thread, dst, base, shared_space);
}

void handle_tcgen05_dealloc_inst(const ptx_instruction *pI,
                                 ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  uint32_t base = tcgen05_read_operand(pI, thread, 0).u32;
  uint32_t ncols = tcgen05_read_operand(pI, thread, 1).u32;
  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "dealloc line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "base=%u ncols=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, base, ncols);
  }
  manager.dealloc(scope, base, ncols);
}

void handle_tcgen05_relinq_inst(const ptx_instruction *pI,
                                ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  manager.relinquish_alloc_permit(tcgen05_tmem_scope(pI, thread));
}

void handle_tcgen05_mma_inst(const ptx_instruction *pI,
                             ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 5);
  assert(tcgen05_has_option(pI, TCGEN05_KIND_F16_OPTION));

  const operand_info &d_tmem = pI->operand_lookup(0);
  const operand_info &a_desc_op = pI->operand_lookup(1);
  unsigned enable_input_d_operand = 4;
  if (pI->operand_lookup(4).is_vector()) {
    assert(pI->get_num_operands() >= 6);
    tcgen05_check_disable_output_lane_zero(pI->operand_lookup(4), thread);
    enable_input_d_operand = 5;
  }
  const operand_info &enable_input_d_op =
      pI->operand_lookup(enable_input_d_operand);

  uint32_t d_address = tcgen05_eval_address(d_tmem, thread);
  uint32_t a_tmem_address =
      a_desc_op.is_memory_operand()
          ? static_cast<uint32_t>(tcgen05_eval_address(a_desc_op, thread))
          : 0;
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "mma line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "d=%u a_mem=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, d_address,
                      a_desc_op.is_memory_operand() ? 1 : 0);
  }
  flash_gpgpu_sim::tcgen05_mma_descriptor_t mma_desc =
      flash_gpgpu_sim::tcgen05_decode_f16_mma_descriptor(
          tcgen05_read_operand(pI, thread, 3).u32, tcgen05_cta_group(pI));
  inst_t::tcgen05_dyn_info_t perf_info;
  perf_info.mma_work = 2ULL * mma_desc.m * mma_desc.n * mma_desc.k;
  const_cast<ptx_instruction *>(pI)->set_tcgen05_dyn_info(thread->get_laneid(),
                                                          perf_info);
  uint64_t a_desc_value = 0;
  if (!a_desc_op.is_memory_operand()) {
    a_desc_value = tcgen05_read_u64_operand(pI, thread, 1).u64;
  }
  uint64_t b_desc_value = tcgen05_read_u64_operand(pI, thread, 2).u64;
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(
        thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
        "mma_desc line=%u a=0x%016llx a_tmem=%u "
        "b=0x%016llx idesc=0x%08x\n",
        pI->source_line(), static_cast<unsigned long long>(a_desc_value),
        a_tmem_address, static_cast<unsigned long long>(b_desc_value),
        tcgen05_read_operand(pI, thread, 3).u32);
  }
  flash_gpgpu_sim::tcgen05_shared_descriptor_t b_desc =
      flash_gpgpu_sim::tcgen05_decode_shared_descriptor(b_desc_value);
  bool enable_input_d = tcgen05_read_enable_input_d(enable_input_d_op, thread);

  std::vector<uint16_t> a_values;
  if (a_desc_op.is_memory_operand()) {
    a_values = manager.read_matrix_packed_u16(scope, a_tmem_address, mma_desc.m,
                                              mma_desc.k);
  } else {
    flash_gpgpu_sim::tcgen05_shared_descriptor_t a_desc =
        flash_gpgpu_sim::tcgen05_decode_shared_descriptor(a_desc_value);
    a_values = tcgen05_read_shared_f16_linearized(
        a_desc, mma_desc.m * mma_desc.k, thread);
  }
  std::vector<uint16_t> b_values = tcgen05_read_shared_f16_linearized(
      b_desc, mma_desc.k * mma_desc.n, thread);

  std::vector<uint32_t> input_d;
  if (enable_input_d) {
    input_d =
        manager.read_matrix_words(scope, d_address, mma_desc.m, mma_desc.n);
  }

  std::vector<uint32_t> output = flash_gpgpu_sim::tcgen05_mma_f16_compute_words(
      mma_desc, a_values, b_values, input_d, enable_input_d);
  manager.write_matrix_words(scope, d_address, output, mma_desc.m, mma_desc.n);
}

void handle_tcgen05_commit_inst(const ptx_instruction *pI,
                                ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 1);
  assert(tcgen05_has_option(pI, TCGEN05_MBARRIER_ARRIVE_ONE_OPTION));

  const operand_info &bar = pI->operand_lookup(0);
  tcgen05_set_commit_mbarrier_info(pI, thread,
                                   tcgen05_eval_address(bar, thread));
}

void handle_tcgen05_ld_inst(const ptx_instruction *pI,
                            ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &addr = pI->operand_lookup(1);
  assert(dst.is_vector());

  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  uint32_t raw_address = tcgen05_eval_address(addr, thread);
  uint32_t address = tcgen05_apply_thread_lane(raw_address, thread);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "ld line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "raw=%u addr=%u n=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, raw_address, address,
                      dst.get_vect_nelem());
  }
  std::vector<uint32_t> values =
      manager.read_words(scope, address, dst.get_vect_nelem());
  tcgen05_write_vector_words(dst, thread, values);
}

void handle_tcgen05_st_inst(const ptx_instruction *pI,
                            ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  const operand_info &addr = pI->operand_lookup(0);
  const operand_info &src = pI->operand_lookup(1);
  assert(src.is_vector());

  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  uint32_t raw_address = tcgen05_eval_address(addr, thread);
  uint32_t address = tcgen05_apply_thread_lane(raw_address, thread);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "st line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "raw=%u addr=%u n=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, raw_address, address,
                      src.get_vect_nelem());
  }
  manager.write_words(scope, address, tcgen05_read_vector_words(src, thread));
}

void handle_tcgen05_wait_inst(const ptx_instruction *pI,
                              ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(pI);
  (void)pI;
  (void)thread;
}

void handle_tcgen05_cp_inst(const ptx_instruction *pI,
                            ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  uint32_t rows = 0;
  uint32_t words_per_row = 0;
  assert(tcgen05_cp_shape_words(pI, &rows, &words_per_row) &&
         "Unsupported TCGen05 CP tile shape");

  const operand_info &dst_tmem = pI->operand_lookup(0);
  uint32_t dst_address =
      static_cast<uint32_t>(tcgen05_eval_address(dst_tmem, thread));
  uint64_t src_desc_value = tcgen05_read_u64_operand(pI, thread, 1).u64;
  flash_gpgpu_sim::tcgen05_shared_descriptor_t src_desc =
      flash_gpgpu_sim::tcgen05_decode_shared_descriptor(src_desc_value);
  std::vector<uint32_t> values = tcgen05_read_shared_words_linearized(
      src_desc, rows, words_per_row, thread);

  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "cp line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "dst=%u rows=%u row_words=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, dst_address, rows, words_per_row);
  }
  manager.write_matrix_words(scope, dst_address, values, rows, words_per_row);
}

void handle_tcgen05_shift_inst(const ptx_instruction *pI,
                               ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 1);
  (void)thread;
}

void handle_tcgen05_fence_inst(const ptx_instruction *pI,
                               ptx_thread_info *thread) {
  (void)pI;
  (void)thread;
}

tcgen05_timing_config_t
tcgen05_timing_config(const shader_core_config *config) {
  tcgen05_timing_config_t result;
  result.mma_issue_interval = config->ptx_opcode_tcgen05_mma_issue_interval;
  result.mma_completion_tail_latency =
      config->ptx_opcode_tcgen05_mma_completion_tail_latency;
  result.mma_f16_flops_per_cycle =
      config->ptx_opcode_tcgen05_mma_f16_flops_per_cycle;
  result.async_queue_depth = config->gpgpu_tcgen05_async_queue_depth;
  return result;
}

int tcgen05_opcode(const warp_inst_t *inst) {
  const ptx_instruction *ptx_inst = dynamic_cast<const ptx_instruction *>(inst);
  return ptx_inst == NULL ? -1 : ptx_inst->get_opcode();
}

tcgen05_op_kind_t tcgen05_timing_kind(int opcode) {
  switch (opcode) {
  case TCGEN05_MMA_OP:
    return TCGEN05_TIMING_MMA;
  case TCGEN05_CP_OP:
    return TCGEN05_TIMING_CP;
  case TCGEN05_SHIFT_OP:
    return TCGEN05_TIMING_SHIFT;
  case TCGEN05_LD_OP:
    return TCGEN05_TIMING_LD;
  case TCGEN05_ST_OP:
    return TCGEN05_TIMING_ST;
  default:
    abort();
  }
}

bool is_tcgen05_timing_data_op(int opcode) {
  return opcode == TCGEN05_MMA_OP || opcode == TCGEN05_CP_OP ||
         opcode == TCGEN05_SHIFT_OP || opcode == TCGEN05_LD_OP ||
         opcode == TCGEN05_ST_OP;
}

static unsigned tcgen05_cp_shape_index(const ptx_instruction *inst) {
  if (tcgen05_has_option(inst, TCGEN05_128X256B_OPTION))
    return 0;
  if (tcgen05_has_option(inst, TCGEN05_128X128B_OPTION))
    return 1;
  if (tcgen05_has_option(inst, TCGEN05_64X128B_OPTION))
    return 2;
  if (tcgen05_has_option(inst, TCGEN05_32X128B_OPTION))
    return 3;
  if (tcgen05_has_option(inst, TCGEN05_4X256B_OPTION))
    return 4;
  fprintf(stderr, "GPGPU-Sim: unsupported TCGen05 CP timing shape: %s\n",
          inst->to_string().c_str());
  abort();
}

static unsigned tcgen05_vector_width_index(unsigned width) {
  unsigned index = 0;
  unsigned value = 1;
  while (value < width && index < 7) {
    value <<= 1;
    ++index;
  }
  if (value != width) {
    fprintf(stderr, "GPGPU-Sim: unsupported TCGen05 vector width x%u\n", width);
    abort();
  }
  return index;
}

tcgen05_op_t tcgen05_timing_op(const ptx_instruction *inst,
                               const ptx_instruction *dynamic_inst,
                               const shader_core_config *config) {
  tcgen05_op_t op;
  const int opcode = inst->get_opcode();
  op.kind = tcgen05_timing_kind(opcode);
  if (opcode == TCGEN05_MMA_OP) {
    assert(dynamic_inst != NULL);
    op.work = dynamic_inst->get_tcgen05_dyn_info(0).mma_work;
    assert(op.work != 0);
  } else if (opcode == TCGEN05_CP_OP) {
    const unsigned index = tcgen05_cp_shape_index(inst);
    op.completion_latency = config->tcgen05_cp_completion_latency[index];
    op.initiation_interval = config->tcgen05_cp_initiation_interval[index];
  } else if (opcode == TCGEN05_LD_OP || opcode == TCGEN05_ST_OP) {
    const unsigned operand_index = opcode == TCGEN05_LD_OP ? 0 : 1;
    const operand_info &vector_operand = inst->operand_lookup(operand_index);
    assert(vector_operand.is_vector());
    const unsigned index =
        tcgen05_vector_width_index(vector_operand.get_vect_nelem());
    if (opcode == TCGEN05_LD_OP) {
      op.completion_latency = config->tcgen05_ld_completion_latency[index];
      op.initiation_interval = config->tcgen05_ld_initiation_interval[index];
    } else {
      op.completion_latency = config->tcgen05_st_completion_latency[index];
      op.initiation_interval = config->tcgen05_st_initiation_interval[index];
    }
  } else {
    assert(opcode == TCGEN05_SHIFT_OP);
    op.completion_latency = config->ptx_opcode_tcgen05_shift_latency;
    op.initiation_interval = config->ptx_opcode_tcgen05_mma_issue_interval;
  }
  return op;
}

} // namespace flash_gpgpu_sim
