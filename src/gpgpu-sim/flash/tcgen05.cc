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
#include "tcgen05/copy.h"
#include "tcgen05/ptx_internal.h"

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

unsigned tcgen05_cta_group(const ptx_instruction *pI) {
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

void tcgen05_assert_cta_group1(const ptx_instruction *pI) {
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

bool tcgen05_is_warp_leader(const ptx_thread_info *thread) {
  return thread->get_laneid() == 0;
}

flash_gpgpu_sim::tcgen05_tmem_scope_t
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

static uint32_t tcgen05_register_address(const ptx_instruction *inst,
                                         uint32_t base, ptx_thread_info *thread,
                                         unsigned reg) {
  unsigned lane = thread->get_laneid();
  const auto tid = thread->get_tid(), ntid = thread->get_ntid();
  const unsigned warp = (tid.x + ntid.x * (tid.y + ntid.y * tid.z)) / 32;
  base = tcgen05_tmem_warp_base(base, warp);
  unsigned lanes = 32, bits = 32;
  if (tcgen05_has_option(inst, TCGEN05_16X256B_OPTION)) {
    lanes = 16;
    bits = 256;
  } else if (tcgen05_has_option(inst, TCGEN05_16X128B_OPTION)) {
    lanes = 16;
    bits = 128;
  } else if (tcgen05_has_option(inst, TCGEN05_16X64B_OPTION)) {
    lanes = 16;
    bits = 64;
  } else if (tcgen05_has_option(inst, TCGEN05_16X32BX2_OPTION)) {
    assert(inst->get_num_operands() == 3);
    const operand_info &offset =
        inst->operand_lookup(inst->get_opcode() == TCGEN05_ST_OP ? 1 : 2);
    assert(offset.is_literal());
    const unsigned split =
        thread->get_operand_value(offset, offset, B32_TYPE, thread, 1).u32;
    if (lane >= 16)
      base += split;
    lane %= 16;
    lanes = 16;
  } else {
    assert(tcgen05_has_option(inst, TCGEN05_32X32B_OPTION));
  }
  return tcgen05_tmem_register_address(
      base, lane, lanes, bits, reg,
      tcgen05_has_option(inst, TCGEN05_PACK_16B_OPTION) ||
          tcgen05_has_option(inst, TCGEN05_UNPACK_16B_OPTION));
}

ptx_reg_t tcgen05_read_operand(const ptx_instruction *pI,
                               ptx_thread_info *thread,
                               unsigned operand_index) {
  assert(operand_index < pI->get_num_operands());
  const operand_info &op = pI->operand_lookup(operand_index);
  return thread->get_operand_value(op, op, B32_TYPE, thread, 1);
}

ptx_reg_t tcgen05_read_u64_operand(const ptx_instruction *pI,
                                   ptx_thread_info *thread,
                                   unsigned operand_index) {
  assert(operand_index < pI->get_num_operands());
  const operand_info &op = pI->operand_lookup(operand_index);
  return thread->get_operand_value(op, op, B64_TYPE, thread, 1);
}

mem_addr_t tcgen05_eval_address(const operand_info &op,
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

std::vector<uint32_t> tcgen05_read_vector_words(const operand_info &src,
                                                ptx_thread_info *thread) {
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

  assert(
      !tcgen05_has_option(pI, RED_OPTION) &&
      "TMEM packed/reduction loads are not qualified by this implementation");
  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &addr = pI->operand_lookup(1);
  assert(dst.is_vector());

  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  uint32_t raw_address = tcgen05_eval_address(addr, thread);
  uint32_t address = tcgen05_register_address(pI, raw_address, thread, 0);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "ld line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "raw=%u addr=%u n=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, raw_address, address,
                      dst.get_vect_nelem());
  }
  std::vector<uint32_t> values(dst.get_vect_nelem());
  const bool packed = tcgen05_has_option(pI, TCGEN05_PACK_16B_OPTION);
  for (unsigned i = 0; i < values.size(); ++i) {
    const auto words = manager.read_words(
        scope, tcgen05_register_address(pI, raw_address, thread, i),
        packed ? 2 : 1);
    values[i] = packed ? (words[0] & 0xffffu) | (words[1] << 16) : words[0];
  }
  tcgen05_write_vector_words(dst, thread, values);
}

void handle_tcgen05_st_inst(const ptx_instruction *pI,
                            ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(pI);
  assert(pI->get_num_operands() >= 2);

  const operand_info &addr = pI->operand_lookup(0);
  const operand_info &src = pI->operand_lookup(
      tcgen05_has_option(pI, TCGEN05_16X32BX2_OPTION) ? 2 : 1);
  assert(src.is_vector());

  flash_gpgpu_sim::tcgen05_tmem_manager_t &manager =
      thread->get_gpu()->get_tcgen05_tmem_manager();
  flash_gpgpu_sim::tcgen05_tmem_scope_t scope = tcgen05_tmem_scope(pI, thread);
  uint32_t raw_address = tcgen05_eval_address(addr, thread);
  uint32_t address = tcgen05_register_address(pI, raw_address, thread, 0);
  if (GPTRACE_CORE(TCGEN05, thread->get_hw_sid())) {
    GPPRINTF_GPU_CORE(thread->get_gpu(), thread->get_hw_sid(), TCGEN05,
                      "st line=%u tid=%u lane=%u scope=(%u,%u,%u) "
                      "raw=%u addr=%u n=%u\n",
                      pI->source_line(), thread->get_tid().x,
                      thread->get_laneid(), scope.sm_id, scope.cta_id,
                      scope.cta_group, raw_address, address,
                      src.get_vect_nelem());
  }
  const auto values = tcgen05_read_vector_words(src, thread);
  const bool unpacked = tcgen05_has_option(pI, TCGEN05_UNPACK_16B_OPTION);
  for (unsigned i = 0; i < values.size(); ++i)
    manager.write_words(
        scope, tcgen05_register_address(pI, raw_address, thread, i),
        unpacked ? std::vector<uint32_t>{values[i] & 0xffffu, values[i] >> 16}
                 : std::vector<uint32_t>{values[i]});
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
  assert(pI->get_num_operands() == 2);
  // Preserve the public PTX-only 4x256b path until it is qualified together
  // with its MMA consumer. It is not part of the new dual-frontend layouts.
  if (tcgen05_has_option(pI, TCGEN05_4X256B_OPTION)) {
    const auto desc = tcgen05_decode_shared_descriptor(
        tcgen05_read_u64_operand(pI, thread, 1).u64);
    assert(!desc.leading_dimension_absolute);
    std::vector<uint32_t> words(32);
    for (unsigned i = 0; i < words.size(); ++i)
      thread->m_shared_mem->read(desc.start_address + i * 4, 4, &words[i]);
    const auto base = tcgen05_eval_address(pI->operand_lookup(0), thread);
    thread->get_gpu()->get_tcgen05_tmem_manager().write_matrix_words(
        tcgen05_tmem_scope(pI, thread), base, words, 4, 8);
    return;
  }
  tcgen05_copy_shape shape;
  if (tcgen05_has_option(pI, TCGEN05_128X256B_OPTION))
    shape = tcgen05_copy_shape::rows128_bits256;
  else if (tcgen05_has_option(pI, TCGEN05_128X128B_OPTION))
    shape = tcgen05_copy_shape::rows128_bits128;
  else if (tcgen05_has_option(pI, TCGEN05_64X128B_OPTION) &&
           tcgen05_has_option(pI, TCGEN05_WARPX2_02_13_OPTION))
    shape = tcgen05_copy_shape::warpx2_02_13;
  else if (tcgen05_has_option(pI, TCGEN05_64X128B_OPTION) &&
           tcgen05_has_option(pI, TCGEN05_WARPX2_01_23_OPTION))
    shape = tcgen05_copy_shape::warpx2_01_23;
  else {
    assert(tcgen05_has_option(pI, TCGEN05_32X128B_OPTION) &&
           tcgen05_has_option(pI, TCGEN05_WARPX4_OPTION));
    shape = tcgen05_copy_shape::warpx4;
  }
  std::vector<uint32_t> addresses;
  const uint64_t descriptor = tcgen05_read_u64_operand(pI, thread, 1).u64;
  const bool valid = tcgen05_copy_addresses(descriptor, shape, addresses);
  assert(valid && "unsupported TMEM copy descriptor");
  std::vector<uint32_t> words(addresses.size());
  for (unsigned i = 0; i < words.size(); ++i)
    thread->m_shared_mem->read(addresses[i], 4, &words[i]);
  const unsigned base = tcgen05_eval_address(pI->operand_lookup(0), thread);
  assert((base >> 16) == 0 && base % (words.size() / 128) == 0);
  thread->get_gpu()->get_tcgen05_tmem_manager().write_matrix_words(
      tcgen05_tmem_scope(pI, thread), base, words, 128, words.size() / 128);
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
  if (ptx_inst)
    return ptx_inst->get_opcode();
  using info = inst_t::tcgen05_static_info_t;
  switch (inst->tcgen05_info.operation) {
  case info::MMA:
    return TCGEN05_MMA_OP;
  case info::COPY:
    return TCGEN05_CP_OP;
  case info::SHIFT:
    return TCGEN05_SHIFT_OP;
  case info::LOAD:
    return TCGEN05_LD_OP;
  case info::STORE:
    return TCGEN05_ST_OP;
  case info::COMMIT:
    return TCGEN05_COMMIT_OP;
  case info::WAIT_LD:
  case info::WAIT_ST:
    return TCGEN05_WAIT_OP;
  default:
    return -1;
  }
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

tcgen05_op_t tcgen05_timing_op(const warp_inst_t *inst,
                               const warp_inst_t *dynamic_inst,
                               const shader_core_config *config) {
  tcgen05_op_t op;
  const auto *ptx = dynamic_cast<const ptx_instruction *>(inst);
  const int opcode = tcgen05_opcode(inst);
  op.kind = tcgen05_timing_kind(opcode);
  if (opcode == TCGEN05_MMA_OP) {
    assert(dynamic_inst != NULL);
    op.work = dynamic_inst->get_tcgen05_dyn_info(0).mma_work;
    assert(op.work != 0);
  } else if (opcode == TCGEN05_CP_OP) {
    const unsigned index =
        ptx ? tcgen05_cp_shape_index(ptx) : inst->tcgen05_info.timing_index;
    assert(index < 5);
    op.completion_latency = config->tcgen05_cp_completion_latency[index];
    op.initiation_interval = config->tcgen05_cp_initiation_interval[index];
  } else if (opcode == TCGEN05_LD_OP || opcode == TCGEN05_ST_OP) {
    unsigned index = inst->tcgen05_info.timing_index;
    if (ptx) {
      const unsigned operand_index =
          opcode == TCGEN05_LD_OP                            ? 0
          : tcgen05_has_option(ptx, TCGEN05_16X32BX2_OPTION) ? 2
                                                             : 1;
      const auto &vector_operand = ptx->operand_lookup(operand_index);
      assert(vector_operand.is_vector());
      index = tcgen05_vector_width_index(vector_operand.get_vect_nelem());
    }
    assert(index < 8);
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
