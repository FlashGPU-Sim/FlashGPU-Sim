#include "../../../trace.h"
#include "../../gpu-sim.h"
#include "operands.h"
#include "ptx_internal.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
namespace flash_gpgpu_sim {
static tcgen05_ws_collector_control
ws_collector_control(const ptx_instruction *inst) {
  constexpr int first = TCGEN05_COLLECTOR_B0_DISCARD_OPTION;
  constexpr int last = TCGEN05_COLLECTOR_B3_LASTUSE_OPTION;
  static_assert(last - first == 15, "WS parser options must stay contiguous");
  static const tcgen05_collector_op operations[] = {
      tcgen05_collector_op::discard, tcgen05_collector_op::fill,
      tcgen05_collector_op::use, tcgen05_collector_op::lastuse};
  tcgen05_ws_collector_control result;
  unsigned count = 0;
  for (int option : inst->get_options()) {
    if (option >= first && option <= last) {
      ++count;
      const unsigned index = option - first;
      result = {index / 4, operations[index % 4]};
    }
  }
  assert(count <= 1 && "Only one WS collector modifier is permitted");
  assert((!count || tcgen05_has_option(inst, TCGEN05_WS_OPTION)) &&
         "Numbered B collectors require WS MMA");
  return result;
}

static bool tcgen05_read_enable_input_d(const operand_info &op,
                                        ptx_thread_info *thread) {
  if (op.is_reg() && op.get_symbol() &&
      op.get_symbol()->type()->get_key().scalar_type() == PRED_TYPE) {
    ptx_reg_t predicate =
        thread->get_operand_value(op, op, PRED_TYPE, thread, 1);
    return ((predicate.pred & 0x1) == 0) != op.is_neg_pred();
  }

  ptx_reg_t value = thread->get_operand_value(op, op, B32_TYPE, thread, 1);
  return value.u32 != 0;
}

static void tcgen05_check_disable_output_lane_zero(const operand_info &op,
                                                   ptx_thread_info *thread,
                                                   unsigned group) {
  assert(op.is_vector());
  assert(op.get_vect_nelem() == 4 * group &&
         "Disable-output-lane vector size must match the CTA group");

  std::vector<uint32_t> lanes = tcgen05_read_vector_words(op, thread);
  for (unsigned i = 0; i < lanes.size(); ++i) {
    assert(lanes[i] == 0 &&
           "TCGen05 MMA disable-output-lane masks are not implemented");
  }
}

static void tcgen05_mma_mixed_ptx(const ptx_instruction *inst,
                                  ptx_thread_info *thread) {
  assert(inst->get_num_operands() == 7);
  assert(tcgen05_has_option(inst, TCGEN05_BLOCK_SCALE_OPTION));
  assert(!tcgen05_has_option(inst, TCGEN05_BLOCK16_OPTION) &&
         !tcgen05_has_option(inst, TCGEN05_SCALE_VEC_2X_OPTION) &&
         !tcgen05_has_option(inst, TCGEN05_SCALE_VEC_4X_OPTION));
  assert(!inst->operand_lookup(1).is_memory_operand() &&
         "MXF8F6F4 TMEM-A is not qualified yet");
  const unsigned group = tcgen05_cta_group(inst);
  const auto targets = tcgen05_ptx_targets(inst, thread);
  const auto desc = tcgen05_decode_mxf8f6f4_mma_descriptor(
      tcgen05_read_operand(inst, thread, 3).u32, group);
  assert(desc.k == 32 && "SM107 mixed MMA requires architecture qualification");
  const unsigned local_m = desc.m / group, local_n = desc.n / group;
  const bool folded = group == 2 && desc.m == 128;
  auto scope = [&](unsigned rank) {
    return tcgen05_tmem_scope_t{targets[rank].sm_id, targets[rank].cta_slot,
                                group};
  };
  auto &manager = thread->get_gpu()->get_tcgen05_tmem_manager();
  const uint32_t destination =
      tcgen05_eval_address(inst->operand_lookup(0), thread);
  const uint32_t sfa = tcgen05_eval_address(inst->operand_lookup(4), thread);
  const uint32_t sfb = tcgen05_eval_address(inst->operand_lookup(5), thread);
  const bool accumulate =
      tcgen05_read_enable_input_d(inst->operand_lookup(6), thread);
  auto read_input = [&](unsigned operand, unsigned rows, uint8_t format,
                        unsigned rank) {
    const auto sd = tcgen05_decode_mma_shared_descriptor(
        tcgen05_read_u64_operand(inst, thread, operand).u64);
    std::vector<uint8_t> values(rows * desc.k);
    const bool ok = tcgen05_gather_mixed_partition(
        sd, rows, desc.k, format, false, 0, values,
        [&](unsigned, uint32_t address, void *dst, unsigned bytes) {
          targets[rank].smem->read(address, bytes, dst);
          return true;
        });
    assert(ok);
    return values;
  };
  std::vector<uint8_t> a, b, sa;
  std::vector<uint32_t> input;
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto ap = read_input(1, local_m, desc.a_type, rank);
    const auto bp = read_input(2, local_n, desc.b_type, rank);
    const auto sp = manager.read_mxf4_scale_matrix(scope(rank), sfa, local_m, 1,
                                                   desc.a_scale_factor_id);
    a.insert(a.end(), ap.begin(), ap.end());
    b.insert(b.end(), bp.begin(), bp.end());
    sa.insert(sa.end(), sp.begin(), sp.end());
    if (accumulate) {
      const auto dp = manager.read_accumulator(scope(rank), destination,
                                               local_m, desc.n, folded);
      input.insert(input.end(), dp.begin(), dp.end());
    }
  }
  const auto sb = manager.read_mxf4_scale_matrix(scope(0), sfb, desc.n, 1,
                                                 desc.b_scale_factor_id);
  const auto output =
      tcgen05_mma_mxf8f6f4_compute_words(desc, a, b, sa, sb, input, accumulate);
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto begin = output.begin() + rank * local_m * desc.n;
    manager.write_accumulator(
        scope(rank), destination,
        std::vector<uint32_t>(begin, begin + local_m * desc.n), local_m, desc.n,
        folded);
  }
  inst_t::tcgen05_dyn_info_t timing;
  timing.mma_work = 2ull * desc.m * desc.n * desc.k;
  const_cast<ptx_instruction *>(inst)->set_tcgen05_dyn_info(
      thread->get_laneid(), timing);
}

// Compact FP4 block32 uses the same descriptor, scale layout, addressing and
// numerical backend as native UTCOMMA. SM107 and CTA2 stay separately gated.
static void tcgen05_mma_mxf4_ptx(const ptx_instruction *inst,
                                 ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(inst);
  assert(inst->get_num_operands() == 7 &&
         tcgen05_has_option(inst, TCGEN05_BLOCK_SCALE_OPTION) &&
         tcgen05_has_option(inst, TCGEN05_BLOCK32_OPTION));
  assert(!tcgen05_has_option(inst, TCGEN05_BLOCK16_OPTION) &&
         !tcgen05_has_option(inst, TCGEN05_SCALE_VEC_2X_OPTION) &&
         !tcgen05_has_option(inst, TCGEN05_SCALE_VEC_4X_OPTION));
  assert(!inst->operand_lookup(1).is_memory_operand());
  const auto desc = tcgen05_decode_mxf4_mma_descriptor(
      tcgen05_read_operand(inst, thread, 3).u32, 1);
  const auto scope = tcgen05_tmem_scope(inst, thread);
  auto &manager = thread->get_gpu()->get_tcgen05_tmem_manager();
  const uint32_t dst = tcgen05_eval_address(inst->operand_lookup(0), thread);
  const uint32_t sa = tcgen05_eval_address(inst->operand_lookup(4), thread);
  const uint32_t sb = tcgen05_eval_address(inst->operand_lookup(5), thread);
  assert(!(dst >> 16) && !(sa >> 16) && !(sb >> 16));
  const bool accumulate =
      tcgen05_read_enable_input_d(inst->operand_lookup(6), thread);
  auto read = [&](unsigned operand, unsigned rows) {
    const auto sd = tcgen05_decode_mma_shared_descriptor(
        tcgen05_read_u64_operand(inst, thread, operand).u64);
    std::vector<uint8_t> values;
    const bool ok = tcgen05_gather_fp4(
        sd, rows, desc.k, values,
        [&](unsigned, uint32_t address, void *dst, unsigned bytes) {
          thread->m_shared_mem->read(address, bytes, dst);
          return true;
        });
    assert(ok);
    return values;
  };
  const auto a = read(1, desc.m), b = read(2, desc.n);
  const auto scales_a = manager.read_mxf4_scale_matrix(
      scope, sa, desc.m, desc.k / 32, desc.a_scale_factor_id);
  const auto scales_b = manager.read_mxf4_scale_matrix(
      scope, sb, desc.n, desc.k / 32, desc.b_scale_factor_id);
  const auto input = accumulate
                         ? manager.read_matrix_words(scope, dst, desc.m, desc.n)
                         : std::vector<uint32_t>{};
  const auto output = tcgen05_mma_mxf4_compute_words(
      desc, a, b, scales_a, scales_b, 32, input, accumulate);
  manager.write_matrix_words(scope, dst, output, desc.m, desc.n);
  inst_t::tcgen05_dyn_info_t timing;
  timing.mma_work = 2ull * desc.m * desc.n * desc.k;
  const_cast<ptx_instruction *>(inst)->set_tcgen05_dyn_info(
      thread->get_laneid(), timing);
}

// PTX adapter for the same qualified TF32 TS forms as native UTCHMMA.
static void tcgen05_mma_tf32_ptx(const ptx_instruction *inst,
                                 ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(inst);
  assert(inst->operand_lookup(1).is_memory_operand() &&
         "TF32 shared-A form has not been qualified");
  const uint32_t idesc = tcgen05_read_operand(inst, thread, 3).u32;
  const unsigned m = ((idesc >> 24) & 31) * 16;
  const unsigned n = ((idesc >> 17) & 63) * 8;
  assert(!(idesc & 0xe080004fu) && ((idesc >> 4) & 3) == 1 &&
         ((idesc >> 7) & 7) == 2 && ((idesc >> 10) & 7) == 2 &&
         !(idesc & (3u << 15)) && (m == 64 || m == 128) && n >= 8 && n <= 256 &&
         n % 8 == 0);
  unsigned enable = 4;
  if (inst->operand_lookup(4).is_vector()) {
    assert(inst->get_num_operands() == 6);
    tcgen05_check_disable_output_lane_zero(inst->operand_lookup(4), thread, 1);
    enable = 5;
  } else
    assert(inst->get_num_operands() == 5);
  const bool accumulate =
      tcgen05_read_enable_input_d(inst->operand_lookup(enable), thread);
  const auto scope = tcgen05_tmem_scope(inst, thread);
  auto &tm = thread->get_gpu()->get_tcgen05_tmem_manager();
  const uint32_t dst = tcgen05_eval_address(inst->operand_lookup(0), thread);
  const uint32_t aptr = tcgen05_eval_address(inst->operand_lookup(1), thread);
  assert(!(aptr >> 16) && !(aptr & 7) && !(dst >> 16));
  const auto bd = tcgen05_decode_mma_shared_descriptor(
      tcgen05_read_u64_operand(inst, thread, 2).u64);
  assert(!bd.base_offset);
  std::vector<uint32_t> a, b(n * 8), input;
  for (unsigned row = 0; row < m; ++row) {
    const unsigned physical = m == 64 ? (row / 16) * 32 + row % 16 : row;
    const unsigned offset = physical << 16;
    const auto values = tm.read_words(scope, aptr + offset, 8);
    a.insert(a.end(), values.begin(), values.end());
    if (accumulate) {
      const auto values = tm.read_words(scope, dst + offset, n);
      input.insert(input.end(), values.begin(), values.end());
    }
  }
  for (unsigned col = 0; col < n; ++col)
    for (unsigned k = 0; k < 8; ++k) {
      const auto address =
          tcgen05_shared_k_major_packed_byte_address(bd, col, k * 4, 32);
      thread->m_shared_mem->read(address, 4, &b[col * 8 + k]);
    }
  const auto out = tcgen05_mma_tf32_compute_words(
      m, n, a, b, input, accumulate, idesc & (1u << 13), idesc & (1u << 14));
  for (unsigned row = 0; row < m; ++row) {
    const unsigned physical = m == 64 ? (row / 16) * 32 + row % 16 : row;
    tm.write_words(scope, dst + (physical << 16),
                   std::vector<uint32_t>(out.begin() + row * n,
                                         out.begin() + (row + 1) * n));
  }
  inst_t::tcgen05_dyn_info_t timing;
  timing.mma_work = 2ull * m * n * 8;
  const_cast<ptx_instruction *>(inst)->set_tcgen05_dyn_info(
      thread->get_laneid(), timing);
}

static void tcgen05_mma_fp8_ptx(const ptx_instruction *inst,
                                ptx_thread_info *thread) {
  tcgen05_assert_cta_group1(inst);
  assert(inst->get_num_operands() == 5);
  assert(!inst->operand_lookup(1).is_memory_operand());
  const auto desc = tcgen05_decode_fp8_mma_descriptor(
      tcgen05_read_operand(inst, thread, 3).u32);
  const auto scope = tcgen05_tmem_scope(inst, thread);
  auto &tm = thread->get_gpu()->get_tcgen05_tmem_manager();
  const uint32_t destination =
      tcgen05_eval_address(inst->operand_lookup(0), thread);
  const bool accumulate =
      tcgen05_read_enable_input_d(inst->operand_lookup(4), thread);
  auto gather = [&](unsigned operand, unsigned rows) {
    const auto sd = tcgen05_decode_mma_shared_descriptor(
        tcgen05_read_u64_operand(inst, thread, operand).u64);
    std::vector<uint8_t> values;
    const bool ok = tcgen05_gather_shared(
        sd, rows, 32, 1, false, values,
        [&](unsigned, uint32_t address, void *dst, unsigned bytes) {
          thread->m_shared_mem->read(address, bytes, dst);
          return true;
        });
    assert(ok);
    return values;
  };
  const auto a = gather(1, desc.m), b = gather(2, desc.n);
  const auto input =
      accumulate ? tm.read_matrix_words(scope, destination, desc.m, desc.n)
                 : std::vector<uint32_t>{};
  const auto output = tcgen05_mma_mxf8f6f4_compute_words(
      desc, a, b, std::vector<uint8_t>(desc.m, 127),
      std::vector<uint8_t>(desc.n, 127), input, accumulate);
  tm.write_matrix_words(scope, destination, output, desc.m, desc.n);
  inst_t::tcgen05_dyn_info_t timing;
  timing.mma_work = 2ull * desc.m * desc.n * desc.k;
  const_cast<ptx_instruction *>(inst)->set_tcgen05_dyn_info(
      thread->get_laneid(), timing);
}

void handle_tcgen05_mma_inst(const ptx_instruction *pI,
                             ptx_thread_info *thread) {
  if (!tcgen05_is_warp_leader(thread))
    return;
  assert(pI->get_num_operands() >= 5);
  const bool ws = tcgen05_has_option(pI, TCGEN05_WS_OPTION);
  const auto collector = ws_collector_control(pI);
  assert((!ws || tcgen05_has_option(pI, TCGEN05_KIND_F16_OPTION)) &&
         "Only F16/BF16 WS MMA has been qualified");
  if (ws)
    assert(pI->get_num_operands() == 5 &&
           "WS zero-column masks are not qualified");
  if (tcgen05_has_option(pI, TCGEN05_KIND_F8F6F4_OPTION)) {
    tcgen05_mma_fp8_ptx(pI, thread);
    return;
  }
  if (tcgen05_has_option(pI, TCGEN05_KIND_MXF8F6F4_OPTION)) {
    tcgen05_mma_mixed_ptx(pI, thread);
    return;
  }
  if (tcgen05_has_option(pI, TCGEN05_KIND_TF32_OPTION)) {
    tcgen05_mma_tf32_ptx(pI, thread);
    return;
  }
  if (tcgen05_has_option(pI, TCGEN05_KIND_MXF4_OPTION)) {
    tcgen05_mma_mxf4_ptx(pI, thread);
    return;
  }
  assert(tcgen05_has_option(pI, TCGEN05_KIND_F16_OPTION));
  const unsigned group = tcgen05_cta_group(pI);
  const auto targets = tcgen05_ptx_targets(pI, thread);

  const operand_info &d_tmem = pI->operand_lookup(0);
  const operand_info &a_desc_op = pI->operand_lookup(1);
  unsigned enable_input_d_operand = 4;
  if (pI->operand_lookup(4).is_vector()) {
    assert(pI->get_num_operands() == 6 && "MMA scale-input-d is not qualified");
    tcgen05_check_disable_output_lane_zero(pI->operand_lookup(4), thread,
                                           group);
    enable_input_d_operand = 5;
  } else {
    assert(pI->get_num_operands() == 5 && "MMA scale-input-d is not qualified");
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
          tcgen05_read_operand(pI, thread, 3).u32, tcgen05_cta_group(pI), ws);
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
      flash_gpgpu_sim::tcgen05_decode_mma_shared_descriptor(
          b_desc_value, mma_desc.transpose_b);
  bool enable_input_d = tcgen05_read_enable_input_d(enable_input_d_op, thread);

  const unsigned local_m = mma_desc.m / group;
  const bool folded = (group == 2 && mma_desc.m == 128) || ws;
  assert(tcgen05_mma_tmem_alignment_supported(local_m, folded, d_address,
                                              a_desc_op.is_memory_operand(),
                                              a_tmem_address));
  if (a_desc_op.is_memory_operand())
    assert(!mma_desc.transpose_a && (!folded || mma_desc.n % 32 == 0));
  auto target_scope = [&](unsigned rank) {
    return tcgen05_tmem_scope_t{targets[rank].sm_id, targets[rank].cta_slot,
                                group};
  };
  auto gather_shared = [&](const tcgen05_shared_descriptor_t &sd, unsigned rows,
                           bool transpose) {
    std::vector<uint16_t> values;
    const bool ok = tcgen05_gather_shared(
        sd, rows, mma_desc.k, group, transpose, values,
        [&](unsigned rank, uint32_t address, void *dst, unsigned bytes) {
          targets[rank].smem->read(address, bytes, dst);
          return true;
        });
    assert(ok);
    return values;
  };
  std::vector<uint16_t> a_values;
  if (!a_desc_op.is_memory_operand()) {
    const auto a_desc = tcgen05_decode_mma_shared_descriptor(
        a_desc_value, mma_desc.transpose_a);
    a_values = gather_shared(a_desc, mma_desc.m, mma_desc.transpose_a);
  }
  std::vector<uint16_t> b_values;
  if (!ws || !collector.reuses())
    b_values = gather_shared(b_desc, mma_desc.n, mma_desc.transpose_b);
  if (ws) {
    const bool valid = manager.ws_half_b(
        scope,
        thread->get_tid().x +
            thread->get_ntid().x * (thread->get_tid().y +
                                    thread->get_ntid().y * thread->get_tid().z),
        collector,
        {mma_desc.n, mma_desc.k, mma_desc.b_type, mma_desc.transpose_b},
        b_values);
    assert(valid && "WS B reuse requires a compatible prior fill");
  }
  std::vector<uint32_t> input_d;
  if (enable_input_d) {
    for (unsigned rank = 0; rank < group; ++rank) {
      const auto part = manager.read_accumulator(target_scope(rank), d_address,
                                                 local_m, mma_desc.n, folded);
      input_d.insert(input_d.end(), part.begin(), part.end());
    }
  }
  std::vector<uint32_t> output;
  if (a_desc_op.is_memory_operand()) {
    std::vector<std::vector<uint16_t>> physical_a(group);
    for (unsigned rank = 0; rank < group; ++rank)
      physical_a[rank] = manager.read_matrix_packed_u16(
          target_scope(rank), a_tmem_address & 0xffff, 128, mma_desc.k);
    output = tcgen05_mma_f16_tmem_a_compute_words(
        mma_desc, physical_a, b_values, input_d, enable_input_d, folded,
        a_tmem_address >> 16);
  } else {
    output = tcgen05_mma_f16_compute_words(mma_desc, a_values, b_values,
                                           input_d, enable_input_d);
  }
  for (unsigned rank = 0; rank < group; ++rank) {
    const auto begin = output.begin() + rank * local_m * mma_desc.n;
    manager.write_accumulator(
        target_scope(rank), d_address,
        std::vector<uint32_t>(begin, begin + local_m * mma_desc.n), local_m,
        mma_desc.n, folded);
  }
}

} // namespace flash_gpgpu_sim
