#include "ptx_global_reduce.h"
#include "../../cuda-sim/ptx_ir.h"
#include "../../cuda-sim/ptx_sim.h"
#include "integer_rmw.h"
#include "panic.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include "ptx.tab.h"
extern void decode_space(memory_space_t &, ptx_thread_info *,
                         const operand_info &, memory_space *&, addr_t &, bool);
namespace flash_gpgpu_sim {
namespace {
unsigned width(const ptx_instruction *inst) {
  const unsigned type = inst->get_type();
  return type == U64_TYPE || type == S64_TYPE || type == B64_TYPE ? 8 : 4;
}
addr_t address(const ptx_instruction *inst, ptx_thread_info *thread,
               memory_space *&mem) {
  const auto &operand = inst->dst();
  addr_t result =
      thread->get_operand_value(operand, operand, U64_TYPE, thread, 1).u64;
  memory_space_t space = inst->get_space();
  if (space == undefined_space)
    space = generic_space;
  decode_space(space, thread, operand, mem, result, false);
  if (space != global_space || result % width(inst))
    panic("red requires aligned global memory");
  return result;
}
// Captured operands outlive the issuing register frame. Async reductions may
// proceed through arithmetic, call/return and EXIT while their request travels.
struct pending_reduce final : atomic_callback_action {
  memory_space *mem;
  addr_t at;
  unsigned bytes;
  uint64_t value;
  integer_rmw_kind kind;
  ptx_thread_info *thread;
  const ptx_instruction *inst;
  void execute() const override {
    uint64_t old;
    apply_integer_rmw(
        value, kind,
        [&](uint64_t &bits) {
          mem->read(at, bytes, &bits);
          return true;
        },
        [&](uint64_t bits) {
          mem->write(at, bytes, &bits, thread, inst);
          return true;
        },
        old);
  }
};

} // namespace
void execute_ptx_global_reduce(const ptx_instruction *inst,
                               ptx_thread_info *thread) {
  const auto &opts = inst->get_options();
  auto has = [&](int o) {
    return std::find(opts.begin(), opts.end(), o) != opts.end();
  };
  auto fail = [&]() {
    panic("unqualified PTX global reduction: " + inst->to_string());
  };
  const unsigned type = inst->get_type();
  const bool add = inst->get_atomic() == ATOMIC_ADD &&
                   (type == U32_TYPE || type == S32_TYPE || type == U64_TYPE ||
                    type == S64_TYPE);
  const bool xorr = inst->get_atomic() == ATOMIC_XOR && type == B64_TYPE;
  if (inst->get_num_operands() != 2 || inst->get_scalar_type().size() != 1 ||
      (!add && !xorr))
    fail();
  for (auto option : opts)
    if (option != ASYNC_OPTION && option != RELEASE_OPTION &&
        option != RELAXED_OPTION && option != GPU_OPTION &&
        option != SYS_OPTION && option != CTA_OPTION &&
        option != CLUSTER_OPTION && option != ATOMIC_ADD &&
        option != ATOMIC_XOR)
      fail();
  if (has(ASYNC_OPTION) &&
      (!add || !has(RELEASE_OPTION) || (!has(GPU_OPTION) && !has(SYS_OPTION)) ||
       has(RELAXED_OPTION)))
    fail();
  if (has(RELEASE_OPTION) && has(RELAXED_OPTION))
    fail();
  if (inst->get_space() != global_space && inst->get_space() != generic_space &&
      inst->get_space() != undefined_space)
    fail();
  memory_space *mem = nullptr;
  thread->m_last_effective_address = address(inst, thread, mem);
  thread->m_last_memory_space = global_space;
  auto pending = std::make_shared<pending_reduce>();
  pending->mem = mem;
  pending->at = thread->m_last_effective_address;
  pending->bytes = width(inst);
  pending->thread = thread;
  pending->inst = inst;
  const auto value =
      thread->get_operand_value(inst->src1(), inst->src1(), type, thread, 1);
  pending->value = pending->bytes == 4 ? value.u32 : value.u64;
  pending->kind = xorr ? integer_rmw_kind::bit_xor : integer_rmw_kind::add;
  thread->m_last_dram_callback.action = std::move(pending);
  thread->m_last_dram_callback.function = nullptr;
  thread->m_last_dram_callback.instruction = inst;
}
} // namespace flash_gpgpu_sim
