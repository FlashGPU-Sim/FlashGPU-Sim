#include "internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result execute_atomg_add(const instruction &inst, warp_state &state,
                              execution_context &context) {
  const bool wide = inst.opcode.find(".64.") != std::string::npos;
  const size_t bytes = wide ? 8 : 4;
  const bool reduction = inst.opcode.rfind("REDG.", 0) == 0;
  if (!has_typed_operands(inst, reduction ? 2 : 4))
    return unsupported(inst,
                       "expected atomic add or discarded-result add operands");
  const operand *predicate =
      reduction ? nullptr : get_operand(inst, 0, operand_kind::kPredicate);
  const operand *destination =
      reduction ? nullptr : get_operand(inst, 1, operand_kind::kRegister);
  const operand *address =
      get_operand(inst, reduction ? 0 : 2, operand_kind::kDescriptorAddress);
  const operand *source =
      get_operand(inst, reduction ? 1 : 3, operand_kind::kRegister);
  if ((!reduction && (!predicate || predicate->index != kTruePredicate ||
                      predicate->negated || !destination)) ||
      !address || !source || address->address_register >= kVectorRegisters - 1)
    return unsupported(inst, "unsupported ATOMG/REDG.ADD operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  (void)state.read_uniform_register(address->descriptor_register);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint64_t location =
        read_register_pair(state, lane, address->address_register) +
        static_cast<uint64_t>(address->address_offset);
    if ((location % bytes) != 0)
      return unsupported(inst,
                         "global reduction address is not naturally aligned");
    const uint64_t value = wide ? read_register_pair(state, lane, source->index)
                                : state.read_register(lane, source->index);
    uint64_t old_value;
    if (!apply_integer_rmw(
            value,
            inst.opcode.find(".XOR.") != std::string::npos
                ? integer_rmw_kind::bit_xor
                : integer_rmw_kind::add,
            [&](uint64_t &bits) {
              return context.memory->read(memory_space::kGlobal, location,
                                          &bits, bytes);
            },
            [&](uint64_t bits) {
              return context.memory->write(memory_space::kGlobal, location,
                                           &bits, bytes);
            },
            old_value))
      return memory_fault(inst, memory_space::kGlobal, location, bytes);
    context.record_memory_access(lane, memory_space::kGlobal, location, bytes,
                                 false);
    if (destination)
      state.write_register(lane, destination->index, old_value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_uniform_reduce(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  const bool wide = inst.opcode.find(".64.") != std::string::npos;
  const size_t bytes = wide ? 8 : 4;
  if (!has_typed_operands(inst, 2) || !context.memory ||
      inst.operands[0].kind != operand_kind::kMemoryAddress ||
      inst.operands[0].has_prefix ||
      inst.operands[0].address_groups.size() != 1 ||
      inst.operands[1].kind != operand_kind::kUniformRegister)
    return unsupported(inst, "expected uniform address and addend");
  operand address = inst.operands[0];
  for (auto &term : address.address_groups[0].terms) {
    if (term.kind == address_term_kind::kUniformRegister)
      term.wide = true;
    else if (term.kind != address_term_kind::kImmediate)
      return unsupported(inst, "nonuniform reduction address");
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t at = 0;
    if (!evaluate_address(address, state, lane, at) || at % bytes)
      return unsupported(inst, "unaligned uniform reduction");
    const uint64_t value =
        wide ? read_uniform_register_pair(state, inst.operands[1].index)
             : state.read_uniform_register(inst.operands[1].index);
    uint64_t old;
    if (!apply_integer_rmw(
            value, integer_rmw_kind::add,
            [&](uint64_t &bits) {
              return context.memory->read(memory_space::kGlobal, at, &bits,
                                          bytes);
            },
            [&](uint64_t bits) {
              return context.memory->write(memory_space::kGlobal, at, &bits,
                                           bytes);
            },
            old))
      return memory_fault(inst, memory_space::kGlobal, at, bytes);
    context.record_memory_access(lane, memory_space::kGlobal, at, bytes, false);
    break;
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}
} // namespace
void register_global_reduce_semantics(frontend &target) {
  target.register_semantics("ATOMG.E.ADD.STRONG.GPU", execute_atomg_add);
  for (const char *op :
       {"REDG.E.ADD.STRONG.GPU", "REDG.E.ADD.S32.STRONG.GPU",
        "REDG.E.ADD.64.STRONG.GPU", "REDG.E.XOR.64.STRONG.GPU"})
    target.register_semantics(op, execute_atomg_add);
  for (const char *scope : {"GPU", "SYS"})
    for (const char *type : {"", "S32.", "64."})
      target.register_semantics(std::string("UREDGR.E.ADD.NA.") + type +
                                    "STRONG." + scope,
                                execute_uniform_reduce);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
