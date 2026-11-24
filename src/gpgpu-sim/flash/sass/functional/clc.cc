#include "internal.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result execute_clc(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (!context.cta || !context.memory || !context.clc_service_available)
    return unsupported(
        inst, "CLC requires CTA shared memory and a launch-queue service");
  const bool joined = inst.operands.size() == 1;
  if ((!joined && inst.operands.size() != 2))
    return unsupported(inst, "CLC requires response and mbarrier addresses");
  for (const auto &operand : inst.operands)
    if (operand.kind != operand_kind::kMemoryAddress || operand.has_prefix ||
        operand.address_groups.size() != (joined ? 2u : 1u))
      return unsupported(inst, "CLC requires uniform shared addresses");
  uint32_t addresses[2];
  for (unsigned i = 0; i < 2; ++i) {
    const auto &terms =
        inst.operands[joined ? 0 : i].address_groups[joined ? i : 0].terms;
    if (terms.size() != 1 ||
        terms[0].kind != address_term_kind::kUniformRegister || terms[0].wide ||
        terms[0].negated || terms[0].index >= kZeroUniformRegister)
      return unsupported(inst, "unqualified CLC uniform address");
    const uint32_t raw = state.read_uniform_register(terms[0].index);
    if ((raw >> 24) != context.cga_cta_id)
      return unsupported(inst, "CLC address must belong to the issuing CTA");
    addresses[i] = raw & 0xffffff;
  }
  const auto imported = context.imported_mbarriers.size();
  if (addresses[0] % 16 || addresses[1] % 8 ||
      !ensure_mbarrier_imported(addresses[1], context))
    return unsupported(inst,
                       "CLC requires aligned response and initialized barrier");
  for (size_t i = imported; i < context.imported_mbarriers.size(); ++i)
    context.imported_mbarriers[i].address |= context.cga_cta_id << 24;
  context.clc_effect.valid = true;
  context.clc_effect.broadcast = inst.opcode == "UGETNEXTWORKID.BROADCAST";
  context.clc_effect.response_address = addresses[0];
  context.clc_effect.barrier_address = addresses[1];
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace
void register_clc_semantics(frontend &target) {
  target.register_semantics("UGETNEXTWORKID.BROADCAST", execute_clc);
  target.register_semantics("UGETNEXTWORKID.SELFCAST", execute_clc);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
