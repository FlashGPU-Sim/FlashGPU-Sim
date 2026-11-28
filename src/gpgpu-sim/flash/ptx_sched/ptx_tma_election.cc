#include "../../../cuda-sim/opcodes.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "ptx_scheduler.h"
#include <map>
#include <set>
#include <vector>

namespace {
const symbol *branch_target(const ptx_instruction *inst) {
  if (inst == NULL || inst->is_label() || inst->get_opcode() != BRA_OP)
    return NULL;
  if (inst->get_label() != NULL)
    return inst->get_label();
  if (inst->get_num_operands() == 1 &&
      (inst->operand_lookup(0).is_label() ||
       inst->operand_lookup(0).get_type() == symbolic_t))
    return inst->operand_lookup(0).get_symbol();
  return NULL;
}

bool is_arch_register(const symbol *reg) {
  return reg != NULL && reg->is_reg() && !reg->is_non_arch_reg() &&
         reg->name() != "_";
}

bool intersects(const std::set<const symbol *> &lhs,
                const std::set<const symbol *> &rhs) {
  for (const symbol *value : lhs)
    if (rhs.find(value) != rhs.end())
      return true;
  return false;
}

bool is_direct_control_flow(const ptx_instruction *inst) {
  if (inst == NULL || inst->is_label())
    return false;
  switch (inst->get_opcode()) {
  case BRA_OP:
  case BRX_OP:
  case BREAK_OP:
  case BREAKADDR_OP:
  case CALL_OP:
  case CALLP_OP:
  case RET_OP:
  case RETP_OP:
  case EXIT_OP:
  case TRAP_OP:
    return true;
  default:
    return false;
  }
}

bool is_tma_opcode(const ptx_instruction *inst) {
  return inst != NULL && !inst->is_label() && inst->get_opcode() == TMA_OP;
}

} // namespace
namespace flash_gpgpu_sim {
void mark_explicit_elected_tma_regions(
    std::list<ptx_instruction *> &instructions) {
  std::vector<ptx_instruction *> stream(instructions.begin(),
                                        instructions.end());
  std::map<const symbol *, unsigned> label_positions;
  for (unsigned i = 0; i < stream.size(); ++i) {
    ptx_instruction *inst = stream[i];
    if (inst != NULL) {
      inst->tma_explicit_elected_region = false;
      if (inst->is_label() && inst->get_label() != NULL)
        label_positions[inst->get_label()] = i;
    }
  }

  for (unsigned elect_index = 0; elect_index < stream.size(); ++elect_index) {
    ptx_instruction *elect = stream[elect_index];
    if (elect == NULL || elect->is_label() || elect->has_pred() ||
        elect->get_opcode() != ELECT_OP || elect->get_num_operands() != 2 ||
        !elect->operand_lookup(0).is_vector() ||
        elect->operand_lookup(0).get_vect_nelem() != 2)
      continue;

    std::set<const symbol *> tainted;
    const operand_info &dst = elect->operand_lookup(0);
    const symbol *lane = dst.vec_symbol_or_null(0);
    const symbol *predicate = dst.vec_symbol_or_null(1);
    if (!is_arch_register(predicate))
      continue;
    tainted.insert(predicate);
    if (is_arch_register(lane))
      tainted.insert(lane);

    unsigned branch_index = stream.size();
    unsigned join_index = stream.size();
    for (unsigned i = elect_index + 1; i < stream.size(); ++i) {
      ptx_instruction *inst = stream[i];
      if (inst == NULL || inst->is_label() || inst->get_opcode() == ELECT_OP)
        break;

      std::set<const symbol *> uses;
      std::set<const symbol *> defs;
      detail::collect_ptx_registers(inst, uses, defs);

      if (inst->get_opcode() == BRA_OP && inst->has_pred() &&
          intersects(uses, tainted)) {
        const symbol *target = branch_target(inst);
        std::map<const symbol *, unsigned>::const_iterator found =
            label_positions.find(target);
        if (found != label_positions.end() && found->second > i + 1) {
          branch_index = i;
          join_index = found->second;
        }
        break;
      }
      if (is_direct_control_flow(inst))
        break;

      const bool influenced = intersects(uses, tainted);
      for (const symbol *definition : defs) {
        if (influenced)
          tainted.insert(definition);
        else
          tainted.erase(definition);
      }
    }

    if (branch_index == stream.size())
      continue;

    // A later warp-role block can share the same lexical join while having a
    // dispatch edge that bypasses this election. Keep the elected/dominated
    // prefix, but do not mark the externally reachable suffix. Internal
    // loop/back edges are allowed because their source already lies inside
    // the elected region.
    unsigned safe_end_index = join_index;
    for (unsigned source = 0; source < stream.size(); ++source) {
      const symbol *target = branch_target(stream[source]);
      std::map<const symbol *, unsigned>::const_iterator found =
          label_positions.find(target);
      if (found == label_positions.end())
        continue;
      const unsigned destination = found->second;
      if (destination > branch_index && destination < join_index &&
          (source <= elect_index || source >= join_index) &&
          destination < safe_end_index)
        safe_end_index = destination;
    }

    for (unsigned i = branch_index + 1; i < safe_end_index; ++i) {
      if (!is_tma_opcode(stream[i]))
        continue;
      stream[i]->tma_explicit_elected_region = true;
    }
  }
}

} // namespace flash_gpgpu_sim
