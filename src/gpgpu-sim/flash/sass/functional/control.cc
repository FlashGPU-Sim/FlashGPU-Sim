#include "internal.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.13 plus warp/CTA control operations.

step_result execute_nop(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 0))
    return unsupported(inst, "NOP must not have operands");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_nanosleep(const instruction &inst, warp_state &state,
                              execution_context &) {
  const operand *duration = get_operand(inst, 0, operand_kind::kImmediate);
  const operand *duration_register =
      get_operand(inst, 0, operand_kind::kRegister);
  if (!has_typed_operands(inst, 1) ||
      (duration == nullptr && duration_register == nullptr) ||
      (duration != nullptr && duration->immediate < 0))
    return unsupported(inst,
                       "expected one immediate or register sleep duration");
  // Functional execution has no elapsed-time state. The guarded wait loop
  // still rechecks its mbarrier predicate on subsequent SASS instructions.
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_usetmaxreg(const instruction &inst, warp_state &state,
                               execution_context &) {
  const bool try_allocate = inst.opcode == "USETMAXREG.TRY_ALLOC.CTAPOOL";
  const size_t count_position = try_allocate ? 1 : 0;
  const operand *success =
      try_allocate ? get_operand(inst, 0, operand_kind::kUniformPredicate)
                   : nullptr;
  const operand *count =
      get_operand(inst, count_position, operand_kind::kImmediate);
  if (!has_typed_operands(inst, try_allocate ? 2 : 1) ||
      (try_allocate && success == nullptr) || count == nullptr ||
      count->immediate <= 0 || count->immediate > kVectorRegisters ||
      (inst.has_guard && inst.guard.kind != operand_kind::kUniformPredicate))
    return unsupported(inst, "unsupported USETMAXREG operand form");
  // Functional state already owns the complete physical register namespace;
  // allocation therefore succeeds without introducing a timing resource.
  if (try_allocate && any_lane_executes(inst, state))
    state.write_uniform_predicate(success->index, true);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

bool instruction_target_pc(const instruction &inst, uint64_t &target_pc) {
  for (const instruction_attribute &attribute : inst.attributes) {
    if (attribute.name != "target-pc")
      continue;
    if (attribute.value.empty() ||
        !std::isdigit(static_cast<unsigned char>(attribute.value.front())))
      return false;
    char *end = nullptr;
    errno = 0;
    const auto value = std::strtoull(attribute.value.c_str(), &end, 0);
    if (errno != 0 || value > std::numeric_limits<uint64_t>::max())
      return false;
    target_pc = value;
    return end == attribute.value.c_str() + attribute.value.size() &&
           (target_pc % kInstructionBytes) == 0;
  }
  return false;
}

step_result execute_call_relative(const instruction &inst, warp_state &state,
                                  execution_context &) {
  uint64_t target_pc = 0;
  if (inst.has_guard || inst.operands.size() != 1 ||
      !instruction_target_pc(inst, target_pc))
    return unsupported(inst, "invalid relative call target");
  state.pc = target_pc;
  return {step_status::kAdvanced, {}};
}

step_result execute_return_relative(const instruction &inst, warp_state &state,
                                    execution_context &) {
  if (inst.has_guard || inst.operands.size() != 2 ||
      inst.operands[0].kind != operand_kind::kRegister)
    return unsupported(inst, "invalid relative return operands");
  bool found = false;
  uint32_t target_pc = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0)
      continue;
    const uint32_t lane_target =
        state.read_register(lane, inst.operands[0].index);
    if (!found) {
      found = true;
      target_pc = lane_target;
    } else if (lane_target != target_pc) {
      return unsupported(inst, "relative return target is not warp uniform");
    }
  }
  if (!found || (target_pc % kInstructionBytes) != 0)
    return unsupported(inst, "invalid relative return target");
  state.pc = target_pc;
  return {step_status::kAdvanced, {}};
}

step_result execute_bra(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 1) && !has_typed_operands(inst, 2))
    return unsupported(inst, "expected a branch target and optional predicate");
  const bool has_branch_predicate = inst.operands.size() == 2;
  if (has_branch_predicate && inst.operands[0].kind != operand_kind::kPredicate)
    return unsupported(inst, "invalid branch predicate");
  const operand *target =
      get_operand(inst, has_branch_predicate ? 1 : 0, operand_kind::kImmediate);
  if (target == nullptr || target->immediate < 0 ||
      (target->immediate % kInstructionBytes) != 0)
    return unsupported(inst, "invalid branch target");
  uint32_t taking = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    bool branch_predicate = true;
    if (has_branch_predicate && !read_predicate_operand(inst.operands[0], state,
                                                        lane, branch_predicate))
      return unsupported(inst, "unsupported branch predicate");
    if ((state.active_mask & (1u << lane)) != 0 && branch_predicate &&
        lane_executes(inst, state, lane))
      taking |= 1u << lane;
  }
  if (taking == 0)
    advance(inst, state);
  else if (taking == state.active_mask)
    state.pc = static_cast<uint64_t>(target->immediate);
  else {
    reconvergence_state *frame = nullptr;
    if (!state.reconvergence_stack.empty()) {
      const unsigned barrier = state.reconvergence_stack.back();
      frame = &state.reconvergence[barrier];
      if (!frame->valid)
        return unsupported(inst, "divergent BRA has an invalid BSSY state");
    } else if (!state.warp_sync_stack.empty()) {
      frame = &state.warp_sync_stack.back();
      if (frame->target_pc != static_cast<uint64_t>(target->immediate))
        return unsupported(inst,
                           "nested branch has a different WARPSYNC target");
    } else {
      const instruction *target_instruction = nullptr;
      if (context.kernel_image != nullptr) {
        const auto found = std::lower_bound(
            context.kernel_image->instructions.begin(),
            context.kernel_image->instructions.end(), target->immediate,
            [](const instruction &candidate, int64_t pc) {
              return candidate.pc < static_cast<uint64_t>(pc);
            });
        if (found != context.kernel_image->instructions.end() &&
            found->pc == static_cast<uint64_t>(target->immediate))
          target_instruction = &*found;
      }
      if (target_instruction == nullptr)
        return unsupported(inst, "divergent BRA has an invalid target");
      if (target_instruction->opcode == "WARPSYNC.ALL" ||
          target_instruction->opcode == "WARPSYNC.COLLECTIVE.ALL") {
        reconvergence_state implicit;
        implicit.valid = true;
        implicit.target_pc = static_cast<uint64_t>(target->immediate);
        implicit.participating_mask = state.active_mask;
        state.warp_sync_stack.push_back(std::move(implicit));
        frame = &state.warp_sync_stack.back();
      } else {
        state.independent_paths.push_back(
            {static_cast<uint64_t>(target->immediate), taking});
        state.active_mask &= ~taking;
        advance(inst, state);
        return {step_status::kAdvanced, {}};
      }
    }
    frame->pending_paths.push_back(
        {static_cast<uint64_t>(target->immediate), taking});
    state.active_mask &= ~taking;
    advance(inst, state);
  }
  return {step_status::kAdvanced, {}};
}

step_result execute_bra_uniform(const instruction &inst, warp_state &state,
                                execution_context &) {
  const bool uses_uniform_predicate =
      has_typed_operands(inst, 2) && !inst.has_guard &&
      inst.operands[0].kind == operand_kind::kUniformPredicate;
  const bool uses_two_predicates =
      has_typed_operands(inst, 3) && !inst.has_guard &&
      inst.operands[0].kind == operand_kind::kPredicate &&
      inst.operands[1].kind == operand_kind::kUniformPredicate;
  const bool uses_lane_guard = has_typed_operands(inst, 1) && inst.has_guard;
  const operand *target = get_operand(inst,
                                      uses_two_predicates      ? 2
                                      : uses_uniform_predicate ? 1
                                                               : 0,
                                      operand_kind::kImmediate);
  if ((!uses_uniform_predicate && !uses_lane_guard && !uses_two_predicates) ||
      target == nullptr || target->immediate < 0 ||
      (target->immediate % kInstructionBytes) != 0)
    return unsupported(inst, "invalid BRA.U operand form");
  bool take = false;
  if (uses_two_predicates) {
    bool uniform = false;
    if (!read_predicate_operand(inst.operands[1], state, 0, uniform))
      return unsupported(inst, "unsupported BRA.U uniform predicate");
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if (!(state.active_mask & (1u << lane)))
        continue;
      if (!read_predicate_operand(inst.operands[0], state, lane, take))
        return unsupported(inst, "unsupported BRA.U lane predicate");
      take = take && uniform;
      break;
    }
  } else if (uses_uniform_predicate) {
    if (!read_predicate_operand(inst.operands[0], state, 0, take))
      return unsupported(inst, "unsupported BRA.U predicate");
  } else {
    // BRA.U with an ordinary predicate guard is emitted when ptxas proves
    // that the lane predicate is warp-uniform (for example, a warpgroup-role
    // test derived from TID.X).  Sample the first active lane and apply the
    // decision to the whole currently active path.
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0)
        continue;
      take = lane_executes(inst, state, lane);
      break;
    }
  }
  if (take)
    state.pc = static_cast<uint64_t>(target->immediate);
  else
    advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bra_not_tid0(const instruction &inst, warp_state &state,
                                 execution_context &context) {
  const operand *target = get_operand(inst, 0, operand_kind::kImmediate);
  if (!has_typed_operands(inst, 1) || inst.has_guard || target == nullptr ||
      target->immediate < 0 || target->immediate % kInstructionBytes != 0)
    return unsupported(inst, "unqualified BRA.U.NOT_TID0 operand form");
  // B200 native probe: the whole active warp falls through only when CTA
  // thread zero is on this active path. Other warps, and warp zero after lane
  // zero has exited, branch uniformly. context.warp_id is CTA-local.
  if (context.warp_id != 0 || !(state.active_mask & 1u))
    state.pc = static_cast<uint64_t>(target->immediate);
  else
    advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bra_uniform_vote(const instruction &inst, warp_state &state,
                                     execution_context &) {
  const bool uses_uniform_predicate =
      has_typed_operands(inst, 2) && !inst.has_guard &&
      inst.operands[0].kind == operand_kind::kUniformPredicate;
  const bool uses_lane_vote = has_typed_operands(inst, 1) && inst.has_guard;
  const operand *target = get_operand(inst, uses_uniform_predicate ? 1 : 0,
                                      operand_kind::kImmediate);
  if ((!uses_uniform_predicate && !uses_lane_vote) || target == nullptr ||
      target->immediate < 0 || (target->immediate % kInstructionBytes) != 0)
    return unsupported(inst, "invalid BRA.U.ANY operand form");
  bool take = false;
  if (uses_uniform_predicate) {
    if (!read_predicate_operand(inst.operands[0], state, 0, take))
      return unsupported(inst, "unsupported BRA.U.ANY predicate");
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) != 0 &&
          lane_executes(inst, state, lane)) {
        take = true;
        break;
      }
    }
  }
  if (take)
    state.pc = static_cast<uint64_t>(target->immediate);
  else
    advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_brxu(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 2) || inst.has_guard ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[0].index >= kUniformRegisters - 1 ||
      inst.operands[1].kind != operand_kind::kImmediate)
    return unsupported(inst, "invalid BRXU operand form");

  uint64_t target = read_uniform_register_pair(state, inst.operands[0].index);
  const int64_t bias = static_cast<int64_t>(inst.pc) +
                       static_cast<int64_t>(kInstructionBytes) +
                       inst.operands[1].immediate;
  if (bias < 0) {
    const uint64_t magnitude = static_cast<uint64_t>(-(bias + 1)) + 1;
    if (target < magnitude)
      return unsupported(inst, "BRXU target underflow");
    target -= magnitude;
  } else {
    const uint64_t magnitude = static_cast<uint64_t>(bias);
    if (target > std::numeric_limits<uint64_t>::max() - magnitude)
      return unsupported(inst, "BRXU target overflow");
    target += magnitude;
  }
  if ((target % kInstructionBytes) != 0)
    return unsupported(inst, "unaligned BRXU target");
  state.pc = target;
  return {step_status::kAdvanced, {}};
}

step_result execute_brx(const instruction &inst, warp_state &state,
                        execution_context &) {
  if (!has_typed_operands(inst, 2) || inst.has_guard ||
      inst.operands[0].kind != operand_kind::kRegister ||
      inst.operands[0].index >= kVectorRegisters - 1 ||
      inst.operands[1].kind != operand_kind::kImmediate)
    return unsupported(inst, "invalid BRX operand form");

  const int64_t bias = static_cast<int64_t>(inst.pc) +
                       static_cast<int64_t>(kInstructionBytes) +
                       inst.operands[1].immediate;
  std::vector<reconvergence_path> paths;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0)
      continue;
    uint64_t lane_target =
        read_register_pair(state, lane, inst.operands[0].index);
    if (bias < 0) {
      const uint64_t magnitude = static_cast<uint64_t>(-(bias + 1)) + 1;
      if (lane_target < magnitude)
        return unsupported(inst, "BRX target underflow");
      lane_target -= magnitude;
    } else {
      const uint64_t magnitude = static_cast<uint64_t>(bias);
      if (lane_target > std::numeric_limits<uint64_t>::max() - magnitude)
        return unsupported(inst, "BRX target overflow");
      lane_target += magnitude;
    }
    if ((lane_target % kInstructionBytes) != 0)
      return unsupported(inst, "unaligned BRX target");
    auto path =
        std::find_if(paths.begin(), paths.end(),
                     [lane_target](const reconvergence_path &candidate) {
                       return candidate.pc == lane_target;
                     });
    if (path == paths.end())
      paths.push_back({lane_target, 1u << lane});
    else
      path->active_mask |= 1u << lane;
  }
  if (paths.empty())
    return unsupported(inst, "BRX has no active target");
  if (paths.size() > 1) {
    // Compiler switch tables may select a different arm for each lane. Use
    // the enclosing BSSY just as for a divergent direct branch, retaining
    // every lane until BSYNC joins the arms. Validate before changing state.
    if (state.reconvergence_stack.empty())
      return unsupported(inst, "divergent BRX requires an explicit BSSY");
    auto &frame = state.reconvergence[state.reconvergence_stack.back()];
    if (!frame.valid)
      return unsupported(inst, "divergent BRX has an invalid BSSY state");
    for (size_t i = paths.size(); i-- > 1;)
      frame.pending_paths.push_back(paths[i]);
  }
  state.pc = paths.front().pc;
  state.active_mask = paths.front().active_mask;
  return {step_status::kAdvanced, {}};
}

step_result execute_convergence_branch(const instruction &inst,
                                       warp_state &state, execution_context &) {
  if (!has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kUniformRegister ||
      inst.operands[1].kind != operand_kind::kImmediate ||
      inst.operands[1].immediate < 0 ||
      (inst.operands[1].immediate % kInstructionBytes) != 0)
    return unsupported(inst, "invalid convergence branch operands");
  if (inst.has_guard) {
    uint32_t executing = 0;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      if ((state.active_mask & (1u << lane)) &&
          lane_executes(inst, state, lane))
        executing |= 1u << lane;
    if (!executing) {
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    if (executing != state.active_mask)
      return unsupported(
          inst, "divergently guarded convergence branch is unqualified");
  }
  const uint32_t convergence_mask =
      state.read_uniform_register(inst.operands[0].index);
  const bool converged = state.active_mask == convergence_mask;
  const bool take = inst.opcode == "BRA.CONV" ? converged : !converged;
  if (take)
    state.pc = static_cast<uint64_t>(inst.operands[1].immediate);
  else
    advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_warpsync(const instruction &, warp_state &,
                             execution_context &);

// Resume an arm after its collective block using its own BSSY frames. Shared
// outer frames stay live, preserving arrivals made by other arms meanwhile.
bool resume_collective_return(warp_state &state) {
  for (auto it = state.collective_returns.begin();
       it != state.collective_returns.end(); ++it) {
    if (it->common_stack != state.reconvergence_stack)
      continue;
    for (const auto &entry : it->local_frames) {
      if (state.reconvergence[entry.first].valid)
        return false;
    }
    for (auto entry = it->local_frames.rbegin();
         entry != it->local_frames.rend(); ++entry) {
      state.reconvergence[entry->first] = entry->second;
      state.reconvergence_stack.push_back(entry->first);
    }
    state.pc = it->pc;
    state.active_mask = it->active_mask;
    state.collective_returns.erase(it);
    return true;
  }
  return false;
}

step_result execute_collective_marker(const instruction &inst,
                                      warp_state &state,
                                      execution_context &context) {
  if (inst.opcode == "ENDCOLLECTIVE") {
    if (!has_typed_operands(inst, 0) || inst.has_guard)
      return unsupported(inst, "ENDCOLLECTIVE must be unpredicated");
    if (state.collective_executing_mask) {
      if (inst.pc + kInstructionBytes != state.collective_return_pc)
        return unsupported(inst, "collective block has unexpected end");
      state.active_mask = state.collective_executing_mask;
      state.pc = state.collective_return_pc;
      state.collective_executing_mask = 0;
      return {step_status::kAdvanced, {}};
    }
  } else if (inst.opcode == "WARPSYNC.COLLECTIVE.ALL") {
    if (!has_typed_operands(inst, 1) || inst.has_guard ||
        inst.operands[0].kind != operand_kind::kImmediate ||
        inst.operands[0].immediate <= static_cast<int64_t>(inst.pc) ||
        inst.operands[0].immediate % kInstructionBytes != 0 ||
        state.collective_executing_mask)
      return unsupported(inst, "unqualified full-warp collective entry");
    if (!state.warp_sync_stack.empty()) {
      instruction rendezvous = inst;
      rendezvous.opcode = "WARPSYNC.ALL";
      rendezvous.operands.clear();
      return execute_warpsync(rendezvous, state, context);
    }
    uint32_t arrived = state.active_mask;
    for (const auto &path : state.collective_waiting) {
      if (arrived & path.active_mask)
        return unsupported(inst, "overlapping collective participants");
      arrived |= path.active_mask;
    }
    if (arrived == 0xffffffffu) {
      if (!state.collective_waiting.empty()) {
        state.collective_executing_mask = state.active_mask;
        state.collective_return_pc = inst.operands[0].immediate;
        for (auto &path : state.collective_waiting)
          state.collective_returns.push_back(std::move(path));
        state.collective_waiting.clear();
        state.active_mask = arrived;
      }
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    // Qualified divergent case: paths are enclosed by explicit BSSY frames.
    // Save only the suffix owned entirely by this arm; leave the shared outer
    // frame available to schedule the other participants.
    if (state.reconvergence_stack.empty() || !state.independent_paths.empty())
      return unsupported(inst, "collective needs an explicit outer BSSY");
    collective_return_path suspended;
    suspended.pc = inst.operands[0].immediate;
    suspended.active_mask = state.active_mask;
    while (!state.reconvergence_stack.empty()) {
      const unsigned id = state.reconvergence_stack.back();
      auto &frame = state.reconvergence[id];
      if (frame.participating_mask & ~state.active_mask)
        break;
      suspended.local_frames.emplace_back(id, frame);
      frame = reconvergence_state{};
      state.reconvergence_stack.pop_back();
    }
    suspended.common_stack = state.reconvergence_stack;
    state.collective_waiting.push_back(std::move(suspended));
    if (resume_collective_return(state))
      return {step_status::kAdvanced, {}};
    if (!state.reconvergence_stack.empty()) {
      auto &outer = state.reconvergence[state.reconvergence_stack.back()];
      if (!outer.pending_paths.empty()) {
        const auto path = outer.pending_paths.back();
        outer.pending_paths.pop_back();
        state.pc = path.pc;
        state.active_mask = path.active_mask;
        return {step_status::kAdvanced, {}};
      }
    }
    return unsupported(inst, "collective has no runnable missing participants");
  } else if (!has_typed_operands(inst, 2) || inst.has_guard ||
             inst.operands[0].kind != operand_kind::kRegister ||
             inst.operands[1].kind != operand_kind::kImmediate ||
             inst.operands[1].immediate < 0 ||
             (inst.operands[1].immediate % kInstructionBytes) != 0) {
    return unsupported(inst, "invalid WARPSYNC.COLLECTIVE operands");
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_warpsync(const instruction &inst, warp_state &state,
                             execution_context &) {
  if (!has_typed_operands(inst, 0))
    return unsupported(inst, "WARPSYNC.ALL must not have operands");
  if (inst.has_guard) {
    uint32_t executing_mask = 0;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) != 0 &&
          lane_executes(inst, state, lane))
        executing_mask |= 1u << lane;
    }
    if (executing_mask == 0) {
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    if (executing_mask != state.active_mask)
      return unsupported(inst, "WARPSYNC.ALL has a nonuniform predicate");
  }
  if (!state.warp_sync_stack.empty()) {
    reconvergence_state &frame = state.warp_sync_stack.back();
    if (!frame.valid || frame.target_pc != inst.pc)
      return unsupported(inst, "WARPSYNC does not match the implicit branch");
    frame.arrived_mask |= state.active_mask;
    if (!frame.pending_paths.empty()) {
      const reconvergence_path path = frame.pending_paths.back();
      frame.pending_paths.pop_back();
      state.active_mask = path.active_mask;
      state.pc = path.pc;
      return {step_status::kAdvanced, {}};
    }
    if (frame.arrived_mask != frame.participating_mask)
      return unsupported(inst, "WARPSYNC is missing a divergent path");
    state.active_mask = frame.participating_mask;
    state.warp_sync_stack.pop_back();
  }

  // A forward branch need not target WARPSYNC directly. When the fallthrough
  // arm reaches it first, run any earlier deferred arm before crossing the
  // synchronization point, then merge that arm when it reaches the same PC.
  for (auto path = state.independent_paths.begin();
       path != state.independent_paths.end();) {
    if (path->pc == inst.pc) {
      state.active_mask |= path->active_mask;
      path = state.independent_paths.erase(path);
    } else {
      ++path;
    }
  }

  // WARPSYNC may occur inside an outer BSSY region before its BSYNC target.
  // In that case the currently executing arm must wait here while the other
  // serialized arms run up to the same WARPSYNC.  Letting the first arm pass
  // would make warp collectives such as ELECT observe only that arm.
  if (!state.reconvergence_stack.empty()) {
    reconvergence_state &outer =
        state.reconvergence[state.reconvergence_stack.back()];
    if (state.active_mask != outer.participating_mask &&
        !outer.pending_paths.empty()) {
      state.independent_paths.push_back({inst.pc, state.active_mask});
      const reconvergence_path path = outer.pending_paths.back();
      outer.pending_paths.pop_back();
      state.active_mask = path.active_mask;
      state.pc = path.pc;
      return {step_status::kAdvanced, {}};
    }
  }

  const auto earlier_path = std::find_if(
      state.independent_paths.rbegin(), state.independent_paths.rend(),
      [&inst](const reconvergence_path &path) { return path.pc < inst.pc; });
  if (earlier_path != state.independent_paths.rend()) {
    const reconvergence_path path = *earlier_path;
    state.independent_paths.erase(std::next(earlier_path).base());
    state.independent_paths.push_back({inst.pc, state.active_mask});
    state.active_mask = path.active_mask;
    state.pc = path.pc;
    return {step_status::kAdvanced, {}};
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bssy(const instruction &inst, warp_state &state,
                         execution_context &) {
  if (!has_typed_operands(inst, 2) || inst.has_guard)
    return unsupported(inst, "expected an unpredicated barrier and target");
  const operand *barrier = get_operand(inst, 0, operand_kind::kBarrierRegister);
  const operand *target = get_operand(inst, 1, operand_kind::kImmediate);
  if (barrier == nullptr || barrier->index >= kBarrierRegisters ||
      target == nullptr || target->immediate < 0 ||
      (target->immediate % kInstructionBytes) != 0)
    return unsupported(inst, "invalid BSSY operand form");
  reconvergence_state &frame = state.reconvergence[barrier->index];
  if (frame.valid)
    return unsupported(inst, "BSSY barrier register is already active");
  frame.valid = true;
  frame.reliable = inst.opcode == "BSSY.RELIABLE";
  frame.target_pc = static_cast<uint64_t>(target->immediate);
  frame.participating_mask = state.active_mask;
  frame.arrived_mask = 0;
  frame.pending_paths.clear();
  frame.escaped_paths.clear();
  state.reconvergence_stack.push_back(barrier->index);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bsync(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 1) || inst.has_guard)
    return unsupported(inst, "expected one unpredicated barrier operand");
  const operand *barrier = get_operand(inst, 0, operand_kind::kBarrierRegister);
  if (barrier == nullptr || barrier->index >= kBarrierRegisters ||
      state.reconvergence_stack.empty() ||
      state.reconvergence_stack.back() != barrier->index)
    return unsupported(inst, "BSYNC does not match the active BSSY");
  reconvergence_state &frame = state.reconvergence[barrier->index];
  if (!frame.valid)
    return unsupported(inst, "BSYNC barrier state is invalid");
  const bool reliable = inst.opcode == "BSYNC.RELIABLE";
  if (frame.reliable != reliable)
    return unsupported(inst, "BSYNC kind does not match the active BSSY");
  frame.arrived_mask |= state.active_mask;
  if (!frame.pending_paths.empty()) {
    const reconvergence_path path = frame.pending_paths.back();
    frame.pending_paths.pop_back();
    state.active_mask = path.active_mask;
    state.pc = path.pc;
    return {step_status::kAdvanced, {}};
  }
  if (frame.arrived_mask != frame.participating_mask) {
    if (resume_collective_return(state))
      return {step_status::kAdvanced, {}};
    return unsupported(inst, "BSYNC is missing a divergent path");
  }
  if (!frame.escaped_paths.empty() && state.reconvergence_stack.size() < 2)
    return unsupported(inst, "BREAK has no outer reconvergence");
  const uint64_t target_pc = frame.target_pc;
  const uint32_t merged_mask = frame.participating_mask;
  std::vector<reconvergence_path> escaped_paths =
      std::move(frame.escaped_paths);
  frame = reconvergence_state{};
  state.reconvergence_stack.pop_back();
  if (!escaped_paths.empty()) {
    reconvergence_state &outer =
        state.reconvergence[state.reconvergence_stack.back()];
    for (const reconvergence_path &path : escaped_paths)
      outer.pending_paths.push_back(path);
  }
  state.active_mask = merged_mask;
  state.pc = target_pc;
  return {step_status::kAdvanced, {}};
}

step_result execute_break(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one barrier operand");
  const operand *barrier = get_operand(inst, 0, operand_kind::kBarrierRegister);
  if (barrier == nullptr || barrier->index >= kBarrierRegisters ||
      state.reconvergence_stack.empty() ||
      state.reconvergence_stack.back() != barrier->index)
    return unsupported(inst, "BREAK does not match the active BSSY");
  reconvergence_state &frame = state.reconvergence[barrier->index];
  const bool reliable = inst.opcode == "BREAK.RELIABLE";
  if (!frame.valid || frame.reliable != reliable)
    return unsupported(inst, "BREAK kind does not match the active BSSY");
  uint32_t breaking = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane))
      breaking |= 1u << lane;
  }
  if (breaking != 0) {
    frame.participating_mask &= ~breaking;
    if (breaking == state.active_mask) {
      if (state.reconvergence_stack.size() < 2)
        return unsupported(inst, "BREAK has no outer reconvergence");
      if (!frame.pending_paths.empty()) {
        frame.escaped_paths.push_back({inst.pc + kInstructionBytes, breaking});
        const reconvergence_path path = frame.pending_paths.back();
        frame.pending_paths.pop_back();
        state.active_mask = path.active_mask;
        state.pc = path.pc;
        return {step_status::kAdvanced, {}};
      }
      if (frame.arrived_mask != frame.participating_mask)
        return unsupported(inst, "BREAK is missing an inner path");

      const uint64_t target_pc = frame.target_pc;
      const uint32_t participating_mask = frame.participating_mask;
      std::vector<reconvergence_path> escaped_paths =
          std::move(frame.escaped_paths);
      frame = reconvergence_state{};
      state.reconvergence_stack.pop_back();
      reconvergence_state &outer =
          state.reconvergence[state.reconvergence_stack.back()];
      if (participating_mask != 0)
        outer.pending_paths.push_back({target_pc, participating_mask});
      for (const reconvergence_path &path : escaped_paths)
        outer.pending_paths.push_back(path);
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    frame.escaped_paths.push_back({inst.pc + kInstructionBytes, breaking});
    state.active_mask &= ~breaking;
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_elect(const instruction &inst, warp_state &state,
                          execution_context &) {
  if (!has_typed_operands(inst, 3))
    return unsupported(inst, "expected three typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kPredicate);
  const operand *leader = get_operand(inst, 1, operand_kind::kUniformRegister);
  if (destination == nullptr || leader == nullptr)
    return unsupported(inst, "expected predicate and uniform destinations");

  unsigned elected_lane = kWarpLanes;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    bool participates = false;
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane) &&
        !read_predicate_operand(inst.operands[2], state, lane, participates))
      return unsupported(inst, "unsupported ELECT participation predicate");
    if (elected_lane == kWarpLanes && lane_executes(inst, state, lane) &&
        participates)
      elected_lane = lane;
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane))
      state.write_predicate(lane, destination->index, lane == elected_lane);
  }
  if (elected_lane != kWarpLanes)
    state.write_uniform_register(leader->index, elected_lane);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_voteu(const instruction &inst, warp_state &state,
                          execution_context &) {
  const bool uniform = inst.opcode.rfind("VOTEU.", 0) == 0;
  // The production logits guard is a lane predicate copied from UP1.
  // Qualify converged predication only: no lane writes on an all-off vote.
  if (inst.has_guard) {
    if (!uniform)
      return unsupported(inst, "guarded nonuniform vote is not qualified");
    uint32_t executing = 0;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      if ((state.active_mask & (1u << lane)) &&
          lane_executes(inst, state, lane))
        executing |= 1u << lane;
    if (!executing) {
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    if (executing != state.active_mask)
      return unsupported(inst, "divergently guarded VOTEU is not qualified");
  }
  if (has_typed_operands(inst, 3)) {
    const operand *mask = get_operand(inst, 0,
                                      uniform ? operand_kind::kUniformRegister
                                              : operand_kind::kRegister);
    const operand *any = get_operand(inst, 1,
                                     uniform ? operand_kind::kUniformPredicate
                                             : operand_kind::kPredicate);
    if (mask == nullptr || any == nullptr)
      return unsupported(inst, "invalid VOTEU ballot destinations");
    uint32_t ballot = 0;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      bool value = false;
      if ((state.active_mask & (1u << lane)) != 0 &&
          !read_predicate_operand(inst.operands[2], state, lane, value))
        return unsupported(inst, "unsupported VOTEU ballot predicate");
      if (value)
        ballot |= uint32_t{1} << lane;
    }
    const bool all = inst.opcode == "VOTEU.ALL" || inst.opcode == "VOTE.ALL";
    const bool result = all ? ballot == state.active_mask : ballot != 0;
    if (uniform) {
      state.write_uniform_register(mask->index, ballot);
      state.write_uniform_predicate(any->index, result);
    } else {
      for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
        if (!(state.active_mask & (1u << lane)))
          continue;
        state.write_register(lane, mask->index, ballot);
        state.write_predicate(lane, any->index, result);
      }
    }
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected a reduction or ballot");
  const operand *destination = get_operand(
      inst, 0,
      uniform ? operand_kind::kUniformPredicate : operand_kind::kPredicate);
  if (destination == nullptr)
    return unsupported(inst, "unexpected vote predicate destination");

  const bool vote_all = inst.opcode == "VOTEU.ALL" || inst.opcode == "VOTE.ALL";
  bool result = vote_all;
  bool saw_active_lane = false;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0)
      continue;
    saw_active_lane = true;
    bool value = false;
    if (!read_predicate_operand(inst.operands[1], state, lane, value))
      return unsupported(inst, "unsupported VOTEU source predicate");
    result = vote_all ? result && value : result || value;
  }
  if (!saw_active_lane)
    return unsupported(inst, "VOTEU requires at least one active lane");
  if (uniform)
    state.write_uniform_predicate(destination->index, result);
  else
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      if (state.active_mask & (1u << lane))
        state.write_predicate(lane, destination->index, result);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_cta_barrier(const instruction &inst, warp_state &state,
                                execution_context &context) {
  const operand *participant_count =
      get_operand(inst, 1, operand_kind::kImmediate);
  const bool has_valid_operand_count =
      has_typed_operands(inst, 1) || has_typed_operands(inst, 2);
  if (!has_valid_operand_count ||
      (inst.operands[0].kind != operand_kind::kImmediate &&
       inst.operands[0].kind != operand_kind::kRegister))
    return unsupported(inst,
                       "expected an immediate or register named barrier id");
  if (state.active_mask != 0xffffffffu) {
    std::ostringstream detail;
    detail << "CTA barrier requires a converged full warp (active mask 0x"
           << std::hex << state.active_mask << ')';
    return unsupported(inst, detail.str());
  }
  uint32_t executing_lanes = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (lane_executes(inst, state, lane))
      executing_lanes |= uint32_t{1} << lane;
  }
  if (executing_lanes == 0) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (executing_lanes != state.active_mask)
    return unsupported(inst, "CTA barrier cannot be predicated per lane");
  if (context.cta == nullptr || context.warp_id >= context.cta->warp_count())
    return unsupported(inst, "CTA barrier requires a multi-warp CTA context");
  const bool packed_register =
      inst.operands.size() == 2 &&
      inst.operands[0].kind == operand_kind::kRegister &&
      inst.operands[1].kind == operand_kind::kRegister &&
      inst.operands[0].index == inst.operands[1].index;
  uint32_t id = 0;
  if (!read_u32_operand(inst.operands[0], state, 0, id) ||
      (!packed_register && id >= kCtaBarrierSlots))
    return unsupported(inst, "named barrier id must be from 0 through 15");
  if (inst.operands[0].kind == operand_kind::kRegister) {
    for (unsigned lane = 1; lane < kWarpLanes; ++lane) {
      uint32_t lane_id = 0;
      if (!read_u32_operand(inst.operands[0], state, lane, lane_id) ||
          lane_id != id)
        return unsupported(inst,
                           "register-selected barrier id must be warp uniform");
    }
  }
  uint32_t expected_threads = 0;
  if (packed_register) {
    // CUDA 13.3 bar.sync reg,reg packs count in bits 31:16 and id in
    // bits 3:0, then emits BAR.SYNC.DEFER_BLOCKING Rn,Rn.
    if (id & 0xfff0u)
      return unsupported(inst, "unqualified packed barrier reserved bits");
    expected_threads = id >> 16;
    id &= 15;
  } else if (participant_count && participant_count->immediate > 0 &&
             participant_count->immediate <= UINT32_MAX) {
    expected_threads = static_cast<uint32_t>(participant_count->immediate);
  }
  if (has_typed_operands(inst, 2) &&
      (!expected_threads || expected_threads % kWarpLanes != 0 ||
       expected_threads > context.cta->warp_count() * kWarpLanes))
    return unsupported(
        inst, "explicit CTA barrier count must name whole configured warps");
  const unsigned expected_warp_count = expected_threads / kWarpLanes;

  functional_named_barrier_effect timing_effect;
  timing_effect.valid = true;
  timing_effect.id = id;
  if (expected_threads)
    timing_effect.participant_count = expected_threads;
  context.record_named_barrier_effect(timing_effect);

  const uint64_t generation = context.cta->generation(id);
  const bool released =
      context.cta->arrive(id, context.warp_id, expected_warp_count);
  advance(inst, state);
  if (released)
    return {step_status::kAdvanced, {}};
  state.waiting_at_cta_barrier = true;
  state.cta_barrier_id = id;
  state.cta_barrier_generation = generation;
  return {step_status::kBlocked, {}};
}

step_result execute_cta_barrier_arrive(const instruction &inst,
                                       warp_state &state,
                                       execution_context &context) {
  step_result result = execute_cta_barrier(inst, state, context);
  if (result.status == step_status::kBlocked) {
    state.waiting_at_cta_barrier = false;
    return {step_status::kAdvanced, {}};
  }
  return result;
}

step_result execute_cta_barrier_deferred(const instruction &inst,
                                         warp_state &state,
                                         execution_context &context) {
  step_result result = execute_cta_barrier(inst, state, context);
  if (result.status != step_status::kBlocked &&
      (result.status != step_status::kAdvanced ||
       !context.named_barrier_effect.valid))
    return result;

  const unsigned id = context.named_barrier_effect.id;
  // The final arrival advances the functional generation immediately, but
  // must still observe the timing handoff delay at its next ordered consumer.
  // Pure functional execution sees the advanced generation and proceeds.
  const uint64_t generation = result.status == step_status::kBlocked
                                  ? state.cta_barrier_generation
                                  : context.cta->generation(id) - 1;
  state.deferred_cta_barrier_pending[id] = true;
  state.deferred_cta_barrier_generation[id] = generation;
  state.waiting_at_cta_barrier = false;
  return {step_status::kAdvanced, {}};
}

step_result execute_exit(const instruction &inst, warp_state &state,
                         execution_context &) {
  const bool has_operand_predicate = has_typed_operands(inst, 1);
  if (!has_typed_operands(inst, 0) && !has_operand_predicate)
    return unsupported(inst, "EXIT accepts only an optional predicate");
  if (has_operand_predicate &&
      inst.operands[0].kind != operand_kind::kPredicate)
    return unsupported(inst, "EXIT operand must be a lane predicate");
  uint32_t exiting = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    bool operand_executes = true;
    if (has_operand_predicate &&
        !read_predicate_operand(inst.operands[0], state, lane,
                                operand_executes))
      return unsupported(inst, "unsupported EXIT predicate operand");
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane) && operand_executes)
      exiting |= (1u << lane);
  }
  state.active_mask &= ~exiting;
  // EXIT retires lanes, not merely the currently scheduled divergent path.
  // Remove those lanes from every enclosing convergence obligation so BSYNC
  // cannot resurrect them, and run suspended siblings before retiring a warp.
  for (unsigned barrier : state.reconvergence_stack)
    state.reconvergence[barrier].participating_mask &= ~exiting;
  for (auto &frame : state.warp_sync_stack)
    frame.participating_mask &= ~exiting;
  while (state.active_mask == 0 && !state.reconvergence_stack.empty()) {
    auto &frame = state.reconvergence[state.reconvergence_stack.back()];
    if (!frame.pending_paths.empty()) {
      const auto path = frame.pending_paths.back();
      frame.pending_paths.pop_back();
      state.active_mask = path.active_mask;
      state.pc = path.pc;
      return {step_status::kAdvanced, {}};
    }
    if (frame.arrived_mask != frame.participating_mask)
      return unsupported(inst, "EXIT is missing a BSSY divergent path");
    if (!frame.escaped_paths.empty() && state.reconvergence_stack.size() < 2)
      return unsupported(inst, "EXIT has escaped paths without outer BSSY");
    const uint64_t target = frame.target_pc;
    const uint32_t arrived = frame.arrived_mask;
    auto escaped = std::move(frame.escaped_paths);
    frame = reconvergence_state{};
    state.reconvergence_stack.pop_back();
    if (!escaped.empty()) {
      auto &outer = state.reconvergence[state.reconvergence_stack.back()];
      for (const auto &path : escaped)
        outer.pending_paths.push_back(path);
    }
    if (arrived) {
      state.active_mask = arrived;
      state.pc = target;
      return {step_status::kAdvanced, {}};
    }
  }
  while (state.active_mask == 0 && !state.warp_sync_stack.empty()) {
    auto &frame = state.warp_sync_stack.back();
    if (!frame.pending_paths.empty()) {
      const auto path = frame.pending_paths.back();
      frame.pending_paths.pop_back();
      state.active_mask = path.active_mask;
      state.pc = path.pc;
      return {step_status::kAdvanced, {}};
    }
    if (frame.arrived_mask != frame.participating_mask)
      return unsupported(inst, "EXIT is missing a WARPSYNC divergent path");
    if (frame.arrived_mask) {
      state.active_mask = frame.arrived_mask;
      state.pc = frame.target_pc;
      return {step_status::kAdvanced, {}};
    }
    state.warp_sync_stack.pop_back();
  }
  if (state.active_mask == 0 && !state.independent_paths.empty()) {
    const reconvergence_path path = state.independent_paths.back();
    state.independent_paths.pop_back();
    state.active_mask = path.active_mask;
    state.pc = path.pc;
    return {step_status::kAdvanced, {}};
  }
  if (state.active_mask == 0)
    return {step_status::kExited, {}};
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_control_semantics(frontend &target) {
  target.register_semantics("NOP", execute_nop);
  target.register_semantics("YIELD", execute_nop);
  target.register_semantics("NANOSLEEP", execute_nanosleep);
  target.register_semantics("NANOSLEEP.SYNCS", execute_nanosleep);
  target.register_semantics("USETMAXREG.TRY_ALLOC.CTAPOOL", execute_usetmaxreg);
  target.register_semantics("USETMAXREG.DEALLOC.CTAPOOL", execute_usetmaxreg);
  target.register_semantics("CALL.REL.NOINC", execute_call_relative);
  target.register_semantics("RET.REL.NODEC", execute_return_relative);
  target.register_semantics("BRA", execute_bra);
  target.register_semantics("BRA.U", execute_bra_uniform);
  target.register_semantics("BRA.U.ANY", execute_bra_uniform_vote);
  target.register_semantics("BRA.U.NOT_TID0", execute_bra_not_tid0);
  target.register_semantics("BRX", execute_brx);
  target.register_semantics("BRXU", execute_brxu);
  target.register_semantics("BRA.DIV", execute_convergence_branch);
  target.register_semantics("BRA.CONV", execute_convergence_branch);
  target.register_semantics("BSSY", execute_bssy);
  target.register_semantics("BSYNC", execute_bsync);
  target.register_semantics("BSSY.RECONVERGENT", execute_bssy);
  target.register_semantics("BSYNC.RECONVERGENT", execute_bsync);
  target.register_semantics("BSSY.RELIABLE", execute_bssy);
  target.register_semantics("BSYNC.RELIABLE", execute_bsync);
  target.register_semantics("BREAK", execute_break);
  target.register_semantics("BREAK.RELIABLE", execute_break);
  target.register_semantics("WARPSYNC.ALL", execute_warpsync);
  target.register_semantics("WARPSYNC.COLLECTIVE", execute_collective_marker);
  target.register_semantics("WARPSYNC.COLLECTIVE.ALL",
                            execute_collective_marker);
  target.register_semantics("ENDCOLLECTIVE", execute_collective_marker);
  target.register_semantics("ELECT", execute_elect);
  target.register_semantics("VOTE.ALL", execute_voteu);
  target.register_semantics("VOTE.ANY", execute_voteu);
  target.register_semantics("VOTEU.ALL", execute_voteu);
  target.register_semantics("VOTEU.ANY", execute_voteu);
  target.register_semantics("UTMACMDFLUSH", execute_nop);
  target.register_semantics("BAR.ARV", execute_cta_barrier_arrive);
  target.register_semantics("BAR.SYNC.DEFER_BLOCKING",
                            execute_cta_barrier_deferred);
  target.register_semantics("EXIT", execute_exit);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
