#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.12: split-phase barrier state and waits.

bool decode_initial_mbarrier(uint64_t value, uint32_t &expected_arrivals) {
  const uint32_t low = static_cast<uint32_t>(value);
  const uint32_t high = static_cast<uint32_t>(value >> 32);
  const uint32_t encoded_pending = low >> 1;
  if ((low & 1u) != 0 || high != (encoded_pending << 11) ||
      encoded_pending >= (uint32_t{1} << 20))
    return false;
  expected_arrivals = (uint32_t{1} << 20) - encoded_pending;
  return expected_arrivals != 0 && expected_arrivals < (uint32_t{1} << 20);
}

bool ensure_mbarrier_imported(uint64_t location, execution_context &context) {
  if (context.cta == nullptr)
    return false;
  if (context.cta->mbarrier_initialized(location))
    return true;
  if (context.memory == nullptr)
    return false;
  uint64_t value = 0;
  uint32_t expected_arrivals = 0;
  return context.memory->read(memory_space::kShared, location, &value,
                              sizeof(value)) &&
         decode_initial_mbarrier(value, expected_arrivals) &&
         context.cta->initialize_mbarrier(location, expected_arrivals);
}

bool record_uniform_mbarrier_effect(const instruction &inst,
                                    const warp_state &state,
                                    execution_context &context,
                                    functional_mbarrier_effect effect) {
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (uint32_t{1} << lane)) != 0 &&
        lane_executes(inst, state, lane)) {
      effect.valid = true;
      context.record_mbarrier_effect(lane, effect);
      return true;
    }
  }
  return false;
}

step_result execute_syncs_exchange(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *source = get_operand(inst, 1, operand_kind::kUniformRegister);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kUniformRegister ||
      source == nullptr)
    return unsupported(inst, "unsupported uniform 64-bit exchange form");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr || context.cta == nullptr)
    return unsupported(inst, "SYNCS exchange requires CTA memory and state");

  uint64_t location = 0;
  if (!evaluate_shared_address(*address, state, 0, location) ||
      (location % alignof(uint64_t)) != 0)
    return unsupported(inst, "invalid aligned SYNCS exchange address");
  const uint64_t new_value = read_uniform_register_pair(state, source->index);
  uint32_t expected_arrivals = 0;
  if (!decode_initial_mbarrier(new_value, expected_arrivals))
    return unsupported(inst, "unrecognized mbarrier.init state encoding");
  if (context.cta->mbarrier_initialized(location))
    return unsupported(inst, "invalid or duplicate mbarrier initialization");

  uint64_t old_value = 0;
  if (!context.memory->read(memory_space::kShared, location, &old_value,
                            sizeof(old_value)) ||
      !context.memory->write(memory_space::kShared, location, &new_value,
                             sizeof(new_value)))
    return memory_fault(inst, memory_space::kShared, location,
                        sizeof(uint64_t));
  write_uniform_register_pair(state, address->prefix_index, old_value);
  if (!context.cta->initialize_mbarrier(location, expected_arrivals))
    return unsupported(inst, "mbarrier initialization state changed");
  if (location > UINT32_MAX)
    return unsupported(inst, "mbarrier address exceeds timing ABI");
  functional_mbarrier_effect timing_effect;
  timing_effect.address = static_cast<uint32_t>(location);
  timing_effect.count = expected_arrivals;
  if (!record_uniform_mbarrier_effect(inst, state, context, timing_effect))
    return unsupported(inst, "executing mbarrier init has no issuing lane");

  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_syncs_arrive(const instruction &inst, warp_state &state,
                                 execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *count = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kRegister ||
      address->prefix_index != kZeroRegister || count == nullptr)
    return unsupported(inst, "only discarded arrival state works");
  if (context.cta == nullptr)
    return unsupported(inst, "SYNCS arrival requires CTA state");

  // mbarrier.complete_tx.shared::cluster lowers to RED.A0TX: it contributes
  // no arrival (A0) and subtracts the register byte count (TX) from the
  // outstanding transaction bytes.  RED is warp-uniform at the timing ABI,
  // so perform one completion after verifying every executing lane names the
  // same barrier and byte count.
  if (inst.opcode == "SYNCS.ARRIVE.TRANS64.RED.A0TX") {
    bool found = false;
    unsigned issuing_lane = 0;
    uint64_t location = 0;
    uint64_t transaction_bytes = 0;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) == 0 ||
          !lane_executes(inst, state, lane))
        continue;
      uint64_t lane_location = 0;
      if (!evaluate_shared_address(*address, state, lane, lane_location) ||
          (lane_location % alignof(uint64_t)) != 0)
        return unsupported(inst, "invalid aligned SYNCS completion address");
      const uint64_t lane_bytes = state.read_register(lane, count->index);
      if (!found) {
        found = true;
        issuing_lane = lane;
        location = lane_location;
        transaction_bytes = lane_bytes;
      } else if (lane_location != location || lane_bytes != transaction_bytes) {
        return unsupported(inst, "non-uniform SYNCS completion operands");
      }
    }
    if (!found) {
      advance(inst, state);
      return {step_status::kAdvanced, {}};
    }
    if (!ensure_mbarrier_imported(location, context))
      return unsupported(inst, "completion references invalid mbarrier state");
    if (location > UINT32_MAX || transaction_bytes > UINT32_MAX)
      return unsupported(inst, "mbarrier completion operand exceeds 32 bits");
    functional_mbarrier_effect timing_effect;
    timing_effect.valid = true;
    timing_effect.address = static_cast<uint32_t>(location);
    timing_effect.count = static_cast<uint32_t>(transaction_bytes);
    context.record_mbarrier_effect(issuing_lane, timing_effect);
    if (!context.cta->complete_mbarrier_transaction(location,
                                                    transaction_bytes))
      return unsupported(inst, "invalid mbarrier transaction completion");
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location) ||
        (location % alignof(uint64_t)) != 0)
      return unsupported(inst, "invalid aligned SYNCS arrival address");
    if (!ensure_mbarrier_imported(location, context))
      return unsupported(inst, "arrival references invalid mbarrier state");

    // cp.async.mbarrier.arrive is lowered to an A0T1 token followed by an
    // ARRIVES.LDGSTSBAR.TRANSCNT completion.  LDGSTS is synchronous in the
    // functional frontend, so the transaction token is already complete and
    // both halves intentionally leave the architectural barrier unchanged.
    if (inst.opcode == "SYNCS.ARRIVE.TRANS64.RED.A0T1")
      continue;

    uint32_t arrivals = 1;
    uint64_t transaction_bytes = 0;
    if (inst.opcode == "SYNCS.ARRIVE.TRANS64.ART0")
      arrivals = state.read_register(lane, count->index);
    else if (inst.opcode == "SYNCS.ARRIVE.TRANS64")
      transaction_bytes = state.read_register(lane, count->index);
    else if (inst.opcode != "SYNCS.ARRIVE.TRANS64.A1T0" &&
             inst.opcode != "SYNCS.ARRIVE.TRANS64.RED.A1T0")
      return unsupported(inst, "unknown SYNCS arrival count mode");

    if (location > UINT32_MAX || (transaction_bytes > UINT32_MAX))
      return unsupported(inst, "mbarrier timing operand exceeds 32 bits");
    functional_mbarrier_effect timing_effect;
    timing_effect.valid = true;
    timing_effect.address = static_cast<uint32_t>(location);
    // SYNCS.ARRIVE.TRANS64 is the combined arrive.expect_tx form. Its timing
    // operand is the transaction-byte count even when that count is zero;
    // substituting the implicit arrival count would create a transaction that
    // can never complete in an empty measurement.
    timing_effect.count = static_cast<uint32_t>(
        inst.opcode == "SYNCS.ARRIVE.TRANS64" ? transaction_bytes : arrivals);
    context.record_mbarrier_effect(lane, timing_effect);

    if (!context.cta->arrive_mbarrier(location, arrivals, transaction_bytes))
      return unsupported(inst, "invalid mbarrier arrival or byte count");
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_arrives_ldgstsbar(const instruction &inst,
                                      warp_state &state,
                                      execution_context &context) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one mbarrier address");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (address == nullptr || address->has_prefix)
    return unsupported(inst, "unsupported LDGSTSBAR arrival address");
  if (context.cta == nullptr)
    return unsupported(inst, "LDGSTSBAR arrival requires CTA state");

  const bool arrival_count = inst.opcode == "ARRIVES.LDGSTSBAR.64.ARVCNT";
  if (!arrival_count && inst.opcode != "ARRIVES.LDGSTSBAR.64.TRANSCNT")
    return unsupported(inst, "unknown LDGSTSBAR arrival mode");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location) ||
        (location % alignof(uint64_t)) != 0)
      return unsupported(inst, "invalid aligned LDGSTSBAR arrival address");
    if (!ensure_mbarrier_imported(location, context))
      return unsupported(inst, "LDGSTSBAR references invalid mbarrier state");
    if (location > UINT32_MAX)
      return unsupported(inst, "LDGSTSBAR address exceeds timing ABI");
    functional_mbarrier_effect timing_effect;
    timing_effect.valid = true;
    timing_effect.address = static_cast<uint32_t>(location);
    context.record_mbarrier_effect(lane, timing_effect);
    if (arrival_count && !context.cta->arrive_mbarrier(location, 1, 0))
      return unsupported(inst, "invalid LDGSTSBAR arrival count");
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_syncs_try_wait_parity(const instruction &inst,
                                          warp_state &state,
                                          execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *phase = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kPredicate || phase == nullptr)
    return unsupported(inst, "expected Pdst[address],Rphase operands");
  if (context.cta == nullptr)
    return unsupported(inst, "SYNCS try-wait requires CTA state");

  bool recorded_timing_effect = false;
  uint32_t pending_mask = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location) ||
        (location % alignof(uint64_t)) != 0)
      return unsupported(inst, "invalid aligned SYNCS try-wait address");
    if (!ensure_mbarrier_imported(location, context)) {
      uint64_t encoded = 0;
      const bool readable =
          context.memory != nullptr &&
          context.memory->read(memory_space::kShared, location, &encoded,
                               sizeof(encoded));
      uint32_t expected_arrivals = 0;
      const bool recognized =
          readable && decode_initial_mbarrier(encoded, expected_arrivals);
      std::ostringstream detail;
      detail << "try-wait references invalid mbarrier state at shared 0x"
             << std::hex << location << " (raw=";
      if (readable)
        detail << "0x" << encoded;
      else
        detail << "unreadable";
      detail << ", init_encoding=" << (recognized ? "yes" : "no") << ")";
      return unsupported(inst, detail.str());
    }
    const bool requested_phase =
        (state.read_register(lane, phase->index) >> 31) != 0;
    const bool complete =
        (context.cta->mbarrier_phase(location) & 1u) != requested_phase;
    state.write_predicate(lane, address->prefix_index, complete);
    if (!complete)
      pending_mask |= uint32_t{1} << lane;
    if (location > UINT32_MAX)
      return unsupported(inst, "mbarrier address exceeds timing ABI");
    // Preserve every lane's question. Only the timing adapter may coalesce
    // uniform waits for the backend's warp-wide release callback.
    functional_mbarrier_effect timing_effect;
    timing_effect.valid = true;
    timing_effect.address = static_cast<uint32_t>(location);
    timing_effect.parity = requested_phase;
    context.record_mbarrier_effect(lane, timing_effect);
    recorded_timing_effect = true;
  }
  if (recorded_timing_effect)
    state.record_mbarrier_try_wait(address->prefix_index, pending_mask);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_syncs_invalidate(const instruction &inst, warp_state &state,
                                     execution_context &context) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one mbarrier address");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (address == nullptr || address->has_prefix)
    return unsupported(inst, "unsupported mbarrier invalidate address");

  uint32_t executing_lanes = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) != 0 &&
        lane_executes(inst, state, lane))
      executing_lanes |= uint32_t{1} << lane;
  }
  if (executing_lanes == 0) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if ((executing_lanes & (executing_lanes - 1)) != 0)
    return unsupported(inst, "mbarrier invalidate requires one lane");
  if (context.cta == nullptr)
    return unsupported(inst, "mbarrier invalidate requires CTA state");

  const unsigned lane = static_cast<unsigned>(__builtin_ctz(executing_lanes));
  uint64_t location = 0;
  if (!evaluate_shared_address(*address, state, lane, location) ||
      (location % alignof(uint64_t)) != 0)
    return unsupported(inst, "invalid aligned mbarrier invalidate address");
  if (!ensure_mbarrier_imported(location, context) ||
      !context.cta->invalidate_mbarrier(location))
    return unsupported(inst, "mbarrier is uninitialized or still pending");
  if (location > UINT32_MAX)
    return unsupported(inst, "mbarrier address exceeds timing ABI");
  functional_mbarrier_effect timing_effect;
  timing_effect.valid = true;
  timing_effect.address = static_cast<uint32_t>(location);
  context.record_mbarrier_effect(lane, timing_effect);
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_syncs_invalidate_all(const instruction &inst,
                                         warp_state &state,
                                         execution_context &) {
  if (!has_typed_operands(inst, 0) || inst.has_guard)
    return unsupported(inst, "SYNCS.CCTL.IVALL must be unpredicated");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_mbarrier_semantics(frontend &target) {
  target.register_semantics("SYNCS.EXCH.64", execute_syncs_exchange);
  target.register_semantics("SYNCS.ARRIVE.TRANS64", execute_syncs_arrive);
  target.register_semantics("SYNCS.ARRIVE.TRANS64.A1T0", execute_syncs_arrive);
  target.register_semantics("SYNCS.ARRIVE.TRANS64.RED.A1T0",
                            execute_syncs_arrive);
  target.register_semantics("SYNCS.ARRIVE.TRANS64.RED.A0T1",
                            execute_syncs_arrive);
  target.register_semantics("SYNCS.ARRIVE.TRANS64.RED.A0TX",
                            execute_syncs_arrive);
  target.register_semantics("SYNCS.ARRIVE.TRANS64.ART0", execute_syncs_arrive);
  target.register_semantics("ARRIVES.LDGSTSBAR.64.ARVCNT",
                            execute_arrives_ldgstsbar);
  target.register_semantics("ARRIVES.LDGSTSBAR.64.TRANSCNT",
                            execute_arrives_ldgstsbar);
  target.register_semantics("SYNCS.PHASECHK.TRANS64.TRYWAIT",
                            execute_syncs_try_wait_parity);
  target.register_semantics("SYNCS.PHASECHK.TRANS64",
                            execute_syncs_try_wait_parity);
  target.register_semantics("SYNCS.CCTL.IV", execute_syncs_invalidate);
  target.register_semantics("SYNCS.CCTL.IVALL", execute_syncs_invalidate_all);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
