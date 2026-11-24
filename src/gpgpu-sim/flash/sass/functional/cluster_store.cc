#include "../../cluster_store.h"
#include "internal.h"
#include <array>
#include <map>
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result execute_stas(const instruction &inst, warp_state &state,
                         execution_context &context) {
  const unsigned bytes = inst.opcode == "STAS.128"  ? 16
                         : inst.opcode == "STAS.64" ? 8
                                                    : 4;
  if (!has_typed_operands(inst, 2) || !context.cluster ||
      inst.operands[0].kind != operand_kind::kMemoryAddress ||
      inst.operands[0].has_prefix ||
      inst.operands[1].kind != operand_kind::kRegister)
    return unsupported(inst, "unqualified cluster STAS form");
  const auto &source = inst.operands[1];
  if (source.negated || source.absolute || source.bitwise_complement ||
      source.wide || source.half_selection != operand_half_selection::kNone ||
      source.matrix_layout != operand_matrix_layout::kNone ||
      (source.index != kZeroRegister &&
       source.index + bytes / 4 > kZeroRegister + 1))
    return unsupported(inst, "unqualified STAS payload registers");
  uint32_t mask = 0;
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((state.active_mask & (1u << lane)) && lane_executes(inst, state, lane))
      mask |= 1u << lane;
  if (!mask) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  struct store_request {
    unsigned lane, owner;
    uint32_t destination, barrier;
    std::array<uint32_t, 4> values;
  };
  std::vector<store_request> requests;
  std::map<std::pair<unsigned, uint32_t>, unsigned> pending_bytes;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(mask & (1u << lane)))
      continue;
    uint64_t packed = 0;
    if (!evaluate_address(inst.operands[0], state, lane, packed))
      return unsupported(inst, "invalid STAS address bundle");
    const uint32_t destination = packed, barrier = packed >> 32;
    cluster_store_request decoded;
    if (!decode_cluster_store(destination, barrier, bytes,
                              context.cluster->ctas.size(), decoded) ||
        decoded.owner >= context.cluster->memories.size())
      return unsupported(inst, "unaligned or mismatched STAS owners");
    const unsigned owner = decoded.owner;
    auto *target = context.cluster->ctas[owner];
    auto *memory = context.cluster->memories[owner];
    if (!memory || !target ||
        !target->mbarrier_initialized(barrier & 0xffffff) ||
        target->mbarrier_pending_transaction_bytes(barrier & 0xffffff) < bytes)
      return unsupported(inst,
                         "STAS completion is not pending on peer barrier");
    auto &total = pending_bytes[{owner, barrier & 0xffffff}];
    total += bytes;
    if (total > target->mbarrier_pending_transaction_bytes(barrier & 0xffffff))
      return unsupported(inst, "STAS warp exceeds pending transaction bytes");
    store_request request{lane, owner, destination & 0xffffff, barrier, {}};
    if (!memory->read(memory_space::kShared, request.destination,
                      request.values.data(), bytes))
      return memory_fault(inst, memory_space::kShared, destination, bytes);
    for (unsigned word = 0; word < bytes / 4; ++word)
      request.values[word] = state.read_register(
          lane,
          source.index == kZeroRegister ? kZeroRegister : source.index + word);
    requests.push_back(request);
  }
  // Validate every participating lane before applying payload or completion.
  for (const auto &request : requests) {
    auto *memory = context.cluster->memories[request.owner];
    auto *target = context.cluster->ctas[request.owner];
    if (!memory->write(memory_space::kShared, request.destination,
                       request.values.data(), bytes))
      return memory_fault(inst, memory_space::kShared, request.destination,
                          bytes);
    target->complete_mbarrier_transaction(request.barrier & 0xffffff, bytes);
    functional_mbarrier_effect effect;
    effect.valid = true;
    effect.address = request.barrier;
    effect.count = bytes;
    context.record_mbarrier_effect(request.lane, effect);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace
void register_cluster_store_semantics(frontend &target) {
  for (const char *op : {"STAS", "STAS.64", "STAS.128"})
    target.register_semantics(op, execute_stas);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
