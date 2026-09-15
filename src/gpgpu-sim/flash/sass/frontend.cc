#include "frontend.h"
#include "../panic.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace flash_gpgpu_sim {
namespace sass {

uint32_t warp_state::read_register(unsigned lane, unsigned reg) const {
  if (lane >= kWarpLanes || reg >= kVectorRegisters) {
    std::ostringstream error;
    error << "SASS register read out of range (lane=" << lane
          << ", register=" << reg << ')';
    panic(error.str());
  }
  return reg == kZeroRegister ? 0 : registers[lane][reg];
}

void warp_state::write_register(unsigned lane, unsigned reg, uint32_t value) {
  if (lane >= kWarpLanes || reg >= kVectorRegisters) {
    std::ostringstream error;
    error << "SASS register write out of range (lane=" << lane
          << ", register=" << reg << ')';
    panic(error.str());
  }
  if (reg != kZeroRegister)
    registers[lane][reg] = value;
}

uint32_t warp_state::read_uniform_register(unsigned reg) const {
  if (reg >= kUniformRegisters)
    panic("SASS uniform register read out of range");
  return reg == kZeroUniformRegister ? 0 : uniform_registers[reg];
}

void warp_state::write_uniform_register(unsigned reg, uint32_t value) {
  if (reg >= kUniformRegisters)
    panic("SASS uniform register write out of range");
  if (reg != kZeroUniformRegister)
    uniform_registers[reg] = value;
}

bool warp_state::read_predicate(unsigned lane, unsigned pred) const {
  if (lane >= kWarpLanes || pred >= kPredicateRegisters)
    panic("SASS predicate read out of range");
  return pred == kTruePredicate || predicates[lane].test(pred);
}

void warp_state::write_predicate(unsigned lane, unsigned pred, bool value) {
  if (lane >= kWarpLanes || pred >= kPredicateRegisters)
    panic("SASS predicate write out of range");
  if (pred != kTruePredicate)
    predicates[lane].set(pred, value);
}

bool warp_state::read_uniform_predicate(unsigned pred) const {
  if (pred >= kPredicateRegisters)
    panic("SASS uniform predicate read out of range");
  return pred == kTruePredicate || uniform_predicates.test(pred);
}

void warp_state::write_uniform_predicate(unsigned pred, bool value) {
  if (pred >= kPredicateRegisters)
    panic("SASS uniform predicate write out of range");
  if (pred != kTruePredicate)
    uniform_predicates.set(pred, value);
}

void warp_state::record_mbarrier_try_wait(unsigned pred,
                                          uint32_t pending_lane_mask) {
  if (pred >= kPredicateRegisters)
    panic("SASS pending mbarrier predicate out of range");
  mbarrier_try_wait = pending_lane_mask == 0
                          ? mbarrier_try_wait_state::kFunctionalComplete
                          : mbarrier_try_wait_state::kFunctionalPending;
  pending_mbarrier_try_wait_predicate = pred;
  pending_mbarrier_try_wait_mask = pending_lane_mask;
}

mbarrier_try_wait_state warp_state::complete_mbarrier_try_wait() {
  const mbarrier_try_wait_state completed = mbarrier_try_wait;
  if (completed == mbarrier_try_wait_state::kFunctionalPending) {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((pending_mbarrier_try_wait_mask & (uint32_t{1} << lane)) != 0)
        write_predicate(lane, pending_mbarrier_try_wait_predicate, true);
    }
  }
  mbarrier_try_wait = mbarrier_try_wait_state::kNone;
  pending_mbarrier_try_wait_mask = 0;
  pending_mbarrier_try_wait_predicate = kTruePredicate;
  return completed;
}

cta_execution_state::cta_execution_state(unsigned warp_count)
    : warp_count_(warp_count) {
  if (warp_count == 0 || warp_count > kMaxCtaWarps)
    panic("CTA warp count must be between 1 and 32");
  expected_warps_ =
      warp_count == 32 ? 0xffffffffu : (uint32_t{1} << warp_count) - 1;
}

uint64_t cta_execution_state::generation(unsigned barrier_id) const {
  if (barrier_id >= kCtaBarrierSlots)
    panic("SASS CTA barrier id out of range");
  return barriers_[barrier_id].generation;
}

bool cta_execution_state::arrive(unsigned barrier_id, unsigned warp_id,
                                 unsigned expected_warp_count) {
  if (barrier_id >= kCtaBarrierSlots)
    panic("SASS CTA barrier id out of range");
  if (warp_id >= warp_count_)
    panic("SASS CTA warp id out of range");
  barrier_state &barrier = barriers_[barrier_id];
  const unsigned live_warps =
      std::bitset<kMaxCtaWarps>(expected_warps_).count();
  if (barrier.arrivals == 0 && barrier.expected_arrivals == 0) {
    barrier.expected_arrivals =
        expected_warp_count == 0 ? live_warps : expected_warp_count;
    barrier.tracks_live_participants = expected_warp_count == 0;
  } else if (expected_warp_count != 0) {
    if (barrier.expected_arrivals != expected_warp_count)
      panic("SASS barrier participant count changed within one generation");
  }
  // Named barriers count thread arrivals, not unique warp identities. Since
  // this frontend only accepts converged full-warp BAR operations, one issue
  // contributes one warp-equivalent arrival. A warp may legally execute a
  // nonblocking BAR.ARV and later contribute again with BAR.SYNC.
  ++barrier.arrivals;
  if (barrier.tracks_live_participants)
    barrier.arrived_warps |= uint32_t{1} << warp_id;
  if (barrier.arrivals < barrier.expected_arrivals)
    return false;
  barrier.arrivals = 0;
  barrier.expected_arrivals = 0;
  barrier.tracks_live_participants = false;
  barrier.arrived_warps = 0;
  ++barrier.generation;
  return true;
}

void cta_execution_state::retire_warp(unsigned warp_id) {
  if (warp_id >= warp_count_)
    panic("SASS CTA warp id out of range");
  const uint32_t warp_bit = uint32_t{1} << warp_id;
  if ((expected_warps_ & warp_bit) == 0)
    panic("SASS CTA warp retired twice");
  expected_warps_ &= ~warp_bit;
  for (barrier_state &barrier : barriers_) {
    if (barrier.expected_arrivals == 0)
      continue;
    if (barrier.tracks_live_participants &&
        (barrier.arrived_warps & warp_bit) == 0 &&
        barrier.expected_arrivals != 0)
      --barrier.expected_arrivals;
    if (barrier.expected_arrivals != 0 &&
        barrier.arrivals >= barrier.expected_arrivals) {
      barrier.arrivals = 0;
      barrier.expected_arrivals = 0;
      barrier.tracks_live_participants = false;
      barrier.arrived_warps = 0;
      ++barrier.generation;
    }
  }
}

bool cta_execution_state::released(unsigned barrier_id,
                                   uint64_t generation) const {
  return this->generation(barrier_id) != generation;
}

bool triggers_deferred_cta_barrier(const instruction &inst) {
  const size_t separator = inst.opcode.find('.');
  const std::string base = inst.opcode.substr(0, separator);
  return base == "LDS" || base == "STS" || base == "LDSM" || base == "STSM" ||
         base == "HGMMA" || base == "QGMMA" || base == "IGMMA" ||
         base == "BGMMA" || base == "ATOMS";
}

bool orders_deferred_cta_barrier(const instruction &inst, architecture arch) {
  if (triggers_deferred_cta_barrier(inst))
    return true;
  const size_t separator = inst.opcode.find('.');
  const std::string base = inst.opcode.substr(0, separator);
  if (base == "BAR")
    return arch == architecture::kSm90 || arch == architecture::kSm120;
  // Publishing an mbarrier arrival/completion is a shared-state side effect.
  // In particular, FA3's no-TMA path completes Q immediately after QueryEmpty;
  // letting this cross an unfinished deferred barrier can lap the consumer's
  // parity wait. Independent phase polling below remains architecture-specific.
  if (inst.opcode.rfind("SYNCS.ARRIVE", 0) == 0)
    return arch == architecture::kSm90 || arch == architecture::kSm120;
  // A TMA store/reduction consumes shared data. FA3's epilogue follows its
  // deferred O-ready barrier directly with UTMASTG, without an intervening
  // LDS/STS. Do not snapshot the reused shared buffer before its writers
  // arrive. This does not change SM90 TMA-load issue penetration.
  if (base == "UTMASTG" || base == "UTMAREDG")
    return arch == architecture::kSm90 || arch == architecture::kSm120;
  if (arch == architecture::kSm120)
    return inst.opcode.rfind("SYNCS.PHASECHK", 0) == 0 ||
           inst.opcode.rfind("UTMA", 0) == 0;
  return false;
}

bool deferred_cta_barriers_released(warp_state &warp,
                                    const cta_execution_state &cta) {
  bool all_released = true;
  for (unsigned id = 0; id < kCtaBarrierSlots; ++id) {
    if (!warp.deferred_cta_barrier_pending[id])
      continue;
    if (cta.released(id, warp.deferred_cta_barrier_generation[id])) {
      warp.deferred_cta_barrier_pending[id] = false;
      continue;
    }
    all_released = false;
  }
  return all_released;
}

bool cta_execution_state::initialize_mbarrier(uint64_t address,
                                              uint32_t expected_arrivals) {
  if (expected_arrivals == 0 || expected_arrivals >= (uint32_t{1} << 20))
    return false;
  return mbarriers_
      .emplace(address,
               mbarrier_state{expected_arrivals, expected_arrivals, 0, 0})
      .second;
}

bool cta_execution_state::mbarrier_initialized(uint64_t address) const {
  return mbarriers_.find(address) != mbarriers_.end();
}

bool cta_execution_state::arrive_mbarrier(uint64_t address, uint32_t arrivals,
                                          uint64_t transaction_bytes) {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end() || arrivals == 0 ||
      arrivals > found->second.pending_arrivals ||
      UINT64_MAX - found->second.pending_transaction_bytes < transaction_bytes)
    return false;
  mbarrier_state &barrier = found->second;
  barrier.pending_arrivals -= arrivals;
  barrier.pending_transaction_bytes += transaction_bytes;
  if (barrier.pending_arrivals == 0 && barrier.pending_transaction_bytes == 0) {
    barrier.pending_arrivals = barrier.expected_arrivals;
    ++barrier.phase;
  }
  return true;
}

bool cta_execution_state::complete_mbarrier_transaction(
    uint64_t address, uint64_t transaction_bytes) {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end() || transaction_bytes == 0 ||
      transaction_bytes > found->second.pending_transaction_bytes)
    return false;
  mbarrier_state &barrier = found->second;
  barrier.pending_transaction_bytes -= transaction_bytes;
  if (barrier.pending_arrivals == 0 && barrier.pending_transaction_bytes == 0) {
    barrier.pending_arrivals = barrier.expected_arrivals;
    ++barrier.phase;
  }
  return true;
}

bool cta_execution_state::invalidate_mbarrier(uint64_t address) {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end() ||
      found->second.pending_transaction_bytes != 0 ||
      found->second.pending_arrivals != found->second.expected_arrivals)
    return false;
  mbarriers_.erase(found);
  return true;
}

uint32_t
cta_execution_state::mbarrier_pending_arrivals(uint64_t address) const {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end())
    panic("uninitialized SASS mbarrier address");
  return found->second.pending_arrivals;
}

uint64_t cta_execution_state::mbarrier_pending_transaction_bytes(
    uint64_t address) const {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end())
    panic("uninitialized SASS mbarrier address");
  return found->second.pending_transaction_bytes;
}

uint64_t cta_execution_state::mbarrier_phase(uint64_t address) const {
  const auto found = mbarriers_.find(address);
  if (found == mbarriers_.end())
    panic("uninitialized SASS mbarrier address");
  return found->second.phase;
}

bool cta_execution_state::register_tensor_map_2d(
    uint64_t descriptor_address, functional_tensor_map_2d descriptor) {
  const bool one_dimensional =
      descriptor.global_dim[1] == 1 && descriptor.box_dim[1] == 1;
  if (descriptor.element_bytes == 0 || descriptor.global_dim[0] == 0 ||
      descriptor.global_dim[1] == 0 || descriptor.box_dim[0] == 0 ||
      descriptor.box_dim[1] == 0 ||
      (!one_dimensional && descriptor.row_stride_bytes == 0) ||
      descriptor.element_stride[0] == 0 || descriptor.element_stride[1] == 0)
    return false;
  return tensor_maps_2d_.emplace(descriptor_address, std::move(descriptor))
      .second;
}

const functional_tensor_map_2d *
cta_execution_state::find_tensor_map_2d(uint64_t descriptor_address) const {
  const auto found = tensor_maps_2d_.find(descriptor_address);
  return found == tensor_maps_2d_.end() ? nullptr : &found->second;
}

void execution_context::clear_instruction_effects() {
  for (auto &lane_accesses : memory_accesses)
    lane_accesses.clear();
  tma_effects.fill({});
  mbarrier_effects.fill({});
  named_barrier_effect = {};
}

bool execution_context::decode_local_memory_offset(unsigned lane,
                                                   uint64_t address,
                                                   uint64_t &offset) const {
  if (lane >= kWarpLanes || local_memory_thread_stride == 0)
    return false;
  const uint64_t thread = thread_linear_id[lane];
  if (thread > (UINT64_MAX - local_memory_base) / local_memory_thread_stride)
    return false;
  const uint64_t thread_base =
      local_memory_base + thread * local_memory_thread_stride;
  if (address < thread_base)
    return false;
  offset = address - thread_base;
  return offset < local_memory_thread_stride;
}

void execution_context::record_memory_access(unsigned lane, memory_space space,
                                             uint64_t address, size_t bytes,
                                             bool write) {
  if (lane >= kWarpLanes || bytes == 0)
    panic("invalid SASS functional memory access");
  memory_accesses[lane].push_back({space, address, bytes, write});
}

void execution_context::record_tma_effect(unsigned lane,
                                          functional_tma_effect effect) {
  if (lane >= kWarpLanes || !effect.valid || tma_effects[lane].valid)
    panic("invalid SASS functional TMA effect");
  tma_effects[lane] = std::move(effect);
}

void execution_context::record_mbarrier_effect(
    unsigned lane, functional_mbarrier_effect effect) {
  if (lane >= kWarpLanes || !effect.valid || mbarrier_effects[lane].valid)
    panic("invalid SASS functional mbarrier effect");
  mbarrier_effects[lane] = effect;
}

void execution_context::record_named_barrier_effect(
    functional_named_barrier_effect effect) {
  if (!effect.valid || named_barrier_effect.valid)
    panic("invalid SASS functional named-barrier effect");
  named_barrier_effect = effect;
}

void frontend::load(const decoder &source, const std::string &image_path,
                    const std::string &kernel_name) {
  add_kernel(source.decode_kernel(image_path, kernel_name));
}

void frontend::add_kernel(kernel decoded_kernel) {
  if (decoded_kernel.name.empty())
    panic("cannot add an unnamed SASS kernel");
  if ((decoded_kernel.arch != architecture::kSm90 &&
       decoded_kernel.arch != architecture::kSm120) ||
      decoded_kernel.instruction_bytes != kInstructionBytes)
    panic("frontend currently accepts only 16-byte SM90/SM120 SASS");
  std::sort(decoded_kernel.instructions.begin(),
            decoded_kernel.instructions.end(),
            [](const instruction &left, const instruction &right) {
              return left.pc < right.pc;
            });
  for (size_t i = 1; i < decoded_kernel.instructions.size(); ++i) {
    if (decoded_kernel.instructions[i - 1].pc ==
        decoded_kernel.instructions[i].pc)
      panic("duplicate pc in SASS kernel");
  }
  for (const instruction &inst : decoded_kernel.instructions) {
    if (inst.pc % decoded_kernel.instruction_bytes != 0)
      panic("unaligned pc in SASS kernel");
  }

  const std::string name = decoded_kernel.name;
  auto inserted = kernels_.emplace(name, std::move(decoded_kernel));
  if (!inserted.second)
    panic("duplicate SASS kernel: " + name);
}

const kernel *frontend::find_kernel(const std::string &kernel_name) const {
  const auto found = kernels_.find(kernel_name);
  return found == kernels_.end() ? nullptr : &found->second;
}

const instruction *frontend::fetch(const std::string &kernel_name,
                                   uint64_t pc) const {
  const kernel *image = find_kernel(kernel_name);
  if (image == nullptr)
    return nullptr;
  const auto found =
      std::lower_bound(image->instructions.begin(), image->instructions.end(),
                       pc, [](const instruction &inst, uint64_t address) {
                         return inst.pc < address;
                       });
  if (found == image->instructions.end() || found->pc != pc)
    return nullptr;
  return &*found;
}

void frontend::register_semantics(const std::string &opcode,
                                  semantic_handler handler) {
  if (opcode.empty() || !handler)
    panic("invalid SASS semantic handler");
  semantics_[opcode] = std::move(handler);
}

step_result frontend::step(const std::string &kernel_name, warp_state &state,
                           execution_context &context) const {
  if (state.waiting_at_cta_barrier) {
    if (context.cta == nullptr ||
        !context.cta->released(state.cta_barrier_id,
                               state.cta_barrier_generation))
      return {step_status::kBlocked, "warp is waiting at a CTA barrier"};
    state.waiting_at_cta_barrier = false;
  }
  const kernel *image = find_kernel(kernel_name);
  const instruction *inst = fetch(kernel_name, state.pc);
  if (inst == nullptr)
    return {step_status::kMissingPc, "no static SASS instruction at warp pc"};
  if (!inst->decoded)
    return {step_status::kUnsupported,
            "the configured SASS decoder could not decode instruction"};
  if (orders_deferred_cta_barrier(
          *inst, image == nullptr ? architecture::kUnknown : image->arch)) {
    const bool has_pending_barrier =
        std::any_of(state.deferred_cta_barrier_pending.begin(),
                    state.deferred_cta_barrier_pending.end(),
                    [](bool pending) { return pending; });
    if (has_pending_barrier &&
        (context.cta == nullptr ||
         !deferred_cta_barriers_released(state, *context.cta)))
      return {step_status::kBlocked,
              "warp consumer is waiting on a deferred CTA barrier"};
  }
  const auto handler = semantics_.find(inst->opcode);
  if (handler == semantics_.end())
    return {step_status::kUnsupported,
            "no functional semantics for SASS opcode " + inst->opcode};
  context.kernel_image = image;
  context.clear_instruction_effects();
  const panic_context diagnostic(inst->opcode, inst->pc, true);
  const step_result result = handler->second(*inst, state, context);
  if (result.status == step_status::kAdvanced ||
      result.status == step_status::kBlocked ||
      result.status == step_status::kExited)
    ++context.instructions_executed;
  return result;
}

cta_executor::cta_executor(const frontend &source, std::string kernel_name,
                           unsigned warp_count, functional_memory *memory)
    : source_(source), kernel_name_(std::move(kernel_name)), cta_(warp_count),
      warps_(warp_count), contexts_(warp_count), exited_(warp_count, false) {
  if (source_.find_kernel(kernel_name_) == nullptr)
    panic("cannot execute an unknown SASS kernel");
  for (unsigned warp_id = 0; warp_id < warp_count; ++warp_id) {
    execution_context &warp_context = contexts_[warp_id];
    warp_context.memory = memory;
    warp_context.cta = &cta_;
    warp_context.warp_id = warp_id;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      warp_context.thread_idx_x[lane] = warp_id * kWarpLanes + lane;
      warp_context.thread_linear_id[lane] = warp_id * kWarpLanes + lane;
    }
  }
}

step_result cta_executor::step() {
  const unsigned count = warp_count();
  bool any_live = false;
  for (unsigned attempt = 0; attempt < count; ++attempt) {
    const unsigned warp_id = next_warp_id_;
    next_warp_id_ = (next_warp_id_ + 1) % count;
    if (exited_[warp_id])
      continue;
    any_live = true;

    warp_state &warp_state = warps_[warp_id];
    execution_context &warp_context = contexts_[warp_id];
    if (warp_state.waiting_at_cta_barrier &&
        !cta_.released(warp_state.cta_barrier_id,
                       warp_state.cta_barrier_generation))
      continue;

    last_warp_id_ = warp_id;
    const uint64_t before = warp_context.instructions_executed;
    step_result result = source_.step(kernel_name_, warp_state, warp_context);
    instructions_executed_ += warp_context.instructions_executed - before;
    if (result.status == step_status::kBlocked)
      return {step_status::kAdvanced, {}};
    if (result.status == step_status::kExited) {
      cta_.retire_warp(warp_id);
      exited_[warp_id] = true;
      if (std::all_of(exited_.begin(), exited_.end(),
                      [](bool exited) { return exited; }))
        return result;
      return {step_status::kAdvanced, {}};
    }
    if (result.status == step_status::kUnsupported ||
        result.status == step_status::kMissingPc) {
      std::ostringstream detail;
      detail << "warp " << warp_id << ": " << result.detail;
      result.detail = detail.str();
    }
    return result;
  }

  if (!any_live)
    return {step_status::kExited, {}};
  return {step_status::kUnsupported,
          "CTA functional execution deadlocked at named barriers"};
}

warp_state &cta_executor::warp(unsigned warp_id) {
  if (warp_id >= warp_count())
    panic("SASS CTA warp id out of range");
  return warps_[warp_id];
}

const warp_state &cta_executor::warp(unsigned warp_id) const {
  if (warp_id >= warp_count())
    panic("SASS CTA warp id out of range");
  return warps_[warp_id];
}

execution_context &cta_executor::context(unsigned warp_id) {
  if (warp_id >= warp_count())
    panic("SASS CTA warp id out of range");
  return contexts_[warp_id];
}

const execution_context &cta_executor::context(unsigned warp_id) const {
  if (warp_id >= warp_count())
    panic("SASS CTA warp id out of range");
  return contexts_[warp_id];
}

} // namespace sass
} // namespace flash_gpgpu_sim
