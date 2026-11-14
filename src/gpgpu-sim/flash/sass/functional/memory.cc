#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.9-9.7.12: memory, atomic, and functional fences.

bool evaluate_constant_address(const address_expression &address,
                               const warp_state &state, unsigned lane,
                               uint64_t &value) {
  value = evaluate_address_terms(address, state, lane);
  return !address.terms.empty();
}

step_result execute_lds(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one typed shared-load operand");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported LDS operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    std::array<uint32_t, 4> value{};
    const bool vector128 = inst.opcode == "LDS.128";
    const bool wide = inst.opcode == "LDS.64";
    const bool u8 = inst.opcode == "LDS.U8";
    const size_t bytes = vector128                  ? 16
                         : wide                     ? 8
                         : u8                       ? 1
                         : inst.opcode == "LDS.U16" ? 2
                                                    : 4;
    if (!evaluate_shared_address(*address, state, lane, location))
      return unsupported(inst, "unsupported shared-load address");
    context.record_memory_access(lane, memory_space::kShared, location, bytes,
                                 false);
    if (!context.memory->read(memory_space::kShared, location, &value, bytes))
      return memory_fault(inst, memory_space::kShared, location, bytes);
    const unsigned words = vector128 ? 4 : wide ? 2 : 1;
    for (unsigned word = 0; word < words; ++word)
      state.write_register(lane, address->prefix_index + word, value[word]);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_sts(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed shared-store operands");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *source = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || address->has_prefix || source == nullptr)
    return unsupported(inst, "unsupported STS operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  const bool vector128 = inst.opcode == "STS.128";
  const bool wide = inst.opcode == "STS.64";
  const bool u8 = inst.opcode == "STS.U8";
  const bool u16 = inst.opcode == "STS.U16";
  const size_t bytes = vector128 ? 16 : wide ? 8 : u8 ? 1 : u16 ? 2 : 4;
  if (vector128 && source->index != kZeroRegister &&
      source->index > kZeroRegister - 3)
    return unsupported(inst, "STS.128 register range is out of bounds");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location))
      return unsupported(inst, "unsupported shared-store address");
    context.record_memory_access(lane, memory_space::kShared, location, bytes,
                                 true);
    if (vector128) {
      std::array<uint32_t, 4> value{};
      for (unsigned word = 0; word < value.size(); ++word)
        value[word] = source->index == kZeroRegister
                          ? 0
                          : state.read_register(lane, source->index + word);
      if (!context.memory->write(memory_space::kShared, location, value.data(),
                                 sizeof(value)))
        return memory_fault(inst, memory_space::kShared, location,
                            sizeof(value));
    } else if (wide) {
      const uint64_t value = read_register_pair(state, lane, source->index);
      if (!context.memory->write(memory_space::kShared, location, &value,
                                 sizeof(value)))
        return memory_fault(inst, memory_space::kShared, location,
                            sizeof(value));
    } else if (u8) {
      const uint8_t value =
          static_cast<uint8_t>(state.read_register(lane, source->index));
      if (!context.memory->write(memory_space::kShared, location, &value,
                                 sizeof(value)))
        return memory_fault(inst, memory_space::kShared, location,
                            sizeof(value));
    } else if (u16) {
      const uint16_t value =
          static_cast<uint16_t>(state.read_register(lane, source->index));
      if (!context.memory->write(memory_space::kShared, location, &value,
                                 sizeof(value)))
        return memory_fault(inst, memory_space::kShared, location,
                            sizeof(value));
    } else {
      const uint32_t value = state.read_register(lane, source->index);
      if (!context.memory->write(memory_space::kShared, location, &value,
                                 sizeof(value)))
        return memory_fault(inst, memory_space::kShared, location,
                            sizeof(value));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

bool evaluate_local_address(const operand &address, const warp_state &state,
                            const execution_context &context, unsigned lane,
                            uint64_t &location) {
  uint64_t virtual_address = 0;
  if (context.local_memory_thread_stride == 0 ||
      !evaluate_address(address, state, lane, virtual_address))
    return false;
  const uint64_t thread = context.thread_linear_id[lane];
  if (thread > (UINT64_MAX - context.local_memory_base) /
                   context.local_memory_thread_stride)
    return false;
  const uint64_t thread_base =
      context.local_memory_base + thread * context.local_memory_thread_stride;
  if (virtual_address > UINT64_MAX - thread_base)
    return false;
  location = thread_base + virtual_address;
  return true;
}

step_result execute_ldl(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one typed local-load operand");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kRegister)
    return unsupported(inst, "unsupported LDL.LU operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  const bool wide = inst.opcode == "LDL.LU.64" || inst.opcode == "LDL.64";
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    uint64_t value = 0;
    if (!evaluate_local_address(*address, state, context, lane, location))
      return unsupported(inst, "unsupported local-load address");
    const size_t bytes = wide ? sizeof(value) : sizeof(uint32_t);
    context.record_memory_access(lane, memory_space::kLocal, location, bytes,
                                 false);
    if (!context.memory->read(memory_space::kLocal, location, &value, bytes))
      return memory_fault(inst, memory_space::kLocal, location, bytes);
    if (wide)
      write_register_pair(state, lane, address->prefix_index, value);
    else
      state.write_register(lane, address->prefix_index,
                           static_cast<uint32_t>(value));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_stl(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed local-store operands");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *source = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || address->has_prefix || source == nullptr)
    return unsupported(inst, "unsupported STL operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  const bool wide = inst.opcode == "STL.64";
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_local_address(*address, state, context, lane, location))
      return unsupported(inst, "unsupported local-store address");
    const uint64_t value = wide ? read_register_pair(state, lane, source->index)
                                : state.read_register(lane, source->index);
    const size_t bytes = wide ? sizeof(uint64_t) : sizeof(uint32_t);
    context.record_memory_access(lane, memory_space::kLocal, location, bytes,
                                 true);
    if (!context.memory->write(memory_space::kLocal, location, &value, bytes))
      return memory_fault(inst, memory_space::kLocal, location, bytes);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ldc(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *source = get_operand(inst, 1, operand_kind::kConstantMemory);
  if (destination == nullptr || source == nullptr)
    return unsupported(inst, "unsupported constant-memory operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  const bool wide = inst.opcode == "LDC.64";
  const bool u8 = inst.opcode == "LDC.U8";
  const bool u16 = inst.opcode == "LDC.U16";
  const size_t bytes = wide ? 8 : u8 ? 1 : u16 ? 2 : 4;
  std::unordered_map<uint64_t, uint64_t> values;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t offset = 0;
    if (!evaluate_constant_address(source->constant_address, state, lane,
                                   offset))
      return unsupported(inst, "unsupported constant-load address");
    auto [loaded, inserted] = values.try_emplace(offset, 0);
    if (inserted && !context.memory->read_constant(
                        source->constant_bank, offset, &loaded->second, bytes))
      return memory_fault(inst, memory_space::kConstant, offset, bytes);
    const uint64_t value = loaded->second;
    context.record_memory_access(lane, memory_space::kConstant, offset, bytes,
                                 false);
    state.write_register(lane, destination->index,
                         static_cast<uint32_t>(value));
    if (wide)
      state.write_register(lane, destination->index + 1,
                           static_cast<uint32_t>(value >> 32));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ldcu(const instruction &inst, warp_state &state,
                         execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination =
      get_operand(inst, 0, operand_kind::kUniformRegister);
  const operand *source = get_operand(inst, 1, operand_kind::kConstantMemory);
  if (destination == nullptr || source == nullptr)
    return unsupported(inst, "unsupported uniform constant-load form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  if (any_lane_executes(inst, state)) {
    const bool wide = inst.opcode == "LDCU.64" || inst.opcode == "ULDC.64";
    const bool vector128 =
        inst.opcode == "LDCU.128" || inst.opcode == "ULDC.128";
    const bool byte = inst.opcode == "LDCU.U8" || inst.opcode == "ULDC.U8";
    const size_t bytes = vector128 ? 16 : wide ? 8 : byte ? 1 : 4;
    uint64_t offset = 0;
    if (!evaluate_constant_address(source->constant_address, state, 0, offset))
      return unsupported(inst, "unsupported uniform constant-load address");
    std::array<uint32_t, 4> value{};
    if (!context.memory->read_constant(source->constant_bank, offset, &value,
                                       bytes))
      return memory_fault(inst, memory_space::kConstant, offset, bytes);
    const unsigned words = vector128 ? 4 : wide ? 2 : 1;
    for (unsigned word = 0; word < words; ++word)
      state.write_uniform_register(destination->index + word, value[word]);
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      if ((state.active_mask & (1u << lane)) != 0 &&
          lane_executes(inst, state, lane))
        context.record_memory_access(lane, memory_space::kConstant, offset,
                                     bytes, false);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ldg(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *address =
      get_operand(inst, 1, operand_kind::kDescriptorAddress);
  if (destination == nullptr || address == nullptr)
    return unsupported(inst, "unsupported global-load operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  (void)state.read_uniform_register(address->descriptor_register);
  const bool u8 = inst.opcode == "LDG.E.U8";
  const bool u16 =
      inst.opcode == "LDG.E.U16" || inst.opcode == "LDG.E.U16.CONSTANT";
  const bool wide = inst.opcode == "LDG.E.64";
  const bool vector128 =
      inst.opcode == "LDG.E.128" || inst.opcode == "LDG.E.128.CONSTANT";
  if (wide && destination->index != kZeroRegister &&
      destination->index > kZeroRegister - 1)
    return unsupported(inst, "LDG.E.64 register range is out of bounds");
  if (vector128 && destination->index != kZeroRegister &&
      destination->index > kZeroRegister - 3)
    return unsupported(inst, "LDG.E.128 register range is out of bounds");
  const size_t bytes = vector128 ? 16 : wide ? 8 : u16 ? 2 : u8 ? 1 : 4;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint64_t location =
        read_register_pair(state, lane, address->address_register) +
        static_cast<uint64_t>(address->address_offset);
    context.record_memory_access(lane, memory_space::kGlobal, location, bytes,
                                 false);
    std::array<uint32_t, 4> value{};
    if (!context.memory->read(memory_space::kGlobal, location, value.data(),
                              bytes))
      return memory_fault(inst, memory_space::kGlobal, location, bytes);
    const unsigned words = vector128 ? 4 : wide ? 2 : 1;
    if (destination->index != kZeroRegister)
      for (unsigned word = 0; word < words; ++word)
        state.write_register(lane, destination->index + word, value[word]);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_stg(const instruction &inst, warp_state &state,
                        execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *address =
      get_operand(inst, 0, operand_kind::kDescriptorAddress);
  const operand *source = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || source == nullptr)
    return unsupported(inst, "unsupported global-store operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  (void)state.read_uniform_register(address->descriptor_register);
  const bool vector128 = inst.opcode == "STG.E.128";
  const bool wide = inst.opcode == "STG.E.64";
  const bool u16 = inst.opcode == "STG.E.U16";
  if (vector128 && source->index != kZeroRegister &&
      source->index > kZeroRegister - 3)
    return unsupported(inst, "STG.E.128 register range is out of bounds");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    const uint64_t location =
        read_register_pair(state, lane, address->address_register) +
        static_cast<uint64_t>(address->address_offset);
    std::array<uint32_t, 4> value{};
    const unsigned words = vector128 ? 4 : wide ? 2 : 1;
    for (unsigned word = 0; word < words; ++word)
      value[word] = source->index == kZeroRegister
                        ? 0
                        : state.read_register(lane, source->index + word);
    const size_t bytes = u16 ? 2 : words * sizeof(uint32_t);
    context.record_memory_access(lane, memory_space::kGlobal, location, bytes,
                                 true);
    if (!context.memory->write(memory_space::kGlobal, location, value.data(),
                               bytes))
      return memory_fault(inst, memory_space::kGlobal, location, bytes);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

bool evaluate_generic_descriptor_address(const operand &address,
                                         const warp_state &state, unsigned lane,
                                         memory_space &space,
                                         uint64_t &location) {
  if (address.kind != operand_kind::kDescriptorAddress ||
      address.address_register >= kVectorRegisters - 1)
    return false;

  const uint32_t descriptor =
      state.read_uniform_register(address.descriptor_register);
  const uint32_t high = state.read_register(lane, address.address_register + 1);
  const uint64_t base =
      read_register_pair(state, lane, address.address_register);
  if (address.address_offset < 0) {
    const uint64_t magnitude =
        static_cast<uint64_t>(-(address.address_offset + 1)) + 1;
    if (base < magnitude)
      return false;
    location = base - magnitude;
  } else {
    const uint64_t offset = static_cast<uint64_t>(address.address_offset);
    if (base > std::numeric_limits<uint64_t>::max() - offset)
      return false;
    location = base + offset;
  }

  // Generic addresses with a null descriptor and a zero high word use the
  // flat shared-memory window.  Compiler-generated global accesses carry the
  // launch descriptor in UR and/or a full 64-bit pointer in the address pair.
  space = descriptor == 0 && high == 0 ? memory_space::kShared
                                       : memory_space::kGlobal;
  return space != memory_space::kShared ||
         location <= std::numeric_limits<uint32_t>::max();
}

step_result execute_generic_memory(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  if (!has_typed_operands(inst, 2) || context.memory == nullptr)
    return unsupported(inst, "invalid generic memory operation");
  const bool load = inst.opcode.rfind("LD.E", 0) == 0;
  const bool vector128 = inst.opcode.find(".128") != std::string::npos;
  const bool u16 = inst.opcode.find(".U16") != std::string::npos;
  const bool u8 = inst.opcode.find(".U8") != std::string::npos;
  const operand *data =
      get_operand(inst, load ? 0 : 1, operand_kind::kRegister);
  const operand *address =
      get_operand(inst, load ? 1 : 0, operand_kind::kDescriptorAddress);
  if (data == nullptr || address == nullptr ||
      address->address_register >= kVectorRegisters - 1)
    return unsupported(inst, "unsupported generic-memory operands");
  if (vector128 && data->index != kZeroRegister &&
      data->index > kZeroRegister - 3)
    return unsupported(inst, "generic .128 register range is out of bounds");

  const unsigned words = vector128 ? 4 : 1;
  const size_t bytes = u16 ? 2 : u8 ? 1 : words * sizeof(uint32_t);

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    memory_space space = memory_space::kShared;
    if (!evaluate_generic_descriptor_address(*address, state, lane, space,
                                             location))
      return unsupported(inst, "invalid generic address");
    context.record_memory_access(lane, space, location, bytes, !load);
    std::array<uint32_t, 4> value{};
    if (load) {
      if (!context.memory->read(space, location, value.data(), bytes))
        return memory_fault(inst, space, location, bytes);
      if (data->index != kZeroRegister)
        for (unsigned word = 0; word < words; ++word)
          state.write_register(lane, data->index + word, value[word]);
    } else {
      for (unsigned word = 0; word < words; ++word)
        value[word] = data->index == kZeroRegister
                          ? 0
                          : state.read_register(lane, data->index + word);
      if (!context.memory->write(space, location, value.data(), bytes))
        return memory_fault(inst, space, location, bytes);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ldgsts(const instruction &inst, warp_state &state,
                           execution_context &context) {
  if (!has_typed_operands(inst, 2) && !has_typed_operands(inst, 3))
    return unsupported(inst, "expected shared and global address operands");
  const operand *destination =
      get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *source =
      get_operand(inst, 1, operand_kind::kDescriptorAddress);
  if (destination == nullptr || destination->has_prefix || source == nullptr)
    return unsupported(inst, "unsupported LDGSTS operand form");
  const operand *copy_enable =
      inst.operands.size() == 3 ? get_operand(inst, 2, operand_kind::kPredicate)
                                : nullptr;
  if (inst.operands.size() == 3 && copy_enable == nullptr)
    return unsupported(inst, "LDGSTS copy enable must be a predicate");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  const bool zero_fill = inst.opcode.find(".ZFILL") != std::string::npos;
  (void)state.read_uniform_register(source->descriptor_register);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t shared_address = 0;
    if (!evaluate_shared_address(*destination, state, lane, shared_address))
      return unsupported(inst, "unsupported LDGSTS shared address");
    const uint64_t global_address =
        read_register_pair(state, lane, source->address_register) +
        static_cast<uint64_t>(source->address_offset);
    std::array<uint32_t, 4> value{};
    bool enabled = true;
    if (copy_enable != nullptr &&
        !read_predicate_operand(*copy_enable, state, lane, enabled))
      return unsupported(inst, "unsupported LDGSTS copy-enable predicate");
    if (enabled) {
      const uint64_t source_offset = zero_fill ? global_address & 0xfu : 0;
      const uint64_t aligned_address = global_address - source_offset;
      const size_t bytes = sizeof(value) - source_offset;
      if (!context.memory->read(memory_space::kGlobal, aligned_address,
                                value.data(), bytes))
        return memory_fault(inst, memory_space::kGlobal, aligned_address,
                            bytes);
      context.record_memory_access(lane, memory_space::kGlobal, aligned_address,
                                   bytes, false);
    }
    if (!context.memory->write(memory_space::kShared, shared_address,
                               value.data(), sizeof(value)))
      return memory_fault(inst, memory_space::kShared, shared_address,
                          sizeof(value));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_atomg_exchange(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  if (!has_typed_operands(inst, 3) && !has_typed_operands(inst, 4))
    return unsupported(inst, "expected three or four typed operands");
  const operand *predicate = get_operand(inst, 0, operand_kind::kPredicate);
  const bool descriptor_form = inst.operands.size() == 4;
  const operand *destination =
      descriptor_form ? get_operand(inst, 1, operand_kind::kRegister) : nullptr;
  const operand *memory_address =
      descriptor_form ? nullptr
                      : get_operand(inst, 1, operand_kind::kMemoryAddress);
  const operand *descriptor_address =
      descriptor_form ? get_operand(inst, 2, operand_kind::kDescriptorAddress)
                      : nullptr;
  const operand *source =
      get_operand(inst, descriptor_form ? 3 : 2, operand_kind::kRegister);
  if (predicate == nullptr || predicate->index != kTruePredicate ||
      predicate->negated || source == nullptr ||
      (descriptor_form &&
       (destination == nullptr || destination->index != kZeroRegister ||
        descriptor_address == nullptr ||
        descriptor_address->address_register >= kVectorRegisters - 1)) ||
      (!descriptor_form &&
       (memory_address == nullptr || !memory_address->has_prefix ||
        memory_address->prefix_kind != operand_kind::kRegister ||
        memory_address->prefix_index != kZeroRegister)))
    return unsupported(inst, "only discarded ATOMG exchange outputs work");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  if (descriptor_form)
    (void)state.read_uniform_register(descriptor_address->descriptor_register);
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (descriptor_form) {
      location = read_register_pair(state, lane,
                                    descriptor_address->address_register) +
                 static_cast<uint64_t>(descriptor_address->address_offset);
    } else {
      if (memory_address->address_groups.size() != 1 ||
          memory_address->address_groups[0].terms.empty())
        return unsupported(inst, "unsupported ATOMG address");
      const auto &terms = memory_address->address_groups[0].terms;
      const address_term &base = terms.front();
      if (base.kind != address_term_kind::kRegister || base.negated ||
          base.index >= kVectorRegisters - 1)
        return unsupported(inst, "unsupported ATOMG address base");
      // The ATOMG address base is a register pair. nvdisasm prints the .64
      // suffix for most forms, but omits it for the form emitted by the SM120
      // TMA mbarrier sanity kernel. Preserve any following uniform/immediate
      // byte offset instead of requiring the address to contain only the base.
      location = read_register_pair(state, lane, base.index);
      for (auto term = terms.begin() + 1; term != terms.end(); ++term) {
        uint64_t component = 0;
        if (term->kind == address_term_kind::kUniformRegister) {
          if (term->index >= kUniformRegisters - 1)
            return unsupported(inst, "unsupported ATOMG uniform offset");
          // Like the vector base, the uniform address component is an implicit
          // pair even though nvdisasm does not print a .64 suffix.
          component = read_uniform_register_pair(state, term->index);
        } else if (term->kind == address_term_kind::kImmediate) {
          component = static_cast<uint64_t>(term->immediate);
        } else {
          return unsupported(inst, "unsupported ATOMG address offset");
        }
        location += term->negated ? uint64_t{0} - component : component;
      }
    }
    uint32_t discarded = 0;
    const uint32_t value = state.read_register(lane, source->index);
    context.record_memory_access(lane, memory_space::kGlobal, location,
                                 sizeof(value), false);
    std::lock_guard<std::mutex> lock(global_atomic_mutex);
    if (!context.memory->read(memory_space::kGlobal, location, &discarded,
                              sizeof(discarded)) ||
        !context.memory->write(memory_space::kGlobal, location, &value,
                               sizeof(value)))
      return memory_fault(inst, memory_space::kGlobal, location,
                          sizeof(uint32_t));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_atomg_add(const instruction &inst, warp_state &state,
                              execution_context &context) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected predicate, old value, address, source");
  const operand *predicate = get_operand(inst, 0, operand_kind::kPredicate);
  const operand *destination = get_operand(inst, 1, operand_kind::kRegister);
  const operand *address =
      get_operand(inst, 2, operand_kind::kDescriptorAddress);
  const operand *source = get_operand(inst, 3, operand_kind::kRegister);
  if (predicate == nullptr || predicate->index != kTruePredicate ||
      predicate->negated || destination == nullptr || address == nullptr ||
      source == nullptr || address->address_register >= kVectorRegisters - 1)
    return unsupported(inst, "unsupported ATOMG.ADD operand form");
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
    if ((location % alignof(uint32_t)) != 0)
      return unsupported(inst, "ATOMG.ADD address must be 4-byte aligned");
    std::lock_guard<std::mutex> lock(global_atomic_mutex);
    uint32_t old_value = 0;
    if (!context.memory->read(memory_space::kGlobal, location, &old_value,
                              sizeof(old_value)))
      return memory_fault(inst, memory_space::kGlobal, location,
                          sizeof(old_value));
    const uint32_t new_value =
        old_value + state.read_register(lane, source->index);
    context.record_memory_access(lane, memory_space::kGlobal, location,
                                 sizeof(new_value), false);
    if (!context.memory->write(memory_space::kGlobal, location, &new_value,
                               sizeof(new_value)))
      return memory_fault(inst, memory_space::kGlobal, location,
                          sizeof(new_value));
    state.write_register(lane, destination->index, old_value);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_atoms_popc_increment(const instruction &inst,
                                         warp_state &state,
                                         execution_context &context) {
  if (!has_typed_operands(inst, 1))
    return unsupported(inst, "expected one shared atomic address");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kRegister ||
      address->prefix_index != kZeroRegister)
    return unsupported(inst, "only discarded ATOMS.POPC outputs work");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");

  std::unordered_map<uint64_t, uint32_t> increments;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location) ||
        (location % alignof(uint32_t)) != 0)
      return unsupported(inst, "invalid ATOMS.POPC shared address");
    ++increments[location];
  }
  for (const auto &increment : increments) {
    uint32_t value = 0;
    if (!context.memory->read(memory_space::kShared, increment.first, &value,
                              sizeof(value)))
      return memory_fault(inst, memory_space::kShared, increment.first,
                          sizeof(value));
    value += increment.second;
    if (!context.memory->write(memory_space::kShared, increment.first, &value,
                               sizeof(value)))
      return memory_fault(inst, memory_space::kShared, increment.first,
                          sizeof(value));
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_atoms_add(const instruction &inst, warp_state &state,
                              execution_context &context) {
  if (!has_typed_operands(inst, 2))
    return unsupported(inst, "expected shared atomic address and source");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *source = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || !address->has_prefix ||
      address->prefix_kind != operand_kind::kRegister || source == nullptr)
    return unsupported(inst, "unsupported ATOMS.ADD operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (!evaluate_shared_address(*address, state, lane, location) ||
        (location % alignof(uint32_t)) != 0)
      return unsupported(inst, "invalid ATOMS.ADD shared address");
    uint32_t value = 0;
    if (!context.memory->read(memory_space::kShared, location, &value,
                              sizeof(value)))
      return memory_fault(inst, memory_space::kShared, location, sizeof(value));
    const uint32_t old_value = value;
    value += state.read_register(lane, source->index);
    if (!context.memory->write(memory_space::kShared, location, &value,
                               sizeof(value)))
      return memory_fault(inst, memory_space::kShared, location, sizeof(value));
    // Capture both address and addend before writing the returned old value:
    // its register may alias either source. RZ naturally discards the result.
    state.write_register(lane, address->prefix_index, old_value);
    context.record_memory_access(lane, memory_space::kShared, location,
                                 sizeof(value), false);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_functional_fence(const instruction &inst, warp_state &state,
                                     execution_context &) {
  bool valid = false;
  if (inst.opcode == "DEPBAR")
    valid = has_typed_operands(inst, 1) &&
            inst.operands[0].kind == operand_kind::kRegisterSet;
  else if (inst.opcode == "DEPBAR.LE")
    valid = has_typed_operands(inst, 2) &&
            inst.operands[0].kind == operand_kind::kScoreboardRegister &&
            inst.operands[1].kind == operand_kind::kImmediate;
  else if (inst.opcode == "CCTL.E.C.LDCU.IV.DEEP" ||
           inst.opcode == "UTMACCTL.IV" || inst.opcode == "UTMACCTL.PF")
    valid = has_typed_operands(inst, 1) &&
            inst.operands[0].kind == operand_kind::kMemoryAddress;
  else if (inst.opcode == "CCTL.IVALL")
    valid = has_typed_operands(inst, 0);
  else if (inst.opcode == "FENCE.VIEW.ASYNC.S")
    valid = has_typed_operands(inst, 0);
  else if (inst.opcode == "MEMBAR.ALL.CTA" || inst.opcode == "MEMBAR.ALL.GPU" ||
           inst.opcode == "MEMBAR.SC.CTA")
    valid = has_typed_operands(inst, 0);
  else if (inst.opcode == "LDGDEPBAR")
    valid = has_typed_operands(inst, 0);
  else if (inst.opcode == "ACQBULK")
    valid = has_typed_operands(inst, 0);
  if (!valid)
    return unsupported(inst, "unsupported functional fence operand form");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_memory_semantics(frontend &target) {
  target.register_semantics("LDS", execute_lds);
  target.register_semantics("LDS.U8", execute_lds);
  target.register_semantics("LDS.U16", execute_lds);
  target.register_semantics("LDS.64", execute_lds);
  target.register_semantics("LDS.128", execute_lds);
  target.register_semantics("STS", execute_sts);
  target.register_semantics("STS.U8", execute_sts);
  target.register_semantics("STS.U16", execute_sts);
  target.register_semantics("STS.64", execute_sts);
  target.register_semantics("STS.128", execute_sts);
  target.register_semantics("LDL", execute_ldl);
  target.register_semantics("LDL.64", execute_ldl);
  target.register_semantics("LDL.LU", execute_ldl);
  target.register_semantics("LDL.LU.64", execute_ldl);
  target.register_semantics("STL", execute_stl);
  target.register_semantics("STL.64", execute_stl);
  target.register_semantics("LDC", execute_ldc);
  target.register_semantics("LDC.U8", execute_ldc);
  target.register_semantics("LDC.U16", execute_ldc);
  target.register_semantics("LDC.64", execute_ldc);
  target.register_semantics("LDCU", execute_ldcu);
  target.register_semantics("LDCU.64", execute_ldcu);
  target.register_semantics("LDCU.128", execute_ldcu);
  target.register_semantics("LDCU.U8", execute_ldcu);
  target.register_semantics("ULDC", execute_ldcu);
  target.register_semantics("ULDC.64", execute_ldcu);
  target.register_semantics("ULDC.128", execute_ldcu);
  target.register_semantics("ULDC.U8", execute_ldcu);
  target.register_semantics("LDG.E", execute_ldg);
  target.register_semantics("LDG.E.STRONG.GPU", execute_ldg);
  target.register_semantics("LDG.E.CONSTANT", execute_ldg);
  target.register_semantics("LDG.E.U8", execute_ldg);
  target.register_semantics("LDG.E.U16", execute_ldg);
  target.register_semantics("LDG.E.U16.CONSTANT", execute_ldg);
  target.register_semantics("LDG.E.64", execute_ldg);
  target.register_semantics("LDG.E.128", execute_ldg);
  target.register_semantics("LDG.E.128.CONSTANT", execute_ldg);
  target.register_semantics("STG.E", execute_stg);
  target.register_semantics("STG.E.U16", execute_stg);
  target.register_semantics("STG.E.64", execute_stg);
  target.register_semantics("STG.E.128", execute_stg);
  target.register_semantics("LD.E", execute_generic_memory);
  target.register_semantics("LD.E.U16.STRONG.SYS", execute_generic_memory);
  target.register_semantics("LD.E.128", execute_generic_memory);
  target.register_semantics("ST.E", execute_generic_memory);
  target.register_semantics("ST.E.128", execute_generic_memory);
  target.register_semantics("LDGSTS.E.BYPASS.LTC128B.128", execute_ldgsts);
  target.register_semantics("LDGSTS.E.BYPASS.128", execute_ldgsts);
  target.register_semantics("LDGSTS.E.BYPASS.128.ZFILL", execute_ldgsts);
  target.register_semantics("ATOMG.E.EXCH.STRONG.GPU", execute_atomg_exchange);
  target.register_semantics("ATOMG.E.ADD.STRONG.GPU", execute_atomg_add);
  target.register_semantics("ATOMS.POPC.INC.32", execute_atoms_popc_increment);
  target.register_semantics("ATOMS.ADD", execute_atoms_add);
  target.register_semantics("DEPBAR", execute_functional_fence);
  target.register_semantics("DEPBAR.LE", execute_functional_fence);
  target.register_semantics("CCTL.E.C.LDCU.IV.DEEP", execute_functional_fence);
  target.register_semantics("CCTL.IVALL", execute_functional_fence);
  target.register_semantics("UTMACCTL.IV", execute_functional_fence);
  target.register_semantics("UTMACCTL.PF", execute_functional_fence);
  target.register_semantics("FENCE.VIEW.ASYNC.S", execute_functional_fence);
  target.register_semantics("MEMBAR.ALL.CTA", execute_functional_fence);
  target.register_semantics("MEMBAR.ALL.GPU", execute_functional_fence);
  target.register_semantics("MEMBAR.SC.CTA", execute_functional_fence);
  target.register_semantics("LDGDEPBAR", execute_functional_fence);
  target.register_semantics("ACQBULK", execute_functional_fence);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
