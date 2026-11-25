#include "../decode/operand_parser.h"
#include "internal.h"
#include <array>
#include <limits>
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {
step_result execute_global256(const instruction &inst, warp_state &state,
                              execution_context &context) {
  const bool store = inst.opcode.rfind("STG.", 0) == 0;
  if (!has_typed_operands(inst, 3) || !context.memory)
    return unsupported(inst,
                       "global256 requires two groups, address and memory");
  const auto *address =
      get_operand(inst, store ? 0 : 2, operand_kind::kDescriptorAddress);
  const auto *first = get_operand(inst, store ? 1 : 0, operand_kind::kRegister);
  const auto *second =
      get_operand(inst, store ? 2 : 1, operand_kind::kRegister);
  if (!address || !first || !second)
    return unsupported(inst, "invalid global256 operands");
  for (const auto *group : {first, second})
    if (group->negated || group->absolute || group->bitwise_complement ||
        group->wide || group->half_selection != operand_half_selection::kNone ||
        group->matrix_layout != operand_matrix_layout::kNone ||
        (group->index != kZeroRegister && group->index > kZeroRegister - 4))
      return unsupported(inst, "invalid global256 register group");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    const uint64_t location =
        read_register_pair(state, lane, address->address_register) +
        static_cast<uint64_t>(address->address_offset);
    if (location % 32)
      return unsupported(inst, "global256 requires 32-byte alignment");
    std::array<uint32_t, 8> values{};
    // LDG prints its upper group first; STG prints its lower group first.
    const unsigned low = store ? first->index : second->index;
    const unsigned high = store ? second->index : first->index;
    if (store)
      for (unsigned word = 0; word < 8; ++word) {
        const unsigned base = word < 4 ? low : high;
        values[word] = base == kZeroRegister
                           ? 0
                           : state.read_register(lane, base + word % 4);
      }
    context.record_memory_access(lane, memory_space::kGlobal, location, 32,
                                 store);
    const bool ok = store ? context.memory->write(memory_space::kGlobal,
                                                  location, values.data(), 32)
                          : context.memory->read(memory_space::kGlobal,
                                                 location, values.data(), 32);
    if (!ok)
      return memory_fault(inst, memory_space::kGlobal, location, 32);
    if (!store)
      for (unsigned word = 0; word < 8; ++word) {
        const unsigned base = word < 4 ? low : high;
        if (base != kZeroRegister)
          state.write_register(lane, base + word % 4, values[word]);
      }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ldg(const instruction &native, warp_state &state,
                        execution_context &context) {
  if (native.opcode.find(".256") != std::string::npos)
    return execute_global256(native, state, context);
  instruction inst = native;
  if (!normalize_ldg_operands(inst) || !has_typed_operands(inst, 2))
    return unsupported(inst, "invalid global-load operands");
  const auto *destination = get_operand(inst, 0, operand_kind::kRegister);
  const auto *address = get_operand(inst, 1, operand_kind::kDescriptorAddress);
  const bool direct = inst.operands[1].kind == operand_kind::kMemoryAddress;
  if (direct)
    address = &inst.operands[1];
  if (!destination || !address || !context.memory)
    return unsupported(inst, "unsupported global-load operands or memory");
  const unsigned words = inst.opcode.find(".128") != std::string::npos  ? 4
                         : inst.opcode.find(".64") != std::string::npos ? 2
                                                                        : 1;
  const bool signed_byte = inst.opcode == "LDG.E.S8";
  const bool byte = signed_byte || inst.opcode == "LDG.E.U8";
  const bool half = inst.opcode.find(".U16") != std::string::npos;
  const size_t bytes = byte ? 1 : half ? 2 : words * 4;
  if (destination->index != kZeroRegister &&
      destination->index > kZeroRegister - words)
    return unsupported(inst, "global-load register range is out of bounds");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!(state.active_mask & (1u << lane)) ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (direct) {
      if (!evaluate_address(*address, state, lane, location))
        return unsupported(inst, "invalid direct global-load address");
    } else {
      location = read_register_pair(state, lane, address->address_register) +
                 static_cast<uint64_t>(address->address_offset);
    }
    context.record_memory_access(lane, memory_space::kGlobal, location, bytes,
                                 false);
    std::array<uint32_t, 4> value{};
    if (!context.memory->read(memory_space::kGlobal, location, value.data(),
                              bytes))
      return memory_fault(inst, memory_space::kGlobal, location, bytes);
    if (signed_byte && (value[0] & 0x80))
      value[0] |= 0xffffff00u;
    if (destination->index != kZeroRegister)
      for (unsigned word = 0; word < words; ++word)
        state.write_register(lane, destination->index + word, value[word]);
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_stg(const instruction &native, warp_state &state,
                        execution_context &context) {
  instruction inst = native;
  if (inst.opcode.find(".256") != std::string::npos)
    return execute_global256(inst, state, context);
  if (!normalize_stg_operands(inst) || !has_typed_operands(inst, 2))
    return unsupported(inst, "expected two typed operands");
  const operand *address =
      get_operand(inst, 0, operand_kind::kDescriptorAddress);
  const bool direct = inst.operands[0].kind == operand_kind::kMemoryAddress;
  if (direct)
    address = &inst.operands[0];
  const operand *source = get_operand(inst, 1, operand_kind::kRegister);
  if (address == nullptr || source == nullptr)
    return unsupported(inst, "unsupported global-store operand form");
  if (context.memory == nullptr)
    return unsupported(inst, "no functional memory is attached");
  if (!direct)
    (void)state.read_uniform_register(address->descriptor_register);
  const bool vector128 =
      inst.opcode == "STG.E.128" || inst.opcode == "STG.E.NA.128";
  const bool wide = inst.opcode == "STG.E.64" ||
                    inst.opcode == "STG.E.64.STRONG.GPU" ||
                    inst.opcode == "STG.E.64.STRONG.SYS";
  const bool u16 = inst.opcode == "STG.E.U16";
  const bool u8 = inst.opcode == "STG.E.U8";
  if (vector128 && source->index != kZeroRegister &&
      source->index > kZeroRegister - 3)
    return unsupported(inst, "STG.E.128 register range is out of bounds");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (1u << lane)) == 0 ||
        !lane_executes(inst, state, lane))
      continue;
    uint64_t location = 0;
    if (direct) {
      if (!evaluate_address(*address, state, lane, location))
        return unsupported(inst, "invalid direct global-store address");
    } else {
      location = read_register_pair(state, lane, address->address_register) +
                 static_cast<uint64_t>(address->address_offset);
    }
    std::array<uint32_t, 4> value{};
    const unsigned words = vector128 ? 4 : wide ? 2 : 1;
    for (unsigned word = 0; word < words; ++word)
      value[word] = source->index == kZeroRegister
                        ? 0
                        : state.read_register(lane, source->index + word);
    const size_t bytes = u8 ? 1 : u16 ? 2 : words * sizeof(uint32_t);
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
  const bool vector64 = inst.opcode.find(".64") != std::string::npos;
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

  const unsigned words = vector128 ? 4 : vector64 ? 2 : 1;
  if (data->index != kZeroRegister && data->index > kZeroRegister - words)
    return unsupported(inst, "generic register range is out of bounds");
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

} // namespace
void register_global_memory_semantics(frontend &target) {
  for (const char *opcode :
       {"LDG.E.NA.EFL2.LTC256B.256.CONSTANT", "LDG.E.NA.ENL2.LTC64B.256",
        "LDG.E.ENL2.LTC128B.256.STRONG.GPU"})
    target.register_semantics(opcode, execute_ldg);
  target.register_semantics("LDG.E", execute_ldg);
  target.register_semantics("LDG.E.STRONG.GPU", execute_ldg);
  target.register_semantics("LDG.E.STRONG.SYS", execute_ldg);
  target.register_semantics("LDG.E.64.STRONG.GPU", execute_ldg);
  target.register_semantics("LDG.E.64.STRONG.SYS", execute_ldg);
  target.register_semantics("LDG.E.CONSTANT", execute_ldg);
  target.register_semantics("LDG.E.U8", execute_ldg);
  target.register_semantics("LDG.E.U16", execute_ldg);
  target.register_semantics("LDG.E.U16.CONSTANT", execute_ldg);
  target.register_semantics("LDG.E.64", execute_ldg);
  target.register_semantics("LDG.E.128", execute_ldg);
  target.register_semantics("LDG.E.NA.128", execute_ldg);
  target.register_semantics("LDG.E.128.CONSTANT", execute_ldg);
  target.register_semantics("STG.E", execute_stg);
  target.register_semantics("STG.E.STRONG.GPU", execute_stg);
  target.register_semantics("STG.E.STRONG.SYS", execute_stg);
  target.register_semantics("STG.E.64.STRONG.GPU", execute_stg);
  target.register_semantics("STG.E.64.STRONG.SYS", execute_stg);
  target.register_semantics("STG.E.U16", execute_stg);
  target.register_semantics("STG.E.U8", execute_stg);
  target.register_semantics("STG.E.64", execute_stg);
  target.register_semantics("STG.E.128", execute_stg);
  target.register_semantics("LD.E", execute_generic_memory);
  target.register_semantics("LD.E.U16.STRONG.SYS", execute_generic_memory);
  target.register_semantics("LD.E.128", execute_generic_memory);
  target.register_semantics("ST.E", execute_generic_memory);
  target.register_semantics("ST.E.STRONG.SYS", execute_generic_memory);
  target.register_semantics("ST.E.128", execute_generic_memory);
  target.register_semantics("LDG.E.ENL2.256", execute_ldg);
  target.register_semantics("LDG.E.S8", execute_ldg);
  target.register_semantics("LDG.E.LTC128B", execute_ldg);
  target.register_semantics("STG.E.ENL2.256", execute_stg);
  target.register_semantics("STG.E.NA.EFL2.256", execute_stg);
  target.register_semantics("STG.E.NA.128", execute_stg);
  target.register_semantics("ST.E.64", execute_generic_memory);
}
} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
