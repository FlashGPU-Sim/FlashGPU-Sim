#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.14-9.7.15: matrix fragment movement and MMA.

step_result execute_ldsm_stsm_m88(const instruction &inst, warp_state &state,
                                  execution_context &context) {
  const bool is_load = inst.opcode.rfind("LDSM.16.", 0) == 0;
  const bool transpose = inst.opcode.find(".MT88") != std::string::npos;
  const unsigned matrix_count =
      inst.opcode.rfind(".4") == inst.opcode.size() - 2   ? 4
      : inst.opcode.rfind(".2") == inst.opcode.size() - 2 ? 2
                                                          : 1;
  if (inst.has_guard ||
      (is_load ? !has_typed_operands(inst, 1) : !has_typed_operands(inst, 2)))
    return unsupported(inst, "unexpected predication or operand count");
  const operand *address = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *store_data =
      is_load ? nullptr : get_operand(inst, 1, operand_kind::kRegister);
  const unsigned data_register = is_load && address != nullptr
                                     ? address->prefix_index
                                 : store_data != nullptr ? store_data->index
                                                         : kVectorRegisters;
  if (address == nullptr || address->address_groups.size() != 1 ||
      (is_load ? (!address->has_prefix ||
                  address->prefix_kind != operand_kind::kRegister)
               : address->has_prefix) ||
      data_register > kVectorRegisters - matrix_count)
    return unsupported(inst, "unsupported M88 register or address form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "M88 matrix transfer requires a full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "matrix transfer requires functional memory");

  std::array<std::array<uint32_t, 4>, kWarpLanes> fragments{};
  std::array<std::array<uint64_t, 4>, kWarpLanes> addresses{};
  std::array<std::array<uint64_t, 4>, kWarpLanes> high_addresses{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned matrix_row = lane / 4;
    const unsigned matrix_column = (lane % 4) * 2;
    for (unsigned matrix = 0; matrix < matrix_count; ++matrix) {
      if (transpose) {
        const unsigned row0 = matrix_column;
        const unsigned row1 = row0 + 1;
        const unsigned column = matrix_row;
        uint64_t row0_address = 0;
        uint64_t row1_address = 0;
        if (!evaluate_shared_address(*address, state, matrix * 8 + row0,
                                     row0_address) ||
            !evaluate_shared_address(*address, state, matrix * 8 + row1,
                                     row1_address) ||
            row0_address > std::numeric_limits<uint64_t>::max() -
                               column * sizeof(uint16_t) ||
            row1_address > std::numeric_limits<uint64_t>::max() -
                               column * sizeof(uint16_t))
          return unsupported(inst, "invalid MT88 shared-memory address");
        addresses[lane][matrix] = row0_address + column * sizeof(uint16_t);
        high_addresses[lane][matrix] = row1_address + column * sizeof(uint16_t);
        if (is_load) {
          uint16_t low = 0;
          uint16_t high = 0;
          if (!context.memory->read(memory_space::kShared,
                                    addresses[lane][matrix], &low, sizeof(low)))
            return memory_fault(inst, memory_space::kShared,
                                addresses[lane][matrix], sizeof(low));
          if (!context.memory->read(memory_space::kShared,
                                    high_addresses[lane][matrix], &high,
                                    sizeof(high)))
            return memory_fault(inst, memory_space::kShared,
                                high_addresses[lane][matrix], sizeof(high));
          fragments[lane][matrix] =
              static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
        } else {
          fragments[lane][matrix] =
              state.read_register(lane, data_register + matrix);
        }
        continue;
      }
      const unsigned address_lane = matrix * 8 + matrix_row;
      uint64_t row_address = 0;
      if (!evaluate_shared_address(*address, state, address_lane,
                                   row_address) ||
          row_address > std::numeric_limits<uint64_t>::max() -
                            matrix_column * sizeof(uint16_t))
        return unsupported(inst, "invalid M88 shared-memory address");
      addresses[lane][matrix] = row_address + matrix_column * sizeof(uint16_t);
      if (is_load) {
        if (!context.memory->read(memory_space::kShared,
                                  addresses[lane][matrix],
                                  &fragments[lane][matrix], sizeof(uint32_t)))
          return memory_fault(inst, memory_space::kShared,
                              addresses[lane][matrix], sizeof(uint32_t));
      } else {
        fragments[lane][matrix] =
            state.read_register(lane, data_register + matrix);
      }
    }
  }

  if (is_load) {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      for (unsigned matrix = 0; matrix < matrix_count; ++matrix) {
        // Each matrix is a distinct shared-memory data-pipe wavefront.  Keep
        // every phase in the functional-to-timing handoff; recording only
        // matrix zero makes .2 and .4 indistinguishable to the bank model.
        context.record_memory_access(
            lane, memory_space::kShared, addresses[lane][matrix],
            transpose ? sizeof(uint16_t) : sizeof(uint32_t), false);
        state.write_register(lane, data_register + matrix,
                             fragments[lane][matrix]);
      }
    }
  } else {
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      for (unsigned matrix = 0; matrix < matrix_count; ++matrix) {
        context.record_memory_access(
            lane, memory_space::kShared, addresses[lane][matrix],
            transpose ? sizeof(uint16_t) : sizeof(uint32_t), true);
        if (transpose) {
          const uint16_t low = static_cast<uint16_t>(fragments[lane][matrix]);
          const uint16_t high =
              static_cast<uint16_t>(fragments[lane][matrix] >> 16);
          if (!context.memory->write(memory_space::kShared,
                                     addresses[lane][matrix], &low,
                                     sizeof(low)))
            return memory_fault(inst, memory_space::kShared,
                                addresses[lane][matrix], sizeof(low));
          if (!context.memory->write(memory_space::kShared,
                                     high_addresses[lane][matrix], &high,
                                     sizeof(high)))
            return memory_fault(inst, memory_space::kShared,
                                high_addresses[lane][matrix], sizeof(high));
        } else if (!context.memory->write(
                       memory_space::kShared, addresses[lane][matrix],
                       &fragments[lane][matrix], sizeof(uint32_t))) {
          return memory_fault(inst, memory_space::kShared,
                              addresses[lane][matrix], sizeof(uint32_t));
        }
      }
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hmma_1688_f32(const instruction &inst, warp_state &state,
                                  execution_context &) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected four typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_base = get_operand(inst, 2, operand_kind::kRegister);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  if (destination == nullptr || a_base == nullptr || b_base == nullptr ||
      c_base == nullptr || destination->negated || a_base->negated ||
      b_base->negated || c_base->negated)
    return unsupported(inst, "unsupported tensor-fragment operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "HMMA.1688 requires a converged full warp");
  const bool bf16 = inst.opcode == "HMMA.1688.F32.BF16";
  if (!bf16 && inst.opcode != "HMMA.1688.F32")
    return unsupported(inst, "unsupported HMMA.1688 input format");
  const auto input_float = bf16 ? bfloat_float : half_float;

  float matrix_a[16][8]{};
  float matrix_b[8][8]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HMMA predication is not implemented");
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    const unsigned k0 = thread * 2;
    const uint32_t a0 = state.read_register(lane, a_base->index);
    const uint32_t a1 = state.read_register(lane, a_base->index + 1);
    const uint32_t b0 = state.read_register(lane, b_base->index);
    matrix_a[group][k0] = input_float(static_cast<uint16_t>(a0));
    matrix_a[group][k0 + 1] = input_float(static_cast<uint16_t>(a0 >> 16));
    matrix_a[group + 8][k0] = input_float(static_cast<uint16_t>(a1));
    matrix_a[group + 8][k0 + 1] = input_float(static_cast<uint16_t>(a1 >> 16));
    matrix_b[k0][group] = input_float(static_cast<uint16_t>(b0));
    matrix_b[k0 + 1][group] = input_float(static_cast<uint16_t>(b0 >> 16));
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    const unsigned column0 = thread * 2;
    const unsigned rows[4] = {group, group, group + 8, group + 8};
    const unsigned columns[4] = {column0, column0 + 1, column0, column0 + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float accumulator =
          c_base->index == kZeroRegister
              ? 0.0f
              : bits_float(state.read_register(lane, c_base->index + output));
      for (unsigned k = 0; k < 8; ++k)
        accumulator = std::fma(matrix_a[rows[output]][k],
                               matrix_b[k][columns[output]], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hmma_16816_f32(const instruction &inst, warp_state &state,
                                   execution_context &) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected four typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_base = get_operand(inst, 2, operand_kind::kRegister);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  if (destination == nullptr || a_base == nullptr || b_base == nullptr ||
      c_base == nullptr || destination->negated || a_base->negated ||
      b_base->negated || c_base->negated ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - 4 || b_base->index > kZeroRegister - 2 ||
      (c_base->index != kZeroRegister && c_base->index > kZeroRegister - 4))
    return unsupported(inst, "unsupported tensor-fragment operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "HMMA.16816 requires a converged full warp");
  const bool bf16 = inst.opcode == "HMMA.16816.F32.BF16";
  if (!bf16 && inst.opcode != "HMMA.16816.F32")
    return unsupported(inst, "unsupported HMMA.16816 input format");
  const auto input_float = bf16 ? bfloat_float : half_float;

  float matrix_a[16][16]{};
  float matrix_b[16][8]{};
  std::array<std::array<float, 4>, kWarpLanes> accumulators{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HMMA predication is not implemented");
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    const unsigned k0 = thread * 2;
    const uint32_t a0 = state.read_register(lane, a_base->index);
    const uint32_t a1 = state.read_register(lane, a_base->index + 1);
    const uint32_t a2 = state.read_register(lane, a_base->index + 2);
    const uint32_t a3 = state.read_register(lane, a_base->index + 3);
    const uint32_t b0 = state.read_register(lane, b_base->index);
    const uint32_t b1 = state.read_register(lane, b_base->index + 1);
    matrix_a[group][k0] = input_float(static_cast<uint16_t>(a0));
    matrix_a[group][k0 + 1] = input_float(static_cast<uint16_t>(a0 >> 16));
    matrix_a[group + 8][k0] = input_float(static_cast<uint16_t>(a1));
    matrix_a[group + 8][k0 + 1] = input_float(static_cast<uint16_t>(a1 >> 16));
    matrix_a[group][k0 + 8] = input_float(static_cast<uint16_t>(a2));
    matrix_a[group][k0 + 9] = input_float(static_cast<uint16_t>(a2 >> 16));
    matrix_a[group + 8][k0 + 8] = input_float(static_cast<uint16_t>(a3));
    matrix_a[group + 8][k0 + 9] = input_float(static_cast<uint16_t>(a3 >> 16));
    matrix_b[k0][group] = input_float(static_cast<uint16_t>(b0));
    matrix_b[k0 + 1][group] = input_float(static_cast<uint16_t>(b0 >> 16));
    matrix_b[k0 + 8][group] = input_float(static_cast<uint16_t>(b1));
    matrix_b[k0 + 9][group] = input_float(static_cast<uint16_t>(b1 >> 16));
    for (unsigned output = 0; output < 4; ++output) {
      accumulators[lane][output] =
          c_base->index == kZeroRegister
              ? 0.0f
              : bits_float(state.read_register(lane, c_base->index + output));
    }
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    const unsigned column0 = thread * 2;
    const unsigned rows[4] = {group, group, group + 8, group + 8};
    const unsigned columns[4] = {column0, column0 + 1, column0, column0 + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float accumulator = accumulators[lane][output];
      for (unsigned k = 0; k < 16; ++k)
        accumulator = std::fma(matrix_a[rows[output]][k],
                               matrix_b[k][columns[output]], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hmma_tf32(const instruction &inst, warp_state &state,
                              execution_context &) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected four typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_base = get_operand(inst, 2, operand_kind::kRegister);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const bool k8 = inst.opcode == "HMMA.1688.F32.TF32";
  if (!k8 && inst.opcode != "HMMA.1684.F32.TF32")
    return unsupported(inst, "unsupported TF32 HMMA shape");
  const unsigned a_registers = k8 ? 4 : 2;
  const unsigned b_registers = k8 ? 2 : 1;
  if (destination == nullptr || a_base == nullptr || b_base == nullptr ||
      c_base == nullptr || destination->negated || a_base->negated ||
      b_base->negated || c_base->negated ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - a_registers ||
      b_base->index > kZeroRegister - b_registers ||
      (c_base->index != kZeroRegister && c_base->index > kZeroRegister - 4))
    return unsupported(inst, "unsupported TF32 tensor-fragment operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "TF32 HMMA requires a converged full warp");

  const unsigned k_count = k8 ? 8 : 4;
  float matrix_a[16][8]{};
  float matrix_b[8][8]{};
  std::array<std::array<float, 4>, kWarpLanes> accumulators{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HMMA predication is not implemented");
    const unsigned group = lane >> 2;
    const unsigned thread = lane & 3;
    matrix_a[group][thread] =
        tf32_float(state.read_register(lane, a_base->index));
    matrix_a[group + 8][thread] =
        tf32_float(state.read_register(lane, a_base->index + 1));
    matrix_b[thread][group] =
        tf32_float(state.read_register(lane, b_base->index));
    if (k8) {
      matrix_a[group][thread + 4] =
          tf32_float(state.read_register(lane, a_base->index + 2));
      matrix_a[group + 8][thread + 4] =
          tf32_float(state.read_register(lane, a_base->index + 3));
      matrix_b[thread + 4][group] =
          tf32_float(state.read_register(lane, b_base->index + 1));
    }
    for (unsigned output = 0; output < 4; ++output) {
      accumulators[lane][output] =
          c_base->index == kZeroRegister
              ? 0.0f
              : bits_float(state.read_register(lane, c_base->index + output));
    }
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned group = lane >> 2;
    const unsigned column0 = (lane & 3) * 2;
    const unsigned rows[4] = {group, group, group + 8, group + 8};
    const unsigned columns[4] = {column0, column0 + 1, column0, column0 + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float accumulator = accumulators[lane][output];
      for (unsigned k = 0; k < k_count; ++k)
        accumulator = std::fma(matrix_a[rows[output]][k],
                               matrix_b[k][columns[output]], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_imma_s8(const instruction &inst, warp_state &state,
                            execution_context &) {
  if (!has_typed_operands(inst, 4))
    return unsupported(inst, "expected four typed operands");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_base = get_operand(inst, 2, operand_kind::kRegister);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const bool k32 = inst.opcode == "IMMA.16832.S8.S8";
  if (!k32 && inst.opcode != "IMMA.16816.S8.S8")
    return unsupported(inst, "unsupported signed IMMA shape");
  const unsigned a_registers = k32 ? 4 : 2;
  const unsigned b_registers = k32 ? 2 : 1;
  if (destination == nullptr || a_base == nullptr || b_base == nullptr ||
      c_base == nullptr || destination->negated || a_base->negated ||
      b_base->negated || c_base->negated ||
      a_base->matrix_layout != operand_matrix_layout::kRow ||
      b_base->matrix_layout != operand_matrix_layout::kColumn ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - a_registers ||
      b_base->index > kZeroRegister - b_registers ||
      (c_base->index != kZeroRegister && c_base->index > kZeroRegister - 4))
    return unsupported(inst, "unsupported signed tensor-fragment operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "IMMA requires a converged full warp");

  const unsigned k_count = k32 ? 32 : 16;
  int32_t matrix_a[16][32]{};
  int32_t matrix_b[32][8]{};
  std::array<std::array<int64_t, 4>, kWarpLanes> accumulators{};
  const auto signed_byte = [](uint32_t packed, unsigned byte) {
    const uint32_t value = (packed >> (byte * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int32_t>(value)
                         : static_cast<int32_t>(value) - 0x100;
  };
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent IMMA predication is not implemented");
    const unsigned group = lane >> 2;
    const unsigned k0 = (lane & 3) * 4;
    const uint32_t a0 = state.read_register(lane, a_base->index);
    const uint32_t a1 = state.read_register(lane, a_base->index + 1);
    const uint32_t b0 = state.read_register(lane, b_base->index);
    for (unsigned byte = 0; byte < 4; ++byte) {
      matrix_a[group][k0 + byte] = signed_byte(a0, byte);
      matrix_b[k0 + byte][group] = signed_byte(b0, byte);
      if (k32) {
        matrix_a[group][k0 + 16 + byte] = signed_byte(a1, byte);
        matrix_a[group + 8][k0 + byte] =
            signed_byte(state.read_register(lane, a_base->index + 2), byte);
        matrix_a[group + 8][k0 + 16 + byte] =
            signed_byte(state.read_register(lane, a_base->index + 3), byte);
        matrix_b[k0 + 16 + byte][group] =
            signed_byte(state.read_register(lane, b_base->index + 1), byte);
      } else {
        matrix_a[group + 8][k0 + byte] = signed_byte(a1, byte);
      }
    }
    for (unsigned output = 0; output < 4; ++output) {
      accumulators[lane][output] =
          c_base->index == kZeroRegister
              ? 0
              : static_cast<int32_t>(
                    state.read_register(lane, c_base->index + output));
    }
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned group = lane >> 2;
    const unsigned column0 = (lane & 3) * 2;
    const unsigned rows[4] = {group, group, group + 8, group + 8};
    const unsigned columns[4] = {column0, column0 + 1, column0, column0 + 1};
    for (unsigned output = 0; output < 4; ++output) {
      int64_t accumulator = accumulators[lane][output];
      for (unsigned k = 0; k < k_count; ++k)
        accumulator += static_cast<int64_t>(matrix_a[rows[output]][k]) *
                       matrix_b[k][columns[output]];
      state.write_register(lane, destination->index + output,
                           static_cast<uint32_t>(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_mma_semantics(frontend &target) {
  target.register_semantics("LDSM.16.M88", execute_ldsm_stsm_m88);
  target.register_semantics("LDSM.16.M88.2", execute_ldsm_stsm_m88);
  target.register_semantics("LDSM.16.M88.4", execute_ldsm_stsm_m88);
  target.register_semantics("LDSM.16.MT88", execute_ldsm_stsm_m88);
  target.register_semantics("LDSM.16.MT88.2", execute_ldsm_stsm_m88);
  target.register_semantics("LDSM.16.MT88.4", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.M88", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.M88.2", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.M88.4", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.MT88", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.MT88.2", execute_ldsm_stsm_m88);
  target.register_semantics("STSM.16.MT88.4", execute_ldsm_stsm_m88);
  target.register_semantics("HMMA.1688.F32", execute_hmma_1688_f32);
  target.register_semantics("HMMA.1688.F32.BF16", execute_hmma_1688_f32);
  target.register_semantics("HMMA.16816.F32", execute_hmma_16816_f32);
  target.register_semantics("HMMA.16816.F32.BF16", execute_hmma_16816_f32);
  target.register_semantics("HMMA.1684.F32.TF32", execute_hmma_tf32);
  target.register_semantics("HMMA.1688.F32.TF32", execute_hmma_tf32);
  target.register_semantics("IMMA.16816.S8.S8", execute_imma_s8);
  target.register_semantics("IMMA.16832.S8.S8", execute_imma_s8);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
