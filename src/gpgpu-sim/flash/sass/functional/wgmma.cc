#include "../decode/wgmma_instruction.h"
#include "internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// Hopper warp-group control has no separately visible functional state once
// the multiply is evaluated synchronously. Timing and asynchronous completion
// remain outside this execution-driven correctness layer.
step_result execute_warpgroup_arrive(const instruction &inst, warp_state &state,
                                     execution_context &) {
  if (inst.has_guard || !has_typed_operands(inst, 0))
    return unsupported(inst, "WARPGROUP.ARRIVE must not have operands");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_warpgroup_depbar(const instruction &inst, warp_state &state,
                                     execution_context &) {
  if (inst.has_guard || !has_typed_operands(inst, 2) ||
      inst.operands[0].kind != operand_kind::kScoreboardRegister ||
      inst.operands[0].index != 0 ||
      inst.operands[1].kind != operand_kind::kImmediate ||
      inst.operands[1].immediate < 0 || inst.operands[1].immediate > 1)
    return unsupported(inst,
                       "only WARPGROUP.DEPBAR.LE gsb0,{0,1} is implemented");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hgmma_commit_group(const instruction &inst,
                                       warp_state &state, execution_context &) {
  if (!is_hgmma_commit_group_sentinel(inst))
    return unsupported(inst, "invalid HGMMA commit-group sentinel");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hgmma_64x8_f32(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  enum class input_format { kF16, kBf16, kTf32 };
  input_format format;
  if (inst.opcode == "HGMMA.64x8x16.F32")
    format = input_format::kF16;
  else if (inst.opcode == "HGMMA.64x8x16.F32.BF16")
    format = input_format::kBf16;
  else if (inst.opcode == "HGMMA.64x8x8.F32.TF32")
    format = input_format::kTf32;
  else
    return unsupported(inst, "unknown register/shared HGMMA input format");
  const bool tf32 = format == input_format::kTf32;
  const unsigned k_extent = tf32 ? 8 : 16;
  const uint64_t element_bytes = tf32 ? sizeof(uint32_t) : sizeof(uint16_t);
  if (inst.has_guard || !has_typed_operands(inst, 6))
    return unsupported(inst, "unexpected predication or operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_descriptor =
      get_operand(inst, 2, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const operand *scale_d =
      get_operand(inst, 4, operand_kind::kUniformPredicate);
  const operand *scoreboard =
      get_operand(inst, 5, operand_kind::kScoreboardRegister);
  if (destination == nullptr || a_base == nullptr || b_descriptor == nullptr ||
      c_base == nullptr || scale_d == nullptr || scoreboard == nullptr ||
      destination->negated || a_base->negated || c_base->negated ||
      b_descriptor->descriptor_transpose_b || scoreboard->index != 0 ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - 4 || c_base->index > kZeroRegister - 4 ||
      b_descriptor->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "unsupported register/shared HGMMA operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "HGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "HGMMA requires functional shared memory");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HGMMA execution is not implemented");
  }

  // gdesc[URn] names a four-register descriptor bundle. In the register/shared
  // form the A descriptor slots URn:n+1 are unused and B occupies URn+2:n+3.
  const uint64_t descriptor =
      read_uniform_register_pair(state, b_descriptor->descriptor_register + 2);
  const uint64_t base = (descriptor & 0x3fffull) << 4;
  const uint64_t leading = ((descriptor >> 16) & 0x3fffull) << 4;
  const uint64_t stride = ((descriptor >> 32) & 0x3fffull) << 4;
  const uint64_t swizzle = descriptor >> 62;
  if (swizzle != 0 || leading == 0 || stride < element_bytes ||
      stride % element_bytes != 0)
    return unsupported(inst, "unsupported HGMMA shared descriptor layout");
  const uint64_t contiguous_k = stride / element_bytes;
  if (contiguous_k == 0 || contiguous_k > k_extent)
    return unsupported(inst, "invalid HGMMA contiguous K span");

  float matrix_a[16][16]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const uint32_t packed = state.read_register(lane, a_base->index + reg);
      const unsigned matrix_row = row + ((reg & 1) != 0 ? 8 : 0);
      if (tf32) {
        const unsigned matrix_k = (lane & 3) + ((reg & 2) != 0 ? 4 : 0);
        matrix_a[matrix_row][matrix_k] = tf32_float(packed);
      } else {
        const unsigned matrix_k = (lane & 3) * 2 + ((reg & 2) != 0 ? 8 : 0);
        const auto convert =
            format == input_format::kBf16 ? bfloat_float : half_float;
        matrix_a[matrix_row][matrix_k] = convert(static_cast<uint16_t>(packed));
        matrix_a[matrix_row][matrix_k + 1] =
            convert(static_cast<uint16_t>(packed >> 16));
      }
    }
  }

  float matrix_b[16][8]{};
  for (unsigned k = 0; k < k_extent; ++k) {
    for (unsigned column = 0; column < 8; ++column) {
      const uint64_t address = base + (k / contiguous_k) * leading +
                               column * stride +
                               (k % contiguous_k) * element_bytes;
      if (tf32) {
        uint32_t value = 0;
        if (!context.memory->read(memory_space::kShared, address, &value,
                                  sizeof(value)))
          return memory_fault(inst, memory_space::kShared, address,
                              sizeof(value));
        matrix_b[k][column] = tf32_float(value);
      } else {
        uint16_t value = 0;
        if (!context.memory->read(memory_space::kShared, address, &value,
                                  sizeof(value)))
          return memory_fault(inst, memory_space::kShared, address,
                              sizeof(value));
        matrix_b[k][column] = format == input_format::kBf16
                                  ? bfloat_float(value)
                                  : half_float(value);
      }
    }
  }

  bool accumulates = false;
  if (!read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid HGMMA scale-D predicate");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    const unsigned column = (lane & 3) * 2;
    const unsigned rows[4] = {row, row, row + 8, row + 8};
    const unsigned columns[4] = {column, column + 1, column, column + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float accumulator =
          accumulates
              ? bits_float(state.read_register(lane, c_base->index + output))
              : 0.0f;
      for (unsigned k = 0; k < k_extent; ++k)
        accumulator = std::fma(matrix_a[rows[output]][k],
                               matrix_b[k][columns[output]], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

struct shared_descriptor {
  uint64_t base = 0;
  uint64_t leading = 0;
  uint64_t stride = 0;
  uint64_t swizzle = 0;
};

shared_descriptor decode_shared_descriptor(uint64_t raw) {
  return {(raw & 0x3fffull) << 4, ((raw >> 16) & 0x3fffull) << 4,
          ((raw >> 32) & 0x3fffull) << 4, raw >> 62};
}

uint64_t swizzle_128b(uint64_t byte_offset) {
  constexpr uint64_t y_mask = uint64_t{0x7} << 7;
  return byte_offset ^ ((byte_offset & y_mask) >> 3);
}

uint64_t swizzle_128b_address(uint64_t base, uint64_t byte_offset) {
  // A GMMA descriptor can begin inside the 1-KiB period of a 128-byte
  // swizzle. The descriptor base participates in the swizzle coordinate;
  // adding an independently swizzled offset produces the wrong K slice.
  constexpr uint64_t period = 128 * 8;
  const uint64_t period_base = base & ~(period - 1);
  const uint64_t period_offset = (base - period_base) + byte_offset;
  return period_base + swizzle_128b(period_offset);
}

uint64_t k_major_f16_address(const shared_descriptor &descriptor, unsigned row,
                             unsigned k) {
  constexpr uint64_t element_bytes = sizeof(uint16_t);
  constexpr unsigned elements_per_16b = 16 / element_bytes;
  const uint64_t logical = uint64_t{row / 8} * descriptor.stride +
                           uint64_t{row % 8} * 128 +
                           uint64_t{k / elements_per_16b} * descriptor.leading +
                           uint64_t{k % elements_per_16b} * element_bytes;
  return swizzle_128b_address(descriptor.base, logical);
}

uint64_t mn_major_f16_address(const shared_descriptor &descriptor, unsigned row,
                              unsigned k) {
  constexpr uint64_t element_bytes = sizeof(uint16_t);
  constexpr unsigned rows_per_swizzle = 128 / element_bytes;
  const uint64_t logical =
      uint64_t{row / rows_per_swizzle} * descriptor.leading +
      uint64_t{row % rows_per_swizzle} * element_bytes +
      uint64_t{k / 8} * descriptor.stride + uint64_t{k % 8} * 128;
  return swizzle_128b_address(descriptor.base, logical);
}

float fp8_float(uint8_t bits, bool e5m2) {
  const bool negative = (bits & 0x80u) != 0;
  const unsigned mantissa_bits = e5m2 ? 2 : 3;
  const unsigned exponent_mask = e5m2 ? 0x1fu : 0x0fu;
  const unsigned mantissa_mask = (1u << mantissa_bits) - 1;
  const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
  const unsigned mantissa = bits & mantissa_mask;
  const int bias = e5m2 ? 15 : 7;
  float value = 0.0f;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa),
                       1 - bias - static_cast<int>(mantissa_bits));
  } else if ((e5m2 && exponent == exponent_mask) ||
             (!e5m2 && exponent == exponent_mask &&
              mantissa == mantissa_mask)) {
    value = mantissa == 0 && e5m2 ? INFINITY : NAN;
  } else {
    value =
        std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << mantissa_bits),
                   static_cast<int>(exponent) - bias);
  }
  return negative ? -value : value;
}

step_result execute_qgmma_64x8x32_f32(const instruction &inst,
                                      warp_state &state,
                                      execution_context &context) {
  bool a_e5m2 = false;
  bool b_e5m2 = false;
  if (inst.opcode == "QGMMA.64x8x32.F32.E4M3.E4M3") {
  } else if (inst.opcode == "QGMMA.64x8x32.F32.E5M2.E5M2") {
    a_e5m2 = true;
    b_e5m2 = true;
  } else if (inst.opcode == "QGMMA.64x8x32.F32.E4M3.E5M2") {
    b_e5m2 = true;
  } else if (inst.opcode == "QGMMA.64x8x32.F32.E5M2.E4M3") {
    a_e5m2 = true;
  } else {
    return unsupported(inst, "unknown FP8 QGMMA input format");
  }
  if (inst.has_guard || !has_typed_operands(inst, 6))
    return unsupported(inst, "unexpected predication or operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_descriptor =
      get_operand(inst, 2, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const operand *scale_d =
      get_operand(inst, 4, operand_kind::kUniformPredicate);
  const operand *scoreboard =
      get_operand(inst, 5, operand_kind::kScoreboardRegister);
  if (destination == nullptr || a_base == nullptr || b_descriptor == nullptr ||
      c_base == nullptr || scale_d == nullptr || scoreboard == nullptr ||
      destination->negated || a_base->negated || c_base->negated ||
      b_descriptor->descriptor_transpose_b || scoreboard->index != 0 ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - 4 || c_base->index > kZeroRegister - 4 ||
      b_descriptor->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "unsupported register/shared QGMMA operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "QGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "QGMMA requires functional shared memory");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent QGMMA execution is not implemented");
  }

  const shared_descriptor descriptor = decode_shared_descriptor(
      read_uniform_register_pair(state, b_descriptor->descriptor_register + 2));
  if (descriptor.swizzle != 0 || descriptor.leading == 0 ||
      descriptor.stride == 0)
    return unsupported(inst, "unsupported QGMMA shared descriptor layout");
  const uint64_t contiguous_k = descriptor.stride;
  if (contiguous_k == 0 || contiguous_k > 32)
    return unsupported(inst, "invalid QGMMA contiguous K span");

  float matrix_a[16][32]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const uint32_t packed = state.read_register(lane, a_base->index + reg);
      const unsigned matrix_row = row + ((reg & 1) != 0 ? 8 : 0);
      const unsigned matrix_k = (lane & 3) * 4 + ((reg & 2) != 0 ? 16 : 0);
      for (unsigned byte = 0; byte < 4; ++byte)
        matrix_a[matrix_row][matrix_k + byte] =
            fp8_float(static_cast<uint8_t>(packed >> (8 * byte)), a_e5m2);
    }
  }

  float matrix_b[32][8]{};
  for (unsigned k = 0; k < 32; ++k) {
    for (unsigned column = 0; column < 8; ++column) {
      const uint64_t address = descriptor.base +
                               (k / contiguous_k) * descriptor.leading +
                               column * descriptor.stride + (k % contiguous_k);
      uint8_t value = 0;
      if (!context.memory->read(memory_space::kShared, address, &value,
                                sizeof(value)))
        return memory_fault(inst, memory_space::kShared, address,
                            sizeof(value));
      matrix_b[k][column] = fp8_float(value, b_e5m2);
    }
  }

  bool accumulates = false;
  if (!read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid QGMMA scale-D predicate");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    const unsigned column = (lane & 3) * 2;
    const unsigned rows[4] = {row, row, row + 8, row + 8};
    const unsigned columns[4] = {column, column + 1, column, column + 1};
    for (unsigned output = 0; output < 4; ++output) {
      float accumulator =
          accumulates
              ? bits_float(state.read_register(lane, c_base->index + output))
              : 0.0f;
      for (unsigned k = 0; k < 32; ++k)
        accumulator = std::fma(matrix_a[rows[output]][k],
                               matrix_b[k][columns[output]], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_igmma_64x8x32(const instruction &inst, warp_state &state,
                                  execution_context &context) {
  bool signed_a = false;
  bool signed_b = false;
  if (inst.opcode == "IGMMA.64x8x32.S8.S8.SAT") {
    signed_a = true;
    signed_b = true;
  } else if (inst.opcode == "IGMMA.64x8x32.U8.U8") {
  } else if (inst.opcode == "IGMMA.64x8x32.S8.U8.SAT") {
    signed_a = true;
  } else if (inst.opcode == "IGMMA.64x8x32.U8.S8") {
    signed_b = true;
  } else {
    return unsupported(inst, "unknown integer IGMMA input format");
  }
  const bool saturates = inst.opcode.find(".SAT") != std::string::npos;
  if (inst.has_guard || !has_typed_operands(inst, 6))
    return unsupported(inst, "unexpected predication or operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_descriptor =
      get_operand(inst, 2, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const operand *scale_d =
      get_operand(inst, 4, operand_kind::kUniformPredicate);
  const operand *scoreboard =
      get_operand(inst, 5, operand_kind::kScoreboardRegister);
  if (destination == nullptr || a_base == nullptr || b_descriptor == nullptr ||
      c_base == nullptr || scale_d == nullptr || scoreboard == nullptr ||
      destination->negated || a_base->negated || c_base->negated ||
      b_descriptor->descriptor_transpose_b || scoreboard->index != 0 ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - 4 || c_base->index > kZeroRegister - 4 ||
      b_descriptor->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "unsupported register/shared IGMMA operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "IGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "IGMMA requires functional shared memory");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent IGMMA execution is not implemented");
  }

  const shared_descriptor descriptor = decode_shared_descriptor(
      read_uniform_register_pair(state, b_descriptor->descriptor_register + 2));
  if (descriptor.swizzle != 0 || descriptor.leading == 0 ||
      descriptor.stride == 0)
    return unsupported(inst, "unsupported IGMMA shared descriptor layout");
  const uint64_t contiguous_k = descriptor.stride;
  if (contiguous_k == 0 || contiguous_k > 32)
    return unsupported(inst, "invalid IGMMA contiguous K span");

  int32_t matrix_a[16][32]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const uint32_t packed = state.read_register(lane, a_base->index + reg);
      const unsigned matrix_row = row + ((reg & 1) != 0 ? 8 : 0);
      const unsigned matrix_k = (lane & 3) * 4 + ((reg & 2) != 0 ? 16 : 0);
      for (unsigned byte = 0; byte < 4; ++byte) {
        const uint8_t value = static_cast<uint8_t>(packed >> (8 * byte));
        matrix_a[matrix_row][matrix_k + byte] =
            signed_a ? static_cast<int8_t>(value) : value;
      }
    }
  }

  int32_t matrix_b[32][8]{};
  for (unsigned k = 0; k < 32; ++k) {
    for (unsigned column = 0; column < 8; ++column) {
      const uint64_t address = descriptor.base +
                               (k / contiguous_k) * descriptor.leading +
                               column * descriptor.stride + (k % contiguous_k);
      uint8_t value = 0;
      if (!context.memory->read(memory_space::kShared, address, &value,
                                sizeof(value)))
        return memory_fault(inst, memory_space::kShared, address,
                            sizeof(value));
      matrix_b[k][column] = signed_b ? static_cast<int8_t>(value) : value;
    }
  }

  bool accumulates = false;
  if (!read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid IGMMA scale-D predicate");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    const unsigned column = (lane & 3) * 2;
    const unsigned rows[4] = {row, row, row + 8, row + 8};
    const unsigned columns[4] = {column, column + 1, column, column + 1};
    for (unsigned output = 0; output < 4; ++output) {
      int64_t accumulator =
          accumulates ? static_cast<int32_t>(
                            state.read_register(lane, c_base->index + output))
                      : 0;
      for (unsigned k = 0; k < 32; ++k)
        accumulator +=
            int64_t{matrix_a[rows[output]][k]} * matrix_b[k][columns[output]];
      if (saturates)
        accumulator = std::max<int64_t>(
            std::numeric_limits<int32_t>::min(),
            std::min<int64_t>(std::numeric_limits<int32_t>::max(),
                              accumulator));
      state.write_register(lane, destination->index + output,
                           static_cast<uint32_t>(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_bgmma_64x8x256(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  if (inst.opcode != "BGMMA.64x8x256.AND.POPC" || inst.has_guard ||
      !has_typed_operands(inst, 6))
    return unsupported(inst, "unsupported BGMMA opcode or operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_descriptor =
      get_operand(inst, 2, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const operand *scale_d =
      get_operand(inst, 4, operand_kind::kUniformPredicate);
  const operand *scoreboard =
      get_operand(inst, 5, operand_kind::kScoreboardRegister);
  if (destination == nullptr || a_base == nullptr || b_descriptor == nullptr ||
      c_base == nullptr || scale_d == nullptr || scoreboard == nullptr ||
      destination->negated || a_base->negated || c_base->negated ||
      b_descriptor->descriptor_transpose_b || scoreboard->index != 0 ||
      destination->index > kZeroRegister - 4 ||
      a_base->index > kZeroRegister - 4 || c_base->index > kZeroRegister - 4 ||
      b_descriptor->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "unsupported register/shared BGMMA operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "BGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "BGMMA requires functional shared memory");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent BGMMA execution is not implemented");
  }

  const shared_descriptor descriptor = decode_shared_descriptor(
      read_uniform_register_pair(state, b_descriptor->descriptor_register + 2));
  constexpr uint64_t element_bytes = sizeof(uint32_t);
  if (descriptor.swizzle != 0 || descriptor.leading == 0 ||
      descriptor.stride < element_bytes ||
      descriptor.stride % element_bytes != 0)
    return unsupported(inst, "unsupported BGMMA shared descriptor layout");
  const uint64_t contiguous_words = descriptor.stride / element_bytes;
  if (contiguous_words == 0 || contiguous_words > 8)
    return unsupported(inst, "invalid BGMMA contiguous K span");

  uint32_t matrix_a[16][8]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const unsigned matrix_row = row + ((reg & 1) != 0 ? 8 : 0);
      const unsigned word = (lane & 3) + ((reg & 2) != 0 ? 4 : 0);
      matrix_a[matrix_row][word] =
          state.read_register(lane, a_base->index + reg);
    }
  }

  uint32_t matrix_b[8][8]{};
  for (unsigned word = 0; word < 8; ++word) {
    for (unsigned column = 0; column < 8; ++column) {
      const uint64_t address = descriptor.base +
                               (word / contiguous_words) * descriptor.leading +
                               column * descriptor.stride +
                               (word % contiguous_words) * element_bytes;
      if (!context.memory->read(memory_space::kShared, address,
                                &matrix_b[word][column], element_bytes))
        return memory_fault(inst, memory_space::kShared, address,
                            element_bytes);
    }
  }

  bool accumulates = false;
  if (!read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid BGMMA scale-D predicate");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    const unsigned column = (lane & 3) * 2;
    const unsigned rows[4] = {row, row, row + 8, row + 8};
    const unsigned columns[4] = {column, column + 1, column, column + 1};
    for (unsigned output = 0; output < 4; ++output) {
      // Non-saturating integer accumulation wraps at 32 bits; signed host
      // addition would be undefined when a popcount crosses INT32_MAX.
      uint32_t accumulator =
          accumulates ? state.read_register(lane, c_base->index + output) : 0;
      for (unsigned word = 0; word < 8; ++word)
        accumulator += static_cast<uint32_t>(__builtin_popcount(
            matrix_a[rows[output]][word] & matrix_b[word][columns[output]]));
      state.write_register(lane, destination->index + output, accumulator);
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hgmma_shared_shared_f32(const instruction &inst,
                                            warp_state &state,
                                            execution_context &context) {
  const unsigned n_extent = inst.opcode == "HGMMA.64x64x16.F32"    ? 64
                            : inst.opcode == "HGMMA.64x80x16.F32"  ? 80
                            : inst.opcode == "HGMMA.64x128x16.F32" ? 128
                            : inst.opcode == "HGMMA.64x176x16.F32" ? 176
                            : inst.opcode == "HGMMA.64x192x16.F32" ? 192
                            : inst.opcode == "HGMMA.64x256x16.F32" ? 256
                                                                   : 0;
  if (n_extent == 0)
    return unsupported(inst, "unsupported shared/shared HGMMA shape");
  const unsigned output_registers = n_extent / 2;
  if (inst.has_guard || inst.operands.size() < 3 || inst.operands.size() > 5 ||
      !has_typed_operands(inst, inst.operands.size()))
    return unsupported(inst, "unexpected predication or operand count");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *descriptors =
      get_operand(inst, 1, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 2, operand_kind::kRegister);
  const operand *scale_d = nullptr;
  const operand *scoreboard = nullptr;
  if (inst.operands.size() >= 4) {
    if (inst.operands[3].kind == operand_kind::kUniformPredicate)
      scale_d = &inst.operands[3];
    else if (inst.operands[3].kind == operand_kind::kScoreboardRegister)
      scoreboard = &inst.operands[3];
    else
      return unsupported(inst, "invalid HGMMA optional operand");
  }
  if (inst.operands.size() == 5) {
    if (scale_d == nullptr ||
        inst.operands[4].kind != operand_kind::kScoreboardRegister)
      return unsupported(inst, "invalid HGMMA scale/scoreboard operands");
    scoreboard = &inst.operands[4];
  }
  if (destination == nullptr || descriptors == nullptr || c_base == nullptr ||
      destination->negated || c_base->negated ||
      (scoreboard != nullptr && scoreboard->index != 0) ||
      destination->index > kZeroRegister - output_registers ||
      (c_base->index != kZeroRegister &&
       c_base->index > kZeroRegister - output_registers) ||
      descriptors->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "unsupported shared/shared HGMMA operand form");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "HGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "HGMMA requires functional shared memory");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HGMMA execution is not implemented");
  }

  const shared_descriptor descriptor_a = decode_shared_descriptor(
      read_uniform_register_pair(state, descriptors->descriptor_register));
  const shared_descriptor descriptor_b = decode_shared_descriptor(
      read_uniform_register_pair(state, descriptors->descriptor_register + 2));
  const uint64_t expected_a_leading =
      descriptors->descriptor_transpose_a ? 0 : 16;
  const uint64_t expected_b_leading =
      descriptors->descriptor_transpose_b ? 0 : 16;
  if (descriptor_a.swizzle != 1 || descriptor_a.leading != expected_a_leading ||
      descriptor_a.stride != 1024 || descriptor_b.swizzle != 1 ||
      descriptor_b.leading != expected_b_leading || descriptor_b.stride != 1024)
    return unsupported(inst,
                       "unsupported shared/shared HGMMA descriptor layout");

  float matrix_a[64][16]{};
  float matrix_b[16][256]{};
  for (unsigned row = 0; row < 64; ++row) {
    for (unsigned k = 0; k < 16; ++k) {
      const uint64_t address = descriptors->descriptor_transpose_a
                                   ? mn_major_f16_address(descriptor_a, row, k)
                                   : k_major_f16_address(descriptor_a, row, k);
      uint16_t value = 0;
      if (!context.memory->read(memory_space::kShared, address, &value,
                                sizeof(value)))
        return memory_fault(inst, memory_space::kShared, address,
                            sizeof(value));
      matrix_a[row][k] = half_float(value);
    }
  }
  for (unsigned column = 0; column < n_extent; ++column) {
    for (unsigned k = 0; k < 16; ++k) {
      const uint64_t address =
          descriptors->descriptor_transpose_b
              ? mn_major_f16_address(descriptor_b, column, k)
              : k_major_f16_address(descriptor_b, column, k);
      uint16_t value = 0;
      if (!context.memory->read(memory_space::kShared, address, &value,
                                sizeof(value)))
        return memory_fault(inst, memory_space::kShared, address,
                            sizeof(value));
      matrix_b[k][column] = half_float(value);
    }
  }

  bool accumulates = scale_d == nullptr;
  if (scale_d != nullptr &&
      !read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid HGMMA scale-D predicate");
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned thread_id = context.thread_idx_x[lane] % 128;
    const unsigned row_in_octet = (thread_id / 4) % 8;
    const unsigned warpgroup_quarter = thread_id / 32;
    const unsigned column_pair = (thread_id % 4) * 2;
    for (unsigned output = 0; output < output_registers; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row =
          row_in_octet + 16 * warpgroup_quarter + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      float accumulator =
          accumulates
              ? bits_float(state.read_register(lane, c_base->index + output))
              : 0.0f;
      for (unsigned k = 0; k < 16; ++k)
        accumulator =
            std::fma(matrix_a[row][k], matrix_b[k][column], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hgmma_register_shared_wide_f32(const instruction &inst,
                                                   warp_state &state,
                                                   execution_context &context) {
  const unsigned n_extent = inst.opcode == "HGMMA.64x64x16.F32"    ? 64
                            : inst.opcode == "HGMMA.64x80x16.F32"  ? 80
                            : inst.opcode == "HGMMA.64x128x16.F32" ? 128
                            : inst.opcode == "HGMMA.64x176x16.F32" ? 176
                            : inst.opcode == "HGMMA.64x192x16.F32" ? 192
                            : inst.opcode == "HGMMA.64x256x16.F32" ? 256
                                                                   : 0;
  const unsigned output_registers = n_extent / 2;
  if (n_extent == 0 || inst.has_guard || inst.operands.size() < 4 ||
      inst.operands.size() > 6 ||
      !has_typed_operands(inst, inst.operands.size()))
    return unsupported(inst, "unsupported wide register/shared HGMMA form");
  const operand *destination = get_operand(inst, 0, operand_kind::kRegister);
  const operand *a_base = get_operand(inst, 1, operand_kind::kRegister);
  const operand *b_descriptor =
      get_operand(inst, 2, operand_kind::kGmmaDescriptor);
  const operand *c_base = get_operand(inst, 3, operand_kind::kRegister);
  const operand *scale_d = nullptr;
  const operand *scoreboard = nullptr;
  if (inst.operands.size() >= 5) {
    if (inst.operands[4].kind == operand_kind::kUniformPredicate)
      scale_d = &inst.operands[4];
    else if (inst.operands[4].kind == operand_kind::kScoreboardRegister)
      scoreboard = &inst.operands[4];
    else
      return unsupported(inst, "invalid wide register/shared HGMMA option");
  }
  if (inst.operands.size() == 6) {
    if (scale_d == nullptr ||
        inst.operands[5].kind != operand_kind::kScoreboardRegister)
      return unsupported(inst,
                         "invalid wide register/shared HGMMA scale/scoreboard");
    scoreboard = &inst.operands[5];
  }
  if (destination == nullptr || a_base == nullptr || b_descriptor == nullptr ||
      c_base == nullptr || (scoreboard != nullptr && scoreboard->index != 0) ||
      destination->negated || a_base->negated || c_base->negated ||
      destination->index > kZeroRegister - output_registers ||
      a_base->index > kZeroRegister - 4 ||
      (c_base->index != kZeroRegister &&
       c_base->index > kZeroRegister - output_registers) ||
      b_descriptor->descriptor_transpose_a ||
      b_descriptor->descriptor_register > kZeroUniformRegister - 3)
    return unsupported(inst, "invalid wide register/shared HGMMA operands");
  if (state.active_mask != 0xffffffffu)
    return unsupported(inst, "HGMMA requires a converged full warp");
  if (context.memory == nullptr)
    return unsupported(inst, "HGMMA requires functional shared memory");
  bool accumulates = c_base->index != kZeroRegister;
  if (scale_d != nullptr &&
      !read_predicate_operand(*scale_d, state, 0, accumulates))
    return unsupported(inst, "invalid wide register/shared scale-D predicate");
  accumulates = accumulates && c_base->index != kZeroRegister;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (!lane_executes(inst, state, lane))
      return unsupported(inst, "divergent HGMMA execution is not implemented");
  }

  const shared_descriptor descriptor_b = decode_shared_descriptor(
      read_uniform_register_pair(state, b_descriptor->descriptor_register + 2));
  const bool valid_leading = b_descriptor->descriptor_transpose_b
                                 ? descriptor_b.leading % 16 == 0
                                 : descriptor_b.leading == 16;
  if (descriptor_b.swizzle != 1 || !valid_leading ||
      descriptor_b.stride != 1024) {
    std::ostringstream detail;
    detail << "unsupported wide register/shared HGMMA descriptor: base="
           << descriptor_b.base << " leading=" << descriptor_b.leading
           << " stride=" << descriptor_b.stride
           << " swizzle=" << descriptor_b.swizzle;
    return unsupported(inst, detail.str());
  }

  float matrix_a[16][16]{};
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row = lane >> 2;
    for (unsigned reg = 0; reg < 4; ++reg) {
      const uint32_t packed = state.read_register(lane, a_base->index + reg);
      const unsigned matrix_row = row + ((reg & 1) != 0 ? 8 : 0);
      const unsigned matrix_k = (lane & 3) * 2 + ((reg & 2) != 0 ? 8 : 0);
      matrix_a[matrix_row][matrix_k] =
          half_float(static_cast<uint16_t>(packed));
      matrix_a[matrix_row][matrix_k + 1] =
          half_float(static_cast<uint16_t>(packed >> 16));
    }
  }
  float matrix_b[16][256]{};
  for (unsigned column = 0; column < n_extent; ++column) {
    for (unsigned k = 0; k < 16; ++k) {
      const uint64_t address =
          b_descriptor->descriptor_transpose_b
              ? mn_major_f16_address(descriptor_b, column, k)
              : k_major_f16_address(descriptor_b, column, k);
      uint16_t value = 0;
      if (!context.memory->read(memory_space::kShared, address, &value,
                                sizeof(value)))
        return memory_fault(inst, memory_space::kShared, address,
                            sizeof(value));
      matrix_b[k][column] = half_float(value);
    }
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const unsigned row_in_octet = lane / 4;
    const unsigned column_pair = (lane % 4) * 2;
    for (unsigned output = 0; output < output_registers; ++output) {
      const unsigned fragment = output % 4;
      const unsigned row = row_in_octet + 8 * ((fragment >> 1) & 1);
      const unsigned column = (output / 4) * 8 + column_pair + (fragment & 1);
      float accumulator =
          accumulates
              ? bits_float(state.read_register(lane, c_base->index + output))
              : 0.0f;
      for (unsigned k = 0; k < 16; ++k)
        accumulator =
            std::fma(matrix_a[row][k], matrix_b[k][column], accumulator);
      state.write_register(lane, destination->index + output,
                           float_bits(accumulator));
    }
  }
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_hgmma_wide_f32(const instruction &inst, warp_state &state,
                                   execution_context &context) {
  if (inst.operands.size() >= 2 &&
      inst.operands[1].kind == operand_kind::kRegister)
    return execute_hgmma_register_shared_wide_f32(inst, state, context);
  return execute_hgmma_shared_shared_f32(inst, state, context);
}

} // namespace

void register_wgmma_semantics(frontend &target) {
  target.register_semantics("WARPGROUP.ARRIVE", execute_warpgroup_arrive);
  target.register_semantics("WARPGROUP.DEPBAR.LE", execute_warpgroup_depbar);
  target.register_semantics("HGMMA.64x8x16.F16", execute_hgmma_commit_group);
  target.register_semantics("HGMMA.64x8x16.F32", execute_hgmma_64x8_f32);
  target.register_semantics("HGMMA.64x8x16.F32.BF16", execute_hgmma_64x8_f32);
  target.register_semantics("HGMMA.64x8x8.F32.TF32", execute_hgmma_64x8_f32);
  target.register_semantics("HGMMA.64x64x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("HGMMA.64x80x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("HGMMA.64x128x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("HGMMA.64x176x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("HGMMA.64x192x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("HGMMA.64x256x16.F32", execute_hgmma_wide_f32);
  target.register_semantics("QGMMA.64x8x32.F32.E4M3.E4M3",
                            execute_qgmma_64x8x32_f32);
  target.register_semantics("QGMMA.64x8x32.F32.E5M2.E5M2",
                            execute_qgmma_64x8x32_f32);
  target.register_semantics("QGMMA.64x8x32.F32.E4M3.E5M2",
                            execute_qgmma_64x8x32_f32);
  target.register_semantics("QGMMA.64x8x32.F32.E5M2.E4M3",
                            execute_qgmma_64x8x32_f32);
  target.register_semantics("IGMMA.64x8x32.S8.S8.SAT", execute_igmma_64x8x32);
  target.register_semantics("IGMMA.64x8x32.U8.U8", execute_igmma_64x8x32);
  target.register_semantics("IGMMA.64x8x32.S8.U8.SAT", execute_igmma_64x8x32);
  target.register_semantics("IGMMA.64x8x32.U8.S8", execute_igmma_64x8x32);
  target.register_semantics("BGMMA.64x8x256.AND.POPC", execute_bgmma_64x8x256);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
