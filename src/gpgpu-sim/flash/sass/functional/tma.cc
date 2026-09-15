#include "internal.h"

#include "../../tensormap.h"
#include "../../tma_reduction.h"
#include "../runtime/tensor_map.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
namespace {

// PTX ISA 9.7.9.26.5: tensor-memory asynchronous copies.

bool record_uniform_tma_effect(const instruction &inst, const warp_state &state,
                               execution_context &context,
                               functional_tma_effect effect) {
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if ((state.active_mask & (uint32_t{1} << lane)) != 0 &&
        lane_executes(inst, state, lane)) {
      effect.valid = true;
      context.record_tma_effect(lane, std::move(effect));
      return true;
    }
  }
  return false;
}

functional_tensor_map_view load_view(const functional_tensor_map_2d &descriptor,
                                     unsigned rank) {
  functional_tensor_map_view result;
  result.rank = rank;
  result.global_address = descriptor.global_address;
  result.element_bytes = descriptor.element_bytes;
  result.element_type = descriptor.element_type;
  result.global_dim[0] = descriptor.global_dim[0];
  result.global_dim[1] = descriptor.global_dim[1];
  result.box_dim[0] = descriptor.box_dim[0];
  result.box_dim[1] = descriptor.box_dim[1];
  result.global_stride_bytes[0] = descriptor.element_bytes;
  result.global_stride_bytes[1] = descriptor.row_stride_bytes;
  result.element_stride[0] = descriptor.element_stride[0];
  result.element_stride[1] = descriptor.element_stride[1];
  result.swizzle_bytes = descriptor.swizzle_bytes;
  result.oob_zero = descriptor.oob_zero;
  return result;
}

template <size_t Rank>
functional_tensor_map_view
load_view(const functional_tensor_map_nd<Rank> &descriptor) {
  static_assert(Rank <= 5, "functional TMA rank exceeds the load view");
  functional_tensor_map_view result;
  result.rank = Rank;
  result.global_address = descriptor.global_address;
  result.element_bytes = descriptor.element_bytes;
  result.element_type = descriptor.element_type;
  std::copy(descriptor.global_dim.begin(), descriptor.global_dim.end(),
            result.global_dim.begin());
  std::copy(descriptor.box_dim.begin(), descriptor.box_dim.end(),
            result.box_dim.begin());
  std::copy(descriptor.global_stride_bytes.begin(),
            descriptor.global_stride_bytes.end(),
            result.global_stride_bytes.begin());
  std::copy(descriptor.element_stride.begin(), descriptor.element_stride.end(),
            result.element_stride.begin());
  result.swizzle_bytes = descriptor.swizzle_bytes;
  result.oob_zero = descriptor.oob_zero;
  return result;
}

step_result execute_utmapf(const instruction &inst, warp_state &state,
                           execution_context &) {
  const unsigned rank = inst.opcode == "UTMAPF.L2.4D" ? 4 : 0;
  if (rank == 0)
    return unsupported(inst, "unsupported TMA prefetch rank or cache level");
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (!has_typed_operands(inst, 1) || addresses == nullptr ||
      addresses->has_prefix || addresses->address_groups.size() != 2)
    return unsupported(inst, "expected [URcoordinates][URdescriptor] operands");

  const address_expression &coordinates = addresses->address_groups[0];
  const address_expression &descriptor = addresses->address_groups[1];
  if (coordinates.terms.size() != 1 || descriptor.terms.size() != 1)
    return unsupported(
        inst, "TMA prefetch operands must each be one uniform register");
  const address_term &coordinate_term = coordinates.terms[0];
  const address_term &descriptor_term = descriptor.terms[0];
  if (coordinate_term.kind != address_term_kind::kUniformRegister ||
      descriptor_term.kind != address_term_kind::kUniformRegister ||
      coordinate_term.wide || descriptor_term.wide || coordinate_term.negated ||
      descriptor_term.negated ||
      coordinate_term.index > kUniformRegisters - rank ||
      descriptor_term.index > kUniformRegisters - 2)
    return unsupported(inst,
                       "unsupported TMA prefetch uniform-register convention");

  // UTMAPF only changes cache residency.  Functional execution validates the
  // coordinate/descriptor register convention, while the later timing
  // adapter is responsible for modeling the L2 prefetch itself.
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_utmaldg(const instruction &inst, warp_state &state,
                            execution_context &context) {
  const unsigned rank = inst.opcode == "UTMALDG.1D"   ? 1
                        : inst.opcode == "UTMALDG.2D" ? 2
                        : inst.opcode == "UTMALDG.3D" ? 3
                        : inst.opcode == "UTMALDG.4D" ? 4
                        : inst.opcode == "UTMALDG.5D" ? 5
                                                      : 0;
  if (rank == 0)
    return unsupported(inst, "unsupported TMA load rank");
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *address_descriptor =
      inst.operands.size() == 2
          ? get_operand(inst, 1, operand_kind::kTensorMapDescriptor)
          : nullptr;
  if ((!has_typed_operands(inst, 1) && !has_typed_operands(inst, 2)) ||
      addresses == nullptr || addresses->has_prefix ||
      addresses->address_groups.size() != 2)
    return unsupported(inst, "expected [URbase][URdescriptor] operands");
  if (inst.operands.size() == 2 && address_descriptor == nullptr)
    return unsupported(inst, "invalid TMA address-space descriptor");

  const address_expression &state_group = addresses->address_groups[0];
  const address_expression &descriptor_group = addresses->address_groups[1];
  if (state_group.terms.size() != 1 || descriptor_group.terms.size() != 1)
    return unsupported(inst, "TMA addresses must each be one uniform register");
  const address_term &state_term = state_group.terms[0];
  const address_term &descriptor_term = descriptor_group.terms[0];
  if (state_term.kind != address_term_kind::kUniformRegister ||
      descriptor_term.kind != address_term_kind::kUniformRegister ||
      state_term.wide || descriptor_term.wide || state_term.negated ||
      descriptor_term.negated ||
      state_term.index > kUniformRegisters - (rank + 2) ||
      descriptor_term.index > kUniformRegisters - 2)
    return unsupported(inst, "unsupported TMA uniform-register convention");

  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr || context.cta == nullptr)
    return unsupported(inst, "UTMALDG requires CTA memory and state");

  // UTMALDG is a uniform instruction: an executing warp issues one tensor
  // transfer regardless of how many active lanes reach the instruction.

  // The target Blackwell forms' canonical nvdisasm text elides consecutive
  // state registers: URbase is the shared destination, URbase+1 is the
  // mbarrier, and the following rank registers are signed coordinates.
  const unsigned state_base = state_term.index;
  const uint64_t shared_address = state.read_uniform_register(state_base);
  const uint64_t mbarrier_address = state.read_uniform_register(state_base + 1);
  std::array<int64_t, 5> coordinates{};
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    coordinates[dimension] = static_cast<int32_t>(
        state.read_uniform_register(state_base + 2 + dimension));
  const uint64_t descriptor_address =
      read_uniform_register_pair(state, descriptor_term.index);
  const memory_space descriptor_space = memory_space::kGlobal;
  if (address_descriptor != nullptr) {
    if (address_descriptor->descriptor_register > kUniformRegisters - 2)
      return unsupported(inst, "TMA address-space descriptor pair overflows");
    // CUDA 13.3 Hopper kernels use this qualifier with tensor maps in the
    // runtime-provided indirect parameter window. Keep the accepted value
    // narrow until another address-space form is validated.
    const uint64_t qualifier = read_uniform_register_pair(
        state, address_descriptor->descriptor_register);
    if (qualifier != 0x1000000000000000ull &&
        qualifier != 0x12f0000000000000ull &&
        qualifier != 0x14f0000000000000ull) {
      std::ostringstream detail;
      detail << "unvalidated Hopper TMA address qualifier 0x" << std::hex
             << qualifier;
      return unsupported(inst, detail.str());
    }
  }

  functional_tensor_map_view tensor_map;
  const functional_tensor_map_2d *registered_tensor_map =
      rank <= 2 ? context.cta->find_tensor_map_2d(descriptor_address) : nullptr;
  if (registered_tensor_map != nullptr) {
    tensor_map = load_view(*registered_tensor_map, rank);
  } else {
    std::array<uint64_t, 8> raw_words{};
    if (!context.memory->read(descriptor_space, descriptor_address,
                              raw_words.data(), sizeof(raw_words)))
      return memory_fault(inst, descriptor_space, descriptor_address,
                          sizeof(raw_words));
    if (rank == 5) {
      const tensor_map_5d_decode_result decoded =
          decode_sm120_tensor_map_5d(raw_words);
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor);
    } else if (rank == 4) {
      std::array<uint64_t, 16> host_words{};
      std::copy(raw_words.begin(), raw_words.end(), host_words.begin());
      tensor_map_4d_decode_result decoded =
          decode_sm120_tensor_map_4d(raw_words);
      if (!decoded.supported && address_descriptor != nullptr) {
        if (!context.memory->read(
                descriptor_space, descriptor_address + sizeof(raw_words),
                host_words.data() + raw_words.size(), sizeof(raw_words)))
          return memory_fault(inst, descriptor_space,
                              descriptor_address + sizeof(raw_words),
                              sizeof(raw_words));
        decoded = decode_flashgpu_host_tensor_map_4d(host_words);
      }
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor);
    } else if (rank == 3) {
      const tensor_map_3d_decode_result decoded =
          decode_sm120_tensor_map_3d(raw_words);
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor);
    } else {
      tensor_map_decode_result decoded =
          rank == 1 ? decode_sm120_tensor_map_1d(raw_words)
                    : decode_sm120_tensor_map_2d(raw_words);
      if (!decoded.supported && rank == 1) {
        std::array<uint64_t, 16> host_words{};
        std::copy(raw_words.begin(), raw_words.end(), host_words.begin());
        if (!context.memory->read(
                descriptor_space, descriptor_address + sizeof(raw_words),
                host_words.data() + raw_words.size(), sizeof(raw_words)))
          return memory_fault(inst, descriptor_space,
                              descriptor_address + sizeof(raw_words),
                              sizeof(raw_words));
        decoded = decode_flashgpu_host_tensor_map_1d(host_words);
      }
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor, rank);
    }
  }
  if (tensor_map.swizzle_bytes != 0 && tensor_map.swizzle_bytes != 32 &&
      tensor_map.swizzle_bytes != 64 && tensor_map.swizzle_bytes != 128)
    return unsupported(inst, "unsupported shared-memory TMA swizzle");

  uint64_t box_elements = 1;
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    if (tensor_map.box_dim[dimension] == 0 ||
        box_elements > std::numeric_limits<uint64_t>::max() /
                           tensor_map.box_dim[dimension])
      return unsupported(inst, "TMA tile element count overflows");
    box_elements *= tensor_map.box_dim[dimension];
  }
  if (tensor_map.element_bytes == 0 ||
      box_elements >
          std::numeric_limits<size_t>::max() / tensor_map.element_bytes)
    return unsupported(inst, "TMA tile byte count overflows host size_t");
  const size_t tile_bytes =
      static_cast<size_t>(box_elements * tensor_map.element_bytes);
  if (tile_bytes == 0 || tile_bytes > UINT32_MAX ||
      shared_address > std::numeric_limits<uint64_t>::max() - (tile_bytes - 1))
    return unsupported(inst, "invalid TMA shared-memory tile range");
  if (!context.cta->mbarrier_initialized(mbarrier_address) ||
      context.cta->mbarrier_pending_transaction_bytes(mbarrier_address) <
          tile_bytes)
    return unsupported(inst, "TMA bytes are not pending on its mbarrier");

  std::vector<uint8_t> tile(tile_bytes, 0);
  for (uint64_t element = 0; element < box_elements; ++element) {
    uint64_t remaining = element;
    uint64_t global_offset = 0;
    bool in_bounds = true;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      const uint64_t local_coordinate =
          remaining % tensor_map.box_dim[dimension];
      remaining /= tensor_map.box_dim[dimension];
      if (local_coordinate != 0 &&
          tensor_map.element_stride[dimension] >
              std::numeric_limits<uint64_t>::max() / local_coordinate) {
        in_bounds = false;
        break;
      }
      const uint64_t delta =
          local_coordinate * tensor_map.element_stride[dimension];
      if (delta > static_cast<uint64_t>(INT64_MAX) ||
          coordinates[dimension] > INT64_MAX - static_cast<int64_t>(delta)) {
        in_bounds = false;
        break;
      }
      const int64_t global_coordinate =
          coordinates[dimension] + static_cast<int64_t>(delta);
      if (global_coordinate < 0 || static_cast<uint64_t>(global_coordinate) >=
                                       tensor_map.global_dim[dimension]) {
        in_bounds = false;
        break;
      }
      const uint64_t unsigned_coordinate =
          static_cast<uint64_t>(global_coordinate);
      if (unsigned_coordinate != 0 &&
          tensor_map.global_stride_bytes[dimension] >
              std::numeric_limits<uint64_t>::max() / unsigned_coordinate)
        return unsupported(inst, "TMA global-memory address overflow");
      const uint64_t dimension_offset =
          unsigned_coordinate * tensor_map.global_stride_bytes[dimension];
      if (global_offset >
          std::numeric_limits<uint64_t>::max() - dimension_offset)
        return unsupported(inst, "TMA global-memory address overflow");
      global_offset += dimension_offset;
    }
    if (!in_bounds) {
      if (!tensor_map.oob_zero)
        return unsupported(inst, "out-of-bounds TMA coordinate");
      continue;
    }
    if (tensor_map.global_address >
        std::numeric_limits<uint64_t>::max() - global_offset)
      return unsupported(inst, "TMA global-memory address overflow");
    const uint64_t global_address = tensor_map.global_address + global_offset;
    const size_t tile_offset =
        static_cast<size_t>(element * tensor_map.element_bytes);
    if (!context.memory->read(memory_space::kGlobal, global_address,
                              tile.data() + tile_offset,
                              tensor_map.element_bytes))
      return memory_fault(inst, memory_space::kGlobal, global_address,
                          tensor_map.element_bytes);
  }

  std::vector<uint8_t> shared_tile;
  const uint8_t *shared_data = tile.data();
  if (tensor_map.swizzle_bytes != 0) {
    shared_tile.resize(tile.size(), 0);
    const uint64_t mask = tensor_map.swizzle_bytes == 128  ? 0x7
                          : tensor_map.swizzle_bytes == 64 ? 0x3
                                                           : 0x1;
    constexpr size_t kSwizzleGranularity = 16;
    for (size_t offset = 0; offset < tile.size();
         offset += kSwizzleGranularity) {
      const size_t bytes = std::min(kSwizzleGranularity, tile.size() - offset);
      const uint64_t swizzled_offset =
          offset ^ (((static_cast<uint64_t>(offset) >> 7) & mask) << 4);
      if (swizzled_offset > tile.size() ||
          bytes > tile.size() - swizzled_offset)
        return unsupported(inst, "TMA swizzle escapes the shared tile");
      std::memcpy(shared_tile.data() + swizzled_offset, tile.data() + offset,
                  bytes);
    }
    shared_data = shared_tile.data();
  }
  if (!context.memory->write(memory_space::kShared, shared_address, shared_data,
                             tile.size()))
    return memory_fault(inst, memory_space::kShared, shared_address,
                        tile.size());
  functional_tma_effect timing_effect;
  timing_effect.destination_address = shared_address;
  timing_effect.source_address = descriptor_address;
  timing_effect.size_bytes = static_cast<uint32_t>(tile_bytes);
  timing_effect.mbarrier_address = static_cast<uint32_t>(mbarrier_address);
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    timing_effect.coordinates[dimension] =
        static_cast<int32_t>(coordinates[dimension]);
  timing_effect.tensor_map = tensor_map;
  if (!record_uniform_tma_effect(inst, state, context,
                                 std::move(timing_effect)))
    return unsupported(inst, "executing TMA load has no issuing lane");
  if (!context.cta->complete_mbarrier_transaction(mbarrier_address,
                                                  tile.size()))
    return unsupported(inst, "mbarrier transaction state changed during TMA");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_utmastg(const instruction &inst, warp_state &state,
                            execution_context &context) {
  const bool reduce_add = inst.opcode == "UTMAREDG.2D.ADD";
  const unsigned rank = inst.opcode == "UTMASTG.1D"   ? 1
                        : inst.opcode == "UTMASTG.2D" ? 2
                        : inst.opcode == "UTMASTG.4D" ? 4
                        : inst.opcode == "UTMASTG.5D" ? 5
                        : reduce_add                  ? 2
                                                      : 0;
  if (rank == 0)
    return unsupported(inst, "unsupported TMA store rank");
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  if (!has_typed_operands(inst, 1) || addresses == nullptr ||
      addresses->has_prefix || addresses->address_groups.size() != 2)
    return unsupported(inst, "expected [URbase][URdescriptor] operands");

  const address_expression &state_group = addresses->address_groups[0];
  const address_expression &descriptor_group = addresses->address_groups[1];
  if (state_group.terms.size() != 1 || descriptor_group.terms.size() != 1)
    return unsupported(inst, "TMA addresses must each be one uniform register");
  const address_term &state_term = state_group.terms[0];
  const address_term &descriptor_term = descriptor_group.terms[0];
  if (state_term.kind != address_term_kind::kUniformRegister ||
      descriptor_term.kind != address_term_kind::kUniformRegister ||
      state_term.wide || descriptor_term.wide || state_term.negated ||
      descriptor_term.negated ||
      state_term.index > kUniformRegisters - (rank + 1) ||
      descriptor_term.index > kUniformRegisters - 2)
    return unsupported(inst, "unsupported TMA uniform-register convention");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr || context.cta == nullptr)
    return unsupported(inst, "UTMASTG requires CTA memory and state");

  // Store state has no mbarrier: URbase is the shared source and the following
  // rank uniform registers are the signed tensor coordinates.
  const unsigned state_base = state_term.index;
  const uint64_t shared_address = state.read_uniform_register(state_base);
  std::array<int64_t, 5> coordinates{};
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    coordinates[dimension] = static_cast<int32_t>(
        state.read_uniform_register(state_base + 1 + dimension));
  const uint64_t descriptor_address =
      read_uniform_register_pair(state, descriptor_term.index);
  functional_tensor_map_view tensor_map;
  const functional_tensor_map_2d *registered_tensor_map =
      rank <= 2 ? context.cta->find_tensor_map_2d(descriptor_address) : nullptr;
  if (registered_tensor_map != nullptr) {
    tensor_map = load_view(*registered_tensor_map, rank);
  } else {
    std::array<uint64_t, 8> raw_words{};
    if (!context.memory->read(memory_space::kGlobal, descriptor_address,
                              raw_words.data(), sizeof(raw_words)))
      return memory_fault(inst, memory_space::kGlobal, descriptor_address,
                          sizeof(raw_words));
    if (rank == 5) {
      tensor_map_5d_decode_result decoded =
          decode_sm120_tensor_map_5d(raw_words);
      std::array<uint64_t, 16> host_words{};
      if (!decoded.supported) {
        std::copy(raw_words.begin(), raw_words.end(), host_words.begin());
        if (!context.memory->read(
                memory_space::kGlobal, descriptor_address + sizeof(raw_words),
                host_words.data() + raw_words.size(), sizeof(raw_words)))
          return memory_fault(inst, memory_space::kGlobal,
                              descriptor_address + sizeof(raw_words),
                              sizeof(raw_words));
        decoded = decode_flashgpu_host_tensor_map_5d(host_words);
      }
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor);
    } else if (rank == 4) {
      tensor_map_4d_decode_result decoded =
          decode_sm120_tensor_map_4d(raw_words);
      if (!decoded.supported) {
        std::array<uint64_t, 16> host_words{};
        std::copy(raw_words.begin(), raw_words.end(), host_words.begin());
        if (!context.memory->read(
                memory_space::kGlobal, descriptor_address + sizeof(raw_words),
                host_words.data() + raw_words.size(), sizeof(raw_words)))
          return memory_fault(inst, memory_space::kGlobal,
                              descriptor_address + sizeof(raw_words),
                              sizeof(raw_words));
        decoded = decode_flashgpu_host_tensor_map_4d(host_words);
      }
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor);
    } else {
      tensor_map_decode_result decoded =
          rank == 1 ? decode_sm120_tensor_map_1d(raw_words)
                    : decode_sm120_tensor_map_2d(raw_words);
      if (!decoded.supported) {
        std::array<uint64_t, 16> host_words{};
        std::copy(raw_words.begin(), raw_words.end(), host_words.begin());
        if (!context.memory->read(
                memory_space::kGlobal, descriptor_address + sizeof(raw_words),
                host_words.data() + raw_words.size(), sizeof(raw_words)))
          return memory_fault(inst, memory_space::kGlobal,
                              descriptor_address + sizeof(raw_words),
                              sizeof(raw_words));
        decoded = rank == 1 ? decode_flashgpu_host_tensor_map_1d(host_words)
                            : decode_flashgpu_host_tensor_map_2d(host_words);
      }
      if (!decoded.supported)
        return unsupported(inst, decoded.detail);
      tensor_map = load_view(decoded.descriptor, rank);
    }
  }
  if (tensor_map.swizzle_bytes != 0 && tensor_map.swizzle_bytes != 32 &&
      tensor_map.swizzle_bytes != 64 && tensor_map.swizzle_bytes != 128)
    return unsupported(inst, "unsupported shared-memory TMA swizzle");
  if (reduce_add &&
      (tensor_map.element_bytes != sizeof(float) ||
       (tensor_map.element_type != tensor_map_element_type::kFloat32 &&
        tensor_map.element_type != tensor_map_element_type::kFloat32Ftz)))
    return unsupported(inst,
                       "UTMAREDG.2D.ADD requires a 4-byte F32 tensor map");

  uint64_t box_elements = 1;
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    if (tensor_map.box_dim[dimension] == 0 ||
        box_elements > std::numeric_limits<uint64_t>::max() /
                           tensor_map.box_dim[dimension])
      return unsupported(inst, "TMA tile element count overflows");
    box_elements *= tensor_map.box_dim[dimension];
  }
  if (tensor_map.element_bytes == 0 ||
      box_elements >
          std::numeric_limits<size_t>::max() / tensor_map.element_bytes)
    return unsupported(inst, "TMA tile byte count overflows host size_t");
  const size_t tile_bytes =
      static_cast<size_t>(box_elements * tensor_map.element_bytes);
  if (tile_bytes == 0 || tile_bytes > UINT32_MAX ||
      shared_address > std::numeric_limits<uint64_t>::max() - (tile_bytes - 1))
    return unsupported(inst, "invalid TMA shared-memory tile range");

  std::vector<uint8_t> shared_tile(tile_bytes);
  if (!context.memory->read(memory_space::kShared, shared_address,
                            shared_tile.data(), shared_tile.size()))
    return memory_fault(inst, memory_space::kShared, shared_address,
                        shared_tile.size());
  std::vector<uint8_t> tile(tile_bytes);
  if (tensor_map.swizzle_bytes == 0) {
    tile = shared_tile;
  } else {
    const uint64_t mask = tensor_map.swizzle_bytes == 128  ? 0x7
                          : tensor_map.swizzle_bytes == 64 ? 0x3
                                                           : 0x1;
    constexpr size_t kSwizzleGranularity = 16;
    for (size_t offset = 0; offset < tile.size();
         offset += kSwizzleGranularity) {
      const size_t bytes = std::min(kSwizzleGranularity, tile.size() - offset);
      const uint64_t swizzled_offset =
          offset ^ (((static_cast<uint64_t>(offset) >> 7) & mask) << 4);
      if (swizzled_offset > shared_tile.size() ||
          bytes > shared_tile.size() - swizzled_offset)
        return unsupported(inst, "TMA swizzle escapes the shared tile");
      std::memcpy(tile.data() + offset, shared_tile.data() + swizzled_offset,
                  bytes);
    }
  }

  for (uint64_t element = 0; element < box_elements; ++element) {
    uint64_t remaining = element;
    uint64_t global_offset = 0;
    bool in_bounds = true;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      const uint64_t local_coordinate =
          remaining % tensor_map.box_dim[dimension];
      remaining /= tensor_map.box_dim[dimension];
      if (local_coordinate > std::numeric_limits<uint64_t>::max() /
                                 tensor_map.element_stride[dimension])
        return unsupported(inst, "TMA element-coordinate overflow");
      const uint64_t delta =
          local_coordinate * tensor_map.element_stride[dimension];
      if (delta > static_cast<uint64_t>(INT64_MAX) ||
          coordinates[dimension] > INT64_MAX - static_cast<int64_t>(delta)) {
        in_bounds = false;
        break;
      }
      const int64_t global_coordinate =
          coordinates[dimension] + static_cast<int64_t>(delta);
      if (global_coordinate < 0 || static_cast<uint64_t>(global_coordinate) >=
                                       tensor_map.global_dim[dimension]) {
        in_bounds = false;
        break;
      }
      const uint64_t unsigned_coordinate =
          static_cast<uint64_t>(global_coordinate);
      const uint64_t stride = tensor_map.global_stride_bytes[dimension];
      if (unsigned_coordinate != 0 &&
          stride > std::numeric_limits<uint64_t>::max() / unsigned_coordinate)
        return unsupported(inst, "TMA global-memory address overflow");
      const uint64_t dimension_offset = unsigned_coordinate * stride;
      if (global_offset >
          std::numeric_limits<uint64_t>::max() - dimension_offset)
        return unsupported(inst, "TMA global-memory address overflow");
      global_offset += dimension_offset;
    }
    if (!in_bounds)
      continue;
    if (tensor_map.global_address >
        std::numeric_limits<uint64_t>::max() - global_offset)
      return unsupported(inst, "TMA global-memory address overflow");
    const uint64_t global_address = tensor_map.global_address + global_offset;
    const size_t tile_offset =
        static_cast<size_t>(element * tensor_map.element_bytes);
    if (reduce_add) {
      std::lock_guard<std::mutex> lock(global_atomic_mutex);
      std::array<uint8_t, sizeof(float)> destination{};
      if (!context.memory->read(memory_space::kGlobal, global_address,
                                destination.data(), destination.size()))
        return memory_fault(inst, memory_space::kGlobal, global_address,
                            destination.size());
      const uint32_t data_type =
          tensor_map.element_type == tensor_map_element_type::kFloat32Ftz
              ? TMA_DTYPE_F32_FTZ
              : TMA_DTYPE_F32;
      apply_tma_tensor_reduction(tma_reduction_op_t::ADD, data_type,
                                 destination.data(), tile.data() + tile_offset,
                                 destination.size());
      if (!context.memory->write(memory_space::kGlobal, global_address,
                                 destination.data(), destination.size()))
        return memory_fault(inst, memory_space::kGlobal, global_address,
                            destination.size());
    } else if (!context.memory->write(memory_space::kGlobal, global_address,
                                      tile.data() + tile_offset,
                                      tensor_map.element_bytes)) {
      return memory_fault(inst, memory_space::kGlobal, global_address,
                          tensor_map.element_bytes);
    }
  }
  functional_tma_effect timing_effect;
  timing_effect.destination_address = descriptor_address;
  timing_effect.source_address = shared_address;
  timing_effect.size_bytes = static_cast<uint32_t>(tile_bytes);
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    timing_effect.coordinates[dimension] =
        static_cast<int32_t>(coordinates[dimension]);
  timing_effect.tensor_map = tensor_map;
  if (!record_uniform_tma_effect(inst, state, context,
                                 std::move(timing_effect)))
    return unsupported(inst, "executing TMA store has no issuing lane");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ublkcp_global_shared(const instruction &inst,
                                         warp_state &state,
                                         execution_context &context) {
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *size = get_operand(inst, 1, operand_kind::kUniformRegister);
  if (!has_typed_operands(inst, 2) || addresses == nullptr || size == nullptr ||
      addresses->has_prefix || addresses->address_groups.size() != 2)
    return unsupported(inst, "expected [URglobal][URshared], URsize operands");
  const address_expression &global_group = addresses->address_groups[0];
  const address_expression &shared_group = addresses->address_groups[1];
  if (global_group.terms.size() != 1 || shared_group.terms.size() != 1)
    return unsupported(inst, "UBLKCP addresses must be uniform registers");
  const address_term &global_term = global_group.terms[0];
  const address_term &shared_term = shared_group.terms[0];
  if (global_term.kind != address_term_kind::kUniformRegister ||
      shared_term.kind != address_term_kind::kUniformRegister ||
      global_term.wide || shared_term.wide || global_term.negated ||
      shared_term.negated || global_term.index > kUniformRegisters - 2)
    return unsupported(inst, "unsupported UBLKCP uniform-register convention");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr)
    return unsupported(inst, "UBLKCP requires functional memory");

  const uint64_t global_address =
      read_uniform_register_pair(state, global_term.index);
  const uint64_t shared_address =
      state.read_uniform_register(shared_term.index);
  const uint64_t size_units = state.read_uniform_register(size->index);
  if (size_units == 0 || size_units > std::numeric_limits<size_t>::max() / 16)
    return unsupported(inst, "invalid UBLKCP transfer size");
  const size_t bytes = static_cast<size_t>(size_units * 16);
  if ((global_address % 16) != 0 || (shared_address % 16) != 0)
    return unsupported(inst, "UBLKCP addresses must be 16-byte aligned");

  std::vector<uint8_t> payload(bytes);
  if (!context.memory->read(memory_space::kShared, shared_address,
                            payload.data(), payload.size()))
    return memory_fault(inst, memory_space::kShared, shared_address,
                        payload.size());
  if (!context.memory->write(memory_space::kGlobal, global_address,
                             payload.data(), payload.size()))
    return memory_fault(inst, memory_space::kGlobal, global_address,
                        payload.size());
  functional_tma_effect timing_effect;
  timing_effect.destination_address = global_address;
  timing_effect.source_address = shared_address;
  timing_effect.size_bytes = static_cast<uint32_t>(bytes);
  if (!record_uniform_tma_effect(inst, state, context,
                                 std::move(timing_effect)))
    return unsupported(inst,
                       "executing uniform bulk store has no issuing lane");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ublkred_global_shared_add_f32(const instruction &inst,
                                                  warp_state &state,
                                                  execution_context &context) {
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *size = get_operand(inst, 1, operand_kind::kUniformRegister);
  const operand *address_descriptor =
      get_operand(inst, 2, operand_kind::kTensorMapDescriptor);
  if (!has_typed_operands(inst, 3) || addresses == nullptr || size == nullptr ||
      address_descriptor == nullptr || addresses->has_prefix ||
      addresses->address_groups.size() != 2)
    return unsupported(
        inst, "expected [URglobal][URshared], URsize, desc[UR] operands");
  const address_expression &global_group = addresses->address_groups[0];
  const address_expression &shared_group = addresses->address_groups[1];
  if (global_group.terms.size() != 1 || shared_group.terms.size() != 1)
    return unsupported(inst, "UBLKRED addresses must be uniform registers");
  const address_term &global_term = global_group.terms[0];
  const address_term &shared_term = shared_group.terms[0];
  if (global_term.kind != address_term_kind::kUniformRegister ||
      shared_term.kind != address_term_kind::kUniformRegister ||
      global_term.wide || shared_term.wide || global_term.negated ||
      shared_term.negated || global_term.index > kUniformRegisters - 2 ||
      address_descriptor->descriptor_register > kUniformRegisters - 2)
    return unsupported(inst, "unsupported UBLKRED uniform-register convention");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr)
    return unsupported(inst, "UBLKRED requires functional memory");

  const uint64_t qualifier = read_uniform_register_pair(
      state, address_descriptor->descriptor_register);
  if (qualifier != 0x1000000000000000ull &&
      qualifier != 0x12f0000000000000ull &&
      qualifier != 0x14f0000000000000ull) {
    std::ostringstream detail;
    detail << "unvalidated Hopper bulk-reduction address qualifier 0x"
           << std::hex << qualifier;
    return unsupported(inst, detail.str());
  }
  const uint64_t global_address =
      read_uniform_register_pair(state, global_term.index);
  const uint64_t shared_address =
      state.read_uniform_register(shared_term.index);
  const uint64_t size_units = state.read_uniform_register(size->index);
  if (size_units == 0 || size_units > std::numeric_limits<size_t>::max() / 16)
    return unsupported(inst, "invalid UBLKRED transfer size");
  const size_t bytes = static_cast<size_t>(size_units * 16);
  if ((global_address % 16) != 0 || (shared_address % 16) != 0)
    return unsupported(inst, "UBLKRED addresses must be 16-byte aligned");

  std::vector<uint8_t> source(bytes);
  std::vector<uint8_t> destination(bytes);
  if (!context.memory->read(memory_space::kShared, shared_address,
                            source.data(), source.size()))
    return memory_fault(inst, memory_space::kShared, shared_address,
                        source.size());
  std::lock_guard<std::mutex> lock(global_atomic_mutex);
  if (!context.memory->read(memory_space::kGlobal, global_address,
                            destination.data(), destination.size()))
    return memory_fault(inst, memory_space::kGlobal, global_address,
                        destination.size());
  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_F32,
                             destination.data(), source.data(), bytes);
  if (!context.memory->write(memory_space::kGlobal, global_address,
                             destination.data(), destination.size()))
    return memory_fault(inst, memory_space::kGlobal, global_address,
                        destination.size());
  functional_tma_effect timing_effect;
  timing_effect.destination_address = global_address;
  timing_effect.source_address = shared_address;
  timing_effect.size_bytes = static_cast<uint32_t>(bytes);
  if (!record_uniform_tma_effect(inst, state, context,
                                 std::move(timing_effect)))
    return unsupported(inst,
                       "executing uniform bulk reduction has no issuing lane");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

step_result execute_ublkcp_shared_global(const instruction &inst,
                                         warp_state &state,
                                         execution_context &context) {
  const operand *addresses = get_operand(inst, 0, operand_kind::kMemoryAddress);
  const operand *size = get_operand(inst, 1, operand_kind::kUniformRegister);
  if (!has_typed_operands(inst, 2) || addresses == nullptr || size == nullptr ||
      addresses->has_prefix || addresses->address_groups.size() != 2)
    return unsupported(inst, "expected [URstate][URglobal], URsize operands");
  const address_expression &state_group = addresses->address_groups[0];
  const address_expression &global_group = addresses->address_groups[1];
  if (state_group.terms.size() != 1 || global_group.terms.size() != 1)
    return unsupported(inst, "UBLKCP addresses must be uniform registers");
  const address_term &state_term = state_group.terms[0];
  const address_term &global_term = global_group.terms[0];
  if (state_term.kind != address_term_kind::kUniformRegister ||
      global_term.kind != address_term_kind::kUniformRegister ||
      state_term.wide || global_term.wide || state_term.negated ||
      global_term.negated || state_term.index > kUniformRegisters - 2 ||
      global_term.index > kUniformRegisters - 2)
    return unsupported(inst, "unsupported UBLKCP uniform-register convention");
  if (!any_lane_executes(inst, state)) {
    advance(inst, state);
    return {step_status::kAdvanced, {}};
  }
  if (context.memory == nullptr || context.cta == nullptr)
    return unsupported(inst, "UBLKCP.S.G requires CTA memory and state");

  const uint64_t shared_address = state.read_uniform_register(state_term.index);
  const uint64_t mbarrier_address =
      state.read_uniform_register(state_term.index + 1);
  const uint64_t global_address =
      read_uniform_register_pair(state, global_term.index);
  const uint64_t size_units = state.read_uniform_register(size->index);
  if (size_units == 0 || size_units > std::numeric_limits<size_t>::max() / 16)
    return unsupported(inst, "invalid UBLKCP transfer size");
  const size_t bytes = static_cast<size_t>(size_units * 16);
  if ((global_address % 16) != 0 || (shared_address % 16) != 0)
    return unsupported(inst, "UBLKCP addresses must be 16-byte aligned");
  if (!context.cta->mbarrier_initialized(mbarrier_address) ||
      context.cta->mbarrier_pending_transaction_bytes(mbarrier_address) < bytes)
    return unsupported(inst, "UBLKCP bytes are not pending on its mbarrier");

  std::vector<uint8_t> payload(bytes);
  if (!context.memory->read(memory_space::kGlobal, global_address,
                            payload.data(), payload.size()))
    return memory_fault(inst, memory_space::kGlobal, global_address,
                        payload.size());
  if (!context.memory->write(memory_space::kShared, shared_address,
                             payload.data(), payload.size()))
    return memory_fault(inst, memory_space::kShared, shared_address,
                        payload.size());
  functional_tma_effect timing_effect;
  timing_effect.destination_address = shared_address;
  timing_effect.source_address = global_address;
  timing_effect.size_bytes = static_cast<uint32_t>(bytes);
  timing_effect.mbarrier_address = static_cast<uint32_t>(mbarrier_address);
  if (!record_uniform_tma_effect(inst, state, context,
                                 std::move(timing_effect)))
    return unsupported(inst, "executing uniform bulk load has no issuing lane");
  if (!context.cta->complete_mbarrier_transaction(mbarrier_address, bytes))
    return unsupported(inst, "mbarrier state changed during UBLKCP");
  advance(inst, state);
  return {step_status::kAdvanced, {}};
}

} // namespace

void register_bulk_copy_semantics(frontend &target) {
  target.register_semantics("UBLKCP.G.S", execute_ublkcp_global_shared);
  target.register_semantics("UBLKCP.S.G", execute_ublkcp_shared_global);
  target.register_semantics("UBLKRED.G.S.ADD.F32.RN",
                            execute_ublkred_global_shared_add_f32);
}

void register_tma_semantics(frontend &target) {
  target.register_semantics("UTMAPF.L2.4D", execute_utmapf);
  target.register_semantics("UTMALDG.1D", execute_utmaldg);
  target.register_semantics("UTMALDG.2D", execute_utmaldg);
  target.register_semantics("UTMALDG.3D", execute_utmaldg);
  target.register_semantics("UTMALDG.4D", execute_utmaldg);
  target.register_semantics("UTMALDG.5D", execute_utmaldg);
  target.register_semantics("UTMASTG.1D", execute_utmastg);
  target.register_semantics("UTMASTG.2D", execute_utmastg);
  target.register_semantics("UTMASTG.4D", execute_utmastg);
  target.register_semantics("UTMASTG.5D", execute_utmastg);
  target.register_semantics("UTMAREDG.2D.ADD", execute_utmastg);
  register_bulk_copy_semantics(target);
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
