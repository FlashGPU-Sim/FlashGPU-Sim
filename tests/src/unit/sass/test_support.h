#pragma once

#include <gtest/gtest.h>

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "abstract_hardware_model.h"
#include "gpgpu-sim/flash/async_proxy_timing.h"
#include "gpgpu-sim/flash/cta_barrier_timing.h"
#include "gpgpu-sim/flash/instruction_dependency_tracker.h"
#include "gpgpu-sim/flash/instruction_fetch_timing.h"
#include "gpgpu-sim/flash/mio_ldsm_timing.h"
#include "gpgpu-sim/flash/sass/decode/operand_parser.h"
#include "gpgpu-sim/flash/sass/decode/sassir_decoder.h"
#include "gpgpu-sim/flash/sass/decode/sm120_control.h"
#include "gpgpu-sim/flash/sass/decode/sm90_control.h"
#include "gpgpu-sim/flash/sass/frontend.h"
#include "gpgpu-sim/flash/sass/functional/functional.h"
#include "gpgpu-sim/flash/sass/runtime/tensor_map.h"
#include "gpgpu-sim/flash/sass/timing/timing_projection.h"
#include "gpgpu-sim/flash/shared_proxy_timing.h"
#include "gpgpu-sim/flash/tensor_core_admission_timing.h"
#include "gpgpu-sim/flash/tensormap.h"

namespace {

using flash_gpgpu_sim::async_proxy_timing;
using flash_gpgpu_sim::instruction_dependency_tracker;
using flash_gpgpu_sim::instruction_loop_buffer_config;
using flash_gpgpu_sim::instruction_loop_refill_delay;
using flash_gpgpu_sim::shared_proxy_timing;
using flash_gpgpu_sim::sass::address_term_kind;
using flash_gpgpu_sim::sass::architecture;
using flash_gpgpu_sim::sass::cta_execution_state;
using flash_gpgpu_sim::sass::cta_executor;
using flash_gpgpu_sim::sass::decode_flashgpu_host_tensor_map_1d;
using flash_gpgpu_sim::sass::decode_flashgpu_host_tensor_map_2d;
using flash_gpgpu_sim::sass::decode_flashgpu_host_tensor_map_4d;
using flash_gpgpu_sim::sass::decode_flashgpu_host_tensor_map_5d;
using flash_gpgpu_sim::sass::decode_sm120_control;
using flash_gpgpu_sim::sass::decode_sm120_tensor_map_1d;
using flash_gpgpu_sim::sass::decode_sm120_tensor_map_2d;
using flash_gpgpu_sim::sass::decode_sm120_tensor_map_3d;
using flash_gpgpu_sim::sass::decode_sm120_tensor_map_4d;
using flash_gpgpu_sim::sass::decode_sm120_tensor_map_5d;
using flash_gpgpu_sim::sass::decode_sm90_control;
using flash_gpgpu_sim::sass::execution_context;
using flash_gpgpu_sim::sass::frontend;
using flash_gpgpu_sim::sass::functional_memory;
using flash_gpgpu_sim::sass::functional_tensor_map_2d;
using flash_gpgpu_sim::sass::instruction;
using flash_gpgpu_sim::sass::kernel;
using flash_gpgpu_sim::sass::mbarrier_try_wait_state;
using flash_gpgpu_sim::sass::memory_space;
using flash_gpgpu_sim::sass::operand_kind;
using flash_gpgpu_sim::sass::operand_matrix_layout;
using flash_gpgpu_sim::sass::parse_instruction_operands;
using flash_gpgpu_sim::sass::parse_operand;
using flash_gpgpu_sim::sass::parse_timing_profile;
using flash_gpgpu_sim::sass::register_sm120_functional_semantics;
using flash_gpgpu_sim::sass::register_sm90_functional_semantics;
using flash_gpgpu_sim::sass::sassir_decoder;
using flash_gpgpu_sim::sass::split_operand_text;
using flash_gpgpu_sim::sass::step_result;
using flash_gpgpu_sim::sass::step_status;
using flash_gpgpu_sim::sass::tensor_map_element_type;
using flash_gpgpu_sim::sass::timing_instruction;
using flash_gpgpu_sim::sass::timing_profile;
using flash_gpgpu_sim::sass::timing_profile_options;
using flash_gpgpu_sim::sass::warp_state;

#ifndef SASS_GENERATED_SASSIR_DIR
#define SASS_GENERATED_SASSIR_DIR "build/generated/sass-unit"
#endif

class mapped_functional_memory final : public functional_memory {
 public:
  struct region {
    memory_space space;
    uint64_t base;
    std::vector<unsigned char> bytes;
    bool writable;
  };

  void map(memory_space space, uint64_t base, size_t bytes, bool writable) {
    regions_.push_back(
        {space, base, std::vector<unsigned char>(bytes), writable});
  }

  template <typename T>
  void initialize(memory_space space, uint64_t address, const T &value) {
    region *target = find(space, address, sizeof(value));
    ASSERT_NE(target, nullptr);
    std::memcpy(target->bytes.data() + (address - target->base), &value,
                sizeof(value));
  }

  template <typename T>
  T inspect(memory_space space, uint64_t address) const {
    const region *target = find(space, address, sizeof(T));
    EXPECT_NE(target, nullptr);
    T value{};
    if (target != nullptr)
      std::memcpy(&value, target->bytes.data() + (address - target->base),
                  sizeof(value));
    return value;
  }

  bool read(memory_space space, uint64_t address, void *data,
            size_t bytes) const override {
    const region *source = find(space, address, bytes);
    if (source == nullptr) return false;
    std::memcpy(data, source->bytes.data() + (address - source->base), bytes);
    if (space == memory_space::kConstant)
      ++constant_reads;
    else if (space == memory_space::kGlobal)
      ++global_reads;
    else if (space == memory_space::kShared)
      ++shared_reads;
    else
      ++local_reads;
    return true;
  }

  bool read_constant(uint32_t bank, uint64_t address, void *data,
                     size_t bytes) const override {
    return read(memory_space::kConstant, (uint64_t{bank} << 32) | address, data,
                bytes);
  }

  bool write(memory_space space, uint64_t address, const void *data,
             size_t bytes) override {
    region *destination = find(space, address, bytes);
    if (destination == nullptr || !destination->writable) return false;
    std::memcpy(destination->bytes.data() + (address - destination->base), data,
                bytes);
    if (space == memory_space::kGlobal)
      ++global_writes;
    else if (space == memory_space::kShared)
      ++shared_writes;
    else if (space == memory_space::kLocal)
      ++local_writes;
    return true;
  }

  mutable unsigned constant_reads = 0;
  mutable unsigned global_reads = 0;
  mutable unsigned shared_reads = 0;
  mutable unsigned local_reads = 0;
  unsigned global_writes = 0;
  unsigned shared_writes = 0;
  unsigned local_writes = 0;

 private:
  region *find(memory_space space, uint64_t address, size_t bytes) {
    const auto found = std::find_if(
        regions_.begin(), regions_.end(), [&](const region &candidate) {
          return candidate.space == space && address >= candidate.base &&
                 address - candidate.base <= candidate.bytes.size() &&
                 bytes <= candidate.bytes.size() - (address - candidate.base);
        });
    return found == regions_.end() ? nullptr : &*found;
  }

  const region *find(memory_space space, uint64_t address, size_t bytes) const {
    const auto found = std::find_if(
        regions_.begin(), regions_.end(), [&](const region &candidate) {
          return candidate.space == space && address >= candidate.base &&
                 address - candidate.base <= candidate.bytes.size() &&
                 bytes <= candidate.bytes.size() - (address - candidate.base);
        });
    return found == regions_.end() ? nullptr : &*found;
  }

  std::vector<region> regions_;
};

// Individually safe memory calls, with a scheduling opportunity inside a
// simulated RMW. Shared by the scalar-atomic and bulk-reduction regressions.
class interleavable_functional_memory final : public functional_memory {
 public:
  mapped_functional_memory storage;
  bool read(memory_space space, uint64_t address, void *data,
            size_t bytes) const override {
    bool ok;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ok = storage.read(space, address, data, bytes);
    }
    if (space == memory_space::kGlobal) std::this_thread::yield();
    return ok;
  }
  bool write(memory_space space, uint64_t address, const void *data,
             size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return storage.write(space, address, data, bytes);
  }

 private:
  mutable std::mutex mutex_;
};

inline uint16_t exact_integer_half(int value) {
  if (value == 0) return 0;
  const uint16_t sign = value < 0 ? 0x8000u : 0;
  const unsigned magnitude = static_cast<unsigned>(value < 0 ? -value : value);
  if (magnitude > 1024) {
    ADD_FAILURE() << "unsupported exact FP16 fixture integer " << value;
    return 0;
  }
  unsigned exponent = 0;
  while ((2u << exponent) <= magnitude) ++exponent;
  const unsigned leading = 1u << exponent;
  const uint16_t mantissa =
      static_cast<uint16_t>((magnitude - leading) << (10 - exponent));
  return sign | static_cast<uint16_t>((exponent + 15) << 10) | mantissa;
}

inline float half_to_float(uint16_t value) {
  const uint32_t sign = value >> 15;
  const uint32_t exponent = (value >> 10) & 0x1f;
  const uint32_t mantissa = value & 0x3ff;
  if (exponent == 0) {
    const float magnitude =
        mantissa == 0 ? 0.0f : std::ldexp(static_cast<float>(mantissa), -24);
    return sign ? -magnitude : magnitude;
  }
  if (exponent == 0x1f) {
    if (mantissa != 0) return std::numeric_limits<float>::quiet_NaN();
    return sign ? -std::numeric_limits<float>::infinity()
                : std::numeric_limits<float>::infinity();
  }
  const float magnitude =
      std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                 static_cast<int>(exponent) - 15);
  return sign ? -magnitude : magnitude;
}

inline instruction official_instruction(uint64_t pc, const std::string &opcode,
                                        const std::string &operands = {},
                                        const std::string &predicate = {}) {
  instruction result;
  result.pc = pc;
  result.opcode = opcode;
  result.operand_text = operands;
  result.predicate_text = predicate;
  result.decoded = true;
  parse_instruction_operands(result);
  return result;
}

inline uint64_t make_control(unsigned stall, bool yield, unsigned write_barrier,
                             unsigned read_barrier, unsigned wait_mask,
                             unsigned reuse_mask) {
  const uint64_t packed =
      stall | (uint64_t{!yield} << 4) | (uint64_t{write_barrier} << 5) |
      (uint64_t{read_barrier} << 8) | (uint64_t{wait_mask} << 11);
  return (packed << 41) | (uint64_t{reuse_mask} << 58);
}

}  // namespace
