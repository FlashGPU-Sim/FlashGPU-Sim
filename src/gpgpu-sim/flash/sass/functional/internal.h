#ifndef FLASH_GPGPU_SIM_SASS_FUNCTIONAL_INTERNAL_H
#define FLASH_GPGPU_SIM_SASS_FUNCTIONAL_INTERNAL_H

#include "functional.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {

// Shared across CTA adapters and scalar/TMA global RMW implementations.
// Host synchronization only: this does not model GPU timing resources.
extern std::mutex global_atomic_mutex;

step_result unsupported(const instruction &inst, const std::string &reason);
step_result memory_fault(const instruction &inst, memory_space space,
                         uint64_t address, size_t bytes);
const operand *get_operand(const instruction &inst, size_t position,
                           operand_kind kind);
bool has_typed_operands(const instruction &inst, size_t count);
bool lane_executes(const instruction &inst, const warp_state &state,
                   unsigned lane);
bool any_lane_executes(const instruction &inst, const warp_state &state);
bool read_predicate_operand(const operand &source, const warp_state &state,
                            unsigned lane, bool &value);
void advance(const instruction &inst, warp_state &state);
uint64_t read_register_pair(const warp_state &state, unsigned lane,
                            unsigned low_register);
uint64_t read_uniform_register_pair(const warp_state &state,
                                    unsigned low_register);
void write_register_pair(warp_state &state, unsigned lane,
                         unsigned low_register, uint64_t value);
void write_uniform_register_pair(warp_state &state, unsigned low_register,
                                 uint64_t value);
bool read_u32_operand(const operand &source, const warp_state &state,
                      unsigned lane, uint32_t &value);
bool read_u64_operand(const operand &source, const warp_state &state,
                      unsigned lane, uint64_t &value);
bool read_f32_operand(const operand &source, const warp_state &state,
                      unsigned lane, float &value);
uint64_t evaluate_address_terms(const address_expression &address,
                                const warp_state &state, unsigned lane);
bool evaluate_address(const operand &address, const warp_state &state,
                      unsigned lane, uint64_t &value);
bool evaluate_shared_address(const operand &address, const warp_state &state,
                             unsigned lane, uint64_t &value);
bool read_special_register(const operand &source,
                           const execution_context &context, unsigned lane,
                           uint32_t &value);
uint32_t float_bits(float value);
float bits_float(uint32_t bits);
float half_float(uint16_t bits);
float bfloat_float(uint16_t bits);
float tf32_float(uint32_t bits);
uint16_t float_half_rn(uint32_t bits);

void register_data_movement_semantics(frontend &target);
void register_integer_semantics(frontend &target);
void register_float_semantics(frontend &target);
void register_control_semantics(frontend &target);
void register_memory_semantics(frontend &target);
void register_mbarrier_semantics(frontend &target);
void register_bulk_copy_semantics(frontend &target);
void register_tma_semantics(frontend &target);
void register_mma_semantics(frontend &target);
void register_wgmma_semantics(frontend &target);

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim

#endif
