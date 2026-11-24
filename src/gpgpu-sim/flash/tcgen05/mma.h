#ifndef FLASH_GPGPU_SIM_TCGEN05_MMA_H
#define FLASH_GPGPU_SIM_TCGEN05_MMA_H

#include "descriptor.h"

#include <cstdint>
#include <vector>

namespace flash_gpgpu_sim {

float tcgen05_f16_to_f32(uint16_t f16);
float tcgen05_bf16_to_f32(uint16_t bf16);
float tcgen05_e2m1_to_f32(uint8_t e2m1);
float tcgen05_mxf8f6f4_to_f32(uint8_t value, uint8_t format);
float tcgen05_ue8m0_to_f32(uint8_t ue8m0);
uint16_t tcgen05_f32_to_f16(float f32);
uint32_t tcgen05_f32_to_bits(float value);
float tcgen05_bits_to_f32(uint32_t value);

// Inputs are already TF32-converted words. Low mantissa bits are ignored.
std::vector<uint32_t> tcgen05_mma_tf32_compute_words(
    unsigned m, unsigned n, const std::vector<uint32_t> &a,
    const std::vector<uint32_t> &b, const std::vector<uint32_t> &input_d,
    bool accumulate, bool negate_a = false, bool negate_b = false);

std::vector<uint32_t> tcgen05_mma_f16_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint16_t> &a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input_d,
    bool enable_input_d);

// Physical A images contain all 128 datapaths per CTA, packed as 16-bit
// elements. Shared by native and PTX adapters, including folded dual GEMM.
std::vector<uint32_t> tcgen05_mma_f16_tmem_a_compute_words(
    const tcgen05_mma_descriptor_t &desc,
    const std::vector<std::vector<uint16_t>> &physical_a,
    const std::vector<uint16_t> &b, const std::vector<uint32_t> &input_d,
    bool enable_input_d, bool folded);

std::vector<uint32_t> tcgen05_mma_mxf4_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint8_t> &a,
    const std::vector<uint8_t> &b, const std::vector<uint8_t> &scale_a,
    const std::vector<uint8_t> &scale_b, uint32_t scale_vector_size,
    const std::vector<uint32_t> &input_d, bool enable_input_d);

std::vector<uint32_t> tcgen05_mma_mxf8f6f4_compute_words(
    const tcgen05_mma_descriptor_t &desc, const std::vector<uint8_t> &a,
    const std::vector<uint8_t> &b, const std::vector<uint8_t> &scale_a,
    const std::vector<uint8_t> &scale_b, const std::vector<uint32_t> &input_d,
    bool enable_input_d);

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TCGEN05_MMA_H
