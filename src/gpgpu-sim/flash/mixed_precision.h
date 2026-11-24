#pragma once
#include "bfloat_arithmetic.h"
#include "fp16_conversion.h"
#include "fp32_fma.h"
namespace flash_gpgpu_sim {
enum class mixed_operation { add, subtract, fma };
inline float mixed_input_to_f32(uint16_t bits, bool bfloat) {
  return bfloat ? bf16_to_f32(bits) : fp16_to_f32(bits);
}
inline float mixed_f32_arithmetic(mixed_operation op, float a, float b, float c,
                                  int rounding, bool saturate) {
  return rounded_f32_fma(a, op == mixed_operation::fma ? b : 1.0f,
                         op == mixed_operation::subtract ? -c : c, rounding,
                         saturate);
}
} // namespace flash_gpgpu_sim
