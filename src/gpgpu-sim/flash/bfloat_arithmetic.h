#pragma once
#include "bfloat_conversion.h"
#include <cfenv>
#include <cmath>

namespace flash_gpgpu_sim {
inline float bf16_to_f32(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
namespace bfloat_detail {
// Host rounding state is thread-local. Preserve the caller's rounding mode.
class nearest_scope {
public:
  nearest_scope() : saved(std::fegetround()) {
    if (saved != FE_TONEAREST)
      std::fesetround(FE_TONEAREST);
  }
  ~nearest_scope() {
    if (saved != FE_TONEAREST)
      std::fesetround(saved);
  }

private:
  int saved;
};
inline bool midpoint(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return std::isfinite(value) && (bits & 0xffff) == 0x8000;
}
} // namespace bfloat_detail
inline uint16_t bf16_add_rn(float a, float b) {
  bfloat_detail::nearest_scope rounding;
  volatile float x = a, y = b;
  volatile float result = x + y;
  return f32_to_bf16_rn(result);
}
inline uint16_t bf16_mul_rn(float a, float b) {
  bfloat_detail::nearest_scope rounding;
  volatile float x = a, y = b;
  float result = x * y;
  // BF16 products are exact in FP64. Avoid a second tie at FP32 underflow.
  if (bfloat_detail::midpoint(result)) {
    const double difference = double(a) * double(b) - double(result);
    if (difference != 0)
      result = std::nextafter(result, difference > 0 ? INFINITY : -INFINITY);
  }
  return f32_to_bf16_rn(result);
}
// BF16 inputs have exact FP64 products. Recover a tiny residual at an FP32
// midpoint before rounding to BF16, including cancellation and subnormals.
inline uint16_t bf16_fma_rn(float a, float b, float c) {
  bfloat_detail::nearest_scope rounding;
  volatile float x = a, y = b, z = c;
  float rounded = std::fma(x, y, z);
  if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
      bfloat_detail::midpoint(rounded)) {
    const double product = double(a) * double(b);
    const double sum = product + double(c);
    const double recovered = sum - product;
    const double residual =
        (product - (sum - recovered)) + (double(c) - recovered);
    const double difference = sum - double(rounded);
    const double direction = difference != 0 ? difference : residual;
    if (direction != 0)
      rounded = std::nextafter(rounded, direction > 0 ? INFINITY : -INFINITY);
  }
  return f32_to_bf16_rn(rounded);
}
} // namespace flash_gpgpu_sim
