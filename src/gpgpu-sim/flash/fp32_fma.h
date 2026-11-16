#pragma once
#include <algorithm>
#include <cfenv>
#include <cmath>
namespace flash_gpgpu_sim {
inline float rounded_f32_fma(float a, float b, float c, int rounding,
                             bool saturate) {
  const int saved = std::fegetround();
  std::fesetround(rounding);
  volatile float x = a, y = b, z = c;
  volatile float rounded = std::fma(x, y, z);
  const float result = rounded;
  std::fesetround(saved);
  return saturate ? (!(result > 0.0f) ? 0.0f : std::min(result, 1.0f)) : result;
}
}
