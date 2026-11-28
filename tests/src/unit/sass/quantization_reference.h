#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Independent host oracle: enumerate representable E4M3 values, rather than
// sharing the simulator's exponent/mantissa conversion implementation.
namespace moe_quant_reference {
inline double positive_value(unsigned code) {
  const unsigned exponent = code >> 3, fraction = code & 7;
  return exponent ? std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7)
                  : std::ldexp(double(fraction), -9);
}
inline uint8_t encode(double value) {
  if (std::isnan(value)) return 0x7f;
  const uint8_t sign = std::signbit(value) ? 0x80 : 0;
  const double magnitude = std::min(std::abs(value), 448.0);
  unsigned best = 0;
  double error = std::numeric_limits<double>::infinity();
  for (unsigned code = 0; code <= 126; ++code) {
    const double candidate = std::abs(magnitude - positive_value(code));
    if (candidate < error || (candidate == error && !(code & 1))) {
      error = candidate;
      best = code;
    }
  }
  return sign | uint8_t(best);
}
struct row {
  std::vector<uint8_t> data;
  std::vector<uint8_t> scales;
};
inline row quantize(const std::vector<double>& input) {
  row result;
  result.data.resize(input.size());
  for (size_t begin = 0; begin < input.size(); begin += 128) {
    double maximum = 1e-10;
    for (size_t i = begin; i < begin + 128; ++i)
      maximum = std::max(maximum, std::abs(input[i]));
    const int exponent = int(std::ceil(std::log2(maximum / 448.0)));
    result.scales.push_back(uint8_t(exponent + 127));
    for (size_t i = begin; i < begin + 128; ++i)
      result.data[i] = encode(std::ldexp(input[i], -exponent));
  }
  return result;
}
} // namespace moe_quant_reference
