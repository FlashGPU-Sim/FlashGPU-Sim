#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
namespace flash_gpgpu_sim {
enum class reduce32_kind {
  add,
  bit_and,
  bit_or,
  bit_xor,
  min_s,
  max_s,
  min_u,
  max_u,
  min_f,
  max_f
};
struct reduce32_operation {
  reduce32_kind kind;
  bool absolute = false;
  bool propagate_nan = false;
};
inline uint32_t reduce32_combine(reduce32_operation op, uint32_t a,
                                 uint32_t b) {
  switch (op.kind) {
  case reduce32_kind::add:
    return a + b;
  case reduce32_kind::bit_and:
    return a & b;
  case reduce32_kind::bit_or:
    return a | b;
  case reduce32_kind::bit_xor:
    return a ^ b;
  case reduce32_kind::min_s:
    return uint32_t(std::min(int32_t(a), int32_t(b)));
  case reduce32_kind::max_s:
    return uint32_t(std::max(int32_t(a), int32_t(b)));
  case reduce32_kind::min_u:
    return std::min(a, b);
  case reduce32_kind::max_u:
    return std::max(a, b);
  case reduce32_kind::min_f:
  case reduce32_kind::max_f: {
    if (op.absolute) {
      a &= 0x7fffffff;
      b &= 0x7fffffff;
    }
    const bool an = (a & 0x7fffffff) > 0x7f800000,
               bn = (b & 0x7fffffff) > 0x7f800000;
    if (an || bn) {
      if (op.propagate_nan || (an && bn))
        return 0x7fffffff;
      return an ? b : a;
    }
    const bool minimum = op.kind == reduce32_kind::min_f;
    if (!(a & 0x7fffffff) && !(b & 0x7fffffff))
      return minimum ? (a | b) : (a & b);
    float x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return (minimum ? x < y : x > y) ? a : b;
  }
  }
  return 0;
}
// All participating source values are snapshotted before any destination write.
// Disjoint groups are allowed, but each group's members must agree on its mask.
inline bool reduce32_groups(reduce32_operation op, uint32_t active,
                            const uint32_t (&values)[32],
                            const uint32_t (&masks)[32],
                            uint32_t (&result)[32]) {
  uint32_t remaining = active;
  while (remaining) {
    const unsigned first = __builtin_ctz(remaining);
    const uint32_t group = masks[first];
    if (!(group & (1u << first)) || (group & ~remaining))
      return false;
    for (unsigned peer = first; peer < 32; ++peer)
      if ((group & (1u << peer)) && masks[peer] != group)
        return false;
    uint32_t value = values[first];
    if (op.kind == reduce32_kind::min_f || op.kind == reduce32_kind::max_f)
      value = reduce32_combine(op, value, value);
    for (unsigned peer = first + 1; peer < 32; ++peer)
      if (group & (1u << peer))
        value = reduce32_combine(op, value, values[peer]);
    for (unsigned peer = first; peer < 32; ++peer)
      if (group & (1u << peer))
        result[peer] = value;
    remaining &= ~group;
  }
  return true;
}
} // namespace flash_gpgpu_sim
