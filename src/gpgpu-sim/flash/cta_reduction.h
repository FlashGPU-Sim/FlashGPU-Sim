#pragma once
#include <cassert>
#include <cstdint>

namespace flash_gpgpu_sim {
enum class cta_reduction_op { bit_and, bit_or, popcount };

// Shared PTX/native reduction arithmetic and generation initialization. Barrier
// arrival and publication remain with the existing CTA synchronization service.
class cta_reduction_accumulator {
public:
  void begin(uint32_t identity) {
    if (!active_) {
      value_ = identity;
      active_ = true;
    }
  }
  void accumulate(cta_reduction_op op, uint32_t contribution) {
    begin(op == cta_reduction_op::bit_and ? 1u : 0u);
    switch (op) {
    case cta_reduction_op::bit_and:
      value_ &= contribution;
      break;
    case cta_reduction_op::bit_or:
      value_ |= contribution;
      break;
    case cta_reduction_op::popcount:
      value_ += contribution;
      break;
    }
  }
  uint32_t value() const {
    assert(active_);
    return value_;
  }
  void reset() { active_ = false; }

private:
  uint32_t value_ = 0;
  bool active_ = false;
};
} // namespace flash_gpgpu_sim
