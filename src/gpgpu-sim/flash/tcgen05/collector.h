#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace flash_gpgpu_sim {
enum class tcgen05_collector_op { discard, fill, use, lastuse };
struct tcgen05_ws_collector_control {
  unsigned slot = 0;
  tcgen05_collector_op op = tcgen05_collector_op::discard;
  bool reuses() const {
    return op == tcgen05_collector_op::use ||
           op == tcgen05_collector_op::lastuse;
  }
};
struct tcgen05_ws_half_signature {
  unsigned n, k, format;
  bool transpose;
  unsigned max_shift = 0;
  bool operator==(const tcgen05_ws_half_signature &other) const {
    return n == other.n && k == other.k && format == other.format &&
           transpose == other.transpose && max_shift == other.max_shift;
  }
};

// CTA-local banks, separated by the issuing thread. They cache unmodified
// logical B values, before negation or any per-instruction arithmetic.
// The owning TMEM scope clears these together with CTA retirement/reset.
class tcgen05_ws_half_collectors {
public:
  bool apply(unsigned owner, tcgen05_ws_collector_control control,
             tcgen05_ws_half_signature signature,
             std::vector<uint16_t> &values) {
    if (control.slot >= 4 || signature.n == 0 || signature.k == 0)
      return false;
    auto found = banks_.find(owner);
    if (control.reuses()) {
      if (found == banks_.end())
        return false;
      auto &entry = found->second[control.slot];
      if (!entry.valid || !(entry.signature == signature))
        return false;
      values = entry.values;
      if (control.op == tcgen05_collector_op::lastuse)
        entry = {};
      return true;
    }
    if (values.size() !=
        std::size_t(signature.n + signature.max_shift) * signature.k)
      return false;
    if (control.op == tcgen05_collector_op::fill) {
      banks_[owner][control.slot] = {true, signature, values};
      return true;
    }
    if (control.op != tcgen05_collector_op::discard)
      return false;
    if (found != banks_.end())
      found->second[control.slot] = {};
    return true;
  }

private:
  struct entry {
    bool valid = false;
    tcgen05_ws_half_signature signature{};
    std::vector<uint16_t> values;
  };
  std::map<unsigned, std::array<entry, 4>> banks_;
};
} // namespace flash_gpgpu_sim
