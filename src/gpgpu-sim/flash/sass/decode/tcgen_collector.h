#pragma once
#include "../../tcgen05/collector.h"
#include <string>

namespace flash_gpgpu_sim {
namespace sass {
// CUDA 13.3 SM100 lowering places reuse/keep and buffer selection on B.
inline bool decode_ws_collector_operand(std::string &operand,
                                        tcgen05_ws_collector_control &control) {
  const auto end = operand.find(']');
  if (end == std::string::npos)
    return false;
  bool reuse = false, keep = false, slot_seen = false;
  unsigned slot = 0;
  size_t at = end + 1;
  while (at < operand.size()) {
    if (operand[at++] != '.')
      return false;
    const auto next = operand.find('.', at);
    const auto token = operand.substr(at, next - at);
    if (token == "B_REUSE" && !reuse)
      reuse = true;
    else if (token == "B_KEEP" && !keep)
      keep = true;
    else if (token.size() == 7 && token.rfind("BUFFER", 0) == 0 &&
             token[6] >= '1' && token[6] <= '3' && !slot_seen) {
      slot = token[6] - '0';
      slot_seen = true;
    } else
      return false;
    at = next == std::string::npos ? operand.size() : next;
  }
  control = {slot, reuse ? (keep ? tcgen05_collector_op::use
                                 : tcgen05_collector_op::lastuse)
                         : (keep ? tcgen05_collector_op::fill
                                 : tcgen05_collector_op::discard)};
  operand.resize(end + 1);
  return true;
}
} // namespace sass
} // namespace flash_gpgpu_sim
