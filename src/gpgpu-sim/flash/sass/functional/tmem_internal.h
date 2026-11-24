#pragma once
#include "../decode/operand_parser.h"
#include "internal.h"
#include "tmem.h"
namespace flash_gpgpu_sim {
namespace sass {
namespace functional_detail {
inline functional_tcgen_state &storage(execution_context &context) {
  if (!context.cta->tcgen)
    context.cta->tcgen = std::make_shared<functional_tcgen_state>();
  return *context.cta->tcgen;
}

inline bool tmem_address(const operand &op, const warp_state &state,
                         uint32_t &value) {
  if (op.text.rfind("tmem[", 0) || op.text.back() != ']')
    return false;
  const auto address = parse_operand(op.text.substr(4));
  if (address.kind != operand_kind::kMemoryAddress || address.has_prefix ||
      address.address_groups.size() != 1)
    return false;
  uint64_t result = 0;
  for (const auto &term : address.address_groups[0].terms) {
    if (term.wide || term.negated)
      return false;
    if (term.kind == address_term_kind::kUniformRegister)
      result += state.read_uniform_register(term.index);
    else if (term.kind == address_term_kind::kImmediate) {
      if (term.immediate < 0) {
        const uint64_t magnitude = uint64_t(-(term.immediate + 1)) + 1;
        if (result < magnitude)
          return false;
        result -= magnitude;
      } else {
        if (uint64_t(term.immediate) > UINT32_MAX - result)
          return false;
        result += uint64_t(term.immediate);
      }
    } else
      return false;
    if (result > UINT32_MAX)
      return false;
  }
  value = result;
  return true;
}

inline bool wrapped_uniform(const operand &op, const char *prefix,
                            operand &inner) {
  const std::string start = std::string(prefix) + "[";
  if (op.text.rfind(start, 0) != 0 || op.text.back() != ']')
    return false;
  inner = parse_operand(
      op.text.substr(start.size(), op.text.size() - start.size() - 1));
  return inner.kind == operand_kind::kUniformRegister && !inner.negated &&
         !inner.absolute && !inner.wide && !inner.bitwise_complement;
}

} // namespace functional_detail
} // namespace sass
} // namespace flash_gpgpu_sim
