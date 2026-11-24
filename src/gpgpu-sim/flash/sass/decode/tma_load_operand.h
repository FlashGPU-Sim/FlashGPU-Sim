#pragma once
#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {

struct tma_load_operands {
  unsigned state_base = 0, map_base = 0, extra_begin = 0;
  const operand *cache_descriptor = nullptr;
};

// Native text names only the first UR of each implicit span. Share validation
// between functional execution and dependency projection, for joined and split
// [state][map] syntax. Callers supply the span for their opcode/rank.
inline bool decode_tma_load_operands(const instruction &inst,
                                     unsigned state_words, bool multicast,
                                     bool allow_cache_descriptor,
                                     tma_load_operands &out) {
  out = {};
  if (!inst.operands_structured || inst.operands.empty())
    return false;
  const auto &first = inst.operands[0];
  if (first.kind != operand_kind::kMemoryAddress || first.has_prefix ||
      (first.address_groups.size() != 1 && first.address_groups.size() != 2))
    return false;
  const bool split = first.address_groups.size() == 1;
  out.extra_begin = split ? 2 : 1;
  if (inst.operands.size() < out.extra_begin)
    return false;
  const auto &second = inst.operands[split ? 1 : 0];
  if (second.kind != operand_kind::kMemoryAddress || second.has_prefix ||
      second.address_groups.size() != (split ? 1u : 2u))
    return false;
  const unsigned cache_index = out.extra_begin + (multicast ? 1 : 0);
  if (allow_cache_descriptor && inst.operands.size() == cache_index + 1) {
    const auto &cache = inst.operands[cache_index];
    if (cache.kind != operand_kind::kTensorMapDescriptor ||
        cache.descriptor_register > kUniformRegisters - 2)
      return false;
    out.cache_descriptor = &cache;
  } else if (inst.operands.size() != cache_index) {
    return false;
  }
  const auto &a = first.address_groups[0];
  const auto &b = second.address_groups[split ? 0 : 1];
  if (a.terms.size() != 1 || b.terms.size() != 1 || !state_words ||
      state_words > kUniformRegisters)
    return false;
  const auto valid = [](const address_term &term, unsigned words) {
    return term.kind == address_term_kind::kUniformRegister && !term.wide &&
           !term.negated && term.index <= kUniformRegisters - words;
  };
  if (!valid(a.terms[0], state_words) || !valid(b.terms[0], 2))
    return false;
  out.state_base = a.terms[0].index;
  out.map_base = b.terms[0].index;
  return true;
}

} // namespace sass
} // namespace flash_gpgpu_sim
