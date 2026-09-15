#include "operand_parser.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace flash_gpgpu_sim {
namespace sass {
namespace {

std::string trim(const std::string &text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
    --end;
  return text.substr(begin, end - begin);
}

bool consume_suffix(std::string &text, const char *suffix) {
  const std::string value(suffix);
  if (text.size() < value.size() ||
      text.compare(text.size() - value.size(), value.size(), value) != 0)
    return false;
  text.erase(text.size() - value.size());
  return true;
}

bool parse_decimal_index(const std::string &text, size_t begin,
                         uint32_t &result) {
  if (begin >= text.size())
    return false;
  uint64_t value = 0;
  for (size_t i = begin; i < text.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(text[i])))
      return false;
    value = value * 10 + static_cast<unsigned>(text[i] - '0');
    if (value > std::numeric_limits<uint32_t>::max())
      return false;
  }
  result = static_cast<uint32_t>(value);
  return true;
}

bool parse_integer(const std::string &text, int64_t &result) {
  if (text.empty())
    return false;
  errno = 0;
  char *end = nullptr;
  if (text[0] == '-') {
    const long long value = std::strtoll(text.c_str(), &end, 0);
    if (errno != 0 || end != text.c_str() + text.size())
      return false;
    result = static_cast<int64_t>(value);
    return true;
  }
  const unsigned long long value = std::strtoull(text.c_str(), &end, 0);
  if (errno != 0 || end != text.c_str() + text.size())
    return false;
  // nvdisasm prints 64-bit immediates as their full unsigned hexadecimal bit
  // pattern (for example 0xffffffffffffffc0). Preserve that pattern in the
  // signed IR field without relying on implementation-defined conversion.
  const unsigned long long signed_max =
      static_cast<unsigned long long>(std::numeric_limits<int64_t>::max());
  result =
      value <= signed_max
          ? static_cast<int64_t>(value)
          : -1 - static_cast<int64_t>(
                     std::numeric_limits<unsigned long long>::max() - value);
  return true;
}

operand parse_register_like(const std::string &original) {
  operand result;
  result.text = original;
  std::string text = original;
  if (!text.empty() && text[0] == '!') {
    result.negated = true;
    text.erase(0, 1);
  } else if (!text.empty() && text[0] == '-') {
    result.negated = true;
    text.erase(0, 1);
  } else if (!text.empty() && text[0] == '~') {
    result.bitwise_complement = true;
    text.erase(0, 1);
  }
  if (consume_suffix(text, ".H0_H0"))
    result.half_selection = operand_half_selection::kLowReplicate;
  else if (consume_suffix(text, ".H1_H1"))
    result.half_selection = operand_half_selection::kHighReplicate;
  if (consume_suffix(text, ".ROW"))
    result.matrix_layout = operand_matrix_layout::kRow;
  else if (consume_suffix(text, ".COL"))
    result.matrix_layout = operand_matrix_layout::kColumn;
  result.reuse = consume_suffix(text, ".reuse");
  result.wide = consume_suffix(text, ".64");
  if (text.size() >= 2 && text.front() == '|' && text.back() == '|') {
    result.absolute = true;
    text = text.substr(1, text.size() - 2);
  }

  if (text == "RZ") {
    result.kind = operand_kind::kRegister;
    result.index = 255;
  } else if (text == "URZ") {
    result.kind = operand_kind::kUniformRegister;
    result.index = 255;
  } else if (text == "PT") {
    result.kind = operand_kind::kPredicate;
    result.index = 7;
  } else if (text == "UPT") {
    result.kind = operand_kind::kUniformPredicate;
    result.index = 7;
  } else if (text == "PR") {
    result.kind = operand_kind::kPredicateRegisterFile;
  } else if (text == "UPR") {
    result.kind = operand_kind::kUniformPredicateRegisterFile;
  } else if (text == "SRZ" || text.rfind("SR_", 0) == 0) {
    result.kind = operand_kind::kSpecialRegister;
  } else if (text.size() > 1 && text[0] == 'R' &&
             parse_decimal_index(text, 1, result.index)) {
    result.kind = operand_kind::kRegister;
  } else if (text.size() > 2 && text.rfind("UR", 0) == 0 &&
             parse_decimal_index(text, 2, result.index)) {
    result.kind = operand_kind::kUniformRegister;
  } else if (text.size() > 1 && text[0] == 'P' &&
             parse_decimal_index(text, 1, result.index)) {
    result.kind = operand_kind::kPredicate;
  } else if (text.size() > 2 && text.rfind("UP", 0) == 0 &&
             parse_decimal_index(text, 2, result.index)) {
    result.kind = operand_kind::kUniformPredicate;
  } else if (text.size() > 1 && text[0] == 'B' &&
             parse_decimal_index(text, 1, result.index)) {
    result.kind = operand_kind::kBarrierRegister;
  } else if (text.size() > 2 && text.rfind("SB", 0) == 0 &&
             parse_decimal_index(text, 2, result.index)) {
    result.kind = operand_kind::kScoreboardRegister;
  } else if (text.size() > 3 && text.rfind("gsb", 0) == 0 &&
             parse_decimal_index(text, 3, result.index)) {
    result.kind = operand_kind::kScoreboardRegister;
  }
  return result;
}

bool parse_address_expression(const std::string &text,
                              address_expression &result);

// Address terms and guards do not carry these value-transforming fields.
// Never silently discard them when narrowing a parsed register operand.
// Negation and width are checked separately where the target IR supports them.
bool has_unrepresented_value_modifier(const operand &value) {
  return value.bitwise_complement || value.absolute ||
         value.half_selection != operand_half_selection::kNone ||
         value.matrix_layout != operand_matrix_layout::kNone;
}

operand parse_constant(const std::string &text) {
  operand result;
  result.text = text;
  if (text.rfind("c[", 0) != 0)
    return result;
  const size_t bank_end = text.find(']', 2);
  if (bank_end == std::string::npos || bank_end + 2 >= text.size() ||
      text[bank_end + 1] != '[' || text.back() != ']')
    return result;
  int64_t bank = 0;
  if (!parse_integer(text.substr(2, bank_end - 2), bank) || bank < 0 ||
      bank > std::numeric_limits<uint32_t>::max() ||
      !parse_address_expression(
          text.substr(bank_end + 2, text.size() - bank_end - 3),
          result.constant_address))
    return result;
  result.kind = operand_kind::kConstantMemory;
  result.constant_bank = static_cast<uint32_t>(bank);
  if (result.constant_address.terms.size() == 1 &&
      result.constant_address.terms[0].kind == address_term_kind::kImmediate &&
      result.constant_address.terms[0].immediate >= 0) {
    result.constant_offset_static = true;
    result.constant_offset =
        static_cast<uint64_t>(result.constant_address.terms[0].immediate);
  }
  return result;
}

operand parse_descriptor(const std::string &text) {
  operand result;
  result.text = text;
  if (text.rfind("desc[", 0) != 0)
    return result;
  const size_t descriptor_end = text.find(']', 5);
  if (descriptor_end == std::string::npos || text.back() != ']')
    return result;
  const operand descriptor =
      parse_register_like(text.substr(5, descriptor_end - 5));
  if (descriptor.kind != operand_kind::kUniformRegister || descriptor.negated ||
      has_unrepresented_value_modifier(descriptor))
    return result;

  if (descriptor_end + 1 == text.size()) {
    result.kind = operand_kind::kTensorMapDescriptor;
    result.descriptor_register = descriptor.index;
    return result;
  }
  if (descriptor_end + 2 >= text.size() || text[descriptor_end + 1] != '[')
    return result;

  const std::string address =
      text.substr(descriptor_end + 2, text.size() - descriptor_end - 3);
  const size_t displacement = address.find_first_of("+-", 1);
  const std::string base_text = address.substr(0, displacement);
  const operand base = parse_register_like(base_text);
  if (base.kind != operand_kind::kRegister || !base.wide || base.negated ||
      has_unrepresented_value_modifier(base))
    return result;
  int64_t offset = 0;
  if (displacement != std::string::npos) {
    std::string offset_text = address.substr(displacement);
    // nvdisasm spells a negative displacement appended to a base as
    // "R2.64+-0x40". The plus is the address separator, not a second sign.
    if (offset_text.rfind("+-", 0) == 0)
      offset_text.erase(0, 1);
    if (!parse_integer(offset_text, offset))
      return result;
  }

  result.kind = operand_kind::kDescriptorAddress;
  result.descriptor_register = descriptor.index;
  result.address_register = base.index;
  result.address_offset = offset;
  return result;
}

operand parse_gmma_descriptor(const std::string &original) {
  operand result;
  result.text = original;
  std::string text = original;
  result.descriptor_transpose_b = consume_suffix(text, ".tnspB");
  result.descriptor_transpose_a = consume_suffix(text, ".tnspA");
  if (text.rfind("gdesc[", 0) != 0 || text.back() != ']')
    return result;
  const operand descriptor =
      parse_register_like(text.substr(6, text.size() - 7));
  if (descriptor.kind != operand_kind::kUniformRegister || descriptor.negated ||
      has_unrepresented_value_modifier(descriptor))
    return result;
  result.kind = operand_kind::kGmmaDescriptor;
  result.descriptor_register = descriptor.index;
  return result;
}

bool parse_address_expression(const std::string &text,
                              address_expression &result) {
  size_t begin = 0;
  bool negate = false;
  while (begin < text.size()) {
    if (text[begin] == '+' || text[begin] == '-') {
      negate = text[begin] == '-';
      ++begin;
    }
    const size_t end = text.find_first_of("+-", begin);
    const std::string token = text.substr(begin, end - begin);
    if (token.empty())
      return false;
    address_term term;
    const operand parsed = parse_register_like(token);
    if (has_unrepresented_value_modifier(parsed)) {
      return false;
    } else if (parsed.kind == operand_kind::kRegister) {
      term.kind = address_term_kind::kRegister;
      term.index = parsed.index;
      term.wide = parsed.wide;
      term.negated = negate || parsed.negated;
    } else if (parsed.kind == operand_kind::kUniformRegister) {
      term.kind = address_term_kind::kUniformRegister;
      term.index = parsed.index;
      term.wide = parsed.wide;
      term.negated = negate || parsed.negated;
    } else {
      int64_t immediate = 0;
      if (!parse_integer(token, immediate))
        return false;
      term.kind = address_term_kind::kImmediate;
      // The IR preserves full 64-bit immediate bit patterns. Negating the
      // minimum signed value wraps to itself in address arithmetic, but
      // evaluating -INT64_MIN in C++ would be undefined behavior.
      term.immediate =
          negate && immediate != std::numeric_limits<int64_t>::min()
              ? -immediate
              : immediate;
    }
    result.terms.push_back(term);
    if (end == std::string::npos)
      break;
    if (end + 1 == text.size())
      return false;
    negate = text[end] == '-';
    begin = end + 1;
  }
  return !result.terms.empty();
}

operand parse_memory_address(const std::string &text) {
  operand result;
  result.text = text;
  const size_t first_open = text.find('[');
  if (first_open == std::string::npos || text.back() != ']')
    return result;
  if (first_open != 0) {
    const operand prefix = parse_register_like(text.substr(0, first_open));
    if (prefix.negated || has_unrepresented_value_modifier(prefix))
      return result;
    if (prefix.kind != operand_kind::kRegister &&
        prefix.kind != operand_kind::kUniformRegister &&
        prefix.kind != operand_kind::kPredicate &&
        prefix.kind != operand_kind::kUniformPredicate)
      return result;
    result.has_prefix = true;
    result.prefix_kind = prefix.kind;
    result.prefix_index = prefix.index;
    result.prefix_wide = prefix.wide;
  }

  size_t position = first_open;
  while (position < text.size()) {
    if (text[position] != '[')
      return operand{};
    const size_t close = text.find(']', position + 1);
    if (close == std::string::npos)
      return operand{};
    address_expression expression;
    if (!parse_address_expression(
            text.substr(position + 1, close - position - 1), expression))
      return operand{};
    result.address_groups.push_back(std::move(expression));
    position = close + 1;
  }
  result.kind = operand_kind::kMemoryAddress;
  return result;
}

operand parse_register_set(const std::string &text) {
  operand result;
  result.text = text;
  if (text.size() < 2 || text.front() != '{' || text.back() != '}')
    return result;
  const std::string contents = text.substr(1, text.size() - 2);
  for (const std::string &element : split_operand_text(contents)) {
    int64_t value = 0;
    if (!parse_integer(element, value) || value < 0 ||
        value > std::numeric_limits<uint32_t>::max())
      return operand{};
    result.register_set.push_back(static_cast<uint32_t>(value));
  }
  result.kind = operand_kind::kRegisterSet;
  return result;
}

} // namespace

std::vector<std::string> split_operand_text(const std::string &text) {
  std::vector<std::string> result;
  if (text.empty())
    return result;
  unsigned square_depth = 0;
  unsigned brace_depth = 0;
  unsigned paren_depth = 0;
  size_t begin = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    switch (text[i]) {
    case '[':
      ++square_depth;
      break;
    case ']':
      if (square_depth != 0)
        --square_depth;
      break;
    case '{':
      ++brace_depth;
      break;
    case '}':
      if (brace_depth != 0)
        --brace_depth;
      break;
    case '(':
      ++paren_depth;
      break;
    case ')':
      if (paren_depth != 0)
        --paren_depth;
      break;
    case ',':
      if (square_depth == 0 && brace_depth == 0 && paren_depth == 0) {
        result.push_back(trim(text.substr(begin, i - begin)));
        begin = i + 1;
      }
      break;
    }
  }
  result.push_back(trim(text.substr(begin)));
  return result;
}

operand parse_operand(const std::string &input) {
  const std::string text = trim(input);
  if (text.empty())
    return operand{};
  operand result = parse_constant(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  result = parse_gmma_descriptor(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  result = parse_descriptor(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  result = parse_memory_address(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  result = parse_register_set(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  result = parse_register_like(text);
  if (result.kind != operand_kind::kOpaque)
    return result;
  int64_t immediate = 0;
  if (parse_integer(text, immediate)) {
    // A leading '-' is already represented in the signed immediate. Discard
    // the register parser's tentative unary-negation flag so execution does
    // not negate a negative immediate a second time.
    result = operand{};
    result.kind = operand_kind::kImmediate;
    result.text = text;
    result.immediate = immediate;
    return result;
  }
  errno = 0;
  char *end = nullptr;
  const double floating = std::strtod(text.c_str(), &end);
  if (errno == 0 && end == text.c_str() + text.size() &&
      (std::isfinite(floating) || text == "INF" || text == "+INF" ||
       text == "-INF")) {
    result = operand{};
    result.kind = operand_kind::kFloatImmediate;
    result.text = text;
    result.float_immediate = floating;
  }
  return result;
}

void parse_instruction_operands(instruction &inst) {
  inst.operands.clear();
  inst.operands_structured = true;
  for (const std::string &text : split_operand_text(inst.operand_text)) {
    inst.operands.push_back(parse_operand(text));
    if (inst.operands.back().kind == operand_kind::kOpaque)
      inst.operands_structured = false;
  }
  inst.has_guard =
      !inst.predicate_text.empty() && inst.predicate_text[0] == '@';
  inst.has_auxiliary_predicate =
      !inst.predicate_text.empty() && inst.predicate_text[0] != '@';
  if (inst.has_guard) {
    std::string guard = inst.predicate_text;
    if (!guard.empty() && guard[0] == '@')
      guard.erase(0, 1);
    inst.guard = parse_operand(guard);
    if ((inst.guard.kind != operand_kind::kPredicate &&
         inst.guard.kind != operand_kind::kUniformPredicate) ||
        inst.guard.wide || has_unrepresented_value_modifier(inst.guard))
      inst.operands_structured = false;
  }
  if (inst.has_auxiliary_predicate) {
    inst.auxiliary_predicate = parse_operand(inst.predicate_text);
    if (inst.auxiliary_predicate.kind == operand_kind::kOpaque)
      inst.operands_structured = false;
  }
}

} // namespace sass
} // namespace flash_gpgpu_sim
