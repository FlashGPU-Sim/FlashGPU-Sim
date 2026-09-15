#include "sassir_decoder.h"
#include "../../panic.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include "operand_parser.h"
#include "sm120_control.h"
#include "sm90_control.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace {

std::vector<std::string> split_tabs(const std::string &line) {
  std::vector<std::string> result;
  size_t begin = 0;
  while (begin <= line.size()) {
    const size_t end = line.find('\t', begin);
    if (end == std::string::npos) {
      result.push_back(line.substr(begin));
      break;
    }
    result.push_back(line.substr(begin, end - begin));
    begin = end + 1;
  }
  return result;
}

unsigned hex_digit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  panic("invalid hexadecimal digit");
}

std::string unescape(const std::string &value) {
  std::string result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '%') {
      result.push_back(value[i]);
      continue;
    }
    if (i + 2 >= value.size())
      panic("truncated percent escape");
    result.push_back(static_cast<char>((hex_digit(value[i + 1]) << 4) |
                                       hex_digit(value[i + 2])));
    i += 2;
  }
  return result;
}

std::vector<uint8_t> parse_hex_bytes(const std::string &value,
                                     const char *field) {
  if (value.size() % 2 != 0)
    panic(std::string("odd-length ") + field);
  std::vector<uint8_t> result;
  result.reserve(value.size() / 2);
  for (size_t index = 0; index < value.size(); index += 2) {
    if (!std::isxdigit(static_cast<unsigned char>(value[index])) ||
        !std::isxdigit(static_cast<unsigned char>(value[index + 1])))
      panic(std::string("invalid ") + field);
    result.push_back(static_cast<uint8_t>((hex_digit(value[index]) << 4) |
                                          hex_digit(value[index + 1])));
  }
  return result;
}

std::string demangle(const std::string &name) {
  int status = 0;
  char *decoded = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
  if (status != 0 || decoded == nullptr) {
    std::free(decoded);
    return {};
  }
  std::string result(decoded);
  std::free(decoded);
  return result;
}

// Nvcc gives functions in anonymous namespaces a volatile numeric component
// in their mangled name.  Rebuilt sources can therefore differ from a decoded
// cubin image even though their ABI and demangled name are identical.  Prefer
// an exact symbol and otherwise accept only one demangled match, preserving
// fail-closed behavior on ambiguity.
std::string resolve_kernel_name(const std::string &sassir_path,
                                const std::string &requested_name) {
  std::ifstream input(sassir_path);
  if (!input)
    panic("cannot open SASS sassir: " + sassir_path);

  const std::string requested_demangled = demangle(requested_name);
  if (requested_demangled.empty())
    return requested_name;

  std::vector<std::string> matches;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("KERNEL\t", 0) != 0)
      continue;
    const std::vector<std::string> columns = split_tabs(line);
    if (columns.size() != 4)
      continue;
    const std::string candidate = unescape(columns[1]);
    if (candidate == requested_name)
      return requested_name;
    if (demangle(candidate) == requested_demangled)
      matches.push_back(candidate);
  }
  if (matches.size() > 1)
    panic("ambiguous demangled SASS kernel match for " + requested_name);
  return matches.empty() ? requested_name : matches.front();
}

uint64_t parse_u64(const std::string &value, const char *field) {
  // strtoull accepts a leading minus and wraps it into the unsigned range.
  // SASSIR unsigned fields must start with a digit (decimal or 0x-prefixed),
  // not a sign/whitespace. Signed legacy token indices use parse_i32 below.
  if (value.empty() || !std::isdigit(static_cast<unsigned char>(value.front())))
    panic(std::string("invalid ") + field + ": " + value);
  char *end = nullptr;
  errno = 0;
  const auto result = std::strtoull(value.c_str(), &end, 0);
  if (errno != 0 || end != value.c_str() + value.size() ||
      result > std::numeric_limits<uint64_t>::max())
    panic(std::string("invalid ") + field + ": " + value);
  return result;
}

uint32_t parse_u32(const std::string &value, const char *field) {
  const uint64_t result = parse_u64(value, field);
  if (result > std::numeric_limits<uint32_t>::max())
    panic(std::string("out-of-range ") + field + ": " + value);
  return static_cast<uint32_t>(result);
}

uint8_t parse_u8(const std::string &value, const char *field) {
  const uint64_t result = parse_u64(value, field);
  if (result > std::numeric_limits<uint8_t>::max())
    panic(std::string("out-of-range ") + field + ": " + value);
  return static_cast<uint8_t>(result);
}

bool parse_bool(const std::string &value, const char *field) {
  const uint64_t result = parse_u64(value, field);
  if (result > 1)
    panic(std::string("invalid ") + field + ": " + value);
  return result != 0;
}

int32_t parse_i32(const std::string &value, const char *field) {
  char *end = nullptr;
  errno = 0;
  const long result = std::strtol(value.c_str(), &end, 0);
  if (errno != 0 || end == value.c_str() ||
      end != value.c_str() + value.size() ||
      result < std::numeric_limits<int32_t>::min() ||
      result > std::numeric_limits<int32_t>::max())
    panic(std::string("invalid ") + field + ": " + value);
  return static_cast<int32_t>(result);
}

void require_columns(const std::vector<std::string> &columns, size_t count,
                     const char *record) {
  if (columns.size() != count) {
    std::ostringstream error;
    error << record << " record has " << columns.size() << " columns; expected "
          << count;
    panic(error.str());
  }
}

void validate_control(const instruction &inst, architecture arch) {
  const control_code raw = arch == architecture::kSm90
                               ? decode_sm90_control(inst.raw.hi)
                               : decode_sm120_control(inst.raw.hi);
  const control_code &sassir = inst.control;
  if (raw.stall != sassir.stall || raw.yield_flag != sassir.yield_flag ||
      raw.write_barrier != sassir.write_barrier ||
      raw.read_barrier != sassir.read_barrier ||
      raw.wait_mask != sassir.wait_mask ||
      raw.reuse_mask != sassir.reuse_mask) {
    std::ostringstream error;
    error << "control word does not match raw instruction at pc 0x" << std::hex
          << inst.pc;
    panic(error.str());
  }
}

void validate_kernel(const kernel &result) {
  if (result.arch != architecture::kSm90 && result.arch != architecture::kSm120)
    panic("only sm90 and sm120 sassirs are supported");
  if (result.instruction_bytes != kInstructionBytes)
    panic("SASS instruction size must be 16 bytes");

  std::vector<bool> parameter_bytes(result.parameter_size, false);
  std::vector<bool> parameter_ordinals(result.parameters.size(), false);
  for (const kernel_parameter &parameter : result.parameters) {
    if (parameter.size == 0 || parameter.ordinal >= parameter_ordinals.size() ||
        parameter_ordinals[parameter.ordinal] ||
        parameter.offset > result.parameter_size ||
        parameter.size > result.parameter_size - parameter.offset)
      panic("invalid kernel parameter ABI metadata");
    parameter_ordinals[parameter.ordinal] = true;
    for (uint32_t byte = parameter.offset;
         byte < parameter.offset + parameter.size; ++byte) {
      if (parameter_bytes[byte])
        panic("overlapping kernel parameter ABI metadata");
      parameter_bytes[byte] = true;
    }
  }
  if (!result.parameters.empty() &&
      (!result.has_parameter_bank || result.parameter_base == 0 ||
       result.parameter_size == 0))
    panic("incomplete kernel parameter bank metadata");

  std::vector<uint32_t> constant_bank_indices;
  for (const kernel_constant_bank &bank : result.constant_banks) {
    if (bank.index == 0 || bank.data.empty() ||
        std::find(constant_bank_indices.begin(), constant_bank_indices.end(),
                  bank.index) != constant_bank_indices.end())
      panic("invalid static constant-bank metadata");
    constant_bank_indices.push_back(bank.index);
  }

  uint64_t previous_pc = 0;
  bool first = true;
  for (const instruction &inst : result.instructions) {
    if (inst.pc % result.instruction_bytes != 0)
      panic("unaligned instruction pc in sassir");
    if (!first && inst.pc <= previous_pc)
      panic("instruction pcs must be strictly increasing");
    if (inst.decoded != !inst.opcode.empty())
      panic("decoded flag does not match opcode presence");
    if (!result.decoder_name.empty() &&
        inst.decoder_source != result.decoder_name)
      panic("instruction decoder source does not match "
            "sassir decoder provenance");
    validate_control(inst, result.arch);
    previous_pc = inst.pc;
    first = false;
  }
}

} // namespace

kernel sassir_decoder::decode_kernel(const std::string &sassir_path,
                                     const std::string &kernel_name) const {
  uint64_t line_number = 0;
  const panic_context diagnostic(sassir_path, line_number);
  std::ifstream input(sassir_path);
  if (!input)
    panic("cannot open SASS sassir: " + sassir_path);
  const std::string resolved_kernel_name =
      resolve_kernel_name(sassir_path, kernel_name);

  kernel result;
  bool saw_header = false;
  bool saw_decoder = false;
  bool in_kernel = false;
  bool wanted_kernel = false;
  bool in_instruction = false;
  bool found_kernel = false;
  unsigned format_version = 0;
  std::string decoder_name;
  std::string decoder_version;
  std::string decoder_schema;
  std::string line;

  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#')
      continue;
    const std::vector<std::string> columns = split_tabs(line);
    const std::string &record = columns.front();

    if (record == "SASSIR") {
      require_columns(columns, 2, "SASSIR");
      format_version = parse_u32(columns[1], "SASSIR version");
      if (saw_header || (format_version != 1 && format_version != 2 &&
                         format_version != 3 && format_version != 4))
        panic("unsupported or duplicate SASSIR header");
      saw_header = true;
    } else if (record == "DECODER") {
      require_columns(columns, 4, "DECODER");
      if (!saw_header || format_version < 2 || in_kernel || saw_decoder)
        panic("misplaced DECODER record");
      decoder_name = unescape(columns[1]);
      decoder_version = unescape(columns[2]);
      decoder_schema = columns[3];
      if (decoder_name.empty() || decoder_version.empty() ||
          decoder_schema.empty())
        panic("incomplete DECODER provenance");
      saw_decoder = true;
    } else if (record == "KERNEL") {
      require_columns(columns, 4, "KERNEL");
      if (!saw_header || in_kernel)
        panic("misplaced KERNEL record");
      if (format_version >= 2 && !saw_decoder)
        panic("SASSIR v2+ requires DECODER provenance");
      in_kernel = true;
      wanted_kernel = unescape(columns[1]) == resolved_kernel_name;
      if (wanted_kernel) {
        if (found_kernel)
          panic("duplicate requested kernel");
        result = kernel{};
        result.name = unescape(columns[1]);
        result.decoder_name = decoder_name;
        result.decoder_version = decoder_version;
        result.decoder_schema = decoder_schema;
        result.arch = columns[2] == "sm90"    ? architecture::kSm90
                      : columns[2] == "sm120" ? architecture::kSm120
                                              : architecture::kUnknown;
        result.instruction_bytes = parse_u32(columns[3], "instruction size");
      }
    } else if (record == "RESOURCES") {
      require_columns(columns, 5, "RESOURCES");
      if (!in_kernel || in_instruction || format_version < 3)
        panic("RESOURCES outside a SASSIR v3 kernel");
      if (wanted_kernel) {
        if (result.has_resources)
          panic("duplicate RESOURCES record");
        result.has_resources = true;
        result.resources.registers = parse_u32(columns[1], "register count");
        result.resources.static_shared =
            parse_u32(columns[2], "static shared memory");
        result.resources.local_memory = parse_u32(columns[3], "local memory");
        result.resources.stack_size = parse_u32(columns[4], "stack size");
      }
    } else if (record == "PARAMBANK") {
      require_columns(columns, 3, "PARAMBANK");
      if (!in_kernel || in_instruction)
        panic("misplaced PARAMBANK record");
      if (wanted_kernel) {
        if (result.has_parameter_bank)
          panic("duplicate PARAMBANK record");
        result.has_parameter_bank = true;
        result.parameter_base = parse_u32(columns[1], "parameter base");
        result.parameter_size = parse_u32(columns[2], "parameter size");
      }
    } else if (record == "PARAM") {
      require_columns(columns, 4, "PARAM");
      if (!in_kernel || in_instruction)
        panic("misplaced PARAM record");
      if (wanted_kernel) {
        if (!result.has_parameter_bank)
          panic("PARAM record precedes PARAMBANK");
        kernel_parameter parameter;
        parameter.ordinal = parse_u32(columns[1], "parameter ordinal");
        parameter.offset = parse_u32(columns[2], "parameter offset");
        parameter.size = parse_u32(columns[3], "parameter size");
        result.parameters.push_back(parameter);
      }
    } else if (record == "CONSTBANK") {
      require_columns(columns, 3, "CONSTBANK");
      if (!in_kernel || in_instruction || format_version < 4)
        panic("CONSTBANK outside a SASSIR v4 kernel");
      if (wanted_kernel) {
        kernel_constant_bank bank;
        bank.index = parse_u32(columns[1], "constant bank index");
        bank.data = parse_hex_bytes(columns[2], "constant bank data");
        result.constant_banks.push_back(std::move(bank));
      }
    } else if (record == "INST") {
      require_columns(columns, format_version == 1 ? 14 : 15, "INST");
      if (!in_kernel || in_instruction)
        panic("misplaced INST record");
      in_instruction = true;
      if (wanted_kernel) {
        instruction inst;
        inst.pc = parse_u64(columns[1], "pc");
        inst.raw.lo = parse_u64(columns[2], "raw lo");
        inst.raw.hi = parse_u64(columns[3], "raw hi");
        inst.decoded = parse_bool(columns[4], "decoded flag");
        if (format_version == 1) {
          inst.opcode = unescape(columns[5]);
          inst.key = unescape(columns[6]);
          inst.modifier_group = unescape(columns[7]);
          inst.control.stall = parse_u8(columns[8], "stall");
          inst.control.yield_flag = parse_bool(columns[9], "yield flag");
          inst.control.write_barrier = parse_u8(columns[10], "write barrier");
          inst.control.read_barrier = parse_u8(columns[11], "read barrier");
          inst.control.wait_mask = parse_u8(columns[12], "wait mask");
          inst.control.reuse_mask = parse_u8(columns[13], "reuse mask");
        } else {
          inst.decoder_source = unescape(columns[5]);
          inst.predicate_text = unescape(columns[6]);
          inst.opcode = unescape(columns[7]);
          inst.operand_text = unescape(columns[8]);
          inst.control.stall = parse_u8(columns[9], "stall");
          inst.control.yield_flag = parse_bool(columns[10], "yield flag");
          inst.control.write_barrier = parse_u8(columns[11], "write barrier");
          inst.control.read_barrier = parse_u8(columns[12], "read barrier");
          inst.control.wait_mask = parse_u8(columns[13], "wait mask");
          inst.control.reuse_mask = parse_u8(columns[14], "reuse mask");
        }
        result.instructions.push_back(std::move(inst));
      }
    } else if (record == "FIELD") {
      require_columns(columns, 7, "FIELD");
      if (!in_instruction)
        panic("FIELD outside an instruction");
      if (wanted_kernel) {
        decoded_field field;
        field.name = unescape(columns[1]);
        field.shift = parse_u32(columns[2], "field shift");
        field.bits = parse_u32(columns[3], "field width");
        field.value = parse_u64(columns[4], "field value");
        field.token_index = parse_i32(columns[5], "field token index");
        field.extraction = unescape(columns[6]);
        result.instructions.back().fields.push_back(std::move(field));
      }
    } else if (record == "ATTR") {
      require_columns(columns, 3, "ATTR");
      if (!in_instruction || format_version < 2)
        panic("ATTR outside a SASSIR v2+ instruction");
      if (wanted_kernel) {
        instruction_attribute attribute;
        attribute.name = unescape(columns[1]);
        attribute.value = unescape(columns[2]);
        if (attribute.name.empty())
          panic("empty instruction attribute name");
        result.instructions.back().attributes.push_back(std::move(attribute));
      }
    } else if (record == "ENDINST") {
      require_columns(columns, 1, "ENDINST");
      if (!in_instruction)
        panic("ENDINST outside an instruction");
      if (wanted_kernel && format_version >= 2)
        parse_instruction_operands(result.instructions.back());
      in_instruction = false;
    } else if (record == "ENDKERNEL") {
      require_columns(columns, 1, "ENDKERNEL");
      if (!in_kernel || in_instruction)
        panic("misplaced ENDKERNEL record");
      if (wanted_kernel) {
        if (format_version >= 3 && !result.has_resources)
          panic("SASSIR v3 kernel has no resources");
        validate_kernel(result);
        found_kernel = true;
      }
      in_kernel = false;
      wanted_kernel = false;
    } else {
      panic("unknown sassir record: " + record);
    }
  }

  if (!saw_header)
    panic("missing SASSIR header");
  if (in_kernel || in_instruction)
    panic("truncated SASS sassir");
  if (!found_kernel)
    panic("kernel not found in SASS sassir: " + kernel_name);
  return result;
}

} // namespace sass
} // namespace flash_gpgpu_sim
