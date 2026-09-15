#include "timing_projection.h"
#include "../../panic.h"
#include "../decode/wgmma_instruction.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace flash_gpgpu_sim {
namespace sass {
namespace {

constexpr unsigned kVectorRegisterBase = 1;
constexpr unsigned kUniformRegisterBase =
    kVectorRegisterBase + kVectorRegisters;
constexpr unsigned kPredicateRegisterBase =
    kUniformRegisterBase + kUniformRegisters;
constexpr unsigned kUniformPredicateRegisterBase =
    kPredicateRegisterBase + kPredicateRegisters;

template <size_t Size>
std::array<unsigned, Size> parse_table(const char *text, const char *name,
                                       bool allow_broadcast = false) {
  if (text == nullptr)
    panic(std::string("missing SASS timing option ") + name);

  std::array<unsigned, Size> result{};
  std::istringstream input(text);
  for (size_t index = 0; index < Size; ++index) {
    unsigned value = 0;
    if (!(input >> value) || value == 0)
      panic(std::string("invalid SASS timing option ") + name + ": " + text);
    result[index] = value;
    if (index + 1 == Size)
      break;
    char separator = 0;
    if (!(input >> separator) || separator != ',') {
      if (allow_broadcast && index == 0 && input.eof()) {
        result.fill(result[0]);
        return result;
      }
      panic(std::string("invalid SASS timing option ") + name + ": " + text);
    }
  }
  input >> std::ws;
  if (!input.eof())
    panic(std::string("invalid SASS timing option ") + name + ": " + text);
  return result;
}

unsigned parse_scalar(const char *text, const char *name) {
  return parse_table<1>(text, name)[0];
}

template <size_t Size>
void validate_pipeline(const std::array<unsigned, Size> &latency,
                       const std::array<unsigned, Size> &initiation,
                       const char *name) {
  for (size_t index = 0; index < Size; ++index) {
    if (initiation[index] > latency[index])
      panic(std::string("SASS timing initiation exceeds latency for ") + name);
  }
}

std::string base_opcode(const std::string &opcode) {
  const size_t dot = opcode.find('.');
  return opcode.substr(0, dot);
}

bool has_modifier(const std::string &opcode, const char *modifier) {
  const std::string needle = std::string(".") + modifier;
  size_t position = opcode.find(needle);
  while (position != std::string::npos) {
    const size_t end = position + needle.size();
    if (end == opcode.size() || opcode[end] == '.')
      return true;
    position = opcode.find(needle, position + 1);
  }
  return false;
}

unsigned canonical_register(operand_kind kind, unsigned index) {
  switch (kind) {
  case operand_kind::kRegister:
    return index == kZeroRegister ? 0 : kVectorRegisterBase + index;
  case operand_kind::kUniformRegister:
    return index == kZeroUniformRegister ? 0 : kUniformRegisterBase + index;
  case operand_kind::kPredicate:
    return index == kTruePredicate ? 0 : kPredicateRegisterBase + index;
  case operand_kind::kUniformPredicate:
    return index == kTruePredicate ? 0 : kUniformPredicateRegisterBase + index;
  default:
    return 0;
  }
}

void append_unique(std::vector<unsigned> &registers, unsigned value) {
  if (value == 0 ||
      std::find(registers.begin(), registers.end(), value) != registers.end())
    return;
  registers.push_back(value);
}

void append_register(std::vector<unsigned> &registers, operand_kind kind,
                     unsigned index, bool wide) {
  append_unique(registers, canonical_register(kind, index));
  if (wide && (kind == operand_kind::kRegister ||
               kind == operand_kind::kUniformRegister))
    append_unique(registers, canonical_register(kind, index + 1));
}

void append_address_expression(std::vector<unsigned> &registers,
                               const address_expression &address) {
  for (const address_term &term : address.terms) {
    if (term.kind == address_term_kind::kRegister)
      append_register(registers, operand_kind::kRegister, term.index,
                      term.wide);
    else if (term.kind == address_term_kind::kUniformRegister)
      append_register(registers, operand_kind::kUniformRegister, term.index,
                      term.wide);
  }
}

void append_source_registers(std::vector<unsigned> &registers,
                             const operand &value) {
  switch (value.kind) {
  case operand_kind::kRegister:
  case operand_kind::kUniformRegister:
  case operand_kind::kPredicate:
  case operand_kind::kUniformPredicate:
    append_register(registers, value.kind, value.index, value.wide);
    break;
  case operand_kind::kPredicateRegisterFile:
    for (unsigned predicate = 0; predicate < kTruePredicate; ++predicate)
      append_register(registers, operand_kind::kPredicate, predicate, false);
    break;
  case operand_kind::kUniformPredicateRegisterFile:
    for (unsigned predicate = 0; predicate < kTruePredicate; ++predicate)
      append_register(registers, operand_kind::kUniformPredicate, predicate,
                      false);
    break;
  case operand_kind::kConstantMemory:
    append_address_expression(registers, value.constant_address);
    break;
  case operand_kind::kDescriptorAddress:
    append_register(registers, operand_kind::kUniformRegister,
                    value.descriptor_register, false);
    append_register(registers, operand_kind::kRegister, value.address_register,
                    true);
    break;
  case operand_kind::kTensorMapDescriptor:
  case operand_kind::kGmmaDescriptor:
    append_register(registers, operand_kind::kUniformRegister,
                    value.descriptor_register, false);
    break;
  case operand_kind::kMemoryAddress:
    if (value.has_prefix)
      append_register(registers, value.prefix_kind, value.prefix_index,
                      value.prefix_wide);
    for (const address_expression &address : value.address_groups)
      append_address_expression(registers, address);
    break;
  case operand_kind::kRegisterSet:
    for (unsigned index : value.register_set)
      append_register(registers, operand_kind::kRegister, index, false);
    break;
  default:
    break;
  }
}

bool reads_register_file(const operand &value) {
  std::vector<unsigned> registers;
  append_source_registers(registers, value);
  return !registers.empty();
}

bool has_destination(const std::string &base) {
  return base != "ST" && base != "STG" && base != "STS" && base != "STL" &&
         base != "STSM" && base != "BRA" && base != "BRX" && base != "BSSY" &&
         base != "BSYNC" && base != "BREAK" && base != "BRXU" &&
         base != "WARPSYNC" && base != "CALL" && base != "EXIT" &&
         base != "RET" && base != "NOP" && base != "YIELD" && base != "BAR" &&
         base != "MEMBAR" && base != "DEPBAR" && base != "CCTL" &&
         base != "UTMACCTL" && base != "UTMACMDFLUSH" && base != "FENCE" &&
         base != "UTMALDG" && base != "UTMASTG" && base != "UTMAPF" &&
         base != "UBLKCP" && base != "UBLKRED" && base != "LDGSTS" &&
         base != "LDGDEPBAR";
}

unsigned memory_width(const std::string &opcode) {
  if (has_modifier(opcode, "128"))
    return 16;
  if (has_modifier(opcode, "64"))
    return 8;
  if (has_modifier(opcode, "16"))
    return 2;
  if (has_modifier(opcode, "8"))
    return 1;
  return 4;
}

unsigned matrix_count(const std::string &opcode) {
  if (has_modifier(opcode, "4"))
    return 4;
  if (has_modifier(opcode, "2"))
    return 2;
  return 1;
}

unsigned combined_destination_width(const instruction &source) {
  return base_opcode(source.opcode) == "LDSM"
             ? matrix_count(source.opcode)
             : std::max(1u, memory_width(source.opcode) / 4);
}

unsigned tensor_dimension(const std::string &opcode) {
  for (unsigned dimension = 1; dimension <= 5; ++dimension) {
    const std::string modifier = std::to_string(dimension) + "D";
    if (has_modifier(opcode, modifier.c_str()))
      return dimension;
  }
  return 0;
}

unsigned integer_profile_index(const std::string &base) {
  if (base == "SHFL")
    return 5;
  if (base == "IMAD" || base == "UIMAD")
    return 3;
  return 0;
}

unsigned fp32_profile_index(const std::string &base) {
  if (base == "FMNMX")
    return 1;
  if (base == "FMUL" || base == "UFMUL")
    return 2;
  if (base == "FFMA" || base == "UFFMA" || base == "HFMA2")
    return 3;
  return 0;
}

unsigned tensor_profile_index(const std::string &opcode) {
  const bool integer = has_modifier(opcode, "S8") || has_modifier(opcode, "U8");
  if (has_modifier(opcode, "16816"))
    return integer ? 6 : 0;
  if (has_modifier(opcode, "1688")) {
    if (has_modifier(opcode, "TF32"))
      return 1;
    if (has_modifier(opcode, "BF16"))
      return 4;
    return 3;
  }
  if (has_modifier(opcode, "16832"))
    return 2;
  if (has_modifier(opcode, "1684"))
    return 5;
  return 0;
}

std::array<unsigned, 3> hmma_source_widths(const std::string &opcode) {
  const bool wide =
      has_modifier(opcode, "16816") ||
      (has_modifier(opcode, "1688") && has_modifier(opcode, "TF32"));
  return {{wide ? 4u : 2u, wide ? 2u : 1u, 4u}};
}

struct wgmma_shape {
  unsigned n = 0;
  unsigned k = 0;
};

wgmma_shape parse_wgmma_shape(const std::string &opcode) {
  const size_t prefix = opcode.find(".64x");
  if (prefix == std::string::npos)
    panic("unsupported SASS WGMMA shape for opcode " + opcode);
  const size_t n_begin = prefix + 4;
  const size_t separator = opcode.find('x', n_begin);
  const size_t k_begin =
      separator == std::string::npos ? std::string::npos : separator + 1;
  const size_t suffix = k_begin == std::string::npos
                            ? std::string::npos
                            : opcode.find('.', k_begin);
  if (separator == std::string::npos || suffix == std::string::npos)
    panic("unsupported SASS WGMMA shape for opcode " + opcode);

  wgmma_shape result;
  const auto dimension = [&opcode](size_t begin, size_t end) {
    const std::string text = opcode.substr(begin, end - begin);
    if (text.empty() || !std::isdigit(static_cast<unsigned char>(text.front())))
      panic("unsupported SASS WGMMA shape for opcode " + opcode);
    char *tail = nullptr;
    errno = 0;
    const auto value = std::strtoul(text.c_str(), &tail, 10);
    if (errno != 0 || tail != text.c_str() + text.size() ||
        value > std::numeric_limits<unsigned>::max())
      panic("unsupported SASS WGMMA shape for opcode " + opcode);
    return static_cast<unsigned>(value);
  };
  result.n = dimension(n_begin, separator);
  result.k = dimension(k_begin, suffix);
  if (result.n == 0 || result.k == 0)
    panic("unsupported SASS WGMMA shape for opcode " + opcode);
  return result;
}

unsigned wgmma_shape_value(unsigned shape_n,
                           const std::array<unsigned, 4> &table) {
  if (shape_n <= 8)
    return table[0];
  if (shape_n <= 16)
    return table[1];
  if (shape_n <= 32)
    return table[2];
  if (shape_n <= 64)
    return table[3];
  return table[3] * ((shape_n + 63) / 64);
}

unsigned wgmma_completion_value(unsigned shape_n,
                                const std::array<unsigned, 4> &table) {
  if (shape_n <= 8)
    return table[0];
  if (shape_n <= 16)
    return table[1];
  if (shape_n <= 32)
    return table[2];
  // The completion tail is an asynchronous pipeline drain, not additional
  // tensor work.  Native N=128/176/256 sweeps retain one tail rather than one
  // tail per 64-column slice.
  return table[3];
}

unsigned wgmma_compute_cycles(const wgmma_shape &shape,
                              unsigned work_per_cycle) {
  if (work_per_cycle == 0)
    panic("SASS WGMMA compute throughput must be nonzero");
  const unsigned long long work =
      2ull * 64ull * static_cast<unsigned long long>(shape.n) * shape.k;
  return static_cast<unsigned>(std::max<unsigned long long>(
      1, (work + work_per_cycle - 1) / work_per_cycle));
}

void classify(const instruction &source, const timing_profile &profile,
              timing_instruction &target) {
  const std::string base = base_opcode(source.opcode);
  const bool uniform_arithmetic =
      base == "UF2I" || base == "UFADD" || base == "UFMUL" || base == "UFFMA" ||
      base == "UI2F" || base == "UIADD3" || base == "UIMAD" ||
      base == "UISETP" || base == "UFSETP" || base == "UFSEL" ||
      base == "ULEA" || base == "ULOP3" || base == "UPLOP3" || base == "UMOV" ||
      base == "UPRMT" || base == "USEL" || base == "USGXT" || base == "USHF" ||
      base == "UVIMNMX";
  const bool uniform_data_movement = base == "UP2UR";
  target.op = ALU_OP;
  target.oprnd_type = INT_OP;
  target.sp_op = INT__OP;
  target.op_pipe = INTP__OP;
  const unsigned integer_index = integer_profile_index(base);
  target.latency = profile.integer_latency[integer_index];
  target.initiation_interval = profile.integer_initiation[integer_index];

  if (base == "LD" || base == "LDG" || base == "LDS" || base == "LDL" ||
      base == "LDC" || base == "LDCU" || base == "ULDC") {
    target.op = LOAD_OP;
    target.op_pipe = MEM__OP;
    target.memory_op = memory_load;
    target.data_size = memory_width(source.opcode);
    target.space = memory_space_t(base == "LD"    ? generic_space
                                  : base == "LDG" ? global_space
                                  : base == "LDS" ? shared_space
                                  : base == "LDL" ? local_space
                                                  : const_space);
    target.cache_op =
        has_modifier(source.opcode, "LU")                     ? CACHE_LAST_USE
        : base == "LDG" && has_modifier(source.opcode, "GPU") ? CACHE_GLOBAL
        : base == "LDG" && has_modifier(source.opcode, "SYS") ? CACHE_VOLATILE
                                                              : CACHE_ALL;
    target.latency = 1;
    target.initiation_interval = 1;
    if (base == "LDS")
      target.set_mio_client();
  } else if (base == "ST" || base == "STG" || base == "STS" || base == "STL") {
    target.op = STORE_OP;
    target.op_pipe = MEM__OP;
    target.memory_op = memory_store;
    target.data_size = memory_width(source.opcode);
    target.space = memory_space_t(base == "ST"    ? generic_space
                                  : base == "STG" ? global_space
                                  : base == "STS" ? shared_space
                                                  : local_space);
    target.cache_op = CACHE_WRITE_BACK;
    target.latency = 1;
    target.initiation_interval = 1;
    if (base == "STS")
      target.set_mio_client();
  } else if (base == "LDSM" || base == "STSM") {
    target.op = base == "LDSM" ? TENSOR_CORE_LOAD_OP : TENSOR_CORE_STORE_OP;
    target.op_pipe = MEM__OP;
    target.memory_op = base == "LDSM" ? memory_load : memory_store;
    target.data_size = has_modifier(source.opcode, "MT88") ? 2 : 4;
    target.space = memory_space_t(shared_space);
    target.cache_op = CACHE_UNDEFINED;
    target.latency = 1;
    target.initiation_interval = 1;
    // Matrix shared-memory operations enter through the MIO frontend before
    // consuming the SM-wide shared data path.
    target.set_mio_client();
  } else if (base == "ATOMG" || source.opcode == "ATOMS.ADD") {
    target.op = LOAD_OP;
    target.op_pipe = MEM__OP;
    target.memory_op = memory_load;
    target.data_size = 4;
    target.space = memory_space_t(global_space);
    // Global atomics are serviced below L1. The SASS functional frontend has
    // already performed the architectural RMW, so this is a timing-only
    // atomic transaction: it must wait for the backend response without
    // invoking the PTX functional callback a second time.
    // ATOMS.ADD uses this same coarse L2 proxy until shared-atomic timing is
    // calibrated. Its functional RMW remains in CTA shared memory; runtime
    // maps only the timing request into the reserved shared generic window.
    target.cache_op = CACHE_GLOBAL;
    target.set_atomic(false);
    target.latency = 1;
    target.initiation_interval = 1;
  } else if (base == "LDGSTS") {
    target.op = ASYNC_COPY_OP;
    target.op_pipe = CP_ASYNC__OP;
    target.memory_op = memory_load;
    target.data_size = 16;
    target.space = memory_space_t(global_space);
    target.cache_op = CACHE_ALL;
    target.latency = profile.cp_async_latency;
    target.initiation_interval = profile.cp_async_initiation;
    target.m_is_ldgsts = true;
  } else if (base == "LDGDEPBAR") {
    target.op = ASYNC_COPY_OP;
    target.op_pipe = CP_ASYNC__OP;
    target.latency = profile.cp_async_commit_latency;
    target.initiation_interval = profile.cp_async_commit_initiation;
    target.m_is_ldgdepbar = true;
  } else if (base == "ARRIVES") {
    const bool transaction_count =
        source.opcode == "ARRIVES.LDGSTSBAR.64.TRANSCNT";
    const bool arrival_count = source.opcode == "ARRIVES.LDGSTSBAR.64.ARVCNT";
    if (!transaction_count && !arrival_count)
      panic(
          "unsupported SASS cp.async mbarrier arrival projection for opcode " +
          source.opcode);
    target.op = ASYNC_COPY_OP;
    target.op_pipe = CP_ASYNC__OP;
    target.latency = profile.cp_async_commit_latency;
    target.initiation_interval = profile.cp_async_commit_initiation;
    target.m_is_cp_async_mbarrier_arrive = true;
    inst_t::async_copy_static_info_t info;
    info.mbarrier_increment_pending = transaction_count;
    target.set_async_copy_static_info(info);
  } else if (source.opcode == "DEPBAR.LE" && source.operands_structured &&
             source.operands.size() == 2 &&
             source.operands[0].kind == operand_kind::kScoreboardRegister &&
             source.operands[0].index == 0 &&
             source.operands[1].kind == operand_kind::kImmediate &&
             source.operands[1].immediate >= 0) {
    target.op = ASYNC_COPY_OP;
    target.op_pipe = CP_ASYNC__OP;
    target.latency = profile.cp_async_wait_latency;
    target.initiation_interval = profile.cp_async_wait_initiation;
    target.m_is_depbar = true;
    target.m_depbar_group_no =
        static_cast<unsigned>(source.operands[1].immediate);
    // Native DEPBAR is shared by ordinary cp.async and TMA store bulk groups.
    // Runtime state selects the live producer class when the wait issues.
    inst_t::tma_static_info_t info;
    info.tma_type = inst_t::tma_static_info_t::TMA_BULK_WAIT;
    info.bulk_wait_num = target.m_depbar_group_no;
    target.set_tma_static_info(info);
  } else if (base == "FADD" || base == "FMUL" || base == "FFMA" ||
             base == "FMNMX" || base == "FSET" || base == "FSETP" ||
             base == "UFSETP" || base == "FSEL" || base == "UFSEL" ||
             base == "F2F" || base == "F2FP" || base == "F2I" ||
             base == "I2F" || base == "I2FP" || base == "UF2I" ||
             base == "UI2F" || base == "UFADD" || base == "UFMUL" ||
             base == "UFFMA" || base == "HADD2" || base == "HFMA2") {
    target.op = SP_OP;
    target.oprnd_type = FP_OP;
    target.sp_op = (base == "FMUL" || base == "UFMUL") ? FP_MUL_OP : FP__OP;
    target.op_pipe = SP__OP;
    const unsigned fp32_index = fp32_profile_index(base);
    target.latency = profile.fp32_latency[fp32_index];
    target.initiation_interval = profile.fp32_initiation[fp32_index];
  } else if (base == "MUFU") {
    target.op = SFU_OP;
    target.oprnd_type = FP_OP;
    target.sp_op = has_modifier(source.opcode, "EX2") ? FP_EXP_OP : FP__OP;
    target.op_pipe = SFU__OP;
    target.latency = profile.sfu_latency;
    target.initiation_interval = profile.sfu_initiation;
    target.set_mio_client();
  } else if (is_hgmma_commit_group_sentinel(source)) {
    inst_t::wgmma_static_info_t info;
    info.operation = inst_t::wgmma_static_info_t::WGMMA_COMMIT_GROUP;
    target.set_wgmma_static_info(info);
  } else if (base == "HGMMA" || base == "QGMMA" || base == "IGMMA" ||
             base == "BGMMA") {
    const wgmma_shape shape = parse_wgmma_shape(source.opcode);
    const bool register_a = source.operands_structured &&
                            source.operands.size() > 1 &&
                            source.operands[1].kind == operand_kind::kRegister;
    const bool integer = base == "IGMMA" || base == "BGMMA";
    const unsigned throughput_index = base == "QGMMA"                       ? 2
                                      : base == "IGMMA"                     ? 3
                                      : base == "BGMMA"                     ? 4
                                      : has_modifier(source.opcode, "TF32") ? 1
                                                                            : 0;
    const std::array<unsigned, 4> &latency =
        register_a ? profile.wgmma_rs_latency : profile.wgmma_ss_latency;
    const std::array<unsigned, 4> &initiation =
        register_a ? profile.wgmma_rs_initiation : profile.wgmma_ss_initiation;
    const std::array<unsigned, 4> &completion =
        register_a ? (integer ? profile.wgmma_int_rs_completion
                              : profile.wgmma_rs_completion)
                   : (integer ? profile.wgmma_int_ss_completion
                              : profile.wgmma_ss_completion);
    target.op = TENSOR_CORE_OP;
    target.oprnd_type = integer ? INT_OP : FP_OP;
    target.sp_op = TENSOR__OP;
    target.op_pipe = TENSOR_CORE__OP;
    target.latency = wgmma_shape_value(shape.n, latency);
    target.initiation_interval = wgmma_shape_value(shape.n, initiation);
    target.wgmma_compute_latency = wgmma_compute_cycles(
        shape, profile.wgmma_compute_throughput[throughput_index]);
    target.wgmma_completion_tail_latency =
        wgmma_completion_value(shape.n, completion);

    inst_t::wgmma_static_info_t info;
    info.operation = inst_t::wgmma_static_info_t::WGMMA_MMA_ASYNC;
    info.accumulator_bytes_per_thread = (shape.n / 2) * sizeof(uint32_t);
    info.register_a_registers_per_thread = register_a ? 4 : 0;
    info.commit_group_after_issue =
        std::any_of(source.operands.begin(), source.operands.end(),
                    [](const operand &value) {
                      return value.kind == operand_kind::kScoreboardRegister;
                    });
    target.set_wgmma_static_info(info);
  } else if (base == "HMMA") {
    target.op = TENSOR_CORE_OP;
    target.oprnd_type = FP_OP;
    target.sp_op = TENSOR__OP;
    target.op_pipe = TENSOR_CORE__OP;
    const unsigned tensor_index = tensor_profile_index(source.opcode);
    target.latency = profile.tensor_latency[tensor_index];
    target.initiation_interval = profile.tensor_initiation[tensor_index];
  } else if (source.opcode == "WARPGROUP.DEPBAR.LE") {
    if (!source.operands_structured || source.operands.size() != 2 ||
        source.operands[0].kind != operand_kind::kScoreboardRegister ||
        source.operands[1].kind != operand_kind::kImmediate ||
        source.operands[1].immediate < 0)
      panic("unsupported SASS warpgroup wait timing projection");
    inst_t::wgmma_static_info_t info;
    info.operation = inst_t::wgmma_static_info_t::WGMMA_WAIT_GROUP;
    info.wait_group_num = static_cast<unsigned>(source.operands[1].immediate);
    target.set_wgmma_static_info(info);
  } else if (base == "BRA" || base == "BRX" || base == "BRXU" ||
             base == "BSSY" || base == "BSYNC" || base == "BREAK" ||
             base == "WARPSYNC" || base == "ENDCOLLECTIVE") {
    target.op = BRANCH_OP;
  } else if (base == "CALL" || base == "RET") {
    target.op = CALL_OPS;
  } else if (base == "EXIT") {
    target.op = EXIT_OPS;
  } else if (base == "BAR") {
    target.op = BARRIER_OP;
    target.bar_type =
        source.opcode.find(".ARV") != std::string::npos ||
                source.opcode.find(".DEFER_BLOCKING") != std::string::npos
            ? ARRIVE
            : SYNC;
    if (!source.operands.empty() &&
        source.operands[0].kind == operand_kind::kImmediate)
      target.set_bar_id(static_cast<unsigned>(source.operands[0].immediate));
    if (source.operands.size() > 1 &&
        source.operands[1].kind == operand_kind::kImmediate)
      target.set_bar_count(static_cast<unsigned>(source.operands[1].immediate));
  } else if (base == "MEMBAR" || base == "FENCE" || base == "ACQBULK") {
    target.op = MEMORY_BARRIER_OP;
    if (source.opcode == "FENCE.VIEW.ASYNC.S")
      target.set_async_proxy_fence();
  } else if (base == "SYNCS") {
    target.op = MBARRIER_OP;
    inst_t::mbarrier_static_info_t info;
    if (source.opcode == "SYNCS.EXCH.64") {
      info.operation = inst_t::mbarrier_static_info_t::MBARRIER_INIT;
    } else if (source.opcode.find(".PHASECHK.") != std::string::npos) {
      info.operation = inst_t::mbarrier_static_info_t::MBARRIER_TRY_WAIT;
    } else if (source.opcode == "SYNCS.ARRIVE.TRANS64.RED.A0TX") {
      info.operation = inst_t::mbarrier_static_info_t::MBARRIER_COMPLETE_TX;
    } else if (source.opcode.find(".ARRIVE.") != std::string::npos) {
      info.operation =
          inst_t::mbarrier_static_info_t::MBARRIER_ARRIVE_EXPECT_TX;
      const bool a1t0 = source.opcode.find(".A1T0") != std::string::npos;
      const bool art0 = source.opcode.find(".ART0") != std::string::npos;
      const bool a0t1 = source.opcode.find(".A0T1") != std::string::npos;
      // The unqualified TRANS64 form performs both halves of
      // mbarrier.arrive.expect_tx: it contributes one arrival and takes the
      // transaction-byte count from its register operand.  A1T0/ART0 are
      // arrival-only forms.  RED.A0T1 remains statically expect-tx-shaped, but
      // the synchronous functional LDGSTS path emits no dynamic effect because
      // that cp.async transaction token has already completed.
      info.arrive = !a0t1;
      info.expect_tx = !a1t0 && !art0;
    } else if (source.opcode.find(".CCTL.IV") != std::string::npos) {
      info.operation = inst_t::mbarrier_static_info_t::MBARRIER_INVAL;
    }
    if (info.operation == inst_t::mbarrier_static_info_t::MBARRIER_INVALID)
      panic("unsupported SASS mbarrier timing projection for opcode " +
            source.opcode);
    target.set_mbarrier_static_info(info);
  } else if (base == "UTMACMDFLUSH") {
    target.op = TENSOR_MEMORY_ACCELERATOR_OP;
    target.latency = profile.cp_async_commit_latency;
    target.initiation_interval = profile.cp_async_commit_initiation;
    inst_t::tma_static_info_t info;
    info.tma_type = inst_t::tma_static_info_t::TMA_BULK_COMMIT;
    target.set_tma_static_info(info);
  } else if (base == "UTMAPF") {
    target.op = TENSOR_MEMORY_ACCELERATOR_OP;
    target.latency = profile.tma_latency;
    target.initiation_interval = profile.tma_initiation;
  } else if (base == "UBLKCP" || base == "UBLKRED") {
    const bool load = source.opcode == "UBLKCP.S.G";
    const bool store = source.opcode == "UBLKCP.G.S" ||
                       source.opcode == "UBLKRED.G.S.ADD.F32.RN";
    if (!load && !store)
      panic("unsupported SASS uniform bulk-copy timing projection for opcode " +
            source.opcode);
    target.op = TENSOR_MEMORY_ACCELERATOR_OP;
    target.latency = profile.tma_latency;
    target.initiation_interval = profile.tma_initiation;
    inst_t::tma_static_info_t info;
    info.tma_type = inst_t::tma_static_info_t::TMA_NORMAL;
    info.dst_space = load ? inst_t::tma_static_info_t::TMA_SHARED_CTA
                          : inst_t::tma_static_info_t::TMA_GLOBAL;
    info.src_space = load ? inst_t::tma_static_info_t::TMA_GLOBAL
                          : inst_t::tma_static_info_t::TMA_SHARED_CTA;
    target.set_tma_static_info(info);
  } else if (base == "UTMALDG" || base == "UTMASTG") {
    const unsigned dimension = tensor_dimension(source.opcode);
    if (dimension == 0)
      panic("unsupported SASS tensor TMA dimension for opcode " +
            source.opcode);
    target.op = TENSOR_MEMORY_ACCELERATOR_OP;
    target.latency = profile.tma_latency;
    target.initiation_interval = profile.tma_initiation;
    inst_t::tma_static_info_t info;
    info.tma_type = inst_t::tma_static_info_t::TMA_TENSOR;
    info.tensor_dim = dimension;
    info.dst_space = base == "UTMALDG"
                         ? inst_t::tma_static_info_t::TMA_SHARED_CTA
                         : inst_t::tma_static_info_t::TMA_GLOBAL;
    info.src_space = base == "UTMALDG"
                         ? inst_t::tma_static_info_t::TMA_GLOBAL
                         : inst_t::tma_static_info_t::TMA_SHARED_CTA;
    target.set_tma_static_info(info);
  } else if (base == "NOP" || base == "NANOSLEEP" || base == "S2R" ||
             base == "S2UR" || base == "CS2R" || base == "CS2UR" ||
             base == "MOV" || base == "MOVM" || base == "IMAD" ||
             base == "IADD3" || base == "ISETP" || base == "LOP3" ||
             base == "SHF" || base == "PRMT" || base == "SEL" ||
             base == "PLOP3" || base == "VOTE" || base == "LEA" ||
             base == "IABS" || base == "FLO" || base == "POPC" ||
             base == "UIADD3" || base == "UIMAD" || base == "UISETP" ||
             base == "UFSETP" || base == "UFSEL" || base == "ULEA" ||
             base == "ULOP3" || base == "UPLOP3" || base == "UMOV" ||
             base == "UPRMT" || base == "USEL" || base == "USHF" ||
             base == "R2UR" || base == "UP2UR" || base == "R2P" ||
             base == "P2R" || base == "LEPC" || base == "ELECT" ||
             base == "VOTEU" || base == "VIMNMX" || base == "UVIMNMX" ||
             base == "IADD" || base == "VIADD" || base == "VIADDMNMX" ||
             base == "SGXT" || base == "USGXT" || base == "SHFL" ||
             base == "DEPBAR" || base == "CCTL" || base == "UTMACCTL" ||
             base == "USETMAXREG" || source.opcode == "WARPGROUP.ARRIVE" ||
             base == "YIELD") {
    // Default integer/ALU classification above is intentional.
  } else {
    panic("unsupported SASS timing projection for opcode " + source.opcode);
  }

  if ((uniform_arithmetic || uniform_data_movement) &&
      profile.uniform_unit_index >= 0) {
    target.op =
        static_cast<op_type>(SPEC_UNIT_START_ID + profile.uniform_unit_index);
    target.op_pipe = SPECIALIZED__OP;
    target.latency = profile.uniform_latency;
    target.initiation_interval = profile.uniform_initiation;
  }
}

void populate_registers(const instruction &source, timing_instruction &target) {
  const std::string base = base_opcode(source.opcode);
  std::vector<unsigned> destinations;
  std::vector<unsigned> sources;
  size_t source_begin = 0;
  unsigned logical_source_operands = 0;
  unsigned logical_register_operands = 0;

  if (source.operands_structured && !source.operands.empty() &&
      (base == "LDS" || base == "LDL" || base == "LDSM" || base == "SYNCS" ||
       base == "ATOMS") &&
      source.operands.front().kind == operand_kind::kMemoryAddress &&
      source.operands.front().has_prefix) {
    const operand &combined = source.operands.front();
    const unsigned width = combined_destination_width(source);
    append_register(destinations, combined.prefix_kind, combined.prefix_index,
                    width > 1);
    if (width > 2)
      for (unsigned word = 2; word < width; ++word)
        append_register(destinations, combined.prefix_kind,
                        combined.prefix_index + word, false);
    for (const address_expression &address : combined.address_groups)
      append_address_expression(sources, address);
    source_begin = 1;
    ++logical_source_operands;
    if (!sources.empty())
      ++logical_register_operands;
  } else if (source.operands_structured && !source.operands.empty() &&
             base == "HMMA") {
    if (source.operands.size() != 4)
      panic("HMMA timing projection requires four operands");
    const operand &destination = source.operands.front();
    append_register(destinations, destination.kind, destination.index, true);
    append_register(destinations, destination.kind, destination.index + 2,
                    true);
    source_begin = 1;
    const auto widths = hmma_source_widths(source.opcode);
    for (size_t index = source_begin; index < source.operands.size(); ++index) {
      const operand &value = source.operands[index];
      const unsigned width = widths[index - source_begin];
      if (canonical_register(value.kind, value.index) != 0)
        for (unsigned word = 0; word < width; ++word)
          append_register(sources, value.kind, value.index + word, false);
      ++logical_source_operands;
      if (reads_register_file(value))
        ++logical_register_operands;
    }
    source_begin = source.operands.size();
  } else if (source.operands_structured && source.operands.size() >= 3 &&
             base == "ATOMG") {
    // nvdisasm prints an auxiliary predicate before the returned-old-value
    // destination. Descriptor forms therefore use operand 1 as the actual
    // register destination and operands 2+ as address/value sources. The
    // compact discarded-result form combines RZ with its address in operand
    // 1 and has no architectural destination.
    if (source.operands.size() >= 4) {
      append_source_registers(destinations, source.operands[1]);
      source_begin = 2;
    } else {
      source_begin = 1;
    }
  } else if (source.operands_structured && source.operands.size() >= 2 &&
             base == "SHFL") {
    for (size_t index = 0; index < 2; ++index)
      append_source_registers(destinations, source.operands[index]);
    source_begin = 2;
  } else if (source.operands_structured && source.operands.size() >= 2 &&
             (base == "PLOP3" || base == "UPLOP3")) {
    append_source_registers(destinations, source.operands[0]);
    append_source_registers(destinations, source.operands[1]);
    source_begin = 2;
  } else if (source.operands_structured && !source.operands.empty() &&
             base == "BRXU" &&
             source.operands[0].kind == operand_kind::kUniformRegister) {
    append_register(sources, source.operands[0].kind, source.operands[0].index,
                    true);
    source_begin = 1;
    ++logical_source_operands;
    ++logical_register_operands;
  } else if (source.operands_structured && !source.operands.empty() &&
             has_destination(base)) {
    const operand &destination = source.operands.front();
    append_source_registers(destinations, destination);
    const bool memory_load = base == "LD" || base == "LDG" || base == "LDS" ||
                             base == "LDL" || base == "LDC" || base == "LDCU" ||
                             base == "ULDC";
    const unsigned bytes =
        has_modifier(source.opcode, "WIDE") ||
                ((base == "MOV" || base == "UMOV" || base == "UIADD3") &&
                 has_modifier(source.opcode, "64"))
            ? 8
            : (memory_load ? memory_width(source.opcode) : 4);
    if (bytes > 4 && destinations.size() == 1 &&
        (destination.kind == operand_kind::kRegister ||
         destination.kind == operand_kind::kUniformRegister)) {
      for (unsigned word = 1; word < bytes / 4; ++word)
        append_register(destinations, destination.kind,
                        destination.index + word, false);
    }
    source_begin = 1;
  }

  for (size_t index = source_begin; index < source.operands.size(); ++index) {
    append_source_registers(sources, source.operands[index]);
    ++logical_source_operands;
    if (reads_register_file(source.operands[index]))
      ++logical_register_operands;
  }
  if (source.has_guard) {
    append_source_registers(sources, source.guard);
    ++logical_source_operands;
    if (reads_register_file(source.guard))
      ++logical_register_operands;
  }
  if (source.has_auxiliary_predicate) {
    append_source_registers(sources, source.auxiliary_predicate);
    ++logical_source_operands;
    if (reads_register_file(source.auxiliary_predicate))
      ++logical_register_operands;
  }

  if (destinations.size() > std::size(target.arch_reg.dst) ||
      sources.size() > std::size(target.arch_reg.src))
    panic("too many SASS timing register operands for " + source.opcode);

  for (size_t index = 0; index < destinations.size(); ++index) {
    target.out[index] = destinations[index];
    target.arch_reg.dst[index] = static_cast<int>(destinations[index]);
  }
  for (size_t index = 0; index < sources.size(); ++index) {
    target.in[index] = sources[index];
    target.arch_reg.src[index] = static_cast<int>(sources[index]);
  }
  target.outcount = destinations.size();
  target.incount = sources.size();
  target.set_num_regs(logical_register_operands);
  target.set_num_operands(logical_source_operands);
}

void append_register_file_operand(timing_instruction &target,
                                  const operand &value, unsigned slot,
                                  unsigned width = 0) {
  if (value.kind != operand_kind::kRegister)
    return;
  if (value.index == kZeroRegister)
    return;

  if (width == 0)
    width = value.wide ? 2 : 1;
  for (unsigned word = 0; word < width; ++word)
    target.add_register_file_source(value.index + word, slot, value.reuse);
}

// Preserve the logical regular-register source position used by the hardware
// reuse cache.  Positions are compacted over regular SASS operands, including
// RZ but excluding immediates, uniform registers, and predicates.  In
// particular, raw reuse-control bits are not positional across opcode classes;
// nvdisasm's per-operand .reuse annotation is the normalized retain decision.
void populate_register_file_sources(const instruction &source,
                                    timing_instruction &target) {
  if (!source.operands_structured)
    return;

  const std::string base = base_opcode(source.opcode);
  size_t source_begin = has_destination(base) ? 1 : 0;
  if (base == "SHFL")
    source_begin = std::min<size_t>(2, source.operands.size());
  else if (base == "ATOMG")
    source_begin = source.operands.size() >= 4 ? 2 : 1;

  unsigned regular_slot = 0;
  for (size_t index = source_begin; index < source.operands.size(); ++index) {
    const operand &value = source.operands[index];
    if (value.kind != operand_kind::kRegister)
      continue;

    unsigned width = 0;
    if (base == "HMMA") {
      const unsigned source_index = index - source_begin;
      width = hmma_source_widths(source.opcode)[source_index];
    }
    append_register_file_operand(target, value, regular_slot, width);
    ++regular_slot;
  }
}

void append_register_file_destination(timing_instruction &target,
                                      operand_kind kind, unsigned index,
                                      unsigned width) {
  if (kind != operand_kind::kRegister || index == kZeroRegister)
    return;
  for (unsigned word = 0; word < width; ++word)
    target.add_register_file_destination(index + word);
}

void populate_register_file_destinations(const instruction &source,
                                         timing_instruction &target) {
  if (!source.operands_structured || source.operands.empty())
    return;

  const std::string base = base_opcode(source.opcode);
  if (!has_destination(base))
    return;

  const bool atomic_has_return_value =
      base == "ATOMG" && source.operands.size() >= 4;
  const operand &destination =
      atomic_has_return_value ? source.operands[1] : source.operands.front();
  if ((base == "LDS" || base == "LDL" || base == "LDSM" || base == "SYNCS" ||
       base == "ATOMS") &&
      destination.kind == operand_kind::kMemoryAddress &&
      destination.has_prefix) {
    const unsigned width = combined_destination_width(source);
    append_register_file_destination(target, destination.prefix_kind,
                                     destination.prefix_index, width);
    return;
  }

  unsigned width = 1;
  if (base == "HMMA")
    width = 4;
  else if (has_modifier(source.opcode, "WIDE") ||
           ((base == "MOV" || base == "UMOV" || base == "UIADD3") &&
            has_modifier(source.opcode, "64")) ||
           ((base == "LD" || base == "LDG" || base == "LDL" || base == "LDC" ||
             base == "LDCU" || base == "ULDC") &&
            memory_width(source.opcode) > 4))
    width = (base == "LD" || base == "LDG" || base == "LDL" || base == "LDC" ||
             base == "LDCU" || base == "ULDC")
                ? memory_width(source.opcode) / 4
                : 2;
  append_register_file_destination(target, destination.kind, destination.index,
                                   width);
}

} // namespace

void project_mbarrier_effects(const execution_context &context,
                              warp_inst_t &target) {
  target.reset_mbarrier_info();
  const bool try_wait = target.get_mbarrier_static_info().operation ==
                        inst_t::mbarrier_static_info_t::MBARRIER_TRY_WAIT;
  const functional_mbarrier_effect *representative = nullptr;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const auto &effect = context.mbarrier_effects[lane];
    if (!effect.valid)
      continue;
    if (try_wait && representative != nullptr) {
      if (effect.address != representative->address ||
          effect.parity != representative->parity) {
        std::ostringstream error;
        error
            << "nonuniform SASS mbarrier try-wait at pc 0x" << std::hex
            << target.pc << ": lane " << std::dec << lane
            << " differs in address or parity; timing requires a uniform wait";
        panic(error.str());
      }
      continue;
    }
    inst_t::mbarrier_info_t info;
    info.bar_id = effect.address;
    info.bar_count = effect.count;
    info.bar_parity = effect.parity;
    target.set_mbarrier_info(lane, info);
    representative = &effect;
  }
}

timing_profile parse_timing_profile(const timing_profile_options &options) {
  timing_profile result;
  result.integer_latency =
      parse_table<6>(options.integer_latency, "integer latency");
  result.integer_initiation =
      parse_table<6>(options.integer_initiation, "integer initiation");
  result.fp32_latency = parse_table<5>(options.fp32_latency, "fp32 latency");
  result.fp32_initiation =
      parse_table<5>(options.fp32_initiation, "fp32 initiation");
  result.sfu_latency = parse_scalar(options.sfu_latency, "sfu latency");
  result.sfu_initiation =
      parse_scalar(options.sfu_initiation, "sfu initiation");
  result.tensor_latency =
      parse_table<7>(options.tensor_latency, "tensor latency", true);
  result.tensor_initiation =
      parse_table<7>(options.tensor_initiation, "tensor initiation", true);
  if (options.wgmma_ss_latency != nullptr)
    result.wgmma_ss_latency =
        parse_table<4>(options.wgmma_ss_latency, "WGMMA SS latency");
  if (options.wgmma_rs_latency != nullptr)
    result.wgmma_rs_latency =
        parse_table<4>(options.wgmma_rs_latency, "WGMMA RS latency");
  if (options.wgmma_ss_initiation != nullptr)
    result.wgmma_ss_initiation =
        parse_table<4>(options.wgmma_ss_initiation, "WGMMA SS initiation");
  if (options.wgmma_rs_initiation != nullptr)
    result.wgmma_rs_initiation =
        parse_table<4>(options.wgmma_rs_initiation, "WGMMA RS initiation");
  if (options.wgmma_ss_completion != nullptr)
    result.wgmma_ss_completion =
        parse_table<4>(options.wgmma_ss_completion, "WGMMA SS completion");
  if (options.wgmma_rs_completion != nullptr)
    result.wgmma_rs_completion =
        parse_table<4>(options.wgmma_rs_completion, "WGMMA RS completion");
  if (options.wgmma_int_ss_completion != nullptr)
    result.wgmma_int_ss_completion = parse_table<4>(
        options.wgmma_int_ss_completion, "integer WGMMA SS completion");
  if (options.wgmma_int_rs_completion != nullptr)
    result.wgmma_int_rs_completion = parse_table<4>(
        options.wgmma_int_rs_completion, "integer WGMMA RS completion");
  if (options.wgmma_compute_throughput != nullptr)
    result.wgmma_compute_throughput = parse_table<5>(
        options.wgmma_compute_throughput, "WGMMA compute throughput");
  result.tma_latency = parse_scalar(options.tma_latency, "TMA latency");
  result.tma_initiation =
      parse_scalar(options.tma_initiation, "TMA initiation");
  result.cp_async_latency =
      parse_scalar(options.cp_async_latency, "cp.async latency");
  result.cp_async_initiation =
      parse_scalar(options.cp_async_initiation, "cp.async initiation");
  result.cp_async_commit_latency =
      parse_scalar(options.cp_async_commit_latency, "cp.async commit latency");
  result.cp_async_commit_initiation = parse_scalar(
      options.cp_async_commit_initiation, "cp.async commit initiation");
  result.cp_async_wait_latency =
      parse_scalar(options.cp_async_wait_latency, "cp.async wait latency");
  result.cp_async_wait_initiation = parse_scalar(
      options.cp_async_wait_initiation, "cp.async wait initiation");

  validate_pipeline(result.integer_latency, result.integer_initiation,
                    "integer pipeline");
  validate_pipeline(result.fp32_latency, result.fp32_initiation,
                    "fp32 pipeline");
  if (result.sfu_initiation > result.sfu_latency)
    panic("SASS timing initiation exceeds latency for sfu pipeline");
  validate_pipeline(result.tensor_latency, result.tensor_initiation,
                    "tensor pipeline");
  validate_pipeline(result.wgmma_ss_latency, result.wgmma_ss_initiation,
                    "WGMMA SS pipeline");
  validate_pipeline(result.wgmma_rs_latency, result.wgmma_rs_initiation,
                    "WGMMA RS pipeline");
  if (result.tma_initiation > result.tma_latency)
    panic("SASS timing initiation exceeds latency for TMA pipeline");
  if (result.cp_async_initiation > result.cp_async_latency ||
      result.cp_async_commit_initiation > result.cp_async_commit_latency ||
      result.cp_async_wait_initiation > result.cp_async_wait_latency)
    panic("SASS timing initiation exceeds latency for cp.async pipeline");
  return result;
}

timing_instruction::timing_instruction(const instruction &source,
                                       const timing_profile &profile)
    : warp_inst_t(), native_opcode_(source.opcode) {
  initialize(source, profile);
}

timing_instruction::timing_instruction(const instruction &source,
                                       const core_config &config,
                                       const timing_profile &profile)
    : warp_inst_t(&config), native_opcode_(source.opcode) {
  initialize(source, profile);
}

void timing_instruction::initialize(const instruction &source,
                                    const timing_profile &profile) {
  const bool resolved_symbol_call =
      source.opcode == "CALL.REL.NOINC" && source.operands.size() == 1 &&
      std::any_of(source.attributes.begin(), source.attributes.end(),
                  [](const instruction_attribute &attribute) {
                    return attribute.name == "target-pc" &&
                           !attribute.value.empty();
                  });
  const bool register_symbol_return =
      source.opcode == "RET.REL.NODEC" && source.operands.size() == 2 &&
      source.operands[0].kind == operand_kind::kRegister;
  if (!source.decoded || (!source.operands_structured &&
                          !resolved_symbol_call && !register_symbol_return))
    panic("SASS timing projection requires decoded, "
          "structured operands at PC " +
          std::to_string(source.pc));

  pc = source.pc;
  isize = kInstructionBytes;
  native_text_.clear();
  if (!source.predicate_text.empty()) {
    native_text_ += source.predicate_text;
    native_text_ += ' ';
  }
  native_text_ += source.opcode;
  if (!source.operand_text.empty()) {
    native_text_ += ' ';
    native_text_ += source.operand_text;
  }
  classify(source, profile, *this);
  populate_registers(source, *this);
  populate_register_file_sources(source, *this);
  populate_register_file_destinations(source, *this);

  inst_t::dependency_control_t dependency;
  dependency.stall_cycles = source.control.stall;
  dependency.yield = source.control.yield_flag;
  dependency.write_barrier = source.control.write_barrier;
  dependency.read_barrier = source.control.read_barrier;
  dependency.wait_mask = source.control.wait_mask;
  dependency.reuse_mask = source.control.reuse_mask;
  set_dependency_control(dependency);
  m_decoded = true;
}

void timing_instruction::print_insn(FILE *fp) const {
  std::fprintf(fp, " [SASS %s @ pc=0x%04llx] ", native_text_.c_str(),
               static_cast<unsigned long long>(pc));
}

} // namespace sass
} // namespace flash_gpgpu_sim
