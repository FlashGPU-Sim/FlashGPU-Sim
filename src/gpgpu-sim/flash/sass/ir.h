#ifndef FLASH_GPGPU_SIM_SASS_IR_H_
#define FLASH_GPGPU_SIM_SASS_IR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace flash_gpgpu_sim {
namespace sass {

constexpr uint32_t kInstructionBytes = 16;
constexpr uint8_t kNoScoreboardBarrier = 7;

enum class architecture { kUnknown, kSm90, kSm120 };

enum class operand_kind {
  kOpaque,
  kRegister,
  kUniformRegister,
  kPredicate,
  kUniformPredicate,
  kPredicateRegisterFile,
  kUniformPredicateRegisterFile,
  kBarrierRegister,
  kScoreboardRegister,
  kSpecialRegister,
  kImmediate,
  kFloatImmediate,
  kConstantMemory,
  kDescriptorAddress,
  kTensorMapDescriptor,
  kGmmaDescriptor,
  kMemoryAddress,
  kRegisterSet,
};

enum class address_term_kind { kRegister, kUniformRegister, kImmediate };

enum class operand_half_selection { kNone, kLowReplicate, kHighReplicate };
enum class operand_matrix_layout { kNone, kRow, kColumn };

struct address_term {
  address_term_kind kind = address_term_kind::kImmediate;
  uint32_t index = 0;
  bool wide = false;
  bool negated = false;
  int64_t immediate = 0;
};

struct address_expression {
  std::vector<address_term> terms;
};

// Typed view of NVIDIA's canonical nvdisasm operand syntax.  text is always
// retained so newly introduced syntax remains inspectable even when kind is
// kOpaque and functional execution has to fail closed.
struct operand {
  operand_kind kind = operand_kind::kOpaque;
  std::string text;
  uint32_t index = 0;
  bool negated = false;
  bool bitwise_complement = false;
  bool absolute = false;
  bool reuse = false;
  bool wide = false;
  operand_half_selection half_selection = operand_half_selection::kNone;
  operand_matrix_layout matrix_layout = operand_matrix_layout::kNone;
  int64_t immediate = 0;
  double float_immediate = 0.0;
  uint32_t constant_bank = 0;
  bool constant_offset_static = false;
  uint64_t constant_offset = 0;
  address_expression constant_address;
  uint32_t descriptor_register = 0;
  bool descriptor_transpose_a = false;
  bool descriptor_transpose_b = false;
  uint32_t address_register = 0;
  int64_t address_offset = 0;
  bool has_prefix = false;
  operand_kind prefix_kind = operand_kind::kOpaque;
  uint32_t prefix_index = 0;
  bool prefix_wide = false;
  std::vector<address_expression> address_groups;
  std::vector<uint32_t> register_set;
};

struct raw_instruction {
  uint64_t lo = 0;
  uint64_t hi = 0;
};

// The scheduling word embedded in a native instruction. yield_flag is the
// decoded scheduling semantic; the corresponding raw control bit is inverted.
struct control_code {
  uint8_t stall = 0;
  bool yield_flag = false;
  uint8_t write_barrier = kNoScoreboardBarrier;
  uint8_t read_barrier = kNoScoreboardBarrier;
  uint8_t wait_mask = 0;
  uint8_t reuse_mask = 0;

  bool waits_on(unsigned barrier) const {
    return barrier < 6 && (wait_mask & (1u << barrier)) != 0;
  }
  bool has_write_barrier() const {
    return write_barrier != kNoScoreboardBarrier;
  }
  bool has_read_barrier() const { return read_barrier != kNoScoreboardBarrier; }
};

// Legacy CuBit bitfield retained solely so historical SASSIR v1 sassirs stay
// inspectable. SASSIR v2 functional semantics do not consume this structure.
struct decoded_field {
  std::string name;
  uint32_t shift = 0;
  uint32_t bits = 0;
  uint64_t value = 0;
  int32_t token_index = -1;
  std::string extraction;
};

struct instruction_attribute {
  std::string name;
  std::string value;
};

struct instruction {
  uint64_t pc = 0;
  raw_instruction raw;
  std::string opcode;
  std::string decoder_source;
  std::string predicate_text;
  std::string operand_text;
  std::string key;
  std::string modifier_group;
  std::vector<decoded_field> fields;
  std::vector<instruction_attribute> attributes;
  bool has_guard = false;
  operand guard;
  bool has_auxiliary_predicate = false;
  operand auxiliary_predicate;
  bool operands_structured = false;
  std::vector<operand> operands;
  control_code control;
  bool decoded = false;
};

struct kernel_parameter {
  uint32_t ordinal = 0;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct kernel_resources {
  uint32_t registers = 0;
  uint32_t static_shared = 0;
  uint32_t local_memory = 0;
  uint32_t stack_size = 0;
};

struct kernel_constant_bank {
  uint32_t index = 0;
  std::vector<uint8_t> data;
};

struct kernel {
  std::string name;
  std::string decoder_name;
  std::string decoder_version;
  std::string decoder_schema;
  architecture arch = architecture::kUnknown;
  uint32_t instruction_bytes = 0;
  bool has_parameter_bank = false;
  uint32_t parameter_base = 0;
  uint32_t parameter_size = 0;
  std::vector<kernel_parameter> parameters;
  std::vector<kernel_constant_bank> constant_banks;
  bool has_resources = false;
  kernel_resources resources;
  std::vector<instruction> instructions;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_IR_H_
