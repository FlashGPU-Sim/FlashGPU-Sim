#ifndef FLASH_GPGPU_SIM_SASS_OPERAND_PARSER_H_
#define FLASH_GPGPU_SIM_SASS_OPERAND_PARSER_H_

#include <string>
#include <vector>

#include "../ir.h"

namespace flash_gpgpu_sim {
namespace sass {

operand parse_operand(const std::string &text);
std::vector<std::string> split_operand_text(const std::string &text);
void parse_instruction_operands(instruction &inst);

// Normalize native operand spellings for both semantics and timing consumers.
// The original instruction text and guard remain unchanged.
bool normalize_ldg_operands(instruction &inst);
bool normalize_stg_operands(instruction &inst);
bool normalize_r2p_operands(instruction &inst, unsigned &byte_selector);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_OPERAND_PARSER_H_
