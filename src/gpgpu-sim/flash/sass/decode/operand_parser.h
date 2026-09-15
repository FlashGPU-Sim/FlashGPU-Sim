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

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_OPERAND_PARSER_H_
