#ifndef FLASH_GPGPU_SIM_SASS_WGMMA_INSTRUCTION_H_
#define FLASH_GPGPU_SIM_SASS_WGMMA_INSTRUCTION_H_

#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {

// SM90 encodes wgmma.commit_group as a non-computing GMMA sentinel rather
// than as a distinct nvdisasm mnemonic.
inline bool is_hgmma_commit_group_sentinel(const instruction &inst) {
  return inst.opcode == "HGMMA.64x8x16.F16" && !inst.has_guard &&
         inst.operands_structured && inst.operands.size() == 5 &&
         inst.operands[0].kind == operand_kind::kRegister &&
         inst.operands[0].index == kZeroRegister && !inst.operands[0].negated &&
         inst.operands[1].kind == operand_kind::kGmmaDescriptor &&
         inst.operands[1].descriptor_register == kZeroUniformRegister &&
         !inst.operands[1].descriptor_transpose_a &&
         !inst.operands[1].descriptor_transpose_b &&
         inst.operands[2].kind == operand_kind::kRegister &&
         inst.operands[2].index == kZeroRegister && !inst.operands[2].negated &&
         inst.operands[3].kind == operand_kind::kUniformPredicate &&
         inst.operands[3].index == kTruePredicate && inst.operands[3].negated &&
         inst.operands[4].kind == operand_kind::kScoreboardRegister &&
         inst.operands[4].index == 0;
}

} // namespace sass
} // namespace flash_gpgpu_sim

#endif
