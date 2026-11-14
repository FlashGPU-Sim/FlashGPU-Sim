#ifndef FLASH_GPGPU_SIM_SASS_FUNCTIONAL_H_
#define FLASH_GPGPU_SIM_SASS_FUNCTIONAL_H_

#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {

// Installs the fail-closed common Hopper functional ISA subset. Architecture-
// specific instructions such as WGMMA are registered by their own module.
void register_sm90_functional_semantics(frontend &target);

// Installs the fail-closed SM120 functional ISA subset. Handlers validate
// NVIDIA's complete mnemonic and typed operand form, so newly seen modifiers
// or operand syntax cannot silently acquire approximate semantics.
void register_sm120_functional_semantics(frontend &target);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_FUNCTIONAL_H_
