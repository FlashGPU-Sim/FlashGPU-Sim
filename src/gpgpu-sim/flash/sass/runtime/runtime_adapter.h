#ifndef FLASH_GPGPU_SIM_SASS_RUNTIME_ADAPTER_H_
#define FLASH_GPGPU_SIM_SASS_RUNTIME_ADAPTER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../ir.h"

class kernel_info_t;
class memory_space;
class OptionParser;

namespace flash_gpgpu_sim {
namespace sass {

class functional_memory;
enum class architecture;

// Result of executing one CUDA grid through the execution-driven functional
// SASS frontend. This path performs no PTX instruction execution and owns no
// timing policy.
struct functional_grid_summary {
  uint64_t ctas_executed = 0;
  uint64_t warps_executed = 0;
  uint64_t instructions_executed = 0;
};

struct runtime_kernel_parameter {
  uint32_t ordinal = 0;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct runtime_kernel_abi {
  uint32_t parameter_base = 0;
  uint32_t parameter_size = 0;
  std::vector<runtime_kernel_parameter> parameters;
};

struct runtime_kernel_resources {
  uint32_t registers = 0;
  uint32_t static_shared = 0;
  uint32_t local_memory = 0;
  uint32_t stack_size = 0;
};

// A non-empty FLASHGPU_SASS_IR or FLASHGPU_SASS_AUTO selects strict SASS
// functional mode. Automatic mode resolves the exact launched kernel from the
// current executable into a local official-decoder cache. Unsupported kernels
// or instructions are fatal; callers must never silently fall back to PTX
// after this returns true.
// Register before config parsing; finalize before CUDA module registration.
// Explicit config selection overrides environment-based mode selection only.
void register_runtime_options(OptionParser *parser);
void finalize_runtime_options();
bool runtime_functional_requested();
// FLASHGPU_SASS_TIMING keeps the same strict SASS decode/functional contract
// but launches it through the execution-driven cycle backend.
bool runtime_timing_requested();
// Associate a Driver API module handle with the exact file passed to
// cuModuleLoad. Runtime API fatbins continue to use FLASHGPU_SASS_BINARY.
void register_runtime_binary(unsigned fatbin_handle,
                             const std::string &binary_path);
std::string runtime_sassir_path(const std::string &kernel_name,
                                unsigned fatbin_handle);
runtime_kernel_abi load_runtime_kernel_abi(const std::string &sassir_path,
                                           const std::string &kernel_name);
runtime_kernel_resources
load_runtime_kernel_resources(const std::string &sassir_path,
                              const std::string &kernel_name);

// Construct one CTA-private architectural memory view over simulator global
// and launch-parameter memory. Shared and local storage live in the returned
// object; timing policy remains in the normal cache/interconnect backend.
std::unique_ptr<functional_memory> make_runtime_functional_memory(
    kernel_info_t &launch, ::memory_space *global_memory, architecture arch,
    uint32_t parameter_base, uint32_t parameter_size,
    const std::vector<kernel_constant_bank> &constant_banks);

functional_grid_summary
execute_runtime_functional_grid(kernel_info_t &launch,
                                ::memory_space *global_memory,
                                const std::string &sassir_path);

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_RUNTIME_ADAPTER_H_
