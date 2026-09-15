#include "runtime_adapter.h"
#include "../../panic.h"

#include "../runtime/launch_geometry.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../../../../cuda-sim/memory.h"
#include "../../../../kernel_info.h"
#include "../../../../option_parser.h"
#include "../decode/sassir_decoder.h"
#include "../functional/functional.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace {

constexpr uint64_t kSm90MinimumLaunchHeader = 0x210;
constexpr uint64_t kSm120MinimumLaunchHeader = 0x380;
constexpr uint64_t kSm90IndirectParameterBase = 0x0000010000000000ull;
constexpr uint32_t kLocalStackPointer = 0x8000;
constexpr uint64_t kDefaultInstructionLimitPerCta = 100000000;

enum class runtime_frontend { environment, ptx, sass_functional, sass_timing };
runtime_frontend selected_frontend = runtime_frontend::environment;
const char *frontend_option = "environment";

uint64_t configured_instruction_limit() {
  const char *text = std::getenv("FLASHGPU_SASS_MAX_INSTRUCTIONS_PER_CTA");
  if (text == nullptr || *text == '\0')
    return kDefaultInstructionLimitPerCta;
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  // strtoull accepts a minus sign and wraps it into the unsigned range.
  // The launch guard requires a positive decimal count, not a signed value.
  if (*text < '0' || *text > '9' || errno != 0 || end == text || *end != '\0' ||
      value == 0 || value > std::numeric_limits<uint64_t>::max())
    panic("invalid FLASHGPU_SASS_MAX_INSTRUCTIONS_PER_CTA");
  return value;
}

template <typename T>
void store_constant(std::vector<unsigned char> &data, uint64_t offset,
                    const T &value) {
  if (offset > data.size() || sizeof(value) > data.size() - offset)
    panic("SASS launch constant is out of range");
  std::memcpy(data.data() + offset, &value, sizeof(value));
}

uint64_t minimum_launch_header(architecture arch) {
  switch (arch) {
  case architecture::kSm90:
    return kSm90MinimumLaunchHeader;
  case architecture::kSm120:
    return kSm120MinimumLaunchHeader;
  case architecture::kUnknown:
    break;
  }
  panic("unsupported SASS launch architecture");
}

class simulator_functional_memory final : public functional_memory {
public:
  simulator_functional_memory(
      kernel_info_t &launch, ::memory_space *global_memory, architecture arch,
      uint32_t parameter_base, uint32_t parameter_size,
      const std::vector<kernel_constant_bank> &constant_banks)
      : parameter_memory_(launch.get_param_memory()),
        global_memory_(global_memory), parameter_base_(parameter_base),
        indirect_parameter_size_(arch == architecture::kSm90 ? parameter_size
                                                             : 0),
        constant_header_(parameter_base),
        shared_memory_("sass_shared", 64 * 1024),
        local_memory_("sass_local", 64 * 1024) {
    if (parameter_memory_ == nullptr || global_memory_ == nullptr)
      panic("SASS launch is missing simulator memory");
    if (parameter_base_ < minimum_launch_header(arch))
      panic("invalid SASS kernel parameter base");
    for (const kernel_constant_bank &bank : constant_banks) {
      const bool inserted =
          constant_banks_.emplace(bank.index, bank.data).second;
      if (!inserted)
        panic("duplicate SASS static constant bank");
    }

    const dim3 block = launch.get_cta_dim();
    const dim3 grid = launch.get_grid_dim();
    const uint64_t descriptor_base = 0;
    if (arch == architecture::kSm90) {
      // Hopper places launch dimensions at the start of constant bank zero;
      // its descriptor table pointer and kernel parameters follow at 0x208
      // and 0x210 respectively.
      store_constant(constant_header_, 0x000, block.x);
      store_constant(constant_header_, 0x004, block.y);
      store_constant(constant_header_, 0x008, block.z);
      store_constant(constant_header_, 0x00c, grid.x);
      store_constant(constant_header_, 0x010, grid.y);
      store_constant(constant_header_, 0x014, grid.z);
      store_constant(constant_header_, 0x028, kLocalStackPointer);
      // Large Hopper by-value parameters are addressed through a runtime
      // pointer at c[0][0x198]. Present the launch parameter memory through a
      // deterministic synthetic global window so execution remains detached
      // from PTX parameter state while preserving the native SASS ABI.
      store_constant(constant_header_, 0x198, kSm90IndirectParameterBase);
      store_constant(constant_header_, 0x208, descriptor_base);
    } else {
      store_constant(constant_header_, 0x358, descriptor_base);
      store_constant(constant_header_, 0x360, block.x);
      store_constant(constant_header_, 0x364, block.y);
      store_constant(constant_header_, 0x368, block.z);
      // SM120 reserves 0x36c between the NTID and NCTAID triples. This layout
      // is part of the launch ABI consumed by compiler-generated LDCs, not PTX
      // constant-memory addressing: gridDim.{x,y,z} live at 0x370..0x378.
      store_constant(constant_header_, 0x370, grid.x);
      store_constant(constant_header_, 0x374, grid.y);
      store_constant(constant_header_, 0x378, grid.z);
      store_constant(constant_header_, 0x37c, kLocalStackPointer);
    }
  }

  bool read(sass::memory_space space, uint64_t address, void *data,
            size_t bytes) const override {
    if (data == nullptr || bytes == 0)
      return false;
    switch (space) {
    case sass::memory_space::kConstant:
      if (address < parameter_base_) {
        if (bytes > constant_header_.size() - address)
          return false;
        std::memcpy(data, constant_header_.data() + address, bytes);
      } else {
        parameter_memory_->read(address - parameter_base_, bytes, data);
      }
      return true;
    case sass::memory_space::kGlobal:
      if (is_indirect_parameter_access(address, bytes)) {
        parameter_memory_->read(address - kSm90IndirectParameterBase, bytes,
                                data);
        return true;
      }
      global_memory_->read(address, bytes, data);
      return true;
    case sass::memory_space::kShared:
      shared_memory_.read(address, bytes, data);
      return true;
    case sass::memory_space::kLocal:
      local_memory_.read(address, bytes, data);
      return true;
    }
    return false;
  }

  bool write(sass::memory_space space, uint64_t address, const void *data,
             size_t bytes) override {
    if (data == nullptr || bytes == 0)
      return false;
    switch (space) {
    case sass::memory_space::kConstant:
      return false;
    case sass::memory_space::kGlobal:
      if (is_indirect_parameter_access(address, bytes))
        return false;
      global_memory_->write(address, bytes, data, nullptr, nullptr);
      return true;
    case sass::memory_space::kShared:
      shared_memory_.write(address, bytes, data, nullptr, nullptr);
      return true;
    case sass::memory_space::kLocal:
      local_memory_.write(address, bytes, data, nullptr, nullptr);
      return true;
    }
    return false;
  }

  bool read_constant(uint32_t bank, uint64_t address, void *data,
                     size_t bytes) const override {
    if (data == nullptr || bytes == 0)
      return false;
    if (bank == 0)
      return read(sass::memory_space::kConstant, address, data, bytes);
    const auto found = constant_banks_.find(bank);
    if (found == constant_banks_.end() || address > found->second.size() ||
        bytes > found->second.size() - address)
      return false;
    std::memcpy(data, found->second.data() + address, bytes);
    return true;
  }

private:
  bool is_indirect_parameter_access(uint64_t address, size_t bytes) const {
    if (indirect_parameter_size_ == 0 || address < kSm90IndirectParameterBase)
      return false;
    const uint64_t offset = address - kSm90IndirectParameterBase;
    return offset <= indirect_parameter_size_ &&
           bytes <= indirect_parameter_size_ - offset;
  }

  ::memory_space *parameter_memory_ = nullptr;
  ::memory_space *global_memory_ = nullptr;
  uint32_t parameter_base_ = 0;
  uint32_t indirect_parameter_size_ = 0;
  std::vector<unsigned char> constant_header_;
  std::unordered_map<uint32_t, std::vector<uint8_t>> constant_banks_;
  memory_space_impl<8192> shared_memory_;
  memory_space_impl<8192> local_memory_;
};

void configure_cta(cta_executor &cta, dim3 cta_id, dim3 block_dim) {
  const uint64_t threads = uint64_t{block_dim.x} * block_dim.y * block_dim.z;
  for (unsigned warp_id = 0; warp_id < cta.warp_count(); ++warp_id) {
    warp_state &warp = cta.warp(warp_id);
    warp.active_mask = runtime_detail::active_mask_for_warp(threads, warp_id);
    execution_context &context = cta.context(warp_id);
    context.cta_id_x = cta_id.x;
    context.cta_id_y = cta_id.y;
    context.cta_id_z = cta_id.z;
    context.cga_cta_id = 0;
    context.local_memory_base = 0;
    context.local_memory_thread_stride = runtime_detail::kLocalThreadStride;
    for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
      const uint64_t linear = uint64_t{warp_id} * kWarpLanes + lane;
      context.thread_linear_id[lane] = static_cast<uint32_t>(linear);
      if (linear >= threads)
        continue;
      context.thread_idx_x[lane] = linear % block_dim.x;
      context.thread_idx_y[lane] = (linear / block_dim.x) % block_dim.y;
      context.thread_idx_z[lane] =
          linear / (uint64_t{block_dim.x} * block_dim.y);
    }
  }
}

std::string cta_location(dim3 cta) {
  std::ostringstream out;
  out << "CTA (" << cta.x << ',' << cta.y << ',' << cta.z << ')';
  return out.str();
}

bool configured(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

const char *required_environment(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    panic(std::string("automatic SASS decode requires ") + name);
  return value;
}

struct runtime_binary_registry {
  std::mutex mutex;
  std::unordered_map<unsigned, std::string> paths;
};

runtime_binary_registry &binary_registry() {
  static runtime_binary_registry registry;
  return registry;
}

std::string runtime_binary(unsigned fatbin_handle) {
  runtime_binary_registry &registry = binary_registry();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.paths.find(fatbin_handle);
    if (found != registry.paths.end())
      return found->second;
  }
  return required_environment("FLASHGPU_SASS_BINARY");
}

std::string trim_output(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                            value.back() == ' ' || value.back() == '\t'))
    value.pop_back();
  const size_t first = value.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? std::string{} : value.substr(first);
}

std::string generate_sassir_dump(const std::string &kernel_name,
                                 unsigned fatbin_handle) {
  const char *python = required_environment("FLASHGPU_SASS_PYTHON");
  const char *tool = required_environment("FLASHGPU_SASS_DECODE_TOOL");
  const std::string binary = runtime_binary(fatbin_handle);
  const char *arch = required_environment("FLASHGPU_SASS_ARCH");
  // Called under automatic_sassir_path's mutex. Dumps are for debugging,
  // never inputs to a later process; only this process reuses decoded paths.
  static unsigned dump_sequence = 0;
  const std::string dump = "_sass_" + std::to_string(getpid()) + "_" +
                           std::to_string(++dump_sequence) + ".sassir";
  const char *nvdisasm = required_environment("FLASHGPU_SASS_NVDISASM");
  const char *cuobjdump = required_environment("FLASHGPU_SASS_CUOBJDUMP");
  const std::string handle = std::to_string(fatbin_handle);

  int output_pipe[2];
  if (pipe(output_pipe) != 0)
    panic(std::string("cannot create SASS decoder pipe: ") +
          std::strerror(errno));
  const pid_t child = fork();
  if (child < 0) {
    const int saved_errno = errno;
    close(output_pipe[0]);
    close(output_pipe[1]);
    panic(std::string("cannot fork SASS decoder: ") +
          std::strerror(saved_errno));
  }
  if (child == 0) {
    close(output_pipe[0]);
    if (dup2(output_pipe[1], STDOUT_FILENO) < 0)
      _exit(126);
    close(output_pipe[1]);
    execl(python, python, tool, "--binary", binary.c_str(), "--kernel",
          kernel_name.c_str(), "--fatbin-handle", handle.c_str(), "--arch",
          arch, "--output", dump.c_str(), "--nvdisasm", nvdisasm, "--cuobjdump",
          cuobjdump, static_cast<char *>(nullptr));
    _exit(127);
  }

  close(output_pipe[1]);
  std::string output;
  char buffer[4096];
  while (true) {
    const ssize_t bytes = read(output_pipe[0], buffer, sizeof(buffer));
    if (bytes > 0) {
      output.append(buffer, static_cast<size_t>(bytes));
      continue;
    }
    if (bytes < 0 && errno == EINTR)
      continue;
    if (bytes < 0) {
      const int saved_errno = errno;
      close(output_pipe[0]);
      int ignored = 0;
      waitpid(child, &ignored, 0);
      panic(std::string("cannot read SASS decoder output: ") +
            std::strerror(saved_errno));
    }
    break;
  }
  close(output_pipe[0]);

  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR)
      panic(std::string("cannot wait for SASS decoder: ") +
            std::strerror(errno));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    panic("official SASS decoder failed");
  output = trim_output(std::move(output));
  if (output.empty() || output.find_first_of("\r\n") != std::string::npos)
    panic("official SASS decoder returned an "
          "invalid sassir path");
  return output;
}

std::string automatic_sassir_path(const std::string &kernel_name,
                                  unsigned fatbin_handle) {
  if (kernel_name.empty())
    panic("automatic SASS decode requires a kernel name");
  if (fatbin_handle == 0)
    panic("automatic SASS decode requires a fatbin handle");
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::string> sassirs;
  std::lock_guard<std::mutex> lock(cache_mutex);
  const std::string key = std::to_string(fatbin_handle) + ":" + kernel_name;
  const auto found = sassirs.find(key);
  if (found != sassirs.end())
    return found->second;
  const std::string path = generate_sassir_dump(kernel_name, fatbin_handle);
  sassirs.emplace(key, path);
  printf("FlashGPU-Sim SASS: dumped official decode for '%s' at %s.\n",
         kernel_name.c_str(), path.c_str());
  return path;
}

} // namespace

void register_runtime_options(OptionParser *parser) {
  option_parser_register(
      parser, "-gpgpu_execution_frontend", OPT_CSTR, &frontend_option,
      "Execution frontend: environment, ptx, sass-functional, sass-timing; "
      "explicit selection overrides SASS mode environment variables",
      "environment");
}

void finalize_runtime_options() {
  if (std::strcmp(frontend_option, "environment") == 0)
    selected_frontend = runtime_frontend::environment;
  else if (std::strcmp(frontend_option, "ptx") == 0)
    selected_frontend = runtime_frontend::ptx;
  else if (std::strcmp(frontend_option, "sass-functional") == 0)
    selected_frontend = runtime_frontend::sass_functional;
  else if (std::strcmp(frontend_option, "sass-timing") == 0)
    selected_frontend = runtime_frontend::sass_timing;
  else
    panic(std::string("invalid -gpgpu_execution_frontend: ") + frontend_option);
}

bool runtime_functional_requested() {
  if (selected_frontend != runtime_frontend::environment)
    return selected_frontend != runtime_frontend::ptx;
  return configured("FLASHGPU_SASS_IR") || configured("FLASHGPU_SASS_AUTO");
}

bool runtime_timing_requested() {
  if (selected_frontend != runtime_frontend::environment)
    return selected_frontend == runtime_frontend::sass_timing;
  return runtime_functional_requested() && configured("FLASHGPU_SASS_TIMING");
}

void register_runtime_binary(unsigned fatbin_handle,
                             const std::string &binary_path) {
  if (fatbin_handle == 0 || binary_path.empty())
    panic("invalid SASS runtime binary registration");
  // Decode is deferred until launch, which may happen after the application
  // changes its working directory. Preserve the file selected by cuModuleLoad.
  char *resolved = realpath(binary_path.c_str(), nullptr);
  if (resolved == nullptr)
    panic("cannot resolve SASS runtime binary " + binary_path + ": " +
          std::strerror(errno));
  const std::string path(resolved);
  std::free(resolved);
  runtime_binary_registry &registry = binary_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.paths.find(fatbin_handle);
  if (found != registry.paths.end() && found->second != path)
    panic("conflicting SASS runtime binary registration");
  registry.paths[fatbin_handle] = path;
}

std::string runtime_sassir_path(const std::string &kernel_name,
                                unsigned fatbin_handle) {
  const char *sassir = std::getenv("FLASHGPU_SASS_IR");
  if (sassir != nullptr && *sassir != '\0')
    return sassir;
  if (!runtime_functional_requested())
    panic("SASS functional mode is not configured");
  return automatic_sassir_path(kernel_name, fatbin_handle);
}

runtime_kernel_abi load_runtime_kernel_abi(const std::string &sassir_path,
                                           const std::string &kernel_name) {
  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(sassir_path, kernel_name);
  if (!image.has_parameter_bank ||
      image.parameter_base < minimum_launch_header(image.arch))
    panic("SASS sassir has no CUDA kernel parameter ABI");
  runtime_kernel_abi result;
  result.parameter_base = image.parameter_base;
  result.parameter_size = image.parameter_size;
  result.parameters.reserve(image.parameters.size());
  for (const kernel_parameter &parameter : image.parameters)
    result.parameters.push_back(
        {parameter.ordinal, parameter.offset, parameter.size});
  return result;
}

runtime_kernel_resources
load_runtime_kernel_resources(const std::string &sassir_path,
                              const std::string &kernel_name) {
  sassir_decoder decoder;
  const kernel image = decoder.decode_kernel(sassir_path, kernel_name);
  if (!image.has_resources)
    panic("SASS sassir has no kernel resource metadata");
  return {image.resources.registers, image.resources.static_shared,
          image.resources.local_memory, image.resources.stack_size};
}

std::unique_ptr<functional_memory> make_runtime_functional_memory(
    kernel_info_t &launch, ::memory_space *global_memory, architecture arch,
    uint32_t parameter_base, uint32_t parameter_size,
    const std::vector<kernel_constant_bank> &constant_banks) {
  return std::make_unique<simulator_functional_memory>(
      launch, global_memory, arch, parameter_base, parameter_size,
      constant_banks);
}

functional_grid_summary
execute_runtime_functional_grid(kernel_info_t &launch,
                                ::memory_space *global_memory,
                                const std::string &sassir_path) {
  const uint64_t threads = launch.threads_per_cta();
  if (threads == 0 || threads > kMaxCtaWarps * kWarpLanes)
    panic("unsupported SASS CTA thread count");
  const unsigned warp_count =
      static_cast<unsigned>((threads + kWarpLanes - 1) / kWarpLanes);
  const uint64_t instruction_limit = configured_instruction_limit();

  sassir_decoder decoder;
  frontend sass_frontend;
  kernel decoded_kernel = decoder.decode_kernel(sassir_path, launch.name());
  const std::string kernel_name = decoded_kernel.name;
  sass_frontend.add_kernel(std::move(decoded_kernel));
  const kernel *image = sass_frontend.find_kernel(kernel_name);
  if (image == nullptr || !image->has_parameter_bank ||
      image->parameter_base < minimum_launch_header(image->arch))
    panic("SASS runtime kernel has no parameter bank");
  if (image->arch == architecture::kSm90)
    register_sm90_functional_semantics(sass_frontend);
  else
    register_sm120_functional_semantics(sass_frontend);

  functional_grid_summary summary;
  while (!launch.no_more_ctas_to_run()) {
    const dim3 cta_id = launch.get_next_cta_id();
    simulator_functional_memory memory(
        launch, global_memory, image->arch, image->parameter_base,
        image->parameter_size, image->constant_banks);
    cta_executor cta(sass_frontend, kernel_name, warp_count, &memory);
    configure_cta(cta, cta_id, launch.get_cta_dim());

    step_result result;
    do {
      result = cta.step();
      if (cta.instructions_executed() > instruction_limit) {
        std::ostringstream detail;
        detail << cta_location(cta_id)
               << " exceeded the SASS instruction limit; warp state:";
        for (unsigned warp_id = 0; warp_id < cta.warp_count(); ++warp_id) {
          const warp_state &warp = cta.warp(warp_id);
          detail << " w" << warp_id << "=pc:0x" << std::hex << warp.pc
                 << "/mask:0x" << warp.active_mask << std::dec;
          if (warp.waiting_at_cta_barrier)
            detail << "/bar:" << warp.cta_barrier_id;
          if (warp.active_mask != 0) {
            const unsigned lane =
                static_cast<unsigned>(__builtin_ctz(warp.active_mask));
            detail << "/lane:" << lane << "/r0-17:";
            for (unsigned reg = 0; reg < 18; ++reg)
              detail << (reg == 0 ? "" : ",") << "0x" << std::hex
                     << warp.read_register(lane, reg) << std::dec;
            for (unsigned reg = 0; reg < 18; ++reg) {
              const uint64_t candidate = warp.read_register(lane, reg);
              if (!cta.cta_state().mbarrier_initialized(candidate))
                continue;
              detail << "/r" << reg << "-mbar:phase="
                     << cta.cta_state().mbarrier_phase(candidate)
                     << ",arrivals="
                     << cta.cta_state().mbarrier_pending_arrivals(candidate)
                     << ",bytes="
                     << cta.cta_state().mbarrier_pending_transaction_bytes(
                            candidate);
            }
          }
        }
        panic(detail.str());
      }
    } while (result.status == step_status::kAdvanced);

    if (result.status != step_status::kExited) {
      std::ostringstream error;
      error << cta_location(cta_id) << " failed after "
            << cta.instructions_executed()
            << " SASS instructions: " << result.detail;
      panic(error.str());
    }
    ++summary.ctas_executed;
    summary.warps_executed += warp_count;
    summary.instructions_executed += cta.instructions_executed();
    launch.increment_cta_id();
  }
  return summary;
}

} // namespace sass
} // namespace flash_gpgpu_sim
