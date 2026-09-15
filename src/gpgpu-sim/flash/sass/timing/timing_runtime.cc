#include "timing_runtime.h"
#include "../../panic.h"

#include "../runtime/launch_geometry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../../libcuda/gpgpu_context.h"
#include "../../../../kernel_info.h"
#include "../../async_proxy_timing.h"
#include "../../instruction_fetch_timing.h"
#include "../../tensormap.h"
#include "../decode/sassir_decoder.h"
#include "../functional/functional.h"
#include "../runtime/runtime_adapter.h"

namespace flash_gpgpu_sim {
namespace sass {
namespace {

bool timing_debug_enabled() {
  static const bool enabled =
      std::getenv("FLASHGPU_SASS_TIMING_DEBUG") != nullptr;
  return enabled;
}

void timing_debug(const char *message) {
  if (!timing_debug_enabled())
    return;
  std::fprintf(stderr, "FlashGPU-Sim SASS timing: %s\n", message);
  std::fflush(stderr);
}

dim3 linear_to_cta(uint64_t linear, dim3 grid) {
  if (grid.x == 0 || grid.y == 0 || grid.z == 0)
    panic("invalid SASS timing grid dimensions");
  dim3 result;
  result.x = static_cast<unsigned>(linear % grid.x);
  linear /= grid.x;
  result.y = static_cast<unsigned>(linear % grid.y);
  linear /= grid.y;
  result.z = static_cast<unsigned>(linear);
  if (result.z >= grid.z)
    panic("SASS timing CTA id is outside the grid");
  return result;
}

timing_profile profile_from_config(const core_config *config) {
  if (config == nullptr)
    panic("SASS timing runtime requires a core config");
  if (config->gpgpu_ctx == nullptr)
    panic("SASS timing runtime requires a simulator context");
  const instruction_timing_config &hardware =
      config->gpgpu_ctx->instruction_timing;
  timing_profile_options options;
  options.integer_latency = hardware.opcode_latency_int;
  options.integer_initiation = hardware.opcode_initiation_int;
  options.fp32_latency = hardware.opcode_latency_fp;
  options.fp32_initiation = hardware.opcode_initiation_fp;
  options.sfu_latency = hardware.opcode_latency_sfu;
  options.sfu_initiation = hardware.opcode_initiation_sfu;
  options.tensor_latency = hardware.opcode_latency_tensor;
  options.tensor_initiation = hardware.opcode_initiation_tensor;
  options.wgmma_ss_latency = hardware.opcode_latency_wgmma_ss;
  options.wgmma_rs_latency = hardware.opcode_latency_wgmma_rs;
  options.wgmma_ss_initiation = hardware.opcode_initiation_wgmma_ss;
  options.wgmma_rs_initiation = hardware.opcode_initiation_wgmma_rs;
  options.wgmma_ss_completion = hardware.opcode_completion_wgmma_ss;
  options.wgmma_rs_completion = hardware.opcode_completion_wgmma_rs;
  options.wgmma_int_ss_completion = hardware.opcode_completion_wgmma_int_ss;
  options.wgmma_int_rs_completion = hardware.opcode_completion_wgmma_int_rs;
  options.wgmma_compute_throughput = hardware.opcode_compute_throughput_wgmma;
  options.tma_latency = hardware.opcode_latency_tma;
  options.tma_initiation = hardware.opcode_initiation_tma;
  options.cp_async_latency = hardware.opcode_latency_cp_async;
  options.cp_async_initiation = hardware.opcode_initiation_cp_async;
  options.cp_async_commit_latency = hardware.opcode_latency_cp_async_commit;
  options.cp_async_commit_initiation =
      hardware.opcode_initiation_cp_async_commit;
  options.cp_async_wait_latency = hardware.opcode_latency_cp_async_wait;
  options.cp_async_wait_initiation = hardware.opcode_initiation_cp_async_wait;
  timing_profile profile = parse_timing_profile(options);
  const shader_core_config *shader_config =
      dynamic_cast<const shader_core_config *>(config);
  if (shader_config != nullptr) {
    profile.async_proxy_fence_extra_stall =
        shader_config->gpgpu_async_proxy_fence_extra_stall;
    profile.async_proxy_fence_dirty_extra_stall =
        shader_config->gpgpu_async_proxy_fence_dirty_extra_stall;
    profile.async_proxy_fence_initiation_stall =
        shader_config->gpgpu_async_proxy_fence_initiation_stall;
    profile.async_proxy_fence_dirty_initiation_stall =
        shader_config->gpgpu_async_proxy_fence_dirty_initiation_stall;
    for (size_t index = 0; index < shader_config->m_specialized_unit.size();
         ++index) {
      const specialized_unit_params &unit =
          shader_config->m_specialized_unit[index];
      if (std::string(unit.name) != "UNIFORM")
        continue;
      profile.uniform_unit_index = static_cast<int>(index);
      profile.uniform_latency = unit.latency;
      profile.uniform_initiation = unit.initiation;
      break;
    }
  }
  return profile;
}

class timing_kernel_image {
public:
  timing_kernel_image(const std::string &sassir_path,
                      const std::string &kernel_name, const core_config *config)
      : profile_(profile_from_config(config)) {
    const shader_core_config *shader_config =
        dynamic_cast<const shader_core_config *>(config);
    if (shader_config != nullptr) {
      loop_buffer_config_.backedge_redirect_latency =
          shader_config->gpgpu_instruction_backedge_redirect_latency;
      loop_buffer_config_.capacity_bytes =
          shader_config->gpgpu_instruction_loop_buffer_bytes;
      loop_buffer_config_.refill_granularity_bytes =
          shader_config->gpgpu_instruction_loop_refill_granularity;
      loop_buffer_config_.refill_latency =
          shader_config->gpgpu_instruction_loop_refill_latency;
      loop_buffer_config_.max_refill_latency =
          shader_config->gpgpu_instruction_loop_refill_max_latency;
    }
    sassir_decoder decoder;
    kernel decoded = decoder.decode_kernel(sassir_path, kernel_name);
    frontend_.add_kernel(std::move(decoded));
    image_ = frontend_.find_kernel(kernel_name);
    if (image_ == nullptr || image_->instructions.empty())
      panic("empty SASS timing kernel image");
    if (!image_->has_parameter_bank)
      panic("SASS timing kernel has no parameter ABI");
    if (image_->arch == architecture::kSm90)
      register_sm90_functional_semantics(frontend_);
    else if (image_->arch == architecture::kSm120)
      register_sm120_functional_semantics(frontend_);
    else
      panic("unsupported SASS timing architecture");

    for (const instruction &source : image_->instructions) {
      auto projected =
          std::make_unique<timing_instruction>(source, *config, profile_);
      const bool inserted =
          instructions_.emplace(source.pc, std::move(projected)).second;
      if (!inserted)
        panic("duplicate PC in SASS timing image");
    }
  }

  const timing_instruction *fetch(uint64_t pc) const {
    const auto found = instructions_.find(pc);
    return found == instructions_.end() ? nullptr : found->second.get();
  }

  uint64_t start_pc() const { return image_->instructions.front().pc; }
  architecture arch() const { return image_->arch; }
  uint32_t parameter_base() const { return image_->parameter_base; }
  uint32_t parameter_size() const { return image_->parameter_size; }
  const std::vector<kernel_constant_bank> &constant_banks() const {
    return image_->constant_banks;
  }
  const std::string &name() const { return image_->name; }
  const timing_profile &profile() const { return profile_; }
  const frontend &functional_frontend() const { return frontend_; }
  unsigned instruction_refill_delay(uint64_t branch_pc, uint64_t target_pc,
                                    unsigned instruction_bytes) const {
    return flash_gpgpu_sim::instruction_loop_refill_delay(
        branch_pc, target_pc, instruction_bytes, loop_buffer_config_);
  }

private:
  timing_profile profile_;
  frontend frontend_;
  const kernel *image_ = nullptr;
  instruction_loop_buffer_config loop_buffer_config_;
  std::unordered_map<uint64_t, std::unique_ptr<timing_instruction>>
      instructions_;
};

std::shared_ptr<const timing_kernel_image>
load_timing_image(const std::string &sassir_path,
                  const std::string &kernel_name, const core_config *config) {
  static std::mutex mutex;
  static std::unordered_map<std::string,
                            std::weak_ptr<const timing_kernel_image>>
      images;
  std::ostringstream key;
  key << sassir_path << '\n' << kernel_name << '\n' << config;
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = images.find(key.str());
  if (found != images.end()) {
    if (std::shared_ptr<const timing_kernel_image> image = found->second.lock())
      return image;
  }
  auto image = std::make_shared<const timing_kernel_image>(sassir_path,
                                                           kernel_name, config);
  images[key.str()] = image;
  return image;
}

struct timing_cta_state {
  timing_cta_state(const timing_kernel_image &image, kernel_info_t &launch,
                   ::memory_space *global_memory, unsigned first_warp,
                   unsigned count, uint64_t linear_id, dim3 cta_id,
                   unsigned virtual_smid)
      : first_hardware_warp(first_warp), linear_cta_id(linear_id), cta(count),
        warps(count), contexts(count), exited(count, false),
        async_proxy(count,
                    {image.profile().async_proxy_fence_extra_stall,
                     image.profile().async_proxy_fence_dirty_extra_stall,
                     image.profile().async_proxy_fence_initiation_stall,
                     image.profile().async_proxy_fence_dirty_initiation_stall}),
        memory(make_runtime_functional_memory(
            launch, global_memory, image.arch(), image.parameter_base(),
            image.parameter_size(), image.constant_banks())) {
    const dim3 block = launch.get_cta_dim();
    const uint64_t threads = launch.threads_per_cta();
    for (unsigned local_warp = 0; local_warp < count; ++local_warp) {
      warp_state &warp = warps[local_warp];
      warp.pc = image.start_pc();
      warp.active_mask =
          runtime_detail::active_mask_for_warp(threads, local_warp);

      execution_context &context = contexts[local_warp];
      context.memory = memory.get();
      context.cta = &cta;
      context.warp_id = local_warp;
      context.virtual_smid = virtual_smid;
      context.cta_id_x = cta_id.x;
      context.cta_id_y = cta_id.y;
      context.cta_id_z = cta_id.z;
      context.local_memory_thread_stride = runtime_detail::kLocalThreadStride;
      for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
        const uint64_t linear = uint64_t{local_warp} * kWarpLanes + lane;
        context.thread_linear_id[lane] = static_cast<uint32_t>(linear);
        if (linear >= threads)
          continue;
        context.thread_idx_x[lane] = linear % block.x;
        context.thread_idx_y[lane] = (linear / block.x) % block.y;
        context.thread_idx_z[lane] = linear / (uint64_t{block.x} * block.y);
      }
    }
  }

  unsigned local_warp(unsigned hardware_warp) const {
    if (hardware_warp < first_hardware_warp ||
        hardware_warp - first_hardware_warp >= warps.size())
      panic("hardware warp does not belong to SASS CTA");
    return hardware_warp - first_hardware_warp;
  }

  unsigned first_hardware_warp = 0;
  uint64_t linear_cta_id = 0;
  cta_execution_state cta;
  std::vector<warp_state> warps;
  std::vector<execution_context> contexts;
  std::vector<bool> exited;
  // The functional CTA state advances a named-barrier generation as soon as
  // its final arrival executes.  Hopper exposes that completion to deferred
  // BAR.SYNC consumers only after a finite handoff delay, however.  Retain the
  // newest completed generation and its timing-visible cycle per barrier.
  std::array<uint64_t, kCtaBarrierSlots> named_barrier_completed_generation{};
  std::array<uint64_t, kCtaBarrierSlots> named_barrier_visible_cycle{};
  async_proxy_timing async_proxy;
  std::unique_ptr<functional_memory> memory;
};

memory_space_t backend_space(memory_space space) {
  switch (space) {
  case memory_space::kConstant:
    return memory_space_t(const_space);
  case memory_space::kGlobal:
    return memory_space_t(global_space);
  case memory_space::kShared:
    return memory_space_t(shared_space);
  case memory_space::kLocal:
    return memory_space_t(local_space);
  }
  panic("invalid SASS functional memory space");
}

uint32_t instruction_execution_mask(const instruction &inst,
                                    const warp_state &warp) {
  if (!inst.has_guard)
    return warp.active_mask;
  uint32_t result = 0;
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const uint32_t lane_mask = uint32_t{1} << lane;
    if ((warp.active_mask & lane_mask) == 0)
      continue;
    bool predicate = false;
    if (inst.guard.kind == operand_kind::kPredicate)
      predicate = warp.read_predicate(lane, inst.guard.index);
    else if (inst.guard.kind == operand_kind::kUniformPredicate)
      predicate = warp.read_uniform_predicate(inst.guard.index);
    else
      panic("invalid SASS instruction guard kind");
    if (predicate != inst.guard.negated)
      result |= lane_mask;
  }
  return result;
}

uint32_t backend_tensor_dtype(const functional_tensor_map_view &view) {
  switch (view.element_type) {
  case tensor_map_element_type::kUnsigned32:
    return TMA_DTYPE_U32;
  case tensor_map_element_type::kFloat16:
    return TMA_DTYPE_F16;
  case tensor_map_element_type::kFloat32:
    return TMA_DTYPE_F32;
  case tensor_map_element_type::kFloat32Ftz:
    return TMA_DTYPE_F32_FTZ;
  case tensor_map_element_type::kUnknown:
    if (view.element_bytes == 1)
      return TMA_DTYPE_U8;
    if (view.element_bytes == 2)
      return TMA_DTYPE_U16;
    if (view.element_bytes == 4)
      return TMA_DTYPE_U32;
    if (view.element_bytes == 8)
      return TMA_DTYPE_U64;
    break;
  }
  panic("unsupported SASS tensor-map element type");
}

uint32_t backend_tensor_swizzle(uint32_t bytes) {
  switch (bytes) {
  case 0:
    return TMA_SWIZZLE_NONE;
  case 32:
    return TMA_SWIZZLE_32B;
  case 64:
    return TMA_SWIZZLE_64B;
  case 128:
    return TMA_SWIZZLE_128B;
  default:
    panic("unsupported SASS tensor-map swizzle");
  }
}

inst_t::tma_dyn_info_t backend_tma_effect(const functional_tma_effect &effect) {
  if (!effect.valid || effect.size_bytes == 0 || effect.tensor_map.rank > 5)
    panic("invalid SASS functional TMA effect");
  inst_t::tma_dyn_info_t result;
  result.dst_addr = effect.destination_address;
  result.src_addr = effect.source_address;
  result.size_in_bytes = effect.size_bytes;
  result.mbar_addr = effect.mbarrier_address;
  std::copy(effect.coordinates.begin(), effect.coordinates.end(),
            result.coords);

  // Hopper UBLKCP/UBLKRED carry linear shared/global addresses directly and
  // therefore have no tensor-map descriptor to reconstruct.  The ordinary
  // UTMA instructions below still require and publish one.
  if (effect.tensor_map.rank == 0)
    return result;

  tensormap_descriptor_t descriptor{};
  const functional_tensor_map_view &view = effect.tensor_map;
  descriptor.fields.globalAddress = view.global_address;
  descriptor.fields.tensorRank = view.rank - 1;
  for (unsigned dimension = 0; dimension < view.rank; ++dimension) {
    descriptor.fields.boxDim[dimension] = view.box_dim[dimension];
    descriptor.fields.globalDim[dimension] = view.global_dim[dimension];
    descriptor.fields.elementStrides[dimension] =
        view.element_stride[dimension];
    if (dimension != 0)
      descriptor.fields.globalStrides[dimension - 1] =
          view.global_stride_bytes[dimension];
  }
  descriptor.fields.tensorDataType = backend_tensor_dtype(view);
  descriptor.fields.interleave = TMA_INTERLEAVE_NONE;
  descriptor.fields.swizzle = backend_tensor_swizzle(view.swizzle_bytes);
  descriptor.fields.oobFill = view.oob_zero ? TMA_OOB_ZERO : TMA_OOB_NAN;
  static_assert(sizeof(result.tensormap_descriptor) ==
                    sizeof(descriptor.raw_bytes),
                "backend tensor-map descriptor size mismatch");
  std::copy_n(descriptor.raw_bytes, sizeof(descriptor.raw_bytes),
              result.tensormap_descriptor);
  result.has_tensormap_descriptor = true;
  return result;
}

} // namespace

class timing_runtime::impl {
public:
  impl(const core_config *config, unsigned shader_id)
      : config_(config), shader_id_(shader_id) {
    if (config_ == nullptr)
      panic("SASS timing runtime requires a core config");
    const shader_core_config *shader_config =
        dynamic_cast<const shader_core_config *>(config_);
    if (shader_config != nullptr)
      named_barrier_release_latency_ =
          shader_config->gpgpu_cta_barrier_release_latency;
  }

  timing_cta_state &cta_for_warp(unsigned hardware_warp) const {
    const auto found = warp_to_cta_.find(hardware_warp);
    if (found == warp_to_cta_.end())
      panic("unbound SASS timing warp");
    const auto cta = ctas_.find(found->second);
    if (cta == ctas_.end())
      panic("SASS timing warp references a released CTA");
    return *cta->second;
  }

  const core_config *config_ = nullptr;
  unsigned shader_id_ = 0;
  unsigned named_barrier_release_latency_ = 0;
  kernel_info_t *launch_ = nullptr;
  ::memory_space *global_memory_ = nullptr;
  std::shared_ptr<const timing_kernel_image> image_;
  std::unordered_map<unsigned, std::unique_ptr<timing_cta_state>> ctas_;
  std::unordered_map<unsigned, unsigned> warp_to_cta_;
};

timing_runtime::timing_runtime(const core_config *config, unsigned shader_id)
    : impl_(std::make_unique<impl>(config, shader_id)) {}

timing_runtime::~timing_runtime() = default;

void timing_runtime::bind(kernel_info_t &launch,
                          ::memory_space *global_memory) {
  if (global_memory == nullptr)
    panic("SASS timing launch has no global memory");
  if (impl_->launch_ == &launch)
    return;
  if (!impl_->ctas_.empty())
    panic("cannot replace an active SASS timing kernel");
  timing_debug("resolving kernel sassir");
  const std::string sassir =
      runtime_sassir_path(launch.name(), launch.get_sass_fatbin_handle());
  timing_debug("projecting static kernel image");
  impl_->image_ = load_timing_image(sassir, launch.name(), impl_->config_);
  timing_debug("static kernel image is ready");
  impl_->launch_ = &launch;
  impl_->global_memory_ = global_memory;
}

void timing_runtime::start_cta(unsigned hardware_cta_id,
                               unsigned first_hardware_warp,
                               unsigned warp_count, uint64_t linear_cta_id) {
  if (impl_->image_ == nullptr || impl_->launch_ == nullptr)
    panic("SASS timing runtime is not bound to a kernel");
  if (warp_count == 0 || warp_count > kMaxCtaWarps)
    panic("invalid SASS timing CTA warp count");
  if (impl_->ctas_.find(hardware_cta_id) != impl_->ctas_.end())
    panic("duplicate SASS timing hardware CTA");
  timing_debug("creating CTA architectural state");
  const dim3 cta_id =
      linear_to_cta(linear_cta_id, impl_->launch_->get_grid_dim());
  auto cta = std::make_unique<timing_cta_state>(
      *impl_->image_, *impl_->launch_, impl_->global_memory_,
      first_hardware_warp, warp_count, linear_cta_id, cta_id,
      impl_->shader_id_);
  for (unsigned local_warp = 0; local_warp < warp_count; ++local_warp) {
    const unsigned hardware_warp = first_hardware_warp + local_warp;
    if (!impl_->warp_to_cta_.emplace(hardware_warp, hardware_cta_id).second)
      panic("duplicate SASS timing hardware warp");
  }
  impl_->ctas_.emplace(hardware_cta_id, std::move(cta));
  timing_debug("CTA architectural state is ready");
}

void timing_runtime::release_cta(unsigned hardware_cta_id) {
  const auto found = impl_->ctas_.find(hardware_cta_id);
  if (found == impl_->ctas_.end())
    return;
  for (unsigned local_warp = 0; local_warp < found->second->warps.size();
       ++local_warp)
    impl_->warp_to_cta_.erase(found->second->first_hardware_warp + local_warp);
  impl_->ctas_.erase(found);
}

void timing_runtime::reset() {
  impl_->warp_to_cta_.clear();
  impl_->ctas_.clear();
  impl_->launch_ = nullptr;
  impl_->global_memory_ = nullptr;
  impl_->image_.reset();
}

const timing_instruction *timing_runtime::fetch(unsigned hardware_warp_id,
                                                uint64_t pc) const {
  (void)impl_->cta_for_warp(hardware_warp_id);
  return impl_->image_ == nullptr ? nullptr : impl_->image_->fetch(pc);
}

uint64_t timing_runtime::pc(unsigned hardware_warp_id) const {
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  return cta.warps[cta.local_warp(hardware_warp_id)].pc;
}

uint32_t timing_runtime::active_mask(unsigned hardware_warp_id) const {
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  return cta.warps[cta.local_warp(hardware_warp_id)].active_mask;
}

uint64_t timing_runtime::linear_cta_id(unsigned hardware_warp_id) const {
  return impl_->cta_for_warp(hardware_warp_id).linear_cta_id;
}

unsigned timing_runtime::local_warp_id(unsigned hardware_warp_id) const {
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  return cta.local_warp(hardware_warp_id);
}

bool timing_runtime::deferred_barrier_can_issue(unsigned hardware_warp_id,
                                                uint64_t architectural_cycle) {
  if (impl_->image_ == nullptr)
    panic("SASS timing runtime has no image");
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  warp_state &warp = cta.warps[cta.local_warp(hardware_warp_id)];
  const instruction *source = impl_->image_->functional_frontend().fetch(
      impl_->image_->name(), warp.pc);
  if (source == nullptr)
    panic("SASS timing PC has no decoded instruction");
  if (!orders_deferred_cta_barrier(*source, impl_->image_->arch()))
    return true;
  bool all_visible = true;
  for (unsigned id = 0; id < kCtaBarrierSlots; ++id) {
    if (!warp.deferred_cta_barrier_pending[id])
      continue;
    const uint64_t awaited = warp.deferred_cta_barrier_generation[id];
    const uint64_t current = cta.cta.generation(id);
    if (current == awaited) {
      all_visible = false;
      continue;
    }
    // At most one not-yet-visible generation can be outstanding: every BAR
    // orders the prior deferred BAR before the warp may enter the next one.
    // If a warp observes a still newer generation, its own completion is
    // necessarily already visible.
    if (current == awaited + 1 &&
        cta.named_barrier_completed_generation[id] == current &&
        architectural_cycle < cta.named_barrier_visible_cycle[id]) {
      all_visible = false;
      continue;
    }
    warp.deferred_cta_barrier_pending[id] = false;
  }
  if (all_visible)
    return true;

  if (std::getenv("FLASHGPU_SASS_DEFERRED_BARRIER_DEBUG") != nullptr &&
      cta.linear_cta_id == 0) {
    uint32_t pending_mask = 0;
    std::ostringstream detail;
    for (unsigned id = 0; id < kCtaBarrierSlots; ++id) {
      if (!warp.deferred_cta_barrier_pending[id])
        continue;
      pending_mask |= uint32_t{1} << id;
      detail << " id=" << id
             << " awaited=" << warp.deferred_cta_barrier_generation[id]
             << " current=" << cta.cta.generation(id);
    }
    const uint64_t report_key =
        (uint64_t{cta.local_warp(hardware_warp_id)} << 56) ^
        (uint64_t{pending_mask} << 40) ^ warp.pc;
    static std::mutex report_mutex;
    static std::unordered_map<uint64_t, bool> reported;
    std::lock_guard<std::mutex> lock(report_mutex);
    if (reported.emplace(report_key, true).second) {
      std::fprintf(stderr,
                   "FlashGPU-Sim SASS deferred barrier: cta=%llu hw_warp=%u "
                   "local_warp=%u pc=0x%llx pending=0x%x%s\n",
                   static_cast<unsigned long long>(cta.linear_cta_id),
                   hardware_warp_id, cta.local_warp(hardware_warp_id),
                   static_cast<unsigned long long>(warp.pc), pending_mask,
                   detail.str().c_str());
      std::fflush(stderr);
    }
  }
  return false;
}

mbarrier_try_wait_state
timing_runtime::complete_mbarrier_try_wait(unsigned hardware_warp_id) {
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  return cta.warps[cta.local_warp(hardware_warp_id)]
      .complete_mbarrier_try_wait();
}

timing_step_result timing_runtime::step(
    unsigned hardware_warp_id, warp_inst_t &dynamic_instruction,
    uint64_t architectural_cycle, uint64_t architectural_core_frequency_hz) {
  if (impl_->image_ == nullptr)
    panic("SASS timing runtime has no image");
  timing_cta_state &cta = impl_->cta_for_warp(hardware_warp_id);
  const unsigned local_warp = cta.local_warp(hardware_warp_id);
  if (cta.exited[local_warp])
    panic("cannot execute an exited SASS timing warp");
  warp_state &warp = cta.warps[local_warp];
  execution_context &context = cta.contexts[local_warp];
  context.architectural_cycle = architectural_cycle;
  context.architectural_core_frequency_hz = architectural_core_frequency_hz;
  context.architectural_cycle_valid = true;
  if (warp.pc != dynamic_instruction.pc)
    panic("SASS timing issue PC does not match frontend PC");

  timing_step_result result;
  result.active_before = warp.active_mask;
  const instruction *source = impl_->image_->functional_frontend().fetch(
      impl_->image_->name(), warp.pc);
  if (source == nullptr)
    panic("SASS timing issue PC has no decoded instruction");
  if (std::getenv("FLASHGPU_SASS_DEFERRED_BARRIER_DEBUG") != nullptr &&
      cta.linear_cta_id == 0 && source->opcode.rfind("SYNCS.ARRIVE", 0) == 0) {
    for (unsigned id = 0; id < kCtaBarrierSlots; ++id) {
      if (!warp.deferred_cta_barrier_pending[id])
        continue;
      std::fprintf(stderr,
                   "SASS deferred arrival: cycle=%llu warp=%u pc=0x%llx "
                   "bar=%u awaited=%llu current=%llu opcode=%s\n",
                   static_cast<unsigned long long>(architectural_cycle),
                   local_warp, static_cast<unsigned long long>(warp.pc), id,
                   static_cast<unsigned long long>(
                       warp.deferred_cta_barrier_generation[id]),
                   static_cast<unsigned long long>(cta.cta.generation(id)),
                   source->opcode.c_str());
    }
  }
  result.executed_mask = instruction_execution_mask(*source, warp);
  result.frontend_result = impl_->image_->functional_frontend().step(
      impl_->image_->name(), warp, context);
  if (context.named_barrier_effect.valid) {
    const unsigned id = context.named_barrier_effect.id;
    const uint64_t generation = cta.cta.generation(id);
    if (generation > cta.named_barrier_completed_generation[id]) {
      cta.named_barrier_completed_generation[id] = generation;
      cta.named_barrier_visible_cycle[id] =
          architectural_cycle + impl_->named_barrier_release_latency_;
    }
  }
  result.instruction_refill_delay = impl_->image_->instruction_refill_delay(
      dynamic_instruction.pc, warp.pc, dynamic_instruction.isize);
  result.active_after = warp.active_mask;
  if (source->opcode.rfind("EXIT", 0) == 0)
    result.exited_mask = result.active_before & ~result.active_after;
  dynamic_instruction.set_active(active_mask_t(result.executed_mask));

  if (result.executed_mask != 0 && source->opcode.rfind("UTMALDG.", 0) == 0)
    cta.async_proxy.record_tma_load();
  if (result.executed_mask != 0 && source->opcode == "FENCE.VIEW.ASYNC.S") {
    inst_t::dependency_control_t dependency =
        dynamic_instruction.get_dependency_control();
    const unsigned extra_stall =
        cta.async_proxy.issue_fence(local_warp, dependency.has_write_barrier());
    if (dependency.stall_cycles + extra_stall > UINT8_MAX)
      panic("SASS async proxy stall exceeds timing ABI");
    dependency.stall_cycles += extra_stall;
    dynamic_instruction.set_dependency_control(dependency);
  }

  project_mbarrier_effects(context, dynamic_instruction);
  dynamic_instruction.reset_tma_dyn_info();
  if (context.named_barrier_effect.valid) {
    dynamic_instruction.set_bar_id(context.named_barrier_effect.id);
    dynamic_instruction.set_bar_count(
        context.named_barrier_effect.participant_count);
  }
  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    if (context.tma_effects[lane].valid)
      dynamic_instruction.set_tma_dyn_info(
          lane, backend_tma_effect(context.tma_effects[lane]));
  }
  const bool generic_memory = source->opcode.rfind("LD.E", 0) == 0 ||
                              source->opcode.rfind("ST.E", 0) == 0;
  const bool shared_atomic_proxy = source->opcode == "ATOMS.ADD";
  bool generic_space_resolved = false;
  if (timing_debug_enabled()) {
    std::fprintf(stderr,
                 "FlashGPU-Sim SASS timing: warp %u issued pc=0x%llx, "
                 "next=0x%llx, before=0x%08x, after=0x%08x\n",
                 hardware_warp_id,
                 static_cast<unsigned long long>(dynamic_instruction.pc),
                 static_cast<unsigned long long>(warp.pc), result.active_before,
                 result.active_after);
    std::fflush(stderr);
  }

  if (result.frontend_result.status == step_status::kUnsupported ||
      result.frontend_result.status == step_status::kMissingPc) {
    std::ostringstream error;
    error << "SASS timing execution failed for warp " << hardware_warp_id
          << " at pc 0x" << std::hex << dynamic_instruction.pc << ": "
          << result.frontend_result.detail;
    panic(error.str());
  }
  if (result.frontend_result.status == step_status::kBlocked &&
      dynamic_instruction.op != BARRIER_OP)
    panic("SASS functional CTA barrier must be represented by timing backend");
  if (result.frontend_result.status == step_status::kExited) {
    cta.cta.retire_warp(local_warp);
    cta.exited[local_warp] = true;
  }

  for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
    const auto &accesses = context.memory_accesses[lane];
    if (accesses.empty()) {
      // A false LDGSTS copy-enable predicate zero-fills shared memory without
      // issuing a global-memory request. Keep the lane in the instruction so
      // group accounting still sees the warp-level copy, but suppress its
      // transaction in the backend coalescer.
      if (dynamic_instruction.m_is_ldgsts &&
          (result.executed_mask & (uint32_t{1} << lane)) != 0)
        dynamic_instruction.set_per_thread_memory_access_size(lane, 0);
      continue;
    }
    if (accesses.size() > MAX_ACCESSES_PER_INSN_PER_THREAD)
      panic("SASS timing projection exceeds backend per-lane memory-access "
            "capacity");
    new_addr_type backend_addresses[MAX_ACCESSES_PER_INSN_PER_THREAD]{};
    size_t access_bytes = 0;
    for (size_t access_index = 0; access_index < accesses.size();
         ++access_index) {
      const execution_context::functional_memory_access &access =
          accesses[access_index];
      const memory_space_t access_space = backend_space(access.space);
      if (generic_memory) {
        if (!generic_space_resolved) {
          dynamic_instruction.space = access_space;
          generic_space_resolved = true;
        } else if (access_space != dynamic_instruction.space) {
          panic("SASS generic instruction spans multiple memory spaces");
        }
      } else if (shared_atomic_proxy && access.space == memory_space::kShared) {
        // Architectural access stays shared. Only the response-driven timing
        // proxy uses L2, with no PTX callback or global-memory value access.
        if (dynamic_instruction.space != memory_space_t(global_space))
          panic("SASS shared atomic requires the global timing proxy");
      } else if (access_space != dynamic_instruction.space) {
        panic("SASS timing memory space disagrees with static projection");
      }
      if (access.write != dynamic_instruction.is_store())
        panic("SASS timing memory direction disagrees with static projection");
      uint64_t backend_address = access.address;
      if (shared_atomic_proxy) {
        if (access.space != memory_space::kShared ||
            access.address >= SHARED_MEM_SIZE_MAX)
          panic("invalid shared address for SASS atomic timing proxy");
        backend_address =
            shared_to_generic(context.virtual_smid, access.address);
      }
      if (access.space == memory_space::kLocal) {
        // Functional SASS memory uses a disjoint region per thread.  The
        // timing backend expects the architectural per-thread local offset and
        // applies its own GPU-wide local-memory mapping before coalescing.
        if (!context.decode_local_memory_offset(lane, access.address,
                                                backend_address))
          panic(
              "SASS timing local-memory address is outside its thread region");
      }
      if (access_index == 0)
        access_bytes = access.bytes;
      else if (access.bytes != access_bytes)
        panic("SASS timing instruction has unequal per-phase access widths");
      backend_addresses[access_index] = backend_address;
    }
    dynamic_instruction.set_addr(lane, backend_addresses, accesses.size());
    if (access_bytes != dynamic_instruction.data_size)
      dynamic_instruction.set_per_thread_memory_access_size(lane, access_bytes);
  }
  // The backend memory statistics switch has no generic-space case even when
  // every lane is predicated off.  No address exists to resolve in that case,
  // so use a harmless concrete space; an executing generic instruction above
  // must always resolve from its recorded functional access.
  if (generic_memory && !generic_space_resolved)
    dynamic_instruction.space = memory_space_t(global_space);
  return result;
}

} // namespace sass
} // namespace flash_gpgpu_sim
