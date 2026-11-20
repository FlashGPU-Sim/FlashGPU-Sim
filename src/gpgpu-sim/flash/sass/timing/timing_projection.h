#ifndef FLASH_GPGPU_SIM_SASS_TIMING_PROJECTION_H_
#define FLASH_GPGPU_SIM_SASS_TIMING_PROJECTION_H_

#include <array>
#include <string>

#include "../../../../abstract_hardware_model.h"
#include "../frontend.h"

namespace flash_gpgpu_sim {
namespace sass {

// Latency policy is supplied by the GPU configuration rather than embedded in
// the ISA decoder.  The arrays preserve the same operation classes as the
// backend configuration instead of collapsing unrelated native instructions
// into a frontend-specific constant.
struct timing_profile {
  // integer: ADD/MAX/MUL/MAD/DIV/SHFL
  std::array<unsigned, 6> integer_latency{{1, 1, 1, 1, 1, 1}};
  std::array<unsigned, 6> integer_initiation{{1, 1, 1, 1, 1, 1}};
  // fp32: ADD/MAX/MUL/MAD/DIV
  std::array<unsigned, 5> fp32_latency{{1, 1, 1, 1, 1}};
  std::array<unsigned, 5> fp32_initiation{{1, 1, 1, 1, 1}};
  unsigned sfu_latency = 1;
  unsigned sfu_initiation = 1;
  // Optional backend execution class for warp-uniform arithmetic. The unit
  // index is resolved from the generic specialized-unit configuration.
  int uniform_unit_index = -1;
  unsigned uniform_latency = 1;
  unsigned uniform_initiation = 1;
  // tensor: m16n8k16.f16, m16n8k8.tf32, m16n8k32.int8,
  // m16n8k8.f16, m16n8k8.bf16, m16n8k4.tf32, m16n8k16.int8.
  std::array<unsigned, 7> tensor_latency{{1, 1, 1, 1, 1, 1, 1}};
  std::array<unsigned, 7> tensor_initiation{{1, 1, 1, 1, 1, 1, 1}};
  // wgmma: m64n8, m64n16, m64n32, and m64n64. Wider native shapes scale the
  // n64 entry in the same way as the PTX frontend.
  std::array<unsigned, 4> wgmma_ss_latency{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_rs_latency{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_ss_initiation{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_rs_initiation{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_ss_completion{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_rs_completion{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_int_ss_completion{{1, 1, 1, 1}};
  std::array<unsigned, 4> wgmma_int_rs_completion{{1, 1, 1, 1}};
  // f16/bf16, tf32, fp8, int8, and b1 operations per SM cycle.
  std::array<unsigned, 5> wgmma_compute_throughput{{1, 1, 1, 1, 1}};
  unsigned tma_latency = 1;
  unsigned tma_initiation = 1;
  unsigned cp_async_latency = 1;
  unsigned cp_async_initiation = 1;
  unsigned cp_async_commit_latency = 1;
  unsigned cp_async_commit_initiation = 1;
  unsigned cp_async_wait_latency = 1;
  unsigned cp_async_wait_initiation = 1;
  // Dynamic async-proxy timing. The legacy extra-stall knob is retained as
  // the clean completion-token delay; the runtime, rather than static
  // projection, decides which delay applies to each fence.
  unsigned async_proxy_fence_extra_stall = 0;
  unsigned async_proxy_fence_dirty_extra_stall = 0;
  unsigned async_proxy_fence_initiation_stall = 0;
  unsigned async_proxy_fence_dirty_initiation_stall = 0;
};

// Raw backend timing options. Keeping this small value object independent of
// ISA semantics makes parsing and validation directly unit-testable; the
// runtime reads the shared hardware instruction_timing_config.
struct timing_profile_options {
  const char *integer_latency = nullptr;
  const char *integer_initiation = nullptr;
  const char *fp32_latency = nullptr;
  const char *fp32_initiation = nullptr;
  const char *sfu_latency = nullptr;
  const char *sfu_initiation = nullptr;
  const char *tensor_latency = nullptr;
  const char *tensor_initiation = nullptr;
  const char *wgmma_ss_latency = nullptr;
  const char *wgmma_rs_latency = nullptr;
  const char *wgmma_ss_initiation = nullptr;
  const char *wgmma_rs_initiation = nullptr;
  const char *wgmma_ss_completion = nullptr;
  const char *wgmma_rs_completion = nullptr;
  const char *wgmma_int_ss_completion = nullptr;
  const char *wgmma_int_rs_completion = nullptr;
  const char *wgmma_compute_throughput = nullptr;
  const char *tma_latency = nullptr;
  const char *tma_initiation = nullptr;
  const char *cp_async_latency = nullptr;
  const char *cp_async_initiation = nullptr;
  const char *cp_async_commit_latency = nullptr;
  const char *cp_async_commit_initiation = nullptr;
  const char *cp_async_wait_latency = nullptr;
  const char *cp_async_wait_initiation = nullptr;
};

timing_profile parse_timing_profile(const timing_profile_options &options);

// Publish per-lane barrier effects; only uniform try-waits can use the
// backend's warp-wide release. Other barrier operations remain per-lane.
void project_mbarrier_effects(const execution_context &context,
                              warp_inst_t &target);

// Static backend instruction projected from one decoded native instruction.
// Functional architectural state remains owned by the SASS frontend; this
// object contains only the fields consumed by timing issue and execution.
class timing_instruction final : public warp_inst_t {
public:
  timing_instruction(const instruction &source,
                     const timing_profile &profile = timing_profile{});
  timing_instruction(const instruction &source, const core_config &config,
                     const timing_profile &profile = timing_profile{});

  const std::string &native_opcode() const { return native_opcode_; }
  const std::string &native_text() const { return native_text_; }
  void print_insn(FILE *fp) const override;

private:
  void initialize(const instruction &source, const timing_profile &profile);
  std::string native_opcode_;
  std::string native_text_;
};

} // namespace sass
} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_SASS_TIMING_PROJECTION_H_
