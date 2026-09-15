#ifndef FLASH_GPGPU_SIM_INSTRUCTION_TIMING_CONFIG_H_
#define FLASH_GPGPU_SIM_INSTRUCTION_TIMING_CONFIG_H_

#include "../../option_parser.h"

namespace flash_gpgpu_sim {

// Target hardware timing shared by every frontend. Legacy -ptx_* option names
// are retained for config compatibility; ownership does not require cuda_sim.
struct instruction_timing_config {
  void reg_options(option_parser_t opp);
  char *opcode_latency_int = nullptr;
  char *opcode_latency_fp = nullptr;
  char *opcode_latency_dp = nullptr;
  char *opcode_latency_sfu = nullptr;
  char *opcode_latency_tensor = nullptr;
  char *opcode_latency_wgmma_ss = nullptr;
  char *opcode_latency_wgmma_rs = nullptr;
  char *opcode_completion_wgmma_ss = nullptr;
  char *opcode_completion_wgmma_rs = nullptr;
  char *opcode_completion_wgmma_int_ss = nullptr;
  char *opcode_completion_wgmma_int_rs = nullptr;
  char *opcode_compute_throughput_wgmma = nullptr;
  char *opcode_latency_tma = nullptr;
  char *opcode_latency_cp_async = nullptr;
  char *opcode_latency_cp_async_commit = nullptr;
  char *opcode_latency_cp_async_wait = nullptr;
  char *opcode_latency_tensormap = nullptr;
  char *opcode_initiation_int = nullptr;
  char *opcode_initiation_fp = nullptr;
  char *opcode_initiation_dp = nullptr;
  char *opcode_initiation_sfu = nullptr;
  char *opcode_initiation_tensor = nullptr;
  char *opcode_initiation_wgmma_ss = nullptr;
  char *opcode_initiation_wgmma_rs = nullptr;
  char *opcode_initiation_tma = nullptr;
  char *opcode_initiation_cp_async = nullptr;
  char *opcode_initiation_cp_async_commit = nullptr;
  char *opcode_initiation_cp_async_wait = nullptr;
  char *opcode_initiation_tensormap = nullptr;
  char *cdp_latency_str = nullptr;
};

} // namespace flash_gpgpu_sim
#endif
