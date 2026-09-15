#include "instruction_timing_config.h"
namespace flash_gpgpu_sim {
void instruction_timing_config::reg_options(option_parser_t opp) {
  option_parser_register(
      opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
      "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,19,25,145,32",
      "1,1,19,25,145,32");
  option_parser_register(
      opp, "-ptx_int64_add_lowering_factor", OPT_UINT32,
      &int64_add_lowering_factor,
      "Scale integer ADD/SUB latency and initiation for 64-bit PTX lowering",
      "1");
  option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR,
                         &opcode_latency_fp,
                         "Opcode latencies for single precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,30",
                         "1,1,1,1,30");
  option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR,
                         &opcode_latency_dp,
                         "Opcode latencies for double precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,335",
                         "8,8,8,8,335");
  option_parser_register(opp, "-ptx_opcode_latency_sfu", OPT_CSTR,
                         &opcode_latency_sfu,
                         "Opcode latencies for SFU instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_latency_ex2", OPT_UINT32,
                         &opcode_latency_ex2,
                         "EX2 latency; 0 inherits SFU latency", "0");
  option_parser_register(opp, "-ptx_opcode_latency_predicate", OPT_UINT32,
                         &opcode_latency_predicate,
                         "SETP/SELP latency; 0 preserves legacy one-cycle ALU",
                         "0");
  option_parser_register(opp, "-ptx_opcode_latency_int_logic", OPT_UINT32,
                         &opcode_latency_int_logic,
                         "32-bit AND/OR/XOR/NOT/SHL/SHR result latency", "1");
  option_parser_register(opp, "-ptx_opcode_initiation_int_logic", OPT_UINT32,
                         &opcode_initiation_int_logic,
                         "32-bit AND/OR/XOR/NOT/SHL/SHR initiation interval",
                         "1");
  option_parser_register(opp, "-ptx_opcode_f32_minmax_use_int", OPT_BOOL,
                         &opcode_f32_minmax_use_int,
                         "Route scalar FP32 MIN/MAX through the integer ALU; "
                         "retain FP MIN/MAX latency and initiation settings",
                         "0");
  option_parser_register(
      opp, "-ptx_opcode_latency_f32x2", OPT_UINT32, &opcode_latency_f32x2,
      "Packed f32x2 arithmetic latency; 0 inherits scalar FP", "0");
  option_parser_register(
      opp, "-ptx_opcode_initiation_f32x2", OPT_UINT32, &opcode_initiation_f32x2,
      "Packed f32x2 arithmetic initiation; 0 inherits scalar FP", "0");
  option_parser_register(opp, "-ptx_opcode_latency_cvt_f16x2_f32", OPT_UINT32,
                         &opcode_latency_cvt_f16x2_f32,
                         "Packed f32 to f16x2 conversion latency", "1");
  option_parser_register(opp, "-ptx_opcode_initiation_cvt_f16x2_f32",
                         OPT_UINT32, &opcode_initiation_cvt_f16x2_f32,
                         "Packed f32 to f16x2 conversion initiation", "1");
  option_parser_register(opp, "-ptx_opcode_latency_tensor", OPT_CSTR,
                         &opcode_latency_tensor,
                         "Opcode latencies for Tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(opp, "-ptx_opcode_latency_wgmma_ss", OPT_CSTR,
                         &opcode_latency_wgmma_ss,
                         "WGMMA tensor pipe latencies for SS operands "
                         "<m64n8,m64n16,m64n32,m64n64>. Default 4,4,4,4",
                         "4,4,4,4");
  option_parser_register(opp, "-ptx_opcode_latency_wgmma_rs", OPT_CSTR,
                         &opcode_latency_wgmma_rs,
                         "WGMMA tensor pipe latencies for RS operands "
                         "<m64n8,m64n16,m64n32,m64n64>. Default 12,12,12,12",
                         "12,12,12,12");
  option_parser_register(
      opp, "-ptx_opcode_completion_wgmma_ss", OPT_CSTR,
      &opcode_completion_wgmma_ss,
      "WGMMA overlappable completion tail latencies for SS operands "
      "<m64n8,m64n16,m64n32,m64n64>. Default 66,66,66,66",
      "66,66,66,66");
  option_parser_register(
      opp, "-ptx_opcode_completion_wgmma_rs", OPT_CSTR,
      &opcode_completion_wgmma_rs,
      "WGMMA overlappable completion tail latencies for RS operands "
      "<m64n8,m64n16,m64n32,m64n64>. Default 64,65,64,64",
      "64,65,64,64");
  option_parser_register(
      opp, "-ptx_opcode_completion_wgmma_int_ss", OPT_CSTR,
      &opcode_completion_wgmma_int_ss,
      "WGMMA overlappable completion tail latencies for int/b1 SS operands "
      "<m64n8,m64n16,m64n32,m64n64>. Default 64,64,64,64",
      "64,64,64,64");
  option_parser_register(
      opp, "-ptx_opcode_completion_wgmma_int_rs", OPT_CSTR,
      &opcode_completion_wgmma_int_rs,
      "WGMMA overlappable completion tail latencies for int/b1 RS operands "
      "<m64n8,m64n16,m64n32,m64n64>. Default 62,62,62,61",
      "62,62,62,61");
  option_parser_register(
      opp, "-ptx_opcode_compute_throughput_wgmma", OPT_CSTR,
      &opcode_compute_throughput_wgmma,
      "WGMMA non-overlappable compute throughput in work/cycle per SM "
      "<f16/bf16,tf32,fp8,int8,b1>. Default 4096,2048,8192,8192,65536",
      "4096,2048,8192,8192,65536");
  option_parser_register(opp, "-ptx_opcode_latency_tma", OPT_CSTR,
                         &opcode_latency_tma,
                         "Opcode latency for TMA (cp.async.bulk) instructions"
                         "Default 33",
                         "33");
  option_parser_register(
      opp, "-ptx_opcode_latency_cp_async", OPT_CSTR, &opcode_latency_cp_async,
      "Opcode latency for ordinary cp.async instructions. Default 7", "7");
  option_parser_register(opp, "-ptx_opcode_latency_cp_async_commit", OPT_CSTR,
                         &opcode_latency_cp_async_commit,
                         "Opcode latency for ordinary cp.async.commit_group "
                         "instructions. Default 7",
                         "7");
  option_parser_register(
      opp, "-ptx_opcode_latency_cp_async_wait", OPT_CSTR,
      &opcode_latency_cp_async_wait,
      "Opcode latency for ordinary cp.async.wait_group/wait_all instructions. "
      "Default 5",
      "5");
  option_parser_register(
      opp, "-ptx_opcode_latency_tensormap", OPT_CSTR, &opcode_latency_tensormap,
      "Opcode latencies for tensormap descriptor instructions "
      "<replace,cp_fenceproxy,fence_proxy_tensormap>Default 1,1,1",
      "1,1,1");
  option_parser_register(
      opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
      "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,4,4,32,4",
      "1,1,4,4,32,4");
  option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR,
                         &opcode_initiation_fp,
                         "Opcode initiation intervals for single precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,5",
                         "1,1,1,1,5");
  option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR,
                         &opcode_initiation_dp,
                         "Opcode initiation intervals for double precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,130",
                         "8,8,8,8,130");
  option_parser_register(opp, "-ptx_opcode_initiation_sfu", OPT_CSTR,
                         &opcode_initiation_sfu,
                         "Opcode initiation intervals for sfu instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_initiation_tensor", OPT_CSTR,
                         &opcode_initiation_tensor,
                         "Opcode initiation intervals for tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(opp, "-ptx_opcode_initiation_wgmma_ss", OPT_CSTR,
                         &opcode_initiation_wgmma_ss,
                         "WGMMA initiation intervals for SS operands "
                         "<m64n8,m64n16,m64n32,m64n64>. Default 4,4,4,4",
                         "4,4,4,4");
  option_parser_register(opp, "-ptx_opcode_initiation_wgmma_rs", OPT_CSTR,
                         &opcode_initiation_wgmma_rs,
                         "WGMMA initiation intervals for RS operands "
                         "<m64n8,m64n16,m64n32,m64n64>. Default 12,12,12,12",
                         "12,12,12,12");
  option_parser_register(
      opp, "-ptx_opcode_initiation_tma", OPT_CSTR, &opcode_initiation_tma,
      "Opcode initiation interval for TMA (cp.async.bulk) instructions"
      "Default 33",
      "33");
  option_parser_register(opp, "-ptx_opcode_initiation_cp_async", OPT_CSTR,
                         &opcode_initiation_cp_async,
                         "Opcode initiation interval for ordinary cp.async "
                         "instructions. Default 7",
                         "7");
  option_parser_register(
      opp, "-ptx_opcode_initiation_cp_async_commit", OPT_CSTR,
      &opcode_initiation_cp_async_commit,
      "Opcode initiation interval for ordinary cp.async.commit_group "
      "instructions. Default 7",
      "7");
  option_parser_register(
      opp, "-ptx_opcode_initiation_cp_async_wait", OPT_CSTR,
      &opcode_initiation_cp_async_wait,
      "Opcode initiation interval for ordinary cp.async.wait_group/wait_all "
      "instructions. Default 5",
      "5");
  option_parser_register(
      opp, "-ptx_opcode_initiation_tensormap", OPT_CSTR,
      &opcode_initiation_tensormap,
      "Opcode initiation intervals for tensormap descriptor instructions "
      "<replace,cp_fenceproxy,fence_proxy_tensormap>Default 1,1,1",
      "1,1,1");
  option_parser_register(opp, "-cdp_latency", OPT_CSTR, &cdp_latency_str,
                         "CDP API latency <cudaStreamCreateWithFlags, \
cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, \
cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>"
                         "Default 7200,8000,100,12000,1600",
                         "7200,8000,100,12000,1600");
}

} // namespace flash_gpgpu_sim
