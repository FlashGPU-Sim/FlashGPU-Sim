// Native instruction timing regression; no GPU launch.
// g++ -std=c++17 -O1 -fno-access-control -Isrc -I/ABS/SIM/BUILD/cuda-sim \
//   -I/usr/local/cuda/include \
//   test/check_int64_lowering.cc /ABS/SIM/LIB/libcudart.so \
//   -Wl,-rpath,/ABS/SIM/LIB -o /tmp/check_int64_lowering
// LD_LIBRARY_PATH=/ABS/SIM/LIB /tmp/check_int64_lowering
#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/ptx_ir.h"
#include "ptx.tab.h"
#include "gpgpu-sim/shader.h"

#include <cstdio>

int main() {
  gpgpu_context context;
  auto options = option_parser_create();
  context.func_sim->ptx_opcocde_latency_options(options);
  bool pass = context.func_sim->int64_add_lowering_factor == 1;
  const char *args[] = {"check", "-ptx_opcode_latency_int", "4,4,4,4,21,14",
                        "-ptx_opcode_initiation_int", "1,1,1,1,4,4",
                        "-ptx_opcode_latency_tensor", "64,64,64,64,64,64,64",
                        "-ptx_opcode_initiation_tensor", "1,1,1,1,1,1,1"};
  option_parser_cmdline(options, sizeof(args) / sizeof(*args), args);
  shader_core_config config(&context);
  config.warp_size = 32;
  for (unsigned factor : {1u, 2u, 0u}) {
    context.func_sim->int64_add_lowering_factor = factor;
    for (int opcode : {ADD_OP, SUB_OP, ADDC_OP, SUBC_OP, MUL_OP}) {
      for (int type : {B64_TYPE, U64_TYPE, S64_TYPE, U32_TYPE, S32_TYPE}) {
        ptx_instruction inst(opcode, nullptr, 0, 0, nullptr, nullptr, {},
                             operand_info(&context), {}, {}, {}, {}, {type},
                             undefined_space, undefined_space, __FILE__,
                             __LINE__, "timing regression", &config, &context);
        inst.set_opcode_and_latency();
        const unsigned scale = factor == 2 && opcode != MUL_OP &&
                                       (type == B64_TYPE || type == U64_TYPE ||
                                        type == S64_TYPE) ? 2 : 1;
        const bool ok = inst.latency == 4 * scale &&
                        inst.initiation_interval == scale && inst.op == INTP_OP;
        if (!ok) std::printf("FAIL factor=%u opcode=%d type=%d latency=%u init=%u\n",
                             factor, opcode, type, inst.latency,
                             inst.initiation_interval);
        pass &= ok;
      }
    }
  }
  // A valid ADD latency need not equal MAX or SHFL. The lowering factor
  // must also size the execution pipeline, not just instruction metadata.
  context.func_sim->int64_add_lowering_factor = 2;
  config.set_pipeline_latency();
  pass &= config.max_int_latency == 14;  // Current H200 pipeline is unchanged.
  const char *wide_args[] = {"check", "-ptx_opcode_latency_int", "12,4,4,4,21,14"};
  option_parser_cmdline(options, sizeof(wide_args) / sizeof(*wide_args), wide_args);
  config.set_pipeline_latency();
  if (config.max_int_latency < 24) {
    std::printf("FAIL integer pipeline depth=%u, need at least 24\n",
                config.max_int_latency);
    pass = false;
  }
  option_parser_destroy(options);
  std::printf("int64 lowering: %s (default, 75 timing controls, pipeline bound)\n",
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
