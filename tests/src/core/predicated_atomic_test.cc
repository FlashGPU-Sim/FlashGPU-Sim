#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdint>

#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"

namespace {
class TestGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  exec_shader_core_ctx *core() {
    return static_cast<exec_shader_core_ctx *>(m_cluster[0]->get_core(0));
  }
};

void CheckAtomicLanes(uint32_t executing_lanes) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.func_sim->ptx_opcocde_latency_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const char *argv[] = {"atomic-test", "-gpgpu_n_clusters", "1",
                        "-gpgpu_n_cores_per_cluster", "1", "-network_mode", "2"};
  option_parser_cmdline(options, 7, argv);
  config.init();
  TestGpu gpu(config, &ctx);
  gpu.init();
  auto *core = gpu.core();
  active_mask_t issued;
  issued.set();
  core->get_shd_warp(0)->init(0, 0, 0, issued, 0, 0);
  warp_inst_t inst(gpu.getShaderCoreConfig());
  inst.issue(issued, 0, 0, 0, 0, 0);
  unsigned expected = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    // Match execute_warp_inst_t's call ordering. ptx_exec_inst either adds
    // the RMW callback or clears the lane's active bit for a false predicate;
    // its caller then invokes this real post-execution hook in both cases.
    if (executing_lanes & (uint32_t{1} << lane)) {
      inst.add_callback(lane, nullptr, nullptr, nullptr, true);
      ++expected;
    } else {
      inst.set_not_active(lane);
    }
    core->checkExecutionStatusAndUpdate(inst, lane, lane);
    const unsigned actual = core->get_shd_warp(0)->get_n_atomic();
    if (actual != expected) {
      fprintf(stderr, "lane %u: atomic count=%u, executed lanes=%u\n",
              lane, actual, expected);
      std::exit(1);
    }
  }
  // The response path retires only executing lanes. No phantom atomic may
  // remain to keep the warp waiting after all real responses have arrived.
  core->decrement_atomic_count(0, expected);
  if (core->get_shd_warp(0)->get_n_atomic() != 0) std::exit(2);
  std::exit(0);
}

TEST(PredicatedAtomicTest, AllLanesExecute) {
  ASSERT_EXIT(CheckAtomicLanes(UINT32_MAX), testing::ExitedWithCode(0), "");
}
TEST(PredicatedAtomicTest, NoLaneExecutes) {
  ASSERT_EXIT(CheckAtomicLanes(0), testing::ExitedWithCode(0), "");
}
TEST(PredicatedAtomicTest, EvenLanesExecute) {
  ASSERT_EXIT(CheckAtomicLanes(0x55555555u), testing::ExitedWithCode(0), "");
}
TEST(PredicatedAtomicTest, OddLanesExecute) {
  ASSERT_EXIT(CheckAtomicLanes(0xaaaaaaaau), testing::ExitedWithCode(0), "");
}
TEST(PredicatedAtomicTest, OnlyFirstLaneExecutes) {
  ASSERT_EXIT(CheckAtomicLanes(1), testing::ExitedWithCode(0), "");
}
TEST(PredicatedAtomicTest, OnlyLastLaneExecutes) {
  ASSERT_EXIT(CheckAtomicLanes(0x80000000u), testing::ExitedWithCode(0), "");
}
}  // namespace
