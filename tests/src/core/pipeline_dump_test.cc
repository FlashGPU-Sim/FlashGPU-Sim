#include <gtest/gtest.h>

#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"

namespace {
// Run a small real GPU in a subprocess so the interconnect globals and parser
// state cannot affect other host unit tests. No kernel or CUDA device is needed.
void CheckDump(unsigned sms_per_cluster, int selected) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.func_sim->ptx_opcocde_latency_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const std::string sms = std::to_string(sms_per_cluster);
  const char *argv[] = {"dump-test", "-gpgpu_n_clusters", "2",
                        "-gpgpu_n_cores_per_cluster", sms.c_str(),
                        "-network_mode", "2"};
  option_parser_cmdline(options, 7, argv);
  config.init();
  exec_gpgpu_sim gpu(config, &ctx);
  gpu.init();
  testing::internal::CaptureStdout();
  gpu.dump_pipeline(/*shader state only=*/1, selected, -1);
  const auto output = testing::internal::GetCapturedStdout();
  std::set<unsigned> seen;
  unsigned occurrences = 0;
  std::istringstream lines(output);
  for (std::string line; std::getline(lines, line);) {
    unsigned sid;
    if (sscanf(line.c_str(), "shader %u at cycle", &sid) == 1) {
      seen.insert(sid);
      ++occurrences;
    }
  }
  std::set<unsigned> expected;
  if (selected >= 0) expected.insert(selected);
  else for (unsigned sid = 0; sid < 2 * sms_per_cluster; ++sid)
    expected.insert(sid);
  if (seen != expected || occurrences != expected.size()) {
    fprintf(stderr, "pipeline dump: expected %zu unique SMs, got %zu (%u entries)\n",
            expected.size(), seen.size(), occurrences);
    std::exit(1);
  }
  std::exit(0);
}

TEST(PipelineDumpTest, IncludesAllSmsAcrossMultiSmClusters) {
  ASSERT_EXIT(CheckDump(2, -1), testing::ExitedWithCode(0), "");
}
TEST(PipelineDumpTest, SingleSmClustersKeepExistingCoverage) {
  ASSERT_EXIT(CheckDump(1, -1), testing::ExitedWithCode(0), "");
}
TEST(PipelineDumpTest, ExplicitSmSelectsOnlyRequestedShader) {
  ASSERT_EXIT(CheckDump(2, 3), testing::ExitedWithCode(0), "");
}
}  // namespace
