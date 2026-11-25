#include <gtest/gtest.h>
#include <string>

#include "../../../libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
void check_idle_dump(int mask) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  auto *gpu = ctx->the_gpgpusim->g_the_gpu;
  ASSERT_NE(gpu, nullptr);
  // No kernel has populated the thread slots. Exercise the real shader dump,
  // including its optional done-cycle path, rather than a formatting helper.
  testing::internal::CaptureStdout();
  gpu->dump_pipeline(mask, 0, 0);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("per warp SIMT control-flow state"), std::string::npos);
}
TEST(PipelineDumpEmptyThreads, SimtStateWithoutDoneCycles) {
  check_idle_dump(1 | 4);
}
TEST(PipelineDumpEmptyThreads, SimtStateWithDoneCycles) {
  check_idle_dump(1 | 4 | 8);
}
}  // namespace
