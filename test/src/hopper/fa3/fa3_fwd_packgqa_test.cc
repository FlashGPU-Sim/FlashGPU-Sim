#define FA3_STANDARD_FORWARD_TU

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "fa3_fwd_packgqa_case.cuh"

namespace fa3_hopper_test {

class Fa3FwdPackGqaFp16IntegrationTest : public ::testing::Test {};

TEST_F(Fa3FwdPackGqaFp16IntegrationTest, Smoke) {
#if defined(FLASH_FWD_PACKGQA_CPASYNC_NOINC)
  SCOPED_TRACE("cp.async.mbarrier.arrive.noinc");
#else
  SCOPED_TRACE("cp.async.mbarrier.arrive");
#endif

  const Fa3PackGqaRunResult result =
      run_fa3_fwd_packgqa_hdim128_fp16();
  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
  ASSERT_TRUE(result.reference_checked);

  constexpr float kOutputAbsTolerance = 5.0e-2f;
  constexpr float kLseAbsTolerance = 5.0e-2f;
  EXPECT_LE(result.max_output_abs_error, kOutputAbsTolerance)
      << "max output abs error at linear index "
      << result.max_output_abs_error_index << ", output0=" << result.output0
      << ", output0_ref=" << result.output0_ref;
  EXPECT_LE(result.max_lse_abs_error, kLseAbsTolerance)
      << "max LSE abs error at linear index "
      << result.max_lse_abs_error_index << ", lse0=" << result.lse0
      << ", lse0_ref=" << result.lse0_ref;
}

}  // namespace fa3_hopper_test
