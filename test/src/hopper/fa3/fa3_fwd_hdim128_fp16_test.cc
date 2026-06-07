#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "fa3_fwd_hdim128_fp16_case.cuh"

namespace fa3_hopper_test {

class Fa3FwdHdim128Fp16IntegrationTest : public ::testing::Test {};

TEST_F(Fa3FwdHdim128Fp16IntegrationTest, FixedForwardCase) {
  Fa3RunResult result = run_fa3_fwd_hdim128_fp16();

  ASSERT_EQ(result.error, cudaSuccess)
      << result.where << " failed: " << cudaGetErrorString(result.error);
}

}  // namespace fa3_hopper_test
