// CUDA integration tests for Hopper TF32 WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaTf32M64N8K8IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaTf32M64N8K8IntegrationTest, RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::TF32>(
      /*seed=*/301, /*scale_d=*/1, /*min_value=*/-3.0f,
      /*max_value=*/3.0f, /*absolute_tolerance=*/0.25f,
      /*relative_tolerance=*/0.03f);
}

TEST_F(WgmmaTf32M64N8K8IntegrationTest, RandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::TF32>(
      /*seed=*/302, /*scale_d=*/0, /*min_value=*/-3.0f,
      /*max_value=*/3.0f, /*absolute_tolerance=*/0.25f,
      /*relative_tolerance=*/0.03f);
}

}  // namespace wgmma_test
