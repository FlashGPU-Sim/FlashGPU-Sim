// CUDA integration tests for Hopper FP8 WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaFp8M64N8K32IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaFp8M64N8K32IntegrationTest, E4M3E4M3RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::FP8E4M3E4M3>(
      /*seed=*/401, /*scale_d=*/1, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/1.0f,
      /*relative_tolerance=*/0.08f);
}

TEST_F(WgmmaFp8M64N8K32IntegrationTest, E5M2E5M2RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::FP8E5M2E5M2>(
      /*seed=*/402, /*scale_d=*/1, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/1.0f,
      /*relative_tolerance=*/0.08f);
}

TEST_F(WgmmaFp8M64N8K32IntegrationTest, E4M3E5M2RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::FP8E4M3E5M2>(
      /*seed=*/403, /*scale_d=*/1, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/1.0f,
      /*relative_tolerance=*/0.08f);
}

TEST_F(WgmmaFp8M64N8K32IntegrationTest, E5M2E4M3RandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::FP8E5M2E4M3>(
      /*seed=*/404, /*scale_d=*/0, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/1.0f,
      /*relative_tolerance=*/0.08f);
}

}  // namespace wgmma_test
