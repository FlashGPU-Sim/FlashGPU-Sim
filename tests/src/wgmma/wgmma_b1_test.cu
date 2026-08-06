// CUDA integration tests for Hopper single-bit WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaB1M64N8K256IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaB1M64N8K256IntegrationTest, AndPopcRandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::B1>(
      /*seed=*/601, /*scale_d=*/1, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

TEST_F(WgmmaB1M64N8K256IntegrationTest, AndPopcRandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::B1>(
      /*seed=*/602, /*scale_d=*/0, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

}  // namespace wgmma_test
