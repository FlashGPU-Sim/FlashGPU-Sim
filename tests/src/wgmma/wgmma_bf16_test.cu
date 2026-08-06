// CUDA integration tests for Hopper BF16 WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaBf16M64N8K16IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaBf16M64N8K16IntegrationTest, RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::BF16>(
      /*seed=*/2026, /*scale_d=*/1, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/0.15f,
      /*relative_tolerance=*/0.03f);
}

TEST_F(WgmmaBf16M64N8K16IntegrationTest, RandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::BF16>(
      /*seed=*/2027, /*scale_d=*/0, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/0.15f,
      /*relative_tolerance=*/0.03f);
}

}  // namespace wgmma_test
