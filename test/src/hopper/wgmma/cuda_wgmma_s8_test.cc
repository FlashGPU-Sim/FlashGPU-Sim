// CUDA integration tests for Hopper 8-bit integer WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaInt8M64N8K32IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaInt8M64N8K32IntegrationTest, S8S8RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::S8S8>(
      /*seed=*/501, /*scale_d=*/1, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

TEST_F(WgmmaInt8M64N8K32IntegrationTest, U8U8RandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::U8U8>(
      /*seed=*/502, /*scale_d=*/0, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

TEST_F(WgmmaInt8M64N8K32IntegrationTest, S8U8RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::S8U8>(
      /*seed=*/503, /*scale_d=*/1, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

TEST_F(WgmmaInt8M64N8K32IntegrationTest, U8S8RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::U8S8>(
      /*seed=*/504, /*scale_d=*/1, /*min_value=*/0.0f,
      /*max_value=*/0.0f, /*absolute_tolerance=*/0.0f,
      /*relative_tolerance=*/0.0f);
}

}  // namespace wgmma_test
