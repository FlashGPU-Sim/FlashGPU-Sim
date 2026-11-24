#include <gtest/gtest.h>
#include "../../../src/gpgpu-sim/flash/tma_cluster_targets.h"

TEST(TmaClusterTargets, CompletionMaskValidatesPairsAndCountsDuplicateOwners) {
  uint32_t owners = 0;
  EXPECT_TRUE(
      flash_gpgpu_sim::tma_completion_targets(4, 15, 1, true, true, owners));
  EXPECT_EQ(owners, 0xa000au);
  EXPECT_TRUE(
      flash_gpgpu_sim::tma_completion_targets(4, 15, 0, true, true, owners));
  EXPECT_EQ(owners, 0x50005u);
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(4, 1, 2, true, false, owners));
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(4, 1, 1, false, false, owners));
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(3, 4, 1, true, true, owners));
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(2, 4, 0, false, true, owners));
}

TEST(TmaClusterTargets, UnicastDestinationAndPeerCompletion) {
  uint32_t owners = 0;
  for (unsigned rank = 0; rank < 16; ++rank) {
    ASSERT_TRUE(flash_gpgpu_sim::tma_completion_targets(
        16, 1u << rank, rank ^ 1, true, false, owners));
    EXPECT_EQ(owners, 1u << (rank ^ 1));
    ASSERT_TRUE(flash_gpgpu_sim::tma_completion_targets(16, 1u << rank, rank,
                                                        false, false, owners));
    EXPECT_EQ(owners, 1u << rank);
  }
}
TEST(TmaClusterTargets, RejectsInvalidClusterAndOwner) {
  uint32_t owners = 0;
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(0, 0, 0, false, false, owners));
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(17, 0, 0, false, false, owners));
  EXPECT_FALSE(
      flash_gpgpu_sim::tma_completion_targets(2, 1, 2, false, false, owners));
  EXPECT_TRUE(
      flash_gpgpu_sim::tma_completion_targets(2, 0, 0, true, true, owners));
  EXPECT_EQ(owners, 0u);
}
