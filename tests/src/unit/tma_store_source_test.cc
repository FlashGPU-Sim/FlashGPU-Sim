#include "gtest/gtest.h"

#include "../../../src/gpgpu-sim/flash/tma_store_source.h"

namespace flash_gpgpu_sim {
namespace {

TEST(TmaStoreSourceTrackerTest, DisablesCalibratedReleaseAtZeroBandwidth) {
  tma_store_source_tracker_t tracker;

  EXPECT_FALSE(tracker.enabled());
  EXPECT_EQ(tracker.schedule(100, 8192), 0u);
}

TEST(TmaStoreSourceTrackerTest, SerializesQueuedSourceTransfers) {
  tma_store_source_tracker_t tracker(64, 0);

  EXPECT_EQ(tracker.schedule(100, 8192), 228u);
  EXPECT_EQ(tracker.schedule(132, 8192), 356u);
  EXPECT_EQ(tracker.schedule(400, 8192), 528u);
}

TEST(TmaStoreSourceTrackerTest, AppliesSetupLatencyBeforeSourceService) {
  tma_store_source_tracker_t tracker(64, 10);

  EXPECT_EQ(tracker.schedule(100, 8192), 238u);
  EXPECT_EQ(tracker.schedule(132, 8192), 366u);
}

} // namespace
} // namespace flash_gpgpu_sim
