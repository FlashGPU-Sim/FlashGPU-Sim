#include "gtest/gtest.h"

#include <limits>

#include "gpgpu-sim/flash/shared_store_timing.h"

namespace {

TEST(SharedIssueTiming, HoldsOnlyUntilConfiguredRecurrence) {
  flash_gpgpu_sim::shared_store_issue_timing_t timing;

  EXPECT_TRUE(timing.ready(0));
  timing.begin(100, 4);
  EXPECT_FALSE(timing.ready(100));
  EXPECT_FALSE(timing.ready(103));
  EXPECT_TRUE(timing.ready(104));
}

TEST(SharedIssueTiming, LoadAndStoreRecurrencesAreIndependent) {
  flash_gpgpu_sim::shared_load_issue_timing_t load;
  flash_gpgpu_sim::shared_store_issue_timing_t store;

  load.begin(100, 4);
  EXPECT_FALSE(load.ready(103));
  EXPECT_TRUE(store.ready(103));
  EXPECT_TRUE(load.ready(104));
}

TEST(IssueIntervalTiming, ZeroIntervalPreservesLegacyReadiness) {
  flash_gpgpu_sim::issue_interval_timing_t timing;

  timing.begin(100, 0);
  EXPECT_TRUE(timing.ready(100));
}

TEST(IssueIntervalTiming, SaturatesWithoutWrapping) {
  flash_gpgpu_sim::issue_interval_timing_t timing;
  const unsigned long long maximum =
      std::numeric_limits<unsigned long long>::max();

  timing.begin(maximum - 1, 4);
  EXPECT_FALSE(timing.ready(maximum - 1));
  EXPECT_TRUE(timing.ready(maximum));
}

TEST(IssueServiceQueueTiming, AllowsBurstThenAppliesBackendBackpressure) {
  flash_gpgpu_sim::issue_service_queue_timing_t timing;

  for (unsigned i = 0; i < 5; ++i) {
    EXPECT_TRUE(timing.ready(100, 2, 4));
    timing.begin(100, 2, 4);
  }
  EXPECT_EQ(timing.pending(), 4u);
  EXPECT_FALSE(timing.ready(100, 2, 4));
  EXPECT_FALSE(timing.ready(101, 2, 4));
  EXPECT_TRUE(timing.ready(102, 2, 4));
  EXPECT_EQ(timing.pending(), 3u);
}

TEST(IssueServiceQueueTiming, ReturnsAssignedBackendServiceCycle) {
  flash_gpgpu_sim::issue_service_queue_timing_t timing;

  EXPECT_EQ(timing.begin(100, 2, 4), 100u);
  EXPECT_EQ(timing.begin(100, 2, 4), 102u);
  EXPECT_EQ(timing.begin(101, 2, 4), 104u);
}

TEST(IssueServiceQueueTiming, ZeroDepthPreservesStrictIssueGate) {
  flash_gpgpu_sim::issue_service_queue_timing_t timing;

  timing.begin(200, 2, 0);
  EXPECT_FALSE(timing.ready(201, 2, 0));
  EXPECT_TRUE(timing.ready(202, 2, 0));
}

}  // namespace
