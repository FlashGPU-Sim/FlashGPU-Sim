#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "gpgpu-sim/flash/tma_issue_timing.h"

namespace {

using flash_gpgpu_sim::tma_issue_timing_t;

TEST(TmaIssueTimingTest, HoldsIssuingWarpUntilInclusiveReadyCycle) {
  tma_issue_timing_t timing;
  timing.begin(100, 168);

  EXPECT_FALSE(timing.ready(267));
  EXPECT_TRUE(timing.ready(268));
}

TEST(TmaIssueTimingTest, ZeroLatencyPreservesLegacyTiming) {
  tma_issue_timing_t timing;
  timing.begin(100, 0);
  EXPECT_TRUE(timing.ready(100));
}

TEST(TmaIssueTimingTest, ResetClearsOutstandingDelay) {
  tma_issue_timing_t timing;
  timing.begin(100, 168);
  timing.reset();
  EXPECT_TRUE(timing.ready(0));
}

TEST(TmaIssueTimingTest, CycleAdditionSaturates) {
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  tma_issue_timing_t timing;
  timing.begin(maximum - 1, 168);
  EXPECT_FALSE(timing.ready(maximum - 1));
  EXPECT_TRUE(timing.ready(maximum));
}

}  // namespace
