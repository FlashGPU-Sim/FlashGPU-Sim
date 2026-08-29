#include <gtest/gtest.h>

#include "gpgpu-sim/flash/ptx_sched/ptx_scheduler.h"

namespace {

using flash_gpgpu_sim::detail::ptx_schedule_priority_precedes;
using flash_gpgpu_sim::detail::ptx_schedule_priority_t;

TEST(PtxSchedulerPriorityTest, EarlierIssueWinsOverLongerRemainingPath) {
  const ptx_schedule_priority_t ready_now = {12, 8, 20};
  const ptx_schedule_priority_t stalled_critical_path = {28, 200, 4};

  EXPECT_TRUE(
      ptx_schedule_priority_precedes(ready_now, stalled_critical_path));
  EXPECT_FALSE(
      ptx_schedule_priority_precedes(stalled_critical_path, ready_now));
}

TEST(PtxSchedulerPriorityTest, LongerRemainingPathBreaksIssueTimeTie) {
  const ptx_schedule_priority_t longer_path = {12, 40, 20};
  const ptx_schedule_priority_t shorter_path = {12, 8, 4};

  EXPECT_TRUE(ptx_schedule_priority_precedes(longer_path, shorter_path));
  EXPECT_FALSE(ptx_schedule_priority_precedes(shorter_path, longer_path));
}

TEST(PtxSchedulerPriorityTest, SourceOrderBreaksEquivalentTie) {
  const ptx_schedule_priority_t earlier_source = {12, 40, 4};
  const ptx_schedule_priority_t later_source = {12, 40, 20};

  EXPECT_TRUE(ptx_schedule_priority_precedes(earlier_source, later_source));
  EXPECT_FALSE(ptx_schedule_priority_precedes(later_source, earlier_source));
}

}  // namespace
