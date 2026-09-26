#include <gtest/gtest.h>

#include "gpgpu-sim/flash/mbarrier.h"

namespace {

using flash_gpgpu_sim::mbarrier_recheck_action_t;

TEST(MBarrierRetryTimingTest, CompletionWinsAtInclusiveDeadline) {
  EXPECT_EQ(flash_gpgpu_sim::mbarrier_classify_recheck(true, 100, 100),
            mbarrier_recheck_action_t::RETURN_TRUE);
}

TEST(MBarrierRetryTimingTest, IncompletePhaseTimesOutAtInclusiveDeadline) {
  EXPECT_EQ(flash_gpgpu_sim::mbarrier_classify_recheck(false, 99, 100),
            mbarrier_recheck_action_t::KEEP_SLEEPING);
  EXPECT_EQ(flash_gpgpu_sim::mbarrier_classify_recheck(false, 100, 100),
            mbarrier_recheck_action_t::RETURN_FALSE);
  EXPECT_EQ(flash_gpgpu_sim::mbarrier_classify_recheck(false, 101, 100),
            mbarrier_recheck_action_t::RETURN_FALSE);
}

TEST(MBarrierRetryTimingTest, DelaysOnlyNotifiedSuccessfulSuspendedWakeups) {
  EXPECT_TRUE(flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(true, true,
                                                                  true, 110));
  EXPECT_FALSE(flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(false, true,
                                                                   true, 110));
  EXPECT_FALSE(flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(true, false,
                                                                   true, 110));
  EXPECT_FALSE(flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(true, true,
                                                                   false, 110));
  EXPECT_FALSE(
      flash_gpgpu_sim::mbarrier_should_delay_phase_wakeup(true, true, true, 0));
}

}  // namespace
