#include <gtest/gtest.h>

#include <cstdlib>

#include "../../../src/gpgpu-sim/flash/cluster_hang_prevent.h"

using namespace flash_gpgpu_sim;

TEST(ClusterHangPrevent, UniquePcCount) {
  unsigned long long hist[8] = {0x100, 0x108, 0x110, 0x100, 0x108, 0x110};
  EXPECT_EQ(unique_pc_count(hist, 6), 3u);
  unsigned long long one[4] = {0x200, 0x200, 0x200, 0x200};
  EXPECT_EQ(unique_pc_count(one, 4), 1u);
}

TEST(ClusterHangPrevent, SpinWatchdogExemptsRecognizedWait) {
  EXPECT_FALSE(spin_watchdog_should_trip(true, 8, /*at_wait=*/true,
                                         /*interest=*/false, /*peer=*/true,
                                         /*unique=*/1, /*watch=*/100));
  EXPECT_FALSE(spin_watchdog_should_trip(true, 8, false, /*interest=*/true,
                                         true, 1, 100));
  EXPECT_FALSE(spin_watchdog_should_trip(true, 8, false, false, /*peer=*/false,
                                         1, 100));
}

TEST(ClusterHangPrevent, SpinWatchdogTripsTightLoopAfterPeerAccess) {
  EXPECT_FALSE(spin_watchdog_should_trip(true, 8, false, false, true, 3, 7));
  EXPECT_TRUE(spin_watchdog_should_trip(true, 8, false, false, true, 3, 8));
  EXPECT_FALSE(spin_watchdog_should_trip(true, 8, false, false, true,
                                         /*unique=*/7, 100));
  EXPECT_FALSE(spin_watchdog_should_trip(false, 8, false, false, true, 1, 100));
  EXPECT_FALSE(spin_watchdog_should_trip(true, 0, false, false, true, 1, 100));
}

TEST(ClusterHangPrevent, MixedBarTryWaitNeedsBothAndDwell) {
  EXPECT_FALSE(mixed_bar_trywait_should_trip(true, 8, true, true, 7));
  EXPECT_TRUE(mixed_bar_trywait_should_trip(true, 8, true, true, 8));
  EXPECT_FALSE(mixed_bar_trywait_should_trip(true, 8, /*partial=*/false, true,
                                             100));
  EXPECT_FALSE(mixed_bar_trywait_should_trip(true, 8, true, /*bar=*/false, 100));
}

TEST(ClusterHangPrevent, PeerArmExpiresAfterQuietAndOnWait) {
  EXPECT_EQ(peer_arm_quiet_limit(78, 135, 43), 78u * 2 + 64u);
  EXPECT_FALSE(peer_access_still_armed(false, false, 0, 220));
  EXPECT_FALSE(peer_access_still_armed(true, /*at_wait=*/true, 0, 220));
  EXPECT_TRUE(peer_access_still_armed(true, false, 0, 220));
  EXPECT_TRUE(peer_access_still_armed(true, false, 219, 220));
  EXPECT_FALSE(peer_access_still_armed(true, false, 220, 220));
}

TEST(ClusterHangPrevent, EnvOverrideThreshold) {
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "32", 1), 0);
  EXPECT_EQ(hang_watchdog_threshold(8192), 32u);
  ASSERT_EQ(unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG"), 0);
  EXPECT_EQ(hang_watchdog_threshold(8192), 8192u);
}
