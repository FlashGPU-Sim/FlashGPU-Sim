#include <gtest/gtest.h>

#include "gpgpu-sim/flash/sass/runtime/launch_geometry.h"

namespace {
using namespace flash_gpgpu_sim::sass;

TEST(SassLaunchGeometryTest, MatchesLaneMembershipForEveryCtaSize) {
  for (unsigned threads = 0; threads <= kMaxCtaWarps * kWarpLanes; ++threads) {
    for (unsigned warp = 0; warp <= kMaxCtaWarps; ++warp) {
      uint32_t expected = 0;
      for (unsigned lane = 0; lane < kWarpLanes; ++lane) {
        if (warp * kWarpLanes + lane < threads) expected |= uint32_t{1} << lane;
      }
      ASSERT_EQ(runtime_detail::active_mask_for_warp(threads, warp), expected)
          << "threads=" << threads << " warp=" << warp;
    }
  }
}

}  // namespace
