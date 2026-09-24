// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/l2cache.h"

#include <gtest/gtest.h>

#include <deque>

namespace {

typedef outstanding_sector_coalescer<unsigned, unsigned> test_coalescer;

TEST(L2TmaCoalescingTest, OneMasterFansOutWaitersInArrivalOrder) {
  test_coalescer coalescer;
  unsigned long long waiter_count = 99;

  EXPECT_FALSE(coalescer.admit_waiter(0x100, 0, waiter_count));
  EXPECT_TRUE(coalescer.admit(0x100, 1, waiter_count));
  EXPECT_EQ(0u, waiter_count);
  EXPECT_TRUE(coalescer.admit_waiter(0x100, 2, waiter_count));
  EXPECT_EQ(1u, waiter_count);
  EXPECT_FALSE(coalescer.admit(0x100, 3, waiter_count));
  EXPECT_EQ(2u, waiter_count);

  std::deque<unsigned> waiters;
  EXPECT_TRUE(coalescer.close_master(1, waiters));
  ASSERT_EQ(2u, waiters.size());
  EXPECT_EQ(2u, waiters[0]);
  EXPECT_EQ(3u, waiters[1]);

  EXPECT_TRUE(coalescer.admit(0x100, 4, waiter_count));
  EXPECT_EQ(0u, waiter_count);
}

TEST(L2TmaCoalescingTest, InterveningWriteStartsANewReadGeneration) {
  test_coalescer coalescer;
  unsigned long long waiter_count = 0;

  EXPECT_TRUE(coalescer.admit(0x200, 10, waiter_count));
  EXPECT_FALSE(coalescer.admit(0x200, 11, waiter_count));
  coalescer.close_address(0x200);
  EXPECT_TRUE(coalescer.admit(0x200, 20, waiter_count));
  EXPECT_FALSE(coalescer.admit(0x200, 21, waiter_count));

  std::deque<unsigned> old_waiters;
  EXPECT_TRUE(coalescer.close_master(10, old_waiters));
  ASSERT_EQ(1u, old_waiters.size());
  EXPECT_EQ(11u, old_waiters.front());

  // Closing the old generation must not remove the new active generation.
  EXPECT_FALSE(coalescer.admit(0x200, 22, waiter_count));
  EXPECT_EQ(2u, waiter_count);
  std::deque<unsigned> new_waiters;
  EXPECT_TRUE(coalescer.close_master(20, new_waiters));
  ASSERT_EQ(2u, new_waiters.size());
  EXPECT_EQ(21u, new_waiters[0]);
  EXPECT_EQ(22u, new_waiters[1]);
}

TEST(L2TmaCoalescingTest, AddressesAndUnknownResponsesAreIndependent) {
  test_coalescer coalescer;
  unsigned long long waiter_count = 0;

  EXPECT_TRUE(coalescer.admit(0x300, 30, waiter_count));
  EXPECT_TRUE(coalescer.admit(0x320, 31, waiter_count));
  EXPECT_FALSE(coalescer.admit(0x300, 32, waiter_count));
  EXPECT_FALSE(coalescer.admit(0x320, 33, waiter_count));

  std::deque<unsigned> waiters;
  EXPECT_FALSE(coalescer.close_master(99, waiters));
  EXPECT_TRUE(waiters.empty());
  EXPECT_TRUE(coalescer.close_master(31, waiters));
  ASSERT_EQ(1u, waiters.size());
  EXPECT_EQ(33u, waiters.front());
}

}  // namespace
