// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/l2cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

struct test_item {
  test_item(unsigned id_value, unsigned sectors_value = 1)
      : id(id_value), sectors(sectors_value) {}

  unsigned id;
  unsigned sectors;
};

typedef rop_delay_output_queue<test_item *> test_queue;

rop_delay_output_service_result service(
    test_queue &queue, unsigned long long cycle, unsigned width,
    std::vector<unsigned> &accepted, unsigned downstream_capacity = 1000) {
  return queue.service(
      cycle, width, [](const test_item *item) { return item->sectors; },
      [&accepted, downstream_capacity]() {
        return accepted.size() >= downstream_capacity;
      },
      [&accepted](const test_item *item) { accepted.push_back(item->id); });
}

std::vector<test_item> make_items(unsigned count, unsigned first_id = 0) {
  std::vector<test_item> items;
  items.reserve(count);
  for (unsigned i = 0; i < count; ++i)
    items.push_back(test_item(first_id + i));
  return items;
}

TEST(RopDelayOutputTest, FixedReadyCycleCannotBeBypassed) {
  test_queue queue;
  test_item item(7);
  std::vector<unsigned> accepted;
  queue.push(&item, 10, false);

  const rop_delay_output_service_result early =
      service(queue, 9, 4, accepted);
  EXPECT_EQ(ROP_DELAY_OUTPUT_NO_READY_WORK, early.reason);
  EXPECT_EQ(0u, early.accepted_items);
  EXPECT_TRUE(accepted.empty());
  EXPECT_EQ(1u, queue.size());

  const rop_delay_output_service_result ready =
      service(queue, 10, 4, accepted);
  EXPECT_EQ(1u, ready.accepted_items);
  EXPECT_EQ(1u, ready.accepted_sectors);
  ASSERT_EQ(1u, accepted.size());
  EXPECT_EQ(7u, accepted[0]);
  EXPECT_TRUE(queue.empty());
}

TEST(RopDelayOutputTest, WidthOnePreservesLegacyOneItemCadence) {
  test_queue queue;
  test_item multi_sector(1, 4);
  test_item sector(2);
  std::vector<unsigned> accepted;
  queue.push(&multi_sector, 0, false);
  queue.push(&sector, 0, false);

  rop_delay_output_service_result result = service(queue, 0, 1, accepted);
  EXPECT_EQ(ROP_DELAY_OUTPUT_WIDTH_LIMITED, result.reason);
  EXPECT_EQ(1u, result.accepted_items);
  EXPECT_EQ(4u, result.accepted_sectors);
  ASSERT_EQ(1u, accepted.size());
  EXPECT_EQ(1u, accepted[0]);

  result = service(queue, 1, 1, accepted);
  EXPECT_EQ(1u, result.accepted_items);
  ASSERT_EQ(2u, accepted.size());
  EXPECT_EQ(2u, accepted[1]);
  EXPECT_TRUE(queue.empty());
}

class RopDelayOutputWidthTest : public ::testing::TestWithParam<unsigned> {};

TEST_P(RopDelayOutputWidthTest, AcceptsExactlyConfiguredReadySectorWidth) {
  const unsigned width = GetParam();
  test_queue queue;
  std::vector<test_item> items = make_items(width + 1);
  std::vector<unsigned> accepted;
  for (unsigned i = 0; i < items.size(); ++i)
    queue.push(&items[i], 4, (i & 1) != 0);

  const rop_delay_output_service_result result =
      service(queue, 4, width, accepted);
  EXPECT_EQ(ROP_DELAY_OUTPUT_WIDTH_LIMITED, result.reason);
  EXPECT_EQ(width, result.accepted_items);
  EXPECT_EQ(width, result.accepted_sectors);
  EXPECT_EQ(width, accepted.size());
  EXPECT_EQ(1u, queue.size());
}

INSTANTIATE_TEST_SUITE_P(WidthsTwoThreeFour, RopDelayOutputWidthTest,
                         ::testing::Values(2u, 3u, 4u));

TEST(RopDelayOutputTest, SameReadyCyclePreservesLegacyLocalPriority) {
  test_queue queue;
  std::vector<test_item> items = make_items(4, 10);
  std::vector<unsigned> accepted;
  queue.push(&items[0], 8, true);
  queue.push(&items[1], 8, false);
  queue.push(&items[2], 8, true);
  queue.push(&items[3], 8, false);

  for (unsigned tick = 0; tick != 4; ++tick) {
    const rop_delay_output_service_result result =
        service(queue, 8 + tick, 1, accepted);
    EXPECT_EQ(1u, result.accepted_items);
  }
  const unsigned expected[] = {11, 13, 10, 12};
  EXPECT_TRUE(std::equal(accepted.begin(), accepted.end(), expected));
  EXPECT_TRUE(queue.empty());
}

TEST(RopDelayOutputTest, EarlierReadyCycleWinsAcrossLocalAndRemoteQueues) {
  test_queue queue;
  test_item later(1);
  test_item earlier(2);
  std::vector<unsigned> accepted;
  queue.push(&later, 12, false);
  queue.push(&earlier, 11, true);

  const rop_delay_output_service_result result =
      service(queue, 12, 2, accepted);
  EXPECT_EQ(2u, result.accepted_items);
  const unsigned expected[] = {2, 1};
  EXPECT_TRUE(std::equal(accepted.begin(), accepted.end(), expected));
}

TEST(RopDelayOutputTest, DownstreamBackpressureRetainsOrderWithoutCredit) {
  test_queue queue;
  std::vector<test_item> items = make_items(5, 20);
  std::vector<unsigned> first;
  for (unsigned i = 0; i < items.size(); ++i)
    queue.push(&items[i], 3, (i & 1) != 0);

  rop_delay_output_service_result result = service(queue, 3, 4, first, 2);
  EXPECT_EQ(ROP_DELAY_OUTPUT_DOWNSTREAM_FULL, result.reason);
  EXPECT_EQ(2u, result.accepted_items);
  EXPECT_EQ(3u, queue.size());

  // A later tick receives only that tick's width; it cannot consume an idle
  // or blocked tick's unused service as a burst credit.
  std::vector<unsigned> second;
  result = service(queue, 4, 2, second);
  EXPECT_EQ(ROP_DELAY_OUTPUT_WIDTH_LIMITED, result.reason);
  EXPECT_EQ(2u, result.accepted_items);
  EXPECT_EQ(1u, queue.size());
  result = service(queue, 5, 2, second);
  EXPECT_EQ(1u, result.accepted_items);
  EXPECT_TRUE(queue.empty());

  first.insert(first.end(), second.begin(), second.end());
  const unsigned expected[] = {20, 22, 24, 21, 23};
  ASSERT_EQ(5u, first.size());
  EXPECT_TRUE(std::equal(first.begin(), first.end(), expected));
}

TEST(RopDelayOutputTest, StatsSeparateWidthAndDownstreamStops) {
  rop_delay_output_service_stats stats;
  rop_delay_output_service_result width;
  width.reason = ROP_DELAY_OUTPUT_WIDTH_LIMITED;
  width.accepted_items = 3;
  width.accepted_sectors = 3;
  stats.record(width);

  rop_delay_output_service_result blocked;
  blocked.reason = ROP_DELAY_OUTPUT_DOWNSTREAM_FULL;
  blocked.accepted_items = 1;
  blocked.accepted_sectors = 1;
  stats.record(blocked);

  EXPECT_EQ(4u, stats.accepted_items);
  EXPECT_EQ(4u, stats.accepted_sectors);
  EXPECT_EQ(2u, stats.service_ticks);
  EXPECT_EQ(3u, stats.max_sectors_per_tick);
  EXPECT_EQ(1u, stats.width_limited_ticks);
  EXPECT_EQ(1u, stats.downstream_full_ticks);
  EXPECT_EQ(1u, stats.queue_full_ticks);
}

#if GTEST_HAS_DEATH_TEST
TEST(RopDelayOutputDeathTest, MultiIssueRejectsNonSectorItems) {
  test_queue queue;
  test_item item(1, 2);
  std::vector<unsigned> accepted;
  queue.push(&item, 0, false);
  EXPECT_DEATH(service(queue, 0, 2, accepted), "32-byte sector children");
}
#endif

}  // namespace
