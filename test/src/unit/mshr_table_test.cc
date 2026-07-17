#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/gpu-cache.h"

namespace {

mem_fetch *fake_request(unsigned long &storage) {
  return reinterpret_cast<mem_fetch *>(&storage);
}

TEST(MshrTableTest, PendingEntryAcceptsMergedAccesses) {
  mshr_table table(/*num_entries=*/2, /*max_merged=*/4);
  unsigned long first_storage = 0;
  unsigned long second_storage = 0;
  mem_fetch *first = fake_request(first_storage);
  mem_fetch *second = fake_request(second_storage);
  constexpr new_addr_type address = 0x1000;

  table.add(address, first, /*is_atomic=*/false);
  ASSERT_TRUE(table.probe(address));
  EXPECT_FALSE(table.probe_ready(address));
  EXPECT_FALSE(table.full(address));

  table.add(address, second, /*is_atomic=*/false);
  EXPECT_FALSE(table.full(address));
}

TEST(MshrTableTest, LateReadJoinsDrainingResponse) {
  mshr_table table(/*num_entries=*/2, /*max_merged=*/4);
  unsigned long first_storage = 0;
  unsigned long second_storage = 0;
  unsigned long late_storage = 0;
  mem_fetch *first = fake_request(first_storage);
  mem_fetch *second = fake_request(second_storage);
  mem_fetch *late = fake_request(late_storage);
  constexpr new_addr_type address = 0x2000;

  table.add(address, first, /*is_atomic=*/false);
  table.add(address, second, /*is_atomic=*/false);

  bool has_atomic = true;
  table.mark_ready(address, has_atomic);
  ASSERT_FALSE(has_atomic);
  EXPECT_FALSE(table.probe(address));
  ASSERT_TRUE(table.probe_ready(address));
  ASSERT_TRUE(table.ready_for_forward(address));

  EXPECT_EQ(table.next_access(), first);
  ASSERT_TRUE(table.probe_ready(address));

  table.add_ready(address, late);
  EXPECT_EQ(table.next_access(), second);
  EXPECT_EQ(table.next_access(), late);
  EXPECT_FALSE(table.access_ready());
  EXPECT_FALSE(table.probe(address));
  EXPECT_FALSE(table.probe_ready(address));
}

TEST(MshrTableTest, FullDrainingEntryRejectsLateForward) {
  mshr_table table(/*num_entries=*/1, /*max_merged=*/2);
  unsigned long first_storage = 0;
  unsigned long second_storage = 0;
  mem_fetch *first = fake_request(first_storage);
  mem_fetch *second = fake_request(second_storage);
  constexpr new_addr_type address = 0x3000;

  table.add(address, first, /*is_atomic=*/false);
  table.add(address, second, /*is_atomic=*/false);

  bool has_atomic = false;
  table.mark_ready(address, has_atomic);
  EXPECT_TRUE(table.probe_ready(address));
  EXPECT_FALSE(table.ready_for_forward(address));
  EXPECT_TRUE(table.full(address));
}

TEST(MshrTableTest, AtomicStateIsReportedWhenResponseArrives) {
  mshr_table table(/*num_entries=*/1, /*max_merged=*/2);
  unsigned long storage = 0;
  mem_fetch *request = fake_request(storage);
  constexpr new_addr_type address = 0x4000;

  table.add(address, request, /*is_atomic=*/true);

  bool has_atomic = false;
  table.mark_ready(address, has_atomic);
  EXPECT_TRUE(has_atomic);
}

}  // namespace
