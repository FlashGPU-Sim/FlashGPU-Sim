#include <gtest/gtest.h>

#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/l2cache.h"
#include "gpgpu-sim/mem_fetch.h"

namespace {

mem_fetch *FakeRequest(unsigned long &storage) {
  return reinterpret_cast<mem_fetch *>(&storage);
}

void AcceptHit(l2_multi_issue_ports &ports) {
  ASSERT_TRUE(ports.can_accept_lookup(1));
  ASSERT_TRUE(ports.data_port_has_capacity());
  ports.accept_lookup(1);
  ASSERT_EQ(ports.accept_data(1, L2_MULTI_ISSUE_HIT_DATA), 1u);
}

TEST(L2PortModelSelectionTest, LegacyAndMultiIssueModesAreMutuallyExclusive) {
  EXPECT_FALSE(l2_multi_issue_port_model_enabled(0));
  EXPECT_TRUE(l2_multi_issue_port_model_enabled(1));
  EXPECT_DEATH_IF_SUPPORTED(l2_multi_issue_port_model_enabled(2), "mode <= 1");
}

TEST(L2PortModelSelectionTest,
     LegacyModeDoesNotActivateMultiIssueSectorAccounting) {
  l2_multi_issue_ports ports;
  ports.configure(3, 3);

  if (l2_multi_issue_port_model_enabled(0)) {
    AcceptHit(ports);
  }

  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 0u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 0u);
}

TEST(L2MultiIssuePortsTest, DataWidthsOneTwoAndThreeAcceptThatManySectors) {
  for (unsigned width = 1; width <= 3; ++width) {
    l2_multi_issue_ports ports;
    ports.configure(/*lookup_width=*/4, width);
    EXPECT_EQ(ports.accept_data(/*pending_sectors=*/4, L2_MULTI_ISSUE_HIT_DATA),
              width);
    EXPECT_EQ(ports.data_remaining(), 0u);
    EXPECT_EQ(ports.stats().data_port_accepted_sectors, width);
  }
}

TEST(L2MultiIssuePortsTest, ThreeIndependentSectorHitsIssueInOneTick) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3);

  AcceptHit(ports);
  AcceptHit(ports);
  AcceptHit(ports);

  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 3u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorDemandUsesTwoPlusOneAtWidthTwo) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/2, /*data_width=*/2);

  AcceptHit(ports);
  AcceptHit(ports);
  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());

  ports.begin_cycle();
  AcceptHit(ports);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 3u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorOperationCompletesAtWidthThree) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/3);
  l2_multi_issue_pending_operation operation;
  operation.start(3);

  EXPECT_TRUE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(operation.remaining_sectors(), 0u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, ThreeSectorOperationUsesTwoPlusOneAtWidthTwo) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/2);
  l2_multi_issue_pending_operation operation;
  operation.start(3);

  EXPECT_FALSE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(operation.remaining_sectors(), 1u);
  ports.begin_cycle();
  EXPECT_TRUE(operation.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 3u);
}

TEST(L2MultiIssuePortsTest, HitAndDirtyEvictionShareTheDataWidth) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3);

  AcceptHit(ports);
  l2_multi_issue_pending_operation dirty_eviction;
  dirty_eviction.start(3);
  EXPECT_FALSE(
      dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 1u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 2u);
}

TEST(L2MultiIssuePortsTest,
     FourSectorDirtyEvictionAtWidthThreeIsNotVisibleEarly) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3);
  l2_multi_issue_pending_operation dirty_eviction;
  dirty_eviction.start(4);
  bool downstream_visible = false;

  if (dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION))
    downstream_visible = true;
  EXPECT_FALSE(downstream_visible);
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 1u);

  ports.begin_cycle();
  if (dirty_eviction.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION))
    downstream_visible = true;
  EXPECT_TRUE(downstream_visible);
  EXPECT_EQ(dirty_eviction.remaining_sectors(), 0u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 4u);
}

TEST(L2MultiIssuePortsTest, CleanMissConsumesOnlyLookupSectorWork) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/3, /*data_width=*/3);

  ASSERT_TRUE(ports.can_accept_lookup(1));
  ports.accept_lookup(1);

  EXPECT_EQ(ports.lookup_remaining(), 2u);
  EXPECT_EQ(ports.data_remaining(), 3u);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 0u);
}

TEST(L2MultiIssuePortsTest,
     ReadyForwardBypassesExhaustedDataPortAndFictionalDirtyCandidate) {
  constexpr new_addr_type kAddress = 0x2000;
  mshr_table mshrs(/*num_entries=*/1, /*max_merged=*/4);
  unsigned long first_storage = 0;
  unsigned long older_merged_storage = 0;
  unsigned long late_storage = 0;
  mem_fetch *first = FakeRequest(first_storage);
  mem_fetch *older_merged = FakeRequest(older_merged_storage);
  mem_fetch *late = FakeRequest(late_storage);

  mshrs.add(kAddress, first, /*is_atomic=*/false);
  mshrs.add(kAddress, older_merged, /*is_atomic=*/false);
  bool has_atomic = true;
  mshrs.mark_ready(kAddress, has_atomic);
  ASSERT_FALSE(has_atomic);
  ASSERT_EQ(mshrs.next_access(), first);
  ASSERT_TRUE(mshrs.probe_ready(kAddress));

  const bool ready_read_forward = l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/false, /*is_atomic=*/false, MISS);
  ASSERT_TRUE(ready_read_forward);
  EXPECT_FALSE(l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/true, /*is_atomic=*/false, MISS));
  EXPECT_FALSE(l2_cache::ready_read_forward_eligible(
      mshrs, kAddress, /*is_write=*/false, /*is_atomic=*/true, MISS));

  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/2, /*data_width=*/1);

  AcceptHit(ports);
  ASSERT_EQ(ports.data_remaining(), 0u);

  // A tag-only probe can nominate a dirty victim after the original filled
  // line has been evicted. A late ordinary read instead joins the ready MSHR,
  // so that prospective replacement must not become data-port work.
  ASSERT_FALSE(l2_multi_issue_needs_data_port(
      MISS, /*prospective_dirty_eviction_sectors=*/4, ready_read_forward));
  ASSERT_TRUE(ports.can_accept_lookup(1));
  ports.accept_lookup(1);
  mshrs.add_ready(kAddress, late);

  EXPECT_EQ(ports.lookup_remaining(), 0u);
  EXPECT_EQ(ports.data_remaining(), 0u);
  EXPECT_EQ(ports.stats().lookup_accepted_sectors, 2u);
  EXPECT_EQ(ports.stats().data_port_accepted_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_hit_sectors, 1u);
  EXPECT_EQ(ports.stats().data_port_dirty_eviction_sectors, 0u);
  EXPECT_EQ(ports.stats().data_port_width_stall_cycles, 0u);

  EXPECT_TRUE(l2_multi_issue_needs_data_port(
      MISS, /*prospective_dirty_eviction_sectors=*/4,
      /*ready_read_forward=*/false));

  EXPECT_EQ(mshrs.next_access(), older_merged);
  EXPECT_EQ(mshrs.next_access(), late);
  EXPECT_FALSE(mshrs.access_ready());
}

TEST(L2MultiIssuePortsTest, WidthStallsCountOncePerPortPerTick) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/1, /*data_width=*/1);
  AcceptHit(ports);

  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.can_accept_lookup(1));
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_FALSE(ports.data_port_has_capacity());
  EXPECT_EQ(ports.stats().lookup_width_stall_cycles, 1u);
  EXPECT_EQ(ports.stats().data_port_width_stall_cycles, 1u);

  ports.begin_cycle();
  EXPECT_TRUE(ports.can_accept_lookup(1));
  EXPECT_TRUE(ports.data_port_has_capacity());
  EXPECT_EQ(ports.stats().lookup_width_stall_cycles, 1u);
  EXPECT_EQ(ports.stats().data_port_width_stall_cycles, 1u);
}

TEST(L2MultiIssuePortsTest, AcceptedSectorStatisticsAreConservative) {
  l2_multi_issue_ports ports;
  ports.configure(/*lookup_width=*/4, /*data_width=*/4);
  ports.accept_lookup(1);  // clean miss
  AcceptHit(ports);
  l2_multi_issue_pending_operation dirty;
  dirty.start(2);
  EXPECT_TRUE(dirty.service_data(ports, L2_MULTI_ISSUE_DIRTY_EVICTION));
  const l2_multi_issue_port_stats &stats = ports.stats();
  EXPECT_EQ(stats.lookup_accepted_sectors, 2u);
  EXPECT_EQ(stats.data_port_accepted_sectors, 3u);
  EXPECT_EQ(
      stats.data_port_accepted_sectors,
      stats.data_port_hit_sectors + stats.data_port_dirty_eviction_sectors);
}

}  // namespace
