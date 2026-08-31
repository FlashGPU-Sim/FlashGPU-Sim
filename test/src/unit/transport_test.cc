// Unit tests for shipped flit-level interconnect primitives.
// Links src/gpgpu-sim/transport.h — no local reimplementation.

#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/transport.h"

TEST(Transport, OccupancyCountsPayloadFlits) {
  bounded_voq_t voq;
  voq.init(/*n_dst=*/2, /*flit_limit_per_dst=*/8);
  transport_packet_metadata_t pkt{};
  pkt.dst = 0;
  pkt.payload_bytes = 128;
  pkt.total_flits = 4;
  pkt.remaining_flits = 4;
  pkt.created_cycle = 10;
  ASSERT_TRUE(voq.push(0, pkt));
  EXPECT_EQ(voq.occupancy_flits(0), 4u);
  EXPECT_EQ(voq.occupancy_flits(1), 0u);
  EXPECT_EQ(voq.occupancy_flits(), 4u);
  EXPECT_EQ(voq.stats().occupancy_high_water, 4u);

  transport_packet_metadata_t done{};
  EXPECT_FALSE(voq.grant_flit(0, /*now=*/12, &done));
  EXPECT_EQ(voq.occupancy_flits(0), 3u);
  EXPECT_EQ(voq.front(0)->remaining_flits, 3u);
  EXPECT_EQ(voq.stats().flits_moved, 1u);

  EXPECT_FALSE(voq.grant_flit(0, 13, nullptr));
  EXPECT_FALSE(voq.grant_flit(0, 14, nullptr));
  EXPECT_TRUE(voq.grant_flit(0, 15, &done));
  EXPECT_TRUE(voq.empty(0));
  EXPECT_EQ(voq.occupancy_flits(0), 0u);
  EXPECT_EQ(voq.stats().packets_out, 1u);
  EXPECT_EQ(voq.stats().latency_samples, 1u);
  EXPECT_EQ(voq.stats().latency_sum, 5u);
}

TEST(Transport, FullDestDoesNotHolSibling) {
  bounded_voq_t voq;
  voq.init(2, /*flit_limit_per_dst=*/2);
  transport_packet_metadata_t a{};
  a.dst = 0;
  a.remaining_flits = 2;
  transport_packet_metadata_t b{};
  b.dst = 1;
  b.remaining_flits = 1;
  ASSERT_TRUE(voq.push(0, a));
  EXPECT_FALSE(voq.can_push(0, 1));
  EXPECT_FALSE(voq.push(0, b));
  EXPECT_TRUE(voq.can_push(1, 1));
  ASSERT_TRUE(voq.push(1, b));
  EXPECT_EQ(voq.occupancy_flits(0), 2u);
  EXPECT_EQ(voq.occupancy_flits(1), 1u);
  EXPECT_EQ(voq.front(1)->remaining_flits, 1u);

  transport_packet_metadata_t completed{};
  EXPECT_TRUE(voq.grant_flit(1, 1, &completed));
  EXPECT_TRUE(voq.empty(1));
  EXPECT_FALSE(voq.empty(0));
}

TEST(Transport, OversizedPacketStreamsThroughBoundedVoq) {
  bounded_voq_t voq;
  voq.init(/*n_dst=*/1, /*flit_limit_per_dst=*/4);
  transport_packet_metadata_t pkt{};
  pkt.remaining_flits = 9;
  ASSERT_TRUE(voq.push(0, pkt));
  EXPECT_EQ(voq.occupancy_flits(0), 4u);
  EXPECT_FALSE(voq.can_push(0, 1));

  transport_packet_metadata_t done{};
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_FALSE(voq.grant_flit(0, i, &done));
    EXPECT_LE(voq.occupancy_flits(0), 4u);
  }
  EXPECT_TRUE(voq.grant_flit(0, 8, &done));
  EXPECT_EQ(done.remaining_flits, 0u);
  EXPECT_EQ(voq.occupancy_flits(0), 0u);
}

TEST(Transport, CreditTakeDoesNotCrossQueue) {
  flit_credit_counters_t credits;
  credits.init(/*n_queues=*/2, /*depth=*/4);
  EXPECT_EQ(credits.remaining(0), 4u);
  EXPECT_EQ(credits.remaining(1), 4u);
  ASSERT_TRUE(credits.take(0, 4));
  EXPECT_EQ(credits.remaining(0), 0u);
  EXPECT_EQ(credits.remaining(1), 4u);
  EXPECT_FALSE(credits.take(0, 1));
  EXPECT_EQ(credits.remaining(1), 4u);
  ASSERT_TRUE(credits.take(1, 1));
  EXPECT_EQ(credits.remaining(1), 3u);
  credits.give(0, 2);
  EXPECT_EQ(credits.remaining(0), 2u);
  EXPECT_EQ(credits.remaining(1), 3u);
}

TEST(Transport, RoundRobinRotates) {
  round_robin_arbiter_t rr;
  rr.init(3);
  const bool all_ready[3] = {true, true, true};
  EXPECT_EQ(rr.grant(all_ready, 3), 0u);
  EXPECT_EQ(rr.grant(all_ready, 3), 1u);
  EXPECT_EQ(rr.grant(all_ready, 3), 2u);
  EXPECT_EQ(rr.grant(all_ready, 3), 0u);

  const bool skip_one[3] = {true, false, true};
  EXPECT_EQ(rr.grant(skip_one, 3), 2u);
  EXPECT_EQ(rr.grant(skip_one, 3), 0u);

  const bool none[3] = {false, false, false};
  EXPECT_EQ(rr.grant(none, 3), ~0u);
}

TEST(Transport, SinkOccupancyIsPayloadFlits) {
  interconnect_sink_t sink;
  sink.init(/*flit_limit=*/3);
  EXPECT_TRUE(sink.can_accept(2));
  ASSERT_TRUE(sink.push(2));
  EXPECT_EQ(sink.occupancy_flits(), 2u);
  EXPECT_EQ(sink.free_flits(), 1u);
  EXPECT_FALSE(sink.push(2));
  ASSERT_TRUE(sink.push(1));
  EXPECT_EQ(sink.occupancy_flits(), 3u);
  sink.pop(1);
  EXPECT_EQ(sink.occupancy_flits(), 2u);
}
