#include <gtest/gtest.h>

#include "gpgpu-sim/flash/shared_bandwidth.h"
#include "gpgpu-sim/flash/tcgen05/timing.h"

using namespace flash_gpgpu_sim;

namespace {

tcgen05_timing_config_t config(unsigned depth = 0) {
  tcgen05_timing_config_t result;
  result.mma_issue_interval = 2;
  result.mma_completion_tail_latency = 3;
  result.mma_f16_flops_per_cycle = 16;
  result.async_queue_depth = depth;
  return result;
}

tcgen05_thread_stream_key_t thread_stream(unsigned cta = 0, unsigned warp = 1,
                                          unsigned lane = 0) {
  tcgen05_thread_stream_key_t result;
  result.hw_cta_id = cta;
  result.warp_id = warp;
  result.lane_id = lane;
  result.cta_group = 1;
  return result;
}

tcgen05_warp_stream_key_t warp_stream(unsigned cta = 0, unsigned warp = 1) {
  tcgen05_warp_stream_key_t result;
  result.hw_cta_id = cta;
  result.warp_id = warp;
  return result;
}

tcgen05_op_t mma(uint64_t work) {
  tcgen05_op_t result;
  result.kind = TCGEN05_TIMING_MMA;
  result.work = work;
  return result;
}

tcgen05_op_t memory_op(tcgen05_op_kind_t kind, unsigned latency,
                       unsigned interval = 1) {
  tcgen05_op_t result;
  result.kind = kind;
  result.completion_latency = latency;
  result.initiation_interval = interval;
  return result;
}

}  // namespace

TEST(Tcgen05TimingTest, MmaUsesSerializedBackendAndCompletionTail) {
  tcgen05_unit_t unit(config());
  const tcgen05_thread_stream_key_t stream = thread_stream();

  unit.enqueue_thread_op(stream, mma(32), 0);  // compute [0,2), tail to 5
  EXPECT_FALSE(unit.can_enqueue(TCGEN05_TIMING_MMA, 1));
  unit.enqueue_thread_op(stream, mma(16), 2);  // compute [2,3), tail to 6
  unit.commit(stream, 0x80);

  for (uint64_t cycle = 0; cycle < 6; ++cycle) {
    unit.cycle(cycle);
    EXPECT_TRUE(unit.take_completion_events().empty());
  }
  unit.cycle(6);
  const std::vector<tcgen05_completion_event_t> events =
      unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, tcgen05_completion_event_t::MBARRIER_ARRIVAL);
  EXPECT_EQ(events[0].mbarrier_addr, 0x80u);
  EXPECT_EQ(unit.stats().completed[TCGEN05_TIMING_MMA], 2u);
}

TEST(Tcgen05TimingTest, CommitCutoffExcludesLaterOperations) {
  tcgen05_unit_t unit(config());
  const tcgen05_thread_stream_key_t stream = thread_stream();

  unit.enqueue_thread_op(stream, mma(16), 0);
  unit.commit(stream, 0x40);
  unit.enqueue_thread_op(stream, mma(160), 2);

  unit.cycle(4);
  ASSERT_EQ(unit.take_completion_events().size(), 1u);
  EXPECT_EQ(unit.queue_occupancy(), 1u);
}

TEST(Tcgen05TimingTest, MmaCarriesFractionalCapacityAcrossBusyPeriod) {
  tcgen05_timing_config_t timing_config = config();
  timing_config.mma_issue_interval = 1;
  tcgen05_unit_t unit(timing_config);
  const tcgen05_thread_stream_key_t stream = thread_stream();

  unit.enqueue_thread_op(stream, mma(17), 0);
  unit.enqueue_thread_op(stream, mma(17), 1);
  unit.commit(stream, 0x44);

  unit.cycle(6);  // ceil(34 / 16) + 3 completion-tail cycles
  EXPECT_EQ(unit.take_completion_events().size(), 1u);
}

TEST(Tcgen05TimingTest, ThreadStreamsAreLaneIsolated) {
  tcgen05_unit_t unit(config());
  unit.enqueue_thread_op(thread_stream(0, 1, 7), mma(160), 0);
  unit.commit(thread_stream(0, 1, 0), 0x20);

  const std::vector<tcgen05_completion_event_t> events =
      unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].mbarrier_addr, 0x20u);
}

TEST(Tcgen05TimingTest, WarpWaitUsesTypeSpecificCutoff) {
  tcgen05_unit_t unit(config());
  const tcgen05_warp_stream_key_t stream = warp_stream();
  unit.enqueue_warp_mem_op(stream, memory_op(TCGEN05_TIMING_LD, 3), 0);
  unit.enqueue_warp_mem_op(stream, memory_op(TCGEN05_TIMING_ST, 8), 0);

  EXPECT_FALSE(unit.wait_ld(stream));
  unit.cycle(3);
  const std::vector<tcgen05_completion_event_t> events =
      unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, tcgen05_completion_event_t::RELEASE_LD_WAIT);
  EXPECT_EQ(unit.queue_occupancy(), 1u);
}

TEST(Tcgen05TimingTest, WarpWaitCutoffExcludesLaterLoads) {
  tcgen05_unit_t unit(config());
  const tcgen05_warp_stream_key_t stream = warp_stream();
  unit.enqueue_warp_mem_op(stream, memory_op(TCGEN05_TIMING_LD, 3), 0);
  EXPECT_FALSE(unit.wait_ld(stream));
  unit.enqueue_warp_mem_op(stream, memory_op(TCGEN05_TIMING_LD, 10), 1);

  unit.cycle(3);
  EXPECT_EQ(unit.take_completion_events().size(), 1u);
  EXPECT_EQ(unit.queue_occupancy(), 1u);
}

TEST(Tcgen05TimingTest, QueueDepthAppliesAcrossOperationKinds) {
  tcgen05_unit_t unit(config(2));
  unit.enqueue_thread_op(thread_stream(), mma(16), 0);
  unit.enqueue_warp_mem_op(warp_stream(), memory_op(TCGEN05_TIMING_LD, 5), 0);

  EXPECT_FALSE(unit.can_enqueue(TCGEN05_TIMING_ST, 0));
  EXPECT_EQ(unit.stats().queue_full_stall_cycles, 1u);
  EXPECT_EQ(unit.stats().max_queue_occupancy, 2u);
}

TEST(Tcgen05TimingTest, CleanupCtaPreservesOtherCtas) {
  tcgen05_unit_t unit(config());
  unit.enqueue_thread_op(thread_stream(0), mma(16), 0);
  unit.enqueue_warp_mem_op(warp_stream(1), memory_op(TCGEN05_TIMING_ST, 5), 0);

  unit.cleanup_cta(0);
  EXPECT_EQ(unit.queue_occupancy(), 1u);
  EXPECT_FALSE(unit.wait_st(warp_stream(1)));
}

TEST(Tcgen05TimingTest, MulticastCommitRetainsMaskAndCtaGroupCutoff) {
  tcgen05_unit_t unit(config());
  auto pair = thread_stream(2, 4, 0);
  pair.cta_group = 2;
  unit.enqueue_thread_op(pair, memory_op(TCGEN05_TIMING_CP, 9), 0);
  unit.commit(pair, 0x180, 0x30);
  auto single = pair;
  single.cta_group = 1;
  unit.commit(single, 0x188);
  unit.cycle(1);
  auto events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].mbarrier_addr, 0x188u);
  EXPECT_EQ(events[0].multicast_mask, 0u);
  unit.cycle(8);
  EXPECT_TRUE(unit.take_completion_events().empty());
  unit.cycle(9);
  events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].multicast_mask, 0x30u);
  EXPECT_EQ(events[0].hw_cta_id, 2u);
  EXPECT_EQ(events[0].warp_id, 4u);
  EXPECT_EQ(events[0].mbarrier_addr, 0x180u);
  unit.cycle(10);
  EXPECT_TRUE(unit.take_completion_events().empty());
}

namespace {
tcgen05_timing_config_t b200_timing() {
  tcgen05_timing_config_t c;
  c.mma_precision_timing = true;
  c.b200_recurrence = true;
  c.mma_f16_flops_per_cycle = 8192;
  unsigned rates[] = {8192, 4096, 16384, 16384, 32768};
  std::copy(rates, rates + 5, c.mma_rates);
  return c;
}
tcgen05_op_t measured_mma(tcgen05_mma_kind kind, tcgen05_native_profile profile,
                          unsigned k, unsigned n = 64) {
  auto op = mma(2ull * 128 * n * k);
  op.mma.kind = kind;
  op.mma.profile = profile;
  op.mma.m = 128;
  op.mma.n = n;
  op.mma.k = k;
  return op;
}
uint64_t completion(tcgen05_unit_t &u, uint64_t from = 0) {
  for (uint64_t c = from; c < 100000; ++c) {
    u.cycle(c);
    if (!u.take_completion_events().empty()) return c;
  }
  return ~uint64_t(0);
}
}  // namespace
TEST(Tcgen05TimingTest, MeasuredDatatypeChainsShareTheFortyEightCyclePeriod) {
  using K = tcgen05_mma_kind;
  using P = tcgen05_native_profile;
  for (const auto &op : {measured_mma(K::half, P::bf16_f32, 16),
                         measured_mma(K::tf32, P::tf32_f32, 8),
                         measured_mma(K::mixed, P::e2m1_e4m3_f32_block32, 32),
                         measured_mma(K::fp4, P::e2m1_f32_block32, 64)}) {
    tcgen05_unit_t u(b200_timing());
    auto s = thread_stream();
    for (unsigned i = 0; i < 8; ++i) u.enqueue_thread_op(s, op, i);
    u.commit(s, 0x40);
    EXPECT_EQ(completion(u, 8), 8u * 48);
    EXPECT_EQ(u.stats().recurrence_table_ops, 8u);
  }
}
TEST(Tcgen05TimingTest, UnknownTypeDoesNotAliasBf16AndTailIsNotRecurrence) {
  auto cfg = b200_timing();
  cfg.mma_completion_tail_latency = 7;
  tcgen05_unit_t u(cfg);
  auto op =
      measured_mma(tcgen05_mma_kind::half, tcgen05_native_profile::unknown, 16);
  auto s = thread_stream();
  u.enqueue_thread_op(s, op, 0);
  u.enqueue_thread_op(s, op, 1);
  u.commit(s, 0x40);
  EXPECT_EQ(completion(u, 2), 71u);  // 2*32 compute, one final visibility tail
  EXPECT_EQ(u.stats().recurrence_fallback_ops, 2u);
}
TEST(Tcgen05TimingTest, PairedMmaReservesBothSmBudgets) {
  auto cfg = b200_timing();
  cfg.b200_recurrence = false;
  tcgen05_unit_t a(cfg), b(cfg);
  auto s = thread_stream();
  s.cta_group = 2;
  auto op = mma(8192 * 20);
  a.enqueue_thread_op(s, op, 0, &b);  // ten cycles on each SM
  b.enqueue_thread_op(thread_stream(), mma(8192 * 3), 1);
  b.commit(thread_stream(), 0x40);
  EXPECT_EQ(completion(b, 1), 13u);
  a.commit(s, 0x40);
  EXPECT_EQ(completion(a), 10u);
}
TEST(Tcgen05TimingTest, SharedSupplyOverlapsComputeAndHoldsPairedResources) {
  auto cfg = b200_timing();
  cfg.b200_recurrence = false;
  tcgen05_unit_t a(cfg), b(cfg);
  auto s = thread_stream();
  s.cta_group = 2;
  a.enqueue_thread_op(s, mma(8192 * 8), 0, &b,
                      [](uint64_t start) { return start + 12; });
  a.commit(s, 0x40);
  EXPECT_EQ(completion(a), 13u);  // max(4,12), not 4+12
  b.enqueue_thread_op(thread_stream(), mma(8192 * 2), 1);
  b.commit(thread_stream(), 0x48);
  EXPECT_EQ(completion(b, 1), 15u);
}
TEST(Tcgen05TimingTest, TmemReadServiceIsPartitionLocal) {
  tcgen05_unit_t u(config());
  auto op = memory_op(TCGEN05_TIMING_LD, 26, 16);
  op.partitioned_mem = true;
  auto s = warp_stream();
  s.partition = 0;
  u.enqueue_warp_mem_op(s, op, 0);
  EXPECT_FALSE(u.can_enqueue(TCGEN05_TIMING_LD, 1, 0));
  EXPECT_TRUE(u.can_enqueue(TCGEN05_TIMING_LD, 1, 1));
  s.warp_id = 2;
  s.partition = 1;
  u.enqueue_warp_mem_op(s, op, 1);
  EXPECT_FALSE(u.wait_ld(s));
  u.cycle(26);
  EXPECT_TRUE(u.take_completion_events().empty());
  u.cycle(27);
  EXPECT_EQ(u.take_completion_events().size(), 1u);
}
TEST(Tcgen05TimingTest, OwnerAdmissionLimitsDoNotBlockIndependentOwners) {
  auto cfg = b200_timing();
  cfg.owner_queue_depth[0] = 1;
  cfg.owner_issue_interval = 8;
  tcgen05_unit_t u(cfg);
  auto a = thread_stream(), b = thread_stream(1);
  u.enqueue_thread_op(a,
                      measured_mma(tcgen05_mma_kind::half,
                                   tcgen05_native_profile::bf16_f32, 16),
                      0);
  EXPECT_FALSE(u.can_enqueue(TCGEN05_TIMING_MMA, 8, ~0u, &a));
  EXPECT_TRUE(u.can_enqueue(TCGEN05_TIMING_MMA, 8, ~0u, &b));
  u.cycle(48);
  EXPECT_TRUE(u.can_enqueue(TCGEN05_TIMING_MMA, 48, ~0u, &a));
}
TEST(Tcgen05TimingTest, CollectorAndTransposeDoNotBorrowDefaultRecurrence) {
  tcgen05_mma_descriptor_t d{};
  d.m = 64;
  d.n = 64;
  d.k = 16;
  d.d_type = 1;
  d.a_type = d.b_type = 1;
  EXPECT_EQ(tcgen05_metadata(tcgen05_mma_kind::half, d).profile,
            tcgen05_native_profile::bf16_f32);
  EXPECT_EQ(
      tcgen05_metadata(tcgen05_mma_kind::half, d, 1, false, true, true).profile,
      tcgen05_native_profile::unknown);
  d.transpose_b = true;
  EXPECT_EQ(tcgen05_metadata(tcgen05_mma_kind::half, d).profile,
            tcgen05_native_profile::unknown);
}
TEST(Tcgen05TimingTest, PackedSharedReadsAndRegionsAreAccountedOncePerOperand) {
  tcgen05_shared_reads r;
  r.add(131070, 2);
  r.add(131071, 2);
  r.add(131072, 2);
  r.begin_operand();
  r.add(131070, 2);
  tcgen05_mma_metadata m;
  tcgen05_account_shared(m, r.reads);
  EXPECT_EQ(m.shared_bytes[0][0], 4u);
  EXPECT_EQ(m.shared_bytes[0][1], 2u);
}

TEST(Tcgen05TimingTest,
     SharedReadContentionIsPerHalfAndWriteDirectionIndependent) {
  tc_shared_calendar_t calendar;
  EXPECT_EQ(calendar.reserve(10, {256, 0}, 128, 128), 12u);
  shared_bandwidth_t ports;
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TC, false, calendar.take(10),
                            128, 128));
  EXPECT_EQ(
      ports.consume_up_to(shared_bandwidth_t::LSU, false, 0, 128, 128, 128),
      0u);
  EXPECT_EQ(
      ports.consume_up_to(shared_bandwidth_t::LSU, false, 1, 128, 128, 128),
      128u);
  EXPECT_EQ(
      ports.consume_up_to(shared_bandwidth_t::TMA, true, 0, 128, 128, 128),
      128u);
  ports.reset();
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TC, false, calendar.take(11),
                            128, 128));
  EXPECT_EQ(
      ports.consume_up_to(shared_bandwidth_t::LSU, false, 0, 128, 128, 128),
      0u);
  ports.reset();
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TC, false, calendar.take(12),
                            128, 128));
  EXPECT_EQ(
      ports.consume_up_to(shared_bandwidth_t::LSU, false, 0, 128, 128, 128),
      128u);
}
TEST(Tcgen05TimingTest, PairedBusyAccountingDoesNotDependOnSmCycleOrder) {
  auto cfg = b200_timing();
  cfg.b200_recurrence = false;
  tcgen05_unit_t a(cfg), b(cfg);
  a.cycle(0);  // issuer already cycled, peer has not
  auto s = thread_stream();
  s.cta_group = 2;
  a.enqueue_thread_op(s, mma(8192 * 20), 0, &b);
  b.cycle(0);
  for (unsigned c = 1; c <= 10; ++c) {
    b.cycle(c);
    a.cycle(c);
  }
  EXPECT_EQ(a.stats().backend_busy_cycles, 10u);
  EXPECT_EQ(b.stats().backend_busy_cycles, 10u);
}

TEST(Tcgen05TimingTest, PipelineTmemWaitCannotCompleteAtSchedulerDeadline) {
  tcgen05_unit_t unit(config());
  const auto s = warp_stream();
  unit.enqueue_warp_mem_op(s, memory_op(TCGEN05_TIMING_LD, 11), 100, 1);
  EXPECT_FALSE(unit.wait_ld(s));
  unit.cycle(
      111);  // Original scheduler-relative completion would be premature.
  EXPECT_TRUE(unit.take_completion_events().empty());
  unit.complete_warp_mem_op(
      1, 116);  // Actual operand/execute/writeback completion.
  unit.cycle(116);
  ASSERT_EQ(unit.take_completion_events().size(), 1u);
  EXPECT_EQ(unit.stats().completed[TCGEN05_TIMING_LD], 1u);
}

TEST(Tcgen05TimingTest, PipelineTmemCompletionPreservesWarpAndWaitCutoff) {
  tcgen05_unit_t unit(config());
  const auto a = warp_stream(0, 0), b = warp_stream(0, 1);
  unit.enqueue_warp_mem_op(a, memory_op(TCGEN05_TIMING_LD, 11), 0, 10);
  EXPECT_FALSE(unit.wait_ld(a));
  unit.enqueue_warp_mem_op(a, memory_op(TCGEN05_TIMING_LD, 11), 1, 11);
  unit.enqueue_warp_mem_op(b, memory_op(TCGEN05_TIMING_LD, 11), 2, 12);
  EXPECT_FALSE(unit.wait_ld(b));
  unit.complete_warp_mem_op(11,
                            16);  // Out of order completion cannot release a.
  unit.cycle(16);
  EXPECT_TRUE(unit.take_completion_events().empty());
  unit.complete_warp_mem_op(10, 17);
  unit.cycle(17);
  auto events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].warp_id, 0u);
  unit.complete_warp_mem_op(12, 18);
  unit.cycle(18);
  events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].warp_id, 1u);
}

TEST(Tcgen05TimingTest, TmemStoreServiceAndCompletionArePartitionLocal) {
  tcgen05_unit_t unit(config());
  auto op = memory_op(TCGEN05_TIMING_ST, 28, 28);
  op.partitioned_mem = true;
  auto first = warp_stream(0, 0), other = warp_stream(0, 1);
  other.partition = 1;
  unit.enqueue_warp_mem_op(first, op, 0, 100);
  EXPECT_FALSE(unit.can_enqueue(TCGEN05_TIMING_ST, 1, 0));
  EXPECT_TRUE(unit.can_enqueue(TCGEN05_TIMING_ST, 1, 1));
  unit.enqueue_warp_mem_op(other, op, 1, 101);
  EXPECT_FALSE(unit.wait_st(first));
  EXPECT_FALSE(unit.wait_st(other));
  unit.cycle(28);
  EXPECT_TRUE(unit.take_completion_events().empty());
  unit.complete_warp_mem_op(101, 35);
  unit.cycle(35);
  auto events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, tcgen05_completion_event_t::RELEASE_ST_WAIT);
  EXPECT_EQ(events[0].warp_id, 1u);
  unit.complete_warp_mem_op(100, 36);
  unit.cycle(36);
  events = unit.take_completion_events();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].warp_id, 0u);
}

TEST(Tcgen05TimingTest, TmaLandingSharesWriteHalfButNotLsuReadPort) {
  tc_shared_calendar_t landing;
  const auto span = shared_bandwidth_t::split(128 * 1024 - 64, 256, 128 * 1024);
  EXPECT_EQ(span[0], 64u);
  EXPECT_EQ(span[1], 192u);
  // Completion is exclusive of the last SRAM service cycle.
  EXPECT_EQ(landing.reserve(10, span, 128, 128), 12u);
  shared_bandwidth_t ports;
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TMA, true, landing.take(10),
                           128, 128));
  EXPECT_EQ(ports.consume_up_to(shared_bandwidth_t::LSU, true, 0, 128, 128, 128), 64u);
  // LSU read/write share the same path even when SRAM directions differ.
  EXPECT_EQ(ports.consume_up_to(shared_bandwidth_t::LSU, false, 1, 128, 128, 128), 64u);
  ports.reset();
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TMA, true, landing.take(11),
                           128, 128));
  EXPECT_EQ(ports.consume_up_to(shared_bandwidth_t::LSU, true, 1, 128, 128, 128), 0u);
  EXPECT_EQ(ports.consume_up_to(shared_bandwidth_t::LSU, true, 0, 128, 128, 128), 128u);
  ports.reset();
  EXPECT_TRUE(ports.consume(shared_bandwidth_t::TMA, true, landing.take(12),
                           128, 128));
  EXPECT_EQ(ports.consume_up_to(shared_bandwidth_t::LSU, true, 1, 128, 128, 128), 128u);
}

TEST(Tcgen05TimingTest, MultipleTmaReturnsCannotOverbookLandingCycles) {
  tc_shared_calendar_t landing;
  EXPECT_EQ(landing.reserve(3, {256, 0}, 128, 128), 5u);
  EXPECT_EQ(landing.reserve(3, {0, 128}, 128, 128), 6u);
  EXPECT_EQ(landing.take(3), (shared_bandwidth_t::bytes_t{128, 0}));
  EXPECT_EQ(landing.take(4), (shared_bandwidth_t::bytes_t{128, 0}));
  EXPECT_EQ(landing.take(5), (shared_bandwidth_t::bytes_t{0, 128}));
  EXPECT_EQ(landing.take(6), (shared_bandwidth_t::bytes_t{}));
}
