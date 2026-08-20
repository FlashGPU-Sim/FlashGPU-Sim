#include <gtest/gtest.h>

#include "gpgpu-sim/flash/tcgen05/timing.h"

using namespace flash_gpgpu_sim;

namespace {

tcgen05_timing_config_t config(unsigned depth = 0) {
  tcgen05_timing_config_t result;
  result.mma_issue_interval = 2;
  result.mma_completion_base = 3;
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
