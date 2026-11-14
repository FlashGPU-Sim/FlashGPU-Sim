#include "../test_support.h"

namespace {

TEST(InstructionFetchTimingTest, ChargesBoundedPhysicalBackedgeRefills) {
  instruction_loop_buffer_config config;
  config.backedge_redirect_latency = 7;
  config.capacity_bytes = 4224;
  config.refill_granularity_bytes = 256;
  config.refill_latency = 7;
  config.max_refill_latency = 13;

  EXPECT_EQ(instruction_loop_refill_delay(0x2000, 0x1000, 16, config), 7u);
  EXPECT_EQ(instruction_loop_refill_delay(0x2070, 0x1000, 16, config), 7u);
  EXPECT_EQ(instruction_loop_refill_delay(0x2071, 0x1000, 16, config), 14u);
  EXPECT_EQ(instruction_loop_refill_delay(0x2170, 0x1000, 16, config), 14u);
  EXPECT_EQ(instruction_loop_refill_delay(0x2171, 0x1000, 16, config), 20u);
  EXPECT_EQ(instruction_loop_refill_delay(0x4000, 0x1000, 16, config), 20u);
  EXPECT_EQ(instruction_loop_refill_delay(0x1000, 0x2000, 16, config), 0u);
}

TEST(SassAsyncProxyTimingTest, TracksCtaEpochPerWarpAndCompletionToken) {
  async_proxy_timing proxy(4, {/*completion_stall=*/28,
                               /*dirty_completion_extra_stall=*/23,
                               /*initiation_stall=*/5,
                               /*dirty_initiation_stall=*/15});

  EXPECT_EQ(proxy.issue_fence(0, true), 28u);
  proxy.record_tma_load();
  proxy.record_tma_load();
  EXPECT_EQ(proxy.producer_epoch(), 2u);

  // A tokenless fence consumes frontend capacity but does not publish the
  // new epoch.  The following completion-token fence therefore remains dirty.
  EXPECT_EQ(proxy.issue_fence(0, false), 15u);
  EXPECT_EQ(proxy.observed_epoch(0), 0u);
  EXPECT_EQ(proxy.issue_fence(0, true), 51u);
  EXPECT_EQ(proxy.observed_epoch(0), 2u);
  EXPECT_EQ(proxy.issue_fence(0, true), 28u);

  // Synchronization is per warp even though TMA history is CTA-scoped.
  EXPECT_EQ(proxy.observed_epoch(1), 0u);
  EXPECT_EQ(proxy.issue_fence(1, true), 51u);
  EXPECT_EQ(proxy.observed_epoch(1), 2u);
}

TEST(SassSharedProxyTimingTest, TracksPerSchedulerVisibilityAndPerWarpFence) {
  shared_proxy_timing proxy(
      8, 4, {/*visibility_latency=*/32, /*initiation_interval=*/4});

  proxy.record_matrix_store(0, 0, 100);
  EXPECT_EQ(proxy.visible_cycle(0), 132u);
  proxy.record_matrix_store(0, 0, 102);
  EXPECT_EQ(proxy.visible_cycle(0), 136u);

  // A different scheduler partition admits independently.
  proxy.record_matrix_store(1, 1, 102);
  EXPECT_EQ(proxy.visible_cycle(1), 134u);
  EXPECT_TRUE(proxy.fence_ready(2, 0));
  EXPECT_FALSE(proxy.fence_ready(0, 135));
  EXPECT_TRUE(proxy.fence_ready(0, 136));

  // Reusing a hardware warp clears its dependency without discarding the
  // scheduler partition's in-flight visibility occupancy.
  proxy.reset_warp(0);
  EXPECT_TRUE(proxy.fence_ready(0, 0));
  proxy.record_matrix_store(0, 0, 103);
  EXPECT_EQ(proxy.visible_cycle(0), 140u);
  proxy.reset();
  proxy.record_matrix_store(0, 0, 1);
  EXPECT_EQ(proxy.visible_cycle(0), 33u);
}

TEST(InstructionDependencyTrackerTest, EnforcesStallsAndExplicitBarriers) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.stall_cycles = 5;
  producer.write_barrier = 2;

  EXPECT_TRUE(tracker.ready(producer, 100));
  tracker.issue(producer, 11, true, 100);
  EXPECT_EQ(tracker.next_issue_cycle(), 105u);
  EXPECT_TRUE(tracker.barrier_pending(2));
  EXPECT_FALSE(tracker.control_delay_ready(104));
  EXPECT_TRUE(tracker.control_delay_ready(105));

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 2;
  EXPECT_EQ(tracker.pending_wait_mask(consumer, 105), uint8_t{1} << 2);
  EXPECT_FALSE(tracker.ready(consumer, 104));
  EXPECT_FALSE(tracker.ready(consumer, 105));
  tracker.issue(producer, 12, true, 105);
  tracker.complete(producer, 11, 105);
  EXPECT_FALSE(tracker.ready(consumer, 110));
  tracker.complete(producer, 12, 110);
  EXPECT_EQ(tracker.pending_wait_mask(consumer, 110), uint8_t{1} << 2);
  EXPECT_FALSE(tracker.ready(consumer, 110));
  EXPECT_EQ(tracker.pending_wait_mask(consumer, 111), 0);
  EXPECT_TRUE(tracker.ready(consumer, 111));
}

TEST(InstructionDependencyTrackerTest, PredicatedOffIssueOnlyAppliesStall) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t control;
  control.stall_cycles = 3;
  control.read_barrier = 4;

  tracker.issue(control, 21, false, 7);
  EXPECT_EQ(tracker.next_issue_cycle(), 10u);
  EXPECT_FALSE(tracker.barrier_pending(4));
  EXPECT_TRUE(tracker.ready({}, 10));
}

TEST(InstructionDependencyTrackerTest,
     HoldsVariableWriteBarrierPastPipelineCompletion) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.write_barrier = 2;
  tracker.issue(producer, 31, true, 100);
  tracker.hold_write_barrier_until(producer, 31, 136);

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 2;
  tracker.complete(producer, 31, 108);
  EXPECT_FALSE(tracker.ready(consumer, 108));
  EXPECT_FALSE(tracker.ready(consumer, 135));
  EXPECT_TRUE(tracker.ready(consumer, 136));
}

TEST(InstructionDependencyTrackerTest, ReleasesReadBarrierAfterOperandRead) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.read_barrier = 4;
  tracker.issue(producer, 21, true, 7);

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 4;
  EXPECT_TRUE(tracker.barrier_pending(4));
  EXPECT_FALSE(tracker.ready(consumer, 7));

  tracker.operands_read(producer, 21, 7);
  EXPECT_TRUE(tracker.barrier_pending(4));
  EXPECT_FALSE(tracker.ready(consumer, 7));
  EXPECT_TRUE(tracker.ready(consumer, 8));
  EXPECT_FALSE(tracker.barrier_pending(4));

  // A later completion remains harmless for the collector-released token.
  tracker.complete(producer, 21, 8);
  EXPECT_TRUE(tracker.ready(consumer, 8));
}

TEST(InstructionDependencyTrackerTest,
     DefersVariableReadBarrierUntilServiceWindow) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.read_barrier = 4;
  tracker.issue(producer, 51, true, 10);

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 4;
  tracker.operands_read(producer, 51, 11, true);
  EXPECT_FALSE(tracker.ready(consumer, 12));

  tracker.begin_service(producer, 51, SFU_OP, 12, 0, 7);
  EXPECT_FALSE(tracker.ready(consumer, 18));
  EXPECT_TRUE(tracker.ready(consumer, 19));
}

TEST(InstructionDependencyTrackerTest, ForwardsSfuResultsOnlyToSpConsumers) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.write_barrier = 2;
  tracker.issue(producer, 31, true, 90);

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 2;
  EXPECT_FALSE(tracker.ready(consumer, 200, SP_OP));
  tracker.begin_service(producer, 31, SFU_OP, 100, 19);
  EXPECT_FALSE(tracker.ready(consumer, 118, SP_OP));
  EXPECT_TRUE(tracker.ready(consumer, 119, SP_OP));
  EXPECT_FALSE(tracker.ready(consumer, 119, SFU_OP));
  EXPECT_TRUE(tracker.barrier_pending(2));

  tracker.complete(producer, 31, 119);
  EXPECT_FALSE(tracker.ready(consumer, 119, SFU_OP));
  EXPECT_TRUE(tracker.ready(consumer, 120, SFU_OP));
}

TEST(MioLdsmTimingTest, LimitsEachSchedulerIndependently) {
  flash_gpgpu_sim::mio_ldsm_timing timing(4);
  EXPECT_TRUE(timing.can_issue(0, 100, 7));
  timing.issue(0, 100, 7);
  EXPECT_FALSE(timing.can_issue(0, 106, 7));
  EXPECT_TRUE(timing.can_issue(0, 107, 7));
  EXPECT_TRUE(timing.can_issue(1, 100, 7));
  timing.issue(1, 100, 7);
  EXPECT_EQ(timing.next_issue_cycle(0), 107u);
  EXPECT_EQ(timing.next_issue_cycle(1), 107u);
}

TEST(CtaBarrierTimingTest, SharesOneAdmissionPathAcrossSchedulers) {
  flash_gpgpu_sim::cta_barrier_timing timing;
  EXPECT_TRUE(timing.can_issue(100, 2));
  timing.issue(100, 2);
  EXPECT_FALSE(timing.can_issue(101, 2));
  EXPECT_TRUE(timing.can_issue(102, 2));
  timing.issue(102, 2);
  EXPECT_EQ(timing.next_issue_cycle(), 104u);
}

TEST(CtaBarrierTimingTest, ResetAndZeroIntervalDisableTheLimit) {
  flash_gpgpu_sim::cta_barrier_timing timing;
  timing.issue(20, 0);
  EXPECT_TRUE(timing.can_issue(20, 0));
  EXPECT_EQ(timing.next_issue_cycle(), 0u);
  timing.issue(20, 2);
  timing.reset();
  EXPECT_TRUE(timing.can_issue(20, 2));
}

TEST(MioLdsmTimingTest, ZeroIntervalDisablesTheLimit) {
  flash_gpgpu_sim::mio_ldsm_timing timing(1);
  timing.issue(0, 20, 0);
  EXPECT_TRUE(timing.can_issue(0, 20, 0));
  EXPECT_EQ(timing.next_issue_cycle(0), 0u);
  timing.issue(0, 20, 5);
  timing.reset();
  EXPECT_TRUE(timing.can_issue(0, 20, 5));
}

TEST(TensorCoreAdmissionTimingTest, AppliesOneLanePerScheduler) {
  flash_gpgpu_sim::tensor_core_admission_timing timing(4);
  EXPECT_TRUE(timing.can_issue(0, 100, 32));
  timing.issue(0, 100, 32);
  EXPECT_FALSE(timing.can_issue(0, 131, 32));
  EXPECT_TRUE(timing.can_issue(0, 132, 32));
  EXPECT_TRUE(timing.can_issue(1, 100, 32));
  timing.issue(1, 100, 16);
  EXPECT_EQ(timing.next_issue_cycle(0), 132u);
  EXPECT_EQ(timing.next_issue_cycle(1), 116u);
}

TEST(TensorCoreAdmissionTimingTest, ResetAndZeroIntervalDisableTheLimit) {
  flash_gpgpu_sim::tensor_core_admission_timing timing(1);
  timing.issue(0, 20, 0);
  EXPECT_TRUE(timing.can_issue(0, 20, 0));
  EXPECT_EQ(timing.next_issue_cycle(0), 0u);
  timing.issue(0, 20, 32);
  timing.reset();
  EXPECT_TRUE(timing.can_issue(0, 20, 32));
}

TEST(InstructionDependencyTrackerTest, SeparatesReadAndWriteRolesOnOneBarrier) {
  instruction_dependency_tracker tracker;
  inst_t::dependency_control_t producer;
  producer.read_barrier = 2;
  producer.write_barrier = 2;
  tracker.issue(producer, 41, true, 10);

  inst_t::dependency_control_t consumer;
  consumer.wait_mask = uint8_t{1} << 2;
  tracker.operands_read(producer, 41, 10);
  EXPECT_FALSE(tracker.ready(consumer, 10));
  EXPECT_FALSE(tracker.ready(consumer, 11));

  tracker.complete(producer, 41, 11);
  EXPECT_FALSE(tracker.ready(consumer, 11));
  EXPECT_TRUE(tracker.ready(consumer, 12));
}

TEST(TimingMetadataTest, InitializesEveryBackendOperandSlot) {
  inst_t inst;
  EXPECT_FALSE(inst.has_explicit_dependency_control());
  for (unsigned value : inst.out) EXPECT_EQ(value, 0u);
  for (unsigned value : inst.in) EXPECT_EQ(value, 0u);
}

TEST(TimingMetadataTest, CarriesFrontendNeutralHardwareSemantics) {
  inst_t inst;

  inst_t::wgmma_static_info_t wgmma;
  wgmma.operation = inst_t::wgmma_static_info_t::WGMMA_MMA_ASYNC;
  wgmma.accumulator_bytes_per_thread = 64;
  wgmma.register_a_registers_per_thread = 4;
  wgmma.commit_group_after_issue = true;
  inst.set_wgmma_static_info(wgmma);

  inst_t::mbarrier_static_info_t mbarrier;
  mbarrier.operation =
      inst_t::mbarrier_static_info_t::MBARRIER_ARRIVE_EXPECT_TX;
  mbarrier.arrive = true;
  mbarrier.expect_tx = true;
  inst.set_mbarrier_static_info(mbarrier);

  inst_t::async_copy_static_info_t async_copy;
  async_copy.has_source_size = true;
  async_copy.source_size = 8;
  async_copy.mbarrier_increment_pending = false;
  inst.set_async_copy_static_info(async_copy);

  EXPECT_TRUE(inst.get_wgmma_static_info().is_mma_async());
  EXPECT_TRUE(inst.get_wgmma_static_info().uses_register_a());
  EXPECT_TRUE(inst.get_wgmma_static_info().commit_group_after_issue);
  EXPECT_EQ(inst.get_wgmma_static_info().accumulator_bytes_per_thread, 64u);
  EXPECT_TRUE(inst.get_mbarrier_static_info().arrive);
  EXPECT_TRUE(inst.get_mbarrier_static_info().expect_tx);
  EXPECT_TRUE(inst.get_async_copy_static_info().has_source_size);
  EXPECT_EQ(inst.get_async_copy_static_info().source_size, 8u);
  EXPECT_FALSE(inst.get_async_copy_static_info().mbarrier_increment_pending);
}

}  // namespace
