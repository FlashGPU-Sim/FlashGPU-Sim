#include "../test_support.h"

namespace {

TEST(SassCtaExecutorTest, DefersFourWarpsAtReusableBarrier) {
  kernel image;
  image.name = "cta_barrier";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.SYNC.DEFER_BLOCKING", "0x0"),
      official_instruction(0x10, "MOV", "R1,0x2a"),
      official_instruction(0x20, "BAR.SYNC.DEFER_BLOCKING", "0x0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "cta_barrier", 4);

  for (unsigned arrival = 0; arrival < 4; ++arrival) {
    ASSERT_EQ(cta.step().status, step_status::kAdvanced);
    EXPECT_EQ(cta.warp(arrival).pc, 0x10u);
    EXPECT_FALSE(cta.warp(arrival).waiting_at_cta_barrier);
    EXPECT_EQ(cta.warp(arrival).deferred_cta_barrier_pending[0], arrival != 3);
  }

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(cta.instructions_executed(), 32u);
  } while (result.status != step_status::kExited);

  EXPECT_EQ(cta.instructions_executed(), 16u);
  for (unsigned warp_id = 0; warp_id < 4; ++warp_id) {
    EXPECT_FALSE(cta.warp(warp_id).waiting_at_cta_barrier);
    for (unsigned lane = 0; lane < 32; ++lane)
      EXPECT_EQ(cta.warp(warp_id).read_register(lane, 1), 42u);
  }
}

TEST(SassCtaExecutorTest,
     DeferredBarrierAllowsAluPenetrationAndBlocksSharedConsumer) {
  kernel image;
  image.name = "deferred_cta_barrier";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.SYNC.DEFER_BLOCKING", "0x0,0x40"),
      official_instruction(0x10, "MOV", "R0,RZ"),
      official_instruction(0x20, "MOV", "R1,0x2a"),
      official_instruction(0x30, "LDS", "R2[R0]"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0, sizeof(uint32_t), true);
  memory.initialize<uint32_t>(memory_space::kShared, 0, 0x12345678u);
  cta_execution_state cta(2);
  warp_state producer;
  warp_state peer;
  execution_context producer_context(&memory);
  producer_context.cta = &cta;
  producer_context.warp_id = 0;
  execution_context peer_context(&memory);
  peer_context.cta = &cta;
  peer_context.warp_id = 1;

  EXPECT_EQ(
      sass.step("deferred_cta_barrier", producer, producer_context).status,
      step_status::kAdvanced);
  EXPECT_TRUE(producer.deferred_cta_barrier_pending[0]);
  EXPECT_EQ(
      sass.step("deferred_cta_barrier", producer, producer_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(
      sass.step("deferred_cta_barrier", producer, producer_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(producer.pc, 0x30u);
  EXPECT_EQ(producer.read_register(0, 1), 42u);

  const uint64_t executed_before_block = producer_context.instructions_executed;
  step_result result =
      sass.step("deferred_cta_barrier", producer, producer_context);
  EXPECT_EQ(result.status, step_status::kBlocked) << result.detail;
  EXPECT_EQ(producer.pc, 0x30u);
  EXPECT_EQ(producer_context.instructions_executed, executed_before_block);

  EXPECT_EQ(sass.step("deferred_cta_barrier", peer, peer_context).status,
            step_status::kAdvanced);
  EXPECT_EQ(cta.generation(0), 1u);
  result = sass.step("deferred_cta_barrier", producer, producer_context);
  EXPECT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(producer.pc, 0x40u);
  EXPECT_EQ(producer.read_register(0, 2), 0x12345678u);
  EXPECT_FALSE(producer.deferred_cta_barrier_pending[0]);
}

TEST(SassCtaExecutorTest, DeferredBarrierTriggerClassesMatchSm90Evidence) {
  EXPECT_FALSE(triggers_deferred_cta_barrier(
      official_instruction(0, "BAR.ARV", "0x1,0x80")));
  EXPECT_FALSE(triggers_deferred_cta_barrier(
      official_instruction(0, "BAR.SYNC.DEFER_BLOCKING", "0x1,0x80")));
  EXPECT_TRUE(triggers_deferred_cta_barrier(
      official_instruction(0, "LDS.128", "R8[R2]")));
  EXPECT_TRUE(triggers_deferred_cta_barrier(
      official_instruction(0, "STS.64", "[R2],R8")));
  EXPECT_TRUE(triggers_deferred_cta_barrier(
      official_instruction(0, "LDSM.16.M88.4", "R8[R2]")));
  EXPECT_TRUE(triggers_deferred_cta_barrier(official_instruction(
      0, "HGMMA.64x64x16.F32", "R24,gdesc[UR4],gdesc[UR6],1")));
  EXPECT_FALSE(triggers_deferred_cta_barrier(official_instruction(
      0, "SYNCS.PHASECHK.TRANS64.TRYWAIT", "P0,[UR4],R2")));
  EXPECT_FALSE(triggers_deferred_cta_barrier(
      official_instruction(0, "WARPGROUP.ARRIVE")));
  EXPECT_FALSE(triggers_deferred_cta_barrier(
      official_instruction(0, "CS2R", "R2,SR_CLOCKLO")));
  EXPECT_FALSE(triggers_deferred_cta_barrier(
      official_instruction(0, "IADD3", "R2,R2,0x1,RZ")));
}

TEST(SassCtaExecutorTest, DeferredBarrierOrderingIsArchitectureSpecific) {
  const instruction arrive = official_instruction(0, "BAR.ARV", "0x1,0x80");
  const instruction deferred =
      official_instruction(0, "BAR.SYNC.DEFER_BLOCKING", "0x1,0x80");
  const instruction phase_check =
      official_instruction(0, "SYNCS.PHASECHK.TRANS64.TRYWAIT", "P0,[UR4],R2");
  const instruction tma_load =
      official_instruction(0, "UTMALDG.2D", "[UR8][UR4],desc[UR28]");
  const instruction mbarrier_arrive =
      official_instruction(0, "SYNCS.ARRIVE.TRANS64", "RZ[R0+URZ],R1", "@P0");
  const instruction mbarrier_complete = official_instruction(
      0, "SYNCS.ARRIVE.TRANS64.RED.A0TX", "RZ[R0+URZ],R1", "@P0");
  EXPECT_TRUE(orders_deferred_cta_barrier(arrive, architecture::kSm90));
  EXPECT_TRUE(orders_deferred_cta_barrier(deferred, architecture::kSm90));
  EXPECT_FALSE(orders_deferred_cta_barrier(phase_check, architecture::kSm90));
  EXPECT_FALSE(orders_deferred_cta_barrier(tma_load, architecture::kSm90));
  EXPECT_TRUE(
      orders_deferred_cta_barrier(mbarrier_arrive, architecture::kSm90));
  EXPECT_TRUE(
      orders_deferred_cta_barrier(mbarrier_complete, architecture::kSm90));
  EXPECT_TRUE(orders_deferred_cta_barrier(arrive, architecture::kSm120));
  EXPECT_TRUE(orders_deferred_cta_barrier(deferred, architecture::kSm120));
  EXPECT_TRUE(orders_deferred_cta_barrier(phase_check, architecture::kSm120));
  EXPECT_TRUE(orders_deferred_cta_barrier(tma_load, architecture::kSm120));
  EXPECT_TRUE(
      orders_deferred_cta_barrier(mbarrier_arrive, architecture::kSm120));
  EXPECT_TRUE(
      orders_deferred_cta_barrier(mbarrier_complete, architecture::kSm120));
}

TEST(SassCtaExecutorTest, DeferredBarrierOrdersAFollowingNamedBarrierArrival) {
  kernel image;
  image.name = "deferred_barrier_then_arrive";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.SYNC.DEFER_BLOCKING", "0x0,0x40"),
      official_instruction(0x10, "MOV", "R1,0x2a"),
      official_instruction(0x20, "BAR.ARV", "0x1,0x40"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  cta_execution_state cta(2);
  warp_state first;
  warp_state peer;
  execution_context first_context;
  first_context.cta = &cta;
  first_context.warp_id = 0;
  execution_context peer_context;
  peer_context.cta = &cta;
  peer_context.warp_id = 1;

  ASSERT_EQ(
      sass.step("deferred_barrier_then_arrive", first, first_context).status,
      step_status::kAdvanced);
  ASSERT_EQ(
      sass.step("deferred_barrier_then_arrive", first, first_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(first.pc, 0x20u);
  EXPECT_EQ(
      sass.step("deferred_barrier_then_arrive", first, first_context).status,
      step_status::kBlocked);
  EXPECT_EQ(cta.generation(1), 0u);

  ASSERT_EQ(
      sass.step("deferred_barrier_then_arrive", peer, peer_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(cta.generation(0), 1u);
  EXPECT_EQ(
      sass.step("deferred_barrier_then_arrive", first, first_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(cta.generation(1), 0u);
}

TEST(SassCtaExecutorTest,
     DeferredBarrierAllowsIndependentMbarrierPollingToMakeProgress) {
  kernel image;
  image.name = "deferred_barrier_mbarrier_poll";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.PHASECHK.TRANS64.TRYWAIT",
                           "P0[UR4],R2"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  cta_execution_state cta(2);
  ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
  ASSERT_TRUE(cta.arrive_mbarrier(0x100, 1, 0));
  ASSERT_EQ(cta.mbarrier_phase(0x100), 1u);

  warp_state state;
  state.write_uniform_register(4, 0x100);
  for (unsigned lane = 0; lane < flash_gpgpu_sim::sass::kWarpLanes; ++lane)
    state.write_register(lane, 2, 0);
  state.deferred_cta_barrier_pending[9] = true;
  state.deferred_cta_barrier_generation[9] = cta.generation(9);

  execution_context context;
  context.cta = &cta;
  context.warp_id = 0;
  const step_result result =
      sass.step("deferred_barrier_mbarrier_poll", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.pc, 0x10u);
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_TRUE(state.deferred_cta_barrier_pending[9]);
}

TEST(SassCtaExecutorTest, QueryEmptyOrdersNoTmaCompletion) {
  kernel image;
  image.name = "query_empty_no_tma";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.SYNC.DEFER_BLOCKING", "0x8,0x40"),
      official_instruction(0x10, "MOV", "R2,0x2a"),
      official_instruction(0x20, "SYNCS.ARRIVE.TRANS64", "RZ[R0+URZ],R1",
                           "@P0"),
      official_instruction(0x30, "SYNCS.ARRIVE.TRANS64.RED.A0TX",
                           "RZ[R0+URZ],R1", "@P0"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  cta_execution_state cta(2);
  ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
  warp_state producer, consumer;
  execution_context producer_context, consumer_context;
  producer_context.cta = consumer_context.cta = &cta;
  producer_context.memory = consumer_context.memory = &memory;
  producer_context.warp_id = 0;
  consumer_context.warp_id = 1;
  producer.write_register(0, 0, 0x100);
  producer.write_register(0, 1, 64);
  producer.write_predicate(0, 0, true);
  for (unsigned lane = 1; lane < flash_gpgpu_sim::sass::kWarpLanes; ++lane)
    producer.write_predicate(lane, 0, false);

  ASSERT_EQ(sass.step("query_empty_no_tma", producer, producer_context).status,
            step_status::kAdvanced);
  ASSERT_EQ(sass.step("query_empty_no_tma", producer, producer_context).status,
            step_status::kAdvanced);
  EXPECT_EQ(producer.read_register(0, 2), 0x2au);
  // Independent address arithmetic may advance, but publishing Q-ready must
  // wait until the consumer has released the old Q buffer.
  ASSERT_EQ(sass.step("query_empty_no_tma", producer, producer_context).status,
            step_status::kBlocked);
  EXPECT_EQ(producer.pc, 0x20u);
  EXPECT_EQ(cta.mbarrier_pending_arrivals(0x100), 1u);
  EXPECT_EQ(cta.mbarrier_phase(0x100), 0u);
  ASSERT_EQ(sass.step("query_empty_no_tma", consumer, consumer_context).status,
            step_status::kAdvanced);
  ASSERT_EQ(sass.step("query_empty_no_tma", producer, producer_context).status,
            step_status::kAdvanced);
  EXPECT_EQ(cta.mbarrier_pending_transaction_bytes(0x100), 64u);
  ASSERT_EQ(sass.step("query_empty_no_tma", producer, producer_context).status,
            step_status::kAdvanced);
  EXPECT_EQ(cta.mbarrier_phase(0x100), 1u);
}

TEST(SassCtaExecutorTest, PendingMbarrierTryWaitCompletesTheOriginalPredicate) {
  kernel image;
  image.name = "pending_mbarrier_predicate";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.PHASECHK.TRANS64.TRYWAIT",
                           "P0[R0+URZ],R2"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  cta_execution_state cta(1);
  ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
  warp_state state;
  state.active_mask = 0x1;
  state.write_register(0, 0, 0x100);
  state.write_register(0, 2, 0);
  execution_context context;
  context.cta = &cta;
  context.warp_id = 0;

  const step_result result =
      sass.step("pending_mbarrier_predicate", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_FALSE(state.read_predicate(0, 0));
  EXPECT_EQ(state.mbarrier_try_wait,
            mbarrier_try_wait_state::kFunctionalPending);
  EXPECT_EQ(state.pending_mbarrier_try_wait_mask, 0x1u);

  timing_instruction projected(*sass.fetch("pending_mbarrier_predicate", 0));
  flash_gpgpu_sim::sass::project_mbarrier_effects(context, projected);
  EXPECT_EQ(projected.get_mbarrier_info(0).bar_id, 0x100u);
  // Duplicate active lanes must not produce duplicate backend releases.
  context.mbarrier_effects[1] = context.mbarrier_effects[0];
  flash_gpgpu_sim::sass::project_mbarrier_effects(context, projected);
  EXPECT_EQ(projected.get_mbarrier_info(1).bar_id, UINT32_MAX);
  context.clear_instruction_effects();
  flash_gpgpu_sim::sass::project_mbarrier_effects(context, projected);
  EXPECT_EQ(projected.get_mbarrier_info(0).bar_id, UINT32_MAX);

  EXPECT_EQ(state.complete_mbarrier_try_wait(),
            mbarrier_try_wait_state::kFunctionalPending);
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_EQ(state.mbarrier_try_wait, mbarrier_try_wait_state::kNone);
  EXPECT_EQ(state.pending_mbarrier_try_wait_mask, 0u);
  EXPECT_EQ(state.complete_mbarrier_try_wait(), mbarrier_try_wait_state::kNone);
}

TEST(SassCtaExecutorTest, CompletedMbarrierTryWaitIsAValidTimingReleaseNoOp) {
  kernel image;
  image.name = "completed_mbarrier_predicate";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.PHASECHK.TRANS64.TRYWAIT",
                           "P0[R0+URZ],R2"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  cta_execution_state cta(1);
  ASSERT_TRUE(cta.initialize_mbarrier(0x100, 1));
  ASSERT_TRUE(cta.arrive_mbarrier(0x100, 1, 0));
  warp_state state;
  state.active_mask = 0x1;
  state.write_register(0, 0, 0x100);
  state.write_register(0, 2, 0);
  execution_context context;
  context.cta = &cta;
  context.warp_id = 0;

  const step_result result =
      sass.step("completed_mbarrier_predicate", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_EQ(state.mbarrier_try_wait,
            mbarrier_try_wait_state::kFunctionalComplete);
  EXPECT_EQ(state.pending_mbarrier_try_wait_mask, 0u);

  EXPECT_EQ(state.complete_mbarrier_try_wait(),
            mbarrier_try_wait_state::kFunctionalComplete);
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_EQ(state.mbarrier_try_wait, mbarrier_try_wait_state::kNone);

  ASSERT_TRUE(cta.initialize_mbarrier(0x108, 1));
  for (bool different_address : {true, false}) {
    state.pc = 0;
    state.active_mask = 3;
    state.write_register(1, 0, different_address ? 0x108 : 0x100);
    state.write_register(1, 2, different_address ? 0 : 0x80000000u);
    ASSERT_EQ(sass.step("completed_mbarrier_predicate", state, context).status,
              step_status::kAdvanced);
    EXPECT_TRUE(state.read_predicate(0, 0));
    EXPECT_FALSE(state.read_predicate(1, 0));
    ASSERT_TRUE(context.mbarrier_effects[0].valid);
    ASSERT_TRUE(context.mbarrier_effects[1].valid);
    timing_instruction projected(*sass.fetch("completed_mbarrier_predicate", 0));
    EXPECT_DEATH(
        flash_gpgpu_sim::sass::project_mbarrier_effects(context, projected),
        "nonuniform SASS mbarrier try-wait");
  }
}

TEST(SassCtaExecutorTest, NamedBarrierArrivalDoesNotBlockTheArrivingWarp) {
  kernel image;
  image.name = "cta_barrier_arrive";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.ARV", "0x2,0x40", "@P0"),
      official_instruction(0x10, "BAR.ARV", "0x1,0x40"),
      official_instruction(0x20, "BAR.SYNC.DEFER_BLOCKING", "R4,0x40"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_execution_state cta(4);
  warp_state arriving_warp;
  execution_context arriving_context;
  arriving_context.cta = &cta;
  arriving_context.warp_id = 0;

  step_result result =
      sass.step("cta_barrier_arrive", arriving_warp, arriving_context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(arriving_warp.pc, 0x10u);
  EXPECT_EQ(cta.generation(2), 0u);
  EXPECT_FALSE(arriving_context.named_barrier_effect.valid);
  result = sass.step("cta_barrier_arrive", arriving_warp, arriving_context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(arriving_warp.pc, 0x20u);
  EXPECT_FALSE(arriving_warp.waiting_at_cta_barrier);
  EXPECT_EQ(cta.generation(1), 0u);
  EXPECT_TRUE(arriving_context.named_barrier_effect.valid);
  EXPECT_EQ(arriving_context.named_barrier_effect.id, 1u);
  EXPECT_EQ(arriving_context.named_barrier_effect.participant_count, 0x40u);

  warp_state synchronizing_warp;
  synchronizing_warp.pc = 0x20;
  for (unsigned lane = 0; lane < 32; ++lane)
    synchronizing_warp.write_register(lane, 4, 1);
  execution_context synchronizing_context;
  synchronizing_context.cta = &cta;
  synchronizing_context.warp_id = 0;
  result = sass.step("cta_barrier_arrive", synchronizing_warp,
                     synchronizing_context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(synchronizing_warp.pc, 0x30u);
  EXPECT_FALSE(synchronizing_warp.waiting_at_cta_barrier);
  EXPECT_EQ(cta.generation(1), 1u);
  EXPECT_TRUE(synchronizing_context.named_barrier_effect.valid);
  EXPECT_EQ(synchronizing_context.named_barrier_effect.id, 1u);
  EXPECT_EQ(synchronizing_context.named_barrier_effect.participant_count,
            0x40u);
}

TEST(SassCtaExecutorTest,
     NamedBarrierCanReceiveRepeatedArrivalsAfterOtherWarpsRetire) {
  kernel image;
  image.name = "repeated_named_barrier_arrivals";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BAR.ARV", "0x1,0x60"),
      official_instruction(0x10, "BAR.ARV", "0x1,0x60"),
      official_instruction(0x20, "BAR.ARV", "0x1,0x60"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_execution_state cta(4);
  cta.retire_warp(2);
  cta.retire_warp(3);
  warp_state state;
  execution_context context;
  context.cta = &cta;
  context.warp_id = 0;

  for (unsigned arrival = 0; arrival < 3; ++arrival) {
    const step_result result =
        sass.step("repeated_named_barrier_arrivals", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(cta.generation(1), arrival == 2 ? 1u : 0u);
  }
}

TEST(SassCtaExecutorTest,
     RetiringArrivedWarpsDoesNotReleaseCurrentBarrierGenerationEarly) {
  cta_execution_state cta(4);

  EXPECT_FALSE(cta.arrive(0, 0));
  EXPECT_FALSE(cta.arrive(0, 1));
  EXPECT_FALSE(cta.arrive(0, 2));
  EXPECT_EQ(cta.generation(0), 0u);

  cta.retire_warp(0);
  cta.retire_warp(1);
  cta.retire_warp(2);
  EXPECT_EQ(cta.generation(0), 0u);

  EXPECT_TRUE(cta.arrive(0, 3));
  EXPECT_EQ(cta.generation(0), 1u);
}

TEST(SassCtaExecutorTest,
     RetiringUnarrivedWarpCompletesImplicitBarrierGeneration) {
  cta_execution_state cta(2);

  EXPECT_FALSE(cta.arrive(0, 0));
  cta.retire_warp(1);
  EXPECT_EQ(cta.generation(0), 1u);
}

TEST(SassCtaExecutorTest, RetiredWarpStopsParticipatingInNamedBarriers) {
  kernel image;
  image.name = "cta_barrier_retirement";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "S2R", "R0,SR_TID.X"),
      official_instruction(0x10, "SHF.R.U32.HI", "R0,RZ,0x5,R0"),
      official_instruction(0x20, "BAR.SYNC.DEFER_BLOCKING", "0x0"),
      official_instruction(0x30, "ISETP.NE.U32.AND", "P0,PT,R0,RZ,PT"),
      official_instruction(0x40, "EXIT", {}, "@!P0"),
      official_instruction(0x50, "BAR.SYNC.DEFER_BLOCKING", "0x0"),
      official_instruction(0x60, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "cta_barrier_retirement", 2);

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(cta.instructions_executed(), 20u);
  } while (result.status != step_status::kExited);

  EXPECT_EQ(cta.cta_state().generation(0), 2u);
  EXPECT_EQ(cta.instructions_executed(), 12u);
}

TEST(SassCtaExecutorTest, InitializesOpaqueSharedMemoryBarrierState) {
  kernel image;
  image.name = "mbarrier_init";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR4,0x1ffffe"),
      official_instruction(0x10, "UMOV", "UR5,0x7ffff800"),
      official_instruction(0x20, "UMOV", "UR8,0x100"),
      official_instruction(0x30, "SYNCS.EXCH.64", "UR10[UR8],UR4"),
      official_instruction(0x40, "MOV", "R0,0x40"),
      official_instruction(0x50, "SYNCS.ARRIVE.TRANS64", "RZ[UR8],R0", "@P0"),
      official_instruction(0x60, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  constexpr uint64_t original = 0x123456789abcdef0ull;
  memory.initialize(memory_space::kShared, 0x100, original);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "mbarrier_init", 1, &memory);
  cta.warp(0).write_predicate(0, 0, true);

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kShared, 0x100),
            0x7ffff800001ffffeull);
  EXPECT_EQ(cta.warp(0).read_uniform_register(10),
            static_cast<uint32_t>(original));
  EXPECT_EQ(cta.warp(0).read_uniform_register(11),
            static_cast<uint32_t>(original >> 32));
  EXPECT_TRUE(cta.cta_state().mbarrier_initialized(0x100));
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 0u);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_transaction_bytes(0x100), 64u);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 0u);
}

TEST(SassCtaExecutorTest, ExecutesArrivalAndArrivalCountModesPerLane) {
  kernel image;
  image.name = "mbarrier_arrival_modes";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.ARRIVE.TRANS64.A1T0", "RZ[R0+URZ],RZ",
                           "@P0"),
      official_instruction(0x10, "SYNCS.ARRIVE.TRANS64.ART0", "RZ[R0+URZ],R1",
                           "@P1"),
      official_instruction(0x20, "SYNCS.ARRIVE.TRANS64.RED.A1T0",
                           "RZ[R0+URZ],RZ", "@P0"),
      official_instruction(0x30, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "mbarrier_arrival_modes", 1, &memory);
  ASSERT_TRUE(cta.cta_state().initialize_mbarrier(0x100, 5));
  cta.warp(0).active_mask = 0x3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    cta.warp(0).write_register(lane, 0, 0x100);
    cta.warp(0).write_register(lane, 1, 3);
    cta.warp(0).write_predicate(lane, 0, true);
  }
  cta.warp(0).write_predicate(0, 1, true);

  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 3u);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 5u);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 1u);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 3u);
}

TEST(SassCtaExecutorTest, PreservesZeroByteArriveExpectTxForTiming) {
  kernel image;
  image.name = "mbarrier_zero_byte_expect_tx";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.ARRIVE.TRANS64", "RZ[R0+URZ],R1",
                           "@P0"),
      official_instruction(0x10, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "mbarrier_zero_byte_expect_tx", 1, &memory);
  ASSERT_TRUE(cta.cta_state().initialize_mbarrier(0x100, 1));
  cta.warp(0).active_mask = 1;
  cta.warp(0).write_register(0, 0, 0x100);
  cta.warp(0).write_register(0, 1, 0);
  cta.warp(0).write_predicate(0, 0, true);

  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 1u);
  ASSERT_TRUE(cta.context(0).mbarrier_effects[0].valid);
  EXPECT_EQ(cta.context(0).mbarrier_effects[0].count, 0u);
}

TEST(SassCtaExecutorTest, CompletesSynchronousLdgstsBarrierArrivals) {
  kernel image;
  image.name = "ldgsts_mbarrier_arrivals";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.ARRIVE.TRANS64.RED.A0T1", "RZ[UR4],RZ"),
      official_instruction(0x10, "ARRIVES.LDGSTSBAR.64.TRANSCNT", "[UR4]"),
      official_instruction(0x20, "ARRIVES.LDGSTSBAR.64.ARVCNT", "[UR4]"),
      official_instruction(0x30, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_executor cta(sass, "ldgsts_mbarrier_arrivals", 1, &memory);
  ASSERT_TRUE(cta.cta_state().initialize_mbarrier(0x100, 2));
  cta.warp(0).active_mask = 0x3;
  cta.warp(0).write_uniform_register(4, 0x100);

  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 2u);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 2u);
  ASSERT_TRUE(cta.context(0).mbarrier_effects[0].valid);
  EXPECT_EQ(cta.context(0).mbarrier_effects[0].address, 0x100u);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 1u);
  ASSERT_TRUE(cta.context(0).mbarrier_effects[0].valid);
  EXPECT_EQ(cta.context(0).mbarrier_effects[0].address, 0x100u);
  timing_instruction projected(*sass.fetch("ldgsts_mbarrier_arrivals", 0x20));
  flash_gpgpu_sim::sass::project_mbarrier_effects(cta.context(0), projected);
  EXPECT_EQ(projected.get_mbarrier_info(0).bar_id, 0x100u);
  EXPECT_EQ(projected.get_mbarrier_info(1).bar_id, 0x100u);
}

TEST(SassCtaExecutorTest, CompletesMbarrierTransactionBytes) {
  kernel image;
  image.name = "mbarrier_complete_tx";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SYNCS.ARRIVE.TRANS64", "RZ[R0+URZ],R1",
                           "@P0"),
      official_instruction(0x10, "SYNCS.ARRIVE.TRANS64.RED.A0TX",
                           "RZ[R0+URZ],R1", "@P0"),
      official_instruction(0x20, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_executor cta(sass, "mbarrier_complete_tx", 1, &memory);
  ASSERT_TRUE(cta.cta_state().initialize_mbarrier(0x100, 1));
  cta.warp(0).active_mask = 1;
  cta.warp(0).write_register(0, 0, 0x100);
  cta.warp(0).write_register(0, 1, 64);
  cta.warp(0).write_predicate(0, 0, true);

  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 0u);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_transaction_bytes(0x100), 64u);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 1u);
  EXPECT_EQ(cta.cta_state().mbarrier_pending_transaction_bytes(0x100), 0u);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 1u);
  ASSERT_TRUE(cta.context(0).mbarrier_effects[0].valid);
  EXPECT_EQ(cta.context(0).mbarrier_effects[0].address, 0x100u);
  EXPECT_EQ(cta.context(0).mbarrier_effects[0].count, 64u);
}

TEST(SassCtaExecutorTest, ImportsOpaqueBarrierStateWrittenToSharedMemory) {
  kernel image;
  image.name = "mbarrier_import";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "SYNCS.CCTL.IVALL"),
      official_instruction(0x10, "SYNCS.ARRIVE.TRANS64.A1T0", "RZ[R0+URZ],RZ",
                           "@P0"),
      official_instruction(0x20, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, sizeof(uint64_t), true);
  memory.initialize<uint64_t>(memory_space::kShared, 0x100,
                              0x7ffff000001ffffcull);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "mbarrier_import", 1, &memory);
  cta.warp(0).active_mask = 1;
  cta.warp(0).write_register(0, 0, 0x100);
  cta.warp(0).write_predicate(0, 0, true);

  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  ASSERT_EQ(cta.step().status, step_status::kAdvanced);
  EXPECT_TRUE(cta.cta_state().mbarrier_initialized(0x100));
  EXPECT_EQ(cta.cta_state().mbarrier_pending_arrivals(0x100), 1u);
}

TEST(SassCtaExecutorTest, LoadsRegisteredTwoDimensionalTensorMap) {
  kernel image;
  image.name = "tma_load_2d";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR24,0x1ffffe"),
      official_instruction(0x10, "UMOV", "UR25,0x7ffff800"),
      official_instruction(0x20, "UMOV", "UR9,0x100"),
      official_instruction(0x30, "SYNCS.EXCH.64", "UR20[UR9],UR24"),
      official_instruction(0x40, "MOV", "R0,0x8"),
      official_instruction(0x50, "SYNCS.ARRIVE.TRANS64", "RZ[UR9],R0", "@P0"),
      official_instruction(0x60, "UMOV", "UR8,0x200"),
      official_instruction(0x70, "UMOV", "UR10,0x1"),
      official_instruction(0x80, "UMOV", "UR11,0x1"),
      official_instruction(0x90, "UMOV", "UR4,0x3000"),
      official_instruction(0xa0, "UTMAPF.L2.4D", "[UR8][UR4]"),
      official_instruction(0xb0, "UTMALDG.2D", "[UR8][UR4],desc[UR28]"),
      official_instruction(0xc0, "MOV", "R7,0x100"),
      official_instruction(0xd0, "MOV", "R6,RZ"),
      official_instruction(0xe0, "SYNCS.PHASECHK.TRANS64.TRYWAIT",
                           "P1[R7+URZ],R6"),
      official_instruction(0xf0, "SYNCS.CCTL.IV", "[UR9]", "@P0"),
      official_instruction(0x100, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1000, 4 * 4 * sizeof(uint16_t), false);
  memory.map(memory_space::kShared, 0x100, 0x200, true);
  for (uint16_t value = 0; value < 16; ++value)
    memory.initialize(memory_space::kGlobal, 0x1000 + value * sizeof(uint16_t),
                      value);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "tma_load_2d", 1, &memory);
  functional_tensor_map_2d tensor_map;
  tensor_map.global_address = 0x1000;
  tensor_map.element_bytes = sizeof(uint16_t);
  tensor_map.global_dim = {{4, 4}};
  tensor_map.box_dim = {{2, 2}};
  tensor_map.row_stride_bytes = 4 * sizeof(uint16_t);
  ASSERT_TRUE(cta.cta_state().register_tensor_map_2d(0x3000, tensor_map));
  cta.warp(0).active_mask = 0xf;
  cta.warp(0).write_predicate(0, 0, true);
  cta.warp(0).write_uniform_register(28, 0);
  cta.warp(0).write_uniform_register(29, 0x10000000);

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kShared, 0x200), 5u);
  EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kShared, 0x202), 6u);
  EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kShared, 0x204), 9u);
  EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kShared, 0x206), 10u);
  EXPECT_EQ(memory.global_reads, 4u);
  EXPECT_EQ(memory.shared_writes, 2u);
  EXPECT_FALSE(cta.cta_state().mbarrier_initialized(0x100));
  EXPECT_TRUE(cta.warp(0).read_predicate(0, 1));
}

TEST(SassCtaExecutorTest, StoresRegisteredTwoDimensionalTensorMap) {
  for (const auto arch : {architecture::kSm90, architecture::kSm120}) {
    SCOPED_TRACE(arch == architecture::kSm90 ? "sm90" : "sm120");
    kernel image;
    image.name = "tma_store_2d";
    image.arch = arch;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0x00, "UMOV", "UR8,0x200"),
        official_instruction(0x10, "UMOV", "UR9,0x1"),
        official_instruction(0x20, "UMOV", "UR10,0x1"),
        official_instruction(0x30, "UMOV", "UR4,0x3000"),
        official_instruction(0x40, "BAR.SYNC.DEFER_BLOCKING", "0x0,0x40"),
        official_instruction(0x50, "UTMASTG.2D", "[UR8][UR4]"),
        official_instruction(0x60, "EXIT"),
    };
    mapped_functional_memory memory;
    memory.map(memory_space::kGlobal, 0x1000, 4 * 4 * sizeof(uint16_t), true);
    memory.map(memory_space::kShared, 0x200, 4 * sizeof(uint16_t), true);
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm90_functional_semantics(sass);
    cta_execution_state cta(2);
    functional_tensor_map_2d tensor_map;
    tensor_map.global_address = 0x1000;
    tensor_map.element_bytes = sizeof(uint16_t);
    tensor_map.global_dim = {{4, 4}};
    tensor_map.box_dim = {{2, 2}};
    tensor_map.row_stride_bytes = 4 * sizeof(uint16_t);
    ASSERT_TRUE(cta.register_tensor_map_2d(0x3000, tensor_map));
    warp_state store_warp, peer;
    execution_context context(&memory);
    context.cta = &cta;
    context.warp_id = 0;
    for (unsigned i = 0; i < 5; ++i)
      ASSERT_EQ(sass.step("tma_store_2d", store_warp, context).status,
                step_status::kAdvanced);
    ASSERT_TRUE(store_warp.deferred_cta_barrier_pending[0]);
    // FA3 publishes O only after all consumer warps have replaced the old
    // shared buffer contents. TMA store must not snapshot those old bytes.
    EXPECT_EQ(sass.step("tma_store_2d", store_warp, context).status,
              step_status::kBlocked);
    EXPECT_EQ(store_warp.pc, 0x50u);
    EXPECT_EQ(memory.shared_reads, 0u);
    EXPECT_EQ(memory.global_writes, 0u);
    const std::array<uint16_t, 4> values{{5, 6, 9, 10}};
    for (size_t i = 0; i < values.size(); ++i)
      memory.initialize(memory_space::kShared, 0x200 + i * sizeof(uint16_t),
                        values[i]);
    execution_context peer_context(&memory);
    peer_context.cta = &cta;
    peer_context.warp_id = 1;
    peer.pc = 0x40;
    ASSERT_EQ(sass.step("tma_store_2d", peer, peer_context).status,
              step_status::kAdvanced);
    ASSERT_EQ(sass.step("tma_store_2d", store_warp, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kGlobal, 0x100a), 5u);
    EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kGlobal, 0x100c), 6u);
    EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kGlobal, 0x1012), 9u);
    EXPECT_EQ(memory.inspect<uint16_t>(memory_space::kGlobal, 0x1014), 10u);
    EXPECT_EQ(memory.shared_reads, 1u);
    EXPECT_EQ(memory.global_writes, 4u);
  }
}

TEST(SassCtaExecutorTest, StoresRegisteredOneDimensionalTensorMap) {
  kernel image;
  image.name = "tma_store_1d";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR8,0x200"),
      official_instruction(0x10, "UMOV", "UR9,0x2"),
      official_instruction(0x20, "UMOV", "UR4,0x3000"),
      official_instruction(0x30, "UTMASTG.1D", "[UR8][UR4]"),
      official_instruction(0x40, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1000, 8 * sizeof(uint32_t), true);
  memory.map(memory_space::kShared, 0x200, 4 * sizeof(uint32_t), false);
  const std::array<uint32_t, 4> values{{5, 6, 7, 8}};
  for (size_t index = 0; index < values.size(); ++index)
    memory.initialize(memory_space::kShared, 0x200 + index * sizeof(uint32_t),
                      values[index]);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  cta_executor cta(sass, "tma_store_1d", 1, &memory);
  functional_tensor_map_2d tensor_map;
  tensor_map.global_address = 0x1000;
  tensor_map.element_bytes = sizeof(uint32_t);
  tensor_map.global_dim = {{8, 1}};
  tensor_map.box_dim = {{4, 1}};
  tensor_map.row_stride_bytes = 0;
  ASSERT_TRUE(cta.cta_state().register_tensor_map_2d(0x3000, tensor_map));

  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (size_t index = 0; index < values.size(); ++index)
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal,
                                       0x1008 + index * sizeof(uint32_t)),
              values[index]);
  EXPECT_EQ(memory.shared_reads, 1u);
  EXPECT_EQ(memory.global_writes, 4u);
}

TEST(SassCtaExecutorTest, StoresCompactFourDimensionalTensorMap) {
  kernel image;
  image.name = "tma_store_4d";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UMOV", "UR6,0x3000"),
      official_instruction(0x10, "UMOV", "UR7,URZ"),
      official_instruction(0x20, "UMOV", "UR8,0x100"),
      official_instruction(0x30, "UMOV", "UR9,URZ"),
      official_instruction(0x40, "UMOV", "UR10,URZ"),
      official_instruction(0x50, "UMOV", "UR11,URZ"),
      official_instruction(0x60, "UMOV", "UR12,URZ"),
      official_instruction(0x70, "UTMASTG.4D", "[UR8][UR6]"),
      official_instruction(0x80, "EXIT"),
  };

  constexpr uint64_t global_base = 0x0000000c00000000ull;
  constexpr size_t elements = 8 * 8 * 8 * 8;
  const std::array<uint64_t, 8> descriptor{{
      global_base,
      0x00000002000003b0ull,
      0x0000008000000010ull,
      0,
      0x0000000700000007ull,
      0x0000000700000007ull,
      0x0700000000000000ull,
      0x0000000000070707ull,
  }};
  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x3000, sizeof(descriptor), false);
  memory.map(memory_space::kGlobal, global_base, elements * sizeof(float),
             true);
  memory.map(memory_space::kShared, 0x100, elements * sizeof(float), false);
  memory.initialize(memory_space::kGlobal, 0x3000, descriptor);
  for (size_t index = 0; index < elements; ++index)
    memory.initialize<float>(memory_space::kShared,
                             0x100 + index * sizeof(float),
                             static_cast<float>(index));

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_executor cta(sass, "tma_store_4d", 1, &memory);
  step_result result;
  do {
    result = cta.step();
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (size_t index = 0; index < elements; ++index)
    EXPECT_EQ(memory.inspect<float>(memory_space::kGlobal,
                                    global_base + index * sizeof(float)),
              static_cast<float>(index));
}

TEST(SassCtaExecutorTest, CopiesUniformBulkTileFromGlobalToShared) {
  kernel image;
  image.name = "uniform_bulk_load";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UBLKCP.S.G", "[UR4][UR6],UR8"),
      official_instruction(0x10, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 0x1000, 64, false);
  memory.map(memory_space::kShared, 0x100, 0x200, true);
  for (uint32_t index = 0; index < 16; ++index)
    memory.initialize(memory_space::kGlobal, 0x1000 + index * sizeof(uint32_t),
                      0x200u + index);

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  cta_executor cta(sass, "uniform_bulk_load", 1, &memory);
  ASSERT_TRUE(cta.cta_state().initialize_mbarrier(0x100, 1));
  ASSERT_TRUE(cta.cta_state().arrive_mbarrier(0x100, 1, 64));
  cta.warp(0).write_uniform_register(4, 0x200);
  cta.warp(0).write_uniform_register(5, 0x100);
  cta.warp(0).write_uniform_register(6, 0x1000);
  cta.warp(0).write_uniform_register(7, 0);
  cta.warp(0).write_uniform_register(8, 4);

  const step_result result = cta.step();
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (uint32_t index = 0; index < 16; ++index)
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared,
                                       0x200 + index * sizeof(uint32_t)),
              0x200u + index);
  ASSERT_TRUE(cta.context(0).tma_effects[0].valid);
  EXPECT_EQ(cta.context(0).tma_effects[0].destination_address, 0x200u);
  EXPECT_EQ(cta.context(0).tma_effects[0].source_address, 0x1000u);
  EXPECT_EQ(cta.context(0).tma_effects[0].size_bytes, 64u);
  EXPECT_EQ(cta.context(0).tma_effects[0].mbarrier_address, 0x100u);
  EXPECT_EQ(cta.cta_state().mbarrier_phase(0x100), 1u);
}

}  // namespace
