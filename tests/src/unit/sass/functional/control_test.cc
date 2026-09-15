#include "../test_support.h"

namespace {

TEST(SassFunctionalTest,
     ExecutesFoundationalIntegerBranchAndSharedMemorySlice) {
  kernel image;
  image.name = "foundation";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x000, "S2R", "R0,SR_LANEID"),
      official_instruction(0x010, "MOV", "R1,R0"),
      official_instruction(0x020, "SHF.L.U32", "R0,R0,0x2,RZ"),
      official_instruction(0x030, "IADD3", "R0,PT,PT,R0,0x100,RZ"),
      official_instruction(0x040, "IADD3", "R1,PT,PT,R1,0x20,RZ"),
      official_instruction(0x050, "STS", "[R0],R1"),
      official_instruction(0x060, "LDS", "R2[R0]"),
      official_instruction(0x070, "UMOV", "UR4,0x100"),
      official_instruction(0x080, "UMOV", "UR5,0x1"),
      official_instruction(0x090, "UIADD3", "UR6,UPT,UPT,UR4,0x40,URZ"),
      official_instruction(0x0a0, "MOV", "R3,UR6"),
      official_instruction(0x0b0, "R2UR", "UR7,R3"),
      official_instruction(0x0c0, "S2UR", "UR8,SR_CTAID.X"),
      official_instruction(0x0d0, "MOV.64", "R10,UR4"),
      official_instruction(0x0e0, "CS2R", "R12,SRZ"),
      official_instruction(0x0f0, "NOP"),
      official_instruction(0x100, "BRA", "0x120"),
      official_instruction(0x110, "MOV", "R2,0xdeadbeef"),
      official_instruction(0x120, "EXIT"),
  };

  mapped_functional_memory memory;
  memory.map(memory_space::kShared, 0x100, 32 * sizeof(uint32_t), true);
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context{&memory};
  context.cta_id_x = 7;

  step_result result;
  do {
    result = sass.step("foundation", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 32u);
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 2), lane + 0x20);
    EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kShared,
                                       0x100 + lane * sizeof(uint32_t)),
              lane + 0x20);
    EXPECT_EQ(state.read_register(lane, 3), 0x140u);
    EXPECT_EQ(state.read_register(lane, 10), 0x100u);
    EXPECT_EQ(state.read_register(lane, 11), 0x1u);
    EXPECT_EQ(state.read_register(lane, 12), 0u);
    EXPECT_EQ(state.read_register(lane, 13), 0u);
  }
  EXPECT_EQ(state.read_uniform_register(6), 0x140u);
  EXPECT_EQ(state.read_uniform_register(7), 0x140u);
  EXPECT_EQ(state.read_uniform_register(8), 7u);
  EXPECT_EQ(memory.shared_reads, 32u);
  EXPECT_EQ(memory.shared_writes, 32u);
  EXPECT_EQ(context.instructions_executed, 18u);
}

TEST(SassFunctionalTest, ExecutesRelativeHelperCallAndRegisterReturn) {
  instruction call =
      official_instruction(0, "CALL.REL.NOINC", "$__internal_helper");
  call.attributes.push_back({"target-pc", "0x30"});
  kernel image;
  image.name = "relative_call";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      std::move(call),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x30, "MOV", "R4,0x1234"),
      official_instruction(0x40, "RET.REL.NODEC", "R2,relative_call"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  state.write_register(0, 2, 0x10);
  state.write_register(1, 2, 0x10);
  execution_context context;

  step_result result;
  do {
    result = sass.step("relative_call", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_EQ(state.read_register(0, 4), 0x1234u);
  EXPECT_EQ(state.read_register(1, 4), 0x1234u);
  EXPECT_EQ(context.instructions_executed, 4u);
}

TEST(SassFunctionalTest, CombinesExitGuardAndOperandPredicate) {
  kernel image;
  image.name = "predicated_exit";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "EXIT", "P1", "@P0"),
      official_instruction(0x10, "MOV", "R0,0x2a"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x7;
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, true);
  state.write_predicate(0, 1, true);
  state.write_predicate(2, 1, true);
  execution_context context;

  step_result result;
  do {
    result = sass.step("predicated_exit", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 0), 0u);
  EXPECT_EQ(state.read_register(1, 0), 42u);
  EXPECT_EQ(state.read_register(2, 0), 42u);
}

TEST(SassFunctionalTest, SerializesDivergentExitPathsWithoutBssy) {
  kernel image;
  image.name = "independent_exit_paths";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA", "0x30", "@P0"),
      official_instruction(0x10, "MOV", "R1,0x11"),
      official_instruction(0x20, "EXIT"),
      official_instruction(0x30, "MOV", "R1,0x22"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, false);
  execution_context context;
  step_result result;
  do {
    result = sass.step("independent_exit_paths", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_EQ(state.read_register(0, 1), 0x22u);
  EXPECT_EQ(state.read_register(1, 1), 0x11u);
  EXPECT_TRUE(state.independent_paths.empty());
  EXPECT_EQ(context.instructions_executed, 5u);
}

TEST(SassFunctionalTest, CombinesBraGuardAndOperandPredicate) {
  kernel image;
  image.name = "dual_predicate_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA", "P1,0x20", "@P0"),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  for (unsigned guard = 0; guard < 2; ++guard) {
    for (unsigned condition = 0; condition < 2; ++condition) {
      warp_state state;
      state.active_mask = 1;
      state.write_predicate(0, 0, guard != 0);
      state.write_predicate(0, 1, condition != 0);
      execution_context context;
      const step_result result =
          sass.step("dual_predicate_branch", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      EXPECT_EQ(state.pc, guard != 0 && condition != 0 ? 0x20u : 0x10u);
    }
  }
}

TEST(SassFunctionalTest, BranchesUniformlyWhenAnyActiveLaneIsGuarded) {
  kernel image;
  image.name = "uniform_any_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA.U.ANY", "0x20", "@P0"),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  for (unsigned matching_lane = 0; matching_lane < 3; ++matching_lane) {
    warp_state state;
    state.active_mask = 0x3;
    if (matching_lane < 2) state.write_predicate(matching_lane, 0, true);
    execution_context context;
    const step_result result = sass.step("uniform_any_branch", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.pc, matching_lane < 2 ? 0x20u : 0x10u);
  }
}

TEST(SassFunctionalTest, BranchesUniformlyOnLanePredicateGuard) {
  kernel image;
  image.name = "uniform_guard_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA.U", "0x20", "@P0"),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  for (unsigned predicate = 0; predicate < 2; ++predicate) {
    warp_state state;
    state.active_mask = 0x3;
    state.write_predicate(0, 0, predicate != 0);
    state.write_predicate(1, 0, predicate != 0);
    execution_context context;
    const step_result result =
        sass.step("uniform_guard_branch", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.pc, predicate != 0 ? 0x20u : 0x10u);
  }
}

TEST(SassFunctionalTest, BranchesUniformlyOnUniformPredicateOperand) {
  kernel image;
  image.name = "uniform_predicate_any_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA.U.ANY", "UP0,0x20"),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  for (unsigned predicate = 0; predicate < 2; ++predicate) {
    warp_state state;
    state.active_mask = 0x3;
    state.write_uniform_predicate(0, predicate != 0);
    execution_context context;
    const step_result result =
        sass.step("uniform_predicate_any_branch", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.pc, predicate != 0 ? 0x20u : 0x10u);
  }
}

TEST(SassFunctionalTest, ExecutesUniformRegisterIndirectBranch) {
  kernel image;
  image.name = "uniform_indirect_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRXU", "UR4,-0x10"),
      official_instruction(0x10, "MOV", "R0,0x11"),
      official_instruction(0x20, "MOV", "R0,0x22"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(4, 0x20);
  state.write_uniform_register(5, 0);
  execution_context context;

  ASSERT_EQ(sass.step("uniform_indirect_branch", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.pc, 0x20u);
  ASSERT_EQ(sass.step("uniform_indirect_branch", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 0), 0x22u);
}

TEST(SassFunctionalTest, ExecutesLaneRegisterIndirectBranch) {
  kernel image;
  image.name = "lane_indirect_branch";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRX", "R4,-0x10"),
      official_instruction(0x10, "MOV", "R0,0x11"),
      official_instruction(0x20, "MOV", "R0,0x22"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 4, 0x20);
    state.write_register(lane, 5, 0);
  }
  execution_context context;

  ASSERT_EQ(sass.step("lane_indirect_branch", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.pc, 0x20u);
  ASSERT_EQ(sass.step("lane_indirect_branch", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 0), 0x22u);
  EXPECT_EQ(state.read_register(1, 0), 0x22u);
}

TEST(SassFunctionalTest, ProducesVoteuBallotMask) {
  kernel image;
  image.name = "voteu_ballot";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "VOTEU.ANY", "UR4,UPT,P0"),
      official_instruction(0x10, "POPC", "R0,UR4"),
      official_instruction(0x20, "UFLO.U32", "UR5,UR4"),
      official_instruction(0x30, "FLO.U32", "R1,UR4"),
      official_instruction(0x40, "ULEA.HI.SX32", "UR6,UR7,UR8,0x1b"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0xf;
  state.write_predicate(0, 0, true);
  state.write_predicate(2, 0, true);
  state.write_uniform_register(7, 0xfffffff0u);
  state.write_uniform_register(8, 9u);
  execution_context context;

  step_result result;
  do {
    result = sass.step("voteu_ballot", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_EQ(state.read_uniform_register(4), 0x5u);
  EXPECT_EQ(state.read_uniform_register(5), 2u);
  EXPECT_EQ(state.read_uniform_register(6), 8u);
  for (unsigned lane = 0; lane < 4; ++lane)
    EXPECT_EQ(state.read_register(lane, 0), 2u);
  for (unsigned lane = 0; lane < 4; ++lane)
    EXPECT_EQ(state.read_register(lane, 1), 2u);
}

TEST(SassFunctionalTest, SelectsConvergenceBranchesFromActiveMask) {
  kernel image;
  image.name = "convergence_branches";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "BRA.DIV", "UR4,0x20"),
      official_instruction(0x10, "EXIT"),
      official_instruction(0x20, "BRA.CONV", "UR4,0x50"),
      official_instruction(0x30, "WARPSYNC.COLLECTIVE", "R0,0x50"),
      official_instruction(0x40, "ENDCOLLECTIVE"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state divergent;
  divergent.active_mask = 1;
  divergent.write_uniform_register(4, 0xffffffffu);
  execution_context divergent_context;
  ASSERT_EQ(
      sass.step("convergence_branches", divergent, divergent_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(divergent.pc, 0x20u);
  ASSERT_EQ(
      sass.step("convergence_branches", divergent, divergent_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(divergent.pc, 0x30u);
  ASSERT_EQ(
      sass.step("convergence_branches", divergent, divergent_context).status,
      step_status::kAdvanced);
  ASSERT_EQ(
      sass.step("convergence_branches", divergent, divergent_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(divergent.pc, 0x50u);

  warp_state converged;
  converged.pc = 0x20;
  converged.write_uniform_register(4, 0xffffffffu);
  execution_context converged_context;
  ASSERT_EQ(
      sass.step("convergence_branches", converged, converged_context).status,
      step_status::kAdvanced);
  EXPECT_EQ(converged.pc, 0x50u);
}

TEST(SassFunctionalTest, ReconvergesLeaderOnlyPathAtWarpSync) {
  kernel image;
  image.name = "warpsync_reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BRA", "0x40", "@P0"),
      official_instruction(0x10, "MOV", "R1,0x2a"),
      official_instruction(0x20, "BRA", "0x40"),
      official_instruction(0x30, "NOP"),
      official_instruction(0x40, "WARPSYNC.ALL"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("warpsync_reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 16u);
  } while (result.status != step_status::kExited);

  EXPECT_TRUE(state.warp_sync_stack.empty());
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 0u : 42u);
}

TEST(SassFunctionalTest, HandlesUniformlyPredicatedWarpSync) {
  kernel image;
  image.name = "predicated_warpsync";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "WARPSYNC.ALL", "", "@P0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  for (bool predicate : {false, true}) {
    warp_state state;
    state.active_mask = 0x3;
    state.write_predicate(0, 0, predicate);
    state.write_predicate(1, 0, predicate);
    execution_context context;
    const step_result result = sass.step("predicated_warpsync", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.pc, 0x10u);
    EXPECT_EQ(state.active_mask, 0x3u);
  }
}

TEST(SassFunctionalTest, ReconvergesDeferredForwardPathAtWarpSync) {
  kernel image;
  image.name = "deferred_warpsync_reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BRA", "0x30", "@P0"),
      official_instruction(0x10, "MOV", "R1,0x11"),
      official_instruction(0x20, "BRA", "0x50"),
      official_instruction(0x30, "MOV", "R1,0x22"),
      official_instruction(0x40, "NOP"),
      official_instruction(0x50, "WARPSYNC.ALL"),
      official_instruction(0x60, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, false);
  execution_context context;

  step_result result;
  do {
    result = sass.step("deferred_warpsync_reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 16u);
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 1), 0x22u);
  EXPECT_EQ(state.read_register(1, 1), 0x11u);
  EXPECT_TRUE(state.independent_paths.empty());
}

TEST(SassFunctionalTest, ReconvergesBssyPathsBeforeWarpCollective) {
  kernel image;
  image.name = "bssy_warpsync_reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BSSY", "B0,0xb0"),
      official_instruction(0x10, "BRA", "0x50", "@P0"),
      official_instruction(0x20, "MOV", "R1,0x11"),
      official_instruction(0x30, "BRA", "0x60"),
      official_instruction(0x40, "NOP"),
      official_instruction(0x50, "MOV", "R1,0x22"),
      official_instruction(0x60, "WARPSYNC.ALL"),
      official_instruction(0x70, "ELECT", "P1,URZ,PT"),
      official_instruction(0x80, "BRA", "0xa0", "@!P1"),
      official_instruction(0x90, "MOV", "R2,0x2a"),
      official_instruction(0xa0, "BSYNC", "B0"),
      official_instruction(0xb0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("bssy_warpsync_reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 32u);
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 0x22u : 0x11u);
    EXPECT_EQ(state.read_register(lane, 2), lane == 0 ? 0x2au : 0u);
  }
  EXPECT_TRUE(state.reconvergence_stack.empty());
  EXPECT_TRUE(state.independent_paths.empty());
}

TEST(SassFunctionalTest, SerializesDivergentPathsAndReconvergesAtBsync) {
  kernel image;
  image.name = "reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BSSY", "B0,0x80"),
      official_instruction(0x10, "BRA", "0x50", "@P0"),
      official_instruction(0x20, "MOV", "R1,0x11"),
      official_instruction(0x30, "BRA", "0x70"),
      official_instruction(0x40, "NOP"),
      official_instruction(0x50, "MOV", "R1,0x22"),
      official_instruction(0x60, "BRA", "0x70"),
      official_instruction(0x70, "BSYNC", "B0"),
      official_instruction(0x80, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 32u);
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 0x22u : 0x11u);
  EXPECT_TRUE(state.reconvergence_stack.empty());
  EXPECT_FALSE(state.reconvergence[0].valid);
  EXPECT_EQ(context.instructions_executed, 9u);
}

TEST(SassFunctionalTest, EscapesReliableRegionAndReconvergesAtOuterBsync) {
  kernel image;
  image.name = "reliable_reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BSSY.RECONVERGENT", "B0,0xc0"),
      official_instruction(0x10, "BSSY.RELIABLE", "B1,0x80"),
      official_instruction(0x20, "BREAK.RELIABLE", "B1", "@P0"),
      official_instruction(0x30, "BRA", "0xa0", "@P0"),
      official_instruction(0x40, "MOV", "R1,0x5"),
      official_instruction(0x50, "BSYNC.RELIABLE", "B1"),
      official_instruction(0x80, "MOV", "R2,0x7"),
      official_instruction(0x90, "BSYNC.RECONVERGENT", "B0"),
      official_instruction(0xa0, "MOV", "R1,0x9"),
      official_instruction(0xb0, "BSYNC.RECONVERGENT", "B0"),
      official_instruction(0xc0, "WARPSYNC.ALL"),
      official_instruction(0xd0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("reliable_reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 32u);
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 9u : 5u);
    EXPECT_EQ(state.read_register(lane, 2), lane % 2 == 0 ? 0u : 7u);
  }
  EXPECT_TRUE(state.reconvergence_stack.empty());
  EXPECT_EQ(context.instructions_executed, 13u);
}

TEST(SassFunctionalTest, RetiresReliableRegionWhenEveryActiveLaneBreaks) {
  kernel image;
  image.name = "all_active_reliable_break";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BSSY.RECONVERGENT", "B0,0xc0"),
      official_instruction(0x10, "BSSY.RELIABLE", "B1,0x80"),
      official_instruction(0x20, "BRA", "0x50", "@P0"),
      official_instruction(0x30, "MOV", "R1,0x1"),
      official_instruction(0x40, "BSYNC.RELIABLE", "B1"),
      official_instruction(0x50, "BREAK.RELIABLE", "B1", "@PT"),
      official_instruction(0x60, "MOV", "R2,0x2"),
      official_instruction(0x70, "BRA", "0xa0"),
      official_instruction(0x80, "MOV", "R3,0x3"),
      official_instruction(0x90, "BRA", "0xa0"),
      official_instruction(0xa0, "BSYNC.RECONVERGENT", "B0"),
      official_instruction(0xb0, "NOP"),
      official_instruction(0xc0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("all_active_reliable_break", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 16u);
  } while (result.status != step_status::kExited);

  EXPECT_EQ(context.instructions_executed, 13u);
  EXPECT_TRUE(state.reconvergence_stack.empty());
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 0u : 1u);
    EXPECT_EQ(state.read_register(lane, 2), lane % 2 == 0 ? 2u : 0u);
    EXPECT_EQ(state.read_register(lane, 3), lane % 2 == 0 ? 0u : 3u);
  }
}

TEST(SassFunctionalTest, EscapesOrdinaryRegionAndReconvergesAtOuterBsync) {
  kernel image;
  image.name = "ordinary_break_reconvergence";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "BSSY", "B0,0xc0"),
      official_instruction(0x10, "BSSY", "B1,0x80"),
      official_instruction(0x20, "BREAK", "B1", "@P0"),
      official_instruction(0x30, "BRA", "0xa0", "@P0"),
      official_instruction(0x40, "MOV", "R1,0x5"),
      official_instruction(0x50, "BSYNC", "B1"),
      official_instruction(0x80, "MOV", "R2,0x7"),
      official_instruction(0x90, "BSYNC", "B0"),
      official_instruction(0xa0, "MOV", "R1,0x9"),
      official_instruction(0xb0, "BSYNC", "B0"),
      official_instruction(0xc0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 0, lane % 2 == 0);
  execution_context context;

  step_result result;
  do {
    result = sass.step("ordinary_break_reconvergence", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    ASSERT_LT(context.instructions_executed, 32u);
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), lane % 2 == 0 ? 9u : 5u);
    EXPECT_EQ(state.read_register(lane, 2), lane % 2 == 0 ? 0u : 7u);
  }
  EXPECT_TRUE(state.reconvergence_stack.empty());
}

TEST(SassFunctionalTest, ElectsOneActiveLaneAndReportsItsLaneId) {
  kernel image;
  image.name = "elect";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ELECT", "P0,UR4,PT"),
      official_instruction(0x10, "ELECT", "P2,UR5,PT", "@P1"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x8000000au;
  for (unsigned lane = 0; lane < 32; ++lane)
    state.write_predicate(lane, 2, true);
  state.write_predicate(3, 1, true);
  state.write_predicate(31, 1, true);
  execution_context context;

  ASSERT_EQ(sass.step("elect", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(4), 1u);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(state.read_predicate(lane, 0), lane == 1);
  ASSERT_EQ(sass.step("elect", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(5), 3u);
  EXPECT_TRUE(state.read_predicate(3, 2));
  EXPECT_FALSE(state.read_predicate(31, 2));
  EXPECT_TRUE(state.read_predicate(1, 2));
  EXPECT_EQ(sass.step("elect", state, context).status, step_status::kExited);
}

TEST(SassFunctionalTest, VotesActivePredicatesIntoUniformPredicates) {
  kernel image;
  image.name = "voteu";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "VOTEU.ALL", "UP0,P0"),
      official_instruction(0x10, "VOTEU.ANY", "UP1,P0"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x0000000fu;
  state.write_predicate(0, 0, true);
  state.write_predicate(2, 0, true);
  execution_context context;

  step_result result;
  do {
    result = sass.step("voteu", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_FALSE(state.read_uniform_predicate(0));
  EXPECT_TRUE(state.read_uniform_predicate(1));
}

}  // namespace
