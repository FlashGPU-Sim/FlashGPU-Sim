#include "../test_support.h"

namespace {

TEST(SassFunctionalTest, PacksSelectedUniformPredicatesIntoARegister) {
  kernel image;
  image.name = "uniform_predicate_pack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UP2UR", "UR45,UPR,UR4,0x3"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(4, 0xa2u);
  state.write_uniform_predicate(0, true);
  state.write_uniform_predicate(1, false);
  execution_context context;
  const step_result result =
      sass.step("uniform_predicate_pack", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.read_uniform_register(45), 0xa1u);
}

TEST(SassFunctionalTest, ExecutesSm90CommonSemantics) {
  kernel image;
  image.name = "sm90_common";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R2,0x2a"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  execution_context context;

  ASSERT_EQ(sass.step("sm90_common", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 2), 42u);
  EXPECT_EQ(sass.step("sm90_common", state, context).status,
            step_status::kExited);
}

TEST(SassFunctionalTest, ExecutesHopperScalarPrologueForms) {
  kernel image;
  image.name = "sm90_prologue";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ULDC", "UR8,c[0][0x0]"),
      official_instruction(0x10, "ULDC.64", "UR10,c[0][0x8]"),
      official_instruction(0x20, "UIADD3", "UR9,UR8,0x2000,URZ"),
      official_instruction(0x30, "IMAD.MOV.U32", "R2,RZ,RZ,R0"),
      official_instruction(0x40, "VIADD", "R3,R2,UR8"),
      official_instruction(0x50, "IMAD.IADD", "R4,R3,0x1,-R2"),
      official_instruction(0x60, "IADD3", "R5,R3,R4,-R2"),
      official_instruction(0x70, "EXIT"),
  };
  mapped_functional_memory memory;
  memory.map(memory_space::kConstant, 0, 16, false);
  memory.initialize(memory_space::kConstant, 0, uint32_t{5});
  memory.initialize(memory_space::kConstant, 8,
                    uint64_t{0x0123456789abcdefull});
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 7);
  execution_context context{&memory};

  step_result result;
  do {
    result = sass.step("sm90_prologue", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_uniform_register(8), 5u);
  EXPECT_EQ(state.read_uniform_register(9), 0x2005u);
  EXPECT_EQ(state.read_uniform_register(10), 0x89abcdefu);
  EXPECT_EQ(state.read_uniform_register(11), 0x01234567u);
  EXPECT_EQ(state.read_register(0, 2), 7u);
  EXPECT_EQ(state.read_register(0, 3), 12u);
  EXPECT_EQ(state.read_register(0, 4), 5u);
  EXPECT_EQ(state.read_register(0, 5), 10u);
}

TEST(SassFunctionalTest, PermutesBytesAndSupportsSignReplication) {
  kernel image;
  image.name = "byte_permute";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "PRMT", "R2,R0,0x5410,R1"),
      official_instruction(0x10, "PRMT", "R3,R0,0xba98,R1"),
      official_instruction(0x20, "UPRMT", "UR8,URZ,0x654,UR7"),
      official_instruction(0x30, "EXIT"),
  };

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0x00ff7f80u);
  state.write_register(0, 1, 0x88776655u);
  state.write_uniform_register(7, 0x11223344u);
  execution_context context;

  step_result result;
  do {
    result = sass.step("byte_permute", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_register(0, 2), 0x66557f80u);
  EXPECT_EQ(state.read_register(0, 3), 0x00ff00ffu);
  EXPECT_EQ(state.read_uniform_register(8), 0x00223344u);
}

TEST(SassFunctionalTest, ReadsFunctionalProfilingSpecialRegisters) {
  kernel image;
  image.name = "profiling_special_registers";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "CS2R", "R0,SR_GLOBALTIMERLO"),
      official_instruction(0x10, "NOP"),
      official_instruction(0x20, "CS2R", "R2,SR_GLOBALTIMERLO"),
      official_instruction(0x30, "S2R", "R4,SR_VIRTUALSMID"),
      official_instruction(0x40, "S2R", "R5,SR_VIRTID"),
      official_instruction(0x50, "CS2R", "R6,SR_CLOCKLO"),
      official_instruction(0x60, "CS2UR", "UR8,SR_CLOCKLO"),
      official_instruction(0x70, "CS2UR.32", "UR10,SR_GLOBALTIMERLO"),
      official_instruction(0x80, "S2UR", "UR12,SR_CLOCKLO"),
      official_instruction(0x90, "CS2R.32", "R14,SR_CLOCKLO"),
      official_instruction(0xa0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  // Functional fallback, timed conversion, and timed execution without Hz.
  for (unsigned mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    warp_state state;
    execution_context context;
    context.warp_id = 3;
    context.virtual_smid = 17;
    context.architectural_cycle = mode == 2 ? 0x100000002ull : 1234;
    context.architectural_core_frequency_hz = mode == 2 ? 0 : 2000000000ull;
    context.architectural_cycle_valid = mode != 0;
    const uint64_t cycle = context.architectural_cycle;
    const uint64_t timer = mode == 2 ? cycle : 617;
    state.write_uniform_register(11, 0xbeef);
    for (unsigned lane = 0; lane < 32; ++lane)
      state.write_register(lane, 15, 0xbeef);
    step_result result;
    do {
      result = sass.step("profiling_special_registers", state, context);
      ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
      ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
    } while (result.status != step_status::kExited);

    for (unsigned lane = 0; lane < 32; ++lane) {
      const uint64_t start = state.read_register(lane, 0) |
                             (uint64_t{state.read_register(lane, 1)} << 32);
      const uint64_t end = state.read_register(lane, 2) |
                           (uint64_t{state.read_register(lane, 3)} << 32);
      EXPECT_EQ(start, mode == 0 ? 0 : timer);
      EXPECT_EQ(end, mode == 0 ? 2 : timer);
      EXPECT_EQ(state.read_register(lane, 4), 17u);
      EXPECT_EQ(state.read_register(lane, 5), 3u << 8);
      EXPECT_EQ(state.read_register(lane, 6), uint32_t(mode == 0 ? 5 : cycle));
      EXPECT_EQ(state.read_register(lane, 7), mode == 0 ? 0 : cycle >> 32);
      EXPECT_EQ(state.read_register(lane, 14), uint32_t(mode == 0 ? 9 : cycle));
      EXPECT_EQ(state.read_register(lane, 15), 0xbeefu);
    }
    EXPECT_EQ(state.read_uniform_register(8), uint32_t(mode == 0 ? 6 : cycle));
    EXPECT_EQ(state.read_uniform_register(9), mode == 0 ? 0 : cycle >> 32);
    EXPECT_EQ(state.read_uniform_register(10), uint32_t(mode == 0 ? 7 : timer));
    EXPECT_EQ(state.read_uniform_register(11), 0xbeefu);
    EXPECT_EQ(state.read_uniform_register(12), uint32_t(mode == 0 ? 8 : cycle));
  }
}

TEST(SassFunctionalTest, RejectsModifiersNotImplementedByScalarReaders) {
  const std::pair<const char *, const char *> cases[] = {
      {"IADD", "R0,|R1|,1"},
      {"IADD", "R0,R1.ROW,1"},
      {"IMAD.WIDE.U32", "R0,R2,R3,|R4|"},
      {"FADD", "R0,R1.H0_H0,R2"},
      {"SEL", "R0,R1,R2,|P0|"},
  };
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.second);
    kernel image;
    image.name = "unsupported_scalar_modifier";
    image.arch = architecture::kSm120;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(0, entry.first, entry.second)};
    ASSERT_TRUE(image.instructions[0].operands_structured);
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, 0x12345678);
    execution_context context;
    EXPECT_EQ(sass.step("unsupported_scalar_modifier", state, context).status,
              step_status::kUnsupported);
    EXPECT_EQ(state.pc, 0u);
    EXPECT_EQ(state.read_register(0, 0), 0x12345678u);
  }
}

TEST(SassFunctionalTest, ExecutesFunctionalCtaRegisterPoolHints) {
  kernel image;
  image.name = "cta_register_pool_hints";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "USETMAXREG.TRY_ALLOC.CTAPOOL", "UP0,0xa0"),
      official_instruction(0x10, "USETMAXREG.DEALLOC.CTAPOOL", "0x20"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("cta_register_pool_hints", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_TRUE(state.read_uniform_predicate(0));
}

TEST(SassFunctionalTest, ExecutesGuardedRegisterToUniformMove) {
  kernel image;
  image.name = "guarded_r2ur";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "R2UR", "UR7,R3", "@P0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0xf;
  state.write_uniform_register(7, 0xdeadbeefu);
  for (unsigned lane = 0; lane < 4; ++lane) {
    state.write_predicate(lane, 0, (lane & 1u) == 0);
    state.write_register(lane, 3, (lane & 1u) == 0 ? 0x1234u : lane);
  }
  execution_context context;

  step_result result;
  do {
    result = sass.step("guarded_r2ur", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_EQ(state.read_uniform_register(7), 0x1234u);
}

TEST(SassFunctionalTest, ExecutesIntegerPredicateShiftAndMultiplyForms) {
  kernel image;
  image.name = "integer_forms";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x000, "MOV", "R0,0xfffffff8"),
      official_instruction(0x010, "IABS", "R1,R0"),
      official_instruction(0x020, "SHF.R.S32.HI", "R2,RZ,0x1f,R0"),
      official_instruction(0x030, "SHF.L.U32", "R3,R1,0x2,RZ"),
      official_instruction(0x040, "IMAD.SHL.U32", "R4,R1,0x8,R3"),
      official_instruction(0x050, "MOV", "R5,0x80000000"),
      official_instruction(0x060, "MOV", "R6,0x2"),
      official_instruction(0x070, "IMAD.HI.U32", "R7,R5,R6,0x1"),
      official_instruction(0x080, "IMAD.WIDE.U32", "R8,R5,R6,RZ"),
      official_instruction(0x090, "LEA.HI", "R10,R2,R0,RZ,0x1"),
      official_instruction(0x0a0, "UMOV", "UR4,0x3"),
      official_instruction(0x0b0, "UMOV", "UR5,0x5"),
      official_instruction(0x0c0, "ULEA", "UR6,UR4,UR5,0x2"),
      official_instruction(0x0d0, "UIMAD", "UR7,UR4,UR5,UR6"),
      official_instruction(0x0e0, "UMOV", "UR9,0xfffffff8"),
      official_instruction(0x0f0, "USHF.R.S32.HI", "UR8,URZ,0x1f,UR9"),
      official_instruction(0x100, "ISETP.LT.AND", "P0,PT,R0,RZ,PT"),
      official_instruction(0x110, "ISETP.LT.U32.AND", "P1,PT,R0,RZ,PT"),
      official_instruction(0x120, "ISETP.GT.U32.AND", "P2,PT,R4,0x5f,PT"),
      official_instruction(0x130, "ISETP.GT.AND", "P3,PT,R1,0x7,PT"),
      official_instruction(0x140, "USHF.R.U32.HI", "UR10,URZ,0x4,0x80000000"),
      official_instruction(0x150, "ULOP3.LUT", "UR11,UR4,UR5,URZ,0xc0,!UPT"),
      official_instruction(0x160, "UIMAD.WIDE.U32", "UR12,UR4,UR5,URZ"),
      official_instruction(0x170, "USGXT.U32", "UR14,UR5,0x3"),
      official_instruction(0x180, "SGXT.U32", "R11,R0,0x3"),
      official_instruction(0x190, "UMOV", "UR15,0xffffffff"),
      official_instruction(0x1a0, "UMOV", "UR16,0x1"),
      official_instruction(0x1b0, "ULEA", "UR17,UP0,UR15,UR16,0x1"),
      official_instruction(0x1c0, "ULEA.HI.X", "UR18,UR15,UR4,UR5,0x1,UP0"),
      official_instruction(0x1d0, "LOP3.LUT", "P4,RZ,R0,0x8,RZ,0xc0,!PT"),
      official_instruction(0x1e0, "IADD.64", "R12,R8,UR12"),
      official_instruction(0x1f0, "MOV", "R20,0xfffffffe"),
      official_instruction(0x200, "MOV", "R21,0x40000000"),
      official_instruction(0x210, "IMAD.HI", "R22,R20,R21,0x1"),
      official_instruction(0x220, "IMAD.HI.U32", "R23,R20,R21,0x1"),
      official_instruction(0x230, "ULEA.HI", "UR20,UR15,UR16,URZ,0x1"),
      official_instruction(0x240, "UIMAD.HI.U32", "UR21,0x80000000,0x2,URZ"),
      official_instruction(0x250, "MOV", "R24,0x5"),
      official_instruction(0x260, "IMAD.MOV", "R25,RZ,RZ,-R24"),
      official_instruction(0x270, "ULOP3.LUT",
                           "UP2,UR22,UR5,0x3,URZ,0xc0,!UPT"),
      official_instruction(0x280, "ULEA.HI.X.SX32", "UR23,UR15,UR4,0x1,UP0"),
      official_instruction(0x290, "MOV", "R26,0x1"),
      official_instruction(0x2a0, "SGXT", "R27,R26,0x1"),
      official_instruction(0x2b0, "MOV", "R28,0x2"),
      official_instruction(0x2c0, "SGXT", "R29,R28,0x2"),
      // Carry must use the old source when destination and source overlap.
      official_instruction(0x2d0, "MOV", "R30,0x80000000"),
      official_instruction(0x2e0, "LEA", "R30,P5,R30,RZ,0x1"),
      official_instruction(0x2f0, "UMOV", "UR24,0x80000000"),
      official_instruction(0x300, "ULEA", "UR24,UP3,UR24,URZ,0x1"),
      official_instruction(0x310, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("integer_forms", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 8u);
    EXPECT_EQ(state.read_register(lane, 2), 0xffffffffu);
    EXPECT_EQ(state.read_register(lane, 3), 32u);
    EXPECT_EQ(state.read_register(lane, 4), 96u);
    EXPECT_EQ(state.read_register(lane, 7), 1u);
    EXPECT_EQ(state.read_register(lane, 8), 0u);
    EXPECT_EQ(state.read_register(lane, 9), 1u);
    EXPECT_EQ(state.read_register(lane, 10), 0xfffffff9u);
    EXPECT_EQ(state.read_register(lane, 11), 0u);
    EXPECT_EQ(state.read_register(lane, 12), 15u);
    EXPECT_EQ(state.read_register(lane, 13), 1u);
    EXPECT_EQ(state.read_register(lane, 22), 0xffffffffu);
    EXPECT_EQ(state.read_register(lane, 23), 0x3fffffffu);
    EXPECT_EQ(state.read_register(lane, 25), 0xfffffffbu);
    EXPECT_EQ(state.read_register(lane, 27), 0xffffffffu);
    EXPECT_EQ(state.read_register(lane, 29), 0xfffffffeu);
    EXPECT_EQ(state.read_register(lane, 30), 0u);
    EXPECT_TRUE(state.read_predicate(lane, 5));
    EXPECT_TRUE(state.read_predicate(lane, 0));
    EXPECT_FALSE(state.read_predicate(lane, 1));
    EXPECT_TRUE(state.read_predicate(lane, 2));
    EXPECT_TRUE(state.read_predicate(lane, 3));
    EXPECT_TRUE(state.read_predicate(lane, 4));
  }
  EXPECT_EQ(state.read_uniform_register(6), 17u);
  EXPECT_EQ(state.read_uniform_register(7), 32u);
  EXPECT_EQ(state.read_uniform_register(8), 0xffffffffu);
  EXPECT_EQ(state.read_uniform_register(10), 0x08000000u);
  EXPECT_EQ(state.read_uniform_register(11), 1u);
  EXPECT_EQ(state.read_uniform_register(12), 15u);
  EXPECT_EQ(state.read_uniform_register(13), 0u);
  EXPECT_EQ(state.read_uniform_register(14), 5u);
  EXPECT_EQ(state.read_uniform_register(17), 0xffffffffu);
  EXPECT_EQ(state.read_uniform_register(18), 10u);
  EXPECT_EQ(state.read_uniform_register(20), 2u);
  EXPECT_EQ(state.read_uniform_register(21), 1u);
  EXPECT_EQ(state.read_uniform_register(22), 1u);
  EXPECT_EQ(state.read_uniform_register(23), 3u);
  EXPECT_EQ(state.read_uniform_register(24), 0u);
  EXPECT_TRUE(state.read_uniform_predicate(3));
  EXPECT_TRUE(state.read_uniform_predicate(0));
  EXPECT_TRUE(state.read_uniform_predicate(2));
}

TEST(SassFunctionalTest, ExecutesRegisterSelectedFunnelShifts) {
  kernel image;
  image.name = "register_selected_funnel_shifts";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0x80000000"),
      official_instruction(0x10, "MOV", "R1,0x4"),
      official_instruction(0x20, "SHF.R.U32.HI", "R2,RZ,R1,R0"),
      official_instruction(0x30, "UMOV", "UR0,0x80000000"),
      official_instruction(0x40, "UMOV", "UR1,0x24"),
      official_instruction(0x50, "USHF.R.U32.HI", "UR2,URZ,UR1,UR0"),
      official_instruction(0x60, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("register_selected_funnel_shifts", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(state.read_register(lane, 2), 0x08000000u);
  EXPECT_EQ(state.read_uniform_register(2), 0x08000000u);
}

TEST(SassFunctionalTest, CombinesIntegerComparisonWithPredicateOr) {
  kernel image;
  image.name = "isetp_or";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.GE.OR", "P0,PT,R0,UR4,P1"),
      official_instruction(0x10, "ISETP.GT.U32.OR", "P2,PT,R2,0x1f,!P1"),
      official_instruction(0x20, "ISETP.LT.OR", "P3,PT,R0,0x4,P1"),
      official_instruction(0x30, "ISETP.GT.OR", "P4,PT,R4,RZ,P1"),
      official_instruction(0x40, "ISETP.EQ.OR", "P5,PT,R0,0x5,!P1"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  for (unsigned lane = 0; lane < 2; ++lane) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, lane == 0 ? 0 : 5);
    state.write_register(0, 2, lane == 0 ? 32 : 0);
    state.write_register(0, 4, 0xffffffffu);
    state.write_uniform_register(4, 4);
    state.write_predicate(0, 1, lane == 0);
    execution_context context;
    const step_result result = sass.step("isetp_or", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_TRUE(state.read_predicate(0, 0));
    ASSERT_EQ(sass.step("isetp_or", state, context).status,
              step_status::kAdvanced);
    EXPECT_TRUE(state.read_predicate(0, 2));
    ASSERT_EQ(sass.step("isetp_or", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 3), lane == 0);
    ASSERT_EQ(sass.step("isetp_or", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 4), lane == 0);
    ASSERT_EQ(sass.step("isetp_or", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 5), lane == 1);
  }
}

TEST(SassFunctionalTest, CombinesIntegerComparisonWithPredicateXor) {
  kernel image;
  image.name = "isetp_xor";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ISETP.NE.XOR", "P1,PT,R2,0x80,!P0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0xf;
  state.write_register(0, 2, 0x80);
  state.write_register(1, 2, 0x81);
  state.write_register(2, 2, 0x80);
  state.write_register(3, 2, 0x81);
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, true);
  state.write_predicate(2, 0, false);
  state.write_predicate(3, 0, false);
  execution_context context;

  const step_result result = sass.step("isetp_xor", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_FALSE(state.read_predicate(0, 1));
  EXPECT_TRUE(state.read_predicate(1, 1));
  EXPECT_TRUE(state.read_predicate(2, 1));
  EXPECT_FALSE(state.read_predicate(3, 1));
}

TEST(SassFunctionalTest, ExtendsIntegerNotEqualAcrossRegisterPairs) {
  kernel image;
  image.name = "isetp_ne_extended";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,RZ"),
      official_instruction(0x10, "MOV", "R1,0x1"),
      official_instruction(0x20, "ISETP.NE.U32.AND", "P0,PT,R0,RZ,PT"),
      official_instruction(0x30, "ISETP.NE.U32.AND.EX", "P0,PT,R1,RZ,PT,P0"),
      official_instruction(0x40, "MOV", "R2,0x1"),
      official_instruction(0x50, "MOV", "R3,RZ"),
      official_instruction(0x60, "ISETP.NE.U32.AND", "P1,PT,R2,RZ,PT"),
      official_instruction(0x70, "ISETP.NE.AND.EX", "P1,PT,R3,RZ,PT,P1"),
      official_instruction(0x80, "MOV", "R4,RZ"),
      official_instruction(0x90, "MOV", "R5,RZ"),
      official_instruction(0xa0, "ISETP.NE.U32.AND", "P2,PT,R4,RZ,PT"),
      official_instruction(0xb0, "ISETP.NE.AND.EX", "P2,PT,R5,RZ,PT,P2"),
      official_instruction(0xc0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("isetp_ne_extended", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_TRUE(state.read_predicate(lane, 0));
    EXPECT_TRUE(state.read_predicate(lane, 1));
    EXPECT_FALSE(state.read_predicate(lane, 2));
  }
}

TEST(SassFunctionalTest, ExtendsIntegerOrderingAcrossRegisterPairs) {
  kernel image;
  image.name = "isetp_order_extended";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "ISETP.GT.U32.AND", "P0,PT,R0,R2,PT"),
      official_instruction(0x10, "ISETP.GT.U32.AND.EX", "P0,PT,R1,R3,PT,P0"),
      official_instruction(0x20, "ISETP.GE.U32.AND", "P1,PT,R4,R6,PT"),
      official_instruction(0x30, "ISETP.GE.U32.AND.EX", "P1,PT,R5,R7,PT,P1"),
      official_instruction(0x40, "ISETP.LT.U32.AND", "P2,PT,R8,R10,PT"),
      official_instruction(0x50, "ISETP.LT.U32.OR.EX", "P2,PT,R9,R11,P5,P2"),
      official_instruction(0x60, "ISETP.EQ.U32.AND", "P3,PT,R12,R14,PT"),
      official_instruction(0x70, "ISETP.EQ.OR.EX", "P3,PT,R13,R15,P5,P3"),
      official_instruction(0x80, "EXIT"),
  };

  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0x00000000u);
  state.write_register(0, 1, 0x00000002u);
  state.write_register(0, 2, 0xffffffffu);
  state.write_register(0, 3, 0x00000001u);
  state.write_register(0, 4, 0x00000000u);
  state.write_register(0, 5, 0x00000001u);
  state.write_register(0, 6, 0x00000001u);
  state.write_register(0, 7, 0x00000001u);
  state.write_register(0, 8, 0x00000000u);
  state.write_register(0, 9, 0x00000001u);
  state.write_register(0, 10, 0x00000001u);
  state.write_register(0, 11, 0x00000001u);
  state.write_register(0, 12, 0x00000007u);
  state.write_register(0, 13, 0x00000004u);
  state.write_register(0, 14, 0x00000007u);
  state.write_register(0, 15, 0x00000004u);
  state.write_predicate(0, 5, false);
  execution_context context;

  step_result result;
  do {
    result = sass.step("isetp_order_extended", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_FALSE(state.read_predicate(0, 1));
  EXPECT_TRUE(state.read_predicate(0, 2));
  EXPECT_TRUE(state.read_predicate(0, 3));
}

TEST(SassFunctionalTest, ExtendsUniformIntegerNotEqualAcrossRegisterPairs) {
  kernel image;
  image.name = "uisetp_ne_extended";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UISETP.NE.AND.EX", "UP2,UPT,UR0,URZ,UPT,UP0"),
      official_instruction(0x10, "UISETP.NE.AND.EX", "UP3,UPT,UR1,URZ,UPT,UP1"),
      official_instruction(0x20, "UISETP.NE.AND.EX", "UP4,UPT,UR0,URZ,UPT,UP1"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 0);
  state.write_uniform_register(1, 1);
  state.write_uniform_predicate(0, true);
  state.write_uniform_predicate(1, false);
  execution_context context;
  step_result result;
  do {
    result = sass.step("uisetp_ne_extended", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_TRUE(state.read_uniform_predicate(2));
  EXPECT_TRUE(state.read_uniform_predicate(3));
  EXPECT_FALSE(state.read_uniform_predicate(4));
}

TEST(SassFunctionalTest, ExtendsUniformIntegerEqualityAndCombinesWithOr) {
  kernel image;
  image.name = "uisetp_eq_or_extended";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "UISETP.EQ.OR.EX", "UP2,UPT,UR0,UR1,UP0,UP1"),
      official_instruction(0x10, "UISETP.EQ.OR.EX", "UP3,UPT,UR0,UR1,UP1,UP0"),
      official_instruction(0x20, "UISETP.EQ.OR.EX", "UP4,UPT,UR0,UR1,UP0,UP0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 7);
  state.write_uniform_register(1, 7);
  state.write_uniform_predicate(0, false);
  state.write_uniform_predicate(1, true);
  execution_context context;
  step_result result;
  do {
    result = sass.step("uisetp_eq_or_extended", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_TRUE(state.read_uniform_predicate(2));
  EXPECT_TRUE(state.read_uniform_predicate(3));
  EXPECT_FALSE(state.read_uniform_predicate(4));
}

TEST(SassFunctionalTest, ProducesHopperIadd3CarryOutput) {
  kernel image;
  image.name = "hopper_iadd3_carry";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "IADD3", "R2,P2,R5,R6,RZ"),
      official_instruction(0x10, "LEA.HI.X.SX32", "R7,R6,R9,0x1,P2"),
      official_instruction(0x20, "IADD3.X", "R8,R7,UR7,RZ,P2,!PT"),
      official_instruction(0x30, "IADD3", "R10,P3,P4,R5,R9,0x2"),
      official_instruction(0x40, "IADD3.X", "R11,RZ,RZ,RZ,P3,P4"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  state.active_mask = 0x3;
  state.write_register(0, 5, 0xffffffffu);
  state.write_register(0, 6, 1u);
  state.write_register(0, 9, 0xffffffffu);
  state.write_register(1, 5, 1u);
  state.write_register(1, 6, 0xffffffffu);
  state.write_register(1, 9, 0u);
  state.write_uniform_register(7, 5u);
  execution_context context;
  const step_result result = sass.step("hopper_iadd3_carry", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.read_register(0, 2), 0u);
  EXPECT_TRUE(state.read_predicate(0, 2));
  EXPECT_EQ(state.read_register(1, 2), 0u);
  EXPECT_TRUE(state.read_predicate(1, 2));
  ASSERT_EQ(sass.step("hopper_iadd3_carry", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 7), 0u);
  EXPECT_EQ(state.read_register(1, 7), 0u);
  ASSERT_EQ(sass.step("hopper_iadd3_carry", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 8), 6u);
  EXPECT_EQ(state.read_register(1, 8), 6u);
  ASSERT_EQ(sass.step("hopper_iadd3_carry", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 10), 0u);
  EXPECT_TRUE(state.read_predicate(0, 3));
  EXPECT_TRUE(state.read_predicate(0, 4));
  ASSERT_EQ(sass.step("hopper_iadd3_carry", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 11), 2u);
}

TEST(SassFunctionalTest, ConsumesHopperImadCarryInput) {
  kernel image;
  image.name = "hopper_imad_carry";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "IMAD.X", "R0,R1,R2,R3,P0"),
      official_instruction(0x10, "IMAD.X", "R4,R1,R2,~R3,P0"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0x3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 1, 2);
    state.write_register(lane, 2, 3);
    state.write_register(lane, 3, 0xffffffffu);
  }
  state.write_predicate(0, 0, true);
  execution_context context;
  const step_result result = sass.step("hopper_imad_carry", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.read_register(0, 0), 6u);
  EXPECT_EQ(state.read_register(1, 0), 5u);
  const step_result complemented =
      sass.step("hopper_imad_carry", state, context);
  ASSERT_EQ(complemented.status, step_status::kAdvanced) << complemented.detail;
  EXPECT_EQ(state.read_register(0, 4), 7u);
  EXPECT_EQ(state.read_register(1, 4), 6u);
}

TEST(SassFunctionalTest, ProducesHopperUniformIadd3CarryOutputs) {
  kernel image;
  image.name = "hopper_uiadd3_carry";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UIADD3", "UR2,UP0,UP1,UR0,UR1,0x2"),
      official_instruction(0x10, "UIADD3.X", "UR3,URZ,URZ,URZ,UP0,UP1"),
      official_instruction(0x20, "UIADD3", "UR4,UP2,UR0,0x1,URZ"),
      official_instruction(0x30, "UIADD3.X", "UR5,URZ,UR6,URZ,UP2,!UPT"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 0xffffffffu);
  state.write_uniform_register(1, 0xffffffffu);
  state.write_uniform_register(6, 5u);
  execution_context context;
  step_result result;
  do {
    result = sass.step("hopper_uiadd3_carry", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_uniform_register(2), 0u);
  EXPECT_TRUE(state.read_uniform_predicate(0));
  EXPECT_TRUE(state.read_uniform_predicate(1));
  EXPECT_EQ(state.read_uniform_register(3), 2u);
  EXPECT_EQ(state.read_uniform_register(4), 0u);
  EXPECT_TRUE(state.read_uniform_predicate(2));
  EXPECT_EQ(state.read_uniform_register(5), 6u);
}

TEST(SassFunctionalTest, ComparesUnsignedRegisterPairs) {
  kernel image;
  image.name = "isetp_u64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.GE.U64.AND", "P0,PT,R0,R2,PT"),
      official_instruction(0x10, "ISETP.EQ.U32.OR", "P1,PT,R0,RZ,P1"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 0x3;
  state.write_register(0, 0, 0);
  state.write_register(0, 1, 1);
  state.write_register(0, 2, 0xffffffffu);
  state.write_register(0, 3, 0);
  state.write_register(1, 0, 0xffffffffu);
  state.write_register(1, 1, 0);
  state.write_register(1, 2, 0);
  state.write_register(1, 3, 1);
  state.write_predicate(1, 1, true);

  execution_context context;
  step_result result;
  do {
    result = sass.step("isetp_u64", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_FALSE(state.read_predicate(1, 0));
  EXPECT_TRUE(state.read_predicate(0, 1));
  EXPECT_TRUE(state.read_predicate(1, 1));
}

TEST(SassFunctionalTest, ExecutesUniformCompareAndR2urStatusForms) {
  kernel image;
  image.name = "uniform_compare_r2ur";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UISETP.GE.U32.AND", "UP0,UPT,UR0,0x3,UPT"),
      official_instruction(0x10, "UISETP.GE.AND", "UP1,UPT,UR1,0x1,UPT"),
      official_instruction(0x20, "R2UR", "P0,UR4,R2"),
      official_instruction(0x30, "R2UR.OR", "P0,UR5,R3"),
      official_instruction(0x40, "R2UR.BROADCAST", "UR6,R2"),
      official_instruction(0x50, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 0x3;
  state.write_uniform_register(0, 3);
  state.write_uniform_register(1, 0xffffffffu);
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 2, 0x1234);
    state.write_register(lane, 3, 0x5678);
    state.write_predicate(lane, 0, true);
  }

  execution_context context;
  step_result result;
  do {
    result = sass.step("uniform_compare_r2ur", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_TRUE(state.read_uniform_predicate(0));
  EXPECT_FALSE(state.read_uniform_predicate(1));
  EXPECT_EQ(state.read_uniform_register(4), 0x1234u);
  EXPECT_EQ(state.read_uniform_register(5), 0x5678u);
  EXPECT_EQ(state.read_uniform_register(6), 0x1234u);
  EXPECT_FALSE(state.read_predicate(0, 0));
  EXPECT_FALSE(state.read_predicate(1, 0));
}

TEST(SassFunctionalTest, ExecutesUniformSignedAndWideComparisons) {
  kernel image;
  image.name = "uniform_compare_extended";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UISETP.NE.U64.AND", "UP0,UPT,UR0,UR2,UPT"),
      official_instruction(0x10, "UISETP.GT.U32.AND", "UP1,UPT,UR4,UR5,UPT"),
      official_instruction(0x20, "UISETP.GT.AND", "UP2,UPT,UR4,UR5,UPT"),
      official_instruction(0x30, "UISETP.NE.AND", "UP3,UPT,UR6,UR7,!UP1"),
      official_instruction(0x40, "UISETP.LT.AND", "UP4,UPT,UR4,UR5,UPT"),
      official_instruction(0x50, "UISETP.EQ.U32.AND", "UP5,UPT,UR6,0x9,UPT"),
      official_instruction(0x60, "USEL", "UR8,UR4,UR5,UP1"),
      official_instruction(0x70, "USEL.64", "UR10,UR0,UR2,UP0"),
      official_instruction(0x80, "UVIMNMX.S32", "UR12,UR4,UR5,UPT"),
      official_instruction(0x90, "UI2F.RP", "UR14,UR13"),
      official_instruction(0xa0, "UF2I.FTZ.U32.TRUNC.NTZ", "UR16,UR15"),
      official_instruction(0xb0, "UISETP.GE.U64.AND", "UP6,UPT,UR0,0xf80,UPT"),
      official_instruction(0xc0, "UVIMNMX.U32", "UR18,UR4,UR5,UPT"),
      official_instruction(0xd0, "UI2F.U32.RP", "UR19,UR4"),
      official_instruction(0xe0, "UISETP.EQ.U32.OR", "UP7,UPT,UR6,0x8,UP1"),
      official_instruction(0xf0, "UISETP.NE.U32.OR", "UP7,UPT,UR6,0x9,UP7"),
      official_instruction(0x100, "UISETP.GT.U32.OR", "UP7,UPT,UR6,0x8,!UP7"),
      official_instruction(0x110, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 0);
  state.write_uniform_register(1, 1);
  state.write_uniform_register(2, 0xffffffffu);
  state.write_uniform_register(3, 0);
  state.write_uniform_register(4, 0xffffffffu);
  state.write_uniform_register(5, 1);
  state.write_uniform_register(6, 9);
  state.write_uniform_register(7, 8);
  state.write_uniform_register(13, 0x01000001u);
  state.write_uniform_register(15, 0x40700000u);

  execution_context context;
  step_result result;
  do {
    result = sass.step("uniform_compare_extended", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_TRUE(state.read_uniform_predicate(0));
  EXPECT_TRUE(state.read_uniform_predicate(1));
  EXPECT_TRUE(state.read_uniform_predicate(6));
  EXPECT_FALSE(state.read_uniform_predicate(2));
  EXPECT_FALSE(state.read_uniform_predicate(3));
  EXPECT_TRUE(state.read_uniform_predicate(4));
  EXPECT_TRUE(state.read_uniform_predicate(5));
  EXPECT_TRUE(state.read_uniform_predicate(7));
  EXPECT_EQ(state.read_uniform_register(8), 0xffffffffu);
  EXPECT_EQ(state.read_uniform_register(10), 0u);
  EXPECT_EQ(state.read_uniform_register(11), 1u);
  EXPECT_EQ(state.read_uniform_register(12), 0xffffffffu);
  EXPECT_EQ(state.read_uniform_register(14), 0x4b800001u);
  EXPECT_EQ(state.read_uniform_register(16), 3u);
  EXPECT_EQ(state.read_uniform_register(18), 1u);
  EXPECT_EQ(state.read_uniform_register(19), 0x4f800000u);
}

TEST(SassFunctionalTest, ExecutesTwoOutputPredicateLut) {
  kernel image;
  image.name = "predicate_lut";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "PLOP3.LUT", "P0,PT,PT,PT,PT,0x80,0x8"),
      official_instruction(0x10, "PLOP3.LUT", "P1,P2,P0,PT,PT,0x8,0x80", "@P0"),
      official_instruction(0x20, "PLOP3.LUT", "P3,PT,PT,PT,UP0,0x80,0x8"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.write_uniform_predicate(0, true);
  execution_context context;

  ASSERT_EQ(sass.step("predicate_lut", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_TRUE(state.read_predicate(lane, 0));
  ASSERT_EQ(sass.step("predicate_lut", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_FALSE(state.read_predicate(lane, 1));
    EXPECT_TRUE(state.read_predicate(lane, 2));
  }
  ASSERT_EQ(sass.step("predicate_lut", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_TRUE(state.read_predicate(lane, 3));
}

TEST(SassFunctionalTest, ExecutesTwoOutputUniformPredicateLut) {
  kernel image;
  image.name = "uniform_predicate_lut";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UPLOP3.LUT", "UP0,UPT,UPT,UPT,UPT,0x80,0x8"),
      official_instruction(0x10, "UPLOP3.LUT",
                           "UP1,UP2,UP0,UPT,!UPT,0x80,0x40"),
      official_instruction(0x20, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  ASSERT_EQ(sass.step("uniform_predicate_lut", state, context).status,
            step_status::kAdvanced);
  EXPECT_TRUE(state.read_uniform_predicate(0));
  ASSERT_EQ(sass.step("uniform_predicate_lut", state, context).status,
            step_status::kAdvanced);
  EXPECT_FALSE(state.read_uniform_predicate(1));
  EXPECT_TRUE(state.read_uniform_predicate(2));
}

TEST(SassFunctionalTest, SelectsVectorRegisterPairs) {
  kernel image;
  image.name = "wide_vector_select";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "SEL.64", "R4,R0,R2,P0"),
      official_instruction(0x10, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 0x3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 0, 0x11111111u);
    state.write_register(lane, 1, 0x22222222u);
    state.write_register(lane, 2, 0x33333333u);
    state.write_register(lane, 3, 0x44444444u);
  }
  state.write_predicate(0, 0, true);
  state.write_predicate(1, 0, false);
  execution_context context;

  ASSERT_EQ(sass.step("wide_vector_select", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x11111111u);
  EXPECT_EQ(state.read_register(0, 5), 0x22222222u);
  EXPECT_EQ(state.read_register(1, 4), 0x33333333u);
  EXPECT_EQ(state.read_register(1, 5), 0x44444444u);
}

TEST(SassFunctionalTest, ExecutesUniformFtzFloatArithmetic) {
  kernel image;
  image.name = "uniform_ftz_float";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UFMUL.FTZ", "UR2,UR0,UR1"),
      official_instruction(0x10, "UFFMA.FTZ", "UR3,UR0,UR1,-UR2"),
      official_instruction(0x20, "UFMUL.FTZ", "UR4,UR5,UR0"),
      official_instruction(0x30, "UFADD.FTZ", "UR6,UR2,-UR1"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);

  warp_state state;
  state.active_mask = 1;
  state.write_uniform_register(0, 0x40000000u);
  state.write_uniform_register(1, 0x40400000u);
  state.write_uniform_register(5, 0x00000001u);
  execution_context context;
  step_result result;
  do {
    result = sass.step("uniform_ftz_float", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);

  EXPECT_EQ(state.read_uniform_register(2), 0x40c00000u);
  EXPECT_EQ(state.read_uniform_register(3), 0u);
  EXPECT_EQ(state.read_uniform_register(4), 0u);
  EXPECT_EQ(state.read_uniform_register(6), 0x40400000u);
}

TEST(SassFunctionalTest, ExecutesConversionMinMaxAndPredicateMovementForms) {
  kernel image;
  image.name = "conversion_forms";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x000, "MOV", "R0,0x1000001"),
      official_instruction(0x010, "I2F.RP", "R1,R0"),
      official_instruction(0x020, "MOV", "R2,0x40700000"),
      official_instruction(0x030, "MUFU.RCP", "R3,R2"),
      official_instruction(0x040, "F2I.FTZ.U32.TRUNC.NTZ", "R4,R2"),
      official_instruction(0x050, "MOV", "R5,0xfffffffd"),
      official_instruction(0x060, "VIMNMX", "R6,R5,0x8,PT"),
      official_instruction(0x070, "VIMNMX.S32", "R7,R5,0x8,!PT"),
      official_instruction(0x080, "ISETP.LT.AND", "P0,PT,R5,RZ,PT"),
      official_instruction(0x090, "MOV", "R8,0xa0"),
      official_instruction(0x0a0, "P2R", "R8,R8,0x1"),
      official_instruction(0x0b0, "SEL", "R9,R5,0x8,P0"),
      official_instruction(0x0c0, "SEL", "R10,R5,0x8,!P0"),
      official_instruction(0x0d0, "MOV", "R11,0xffffffc0"),
      official_instruction(0x0e0, "LEA.HI.SX32", "R12,R11,0x5,0x1a"),
      official_instruction(0x0f0, "MOV", "R13,0x3fc00000"),
      official_instruction(0x100, "MOV", "R14,0xc0100000"),
      official_instruction(0x110, "F2FP.F16.F32.PACK_AB", "R15,R13,R14"),
      official_instruction(0x120, "I2F.U32.RP", "R16,0x80000000"),
      official_instruction(0x130, "VIMNMX.U32", "R17,R5,0x8,!PT"),
      official_instruction(0x140, "I2F.U16", "R18,0xffff8001"),
      official_instruction(0x150, "I2FP.F32.U32.RZ", "R19,0x1000003"),
      official_instruction(0x160, "MOV", "R20,0x3f7ffffe"),
      official_instruction(0x170, "MOV", "R21,0x3f800001"),
      official_instruction(0x180, "FMUL.RZ", "R22,R20,R21"),
      official_instruction(0x190, "F2I.U32.TRUNC.NTZ", "R23,R2"),
      official_instruction(0x1a0, "MOV", "R24,0x5"),
      official_instruction(0x1b0, "VIADDMNMX", "R25,R24,0x8,0xa,PT"),
      official_instruction(0x1c0, "VIADDMNMX", "R26,R24,0x8,0xa,!PT"),
      official_instruction(0x1d0, "MUFU.RCP", "R27,-R13"),
      official_instruction(0x1e0, "F2I.U32.TRUNC.NTZ", "R28,-R14"),
      official_instruction(0x1f0, "F2FP.F16.F32.PACK_AB", "R29,-R13,-R14"),
      official_instruction(0x200, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("conversion_forms", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0x4b800001u);
    EXPECT_EQ(state.read_register(lane, 3), 0x3e888889u);
    EXPECT_EQ(state.read_register(lane, 4), 3u);
    EXPECT_EQ(state.read_register(lane, 6), 0xfffffffdu);
    EXPECT_EQ(state.read_register(lane, 7), 8u);
    EXPECT_EQ(state.read_register(lane, 8), 0xa1u);
    EXPECT_EQ(state.read_register(lane, 9), 0xfffffffdu);
    EXPECT_EQ(state.read_register(lane, 10), 8u);
    EXPECT_EQ(state.read_register(lane, 12), 4u);
    EXPECT_EQ(state.read_register(lane, 15), 0x3e00c080u);
    EXPECT_EQ(state.read_register(lane, 16), 0x4f000000u);
    EXPECT_EQ(state.read_register(lane, 17), 0xfffffffdu);
    EXPECT_EQ(state.read_register(lane, 18), 0x47000100u);
    EXPECT_EQ(state.read_register(lane, 19), 0x4b800001u);
    EXPECT_EQ(state.read_register(lane, 22), 0x3f7fffffu);
    EXPECT_EQ(state.read_register(lane, 23), 3u);
    EXPECT_EQ(state.read_register(lane, 25), 10u);
    EXPECT_EQ(state.read_register(lane, 26), 13u);
    EXPECT_EQ(state.read_register(lane, 27), 0xbf2aaaabu);
    EXPECT_EQ(state.read_register(lane, 28), 2u);
    EXPECT_EQ(state.read_register(lane, 29), 0xbe004080u);
  }
}

TEST(SassFunctionalTest, ExecutesI2fpUnsignedRoundNearestAndZero) {
  kernel image;
  image.name = "i2fp_unsigned_rounding";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0x1000003"),
      official_instruction(0x10, "I2FP.F32.U32", "R1,R0"),
      official_instruction(0x20, "I2FP.F32.U32.RZ", "R2,R0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("i2fp_unsigned_rounding", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0x4b800002u);
    EXPECT_EQ(state.read_register(lane, 2), 0x4b800001u);
  }
}

TEST(SassFunctionalTest, ExecutesUnsignedViaddMinMax) {
  kernel image;
  image.name = "unsigned_viadd_minmax";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0xfffffff0"),
      official_instruction(0x10, "MOV", "R1,0x8"),
      official_instruction(0x20, "VIADDMNMX.U32", "R2,R0,R1,0x4,PT"),
      official_instruction(0x30, "VIADDMNMX.U32", "R3,R0,R1,0x4,!PT"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);

  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("unsigned_viadd_minmax", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 2), 4u);
    EXPECT_EQ(state.read_register(lane, 3), 0xfffffff8u);
  }
}

TEST(SassFunctionalTest, ExecutesI2fpSignedRoundNearestAndZero) {
  kernel image;
  image.name = "i2fp_signed_rounding";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0xfefffffd"),
      official_instruction(0x10, "I2FP.F32.S32", "R1,R0"),
      official_instruction(0x20, "I2FP.F32.S32.RZ", "R2,R0"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  step_result result;
  do {
    result = sass.step("i2fp_signed_rounding", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0xcb800002u);
    EXPECT_EQ(state.read_register(lane, 2), 0xcb800001u);
  }
}

TEST(SassFunctionalTest, ExecutesUnsignedI2fWidthAndRoundingForms) {
  kernel image;
  image.name = "i2f_unsigned_width_rounding";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0xffff"),
      official_instruction(0x10, "I2F.U8", "R3,R0"),
      official_instruction(0x20, "I2F.U16.RZ", "R1,R0"),
      official_instruction(0x30, "I2F.U64.RP", "R2,UR0"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.write_uniform_register(0, 1u);
  state.write_uniform_register(1, 1u);
  execution_context context;
  step_result result;
  do {
    result = sass.step("i2f_unsigned_width_rounding", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0x477fff00u);
    EXPECT_EQ(state.read_register(lane, 2), 0x4f800001u);
    EXPECT_EQ(state.read_register(lane, 3), 0x437f0000u);
  }
}

TEST(SassFunctionalTest, PreservesMufuReciprocalIntegerDivisionRefinement) {
  kernel image;
  image.name = "mufu_integer_division";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0xaa"),
      official_instruction(0x10, "I2F.U32.RP", "R1,R0"),
      official_instruction(0x20, "MUFU.RCP", "R1,R1"),
      official_instruction(0x30, "IADD", "R2,R1,0xffffffe"),
      official_instruction(0x40, "F2I.FTZ.U32.TRUNC.NTZ", "R3,R2"),
      official_instruction(0x50, "MOV", "R9,R3"),
      official_instruction(0x60, "IADD", "R4,RZ,-R3"),
      official_instruction(0x70, "MOV", "R2,RZ"),
      official_instruction(0x80, "IMAD", "R4,R4,R0,RZ"),
      official_instruction(0x90, "IMAD.HI.U32", "R2,R3,R4,R2"),
      official_instruction(0xa0, "MOV", "R3,0x4a9"),
      official_instruction(0xb0, "IMAD.HI.U32", "R8,R2,R3,RZ"),
      official_instruction(0xc0, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  step_result result;
  do {
    result = sass.step("mufu_integer_division", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0x3bc0c0c1u);
    EXPECT_EQ(state.read_register(lane, 9), 0x0181817eu);
    EXPECT_EQ(state.read_register(lane, 2), 0x01818181u);
    EXPECT_EQ(state.read_register(lane, 8), 7u);
  }
}

TEST(SassFunctionalTest, ReadsHighImadAddendsAsRegisterPairs) {
  kernel image;
  image.name = "high_imad_pair_addend";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R0,0x1"),
      official_instruction(0x10, "MOV", "R2,RZ"),
      official_instruction(0x20, "MOV", "R3,0xffffffff"),
      official_instruction(0x30, "IMAD.HI.U32", "R2,R3,R0,R2"),
      official_instruction(0x40, "UMOV", "UR0,0x1"),
      official_instruction(0x50, "UMOV", "UR2,URZ"),
      official_instruction(0x60, "UMOV", "UR3,0xffffffff"),
      official_instruction(0x70, "UIMAD.HI.U32", "UR2,UR3,UR0,UR2"),
      official_instruction(0x80, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  step_result result;
  do {
    result = sass.step("high_imad_pair_addend", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 2), 0xffffffffu);
    EXPECT_EQ(state.read_register(lane, 3), 0xffffffffu);
  }
  EXPECT_EQ(state.read_uniform_register(2), 0xffffffffu);
  EXPECT_EQ(state.read_uniform_register(3), 0xffffffffu);
}

TEST(SassFunctionalTest, ShufflesWithinImmediateAndRegisterSelectedSegments) {
  kernel image;
  image.name = "shuffle_index";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "SHFL.IDX", "P0,R4,R2,R3,0x181f"),
      official_instruction(0x10, "SHFL.IDX", "P1,R6,R2,R7,R5"),
      official_instruction(0x20, "SHFL.IDX", "P2,R8,R2,0x7,0x1803"),
      official_instruction(0x30, "SHFL.BFLY", "P3,R9,R2,0x8,0x1807"),
      official_instruction(0x40, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 2, 100 + lane);
    state.write_register(lane, 3, 7);
    state.write_register(lane, 5, 0x181f);
    state.write_register(lane, 7, 0);
  }
  execution_context context;

  ASSERT_EQ(sass.step("shuffle_index", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 4), 100 + (lane & 0x18) + 7);
    EXPECT_TRUE(state.read_predicate(lane, 0));
  }
  ASSERT_EQ(sass.step("shuffle_index", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 6), 100 + (lane & 0x18));
    EXPECT_TRUE(state.read_predicate(lane, 1));
  }
  ASSERT_EQ(sass.step("shuffle_index", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 8), 100 + lane);
    EXPECT_FALSE(state.read_predicate(lane, 2));
  }
  ASSERT_EQ(sass.step("shuffle_index", state, context).status,
            step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 9), 100 + lane);
    EXPECT_FALSE(state.read_predicate(lane, 3));
  }
}

TEST(SassFunctionalTest, ConvertsSelectedHalfWordsToFp32) {
  kernel image;
  image.name = "half_word_conversion";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R2,0xc0803e00"),
      official_instruction(0x10, "HADD2.F32", "R3,-RZ,R2.H0_H0"),
      official_instruction(0x20, "HADD2.F32", "R4,-RZ,R2.H1_H1"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  step_result result;
  do {
    result = sass.step("half_word_conversion", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 3), 0x3fc00000u);
    EXPECT_EQ(state.read_register(lane, 4), 0xc0100000u);
  }
}

TEST(SassFunctionalTest, AddsPackedHalfWords) {
  kernel image;
  image.name = "packed_half_add";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x00, "MOV", "R2,0x40003c00"),
      official_instruction(0x10, "MOV", "R3,0x44004200"),
      official_instruction(0x20, "HADD2", "R4,R2,R3"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;

  step_result result;
  do {
    result = sass.step("packed_half_add", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
  } while (result.status != step_status::kExited);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(state.read_register(lane, 4), 0x46004400u);
}

TEST(SassFunctionalTest, ExecutesAttentionFloatAndWarpReductionForms) {
  kernel image;
  image.name = "attention_float_forms";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0x000, "MOV", "R0,0x3f800000"),
      official_instruction(0x010, "FMUL", "R1,R0,0.5"),
      official_instruction(0x020, "FADD", "R2,R1,1.5"),
      official_instruction(0x030, "FSETP.GEU.AND", "P0,PT,|R2|,2.0,PT"),
      official_instruction(0x040, "FSEL", "R3,R2,-INF,P0"),
      official_instruction(0x050, "FMNMX", "R4,R3,3.0,!PT"),
      official_instruction(0x060, "MUFU.EX2", "R5,R0"),
      official_instruction(0x070, "FSETP.NEU.AND", "P1,PT,R3,-INF,PT"),
      official_instruction(0x080, "FSETP.GT.AND", "P2,PT,|R3|,1.5,PT"),
      official_instruction(0x090, "FADD", "R8,R4,-R0"),
      official_instruction(0x0a0, "S2R", "R6,SR_LANEID"),
      official_instruction(0x0b0, "SHFL.BFLY", "PT,R7,R6,0x2,0x1f"),
      official_instruction(0x0c0, "UFSETP.GEU.AND", "UP2,UPT,|UR0|,2.0,UPT"),
      official_instruction(0x0d0, "UFSETP.GT.AND", "UP3,UPT,|UR0|,2.5,UPT"),
      official_instruction(0x0e0, "UFSEL", "UR1,UR0,1.0,UP2"),
      official_instruction(0x0f0, "UI2F.U16", "UR2,UR3"),
      official_instruction(0x100, "FSETP.NE.AND", "P3,PT,R0,RZ,PT"),
      official_instruction(0x110, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.write_uniform_register(0, 0x40000000u);  // 2.0f
  state.write_uniform_register(3, 3u);
  execution_context context;
  step_result result;
  do {
    result = sass.step("attention_float_forms", state, context);
    ASSERT_NE(result.status, step_status::kUnsupported) << result.detail;
    ASSERT_NE(result.status, step_status::kMissingPc) << result.detail;
  } while (result.status != step_status::kExited);
  EXPECT_TRUE(state.read_uniform_predicate(2));
  EXPECT_FALSE(state.read_uniform_predicate(3));
  EXPECT_EQ(state.read_uniform_register(1), 0x40000000u);
  EXPECT_EQ(state.read_uniform_register(2), 0x40400000u);

  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 1), 0x3f000000u);
    EXPECT_EQ(state.read_register(lane, 2), 0x40000000u);
    EXPECT_EQ(state.read_register(lane, 3), 0x40000000u);
    EXPECT_EQ(state.read_register(lane, 4), 0x40400000u);
    EXPECT_EQ(state.read_register(lane, 5), 0x40000000u);
    EXPECT_EQ(state.read_register(lane, 7), (lane ^ 2u));
    EXPECT_EQ(state.read_register(lane, 8), 0x40000000u);
    EXPECT_TRUE(state.read_predicate(lane, 0));
    EXPECT_TRUE(state.read_predicate(lane, 1));
    EXPECT_TRUE(state.read_predicate(lane, 2));
    EXPECT_TRUE(state.read_predicate(lane, 3));
  }
}

}  // namespace
