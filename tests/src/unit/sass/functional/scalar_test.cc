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
      // The shift drops high bits; carry comes only from the low-word add.
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
    EXPECT_FALSE(state.read_predicate(lane, 5));
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
  EXPECT_EQ(state.read_uniform_register(18), 9u);
  EXPECT_EQ(state.read_uniform_register(20), 2u);
  EXPECT_EQ(state.read_uniform_register(21), 1u);
  EXPECT_EQ(state.read_uniform_register(22), 1u);
  EXPECT_EQ(state.read_uniform_register(23), 2u);
  EXPECT_EQ(state.read_uniform_register(24), 0u);
  EXPECT_FALSE(state.read_uniform_predicate(3));
  EXPECT_FALSE(state.read_uniform_predicate(0));
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
  // No .W qualifier: a shift of 36 clamps, rather than wrapping to four.
  EXPECT_EQ(state.read_uniform_register(2), 0u);
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

TEST(SassFunctionalTest, R2pByteSelectorsPreserveMaskAndInactiveLanes) {
  for (unsigned byte = 0; byte < 4; ++byte) {
    for (bool explicit_pr : {false, true}) {
      kernel image;
      image.name = "r2p_bytes";
      image.arch = architecture::kSm120;
      image.instruction_bytes = 16;
      const std::string operands = std::string(explicit_pr ? "PR," : "") +
                                   "R6.B" + std::to_string(byte) + ",0x3";
      image.instructions = {
          official_instruction(0, "R2P", operands.c_str(), "@P6")};
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm120_functional_semantics(sass);
      warp_state state;
      state.active_mask = 3;
      for (unsigned lane = 0; lane < 3; ++lane) {
        state.write_register(lane, 6, 0x01020300);
        for (unsigned pred = 0; pred < 7; ++pred)
          state.write_predicate(lane, pred, true);
      }
      state.write_predicate(1, 6, false);
      execution_context context;
      const auto result = sass.step("r2p_bytes", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      const unsigned value = (0x01020300u >> (byte * 8));
      for (unsigned pred = 0; pred < 7; ++pred) {
        EXPECT_EQ(state.read_predicate(0, pred),
                  pred < 2 ? bool(value & (1u << pred)) : true);
        EXPECT_EQ(state.read_predicate(1, pred), pred != 6);
        EXPECT_TRUE(state.read_predicate(2, pred));
      }
    }
  }
}

}  // namespace

#include "../quantization_reference.h"
namespace {
TEST(SassFunctionalTest, DeepInferFp8BfloatUnpackRetainsExactImmediateBits) {
  kernel image;
  image.name = "fp8_unpack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "F2FP.BF16.E4M3.UNPACK_B",
                           "R8,R2,4.5736980577097704378e-41,H0"),
      official_instruction(16, "F2FP.BF16.E4M3.UNPACK_B",
                           "R9,R2.reuse.H1,R3,H0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0, 2, 0x40384038);
  state.write_register(0, 3, 0x7e80);
  state.write_register(1, 2, 0x7f008000);
  state.write_register(1, 3, 0x7f7f);
  execution_context context;
  auto r = sass.step("fp8_unpack", state, context);
  ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  EXPECT_EQ(state.read_register(0, 8), 0x40003f80u);
  EXPECT_EQ(state.read_register(1, 8), 0x80000000u);
  r = sass.step("fp8_unpack", state, context);
  ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  EXPECT_EQ(state.read_register(0, 9), 0x3f804000u);
  EXPECT_EQ(state.read_register(1, 9), 0x7fff0000u);
}

TEST(SassFunctionalTest, ConvertsFiniteF32ToUnsigned64Pair) {
  kernel image;
  image.name = "f2i_u64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "F2I.U64.TRUNC", "R4,R4")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0, 4, 0x3fe00000); // 1.75 -> 1
  state.write_register(1, 4, 0x53800000); // 2^40 -> high word 256
  state.write_register(2, 4, 0x7fc00000); // inactive NaN must not execute
  state.write_register(2, 5, 99);
  execution_context context;
  ASSERT_EQ(sass.step("f2i_u64", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 1u);
  EXPECT_EQ(state.read_register(0, 5), 0u);
  EXPECT_EQ(state.read_register(1, 4), 0u);
  EXPECT_EQ(state.read_register(1, 5), 256u);
  EXPECT_EQ(state.read_register(2, 4), 0x7fc00000u);
  EXPECT_EQ(state.read_register(2, 5), 99u);
  state.pc = 0;
  state.active_mask = 1;
  state.write_register(0, 4, 0x5f800000); // 2^64 is unqualified, not a C++ cast
  EXPECT_EQ(sass.step("f2i_u64", state, context).status, step_status::kUnsupported);
}

TEST(SassFunctionalTest, NativeUniformSignedGeOrPredicate) {
  kernel image;
  image.name = "uniform_ge_or";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "UISETP.GE.OR", "UP0,UPT,UR0,UR1,!UP1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (int32_t a : {-1, 2, 3}) {
    for (bool gate : {false, true}) {
      warp_state state;
      state.write_uniform_register(0, uint32_t(a));
      state.write_uniform_register(1, 2);
      state.write_uniform_predicate(1, gate);
      execution_context context;
      ASSERT_EQ(sass.step("uniform_ge_or", state, context).status, step_status::kAdvanced);
      EXPECT_EQ(state.read_uniform_predicate(0), a >= 2 || !gate);
    }
  }
}

TEST(SassFunctionalTest, NativeSignedMinMaxWithDiscardedPredicateOutputs) {
  kernel image;
  image.name = "signed_minmax";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "VIMNMX", "R0,PT,PT,R1,R2,PT"),
      official_instruction(16, "VIMNMX.S32", "R3,PT,PT,R1,R2,!PT"),
      official_instruction(32, "VIMNMX", "R4,P0,PT,R1,R2,PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 1, uint32_t(-7));
  state.write_register(0, 2, 4);
  state.write_register(1, 0, 99);
  execution_context context;
  ASSERT_EQ(sass.step("signed_minmax", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 0), uint32_t(-7));
  EXPECT_EQ(state.read_register(1, 0), 99u);
  ASSERT_EQ(sass.step("signed_minmax", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 3), 4u);
  EXPECT_EQ(sass.step("signed_minmax", state, context).status,
            step_status::kUnsupported);
}

TEST(SassFunctionalTest, BlackwellMinMaxDiscardedPredicateOutputs) {
  kernel image;
  image.name = "vimnmx_discard";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "VIMNMX.U32", "R0,PT,PT,R1,0x2,PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (uint32_t value : {0u, 1u, 2u, 0xffffffffu}) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 1, value);
    execution_context context;
    ASSERT_EQ(sass.step("vimnmx_discard", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 0), std::min(value, 2u));
  }
}

TEST(SassFunctionalTest, BlackwellFfma2BroadcastAndAliasedPairs) {
  kernel image;
  image.name = "ffma2";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "FFMA2", "R2,R0.reuse.F32,R2.F32x2.HI_LO,R4.F32x2.HI_LO", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 0, 0x40000000u); // 2
    state.write_register(lane, 2, 0x40400000u); // 3
    state.write_register(lane, 3, 0xc0800000u); // -4
    state.write_register(lane, 4, 0x3f800000u); // 1
    state.write_register(lane, 5, 0x3f000000u); // .5
    state.write_predicate(lane, 0, lane == 0);
  }
  execution_context context;
  const auto result = sass.step("ffma2", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(state.read_register(0, 2), 0x40e00000u); // 7
  EXPECT_EQ(state.read_register(0, 3), 0xc0f00000u); // -7.5
  EXPECT_EQ(state.read_register(1, 2), 0x40400000u);
  EXPECT_EQ(state.read_register(1, 3), 0xc0800000u);
}

TEST(SassFunctionalTest, CutlassUnsignedUniformLessAndFloatEqualityFtz) {
  kernel image;
  image.name = "cutlass_compare";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0,"UISETP.LT.U32.AND","UP0,UPT,UR0,UR1,UPT"),
      official_instruction(16,"FSETP.EQ.FTZ.AND","P0,PT,R0,RZ,PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for(uint32_t a : {0u,1u,0xffffffffu}) for(uint32_t b : {0u,1u,0x80000000u}) {
    warp_state state;
    state.active_mask=1;
    state.write_uniform_register(0,a);
    state.write_uniform_register(1,b);
    execution_context context;
    ASSERT_EQ(sass.step("cutlass_compare",state,context).status,step_status::kAdvanced);
    EXPECT_EQ(state.read_uniform_predicate(0),a<b);
  }
  for(uint32_t bits : {0u,1u,0x80000001u,0x00800000u,0x7fc00000u}) {
    warp_state state;
    state.pc=16; state.active_mask=1;
    state.write_register(0,0,bits);
    execution_context context;
    ASSERT_EQ(sass.step("cutlass_compare",state,context).status,step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0,0),(bits&0x7fffffffu)<0x00800000u);
  }
}

TEST(SassFunctionalTest, CutlassExtendedEqualityWithUnsignedHighWord) {
  kernel image;
  image.name = "eq_u32_ex";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.EQ.U32.OR.EX", "P6,PT,R3,RZ,!P3,P0"),
      official_instruction(16, "ISETP.EQ.U32.AND.EX", "P6,PT,R3,RZ,!P3,P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (uint32_t high : {0u, 1u, 0x80000000u})
    for (bool low_equal : {false, true}) for (bool force : {false, true}) {
      warp_state state;
      state.active_mask = 1;
      state.write_register(0, 3, high);
      state.write_predicate(0, 0, low_equal);
      state.write_predicate(0, 3, !force);
      execution_context context;
      ASSERT_EQ(sass.step("eq_u32_ex", state, context).status, step_status::kAdvanced);
      EXPECT_EQ(state.read_predicate(0, 6), (high == 0 && low_equal) || force);
      ASSERT_EQ(sass.step("eq_u32_ex", state, context).status, step_status::kAdvanced);
      EXPECT_EQ(state.read_predicate(0, 6), high == 0 && low_equal && force);
    }
}

TEST(SassFunctionalTest, BlackwellFfma2RejectsUnqualifiedSelectors) {
  for (const char* operands : {"R2,R0.F32,R2.F32x2.HI_HI,R4.F32x2.HI_LO",
                               "R2,R0,R2,R4", "R254,R0.F32,R2.F32x2.HI_LO,R4.F32x2.HI_LO"}) {
    kernel image;
    image.name = "bad_ffma2";
    image.arch = architecture::kSm120;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(0, "FFMA2", operands)};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    warp_state state;
    execution_context context;
    EXPECT_EQ(sass.step("bad_ffma2", state, context).status, step_status::kUnsupported);
  }
}

TEST(SassFunctionalTest, BlackwellFfma2MlaPairWithUniformAndScalarBroadcasts) {
  kernel image;
  image.name = "mla_ffma2";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(
      0, "FFMA2", "R4,R4.F32x2.HI_LO,UR10.F32,R71.reuse.F32")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 4, 0x40400000); // 3
  state.write_register(0, 5, 0xc0800000); // -4
  state.write_uniform_register(10, 0x40000000); // 2
  state.write_register(0, 71, 0x3f800000); // 1
  execution_context context;
  ASSERT_EQ(sass.step("mla_ffma2", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x40e00000u); // 7
  EXPECT_EQ(state.read_register(0, 5), 0xc0e00000u); // -7
}

TEST(SassFunctionalTest, BlackwellMlaGreaterUnorderedPredicate) {
  kernel image;
  image.name = "gtu";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "FSETP.GTU.AND", "P0,PT,R1,R2,P1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const float values[] = {-1, 0, 1, std::numeric_limits<float>::quiet_NaN()};
  for (float a : values) for (float b : values) for (bool enable : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    uint32_t bits;
    std::memcpy(&bits, &a, 4); state.write_register(0, 1, bits);
    std::memcpy(&bits, &b, 4); state.write_register(0, 2, bits);
    state.write_predicate(0, 1, enable);
    execution_context context;
    ASSERT_EQ(sass.step("gtu", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 0), enable && (std::isnan(a) || std::isnan(b) || a > b));
  }
}

TEST(SassFunctionalTest, BlackwellFadd2PairsAndZeroBroadcast) {
  kernel image;
  image.name = "add_pair";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "FADD2", "R4,R4.F32x2.HI_LO,RZ.F32"),
      official_instruction(16, "FADD2", "R4,R4.F32x2.HI_LO,R6.F32x2.HI_LO"),
      official_instruction(32, "FMUL2", "R4,R4.F32x2.HI_LO,R7.F32")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 4, 0x40400000); // 3
  state.write_register(0, 5, 0xc0800000); // -4
  state.write_register(0, 6, 0x3f800000); // 1
  state.write_register(0, 7, 0x3f000000); // .5
  execution_context context;
  ASSERT_EQ(sass.step("add_pair", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x40400000u);
  EXPECT_EQ(state.read_register(0, 5), 0xc0800000u);
  ASSERT_EQ(sass.step("add_pair", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x40800000u); // 4
  EXPECT_EQ(state.read_register(0, 5), 0xc0600000u); // -3.5
  ASSERT_EQ(sass.step("add_pair", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0x40000000u); // 2
  EXPECT_EQ(state.read_register(0, 5), 0xbfe00000u); // -1.75
}

TEST(SassFunctionalTest, BlackwellUniformSigned64ShiftLowWord) {
  kernel image;
  image.name = "ushf_s64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "USHF.R.S64", "UR4,UR4,UR6,UR5")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (unsigned amount : {0u, 7u, 31u, 32u, 47u, 63u, 64u, 127u, UINT32_MAX}) {
    warp_state state;
    state.write_uniform_register(4, 0x12345678);
    state.write_uniform_register(5, 0x87654321);
    state.write_uniform_register(6, amount);
    execution_context context;
    const auto result = sass.step("ushf_s64", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(
        state.read_uniform_register(4),
        uint32_t(int64_t(0x8765432112345678ull) >> std::min(amount, 63u)));
  }
}

TEST(SassFunctionalTest, BlackwellUniformUnsigned64ShiftLowWord) {
  kernel image;
  image.name = "ushf_u64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "USHF.R.U64", "UR4,UR4,UR6,UR5")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (unsigned amount : {0u, 7u, 31u, 32u, 47u, 63u, 64u, 127u, UINT32_MAX}) {
    warp_state state;
    state.write_uniform_register(4, 0x12345678);
    state.write_uniform_register(5, 0x87654321);
    state.write_uniform_register(6, amount);
    execution_context context;
    const auto result = sass.step("ushf_u64", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.read_uniform_register(4),
              amount >= 64 ? 0u : uint32_t(0x8765432112345678ull >> amount));
  }
}

TEST(SassFunctionalTest, BlackwellUniformLeaHighCarryOutputAndAlias) {
  kernel image;
  image.name = "ulea_hi_carry";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "ULEA.HI", "UR4,UP0,UR4,UR6,URZ,0x7")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (uint32_t a : {0u, 0xffffffffu}) for (uint32_t b : {0u, 0xffffffffu}) {
    warp_state state;
    state.write_uniform_register(4, a);
    state.write_uniform_register(6, b);
    execution_context context;
    const auto result = sass.step("ulea_hi_carry", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    const uint64_t expected = uint64_t(a >> 25) + b;
    EXPECT_EQ(state.read_uniform_register(4), uint32_t(expected));
    EXPECT_EQ(state.read_uniform_predicate(0), bool(expected >> 32));
  }
}

TEST(SassFunctionalTest, BlackwellUnsignedLessEqualAndPredicate) {
  kernel image;
  image.name = "le_unsigned";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "ISETP.LE.U32.AND", "P1,PT,R0,0x1,P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (uint32_t a : {0u, 1u, 0xffffffffu}) for (bool input : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, a);
    state.write_predicate(0, 0, input);
    execution_context context;
    ASSERT_EQ(sass.step("le_unsigned", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 1), a <= 1 && input);
  }
}

TEST(SassFunctionalTest, BlackwellFp8PackingRoundingSaturationAndAlias) {
  kernel image;
  image.name = "fp8_pack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", "R0,R0,R1,RZ")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  std::vector<float> values = {0.f, -0.f, 448.f, 1000.f, INFINITY, -INFINITY, NAN};
  for (unsigned code = 0; code < 126; ++code) {
    const double lo = moe_quant_reference::positive_value(code);
    const double hi = moe_quant_reference::positive_value(code + 1);
    values.push_back(float(lo));
    values.push_back(float((lo + hi) / 2)); // exact RNE halfway cases
    values.push_back(std::nextafter(float((lo + hi) / 2), float(lo)));
    values.push_back(std::nextafter(float((lo + hi) / 2), float(hi)));
  }
  for (float value : values) {
    warp_state state;
    state.active_mask = 1;
    uint32_t a, b;
    const float negative = -value;
    std::memcpy(&a, &value, 4); std::memcpy(&b, &negative, 4);
    state.write_register(0, 0, a); state.write_register(0, 1, b);
    execution_context context;
    const auto result = sass.step("fp8_pack", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.read_register(0, 0),
              (uint32_t(moe_quant_reference::encode(value)) << 8) |
              moe_quant_reference::encode(negative)) << value;
  }
}

TEST(SassFunctionalTest, BlackwellFp8MergesLowHalfAndAllowsDestinationAlias) {
  kernel image;
  image.name = "fp8_merge";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C", "R3,R1,R2,R3")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  execution_context context;
  for(unsigned lane=0;lane<32;++lane) {
    state.write_register(lane,1,0x3f800000); // 1 -> 0x38
    state.write_register(lane,2,0xc0000000); // -2 -> 0xc0
    state.write_register(lane,3,0xa5f01234u ^ (lane*0x01010101u));
  }
  ASSERT_EQ(sass.step("fp8_merge", state, context).status, step_status::kAdvanced);
  for(unsigned lane=0;lane<32;++lane)
    EXPECT_EQ(state.read_register(lane,3),
              ((0xa5f01234u ^ (lane*0x01010101u))<<16)|0x38c0u);
}

TEST(SassFunctionalTest, BlackwellUnsignedComparisonsCombineOrPredicate) {
  kernel image;
  image.name = "compare_or";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.LT.U32.OR", "P1,PT,R0,R1,P0"),
      official_instruction(16, "ISETP.GE.U32.OR", "P2,PT,R0,R1,P0"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (uint32_t value : {0u, 1u, 0xffffffffu}) for (bool input : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, value);
    state.write_register(0, 1, 1);
    state.write_predicate(0, 0, input);
    execution_context context;
    ASSERT_EQ(sass.step("compare_or", state, context).status, step_status::kAdvanced);
    ASSERT_EQ(sass.step("compare_or", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 1), value < 1 || input);
    EXPECT_EQ(state.read_predicate(0, 2), value >= 1 || input);
  }
}

TEST(SassFunctionalTest, BlackwellTernaryFloatMinMaxAndNanPredicate) {
  kernel image;
  image.name = "quant_float";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "FMNMX3.FTZ", "R4,|R0|,R1,R2,!PT"),
      official_instruction(16, "FMNMX3.FTZ", "R5,|R0|,R1,R2,PT"),
      official_instruction(32, "FSETP.NAN.FTZ.AND", "P1,PT,R0,R3,P0"),
      official_instruction(48, "FSETP.GT.FTZ.AND", "P2,PT,R1,RZ,PT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.write_register(lane, 0, 0xc0000000u); // -2
    state.write_register(lane, 1, 1); // positive subnormal, flushed to zero
    state.write_register(lane, 2, 0x40400000u); // 3
    state.write_register(lane, 3, lane % 2 ? 0x7fc00000u : 0x3f800000u);
    state.write_predicate(lane, 0, lane < 16);
  }
  execution_context context;
  for (unsigned i = 0; i < 4; ++i)
    ASSERT_EQ(sass.step("quant_float", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(state.read_register(lane, 4), 0x40400000u);
    EXPECT_EQ(state.read_register(lane, 5), 0u);
    EXPECT_EQ(state.read_predicate(lane, 1), lane % 2 && lane < 16);
    EXPECT_FALSE(state.read_predicate(lane, 2));
  }
}

TEST(SassFunctionalTest, BfloatFmaImmediateHasHighThenLowMultiplier) {
  kernel image;
  image.name = "bf16_fma_literal";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "HFMA2.BF16_V2", "R2,R0,2,0.5,R1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0xc0004080u); // high=-2, low=4
  state.write_register(0, 1, 0x3f80bf80u); // high=1, low=-1
  execution_context context;
  ASSERT_EQ(sass.step("bf16_fma_literal", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 2), 0xc0403f80u); // high=-3, low=1
}

TEST(SassFunctionalTest, SignedExtendedGeUsesSignedHighAndLowPredicate) {
  kernel image;
  image.name = "signed_ge_extended";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "ISETP.GE.AND.EX", "P0,PT,R0,R1,PT,P1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (int high : {-1, 0, 1}) for (bool low_ge : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, uint32_t(high)); state.write_register(0, 1, 0);
    state.write_predicate(0, 1, low_ge);
    execution_context context;
    ASSERT_EQ(sass.step("signed_ge_extended", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 0), high > 0 || (high == 0 && low_ge));
  }
}

TEST(SassFunctionalTest, UniformSignedExtendedGeUsesSignedHighAndLowPredicate) {
  kernel image;
  image.name = "uniform_signed_ge_extended";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "UISETP.GE.AND.EX", "UP0,UPT,UR0,UR1,UPT,UP1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (int high : {-1, 0, 1}) for (bool low_ge : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_uniform_register(0, uint32_t(high)); state.write_uniform_register(1, 0);
    state.write_uniform_predicate(1, low_ge);
    execution_context context;
    ASSERT_EQ(sass.step("uniform_signed_ge_extended", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_uniform_predicate(0), high > 0 || (high == 0 && low_ge));
  }
}

TEST(SassFunctionalTest, SignedExtendedGtUsesSignedHighAndLowPredicate) {
  kernel image;
  image.name = "signed_gt_extended";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "ISETP.GT.AND.EX", "P0,PT,R0,R1,PT,P1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (int high : {-1, 0, 1}) for (bool low_gt : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 0, uint32_t(high)); state.write_register(0, 1, 0);
    state.write_predicate(0, 1, low_gt);
    execution_context context;
    ASSERT_EQ(sass.step("signed_gt_extended", state, context).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_predicate(0, 0), high > 0 || (high == 0 && low_gt));
  }
}

TEST(SassFunctionalTest, BlackwellBfloatAddRoundsAtEachInstruction) {
  kernel image;
  image.name = "bf16_add";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "HADD2.BF16_V2", "R2,R0,R1"),
      official_instruction(0x10, "HADD2.BF16_V2", "R2,R2,R1"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0xc0003f80u); // {-2, 1}
  state.write_register(0, 1, 0x3f803b80u); // {1, 1/256}: tie at 1 rounds even.
  execution_context context;
  ASSERT_EQ(sass.step("bf16_add", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 2), 0xbf803f80u);
  ASSERT_EQ(sass.step("bf16_add", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 2), 0x00003f80u);
}

TEST(SassFunctionalTest, TanhHandlesSignedZeroInfinityAndPredicate) {
  kernel image;
  image.name = "tanh";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "MUFU.TANH", "R1,R0", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 31;
  const uint32_t inputs[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u, 0};
  const uint32_t expected[] = {0, 0x80000000u, 0x3f800000u, 0xbf800000u, 99};
  for (unsigned i = 0; i < 5; ++i) {
    state.write_register(i, 0, inputs[i]); state.write_register(i, 1, 99);
    state.write_predicate(i, 0, i != 4);
  }
  execution_context context;
  ASSERT_EQ(sass.step("tanh", state, context).status, step_status::kAdvanced);
  for (unsigned i = 0; i < 5; ++i) EXPECT_EQ(state.read_register(i, 1), expected[i]);
}

TEST(SassFunctionalTest, BlackwellBfloatPackOrderAndTiesToEven) {
  kernel image;
  image.name = "bf16_pack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "F2FP.BF16.F32.PACK_AB", "R2,R0,R1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t a[] = {0x3f800000, 0x3f808000, 0x3f818000, 0x80000000, 0x7f800000};
  const uint32_t b[] = {0xc0000000, 0xbf808000, 0xbf818000, 0, 0xff800000};
  const uint32_t expected[] = {0x3f80c000, 0x3f80bf80, 0x3f82bf82, 0x80000000, 0x7f80ff80};
  warp_state state;
  state.active_mask = 31;
  for (unsigned i = 0; i < 5; ++i) {
    state.write_register(i, 0, a[i]);
    state.write_register(i, 1, b[i]);
  }
  execution_context context;
  ASSERT_EQ(sass.step("bf16_pack", state, context).status, step_status::kAdvanced);
  for (unsigned i = 0; i < 5; ++i) EXPECT_EQ(state.read_register(i, 2), expected[i]);
}

TEST(SassFunctionalTest, BlackwellUniformExtendedAddDiscardedOutputs) {
  kernel image;
  image.name = "uniform_extended_add";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "UIADD3.X",
      "UR5,UPT,UPT,URZ,UR5,URZ,UP0,!UPT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (bool carry : {false, true}) {
    warp_state state;
    state.active_mask = 1;
    state.write_uniform_register(5, 0xffffffffu);
    state.write_uniform_predicate(0, carry);
    execution_context context;
    ASSERT_EQ(sass.step("uniform_extended_add", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_uniform_register(5), carry ? 0u : 0xffffffffu);
  }
}

TEST(SassFunctionalTest, ReciprocalSquareRootNormalAndSpecialValues) {
  kernel image;
  image.name = "rsq";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "MUFU.RSQ", "R1,R0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const float inputs[] = {4.0f, 16.0f, 0.0f, -0.0f, -1.0f,
                         std::numeric_limits<float>::infinity()};
  warp_state state;
  state.active_mask = 63;
  for (unsigned lane = 0; lane < 6; ++lane) {
    uint32_t bits;
    std::memcpy(&bits, &inputs[lane], 4);
    state.write_register(lane, 0, bits);
  }
  state.write_register(6, 1, 123);
  execution_context context;
  ASSERT_EQ(sass.step("rsq", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 1), 0x3f000000u);
  EXPECT_EQ(state.read_register(1, 1), 0x3e800000u);
  EXPECT_EQ(state.read_register(2, 1), 0x7f800000u);
  EXPECT_EQ(state.read_register(3, 1), 0xff800000u);
  const uint32_t nan_bits = state.read_register(4, 1);
  EXPECT_EQ(nan_bits & 0x7f800000u, 0x7f800000u);
  EXPECT_NE(nan_bits & 0x007fffffu, 0u);
  EXPECT_EQ(state.read_register(5, 1), 0u);
  EXPECT_EQ(state.read_register(6, 1), 123u);
}

TEST(SassFunctionalTest, BlackwellExtendedUniformNotEqual) {
  kernel image;
  image.name = "blackwell_pointer_compare";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UISETP.NE.U32.AND.EX",
                           "UP0,UPT,UR5,URZ,UP1,UP0"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (unsigned high : {0u, 1u, 0x80000000u})
    for (bool low_ne : {false, true})
      for (bool guard : {false, true}) {
        warp_state state;
        state.active_mask = 1;
        state.write_uniform_register(5, high);
        state.write_uniform_predicate(0, low_ne);
        state.write_uniform_predicate(1, guard);
        execution_context context;
        const auto result = sass.step("blackwell_pointer_compare", state, context);
        ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
        EXPECT_EQ(state.read_uniform_predicate(0), (high != 0 || low_ne) && guard);
      }
}

TEST(SassFunctionalTest, BlackwellIadd3xDiscardedCarryOutputs) {
  for (auto arch : {architecture::kSm120, architecture::kSm120}) {
    kernel image;
    image.name = "blackwell_subtract";
    image.arch = arch;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "IADD3.X", "R9,PT,PT,~R3,R7,RZ,P0,!PT"),
    };
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    warp_state state;
    state.active_mask = 3;
    for (unsigned lane = 0; lane < 2; ++lane) {
      state.write_register(lane, 3, 10);
      state.write_register(lane, 7, 17);
      state.write_predicate(lane, 0, lane == 0);
    }
    state.write_register(2, 9, 123);
    execution_context context;
    const auto result = sass.step("blackwell_subtract", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.read_register(0, 9), 7u);
    EXPECT_EQ(state.read_register(1, 9), 6u);
    EXPECT_EQ(state.read_register(2, 9), 123u);
  }
}

TEST(SassFunctionalTest, BlackwellIadd3xRejectsUnverifiedLiveCarryOutputs) {
  kernel image;
  image.name = "blackwell_live_carry";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "IADD3.X", "R9,P1,P2,R3,R7,RZ,P0,!PT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 9, 123);
  execution_context context;
  EXPECT_EQ(sass.step("blackwell_live_carry", state, context).status,
            step_status::kUnsupported);
  EXPECT_EQ(state.read_register(0, 9), 123u);
}

TEST(SassFunctionalTest, CombinesUnsignedEqualityWithPredicateXor) {
  kernel image;
  image.name = "isetp_eq_u32_xor";
  image.arch = architecture::kSm90;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.EQ.U32.XOR", "P1,PT,R2,0x80,!P0"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm90_functional_semantics(sass);
  warp_state state;
  state.active_mask = 0xf;
  for (unsigned lane = 0; lane < 4; ++lane) {
    state.write_register(lane, 2, lane % 2 ? 0x81 : 0x80);
    state.write_predicate(lane, 0, lane < 2);
  }
  execution_context context;
  const auto result = sass.step("isetp_eq_u32_xor", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_TRUE(state.read_predicate(0, 1));
  EXPECT_FALSE(state.read_predicate(1, 1));
  EXPECT_FALSE(state.read_predicate(2, 1));
  EXPECT_TRUE(state.read_predicate(3, 1));
}

TEST(SassFunctionalTest, CombinesUniformEqualityWithPredicateXor) {
  kernel image;
  image.name = "uisetp_eq_u32_xor";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "UISETP.EQ.U32.XOR", "UP1,UPT,UR2,0x80,!UP0"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  execution_context context;
  for (unsigned condition = 0; condition < 4; ++condition) {
    warp_state state;
    // A uniform instruction must also work when lane zero is inactive.
    state.active_mask = 0x80000000u;
    state.write_uniform_register(2, condition % 2 ? 0x81 : 0x80);
    state.write_uniform_predicate(0, condition < 2);
    const auto result = sass.step("uisetp_eq_u32_xor", state, context);
    ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
    EXPECT_EQ(state.read_uniform_predicate(1), condition == 0 || condition == 3);
  }
}

TEST(SassFunctionalTest, PreservesNoBorrowWhenSubtractingZero) {
  for (bool uniform : {false, true}) {
    kernel image;
    image.name = "subtract_zero_carry";
    image.arch = architecture::kSm120;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, uniform ? "UIADD3" : "IADD3",
            uniform ? "UR14,UP0,UPT,-UR13,UR2,URZ"
                    : "R14,P0,PT,-R13,R2,RZ"),
        official_instruction(0x10, "IMAD.X", "R6,R11,0x1,~R6,P0"),
    };
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm120_functional_semantics(sass);
    for (uint32_t subtrahend : {0u, 1u, 4095u, 4096u}) {
      warp_state state;
      state.active_mask = 1;
      state.write_register(0, 2, 4095);
      state.write_register(0, 13, subtrahend);
      state.write_register(0, 11, 0);
      state.write_register(0, 6, 0);
      state.write_uniform_register(2, 4095);
      state.write_uniform_register(13, subtrahend);
      execution_context context;
      ASSERT_EQ(sass.step("subtract_zero_carry", state, context).status,
                step_status::kAdvanced);
      EXPECT_EQ(uniform ? state.read_uniform_register(14) : state.read_register(0, 14),
                4095u - subtrahend);
      EXPECT_EQ(uniform ? state.read_uniform_predicate(0) : state.read_predicate(0, 0),
                subtrahend <= 4095u);
      if (!uniform) {
        ASSERT_EQ(sass.step("subtract_zero_carry", state, context).status,
                  step_status::kAdvanced);
        EXPECT_EQ(state.read_register(0, 6), subtrahend <= 4095u ? 0u : 0xffffffffu);
      }
    }
  }
}

TEST(SassFunctionalTest, ComparesSigned64PairsWithUniformSources) {
  kernel image;
  image.name = "isetp_s64";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.GE.S64.AND", "P0,PT,R0,R2,PT"),
      official_instruction(0x10, "ISETP.GE.S64.AND", "P1,PT,R0,UR4,PT"),
      official_instruction(0x20, "ISETP.GE.S64.AND", "P2,PT,R0,UR4,!PT"),
      official_instruction(0x30, "EXIT"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  // Check upper-word ordering, negative versus positive, signed extrema and
  // equality; the uniform operand is 2^32, so reading UR4 alone is wrong.
  const std::array<int64_t, 8> a = {
      int64_t{1} << 32, (int64_t{1} << 32) - 1, -1, 0,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(), -(int64_t{1} << 32), -1};
  const std::array<int64_t, 8> b = {
      (int64_t{1} << 32) - 1, int64_t{1} << 32, 0, -1,
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::min(), -1, -1};
  warp_state state;
  state.active_mask = 0xff;
  state.write_uniform_register(4, 0);
  state.write_uniform_register(5, 1);
  for (unsigned lane = 0; lane < a.size(); ++lane) {
    state.write_register(lane, 0, static_cast<uint64_t>(a[lane]));
    state.write_register(lane, 1, static_cast<uint64_t>(a[lane]) >> 32);
    state.write_register(lane, 2, static_cast<uint64_t>(b[lane]));
    state.write_register(lane, 3, static_cast<uint64_t>(b[lane]) >> 32);
    state.write_predicate(lane, 2, true);
  }
  execution_context context;
  for (unsigned step = 0; step < 4; ++step) {
    const auto result = sass.step("isetp_s64", state, context);
    ASSERT_EQ(result.status, step == 3 ? step_status::kExited
                                      : step_status::kAdvanced) << result.detail;
  }
  for (unsigned lane = 0; lane < a.size(); ++lane) {
    EXPECT_EQ(state.read_predicate(lane, 0), a[lane] >= b[lane]) << lane;
    EXPECT_EQ(state.read_predicate(lane, 1), a[lane] >= (int64_t{1} << 32)) << lane;
    EXPECT_FALSE(state.read_predicate(lane, 2)) << lane;
  }
}

TEST(SassFunctionalTest, ComparesUnsigned64EqualityAndCombinesPredicate) {
  kernel image;
  image.name = "u64_eq";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "ISETP.EQ.U64.AND", "P0,PT,R4,UR4,PT"),
      official_instruction(16, "ISETP.EQ.U64.OR", "P1,PT,R4,RZ,!P2"),
  };
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 7;
  state.write_uniform_register(4, 0);
  state.write_uniform_register(5, 1);
  for (unsigned lane = 0; lane < 3; ++lane) {
    state.write_register(lane, 4, 0);
    state.write_register(lane, 5, lane);
    state.write_predicate(lane, 2, lane != 2);
  }
  execution_context context;
  ASSERT_EQ(sass.step("u64_eq", state, context).status, step_status::kAdvanced);
  ASSERT_EQ(sass.step("u64_eq", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 3; ++lane) {
    EXPECT_EQ(state.read_predicate(lane, 0), lane == 1);
    EXPECT_EQ(state.read_predicate(lane, 1), lane == 0 || lane == 2);
  }
}

TEST(SassFunctionalTest, Dsv4DirectedAddPreservesTinyNegativeResidual) {
  kernel image;
  image.name = "rz_add";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "FADD.FTZ.RZ", "R0,R1,R2")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  // Tiny opposite-sign normals are lost in double precision too. RZ must
  // nevertheless choose the adjacent float toward zero. Include FTZ and
  // overflow.
  const uint32_t inputs[][3] = {{0x3f800000, 0x8d800000, 0x3f7fffff},
                                {0xbf800000, 0x0d800000, 0xbf7fffff},
                                {0x3f800000, 0x00000001, 0x3f800000},
                                {0x7f7fffff, 0x7f7fffff, 0x7f7fffff},
                                {0x3f800000, 0xbf800000, 0}};
  for (auto &v : inputs) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 1, v[0]);
    state.write_register(0, 2, v[1]);
    execution_context context;
    ASSERT_EQ(sass.step("rz_add", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 0), v[2]);
  }
}

TEST(SassFunctionalTest, Dsv4BfloatMinMaxHalfReplicationAndPredicate) {
  kernel image;
  image.name = "bfmin";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "HMNMX2.BF16_V2", "R0,R1,R2.reuse.H0_H0,PT"),
      official_instruction(16, "HMNMX2.BF16_V2", "R3,R1,R2.H1_H1,!PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  execution_context context;
  state.write_register(0, 1, 0xc0404080);  // low=4, high=-3
  state.write_register(0, 2, 0xc0004000);  // low=2, high=-2
  state.write_register(1, 0, 123);
  ASSERT_EQ(sass.step("bfmin", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 0), 0xc0404000u);
  EXPECT_EQ(state.read_register(1, 0), 123u);
  ASSERT_EQ(sass.step("bfmin", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 3), 0xc0004080u);
}

TEST(SassFunctionalTest, Dsv4Signed64ShiftLowWordAndClampedBoundary) {
  kernel image;
  image.name = "s64shift";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "SHF.R.S64", "R0,R1,R2,R3")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  for (unsigned amount : {0u, 24u, 31u, 32u, 63u, 64u, 127u}) {
    warp_state state;
    state.active_mask = 1;
    execution_context ctx;
    state.write_register(0, 1, 0x12345678);
    state.write_register(0, 2, amount);
    state.write_register(0, 3, 0xabcdef00);
    auto result = sass.step("s64shift", state, ctx);
    ASSERT_EQ(result.status, step_status::kAdvanced);
    EXPECT_EQ(
        state.read_register(0, 0),
        uint32_t(int64_t(0xabcdef0012345678ull) >> std::min(amount, 63u)));
  }
}

TEST(SassFunctionalTest, Dsv4SqrtFlushesSubnormalAndPreservesNegativeZero) {
  kernel image;
  image.name = "sqrt";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "MUFU.SQRT", "R0,R1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t v[][2] = {{0x40800000, 0x40000000},
                           {1, 0},
                           {0x80000001, 0x80000000},
                           {0x80000000, 0x80000000},
                           {0x7f800000, 0x7f800000}};
  for (auto &x : v) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 1, x[0]);
    execution_context ctx;
    ASSERT_EQ(sass.step("sqrt", state, ctx).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 0), x[1]);
  }
}

TEST(SassFunctionalTest, Dsv4BfloatMinMaxNativeNanAndSignedZeroOracle) {
  kernel image;
  image.name = "bfedge";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "HMNMX2.BF16_V2", "R0,R1,R2,PT"),
      official_instruction(16, "HMNMX2.BF16_V2", "R3,R1,R2,!PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t cases[][4] = {
      {0x80000000, 0x00008000, 0x80008000, 0},
      {0x7fc07fc0, 0x3f80bf80, 0x3f80bf80, 0x3f80bf80}};
  for (auto &v : cases) {
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 1, v[0]);
    state.write_register(0, 2, v[1]);
    execution_context ctx;
    ASSERT_EQ(sass.step("bfedge", state, ctx).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 0), v[2]);
    ASSERT_EQ(sass.step("bfedge", state, ctx).status, step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 3), v[3]);
  }
}

TEST(SassFunctionalTest, Dsv4FiniteWarpMaxAndScalarHalf) {
  kernel image;
  image.name = "dsv4_quant_reduce";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "CREDUX.MAX.F32", "UR4,R2"),
      official_instruction(16, "F2F.F16.F32", "R3,R2")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0, 2, 0x3f800000); // 1
  state.write_register(1, 2, 0x40000000); // 2
  state.write_register(2, 2, 0x42c80000); // inactive 100
  state.write_register(0, 3, 0xffffffff);
  execution_context context;
  ASSERT_EQ(sass.step("dsv4_quant_reduce", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(4), 0x40000000u);
  ASSERT_EQ(sass.step("dsv4_quant_reduce", state, context).status,
            step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 3), 0x3c00u);
  EXPECT_EQ(state.read_register(1, 3), 0x4000u);
}

TEST(SassFunctionalTest, Ue8m0PackNativeBoundariesAndMergeAliasing) {
  kernel image;
  image.name = "ue8_pack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP",
                           "R3,R1,R2,R3", "@P0"),
      official_instruction(16, "F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP",
                           "R1,R1,R2,RZ", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t bits[] = {
      0, 0x80000000, 1, 0x003fffff, 0x00400000, 0x00400001,
      0x007fffff, 0x00800000, 0x00800001, 0x3f800000, 0x3f800001,
      0xbf800000, 0x80000001, 0x7f000000, 0x7f000001, 0x7f7fffff,
      0x7f800000, 0xff800000, 0x7fc12345, 0xff812345};
  const uint32_t code[] = {0, 0, 0, 0, 0, 1, 1, 1, 2, 127, 128,
                           127, 0, 254, 254, 254, 254, 254, 255, 255};
  for (unsigned i = 0; i < 20; ++i) {
    warp_state state;
    state.active_mask = 3;
    state.write_predicate(0, 0, true);
    state.write_predicate(1, 0, false);
    state.write_register(0, 1, bits[i]);
    state.write_register(0, 2, bits[19 - i]);
    state.write_register(0, 3, 0xbeefa59c);
    state.write_register(1, 3, 0x12345678);
    execution_context context;
    ASSERT_EQ(sass.step("ue8_pack", state, context).status,
              step_status::kAdvanced);
    const uint32_t packed = (code[i] << 8) | code[19 - i];
    EXPECT_EQ(state.read_register(0, 3), 0xa59c0000u | packed);
    EXPECT_EQ(state.read_register(1, 3), 0x12345678u);
    ASSERT_EQ(sass.step("ue8_pack", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 1), packed);
  }
}

TEST(SassFunctionalTest, Fp4ScaledBfloatUnpackNativeBoundaries) {
  kernel image;
  image.name = "fp4_unpack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(
      0, "F2FP.BF16.E2M1.UNPACK_B", "R1,R1,R2.reuse.H0", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t cases[][3] = {
      {0x80, 0x7f7f, 0x80000000}, {0xf7, 0x0000, 0x81400140},
      {0x31, 0x0101, 0x00c00040}, {0xf7, 0xfefe, 0xff807f80},
      {0x00, 0xffff, 0x7fff7fff}, {0x72, 0x7f7f, 0x40c03f80},
      {0x22, 0x807f, 0x40003f80}};
  for (const auto &test : cases) {
    warp_state state;
    state.active_mask = 3;
    state.write_predicate(0, 0, true);
    state.write_predicate(1, 0, false);
    state.write_register(0, 1, test[0]);
    state.write_register(0, 2, test[1]);
    state.write_register(1, 1, 0x12345678);
    execution_context context;
    ASSERT_EQ(sass.step("fp4_unpack", state, context).status,
              step_status::kAdvanced);
    EXPECT_EQ(state.read_register(0, 1), test[2]);
    EXPECT_EQ(state.read_register(1, 1), 0x12345678u);
  }
}

TEST(SassFunctionalTest, Fp4BroadcastUnpackAllByteSelectors) {
  kernel image;
  image.name = "fp4_broadcast";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  for (unsigned payload = 0; payload < 4; ++payload)
    for (unsigned scale = 0; scale < 4; ++scale)
      image.instructions.push_back(official_instruction(
          (payload * 4 + scale) * 16, "F2FP.BF16.E2M1.UNPACK_B",
          "R4,R1.reuse.B" + std::to_string(payload) + ",R2.reuse.B" +
              std::to_string(scale)));
  image.instructions.push_back(official_instruction(
      256, "F2FP.BF16.E2M1.UNPACK_B", "R4,R1.B4,R2.B0"));
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  const uint32_t expected[4][4] = {
      {0x00400040u, 0x3f803f80u, 0x7f007f00u, 0x7fff7fffu},
      {0x80000000u, 0x80000000u, 0x80000000u, 0x7fff7fffu},
      {0x00600020u, 0x3fc03f00u, 0x7f407e80u, 0x7fff7fffu},
      {0x81400140u, 0xc0c040c0u, 0xff807f80u, 0x7fff7fffu}};
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 1, 0xf7318022);
  state.write_register(0, 2, 0xfffe7f00);
  execution_context context;
  for (unsigned payload = 0; payload < 4; ++payload)
    for (unsigned scale = 0; scale < 4; ++scale) {
      ASSERT_EQ(sass.step("fp4_broadcast", state, context).status,
                step_status::kAdvanced);
      EXPECT_EQ(state.read_register(0, 4), expected[payload][scale]);
    }
  EXPECT_EQ(sass.step("fp4_broadcast", state, context).status,
            step_status::kUnsupported);
  EXPECT_EQ(state.read_register(0, 4), expected[3][3]);
}

TEST(SassFunctionalTest, Dsv4VoteReturnsBallotAndAny) {
  kernel image;
  image.name = "dsv4_vote";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "VOTE.ANY", "R2,P1,P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 7;
  state.write_predicate(1, 0, true);
  state.write_predicate(3, 0, true); // inactive lane must not enter ballot
  execution_context context;
  ASSERT_EQ(sass.step("dsv4_vote", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 3; ++lane) {
    EXPECT_EQ(state.read_register(lane, 2), 2u);
    EXPECT_TRUE(state.read_predicate(lane, 1));
  }
}

TEST(SassFunctionalTest, Dsv4Fp8UnpackSpecialValuesAndInactiveLanes) {
  kernel image;
  image.name = "dsv4_fp8_unpack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "F2FP.F16.E4M3.UNPACK_B", "R2,R1", "@P0"),
      official_instruction(16, "F2FP.F16.E4M3.UNPACK_B", "R3,R1.H1", "@P0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 7;
  for (unsigned lane = 0; lane < 4; ++lane) {
    state.write_predicate(lane, 0, lane != 2);
    state.write_register(lane, 1, lane == 0 ? 0xff7f8000u : 0xfe7e8101u);
    state.write_register(lane, 2, 0xdeadbeefu);
    state.write_register(lane, 3, 0xdeadbeefu);
  }
  execution_context context;
  ASSERT_EQ(sass.step("dsv4_fp8_unpack", state, context).status, step_status::kAdvanced);
  ASSERT_EQ(sass.step("dsv4_fp8_unpack", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 2), 0x80000000u); // +0, -0
  EXPECT_EQ(state.read_register(0, 3), 0x7fff7fffu); // both NaN signs
  EXPECT_EQ(state.read_register(1, 2), 0x98001800u); // +/- 2^-9
  EXPECT_EQ(state.read_register(1, 3), 0xdf005f00u); // +/- 448
  for (unsigned lane : {2u, 3u}) {
    EXPECT_EQ(state.read_register(lane, 2), 0xdeadbeefu);
    EXPECT_EQ(state.read_register(lane, 3), 0xdeadbeefu);
  }
}

TEST(SassFunctionalTest, ScalarBfloatConversionRoundsTiesToEvenAndClearsHighBits) {
  kernel image;
  image.name = "scalar_bfloat";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "F2F.BF16.F32", "R2,R1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 15;
  const unsigned inputs[] = {0x3f808000, 0x3f818000, 0x80000000, 0x7f800000};
  const unsigned expected[] = {0x3f80, 0x3f82, 0x8000, 0x7f80};
  for (unsigned lane = 0; lane < 4; ++lane) {
    state.write_register(lane, 1, inputs[lane]);
    state.write_register(lane, 2, 0xdeadbeef);
  }
  execution_context context;
  ASSERT_EQ(sass.step("scalar_bfloat", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 4; ++lane)
    EXPECT_EQ(state.read_register(lane, 2), expected[lane]);
}

TEST(SassFunctionalTest, FchkDivisionRangeBoundariesAndGuard) {
  kernel image;
  image.name="fchk"; image.arch=architecture::kSm120; image.instruction_bytes=16;
  image.instructions={official_instruction(0,"FCHK","P0,R0,R1","@P1"),
                      official_instruction(16,"FCHK","P0,R0,28672")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  struct Case { unsigned a,b;bool slow; };
  for(auto c : {Case{24,100,true},Case{25,100,false},Case{254,128,false},
       Case{255,128,true},Case{127,1,true},Case{127,2,false},Case{127,251,false},
       Case{127,252,true},Case{25,149,false},Case{25,150,true},Case{129,2,true},
       Case{128,2,false},Case{0,127,true},Case{127,0,true}}) {
    warp_state state;execution_context context;
    for(unsigned lane=0;lane<32;++lane){
      state.write_register(lane,0,(c.a<<23)|(lane&1 ? 0x807fffff : 0));
      state.write_register(lane,1,(c.b<<23)|(lane&2 ? 0x803fffff : 0));
      state.write_predicate(lane,0,!c.slow);state.write_predicate(lane,1,lane<16);
    }
    ASSERT_EQ(sass.step("fchk",state,context).status,step_status::kAdvanced);
    for(unsigned lane=0;lane<32;++lane)EXPECT_EQ(state.read_predicate(lane,0),lane<16?c.slow:!c.slow);
  }
  warp_state state;execution_context context;state.pc=16;
  for(unsigned lane=0;lane<32;++lane)state.write_register(lane,0,0x3f800000);
  ASSERT_EQ(sass.step("fchk",state,context).status,step_status::kAdvanced);
  EXPECT_FALSE(state.read_predicate(0,0));
}

TEST(SassFunctionalTest, FfmaSaturatesNaNAndSignedZeroToPositiveZero) {
  kernel image;image.name="fma_sat";image.arch=architecture::kSm120;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"FFMA.SAT","R0,R0,R1,R2")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  const unsigned values[]={0,0x80000000u,1,0x3f000000u,0xbf800000u,0x40000000u,0x7fc12345u,0x7f800000u};
  const unsigned expected[]={0,0,1,0x3f000000u,0,0x3f800000u,0,0x3f800000u};
  warp_state state;execution_context context;
  for(unsigned lane=0;lane<32;++lane){state.write_register(lane,0,values[lane%8]);state.write_register(lane,1,0x3f800000);state.write_register(lane,2,0x80000000);}
  ASSERT_EQ(sass.step("fma_sat",state,context).status,step_status::kAdvanced);
  for(unsigned lane=0;lane<32;++lane)EXPECT_EQ(state.read_register(lane,0),expected[lane%8]);
}

TEST(SassFunctionalTest, FaddFtzSatHandlesBoundsNaNAndSubnormals) {
  kernel image; image.name="add_sat"; image.arch=architecture::kSm120; image.instruction_bytes=16;
  image.instructions={official_instruction(0,"FADD.FTZ.SAT","R0,R0,R1")};
  frontend sass; sass.add_kernel(std::move(image)); register_sm120_functional_semantics(sass);
  const unsigned values[]={0,0x80000000u,1,0x3f000000u,0xbf800000u,0x40000000u,0x7fc12345u,0x7f800000u};
  const unsigned expected[]={0,0,0,0x3f000000u,0,0x3f800000u,0,0x3f800000u};
  warp_state state; execution_context context;
  for(unsigned lane=0;lane<32;++lane) {
    state.write_register(lane,0,values[lane%8]);state.write_register(lane,1,0x80000000u);
  }
  ASSERT_EQ(sass.step("add_sat",state,context).status,step_status::kAdvanced);
  for(unsigned lane=0;lane<32;++lane) EXPECT_EQ(state.read_register(lane,0),expected[lane%8]);
}

TEST(SassFunctionalTest, DirectedFtzFmaFlushesInputsAndResultsAndRoundsDown) {
  const unsigned inputs[][3] = {
      {0x3f800000, 0x3f800000, 0x8d800000}, // Tiny normal residual affects RM.
      {1, 0x7f000000, 0},                 // Subnormal input is flushed first.
      {0x00800000, 0x3f000000, 0},        // Subnormal result is flushed.
      {0x80800000, 0x3f000000, 0x80000000},
      {0x3f800000, 0x3f800000, 0xbf800000},
      {0x7f7fffff, 0x40000000, 0}};
  const unsigned expected[] = {0x3f7fffff, 0, 0, 0x80000000, 0x80000000, 0x7f7fffff};
  kernel image; image.name="directed_ftz"; image.arch=architecture::kSm120; image.instruction_bytes=16;
  image.instructions={official_instruction(0,"FFMA.FTZ.RM","R0,R0,R1,R2")};
  frontend sass; sass.add_kernel(std::move(image)); register_sm120_functional_semantics(sass);
  warp_state state; execution_context context;
  for (unsigned lane=0; lane<32; ++lane)
    for (unsigned j=0; j<3; ++j) state.write_register(lane,j,inputs[lane%6][j]);
  ASSERT_EQ(sass.step("directed_ftz",state,context).status,step_status::kAdvanced);
  for (unsigned lane=0; lane<32; ++lane) EXPECT_EQ(state.read_register(lane,0),expected[lane%6]);
}

TEST(SassFunctionalTest, BfloatThreeWayMinmaxReadsBothHalvesBeforeAliasedWrite) {
  for(bool minimum : {false,true}) {
    kernel image;image.name="bf16minmax3";image.arch=architecture::kSm120;image.instruction_bytes=16;
    image.instructions={official_instruction(0,"VHMNMX.BF16_V2",minimum?"R4,R4,R5,R6,PT":"R4,R4,R5,R6,!PT")};
    frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
    warp_state state;state.active_mask=1;
    state.write_register(0,4,0x80003f80);state.write_register(0,5,0x4000bf80);state.write_register(0,6,0x00004040);
    execution_context context;auto result=sass.step("bf16minmax3",state,context);
    ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
    EXPECT_EQ(state.read_register(0,4),minimum?0x8000bf80u:0x40004040u);
  }
}

TEST(SassFunctionalTest, BfloatRegisterFmaModifiersAliasingAndSingleRounding) {
  kernel image;
  image.name = "bf16_register_fma";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "HFMA2.BF16_V2", "R0,-RZ.H0_H0,RZ.H0_H0,|R0|"),
      official_instruction(16, "HFMA2.BF16_V2", "R1,-R0.H1_H1,R2.H0_H0,R1"),
      official_instruction(32, "HFMA2.BF16_V2", "R3,R3,R4,R5")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 0, 0xc0008000u); // (-2,-0) -> (2,+0)
  state.write_register(0, 1, 0x40803f80u); // (4,1)
  state.write_register(0, 2, 0x00003f00u); // low = .5
  // 1.5*(1+1/128) = a BF16 midpoint. Tiny +epsilon must round up,
  // whereas FP32 intermediate rounding followed by BF16 rounds down.
  state.write_register(0, 3, 0xbfc03fc0u);
  state.write_register(0, 4, 0x3f813f81u);
  state.write_register(0, 5, 0x80010001u);
  execution_context context;
  ASSERT_EQ(sass.step("bf16_register_fma", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 0), 0x40000000u);
  ASSERT_EQ(sass.step("bf16_register_fma", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 1), 0x40400000u);
  ASSERT_EQ(sass.step("bf16_register_fma", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 3), 0xbfc23fc2u);
}

TEST(SassFunctionalTest, BfloatMinmaxImmediateOrderAndFp8MergeAliasing) {
  kernel image;
  image.name = "bf16_quant";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
    official_instruction(0, "HMNMX2.BF16_V2", "R0,R0,224,448,PT"),
    official_instruction(16, "HMNMX2.BF16_V2", "R1,R1,-224,-448,!PT"),
    official_instruction(32, "F2FP.SATFINITE.E4M3.BF16.UNPACK_B_MERGE_C", "R0,R0,R2"),
    official_instruction(48, "F2FP.SATFINITE.E4M3.BF16.UNPACK_B_MERGE_C", "R0,R1,R0")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0,0,0x44004400u); // 512 in both halves
  state.write_register(0,1,0xc400c400u); // -512 in both halves
  state.write_register(0,2,0xbeef1234u);
  state.write_register(1,0,0xff887f81u); // signaling NaNs, both signs
  state.write_register(1,1,0x7f81ff88u);
  state.write_register(1,2,0xbeef1234u);
  execution_context context;
  ASSERT_EQ(sass.step("bf16_quant",state,context).status,step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0,0),0x436043e0u);
  EXPECT_EQ(state.read_register(1,0),0x436043e0u);
  ASSERT_EQ(sass.step("bf16_quant",state,context).status,step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0,1),0xc360c3e0u);
  EXPECT_EQ(state.read_register(1,1),0xc360c3e0u);
  ASSERT_EQ(sass.step("bf16_quant",state,context).status,step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0,0),0x1234767eu);
  EXPECT_EQ(state.read_register(1,0),0x1234767eu);
  ASSERT_EQ(sass.step("bf16_quant",state,context).status,step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0,0),0x767ef6feu);
  EXPECT_EQ(state.read_register(1,0),0x767ef6feu);
}

TEST(SassFunctionalTest, GreaterEqualUnorderedFlushesBeforePredicateAnd) {
  kernel image;
  image.name = "geu_ftz";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "FSETP.GEU.FTZ.AND", "P0,PT,R0,R1,P1")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  struct Case { uint32_t a, b; bool expected; };
  const Case cases[] = {{0x80000001,0,true}, {0,1,true},
    {0x7fc00000,0,true}, {0,0x7fc00000,true}, {0xbf800000,0,false},
    {0x3f800000,0x3f800000,true}, {0x7f800000,0x3f800000,true}};
  for (const auto &test : cases) {
    for (bool gate : {false,true}) {
      warp_state state;
      state.active_mask = 1;
      state.write_register(0,0,test.a);state.write_register(0,1,test.b);
      state.write_predicate(0,1,gate);
      state.write_predicate(1,0,true);
      execution_context context;
      auto result = sass.step("geu_ftz",state,context);
      ASSERT_EQ(result.status,step_status::kAdvanced) << result.detail;
      EXPECT_EQ(state.read_predicate(0,0),test.expected && gate);
      EXPECT_TRUE(state.read_predicate(1,0));
    }
  }
}

TEST(SassFunctionalTest, LeaHighCarryOutputPreservesAliasedSources) {
  kernel image;
  image.name="lea_high_output";image.arch=architecture::kSm120;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"LEA.HI","R1,P0,R0,R1,R2,0x1d")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  struct Case {uint32_t a,b,c;};
  for(const auto &v: {Case{0,0,0},Case{0xffffffffu,0xffffffffu,0xffffffffu},
                     Case{8,0xffffffffu,0},Case{7,0xffffffffu,0},Case{0,0xffffffffu,1}}) {
    warp_state state;state.active_mask=1;
    state.write_register(0,0,v.a);state.write_register(0,1,v.b);state.write_register(0,2,v.c);
    state.write_register(1,1,0xdeadbeef);state.write_predicate(1,0,true);
    execution_context context;
    auto result=sass.step("lea_high_output",state,context);
    ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
    const uint64_t joined=(uint64_t(v.c)<<32)|v.a;
    uint64_t sum=uint64_t(uint32_t((joined<<29)>>32))+v.b;
    EXPECT_EQ(state.read_register(0,1),uint32_t(sum));
    EXPECT_EQ(state.read_predicate(0,0),(sum>>32)!=0);
    EXPECT_EQ(state.read_register(1,1),0xdeadbeefu);EXPECT_TRUE(state.read_predicate(1,0));
  }
}

TEST(SassFunctionalTest, BfloatSatfinitePackClampsOverflowAndKeepsNan) {
  kernel image;image.name="bf16_sat";image.arch=architecture::kSm120;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"F2FP.SATFINITE.BF16.F32.PACK_AB","R0,R0,R1")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  const uint32_t input[]={0,0x80000000,1,0x80000001,0x7fff,0x8000,0x8001,0x18000,
    0x3f808000,0x3f818000,0x7f7f0000,0x7f7f7fff,0x7f7f8000,0x7f800000,
    0xff7fffff,0xff800000,0x7fc12345,0xff812345};
  const uint32_t half[]={0,0x8000,0,0x8000,0,0,1,2,0x3f80,0x3f82,
    0x7f7f,0x7f7f,0x7f7f,0x7f7f,0xff7f,0xff7f,0x7fff,0x7fff};
  warp_state state;state.active_mask=(1u<<18)-1;
  for(unsigned i=0;i<18;++i){state.write_register(i,0,input[i]);state.write_register(i,1,input[17-i]);}
  state.write_register(31,0,0xdeadbeef);
  execution_context context;auto result=sass.step("bf16_sat",state,context);
  ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
  for(unsigned i=0;i<18;++i)EXPECT_EQ(state.read_register(i,0),(half[i]<<16)|half[17-i]);
  EXPECT_EQ(state.read_register(31,0),0xdeadbeefu);
}

TEST(SassFunctionalTest, SignedLessEqualOrReadsAliasedNegatedPredicate) {
  kernel image;image.name="le_or";image.arch=architecture::kSm120;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"ISETP.LE.OR","P1,PT,R0,UR29,!P1","@P2")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  const uint32_t values[]={0x80000000u,0xffffffffu,0,1,0x7fffffffu};
  warp_state state;state.active_mask=0xffff;state.write_uniform_register(29,0);
  for(unsigned lane=0;lane<32;++lane) {
    state.write_register(lane,0,values[lane%5]);state.write_predicate(lane,1,lane%2==0);
    state.write_predicate(lane,2,lane%3!=0);
  }
  execution_context context;auto result=sass.step("le_or",state,context);
  ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
  for(unsigned lane=0;lane<32;++lane) {
    bool old=lane%2==0;
    bool expected=lane<16&&lane%3!=0 ? (lane%5<=2)||!old : old;
    EXPECT_EQ(state.read_predicate(lane,1),expected)<<lane;
  }
}

TEST(SassFunctionalTest, WrappedLeftFunnelUsesHighWordAndModulo32Count) {
  for(unsigned shift:{0u,1u,8u,31u,32u,33u,63u,255u,0xffffffffu}) {
    for(bool immediate:{false,true}) {
      kernel image;image.name="shf_wrap";image.arch=architecture::kSm120;image.instruction_bytes=16;
      image.instructions={official_instruction(0,"SHF.L.W.U32.HI","R1,R1.reuse,"+(immediate?std::to_string(shift):"R3")+",R2")};
      frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
      warp_state state;state.active_mask=1;state.write_register(0,1,0x12345678);state.write_register(0,2,0xfedcba98);state.write_register(0,3,shift);
      execution_context context;auto result=sass.step("shf_wrap",state,context);ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
      unsigned n=shift&31;uint32_t expected=n?(0xfedcba98u<<n)|(0x12345678u>>(32-n)):0xfedcba98u;
      EXPECT_EQ(state.read_register(0,1),expected);
    }
  }
}

TEST(SassFunctionalTest, UniformSignedGreaterOrHandlesSignBoundaryAndNegatedPredicate) {
  for(uint32_t a:{0u,31u,32u,0x7fffffffu,0x80000000u,0xffffffffu})
  for(uint32_t b:{0u,31u,0x7fffffffu,0x80000000u,0xffffffffu})
  for(bool input:{false,true}) {
    kernel image;image.name="uisetp_gt_or";image.arch=architecture::kSm120;image.instruction_bytes=16;
    image.instructions={official_instruction(0,"UISETP.GT.OR","UP1,UPT,UR11,UR12,!UP1")};
    frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
    warp_state state;state.write_uniform_register(11,a);state.write_uniform_register(12,b);state.write_uniform_predicate(1,input);
    execution_context context;auto result=sass.step("uisetp_gt_or",state,context);ASSERT_EQ(result.status,step_status::kAdvanced)<<result.detail;
    EXPECT_EQ(state.read_uniform_predicate(1),(static_cast<int32_t>(a)>static_cast<int32_t>(b))||!input);
  }
  kernel image;image.name="uisetp_immediate";image.arch=architecture::kSm120;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"UISETP.GT.OR","UP0,UPT,UR11,0x1f,!UP1")};
  frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  warp_state state;state.write_uniform_register(11,32);state.write_uniform_predicate(1,true);
  execution_context context;ASSERT_EQ(sass.step("uisetp_immediate",state,context).status,step_status::kAdvanced);
  EXPECT_TRUE(state.read_uniform_predicate(0));
}

TEST(SassFunctionalTest, NativeFp32PairsSupportAbsBroadcastAndReversedSources) {
  kernel image; image.name="native_pairs"; image.arch=architecture::kSm120; image.instruction_bytes=16;
  image.instructions={
    official_instruction(0,"FADD2","R4,R4.F32x2.HI_LO,|R4|.F32x2.HI_LO"),
    official_instruction(16,"FMUL2","R6,R4.F32x2.HI_LO,-1"),
    official_instruction(32,"FADD2","R8,R4.F32x2.HI_LO,R6.F32x2.LO_HI"),
    official_instruction(48,"FMUL2","R10,R8.F32x2.HI_LO,0.00390625")};
  frontend sass; sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
  warp_state state;state.active_mask=1;
  state.write_register(0,4,0xc0000000);state.write_register(0,5,0x40400000);
  execution_context context;
  for(unsigned i=0;i<4;++i){auto r=sass.step("native_pairs",state,context);ASSERT_EQ(r.status,step_status::kAdvanced)<<r.detail;}
  EXPECT_EQ(state.read_register(0,4),0u);EXPECT_EQ(state.read_register(0,5),0x40c00000u);
  EXPECT_EQ(state.read_register(0,8),0xc0c00000u);EXPECT_EQ(state.read_register(0,9),0x40c00000u);
  EXPECT_EQ(state.read_register(0,10),0xbcc00000u);EXPECT_EQ(state.read_register(0,11),0x3cc00000u);
}


TEST(SassFunctionalTest, NativeUniformArithmeticAliasesPreserveWrapAndSignedComparison) {
 kernel image;image.name="aliases";image.arch=architecture::kSm120;image.instruction_bytes=16;
 image.instructions={official_instruction(0,"UIMAD.U32","UR4,UR2,0x8,UR3"),
  official_instruction(16,"IMAD.SHL","R4,R2,0x8,RZ"),
  official_instruction(32,"UISETP.EQ.OR","UP0,UPT,UR4,0x1,!UPT"),
  official_instruction(48,"UISETP.LT.OR","UP1,UPT,UR2,0x0,!UPT")};
 frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
 warp_state state;state.active_mask=1;state.write_uniform_register(2,0xffffffff);state.write_uniform_register(3,9);state.write_register(0,2,0xffffffff);
 execution_context context;
 for(unsigned i=0;i<4;++i){auto r=sass.step("aliases",state,context);ASSERT_EQ(r.status,step_status::kAdvanced)<<r.detail;}
 EXPECT_EQ(state.read_uniform_register(4),1u);EXPECT_EQ(state.read_register(0,4),0xfffffff8u);
 EXPECT_TRUE(state.read_uniform_predicate(0));EXPECT_TRUE(state.read_uniform_predicate(1));
}

TEST(SassFunctionalTest, Lop3PandReadsAliasedPredicateBeforeWritingIt) {
 kernel image;image.name="pand";image.arch=architecture::kSm120;image.instruction_bytes=16;
 image.instructions={official_instruction(0,"LOP3.LUT.PAND","P0,R4,R1,R2,RZ,0x30,P0")};
 frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
 warp_state state;state.active_mask=15;
 for(unsigned lane=0;lane<4;++lane){state.write_register(lane,1,lane&1?8:0);state.write_register(lane,2,0);state.write_predicate(lane,0,lane&2);}
 execution_context context;auto r=sass.step("pand",state,context);ASSERT_EQ(r.status,step_status::kAdvanced)<<r.detail;
 for(unsigned lane=0;lane<4;++lane){EXPECT_EQ(state.read_register(lane,4),lane&1?8u:0u);EXPECT_EQ(state.read_predicate(lane,0),lane==3);}
}

TEST(SassFunctionalTest, NumericReductionsOrderInfinitiesAndStillRejectNan) {
  kernel image;
  image.name = "infinite_reduce";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "CREDUX.MAX.F32", "UR4,R2"),
      official_instruction(16, "VHMNMX.BF16_V2", "R4,R4,R5,R6,PT"),
      official_instruction(32, "VHMNMX.BF16_V2", "R7,R4,R5,R6,!PT")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0, 2, 0xff800000u);
  state.write_register(1, 2, 0x7f800000u);
  state.write_register(2, 2, 0x7fc00000u); // inactive NaN
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 4, 0xff807f80u);
    state.write_register(lane, 5, 0x7f80ff80u);
    state.write_register(lane, 6, 0x3f803f80u);
  }
  execution_context context;
  ASSERT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(4), 0x7f800000u);
  ASSERT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 4), 0xff80ff80u);
  ASSERT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0, 7), 0x7f803f80u);
  state.pc = 0;
  state.write_register(0, 2, 0x7fc01234u);
  EXPECT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kUnsupported);
  state.pc = 16;
  state.write_register(0, 4, 0x7fc1u);
  EXPECT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kUnsupported);
}

TEST(SassFunctionalTest, DeepInferFp4PackRoundsTiesAndMergesBytes) {
  kernel image;
  image.name = "fp4_pack";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C",
                           "R8,R2,R3,R4"),
      official_instruction(16, "F2FP.SATFINITE.E2M1.BF16.UNPACK_B_MERGE_C",
                           "R9,R5,R4")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  state.write_register(0, 2, 0x3f400000);  // 0.75 -> 1 (even code 2)
  state.write_register(0, 3, 0xbfa00000);  // -1.25 -> -1 (even code 10)
  state.write_register(0, 4, 0x12345678);
  state.write_register(0, 5, 0x3f40bfa0);
  state.write_register(1, 2, 0x7fc00000);
  state.write_register(1, 3, 0xff800000);
  state.write_register(1, 4, 0);
  state.write_register(1, 5, 0x7fc0ff80);
  execution_context context;
  auto r = sass.step("fp4_pack", state, context);
  ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  EXPECT_EQ(state.read_register(0, 8), 0x3456782au);
  EXPECT_EQ(state.read_register(1, 8), 0x7fu);
  r = sass.step("fp4_pack", state, context);
  ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  EXPECT_EQ(state.read_register(0, 9), 0x3456782au);
  EXPECT_EQ(state.read_register(1, 9), 0x7fu);
}

TEST(SassFunctionalTest, DeepInferHalfFmaFp4ImmediateAndBfloatNegation) {
  kernel image;
  image.name = "mixed_half";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "FHFMA.BF16", "R8,R2.H1,R3.H1,R4"),
      official_instruction(16, "F2FP.BF16.E2M1.UNPACK_B",
                           "R9,R5,4.5736980577097704378e-41,H0"),
      official_instruction(32, "HADD2.BF16_V2", "R10,-R3.H1_H1,R2.H0_H0"),
      official_instruction(48, "BREV", "R11,R6"),
      official_instruction(64, "R2P", "R6.B1,0xf")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 1;
  state.write_register(0, 2, 0x40003f80);
  state.write_register(0, 3, 0x40403f00);
  state.write_register(0, 4, 0x3f800000);
  state.write_register(0, 5, 0xc2);
  state.write_register(0, 6, 0x500);
  execution_context context;
  for (unsigned i = 0; i < 5; ++i) {
    auto r = sass.step("mixed_half", state, context);
    ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  }
  EXPECT_EQ(state.read_register(0, 8), 0x40e00000u);
  EXPECT_EQ(state.read_register(0, 9), 0xc0003f80u);
  EXPECT_EQ(state.read_register(0, 10), 0xc000c000u);
  EXPECT_EQ(state.read_register(0, 11), 0xa00000u);
  EXPECT_TRUE(state.read_predicate(0, 0));
  EXPECT_FALSE(state.read_predicate(0, 1));
  EXPECT_TRUE(state.read_predicate(0, 2));
  EXPECT_FALSE(state.read_predicate(0, 3));
}

TEST(SassFunctionalTest, DeepInferTopkMaskUnorderedComparisonAndPredicateByte) {
  kernel image;
  image.name = "topk_scalar";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "FSET.BF.GT.AND", "R8,R2,R3,PT"),
      official_instruction(16, "FSET.BF.EQ.AND", "R9,R2,R3,PT"),
      official_instruction(32, "FSETP.LEU.AND", "P0,PT,R2,R3,PT"),
      official_instruction(48, "FSETP.NEU.OR", "P1,PT,R2,R3,PT"),
      official_instruction(64, "P2R.B1", "R10,R4,0x3"),
      official_instruction(80, "FLO.U32.SH", "R11,R5"),
      official_instruction(96, "F2FP.RELU.BF16.F32.PACK_AB", "R12,R3,R2")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.active_mask = 3;
  for (unsigned l = 0; l < 2; ++l) {
    state.write_register(l, 2, l ? 0x7fffffff : 0x3f800000);
    state.write_register(l, 3, 0xbf800000);
    state.write_register(l, 4, 0xa5a504ff);
    state.write_register(l, 5, 0x100);
  }
  execution_context context;
  for (unsigned i = 0; i < 7; ++i) {
    auto r = sass.step("topk_scalar", state, context);
    ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  }
  EXPECT_EQ(state.read_register(0, 8), 0xffffffffu);
  EXPECT_EQ(state.read_register(1, 8), 0u);
  EXPECT_EQ(state.read_register(0, 9), 0u);
  EXPECT_EQ(state.read_register(1, 9), 0u);
  EXPECT_FALSE(state.read_predicate(0, 0));
  EXPECT_TRUE(state.read_predicate(1, 0));
  EXPECT_EQ(state.read_register(0, 10), 0xa5a506ffu);
  EXPECT_EQ(state.read_register(1, 10), 0xa5a507ffu);
  EXPECT_EQ(state.read_register(0, 11), 23u);
  EXPECT_EQ(state.read_register(0, 12), 0x3f80u);
}

TEST(SassFunctionalTest, DeepInferUniformBitOpsAndScalarBfloatAdd) {
  kernel image;
  image.name = "uniform_bits";
  image.arch = architecture::kSm120;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "UBREV", "UR4,UR6"),
                        official_instruction(16, "UFLO.U32.SH", "UR5,UR6"),
                        official_instruction(32, "FHADD.BF16", "R8,R2.H1,R3")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm120_functional_semantics(sass);
  warp_state state;
  state.write_uniform_register(6, 0x100);
  state.write_register(0, 2, 0x40003f80);
  state.write_register(0, 3, 0x40400000);
  execution_context context;
  for (unsigned i = 0; i < 3; ++i) {
    auto r = sass.step("uniform_bits", state, context);
    ASSERT_EQ(r.status, step_status::kAdvanced) << r.detail;
  }
  EXPECT_EQ(state.read_uniform_register(4), 0x800000u);
  EXPECT_EQ(state.read_uniform_register(5), 23u);
  EXPECT_EQ(state.read_register(0, 8), 0x40a00000u);
}
}
