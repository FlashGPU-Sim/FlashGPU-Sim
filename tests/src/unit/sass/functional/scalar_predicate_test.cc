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

TEST(SassFunctionalTest, IntegerCompareModifierProductAndDualOutputs) {
  const std::array<std::string,6> comparisons={"LT","LE","GT","GE","EQ","NE"};
  for(unsigned cmp=0;cmp<6;++cmp) for(bool uniform:{false,true})
    for(bool unsign:{false,true}) for(bool extended:{false,true})
      for(bool extension:{false,true}) for(bool input:{false,true})
        for(const std::string boolean:{"AND","OR","XOR"}) {
          const std::string op=(uniform?"UISETP.":"ISETP.")+comparisons[cmp]+(unsign?".U32":"")+"."+boolean+(extended?".EX":"");
          const std::string p=uniform?"UP":"P",r=uniform?"UR":"R";
          kernel image;image.name="integer_product";image.arch=architecture::kSm120;image.instruction_bytes=16;
          image.instructions={official_instruction(0,op,p+"0,"+p+"1,"+r+"2,"+r+"3,"+p+"2"+(extended?","+p+"3":""))};
          frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
          warp_state state;state.active_mask=1;
          const uint32_t a=0xffffffffu,b=extended?a:1;
          state.write_register(0,2,a);state.write_register(0,3,b);state.write_uniform_register(2,a);state.write_uniform_register(3,b);
          state.write_predicate(0,2,input);state.write_predicate(0,3,extension);state.write_uniform_predicate(2,input);state.write_uniform_predicate(3,extension);
          execution_context context;const auto result=sass.step("integer_product",state,context);
          ASSERT_EQ(result.status,step_status::kAdvanced)<<op<<result.detail;
          const bool value=extended?extension:cmp<2?!unsign:cmp<4?unsign:cmp==5;
          auto combine=[&](bool v){return boolean=="AND"?v&&input:boolean=="OR"?v||input:v!=input;};
          EXPECT_EQ(uniform?state.read_uniform_predicate(0):state.read_predicate(0,0),combine(value))<<op;
          EXPECT_EQ(uniform?state.read_uniform_predicate(1):state.read_predicate(0,1),combine(!value))<<op;
        }
}

} // namespace
