#include "../test_support.h"

namespace {

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

TEST(SassFunctionalTest, FloatingPredicateProductWritesBothOutputs) {
  const std::array<std::string,14> comparisons={"LT","LE","GT","GE","EQ","NE","LTU","LEU","GTU","GEU","EQU","NEU","NUM","NAN"};
  const std::array<bool,14> finite={true,true,false,false,false,true,true,true,false,false,false,true,true,false};
  for(unsigned cmp=0;cmp<comparisons.size();++cmp)
    for(bool uniform:{false,true}) for(bool ftz:{false,true})
      for(bool nan:{false,true}) for(bool input:{false,true})
        for(const std::string boolean:{"AND","OR","XOR"}) {
          const std::string op=(uniform?"UFSETP.":"FSETP.")+comparisons[cmp]+(ftz?".FTZ":"")+"."+boolean;
          const std::string p=uniform?"UP":"P",r=uniform?"UR":"R";
          kernel image;image.name="compare_product";image.arch=architecture::kSm120;image.instruction_bytes=16;
          image.instructions={official_instruction(0,op,p+"0,"+p+"1,"+r+"2,"+r+"3,"+p+"2")};
          frontend sass;sass.add_kernel(std::move(image));register_sm120_functional_semantics(sass);
          warp_state state;state.active_mask=1;
          state.write_register(0,2,nan?0x7fc00000u:0x3f800000u);state.write_register(0,3,0x40000000u);
          state.write_uniform_register(2,nan?0x7fc00000u:0x3f800000u);state.write_uniform_register(3,0x40000000u);
          state.write_predicate(0,2,input);state.write_uniform_predicate(2,input);
          execution_context context;const auto result=sass.step("compare_product",state,context);
          ASSERT_EQ(result.status,step_status::kAdvanced)<<op<<result.detail;
          const bool value=nan?(cmp>=6&&cmp!=12):finite[cmp];
          auto combine=[&](bool v){return boolean=="AND"?v&&input:boolean=="OR"?v||input:v!=input;};
          EXPECT_EQ(uniform?state.read_uniform_predicate(0):state.read_predicate(0,0),combine(value))<<op;
          EXPECT_EQ(uniform?state.read_uniform_predicate(1):state.read_predicate(0,1),combine(!value))<<op;
        }
}

} // namespace
