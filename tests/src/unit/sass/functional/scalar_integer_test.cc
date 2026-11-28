#include "../test_support.h"

namespace {

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

} // namespace
