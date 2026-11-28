#include "../test_support.h"

namespace {

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
    // BFLY checks the upper clamp only. Lanes 8..15 and 24..31
    // may select below their segment; there is no additional lower clamp.
    const bool valid = (lane & 8) != 0;
    EXPECT_EQ(state.read_register(lane, 9), 100 + (valid ? lane ^ 8 : lane));
    EXPECT_EQ(state.read_predicate(lane, 3), valid);
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

TEST(SassFunctionalTest, NumericReductionsHandleNanAndBfloatMinmaxStaysChecked) {
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
  // The shared reduction backend implements non-propagating MIN/MAX:
  // a numeric operand wins over NaN without the .NAN modifier.
  EXPECT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kAdvanced);
  EXPECT_EQ(state.read_uniform_register(4), 0x7f800000u);
  state.pc = 16;
  state.write_register(0, 4, 0x7fc1u);
  EXPECT_EQ(sass.step("infinite_reduce", state, context).status, step_status::kUnsupported);
}

TEST(SassFunctionalTest, Fp32RoundingFtzSaturationModifierProduct) {
  for (const std::string base : {"FADD", "FMUL", "FFMA"})
    for (bool uniform : {false, true})
      for (bool ftz : {false, true})
        for (bool sat : {false, true})
          for (const std::string rounding : {"", ".RM", ".RP", ".RZ"})
            for (unsigned sample = 0; sample < 3; ++sample) {
              const bool mul = base == "FMUL", fma = base == "FFMA";
              uint32_t a = mul ? 0x3f800001u : 0x3f800000u;
              uint32_t b = mul ? 0x3f800001u : fma ? 0x3f800000u : 0x33000000u;
              uint32_t c = 0x33000000u;
              uint32_t expected = mul ? 0x3f800002u : 0x3f800000u;
              if (rounding == ".RP") ++expected;
              if (sat) expected = 0x3f800000u;
              if (sample == 1) {
                a = 0xbf800000u; b = (mul || fma) ? 0x3f800000u : 0;
                c = 0; expected = sat ? 0 : 0xbf800000u;
              } else if (sample == 2) {
                a = 1; b = (mul || fma) ? 0x3f800000u : 0;
                c = 0; expected = ftz ? 0 : 1;
              }
              const std::string opcode = (uniform ? "U" : "") + base +
                  (ftz ? ".FTZ" : "") + rounding + (sat ? ".SAT" : "");
              const std::string r = uniform ? "UR" : "R";
              kernel image; image.name="modifier_product";
              image.arch=architecture::kSm120; image.instruction_bytes=16;
              image.instructions={official_instruction(0,opcode,r+"8,"+r+"2,"+r+"3"+(fma?","+r+"4":""))};
              frontend sass; sass.add_kernel(std::move(image));
              register_sm120_functional_semantics(sass);
              warp_state state; state.active_mask=1;
              state.write_register(0,2,a);state.write_register(0,3,b);state.write_register(0,4,c);
              state.write_uniform_register(2,a);state.write_uniform_register(3,b);state.write_uniform_register(4,c);
              execution_context context;
              const auto result=sass.step("modifier_product",state,context);
              ASSERT_EQ(result.status,step_status::kAdvanced)<<opcode<<": "<<result.detail;
              EXPECT_EQ(uniform?state.read_uniform_register(8):state.read_register(0,8),expected)<<opcode<<" sample="<<sample;
            }
}

} // namespace
