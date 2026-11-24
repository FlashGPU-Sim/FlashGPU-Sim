#include "../test_support.h"
#include "../quantization_reference.h"

namespace {

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

} // namespace
