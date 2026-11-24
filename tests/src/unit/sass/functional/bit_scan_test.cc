#include "../test_support.h"

TEST(NativeBitScan, SignedUnsignedUniformAndLaneForms) {
  struct Case { uint32_t input, unsigned_index, signed_index; };
  for (const char *opcode : {"FLO", "FLO.SH", "FLO.U32", "FLO.U32.SH",
                              "UFLO", "UFLO.SH", "UFLO.U32", "UFLO.U32.SH"}) {
    const std::string op(opcode);
    const bool uniform = op[0] == 'U';
    const bool sign = op.find(".U32") == std::string::npos;
    const bool shift = op.find(".SH") != std::string::npos;
    for (const Case c : {Case{0, UINT32_MAX, UINT32_MAX},
                         Case{UINT32_MAX, 31, UINT32_MAX},
                         Case{0x80000000, 31, 30}, Case{1, 0, 0},
                         Case{0xfffffffe, 31, 0}, Case{0x7fffffff, 30, 30}}) {
      kernel image;
      image.name = "bit_scan";
      image.arch = architecture::kSm100;
      image.instruction_bytes = 16;
      image.instructions = {official_instruction(0, opcode,
                                  uniform ? "UR4,UR4" : "R4,R4")};
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm100_functional_semantics(sass);
      warp_state state;
      state.active_mask = 0xa; // nonzero first active lane
      state.write_uniform_register(4, c.input);
      for (unsigned lane = 0; lane < 4; ++lane)
        state.write_register(lane, 4, c.input);
      execution_context context;
      auto result = sass.step("bit_scan", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      unsigned expected = sign ? c.signed_index : c.unsigned_index;
      if (shift && expected != UINT32_MAX) expected = 31 - expected;
      if (uniform) EXPECT_EQ(state.read_uniform_register(4), expected) << opcode;
      else {
        EXPECT_EQ(state.read_register(1, 4), expected) << opcode;
        EXPECT_EQ(state.read_register(3, 4), expected) << opcode;
        EXPECT_EQ(state.read_register(0, 4), c.input);
        EXPECT_EQ(state.read_register(2, 4), c.input);
      }
    }
  }
}
