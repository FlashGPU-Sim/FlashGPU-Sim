#include <gtest/gtest.h>
#include <cstring>

#include "gpgpu-sim/flash/sass/decode/operand_parser.h"
#include "gpgpu-sim/flash/sass/functional/functional.h"

namespace {
using namespace flash_gpgpu_sim::sass;

void check_signed_high_lea(bool uniform) {
  for (const auto arch : {architecture::kSm90, architecture::kSm120}) {
    frontend source;
    if (arch == architecture::kSm90)
      register_sm90_functional_semantics(source);
    else
      register_sm120_functional_semantics(source);
    kernel image;
    image.name = "signed_high_lea";
    image.arch = arch;
    image.instruction_bytes = kInstructionBytes;
    for (unsigned shift = 0; shift < 32; ++shift) {
      instruction inst;
      inst.pc = shift * kInstructionBytes;
      inst.opcode = uniform ? "ULEA.HI.SX32" : "LEA.HI.SX32";
      inst.operand_text =
          (uniform ? "UR0,UR1,UR2," : "R0,R1,R2,") + std::to_string(shift);
      inst.decoded = true;
      parse_instruction_operands(inst);
      ASSERT_TRUE(inst.operands_structured);
      image.instructions.push_back(inst);
    }
    source.add_kernel(std::move(image));
    for (const uint32_t bits :
         {0u, 1u, 0x7fffffffu, 0x80000000u, 0xfffffff0u, 0xffffffffu}) {
      for (const uint32_t addend : {0u, 9u, 0xffffffffu}) {
        warp_state state;
        state.active_mask = 1;
        if (uniform) {
          state.write_uniform_register(1, bits);
          state.write_uniform_register(2, addend);
        } else {
          state.write_register(0, 1, bits);
          state.write_register(0, 2, addend);
          state.write_register(1, 0, 0x12345678u);
        }
        execution_context context;
        for (unsigned shift = 0; shift < 32; ++shift) {
          SCOPED_TRACE(::testing::Message()
                       << "arch=" << static_cast<int>(arch) << " shift="
                       << shift << " bits=" << bits << " addend=" << addend);
          // Independent unsigned reference: sign-extend, shift, select HI.
          const uint64_t extended =
              uint64_t{bits} |
              ((bits & 0x80000000u) ? 0xffffffff00000000ull : 0ull);
          const uint32_t expected =
              static_cast<uint32_t>((extended << shift) >> 32) + addend;
          const auto result = source.step("signed_high_lea", state, context);
          ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
          EXPECT_EQ(uniform ? state.read_uniform_register(0)
                            : state.read_register(0, 0),
                    expected);
          if (!uniform) EXPECT_EQ(state.read_register(1, 0), 0x12345678u);
        }
      }
    }
  }
}

TEST(SassIntegerBoundaryTest, SignedHighLeaAllShiftAmounts) {
  check_signed_high_lea(false);
}

TEST(SassIntegerBoundaryTest, UniformSignedHighLeaAllShiftAmounts) {
  check_signed_high_lea(true);
}

TEST(SassIntegerBoundaryTest, BinaryWarpgroupMmaWrapsAccumulator) {
  struct ones_memory : functional_memory {
    bool read(memory_space, uint64_t, void *data, size_t bytes) const override {
      std::memset(data, 0xff, bytes);
      return true;
    }
    bool write(memory_space, uint64_t, const void *, size_t) override {
      return false;
    }
  } memory;
  instruction inst;
  inst.decoded = true;
  inst.opcode = "BGMMA.64x8x256.AND.POPC";
  inst.operand_text = "R8,R0,gdesc[UR4],R8,UPT,gsb0";
  parse_instruction_operands(inst);
  kernel image;
  image.name = "binary_mma_wrap";
  image.arch = architecture::kSm90;
  image.instruction_bytes = kInstructionBytes;
  image.instructions.push_back(inst);
  frontend source;
  source.add_kernel(std::move(image));
  register_sm90_functional_semantics(source);
  for (uint32_t initial : {0u, 0x7fffffffu, 0xffffff80u, 0xffffffffu}) {
    warp_state state;
    state.write_uniform_register(6, 1u << 16);  // B leading = 16 bytes.
    state.write_uniform_register(7, 2);         // B stride = 32 bytes.
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      for (unsigned reg = 0; reg < 4; ++reg) {
        state.write_register(lane, reg, 0xffffffffu);
        state.write_register(lane, 8 + reg, initial);
      }
    execution_context context{&memory};
    ASSERT_EQ(source.step("binary_mma_wrap", state, context).status,
              step_status::kAdvanced);
    const uint32_t expected = static_cast<uint32_t>(uint64_t{initial} + 256);
    for (unsigned lane = 0; lane < kWarpLanes; ++lane)
      for (unsigned reg = 0; reg < 4; ++reg)
        EXPECT_EQ(state.read_register(lane, 8 + reg), expected);
  }
}
}  // namespace
