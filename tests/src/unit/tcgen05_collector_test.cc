#include <gtest/gtest.h>
#include "gpgpu-sim/flash/sass/decode/operand_parser.h"
#include "gpgpu-sim/flash/sass/decode/tcgen_collector.h"
#include "gpgpu-sim/flash/tcgen05/tmem.h"
using namespace flash_gpgpu_sim;
using Op = tcgen05_collector_op;
namespace {
constexpr tcgen05_ws_half_signature shape{64, 16, 0, false};
std::vector<uint16_t> data(unsigned value) {
  return std::vector<uint16_t>(1024, value);
}
TEST(Tcgen05Collector, SlotsAndOwnersDoNotAlias) {
  tcgen05_ws_half_collectors state;
  for (unsigned slot = 0; slot < 4; ++slot)
    for (unsigned owner : {0u, 32u}) {
      auto v = data(slot + owner);
      ASSERT_TRUE(state.apply(owner, {slot, Op::fill}, shape, v));
    }
  for (unsigned slot = 0; slot < 4; ++slot)
    for (unsigned owner : {0u, 32u}) {
      auto v = data(999);
      ASSERT_TRUE(state.apply(owner, {slot, Op::use}, shape, v));
      EXPECT_EQ(v, data(slot + owner));
    }
}
TEST(Tcgen05Collector, LastuseAndDiscardInvalidateOnlySelectedSlot) {
  tcgen05_ws_half_collectors state;
  auto v = data(3);
  ASSERT_TRUE(state.apply(0, {0, Op::fill}, shape, v));
  ASSERT_TRUE(state.apply(0, {1, Op::fill}, shape, v));
  ASSERT_TRUE(state.apply(0, {0, Op::lastuse}, shape, v));
  EXPECT_EQ(v, data(3));
  EXPECT_FALSE(state.apply(0, {0, Op::use}, shape, v));
  v = data(7);
  ASSERT_TRUE(state.apply(0, {1, Op::discard}, shape, v));
  EXPECT_EQ(v, data(7));
  EXPECT_FALSE(state.apply(0, {1, Op::use}, shape, v));
  EXPECT_TRUE(state.apply(0, {0, Op::fill}, shape, v));
  EXPECT_TRUE(state.apply(0, {0, Op::lastuse}, shape, v));
  EXPECT_EQ(v, data(7));
}
TEST(Tcgen05Collector, RejectsMissingIncompatibleAndMalformedInputs) {
  tcgen05_ws_half_collectors state;
  auto v = data(3);
  EXPECT_FALSE(state.apply(0, {0, Op::use}, shape, v));
  EXPECT_FALSE(state.apply(0, {4, Op::fill}, shape, v));
  ASSERT_TRUE(state.apply(0, {0, Op::fill}, shape, v));
  v = data(99);
  for (auto other : {tcgen05_ws_half_signature{128, 16, 0, false},
                     tcgen05_ws_half_signature{64, 32, 0, false},
                     tcgen05_ws_half_signature{64, 16, 1, false},
                     tcgen05_ws_half_signature{64, 16, 0, true}})
    EXPECT_FALSE(state.apply(0, {0, Op::lastuse}, other, v));
  EXPECT_EQ(v, data(99));
  ASSERT_TRUE(state.apply(0, {0, Op::use}, shape, v));
  EXPECT_EQ(v, data(3));
  v.clear();
  EXPECT_FALSE(state.apply(0, {0, Op::fill}, shape, v));
  EXPECT_TRUE(state.apply(0, {0, Op::use}, shape, v));
}
TEST(Tcgen05Collector, CtaRetirementAndResetClearBanks) {
  tcgen05_tmem_manager_t memory;
  auto v = data(5);
  ASSERT_TRUE(memory.ws_half_b({0, 0}, 0, {0, Op::fill}, shape, v));
  ASSERT_TRUE(memory.ws_half_b({0, 1}, 0, {0, Op::fill}, shape, v));
  memory.clear_cta(0, 0);
  EXPECT_FALSE(memory.ws_half_b({0, 0}, 0, {0, Op::use}, shape, v));
  EXPECT_TRUE(memory.ws_half_b({0, 1}, 0, {0, Op::use}, shape, v));
  memory.reset();
  EXPECT_FALSE(memory.ws_half_b({0, 1}, 0, {0, Op::use}, shape, v));
}
TEST(Tcgen05Collector, NativeControlsAndContextGate) {
  for (unsigned slot = 0; slot < 4; ++slot)
    for (auto op : {Op::fill, Op::use, Op::lastuse, Op::discard}) {
      std::string text = "gdesc[UR4]";
      if (op == Op::use || op == Op::lastuse) text += ".B_REUSE";
      if (op == Op::use || op == Op::fill) text += ".B_KEEP";
      if (slot) text += ".BUFFER" + std::to_string(slot);
      auto raw = text;
      tcgen05_ws_collector_control control;
      ASSERT_TRUE(sass::decode_ws_collector_operand(text, control));
      EXPECT_EQ(text, "gdesc[UR4]");
      EXPECT_EQ(control.slot, slot);
      EXPECT_EQ(control.op, op);
      sass::instruction inst;
      inst.opcode = "UTCHMMA.WS";
      inst.operand_text =
          "gdesc[UR2]," + raw + ",tmem[UR6],tmem[URZ],idesc[UR8],UP0";
      sass::parse_instruction_operands(inst);
      ASSERT_TRUE(inst.operands_structured);
      EXPECT_EQ(inst.operands[1].descriptor_register, 4u);
      if (raw != "gdesc[UR4]") {
        inst.opcode = "UTCHMMA";
        sass::parse_instruction_operands(inst);
        EXPECT_FALSE(inst.operands_structured);
      }
    }
  for (auto suffix : {".BUFFER4", ".B_KEEP.B_KEEP", ".BUFFER1.BUFFER2",
                      ".B_REUSE.BAD", "."}) {
    std::string text = std::string("gdesc[UR4]") + suffix;
    tcgen05_ws_collector_control c;
    EXPECT_FALSE(sass::decode_ws_collector_operand(text, c));
  }
}
}  // namespace
