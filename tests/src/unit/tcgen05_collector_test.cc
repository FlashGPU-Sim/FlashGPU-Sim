#include <gtest/gtest.h>
#include "gpgpu-sim/flash/sass/decode/operand_parser.h"
#include "gpgpu-sim/flash/sass/decode/tcgen_collector.h"
#include "gpgpu-sim/flash/tcgen05/tmem.h"
#include "gpgpu-sim/flash/tcgen05/ws_mask.h"
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

TEST(Tcgen05WsMask, PtxExamplesAndSegmentStartCounts) {
  const uint64_t base = (1ull << 39) | (2ull << 40) | (3ull << 48);
  // Bits written here in column order, from low to high, as in the manual.
  const char *patterns[] = {"0000111", "1110000", "1100001", "0011100", "0001110"};
  for (unsigned col = 0; col < 128; ++col)
    EXPECT_EQ(tcgen05_ws_zero_column(base, 128, 128, col), patterns[0][col % 7] == '1');
  for (unsigned col = 0; col < 128; ++col) {
    const auto raw = base | (1ull << 32);
    EXPECT_EQ(tcgen05_ws_zero_column(raw, 64, 128, col),
              patterns[col < 64 ? 1 : 0][(col % 64) % 7] == '1');
  }
  const uint64_t raw = base | (3ull << 32) | (1ull << 8) | (2ull << 16) | (1ull << 24);
  for (unsigned col = 0; col < 128; ++col)
    EXPECT_EQ(tcgen05_ws_zero_column(raw, 32, 128, col),
              patterns[col < 32 ? 1 : col < 64 ? 2 : col < 96 ? 3 : 4][(col % 32) % 7] == '1');
  for (unsigned col = 0; col < 256; ++col)
    EXPECT_FALSE(tcgen05_ws_zero_column(raw & ~(1ull << 39), 64, 256, col));
}

TEST(Tcgen05WsMask, ReservedBitsAndShiftBounds) {
  for (unsigned max : {0u, 8u, 16u, 32u}) {
    const unsigned code = max == 0 ? 0 : max == 8 ? 1 : max == 16 ? 2 : 3;
    EXPECT_EQ(tcgen05_ws_max_shift(code << 30), max);
    for (unsigned shift = 0; shift < 64; ++shift)
      EXPECT_EQ(tcgen05_ws_mask_supported(uint64_t(shift) << 56, 64, 64, max),
                shift <= max);
  }
  for (unsigned bit : {36u, 37u, 38u, 62u, 63u})
    EXPECT_FALSE(tcgen05_ws_mask_supported(1ull << bit, 64, 64, 0));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 32, 128, 32));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 64, 128, 3));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 64, 128, 33));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 0, 128, 0));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 64, 0, 0));
  EXPECT_FALSE(tcgen05_ws_mask_supported(0, 64, 63, 0));
  EXPECT_TRUE(tcgen05_ws_mask_supported(1ull << 39, 64, 64, 0));
}

TEST(Tcgen05WsMask, CollectorKeepsRawShiftWindowAcrossPerInstructionMasks) {
  for (bool transpose : {false, true}) {
    tcgen05_ws_half_collectors state;
    tcgen05_ws_half_signature signature{64, 16, 0, transpose, 32};
    std::vector<uint16_t> raw(96 * 16);
    for (unsigned c = 0; c < 96; ++c)
      for (unsigned k = 0; k < 16; ++k)
        raw[transpose ? k * 96 + c : c * 16 + k] = 100 + c + k;
    auto values = raw;
    ASSERT_TRUE(state.apply(0, {0, Op::fill}, signature, values));
    const uint64_t all_zero = (1ull << 39) | (255ull << 40) | (3ull << 32);
    ASSERT_TRUE(tcgen05_ws_select_b(all_zero, 64, 64, 16, 32, transpose, values));
    EXPECT_EQ(values, std::vector<uint16_t>(64 * 16));
    ASSERT_TRUE(state.apply(0, {0, Op::use}, signature, values));
    EXPECT_EQ(values, raw);
    ASSERT_TRUE(tcgen05_ws_select_b(32ull << 56, 64, 64, 16, 32, transpose, values));
    for (unsigned c = 0; c < 64; ++c)
      for (unsigned k = 0; k < 16; ++k)
        EXPECT_EQ(values[transpose ? k * 64 + c : c * 16 + k], 132 + c + k);
    auto mismatch = signature;
    mismatch.max_shift = 16;
    EXPECT_FALSE(state.apply(0, {0, Op::use}, mismatch, values));
    ASSERT_TRUE(state.apply(0, {0, Op::lastuse}, signature, values));
    EXPECT_EQ(values, raw);
    EXPECT_FALSE(state.apply(0, {0, Op::use}, signature, values));
  }
}

TEST(Tcgen05WsMask, RejectionDoesNotAlterOperandOrCollector) {
  tcgen05_ws_half_collectors state;
  auto values = data(17), previous = values;
  ASSERT_TRUE(state.apply(0, {0, Op::fill}, shape, values));
  EXPECT_FALSE(tcgen05_ws_select_b(1ull << 36, 64, 64, 16, 0, false, values));
  EXPECT_EQ(values, previous);
  EXPECT_FALSE(tcgen05_ws_select_b(0, 64, 128, 16, 0, false, values));
  EXPECT_EQ(values, previous);
  ASSERT_TRUE(state.apply(0, {0, Op::use}, shape, values));
  EXPECT_EQ(values, previous);
}

TEST(Tcgen05WsMask, B200StartCountSaturatesWithinFirstSpan) {
  // Apply the native B200 first-span clamp to unequal skip/use spans.
  // First span is one column; subsequent spans retain their full length.
  const char *bits[] = {"01110000000111", "10000000111000"};
  for (unsigned m : {32u, 64u, 128u}) {
    const unsigned width = 128 / (128 / m);
    for (unsigned fs : {0u, 1u}) {
      for (unsigned start : {7u, 63u, 64u, 254u, 255u}) {
        uint64_t raw = (1ull << 39) | (2ull << 40) | (6ull << 48);
        for (unsigned part = 0; part < 128 / m; ++part)
          raw |= (uint64_t(start) << (8 * part)) | (uint64_t(fs) << (32 + part));
        for (unsigned part = 0; part < 128 / m; ++part)
          for (unsigned col = 0; col < 14; ++col)
            EXPECT_EQ(tcgen05_ws_zero_column(raw, m, 128, part * width + col),
                      bits[fs][col] == '1');
      }
    }
  }
}
