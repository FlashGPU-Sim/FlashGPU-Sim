#include <gtest/gtest.h>

#include "gpgpu-sim/flash/sass/decode/operand_parser.h"

namespace {
using namespace flash_gpgpu_sim::sass;

TEST(SassAddressModifiersTest, RejectsValueModifiersLostByAddressIR) {
  for (const char *text :
       {"[|R0|]", "[R0.H0_H0]", "[R0.ROW]", "c[0][|UR0|]", "-R0[R2]", "~P0[R2]",
        "desc[-UR4]", "desc[|UR4|]", "desc[UR4][-R2.64]", "desc[UR4][|R2|.64]",
        "gdesc[~UR4]", "gdesc[UR4.H1_H1]"}) {
    const auto parsed = parse_operand(text);
    EXPECT_EQ(parsed.kind, operand_kind::kOpaque) << text;
    EXPECT_EQ(parsed.text, text);
  }
}

TEST(SassAddressModifiersTest, KeepsRepresentableAddressForms) {
  const auto address = parse_operand("[R2.64+-0x40]");
  ASSERT_EQ(address.kind, operand_kind::kMemoryAddress);
  EXPECT_TRUE(address.address_groups[0].terms[0].wide);
  EXPECT_EQ(address.address_groups[0].terms[1].immediate, -64);
  const auto descriptor = parse_operand("desc[UR4][R2.64+-0x40]");
  ASSERT_EQ(descriptor.kind, operand_kind::kDescriptorAddress);
  EXPECT_EQ(descriptor.descriptor_register, 4u);
  EXPECT_EQ(descriptor.address_register, 2u);
  EXPECT_EQ(descriptor.address_offset, -64);
  const auto gmma = parse_operand("gdesc[UR4].tnspA.tnspB");
  ASSERT_EQ(gmma.kind, operand_kind::kGmmaDescriptor);
  EXPECT_TRUE(gmma.descriptor_transpose_a);
  EXPECT_TRUE(gmma.descriptor_transpose_b);
}

TEST(SassAddressModifiersTest, RejectsUnrepresentedGuardModifiers) {
  for (const char *guard : {"@~P0", "@|P0|", "@P0.H0_H0", "@P0.64"}) {
    instruction inst;
    inst.predicate_text = guard;
    inst.operand_text = "R0, R1";
    parse_instruction_operands(inst);
    EXPECT_FALSE(inst.operands_structured) << guard;
  }
  instruction inst;
  inst.predicate_text = "@!UP0";
  parse_instruction_operands(inst);
  EXPECT_TRUE(inst.operands_structured);
  EXPECT_TRUE(inst.guard.negated);
}

}  // namespace
