#include <gtest/gtest.h>

#include <limits>

#include "gpgpu-sim/flash/sass/decode/operand_parser.h"

namespace {
using namespace flash_gpgpu_sim::sass;

TEST(SassAddressExpressionTest, RejectsTrailingOperators) {
  for (const char *text :
       {"[R0+]", "[R0-]", "[UR4+0x10+]", "c[0x0][R0+]", "[R0][UR4-]"}) {
    EXPECT_EQ(parse_operand(text).kind, operand_kind::kOpaque) << text;
  }
}

TEST(SassAddressExpressionTest, PreservesNegativeDisplacements) {
  for (const char *text : {"[R0+-0x40]", "[R0-0x40]"}) {
    const operand value = parse_operand(text);
    ASSERT_EQ(value.kind, operand_kind::kMemoryAddress) << text;
    ASSERT_EQ(value.address_groups.size(), 1u);
    ASSERT_EQ(value.address_groups[0].terms.size(), 2u);
    EXPECT_EQ(value.address_groups[0].terms[1].immediate, -64);
  }
}

TEST(SassAddressExpressionTest, NegatesMinimumImmediateModulo64Bits) {
  const operand value = parse_operand("[R0-0x8000000000000000]");
  ASSERT_EQ(value.kind, operand_kind::kMemoryAddress);
  ASSERT_EQ(value.address_groups.size(), 1u);
  ASSERT_EQ(value.address_groups[0].terms.size(), 2u);
  EXPECT_EQ(value.address_groups[0].terms[1].immediate,
            std::numeric_limits<int64_t>::min());
}

}  // namespace
