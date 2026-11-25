#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "../../../libcuda/gpgpu_context.h"
#include "cuda-sim/opcodes.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
namespace {
void check_literals(const std::vector<std::pair<std::string, unsigned long long>> &values,
                    const char *name) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  const std::string path = std::string(name) + ".ptx";
  {
    std::ofstream f(path);
    f << ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry "
      << name << "() {\n.reg .b64 d;\n";
    for (const auto &v : values) f << "mov.b64 d, " << v.first << ";\n";
    f << "ret;\n}\n";
    ASSERT_TRUE(f.good());
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
  std::remove(path.c_str());
  ASSERT_NE(symbols, nullptr);
  auto *func = symbols->lookup_function(name);
  ASSERT_NE(func, nullptr);
  func->do_pdom();
  unsigned pc = func->get_start_PC();
  for (const auto &v : values) {
    const auto *inst = func->get_dyn_inst(pc);
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->src1().get_literal_value().u64, v.second) << v.first;
    pc += inst->inst_size();
  }
}
TEST(PtxIntegerLiteral, UnsignedSuffixAndAllRadices) {
  check_literals({{"0U", 0}, {"42U", 42}, {"0x3340U", 0x3340},
                  {"052U", 42}, {"0b101010U", 42}, {"0XFAU", 250},
                  {"0B11", 3}, {"052", 42}, {"-052U", 0ull - 42},
                  {"-0x2aU", 0ull - 42}, {"-0b101010U", 0ull - 42}, {"-1U", ~0ull}}, "literal_radices");
}
TEST(PtxIntegerLiteral, DecimalAboveSignedRangeWithoutSuffix) {
  check_literals({{"9223372036854775808", 0x8000000000000000ull},
                  {"18446744073709551615", ~0ull},
                  {"-9223372036854775808", 0x8000000000000000ull}},
                 "literal_unsigned_decimal");
}
TEST(PtxIntegerLiteral, Full64BitRangeAndNegation) {
  check_literals({{"9223372036854775807", 0x7fffffffffffffffull},
                  {"9223372036854775808", 0x8000000000000000ull},
                  {"18446744073709551615U", ~0ull},
                  {"18446744073709551615", ~0ull},
                  {"0xffffffffffffffffU", ~0ull},
                  {"01777777777777777777777U", ~0ull},
                  {"0b1111111111111111111111111111111111111111111111111111111111111111U", ~0ull},
                  {"-9223372036854775808", 0x8000000000000000ull},
                  {"-18446744073709551615U", 1}, {"-42", 0ull - 42}},
                 "literal_boundaries");
}
TEST(PtxIntegerLiteral, RejectOverflowInsteadOfSaturating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(check_literals({{"18446744073709551616U", 0}}, "literal_overflow_decimal"), "integer literal out of range");
  EXPECT_DEATH(check_literals({{"0x10000000000000000U", 0}}, "literal_overflow_hex"), "integer literal out of range");
}
TEST(PtxIntegerLiteral, RejectInvalidOctalAndNegativeOverflow) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(check_literals({{"08", 0}}, "literal_invalid_octal"),
               "integer literal out of range");
  EXPECT_DEATH(check_literals({{"-18446744073709551616", 0}},
                              "literal_negative_overflow"),
               "integer literal out of range");
}
}  // namespace
