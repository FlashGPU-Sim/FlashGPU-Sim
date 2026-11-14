#include <gtest/gtest.h>

#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

#include "gpgpu-sim/flash/panic.h"
#include "gpgpu-sim/flash/sass/decode/sassir_decoder.h"

namespace {
class SassirNumbersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char name[] = "/tmp/sassir-numbers-XXXXXX";
    const int fd = mkstemp(name);
    ASSERT_GE(fd, 0);
    close(fd);
    path = name;
  }
  void TearDown() override {
    if (!path.empty()) std::remove(path.c_str());
  }
  flash_gpgpu_sim::sass::kernel decode(const std::string &raw_lo) {
    {
      std::ofstream out(path);
      out << "SASSIR\t1\nKERNEL\tk\tsm90\t16\nINST\t0\t" << raw_lo
          << "\t0\t1\tNOP\t\t\t0\t1\t0\t0\t0\t0\n"
             "FIELD\tx\t0\t1\t0\t-1\tlegacy\nENDINST\nENDKERNEL\n";
    }
    return flash_gpgpu_sim::sass::sassir_decoder{}.decode_kernel(path, "k");
  }
  std::string path;
};

TEST_F(SassirNumbersTest, RejectsNegativeUnsignedBitPatterns) {
  for (const char *text : {"-1", "-0x10", " -1", "-18446744073709551615"})
    EXPECT_DEATH(decode(text), ":3: invalid raw lo:") << text;
}

TEST_F(SassirNumbersTest, AcceptsFullUnsignedRangeAndSignedTokenSentinel) {
  const auto image = decode("0xffffffffffffffff");
  ASSERT_EQ(image.instructions.size(), 1u);
  EXPECT_EQ(image.instructions[0].raw.lo, std::numeric_limits<uint64_t>::max());
  ASSERT_EQ(image.instructions[0].fields.size(), 1u);
  EXPECT_EQ(image.instructions[0].fields[0].token_index, -1);
}

TEST_F(SassirNumbersTest, RejectsOverflowAndTrailingJunk) {
  for (const char *text :
       {"18446744073709551616", "0x10000000000000000", "12junk", ""})
    EXPECT_DEATH(decode(text), ":3: invalid raw lo:") << text;
}

TEST(FlashPanicDeathTest, PrintsContextAndTerminatesWithSigabrt) {
  const std::string file = "test.sassir";
  const uint64_t line = 7;
  const std::string opcode = "HMMA";
  const uint64_t pc = 0x120;
  const flash_gpgpu_sim::panic_context outer(file, line);
  const flash_gpgpu_sim::panic_context inner(opcode, pc, true);
  EXPECT_EXIT(
      flash_gpgpu_sim::panic("invalid operand"),
      ::testing::KilledBySignal(SIGABRT),
      "FlashGPU-Sim panic: test.sassir:7: HMMA at pc 0x120: invalid operand");
}

TEST(FlashPanicDeathTest, RestoresContextAfterNormalReturn) {
  EXPECT_EQ(flash_gpgpu_sim::panic_context::current(), nullptr);
  const std::string file = "test.sassir";
  const uint64_t line = 2;
  {
    const flash_gpgpu_sim::panic_context outer(file, line);
    {
      const flash_gpgpu_sim::panic_context inner(file, line);
      EXPECT_EQ(flash_gpgpu_sim::panic_context::current(), &inner);
    }
    EXPECT_EQ(flash_gpgpu_sim::panic_context::current(), &outer);
  }
  EXPECT_EQ(flash_gpgpu_sim::panic_context::current(), nullptr);
}
}  // namespace

TEST_F(SassirNumbersTest, InitializedLocalGlobalCarriesExactBytes) {
  {
    std::ofstream out(path);
    out << "SASSIR\t4\nDECODER\ttest\t1\tfixture\nKERNEL\tk\tsm120\t16\nRESOURC"
           "ES\t8\t0\t0\t0\n"
           "CONSTBANK\t4\t0000000000000000\n"
           "CONSTGLOBAL_INIT\t4\t0\tlocal_string\t4\t1\t41424300\n"
           "ENDKERNEL\n";
  }
  const auto image =
      flash_gpgpu_sim::sass::sassir_decoder{}.decode_kernel(path, "k");
  ASSERT_EQ(image.constant_globals.size(), 1u);
  EXPECT_EQ(image.constant_globals[0].addend, 1u);
  EXPECT_EQ(image.constant_globals[0].initial_data,
            (std::vector<uint8_t>{65, 66, 67, 0}));
}

TEST_F(SassirNumbersTest, RejectsTruncatedLocalGlobalInitializer) {
  {
    std::ofstream out(path);
    out << "SASSIR\t4\nDECODER\ttest\t1\tfixture\nKERNEL\tk\tsm120\t16\nRESOURC"
           "ES\t8\t0\t0\t0\n"
           "CONSTGLOBAL_INIT\t4\t0\tlocal_string\t4\t0\t4142\n"
           "ENDKERNEL\n";
  }
  EXPECT_DEATH(flash_gpgpu_sim::sass::sassir_decoder{}.decode_kernel(path, "k"),
               "initializer size mismatch");
}
