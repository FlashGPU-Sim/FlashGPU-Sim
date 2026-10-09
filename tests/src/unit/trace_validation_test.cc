#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <unistd.h>

#include "../trace/trace_test_utils.h"

TEST(TraceValidationTest, RejectsNonfiniteOutputEvenAgainstZero) {
  EXPECT_TRUE(fp16_matches_reference(0, 0));
  EXPECT_TRUE(fp16_matches_reference(0x3c00, 0x3c00));
  EXPECT_FALSE(fp16_matches_reference(0x4000, 0x3c00));
  for (uint16_t nonfinite : {uint16_t(0x7e00), uint16_t(0x7c00),
                             uint16_t(0xfc00)}) {
    EXPECT_FALSE(fp16_matches_reference(nonfinite, 0));
    EXPECT_FALSE(fp16_matches_reference(0, nonfinite));
  }
}

TEST(TraceValidationTest, RejectsTruncatedReferenceFile) {
  char path[] = "/tmp/flashgpu-trace-reference-XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  const uint16_t expected = 0x3c00;
  const ssize_t written = write(fd, &expected, sizeof(expected));
  close(fd);
  std::array<uint16_t, 2> data{};
  const int short_result = read_exact_file(path, data.data(), sizeof(data));
  const int exact_result = read_exact_file(path, data.data(), sizeof(expected));
  unlink(path);
  ASSERT_EQ(written, sizeof(expected));
  EXPECT_NE(short_result, 0);
  EXPECT_EQ(exact_result, 0);
  EXPECT_EQ(data[0], expected);
}
