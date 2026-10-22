// Remote DSM stores go through the intra-GPC fabric only.

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace {

std::string find_repo_file(const char *rel) {
  std::string prefix;
  for (int up = 0; up < 6; up++) {
    const std::string path = prefix + rel;
    std::ifstream in(path);
    if (in)
      return path;
    prefix += "../";
  }
  return {};
}

}  // namespace

TEST(DsmStorePath, StImplUsesFabricAndAbortsWhenDisabled) {
  const std::string path = find_repo_file("src/cuda-sim/instructions.cc");
  ASSERT_FALSE(path.empty()) << "cannot find src/cuda-sim/instructions.cc";
  std::ifstream in(path);
  ASSERT_TRUE(in);
  std::string src((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  EXPECT_NE(src.find("dsm_note_fabric"), std::string::npos)
      << "st_impl must inject remote stores through the DSM fabric";
  EXPECT_NE(src.find("abort_dsm_disabled"), std::string::npos)
      << "remote DSM without -gpgpu_dsm_enable 1 must abort";
  EXPECT_EQ(src.find("issue_remote_dsm_store"), std::string::npos);
}

TEST(DsmStorePath, FabricMasterSwitchRegistered) {
  const std::string path = find_repo_file("src/gpgpu-sim/gpu-sim.cc");
  ASSERT_FALSE(path.empty()) << "cannot find src/gpgpu-sim/gpu-sim.cc";
  std::ifstream in(path);
  ASSERT_TRUE(in);
  std::string src((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  EXPECT_NE(src.find("\"-gpgpu_dsm_enable\""), std::string::npos);
}
