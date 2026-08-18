// Unit tests for the shipped remote-DSM store issue path.
// Includes src/gpgpu-sim/flash/cluster_dsm_store.h — no local reimplementation.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include "../../../src/gpgpu-sim/flash/cluster_dsm_store.h"

using flash_gpgpu_sim::dsm_store_immediate_enabled;
using flash_gpgpu_sim::dsm_store_immediate_env_name;
using flash_gpgpu_sim::issue_remote_dsm_store;

namespace {

void unset_immediate_env() {
  unsetenv(dsm_store_immediate_env_name());
}

std::string find_repo_file(const char *rel) {
  // Walk cwd and a few parents: unit tests run from test/run/<config>/.
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

TEST(DsmStoreImmediate, EnvUnsetUsesConfigured) {
  unset_immediate_env();
  EXPECT_TRUE(dsm_store_immediate_enabled(true));
  EXPECT_FALSE(dsm_store_immediate_enabled(false));
}

TEST(DsmStoreImmediate, EnvZeroOverridesConfiguredOn) {
  ASSERT_EQ(setenv(dsm_store_immediate_env_name(), "0", 1), 0);
  EXPECT_FALSE(dsm_store_immediate_enabled(true));
  EXPECT_FALSE(dsm_store_immediate_enabled(false));
  unset_immediate_env();
}

TEST(DsmStoreImmediate, EnvOneOverridesConfiguredOff) {
  ASSERT_EQ(setenv(dsm_store_immediate_env_name(), "1", 1), 0);
  EXPECT_TRUE(dsm_store_immediate_enabled(false));
  EXPECT_TRUE(dsm_store_immediate_enabled(true));
  unset_immediate_env();
}

TEST(DsmStoreImmediate, IssuePathInjectsWithoutPeerWriteWhenDelayed) {
  unset_immediate_env();
  int injects = 0;
  int writes = 0;
  const bool handled = issue_remote_dsm_store(
      /*is_remote=*/true, /*noc_enabled=*/true, /*immediate=*/false,
      [&]() {
        injects++;
        return true;
      },
      [&]() { writes++; });
  EXPECT_TRUE(handled);
  EXPECT_EQ(injects, 1);
  EXPECT_EQ(writes, 0) << "delayed store must not write peer smem at issue";
}

TEST(DsmStoreImmediate, IssuePathWritesPeerWhenImmediate) {
  unset_immediate_env();
  int injects = 0;
  int writes = 0;
  const bool handled = issue_remote_dsm_store(
      /*is_remote=*/true, /*noc_enabled=*/true, /*immediate=*/true,
      [&]() {
        injects++;
        return true;
      },
      [&]() { writes++; });
  EXPECT_TRUE(handled);
  EXPECT_EQ(injects, 1);
  EXPECT_EQ(writes, 1);
}

TEST(DsmStoreImmediate, NoCOffLeavesWriteToCaller) {
  int injects = 0;
  int writes = 0;
  const bool handled = issue_remote_dsm_store(
      /*is_remote=*/true, /*noc_enabled=*/false, /*immediate=*/false,
      [&]() {
        injects++;
        return true;
      },
      [&]() { writes++; });
  EXPECT_FALSE(handled);
  EXPECT_EQ(injects, 0);
  EXPECT_EQ(writes, 0);
}

TEST(DsmStoreImmediate, LocalStoreDoesNotTouchNoC) {
  int injects = 0;
  int writes = 0;
  const bool handled = issue_remote_dsm_store(
      /*is_remote=*/false, /*noc_enabled=*/true, /*immediate=*/true,
      [&]() {
        injects++;
        return true;
      },
      [&]() { writes++; });
  EXPECT_FALSE(handled);
  EXPECT_EQ(injects, 0);
  EXPECT_EQ(writes, 0);
}

TEST(DsmStoreImmediate, FailedInjectDoesNotWritePeer) {
  int writes = 0;
  const bool handled = issue_remote_dsm_store(
      true, true, true, [&]() { return false; }, [&]() { writes++; });
  EXPECT_FALSE(handled);
  EXPECT_EQ(writes, 0);
}

TEST(DsmStoreImmediate, StImplUsesShippedIssuePath) {
  const std::string path = find_repo_file("src/cuda-sim/instructions.cc");
  ASSERT_FALSE(path.empty()) << "cannot find src/cuda-sim/instructions.cc";
  std::ifstream in(path);
  ASSERT_TRUE(in);
  std::string src((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  EXPECT_NE(src.find("dsm_store_immediate_enabled"), std::string::npos)
      << "st_impl must read the store-immediate knob/env via the shipped helper";
  EXPECT_NE(src.find("issue_remote_dsm_store"), std::string::npos)
      << "st_impl must call issue_remote_dsm_store (inject + gated peer write)";
  EXPECT_NE(src.find("gpgpu_dsm_store_immediate"), std::string::npos);
}

TEST(DsmStoreImmediate, KnobRegisteredWithDefaultOn) {
  const std::string path = find_repo_file("src/gpgpu-sim/gpu-sim.cc");
  ASSERT_FALSE(path.empty()) << "cannot find src/gpgpu-sim/gpu-sim.cc";
  std::ifstream in(path);
  ASSERT_TRUE(in);
  std::string src((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  EXPECT_NE(src.find("\"-gpgpu_dsm_store_immediate\""), std::string::npos);
  // option_parser default string is the last argument; must be "1".
  const auto pos = src.find("\"-gpgpu_dsm_store_immediate\"");
  ASSERT_NE(pos, std::string::npos);
  const auto tail = src.substr(pos, 600);
  EXPECT_NE(tail.find("\"1\""), std::string::npos)
      << "parsed default for -gpgpu_dsm_store_immediate must stay 1";
}
