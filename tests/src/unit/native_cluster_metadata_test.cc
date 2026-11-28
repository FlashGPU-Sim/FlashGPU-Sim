#include <gtest/gtest.h>
#include <unistd.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include "gpgpu-sim/flash/sass/decode/sassir_decoder.h"
#include "gpgpu-sim/flash/sass/runtime/runtime_adapter.h"
namespace {
class NativeClusterMetadata : public ::testing::Test {
 protected:
  void SetUp() override {
    char name[] = "/tmp/native-cluster-XXXXXX";
    const int fd = mkstemp(name);
    ASSERT_GE(fd, 0);
    close(fd);
    path = name;
  }
  void TearDown() override {
    if (!path.empty()) std::remove(path.c_str());
  }
  void write(const std::string &metadata) {
    std::ofstream out(path);
    out << "SASSIR\t3\nDECODER\ttest\t1\tfixture\nKERNEL\tk\tsm100\t16\n"
           "RESOURCES\t8\t0\t0\t0\n"
        << metadata
        << "ENDKERNEL\n"
           "KERNEL\tother\tsm100\t16\nRESOURCES\t16\t0\t0\t0\nENDKERNEL\n";
  }
  auto decode() {
    return flash_gpgpu_sim::sass::sassir_decoder{}.decode_kernel(path, "k");
  }
  std::string path;
};
TEST_F(NativeClusterMetadata,
       DimensionsFlowIntoLaunchResourcesOnlyForTheirKernel) {
  write("CLUSTERDIMS\t2\t1\t1\n");
  EXPECT_EQ(decode().required_cluster, (std::array<uint32_t, 3>{2, 1, 1}));
  EXPECT_EQ(flash_gpgpu_sim::sass::load_runtime_kernel_resources(path, "k")
                .required_cluster,
            (std::array<uint32_t, 3>{2, 1, 1}));
  EXPECT_EQ(flash_gpgpu_sim::sass::load_runtime_kernel_resources(path, "other")
                .required_cluster,
            (std::array<uint32_t, 3>{0, 0, 0}));
  write("");
  EXPECT_EQ(decode().required_cluster, (std::array<uint32_t, 3>{0, 0, 0}));
}
TEST_F(NativeClusterMetadata, RejectsZeroDuplicateAndOverflowDimensions) {
  for (const char *metadata :
       {"CLUSTERDIMS\t0\t1\t1\n", "CLUSTERDIMS\t-1\t1\t1\n",
        "CLUSTERDIMS\t4294967296\t1\t1\n", "CLUSTERDIMS\t65536\t65536\t1\n",
        "CLUSTERDIMS\t2\t1\t1\nCLUSTERDIMS\t2\t1\t1\n"}) {
    write(metadata);
    EXPECT_DEATH(decode(), "cluster|CLUSTERDIMS");
  }
}
}  // namespace
