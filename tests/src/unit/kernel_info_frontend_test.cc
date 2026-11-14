#include <gtest/gtest.h>
#include <cstring>
#include <new>

#include "libcuda/gpgpu_context.h"
#include "src/kernel_info.h"

TEST(KernelInfoFrontend, NativeLaunchStartsWithoutClusterState) {
  // Reused heap storage must not turn an ordinary native launch into a
  // cluster launch that the per-SM dispatcher silently skips.
  alignas(kernel_info_t) unsigned char storage[sizeof(kernel_info_t)];
  for (int fill : {0, 0xff}) {
    std::memset(storage, fill, sizeof(storage));
    auto *kernel = new (storage) kernel_info_t(
        dim3(2, 1, 1), dim3(32, 1, 1), "native_launch_defaults",
        GPGPU_Context(), 0, 1);
    EXPECT_FALSE(kernel->is_cluster_launch());
    EXPECT_EQ(kernel->get_ctas_per_cluster(), 1u);
    EXPECT_EQ(kernel->get_cluster_dim().x, 0u);
    EXPECT_FALSE(kernel->has_open_tb_cluster());
    EXPECT_EQ(kernel->open_tb_phys_cluster(), (unsigned)-1);
    EXPECT_EQ(kernel->open_tb_cluster_group(), (unsigned)-1);
    kernel->increment_cta_id();
    EXPECT_EQ(kernel->get_next_cta_id().x, 1u);
    kernel->~kernel_info_t();
  }
}
