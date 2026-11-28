#include <gtest/gtest.h>
#include "gpgpu-sim/flash/sass/frontend.h"
#include "gpgpu-sim/flash/sass/runtime/runtime_adapter.h"
#include "libcuda/gpgpu_context.h"
#include "src/cuda-sim/memory.h"
#include "src/kernel_info.h"

namespace {
namespace sass = flash_gpgpu_sim::sass;
TEST(SassParameterAlias, BlackwellUsesTheExistingLaunchParameterBacking) {
  for (auto arch : {sass::architecture::kSm100, sass::architecture::kSm103}) {
    kernel_info_t kernel(dim3(1), dim3(32), "parameter_alias", GPGPU_Context(),
                         0, 1);
    memory_space_impl<8192> global("alias_global", 32);
    auto memory = sass::make_runtime_functional_memory(kernel, &global, arch,
                                                       0x380, 512, {});
    uint64_t address = 0;
    ASSERT_TRUE(
        memory->read(sass::memory_space::kConstant, 0x348, &address, 8));
    ASSERT_NE(address, 0u);
    uint32_t input = 0x12345678, output = 0;
    kernel.get_param_memory()->write(384, 4, &input, nullptr, nullptr);
    ASSERT_TRUE(
        memory->read(sass::memory_space::kGlobal, address + 384, &output, 4));
    EXPECT_EQ(output, input);
    input =
        0xabcdef01;  // Both paths must observe updates; no copied alias buffer.
    kernel.get_param_memory()->write(384, 4, &input, nullptr, nullptr);
    ASSERT_TRUE(
        memory->read(sass::memory_space::kGlobal, address + 384, &output, 4));
    EXPECT_EQ(output, input);
    ASSERT_TRUE(
        memory->read(sass::memory_space::kConstant, 0x380 + 384, &output, 4));
    EXPECT_EQ(output, input);
    EXPECT_FALSE(
        memory->write(sass::memory_space::kGlobal, address + 384, &output, 4));
    uint8_t last = 0x92, got = 0;
    kernel.get_param_memory()->write(511, 1, &last, nullptr, nullptr);
    ASSERT_TRUE(
        memory->read(sass::memory_space::kGlobal, address + 511, &got, 1));
    EXPECT_EQ(got, last);
    uint32_t outside = 0x87654321;
    global.write(address + 512, 4, &outside, nullptr, nullptr);
    ASSERT_TRUE(
        memory->read(sass::memory_space::kGlobal, address + 512, &output, 4));
    EXPECT_EQ(output, outside);
  }
}
TEST(SassParameterAlias, Sm120ReservedHeaderRemainsZero) {
  kernel_info_t kernel(dim3(1), dim3(32), "parameter_alias", GPGPU_Context(), 0,
                       1);
  memory_space_impl<8192> global("alias_global", 32);
  auto memory = sass::make_runtime_functional_memory(
      kernel, &global, sass::architecture::kSm120, 0x380, 512, {});
  uint64_t address = 1;
  ASSERT_TRUE(memory->read(sass::memory_space::kConstant, 0x348, &address, 8));
  EXPECT_EQ(address, 0u);
}
}  // namespace
