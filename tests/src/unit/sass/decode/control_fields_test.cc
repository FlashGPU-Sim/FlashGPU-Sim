#include <gtest/gtest.h>

#include "gpgpu-sim/flash/sass/decode/sm120_control.h"
#include "gpgpu-sim/flash/sass/decode/sm90_control.h"

namespace {

TEST(SassControlFieldsTest, ExhaustsSharedSchedulingFields) {
  for (uint32_t packed = 0; packed < (1u << 17); ++packed) {
    const uint64_t raw = uint64_t{packed} << 41;
    for (const auto control :
         {flash_gpgpu_sim::sass::decode_sm90_control(raw),
          flash_gpgpu_sim::sass::decode_sm120_control(raw)}) {
      ASSERT_EQ(control.stall, packed & 0xf);
      ASSERT_EQ(control.yield_flag, ((packed >> 4) & 1) == 0);
      ASSERT_EQ(control.write_barrier, (packed >> 5) & 7);
      ASSERT_EQ(control.read_barrier, (packed >> 8) & 7);
      ASSERT_EQ(control.wait_mask, (packed >> 11) & 0x3f);
      ASSERT_EQ(control.reuse_mask, 0);
    }
  }
}

TEST(SassControlFieldsTest, KeepsArchitectureSpecificReuseWidths) {
  for (uint64_t reuse = 0; reuse < 16; ++reuse) {
    const uint64_t raw = (reuse << 58) | (uint64_t{1} << 63);
    EXPECT_EQ(flash_gpgpu_sim::sass::decode_sm90_control(raw).reuse_mask,
              reuse);
    EXPECT_EQ(flash_gpgpu_sim::sass::decode_sm120_control(raw).reuse_mask,
              reuse & 7);
  }
}

}  // namespace
