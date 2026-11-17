#include <gtest/gtest.h>

#include "../../../src/gpgpu-sim/flash/instruction_cache/address_mapping.h"

namespace {

using flash_gpgpu_sim::instruction_address_mapper;

TEST(InstructionAddressMappingTest, IdentityScalePreservesFetch) {
  const instruction_address_mapper mapper(/*scale=*/1, /*line_size=*/128);
  const auto fetch = mapper.map_fetch(/*functional_pc=*/0x40,
                                      /*max_functional_bytes=*/16,
                                      /*cache_base=*/0x100000);
  EXPECT_EQ(fetch.cache_address, 0x100040u);
  EXPECT_EQ(fetch.cache_bytes, 16u);
  EXPECT_EQ(fetch.functional_bytes, 16u);
  EXPECT_EQ(mapper.functional_pc(fetch.cache_address, 0x100000), 0x40u);
}

TEST(InstructionAddressMappingTest, ScaleTwoModelsSixteenByteSassSlots) {
  const instruction_address_mapper mapper(/*scale=*/2, /*line_size=*/128);
  const auto fetch = mapper.map_fetch(/*functional_pc=*/0x20,
                                      /*max_functional_bytes=*/16,
                                      /*cache_base=*/0x100000);
  EXPECT_EQ(fetch.cache_address, 0x100040u);
  EXPECT_EQ(fetch.cache_bytes, 32u);
  EXPECT_EQ(fetch.functional_bytes, 16u);
  EXPECT_EQ(mapper.functional_pc(fetch.cache_address, 0x100000), 0x20u);
}

TEST(InstructionAddressMappingTest, LineCrossingTruncatesBothByteDomains) {
  const instruction_address_mapper mapper(/*scale=*/2, /*line_size=*/128);
  const auto fetch = mapper.map_fetch(/*functional_pc=*/0x38,
                                      /*max_functional_bytes=*/16,
                                      /*cache_base=*/0x100000);
  EXPECT_EQ(fetch.cache_address, 0x100070u);
  EXPECT_EQ(fetch.cache_bytes, 16u);
  EXPECT_EQ(fetch.functional_bytes, 8u);
  EXPECT_EQ(mapper.functional_bytes(fetch.cache_bytes), 8u);
}

}  // namespace
