#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include "gpgpu-sim/flash/shared_bulk.h"
using namespace flash_gpgpu_sim;
TEST(SharedBulk, ExtentBoundaries) {
  EXPECT_TRUE(valid_shared_bulk_extent(8, 16777216));
  EXPECT_TRUE(valid_shared_bulk_extent(0, 0));
  EXPECT_FALSE(valid_shared_bulk_extent(4, 8));
  EXPECT_FALSE(valid_shared_bulk_extent(8, 4));
  EXPECT_FALSE(valid_shared_bulk_extent(0, 16777224));
  EXPECT_FALSE(valid_shared_bulk_extent(UINT64_MAX-7, 8));
}
TEST(SharedBulk, GuardsAndEmptyExtent) {
  std::array<unsigned char,32> bytes; bytes.fill(123);
  unsigned calls=0;
  auto writer=[&](uint64_t at,const void *data,size_t n) {
    ++calls; std::memcpy(bytes.data()+at,data,n); return true;
  };
  EXPECT_TRUE(zero_shared_bulk(8,0,writer));
  EXPECT_EQ(calls,0u);
  EXPECT_FALSE(zero_shared_bulk(3,8,writer));
  EXPECT_EQ(calls,0u);
  EXPECT_TRUE(zero_shared_bulk(8,16,writer));
  EXPECT_EQ(calls,1u);
  for(unsigned i=0;i<32;++i) EXPECT_EQ(bytes[i],i>=8&&i<24?0:123);
}
TEST(SharedBulk, PropagatesWriteFailure) {
  EXPECT_FALSE(zero_shared_bulk(0,8,[](uint64_t,const void*,size_t){return false;}));
}
