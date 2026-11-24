#include <gtest/gtest.h>
#include "gpgpu-sim/flash/warp_reduce.h"
using namespace flash_gpgpu_sim;
TEST(WarpReduce,MaskValidationAndDisjointGroups) {
  uint32_t values[32]{},masks[32]{},out[32]{};
  for(unsigned i=0;i<32;++i) {values[i]=i;masks[i]=0xffffu << (i&16);}
  ASSERT_TRUE(reduce32_groups({reduce32_kind::add},0xffffffff,values,masks,out));
  EXPECT_EQ(out[0],120u);EXPECT_EQ(out[31],376u);
  masks[3]=~0u; EXPECT_FALSE(reduce32_groups({reduce32_kind::add},~0u,values,masks,out));
  masks[3]=0xffff;EXPECT_FALSE(reduce32_groups({reduce32_kind::add},0xfffffffe,values,masks,out));
  masks[0]=0xfffe;EXPECT_FALSE(reduce32_groups({reduce32_kind::add},~0u,values,masks,out));
}
TEST(WarpReduce,FloatSpecialValues) {
  for(auto kind:{reduce32_kind::min_f,reduce32_kind::max_f}) {
    EXPECT_EQ(reduce32_combine({kind},0x7f800001,0xff800000),0xff800000u);
    EXPECT_EQ(reduce32_combine({kind},0xff800001,0x7fc00000),0x7fffffffu);
    EXPECT_EQ(reduce32_combine({kind,false,true},0x7f800001,0x3f800000),0x7fffffffu);
    EXPECT_EQ(reduce32_combine({kind,true},0xbf800000,0x3f800000),0x3f800000u);
  }
  EXPECT_EQ(reduce32_combine({reduce32_kind::min_f},0,0x80000000),0x80000000u);
  EXPECT_EQ(reduce32_combine({reduce32_kind::max_f},0,0x80000000),0u);
  EXPECT_EQ(reduce32_combine({reduce32_kind::add},0xffffffff,2),1u);
}
