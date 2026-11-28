#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {
__global__ void wrapping_funnel(unsigned *out) {
  const unsigned tid=threadIdx.x;
  unsigned result;
  const unsigned a=0x12345678u+tid, b=0xfedcba98u-tid;
#define SHIFT(OP,J) asm volatile(OP " %0,%1,%2,%3;" : "=r"(result) : "r"(a),"r"(b),"r"(tid)); out[tid*4+J]=result;
  SHIFT("shf.r.wrap.b32",0) SHIFT("shf.l.wrap.b32",1)
  SHIFT("shf.r.clamp.b32",2) SHIFT("shf.l.clamp.b32",3)
#undef SHIFT
}
TEST(WarpShiftLaunch,AllWrappedFunnelCounts) {
  unsigned *out;ASSERT_EQ(cudaMalloc(&out,128*4*4),cudaSuccess);
  wrapping_funnel<<<1,128>>>(out);ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  unsigned actual[128*4];ASSERT_EQ(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<128;++i) { unsigned n=i&31,a=0x12345678u+i,b=0xfedcba98u-i;
    EXPECT_EQ(actual[i*4],n ? (a>>n)|(b<<(32-n)):a) << i;
    EXPECT_EQ(actual[i*4+1],n ? (b<<n)|(a>>(32-n)):b) << i;
    EXPECT_EQ(actual[i*4+2],i>=32 ? b : i ? (a>>i)|(b<<(32-i)):a) << i;
    EXPECT_EQ(actual[i*4+3],i>=32 ? a : i ? (b<<i)|(a>>(32-i)):b) << i;
  }
  EXPECT_EQ(cudaFree(out),cudaSuccess);
}
}
