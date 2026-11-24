#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace {
// Every reduction aliases its source; an implementation executed per lane
// would reread results written by earlier lanes and corrupt most operations.
__global__ void reduce_all(const unsigned *input,unsigned *out,unsigned members, unsigned mode) {
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x & 31;
  unsigned mask=mode==2 ? (0xffu << (lane & ~7u)) : members;
  if (!(mask & (1u<<lane))) { for(unsigned j=0;j<17;++j) out[tid*17+j]=0xdeadbeef; return; }
  unsigned r;
#define REDUCE(OP,TYPE,INDEX) r=input[tid]; asm volatile("redux.sync." OP "." TYPE " %0,%0,%1;" : "+r"(r) : "r"(mask)); out[tid*17+INDEX]=r;
  REDUCE("add","u32",0) REDUCE("add","s32",1)
  REDUCE("min","u32",2) REDUCE("max","u32",3)
  REDUCE("min","s32",4) REDUCE("max","s32",5)
  REDUCE("and","b32",6) REDUCE("or","b32",7) REDUCE("xor","b32",8)
  REDUCE("min","f32",9) REDUCE("max","f32",10)
  REDUCE("min.abs","f32",11) REDUCE("max.abs","f32",12)
  REDUCE("min.NaN","f32",13) REDUCE("max.NaN","f32",14)
  REDUCE("min.abs.NaN","f32",15) REDUCE("max.abs.NaN","f32",16)
#undef REDUCE
}
unsigned reference(const std::vector<unsigned> &in,unsigned base,unsigned mask,unsigned op) {
  unsigned result=0; bool first=true;
  for(unsigned i=0;i<32;++i) {
    if(!(mask & (1u<<i))) continue;
    unsigned b=in[base+i];
    if(op<9) {
      if(first) result=b;
      else switch(op) {
        case 0:case 1:result+=b;break;
        case 2:result=std::min(result,b);break;
        case 3:result=std::max(result,b);break;
        case 4:result=unsigned(std::min(int(result),int(b)));break;
        case 5:result=unsigned(std::max(int(result),int(b)));break;
        case 6:result&=b;break;case 7:result|=b;break;case 8:result^=b;break;
      }
      first=false;continue;
    }
    const bool absolute=op==11 || op==12 || op==15 || op==16;
    if(absolute) b&=0x7fffffff;
    if(first) { result=b; first=false; continue; }
    float a,x;std::memcpy(&a,&result,4);std::memcpy(&x,&b,4);
    if(std::isnan(a) || std::isnan(x)) {
      if(op>=13 || (std::isnan(a)&&std::isnan(x))) result=0x7fffffff;
      else if(std::isnan(a)) result=b;
    } else {
      const bool minimum=op==9 || op==11 || op==13 || op==15;
      float c=minimum ? std::fmin(a,x) : std::fmax(a,x);
      if(a==0 && x==0) c=(minimum ? (std::signbit(a)||std::signbit(x)) : (std::signbit(a)&&std::signbit(x))) ? -0.f:0.f;
      std::memcpy(&result,&c,4);
    }
  }
  float f;std::memcpy(&f,&result,4);
  if(op>=9 && std::isnan(f)) return 0x7fffffff;
  return result;
}
void check(unsigned members,unsigned mode,unsigned data) {
  std::vector<unsigned> in(256),out(256*17);
  const unsigned special[]={0,0x80000000,0x7f800000,0xff800000,0x7fffffff,0xffc00001,1,0x80000001,0x00800000,0x80800000,0x3f800000,0xbf800000};
  for(unsigned i=0;i<in.size();++i) {
    in[i]=data==0 ? (0xf1234567u+i*0x1009131u) : data==1 ? special[i%12] : data==2 ? (i&1 ? 0xffc00001:0x7fffffff) : (i&1 ? 0x80000000:0);
  }
  unsigned *a,*b;
  ASSERT_EQ(cudaMalloc(&a,in.size()*4),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&b,out.size()*4),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(a,in.data(),in.size()*4,cudaMemcpyHostToDevice),cudaSuccess);
  reduce_all<<<4,64>>>(a,b,members,mode);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(),b,out.size()*4,cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<in.size();++i) for(unsigned op=0;op<17;++op) {
    unsigned mask=mode==2 ? 0xffu << (i&24):members;
    unsigned expected=!(mask & (1u<<(i&31))) ? 0xdeadbeef : reference(in,i&~31u,mask,op);
    ASSERT_EQ(out[i*17+op],expected) << "thread=" << i << " op=" << op;
  }
  EXPECT_EQ(cudaFree(a),cudaSuccess);EXPECT_EQ(cudaFree(b),cudaSuccess);
}
TEST(WarpReduceLaunch,FullWarpSignedOverflowAndAlias) {check(0xffffffff,0,0);}
TEST(WarpReduceLaunch,PartialWarpLeaderNotZero) {check(0xaaaaaaaau,1,1);}
TEST(WarpReduceLaunch,DisjointGroups) {check(0xffffffff,2,0);}
TEST(WarpReduceLaunch,FloatNaNsInfinitiesAndSubnormals) {check(0xffffffff,0,1);}
TEST(WarpReduceLaunch,AllNaNs) {check(0xffffffff,0,2);}
TEST(WarpReduceLaunch,SignedZeros) {check(0xffffffff,0,3);}
} // namespace

namespace {
__global__ void predicated_reduce(unsigned *out) {
  const unsigned tid=threadIdx.x;
  unsigned x=tid+1;
  asm volatile("{ .reg .pred p; .reg .b32 odd; and.b32 odd,%1,1; setp.ne.u32 p,odd,0; @p redux.sync.add.u32 %0,%0,0xaaaaaaaa; }" : "+r"(x) : "r"(tid));
  out[tid]=x;
}
TEST(WarpReduceLaunch,PredicatedSourceDestinationAlias) {
  unsigned *out;ASSERT_EQ(cudaMalloc(&out,64*4),cudaSuccess);
  predicated_reduce<<<1,64>>>(out);ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  unsigned actual[64];ASSERT_EQ(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<64;++i) EXPECT_EQ(actual[i],i&1 ? 272+(i/32)*512:i+1) << i;
  EXPECT_EQ(cudaFree(out),cudaSuccess);
}
}

// The same standalone regression also qualifies the generated native wrapper.
#include "../funnel_shift/launch_test.cu"
