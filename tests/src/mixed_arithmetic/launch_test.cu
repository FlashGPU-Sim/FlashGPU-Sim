#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
struct Input { unsigned packed; float c; };
template<unsigned Op, bool Bfloat>
__global__ void mixed(const Input *in, float *out) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned short a=in[i].packed, b=in[i].packed>>16;
  const float c=in[i].c;
  float d;
  if constexpr (Op == 0 && Bfloat == false) {
    asm volatile("add.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+0]=d;
    asm volatile("add.rn.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+1]=d;
    asm volatile("add.rz.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+2]=d;
    asm volatile("add.rm.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+3]=d;
    asm volatile("add.rp.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+4]=d;
    asm volatile("add.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+5]=d;
    asm volatile("add.rn.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+6]=d;
    asm volatile("add.rz.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+7]=d;
    asm volatile("add.rm.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+8]=d;
    asm volatile("add.rp.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+9]=d;
  }
  if constexpr (Op == 0 && Bfloat == true) {
    asm volatile("add.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+0]=d;
    asm volatile("add.rn.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+1]=d;
    asm volatile("add.rz.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+2]=d;
    asm volatile("add.rm.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+3]=d;
    asm volatile("add.rp.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+4]=d;
    asm volatile("add.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+5]=d;
    asm volatile("add.rn.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+6]=d;
    asm volatile("add.rz.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+7]=d;
    asm volatile("add.rm.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+8]=d;
    asm volatile("add.rp.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+9]=d;
  }
  if constexpr (Op == 1 && Bfloat == false) {
    asm volatile("sub.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+0]=d;
    asm volatile("sub.rn.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+1]=d;
    asm volatile("sub.rz.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+2]=d;
    asm volatile("sub.rm.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+3]=d;
    asm volatile("sub.rp.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+4]=d;
    asm volatile("sub.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+5]=d;
    asm volatile("sub.rn.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+6]=d;
    asm volatile("sub.rz.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+7]=d;
    asm volatile("sub.rm.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+8]=d;
    asm volatile("sub.rp.sat.f32.f16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+9]=d;
  }
  if constexpr (Op == 1 && Bfloat == true) {
    asm volatile("sub.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+0]=d;
    asm volatile("sub.rn.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+1]=d;
    asm volatile("sub.rz.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+2]=d;
    asm volatile("sub.rm.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+3]=d;
    asm volatile("sub.rp.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+4]=d;
    asm volatile("sub.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+5]=d;
    asm volatile("sub.rn.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+6]=d;
    asm volatile("sub.rz.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+7]=d;
    asm volatile("sub.rm.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+8]=d;
    asm volatile("sub.rp.sat.f32.bf16 %0,%1,%2;" : "=f"(d) : "h"(a), "f"(c));
    out[i*10+9]=d;
  }
  if constexpr (Op == 2 && Bfloat == false) {
    asm volatile("fma.rn.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+0]=d;
    asm volatile("fma.rn.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+1]=d;
    asm volatile("fma.rz.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+2]=d;
    asm volatile("fma.rm.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+3]=d;
    asm volatile("fma.rp.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+4]=d;
    asm volatile("fma.rn.sat.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+5]=d;
    asm volatile("fma.rn.sat.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+6]=d;
    asm volatile("fma.rz.sat.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+7]=d;
    asm volatile("fma.rm.sat.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+8]=d;
    asm volatile("fma.rp.sat.f32.f16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+9]=d;
  }
  if constexpr (Op == 2 && Bfloat == true) {
    asm volatile("fma.rn.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+0]=d;
    asm volatile("fma.rn.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+1]=d;
    asm volatile("fma.rz.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+2]=d;
    asm volatile("fma.rm.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+3]=d;
    asm volatile("fma.rp.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+4]=d;
    asm volatile("fma.rn.sat.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+5]=d;
    asm volatile("fma.rn.sat.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+6]=d;
    asm volatile("fma.rz.sat.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+7]=d;
    asm volatile("fma.rm.sat.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+8]=d;
    asm volatile("fma.rp.sat.f32.bf16 %0,%1,%2,%3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    out[i*10+9]=d;
  }
}
// Oracle decodes the 16-bit format arithmetically, without simulator helpers.
// Finite cases use c=0 or nearby exponents, so long double retains every
// input/product/sum bit needed to decide all four FP32 rounding directions.
long double decode(uint16_t v,bool bf) {
  const unsigned mant=bf?7:10, expbits=bf?8:5, bias=bf?127:15;
  const unsigned fraction=v&((1u<<mant)-1), exponent=(v>>mant)&((1u<<expbits)-1);
  long double x;
  if(exponent==((1u<<expbits)-1)) x=fraction?NAN:INFINITY;
  else x=std::ldexp((long double)(fraction+(exponent?(1u<<mant):0)),
                    int(exponent?exponent:1)-int(bias)-int(mant));
  return v&0x8000 ? -x : x;
}
float rounded(long double exact,unsigned mode,bool sat) {
  float x=static_cast<float>(exact);
  // Conversion starts at RN; adjust by one FP32 ULP for directed modes.
  if(!std::isnan(exact)) {
    if(mode==1 && ((exact>0 && (long double)x>exact) || (exact<0 && (long double)x<exact))) x=std::nextafter(x,0.0f);
    if(mode==2 && (long double)x>exact) x=std::nextafter(x,-INFINITY);
    if(mode==3 && (long double)x<exact) x=std::nextafter(x,INFINITY);
    if(exact==0 && mode==2 && !std::signbit(exact)) x=-0.0f;
  }
  return sat?(!(x>0)?0.0f:std::min(x,1.0f)):x;
}
template<unsigned Op,bool Bfloat> void check() {
  std::vector<Input> in(256);
  const uint16_t one=Bfloat?0x3f80:0x3c00;
  const uint16_t edge[]={0,0x8000,1,0x8001, uint16_t(Bfloat?0x007f:0x03ff),
    uint16_t(Bfloat?0x0080:0x0400), uint16_t(one-1),one,uint16_t(one+1),
    uint16_t(Bfloat?0x7f7f:0x7bff),uint16_t(Bfloat?0xff7f:0xfbff),
    uint16_t(Bfloat?0x7f80:0x7c00),uint16_t(Bfloat?0xff80:0xfc00),
    uint16_t(Bfloat?0x7fc1:0x7e01)};
  for(unsigned i=0;i<256;++i) {
    uint16_t a,b;float c;
    if(i<128) {a=edge[i%14];b=one;c=0.0f;}
    else {a=one+(i%7)-3;b=one+(i%5)-2;c=i%3? -1.0f:std::ldexp(1.0f,-24);}
    if(i%2) a^=0x8000;
    in[i]={unsigned(a)|(unsigned(b)<<16),c};
  }
  if constexpr (Bfloat) {
    // FMA must not round the product before adding c: half an FP32
    // subnormal plus one subnormal, and an overflowing product canceled by c.
    in[250]={0x37000001u,std::ldexp(1.0f,-149)};
    in[251]={0x40007f7fu,-0x1.fffffep127f};
  }
  Input *device;float *output;
  ASSERT_EQ(cudaMalloc(&device,in.size()*sizeof(Input)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output,in.size()*10*sizeof(float)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device,in.data(),in.size()*sizeof(Input),cudaMemcpyHostToDevice),cudaSuccess);
  mixed<Op,Bfloat><<<4,64>>>(device,output);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  std::vector<float> actual(in.size()*10);
  ASSERT_EQ(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost),cudaSuccess);
  const unsigned mode[5]={0,0,1,2,3};
  for(unsigned i=0;i<in.size();++i) {
    const long double a=decode(in[i].packed,Bfloat),b=decode(in[i].packed>>16,Bfloat),c=in[i].c;
    const long double exact=Op==0?a+c:Op==1?a-c:a*b+c;
    for(unsigned j=0;j<10;++j) {
      const float expected=rounded(exact,mode[j%5],j>=5),got=actual[i*10+j];
      if(std::isnan(expected)) EXPECT_TRUE(std::isnan(got)) << i << ':' << j;
      else EXPECT_EQ(got,expected) << i << ':' << j;
    }
  }
  EXPECT_EQ(cudaFree(device),cudaSuccess);EXPECT_EQ(cudaFree(output),cudaSuccess);
}
TEST(MixedArithmeticLaunch, HalfAdd) {check<0,false>();}
TEST(MixedArithmeticLaunch, HalfSub) {check<1,false>();}
TEST(MixedArithmeticLaunch, HalfFma) {check<2,false>();}
TEST(MixedArithmeticLaunch, BfloatAdd) {check<0,true>();}
TEST(MixedArithmeticLaunch, BfloatSub) {check<1,true>();}
TEST(MixedArithmeticLaunch, BfloatFma) {check<2,true>();}
} // namespace
