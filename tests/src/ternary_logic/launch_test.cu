#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
template <unsigned Lut, unsigned Mode>
__device__ __forceinline__ void apply(unsigned a, unsigned b, unsigned c, unsigned flag,
                                      unsigned *out) {
  unsigned d = 0, p = 0;
  if constexpr (Mode == 0)
    asm volatile("lop3.b32 %0,%1,%2,%3,%4;" : "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(Lut));
  else if constexpr (Mode == 1)
    asm volatile("{ .reg .pred p,q; setp.ne.u32 q,%6,0; lop3.and.b32 %0|p,%2,%3,%4,%5,q; selp.u32 %1,1,0,p; }"
                 : "=r"(d), "=r"(p) : "r"(a), "r"(b), "r"(c), "n"(Lut), "r"(flag));
  else if constexpr (Mode == 2)
    asm volatile("{ .reg .pred p,q; setp.ne.u32 q,%6,0; lop3.or.b32 %0|p,%2,%3,%4,%5,q; selp.u32 %1,1,0,p; }"
                 : "=r"(d), "=r"(p) : "r"(a), "r"(b), "r"(c), "n"(Lut), "r"(flag));
  else if constexpr (Mode == 3)
    asm volatile("{ .reg .pred p,q; setp.ne.u32 q,%5,0; lop3.and.b32 _|p,%1,%2,%3,%4,q; selp.u32 %0,1,0,p; }"
                 : "=r"(p) : "r"(a), "r"(b), "r"(c), "n"(Lut), "r"(flag));
  else
    asm volatile("{ .reg .pred p,q; setp.ne.u32 q,%5,0; lop3.or.b32 _|p,%1,%2,%3,%4,q; selp.u32 %0,1,0,p; }"
                 : "=r"(p) : "r"(a), "r"(b), "r"(c), "n"(Lut), "r"(flag));
  out[(Lut * 32 + threadIdx.x) * 2] = d;
  out[(Lut * 32 + threadIdx.x) * 2 + 1] = p;
}
template<unsigned Mode, unsigned... Lut>
__device__ void all(unsigned a, unsigned b, unsigned c, unsigned flag, unsigned *out,
                    std::integer_sequence<unsigned, Lut...>) {
  (apply<Lut, Mode>(a,b,c,flag,out), ...);
}
template<unsigned Mode>
__global__ void logic(const unsigned *in, unsigned *out) {
  unsigned tid = threadIdx.x;
  all<Mode>(in[tid], in[32+tid], in[64+tid], tid&1, out,
            std::make_integer_sequence<unsigned, 256>{});
}
template<unsigned Mode>
void check() {
  std::vector<unsigned> in(96);
  for(unsigned t=0;t<32;++t) {
    // Every input combination at every bit, plus mixed words and both q values.
    in[t] = t<16 ? ((t&8)?~0u:0) : 0xf0f00ff0u * t;
    in[32+t] = t<16 ? ((t&4)?~0u:0) : 0xcccc3333u * t;
    in[64+t] = t<16 ? ((t&2)?~0u:0) : 0xaaaa5555u * t;
  }
  unsigned *di,*dout;
  ASSERT_EQ(cudaMalloc(&di,in.size()*4),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dout,256*32*2*4),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(di,in.data(),in.size()*4,cudaMemcpyHostToDevice),cudaSuccess);
  logic<Mode><<<1,32>>>(di,dout);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  std::vector<unsigned> out(256*32*2);
  ASSERT_EQ(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned lut=0;lut<256;++lut) for(unsigned t=0;t<32;++t) {
    unsigned d=0;
    for(unsigned bit=0;bit<32;++bit) {
      const unsigned index=(((in[t]>>bit)&1)*4)+(((in[32+t]>>bit)&1)*2)+((in[64+t]>>bit)&1);
      d |= ((lut>>index)&1)<<bit;
    }
    const unsigned p=Mode==0?0:Mode==1||Mode==3?(d!=0)&&(t&1):(d!=0)||(t&1);
    ASSERT_EQ(out[(lut*32+t)*2],Mode>=3?0:d) << "LUT="<<lut<<" lane="<<t;
    ASSERT_EQ(out[(lut*32+t)*2+1],p) << "LUT="<<lut<<" lane="<<t;
  }
  EXPECT_EQ(cudaFree(di),cudaSuccess);EXPECT_EQ(cudaFree(dout),cudaSuccess);
}
TEST(Lop3Launch, AllLookupTables) { check<0>(); }
TEST(Lop3Launch, PredicateAnd) { check<1>(); }
TEST(Lop3Launch, PredicateOr) { check<2>(); }
TEST(Lop3Launch, SinkAnd) { check<3>(); }
TEST(Lop3Launch, SinkOr) { check<4>(); }
}
