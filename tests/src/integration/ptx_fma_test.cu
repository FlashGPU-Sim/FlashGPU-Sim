#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void packed_fma_single_round(unsigned long long *out) {
  const unsigned long long a=0x3f8000013f800001ull;
  const unsigned long long b=0x3f7ffffe3f7ffffeull;
  const unsigned long long c=0xbf800000bf800000ull;
  unsigned long long d;
  asm volatile("fma.rn.f32x2 %0,%1,%2,%3;" : "=l"(d) : "l"(a),"l"(b),"l"(c));
  out[threadIdx.x]=d;
}
TEST(PtxFma, PackedFmaUsesOneRoundingStep) {
  unsigned long long *dst=nullptr;
  ASSERT_EQ(cudaMalloc(&dst,32*8),cudaSuccess);
  packed_fma_single_round<<<1,32>>>(dst);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  std::vector<unsigned long long> values(32);
  ASSERT_EQ(cudaMemcpy(values.data(),dst,32*8,cudaMemcpyDeviceToHost),cudaSuccess);
  // (1+2^-23)*(1-2^-23)-1 = -2^-46 exactly; two roundings produce zero.
  for (auto value:values) EXPECT_EQ(value,0xa8800000a8800000ull);
  ASSERT_EQ(cudaFree(dst),cudaSuccess);
}

__global__ void scalar_fma_single_round(unsigned *out) {
  float d;
  asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(d)
               : "f"(__uint_as_float(0x3f800001)),
                 "f"(__uint_as_float(0x3f7ffffe)), "f"(-1.0f));
  const unsigned i=threadIdx.x*5;
  out[i]=__float_as_uint(d);
  asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(d)
               : "f"(__uint_as_float(0x80000001)), "f"(1.0f), "f"(-0.0f));
  out[i+1]=__float_as_uint(d);
  asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(d)
               : "f"(-0x1p-126f), "f"(0.5f), "f"(-0.0f));
  out[i+2]=__float_as_uint(d);
  asm volatile("fma.rn.sat.f32 %0,%1,%2,%3;" : "=f"(d)
               : "f"(__uint_as_float(0x7fc00001)), "f"(1.0f), "f"(0.0f));
  out[i+3]=__float_as_uint(d);
  asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(d)
               : "f"(0x1p-126f), "f"(0.5f), "f"(0.0f));
  out[i+4]=__float_as_uint(d);
}
TEST(PtxFma, ScalarFmaUsesOneRoundingStep) {
  unsigned *dst=nullptr;unsigned values[160];
  ASSERT_EQ(cudaMalloc(&dst,sizeof(values)),cudaSuccess);
  scalar_fma_single_round<<<1,32>>>(dst);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(values,dst,sizeof(values),cudaMemcpyDeviceToHost),cudaSuccess);
  // (1+2^-23)*(1-2^-23)-1 = -2^-46. A separately rounded product yields 0.
  const unsigned expected[]={0xa8800000u,0x80000000u,0x80000000u,0u,0x00400000u};
  for(unsigned i=0;i<160;++i) EXPECT_EQ(values[i],expected[i%5]);
  ASSERT_EQ(cudaFree(dst),cudaSuccess);
}
