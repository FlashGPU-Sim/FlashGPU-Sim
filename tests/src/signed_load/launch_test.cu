#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
namespace {
__global__ void byte_loads(const signed char *input, long long *output) {
  int s32,u32;long long s64;unsigned long long u64;
  const signed char *p=input+threadIdx.x;
  asm volatile("ld.global.s8 %0,[%1];" : "=r"(s32) : "l"(p));
  asm volatile("ld.global.s8 %0,[%1];" : "=l"(s64) : "l"(p));
  asm volatile("ld.global.u8 %0,[%1];" : "=r"(u32) : "l"(p));
  asm volatile("ld.global.u8 %0,[%1];" : "=l"(u64) : "l"(p));
  output[4*threadIdx.x]=s32;output[4*threadIdx.x+1]=s64;
  output[4*threadIdx.x+2]=u32;output[4*threadIdx.x+3]=u64;
}
TEST(PtxSignedByteLoad, SignedAndUnsignedInto32And64BitDestinations) {
  std::array<signed char,32> input;
  for(unsigned i=0;i<32;++i)input[i]=static_cast<signed char>(i*8);
  input[0]=-1;input[1]=0;input[2]=1;input[3]=127;input[4]=-128;
  std::array<long long,128> output{};signed char *a;long long *b;
  ASSERT_EQ(cudaMalloc(&a,sizeof(input)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&b,sizeof(output)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(a,input.data(),sizeof(input),cudaMemcpyHostToDevice),cudaSuccess);
  byte_loads<<<1,32>>>(a,b);ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(),b,sizeof(output),cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<32;++i) {
    EXPECT_EQ(output[4*i],int(input[i]));EXPECT_EQ(output[4*i+1],int(input[i]));
    EXPECT_EQ(output[4*i+2],static_cast<unsigned char>(input[i]));
    EXPECT_EQ(output[4*i+3],static_cast<unsigned char>(input[i]));
  }
  EXPECT_EQ(cudaFree(a),cudaSuccess);EXPECT_EQ(cudaFree(b),cudaSuccess);
}
} // namespace
