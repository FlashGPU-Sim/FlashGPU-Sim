#include <cuda_runtime.h>
#include <gtest/gtest.h>
namespace {
__global__ void grouped_loads(const unsigned *input,unsigned *out) {
  unsigned tid=threadIdx.x, value=0;
#pragma unroll
  for(unsigned group=0;group<4;++group) {
    if((tid&31)/8==group) {
      asm volatile("ld.global.u32 %0,[%1];" : "=r"(value) : "l"(input+tid));
    }
  }
  __syncwarp();
  out[tid]=value;
}
TEST(CachedLoadCompletion,DisjointLaneProducersSameRegister) {
  unsigned input[64],actual[64];for(unsigned i=0;i<64;++i) input[i]=0x12345678u+17*i;
  unsigned *a,*b;ASSERT_EQ(cudaMalloc(&a,sizeof(input)),cudaSuccess);ASSERT_EQ(cudaMalloc(&b,sizeof(input)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(a,input,sizeof(input),cudaMemcpyHostToDevice),cudaSuccess);
  grouped_loads<<<1,64>>>(a,b);ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual,b,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<64;++i) EXPECT_EQ(actual[i],input[i]) << i;
  EXPECT_EQ(cudaFree(a),cudaSuccess);EXPECT_EQ(cudaFree(b),cudaSuccess);
}
}
