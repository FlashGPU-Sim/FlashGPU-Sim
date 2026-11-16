#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
__global__ void cta_predicate_reduce(unsigned *out) {
  for (unsigned round=0;round<4;++round) {
    unsigned value = round==0 ? 0 : round==1 ? threadIdx.x==blockDim.x-1 : round==2;
    unsigned any=__syncthreads_or(value),all=__syncthreads_and(value);
    out[(blockIdx.x*4+round)*blockDim.x+threadIdx.x]=any+2*all;
  }
}
TEST(CtaReduction, CtaPredicateReductionRetainsEachGeneration) {
  unsigned *dst=nullptr;
  ASSERT_EQ(cudaMalloc(&dst,3*4*128*4),cudaSuccess);
  for (unsigned threads : {32u,64u,128u}) {
    cta_predicate_reduce<<<3,threads>>>(dst);
    ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
    std::vector<unsigned> out(threads*4*3);
    ASSERT_EQ(cudaMemcpy(out.data(),dst,out.size()*4,cudaMemcpyDeviceToHost),cudaSuccess);
    for (unsigned i=0;i<out.size();++i) EXPECT_EQ(out[i],(i/threads)%4==0?0u:(i/threads)%4==1?1u:(i/threads)%4==2?3u:0u);
  }
  ASSERT_EQ(cudaFree(dst),cudaSuccess);
}
