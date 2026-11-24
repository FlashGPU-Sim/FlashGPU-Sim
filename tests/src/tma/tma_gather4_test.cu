#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include "../tma/cp_kernels.cuh"

namespace {
__global__ void gather_four(const CUtensorMap *map, unsigned *output, int x,
                            int y0, int y1, int y2, int y3) {
  __shared__ __align__(128) unsigned tile[4 * 32];
  __shared__ __align__(8) unsigned long long barrier;
  if (threadIdx.x == 0) mbarrier_init(&barrier, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(&barrier, sizeof(tile));
    const unsigned shared = __cvta_generic_to_shared(tile);
    const unsigned bar = __cvta_generic_to_shared(&barrier);
    asm volatile("cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
                 ::"r"(shared),"l"(map),"r"(x),"r"(y0),"r"(y1),"r"(y2),"r"(y3),"r"(bar):"memory");
  }
  wait(&barrier, 0);
#pragma unroll
  for (unsigned i = 0; i < 4; ++i)
    output[threadIdx.x + i * 32] = tile[threadIdx.x + i * 32];
}

TEST(TmaGather4Integration, RepeatedPermutedAndOutOfBoundsRows) {
  std::vector<unsigned> input(64*8);
  for (unsigned i=0;i<input.size();++i) input[i]=i*31+7;
  unsigned *device_input=nullptr, *device_output=nullptr;
  CUtensorMap *device_map=nullptr;
  ASSERT_EQ(cudaMalloc(&device_input,input.size()*sizeof(unsigned)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output,128*sizeof(unsigned)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_map,sizeof(CUtensorMap)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input,input.data(),input.size()*sizeof(unsigned),cudaMemcpyHostToDevice),cudaSuccess);
  alignas(128) CUtensorMap map{};
  const cuuint64_t dimensions[2]={64,8}, strides[1]={64*sizeof(unsigned)};
  const cuuint32_t box[2]={32,1}, steps[2]={1,1};
  ASSERT_EQ(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,
      device_input,dimensions,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),CUDA_SUCCESS);
  ASSERT_EQ(cudaMemcpy(device_map,&map,sizeof(map),cudaMemcpyHostToDevice),cudaSuccess);
  for (int x : {0,48}) {
    gather_four<<<1,32>>>(device_map,device_output,x,5,-1,5,8);
    ASSERT_EQ(cudaGetLastError(),cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
    std::array<unsigned,128> result{};
    ASSERT_EQ(cudaMemcpy(result.data(),device_output,sizeof(result),cudaMemcpyDeviceToHost),cudaSuccess);
    const int rows[4]={5,-1,5,8};
    for (unsigned slot=0;slot<4;++slot)
      for (unsigned col=0;col<32;++col) {
        unsigned expected=0;
        if (rows[slot]>=0 && rows[slot]<8 && x+col<64) expected=input[rows[slot]*64+x+col];
        EXPECT_EQ(result[slot*32+col],expected) << "slot="<<slot<<" col="<<col;
      }
  }
  EXPECT_EQ(cudaFree(device_map),cudaSuccess);
  EXPECT_EQ(cudaFree(device_output),cudaSuccess);
  EXPECT_EQ(cudaFree(device_input),cudaSuccess);
}
}
