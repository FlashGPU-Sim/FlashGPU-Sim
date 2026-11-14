// End-to-end SASS loader regression: module constant-bank relocations,
// zero initialization, symbol copy offsets, and state shared by two kernels.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
__device__ unsigned long long values[4];
__device__ unsigned long long observed;
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(2); } } while(0)
__global__ void write_globals() { if (threadIdx.x != 0) return; observed = values[0]; values[2] = values[1] + 7; }
__global__ void read_globals() { if (threadIdx.x != 0) return; values[3] = values[2] * 3; }
int main(int argc, char **argv) {
  const auto attr = argc > 1 && argv[1][0] == 'c'
      ? cudaFuncAttributePreferredSharedMemoryCarveout
      : cudaFuncAttributeMaxDynamicSharedMemorySize;
  CUDA(cudaFuncSetAttribute(write_globals, attr,
      attr == cudaFuncAttributeMaxDynamicSharedMemorySize ? 4096 : 100));
  CUDA(cudaFuncSetAttribute(read_globals, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
  if (cudaFuncSetAttribute(write_globals, cudaFuncAttributeMaxDynamicSharedMemorySize, -1) != cudaErrorInvalidValue) return 3;
  if (cudaFuncSetAttribute(read_globals, cudaFuncAttributePreferredSharedMemoryCarveout, 101) != cudaErrorInvalidValue) return 4;
  if (cudaFuncSetAttribute(static_cast<const void *>(nullptr), attr, 0) != cudaErrorInvalidDeviceFunction) return 5;
  cudaGetLastError();
  unsigned long long input=11, out[4]={}, zero=99;
  CUDA(cudaMemcpyToSymbol(values,&input,sizeof(input),sizeof(input)));
  write_globals<<<1,32>>>();
  CUDA(cudaDeviceSynchronize());
  read_globals<<<1,32>>>();
  CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpyFromSymbol(out,values,sizeof(out)));
  CUDA(cudaMemcpyFromSymbol(&zero,observed,sizeof(zero)));
  CUDA(cudaDeviceSynchronize());
  unsigned bad=zero!=0 || out[0]!=0 || out[1]!=11 || out[2]!=18 || out[3]!=54;
  std::printf("ORACLE device_globals mismatches=%u values=%llu,%llu,%llu,%llu observed=%llu\n",bad,out[0],out[1],out[2],out[3],zero);
  return bad!=0;
}
