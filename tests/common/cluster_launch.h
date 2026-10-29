// Host helpers for CUDA Thread Block Cluster launches under FlashGPU-Sim.
// Wraps cudaLaunchKernelExC + cudaLaunchAttributeClusterDimension so tests
// do not need to hand-roll launch configs.

#ifndef FLASH_TEST_CLUSTER_LAUNCH_H
#define FLASH_TEST_CLUSTER_LAUNCH_H

#include <cuda_runtime.h>

namespace flash_test {

// Launch a kernel with an explicit Thread Block Cluster dimension.
// Falls back to cudaLaunchKernel if the attribute path is unavailable
// (should not happen on this branch after ExC is implemented).
inline cudaError_t launch_kernel_with_cluster(const void *func, dim3 grid,
                                              dim3 block, dim3 cluster_dim,
                                              void **args,
                                              size_t dynamic_smem = 0,
                                              cudaStream_t stream = 0) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = dynamic_smem;
  cfg.stream = stream;

  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_dim.x;
  attr.val.clusterDim.y = cluster_dim.y;
  attr.val.clusterDim.z = cluster_dim.z;

  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  return cudaLaunchKernelExC(&cfg, func, args);
}

}  // namespace flash_test

#endif  // FLASH_TEST_CLUSTER_LAUNCH_H
