// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory tma_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

#pragma once

#include "cp_kernels.cuh"

// Kernel wrapper for cp with configurable producer warps
template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1,
          CP_METHOD METHOD = CP_METHOD::CP_ASYNC>
struct CPKernelWrapper {
  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(
        cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS,
                     NUM_CONSUMER_WARPS, METHOD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS,
                 NUM_CONSUMER_WARPS, METHOD>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};

// Kernel wrapper for TMA using the unified cp kernel with configurable warp counts
template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1>
struct TMAKernelWrapper {
  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(
        cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS, CP_METHOD::TMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS, CP_METHOD::TMA>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};

// Kernel wrapper for normal load (using cp kernel with NORMAL_LOAD method)
template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1>
using NormalLoadKernelWrapper =
    CPKernelWrapper<NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS,
                    CP_METHOD::NORMAL_LOAD>;

// Convenient aliases for cp.async
template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1>
using CPAsyncKernelWrapper =
    CPKernelWrapper<NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS,
                    CP_METHOD::CP_ASYNC>;