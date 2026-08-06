#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#define C10_CUDA_CHECK(EXPR)                                                \
  do {                                                                      \
    const cudaError_t status__ = (EXPR);                                    \
    if (status__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status__));                           \
    }                                                                       \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
