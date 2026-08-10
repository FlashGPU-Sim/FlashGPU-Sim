#pragma once

#include <cstdint>
#include <tuple>

#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace at::cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    at::PhiloxCudaState arg) {
  if (arg.captured_) {
    return std::make_tuple(static_cast<uint64_t>(*arg.seed_.ptr),
                           static_cast<uint64_t>(*arg.offset_.ptr +
                                                 arg.offset_intragraph_));
  }
  return std::make_tuple(arg.seed_.val, arg.offset_.val);
}

}  // namespace at::cuda::philox
