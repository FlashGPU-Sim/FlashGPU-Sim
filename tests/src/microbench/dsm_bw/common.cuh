// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory dsm_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace dsm {

__device__ __forceinline__ void mbarrier_init(
    unsigned long long *barrier, unsigned expected_arrivals) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(address), "r"(expected_arrivals));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    unsigned long long *barrier, unsigned expected_bytes) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(address), "r"(expected_bytes));
}

} // namespace dsm
