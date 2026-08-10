#ifndef FLASHGPU_TEST_PTX_MBARRIER_CUH_
#define FLASHGPU_TEST_PTX_MBARRIER_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace flashgpu::test::ptx {

template <typename T>
__device__ __forceinline__ uint32_t smem_u32_addr(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <typename T>
__device__ __forceinline__ uint64_t smem_u64_addr(T* ptr) {
  return static_cast<uint64_t>(__cvta_generic_to_shared(ptr));
}

template <typename T>
__device__ __forceinline__ void mbarrier_init(T* bar_addr,
                                              uint32_t expected_arrivals) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(expected_arrivals));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive(T* bar_addr) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(bar_ptr));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive_count(T* bar_addr,
                                                      uint32_t count) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(count));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive_expect_tx(T* bar_addr,
                                                          uint32_t bytes) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(bytes));
}

template <typename T>
__device__ __forceinline__ bool mbarrier_try_wait_parity(T* bar_addr,
                                                         uint32_t parity) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  uint32_t complete = 0;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(complete)
      : "r"(bar_ptr), "r"(parity));
  return complete != 0;
}

template <typename T>
__device__ __forceinline__ void mbarrier_wait_parity(T* bar_addr,
                                                     uint32_t parity) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "wait_loop%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra.uni wait_loop%=;\n"
      "}\n"
      :
      : "r"(bar_ptr), "r"(parity));
}

template <typename T>
__device__ __forceinline__ void mbarrier_inval(T* bar_addr) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" : : "r"(bar_ptr));
}

}  // namespace flashgpu::test::ptx

#endif  // FLASHGPU_TEST_PTX_MBARRIER_CUH_
