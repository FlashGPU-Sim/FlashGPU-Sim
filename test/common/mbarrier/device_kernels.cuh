#ifndef TEST_COMMON_MBARRIER_DEVICE_KERNELS_CUH
#define TEST_COMMON_MBARRIER_DEVICE_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

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
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(expected_arrivals));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive(T* bar_addr) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar_ptr));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive_count(T* bar_addr,
                                                      uint32_t count) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
               "r"(count));
}

template <typename T>
__device__ __forceinline__ void mbarrier_arrive_expect_tx(T* bar_addr,
                                                          uint32_t bytes) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(bytes));
}

template <typename T>
__device__ __forceinline__ void mbarrier_expect_tx(T* bar_addr,
                                                   uint32_t expected_tx) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile(
      "mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
      "r"(expected_tx));
}

template <typename T>
__device__ __forceinline__ bool mbarrier_try_wait_parity(T* bar_addr,
                                                         uint32_t parity) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  uint32_t complete = 0;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(complete)
               : "r"(bar_ptr), "r"(parity));
  return complete != 0;
}

template <typename T>
__device__ __forceinline__ void mbarrier_wait(T* bar_addr, uint32_t parity) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("{\n"
               ".reg .pred p;\n"
               "wait_loop%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra.uni wait_loop%=;\n"
               "}\n" ::"r"(bar_ptr),
               "r"(parity));
}

template <typename T>
__device__ __forceinline__ void mbarrier_wait_parity(T* bar_addr,
                                                     uint32_t parity) {
  mbarrier_wait(bar_addr, parity);
}

template <typename T>
__device__ __forceinline__ void mbarrier_inval(T* bar_addr) {
  const uint32_t bar_ptr = smem_u32_addr(bar_addr);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(bar_ptr));
}

__device__ __forceinline__ void tensormap_set_global_address(uint64_t tmap_smem,
                                                             uint64_t addr) {
  asm volatile(
      "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n"
      ::"l"(tmap_smem),
      "l"(addr));
}

__device__ __forceinline__ void tensormap_cp_fenceproxy(uint64_t global_tmap,
                                                        uint64_t smem_tmap) {
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release."
      "gpu.sync.aligned [%0], [%1], 0x80;\n" ::"l"(global_tmap),
      "l"(smem_tmap));
}

__device__ __forceinline__ void fence_proxy_tensormap_acquire(
    uint64_t global_tmap) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 0x80;\n"
               "cp.async.bulk.commit_group;\n"
               "cp.async.bulk.wait_group.read 0;\n" ::"l"(global_tmap));
}

__device__ __forceinline__ void cp_async_bulk_tensor_1d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0,
    uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2}], [%3];\n" ::"r"(smem_addr),
      "l"(tmap_addr), "r"(coord0), "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr),
      "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_addr),
      "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2), "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, int32_t coord3, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_addr),
      "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3),
      "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_5d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, int32_t coord3, int32_t coord4, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(smem_addr),
      "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3),
      "r"(coord4), "r"(mbar_addr));
}

#endif  // TEST_COMMON_MBARRIER_DEVICE_KERNELS_CUH
