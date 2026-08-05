#ifndef FLASHGPU_TEST_PTX_TMA_CUH_
#define FLASHGPU_TEST_PTX_TMA_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace flashgpu::test::ptx {

__device__ __forceinline__ void tensormap_set_global_address(uint64_t tmap_smem,
                                                             uint64_t address) {
  asm volatile(
      "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], "
      "%1;\n"
      :
      : "l"(tmap_smem), "l"(address));
}

__device__ __forceinline__ void tensormap_cp_fenceproxy(uint64_t global_tmap,
                                                        uint64_t smem_tmap) {
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release."
      "gpu.sync.aligned [%0], [%1], 0x80;\n"
      :
      : "l"(global_tmap), "l"(smem_tmap));
}

__device__ __forceinline__ void fence_proxy_tensormap_acquire(
    uint64_t global_tmap) {
  asm volatile(
      "fence.proxy.tensormap::generic.acquire.gpu [%0], 0x80;\n"
      "cp.async.bulk.commit_group;\n"
      "cp.async.bulk.wait_group.read 0;\n"
      :
      : "l"(global_tmap));
}

__device__ __forceinline__ void cp_async_bulk_tensor_1d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0,
    uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2}], [%3];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(coord1),
        "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, int32_t coord3, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(coord3), "r"(mbar_addr));
}

__device__ __forceinline__ void cp_async_bulk_tensor_5d_load(
    uint32_t smem_addr, uint64_t tmap_addr, int32_t coord0, int32_t coord1,
    int32_t coord2, int32_t coord3, int32_t coord4, uint32_t mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(coord3), "r"(coord4), "r"(mbar_addr));
}

}  // namespace flashgpu::test::ptx

#endif  // FLASHGPU_TEST_PTX_TMA_CUH_
