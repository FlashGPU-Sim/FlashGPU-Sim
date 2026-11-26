#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreads =
    17 * 32;  // One TMA warp plus up to 16 background warps.
constexpr size_t kSmemBytes = 224 * 1024;
constexpr int kTileX = 256;
#ifdef PAIR_TMA_32K
constexpr int kTileY = 32;
#else
constexpr int kTileY = 64;
#endif
constexpr int kTileElements = kTileX * kTileY;
constexpr int kTileBytes = kTileElements * sizeof(float);
constexpr int kPipelineSlots1 = 1;
constexpr int kPipelineSlots2 = 2;
constexpr int kPipelineSlots4 = 4;
constexpr int kPipelineSlots6 = 6;

static_assert(kTileBytes == 32 * 1024 || kTileBytes == 64 * 1024);
static_assert(kTileBytes % 128 == 0, "TMA tile must be 128-byte aligned");

enum LocalityMode {
  kLocalityStride = 0,
  kLocalityHot = 1,
  kLocalityBlockHot = 2,
};

#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                           \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define CU_CHECK(expr)                                                    \
  do {                                                                    \
    CUresult err__ = (expr);                                              \
    if (err__ != CUDA_SUCCESS) {                                          \
      const char* name__ = nullptr;                                       \
      const char* text__ = nullptr;                                       \
      cuGetErrorName(err__, &name__);                                     \
      cuGetErrorString(err__, &text__);                                   \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s: %s\n", __FILE__, \
                   __LINE__, name__ ? name__ : "unknown",                 \
                   text__ ? text__ : "unknown");                          \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

__host__ __device__ inline uintptr_t align_up(uintptr_t value,
                                              uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
               :
               : "r"(smem_addr_u32(bar))
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(smem_addr_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar,
                                                     uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "waitLoop%=: \n"
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
      "@!complete bra.uni waitLoop%=;\n"
      "}\n"
      :
      : "r"(smem_addr_u32(bar)), "r"(parity)
      : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
               :
               : "r"(smem_addr_u32(bar))
               : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d(float* dst, const CUtensorMap* map,
                                            int32_t coord0, int32_t coord1,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(smem_addr_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)),
        "r"(coord0), "r"(coord1), "r"(smem_addr_u32(bar))
      : "memory");
}

__device__ __forceinline__ uint64_t tile_for_iteration(int iteration, int tiles,
                                                       LocalityMode locality) {
  if (locality == kLocalityHot) {
    return 0;
  }
  if (locality == kLocalityBlockHot) {
    return static_cast<uint64_t>(blockIdx.x);
  }
  return (static_cast<uint64_t>(iteration) * gridDim.x + blockIdx.x) %
         static_cast<uint64_t>(tiles);
}

template <int kPipelineSlots>
__device__ __forceinline__ uint32_t stream_tma_tiles(
    const CUtensorMap* map, float* slots, uint64_t* bars, int first_iteration,
    int count, int tiles, LocalityMode locality) {
  if (count <= 0) {
    return 0;
  }

#pragma unroll
  for (int slot = 0; slot < kPipelineSlots; ++slot) {
    mbarrier_init(bars + slot);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  fence_proxy_async();

  const int initial = min(count, kPipelineSlots);
  for (int i = 0; i < initial; ++i) {
    const uint64_t tile =
        tile_for_iteration(first_iteration + i, tiles, locality);
    mbarrier_arrive_expect_tx(bars + i, kTileBytes);
    tma_load_2d(slots + i * kTileElements, map, 0,
                static_cast<int32_t>(tile * kTileY), bars + i);
  }

  uint32_t sink = 0;
  // Issue a statically addressed round. Avoid modulo/division and a back edge
  // after every small tile; those throttle the one issuing thread.
  uint32_t parity = 0;
#pragma unroll 1
  for (int round = kPipelineSlots; round < count; round += kPipelineSlots) {
#pragma unroll
    for (int slot = 0; slot < kPipelineSlots; ++slot) {
      mbarrier_wait_parity(bars + slot, parity);
      const uint64_t tile = tile_for_iteration(first_iteration + round + slot,
                                                tiles, locality);
      mbarrier_arrive_expect_tx(bars + slot, kTileBytes);
      tma_load_2d(slots + slot * kTileElements, map, 0,
                  static_cast<int32_t>(tile * kTileY), bars + slot);
    }
    parity ^= 1;
  }
#pragma unroll
  for (int slot = 0; slot < kPipelineSlots; ++slot)
    mbarrier_wait_parity(bars + slot, parity);

#pragma unroll
  for (int slot = 0; slot < kPipelineSlots; ++slot) {
    mbarrier_inval(bars + slot);
  }
  return sink;
}

CUtensorMap encode_map(float* base, int tiles) {
  CUtensorMap map{};
  const uint64_t global_dim[2] = {kTileX,
                                  static_cast<uint64_t>(tiles) * kTileY};
  const uint64_t global_stride[1] = {kTileX * sizeof(float)};
  const uint32_t box_dim[2] = {kTileX, kTileY};
  const uint32_t element_stride[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, global_dim, global_stride,
      box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

} // namespace
