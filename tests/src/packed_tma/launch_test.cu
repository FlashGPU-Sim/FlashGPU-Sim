#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

namespace {
struct alignas(64) PackedMap {
  uint64_t words[16]{};
};
template <bool Expanded, unsigned Rank = (Expanded ? 4 : 2)>
__global__ void packed_load(const __grid_constant__ PackedMap map,
                            unsigned *out) {
  constexpr unsigned bytes = Expanded ? 16384 : 8192;
  constexpr unsigned base_words = Expanded ? 32 : 16;
  __shared__ __align__(1024) unsigned storage[(bytes + 256) / 4];
  __shared__ __align__(8) unsigned long long barrier;
  for (unsigned i = threadIdx.x; i < (bytes + 256) / 4; i += blockDim.x)
    storage[i] = 0xa5a5a5a5u;
  __syncthreads();
  const unsigned dst = __cvta_generic_to_shared(storage + base_words);
  const unsigned bar = __cvta_generic_to_shared(&barrier);
  const int c = blockIdx.x;
  const int x = c == 1 ? -32 : c == 2 ? (Expanded ? 224 : 96) : 0;
  const int y = c == 2 ? 127 : c == 3 ? -128 : c;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],8192;" ::"r"(bar)
        : "memory");
    if constexpr (Rank == 4)
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%2,%3,1,1}],[%4];" ::"r"(dst),
          "l"(&map), "r"(x), "r"(y), "r"(bar)
          : "memory");
    else if constexpr (Rank == 3)
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%2,%3,1}],[%4];" ::"r"(dst),
          "l"(&map), "r"(x), "r"(y), "r"(bar)
          : "memory");
    else
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%2,%3}],[%4];" ::"r"(dst),
          "l"(&map), "r"(x), "r"(y), "r"(bar)
          : "memory");
    asm volatile(
        "{.reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],0; @!p bra wait;}" ::"r"(bar)
        : "memory");
  }
  __syncthreads();
  unsigned *result = out + c * (bytes / 4 + 2);
  for (unsigned i = threadIdx.x * 4; i < bytes; i += blockDim.x * 4) {
    const unsigned offset = i ^ ((((dst + i) >> 7) & (Expanded ? 7 : 3)) << 4);
    result[1 + i / 4] = storage[base_words + offset / 4];
  }
  if (threadIdx.x == 0) {
    result[0] = storage[base_words - 1];
    result[bytes / 4 + 1] = storage[base_words + bytes / 4];
  }
}

template <bool Expanded, unsigned Rank = (Expanded ? 4 : 2)>
void check_packed(uint32_t control, bool host = false) {
  constexpr unsigned cols = Expanded ? 256 : 128, rows = 129;
  constexpr unsigned shared_bytes = Expanded ? 16384 : 8192;
  constexpr unsigned total = cols / 2 * rows *
                             (Rank == 4   ? 4
                              : Rank == 3 ? 2
                                          : 1);
  std::vector<unsigned char> input(total);
  for (unsigned i = 0; i < total; ++i)
    input[i] = (i * 37 + (i >> 7) + 19) % 251;
  unsigned char *in;
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&in, total), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 4 * (shared_bytes + 8)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), total, cudaMemcpyHostToDevice),
            cudaSuccess);
  PackedMap map;
  map.words[0] = uint64_t(in);
  map.words[1] = control | (uint64_t(cols / 2 / 16) << 32);
  map.words[4] = uint64_t(cols - 1) | (uint64_t(rows - 1) << 32);
  if constexpr (Rank >= 3) {
    map.words[2] = uint64_t(cols / 2 * rows / 16) |
                   (Rank == 4 ? (uint64_t(cols / 2 * rows * 2 / 16) << 32) : 0);
    map.words[5] = Rank == 4 ? 0x100000001ull : 1;
  }
  map.words[6] = 0x7f00000000000000ull;
  map.words[7] = 127;
  map.words[9] = Expanded ? 0x400 : 0x200;
  if (host) {
    const unsigned rank = Rank;
    const uint64_t dims[] = {cols, rows, 2, 2};
    const uint64_t strides[] = {cols / 2, cols / 2 * rows, cols / 2 * rows * 2};
    const unsigned box[] = {128, 128, 1, 1}, element_strides[] = {1, 1, 1, 1};
    ASSERT_EQ(
        cuTensorMapEncodeTiled(
            reinterpret_cast<CUtensorMap *>(&map),
            Expanded ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                     : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
            rank, in, dims, strides, box, element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            Expanded ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        CUDA_SUCCESS);
  }
  packed_load<Expanded, Rank><<<4, 128>>>(map, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned char> actual(4 * (shared_bytes + 8)),
      expected(actual.size(), 0);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned c = 0; c < 4; ++c) {
    const unsigned base = c * (shared_bytes + 8);
    for (unsigned i = 0; i < 4; ++i)
      expected[base + i] = expected[base + shared_bytes + 4 + i] = 0xa5;
    const int x0 = c == 1 ? -32 : c == 2 ? int(cols - 32) : 0;
    const int y0 = c == 2 ? 127 : c == 3 ? -128 : int(c);
    for (unsigned row = 0; row < 128; ++row)
      for (unsigned col = 0; col < 128; ++col) {
        const int x = x0 + col, y = y0 + row;
        if (x < 0 || y < 0 || x >= int(cols) || y >= int(rows)) continue;
        const unsigned address = (cols / 2 * rows *
                                  (Rank == 4   ? 3
                                   : Rank == 3 ? 1
                                               : 0)) +
                                 y * cols / 2 + x / 2;
        const unsigned nibble = (input[address] >> ((x % 2) * 4)) & 15;
        const unsigned byte =
            row * (Expanded ? 128 : 64) +
            (Expanded ? (col / 16) * 16 + col % 16 / 2 : col / 2);
        expected[base + 4 + byte] |= nibble << ((col % 2) * 4);
      }
  }
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(PackedTma, Compact2dControl64590) { check_packed<false>(0x64590); }
TEST(PackedTma, Compact2dControl264590) { check_packed<false>(0x264590); }
TEST(PackedTma, Compact3dControl2645a0) { check_packed<false, 3>(0x2645a0); }
TEST(PackedTma, HostCompact3d) { check_packed<false, 3>(0, true); }
TEST(PackedTma, Expanded4dWeights) { check_packed<true>(0x266630); }
TEST(PackedTma, HostCompact2d) { check_packed<false>(0, true); }
TEST(PackedTma, HostExpanded4d) { check_packed<true>(0, true); }
}  // namespace
