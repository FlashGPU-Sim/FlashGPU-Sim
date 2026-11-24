#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
struct alignas(64) Map {
  uint64_t word[16]{};
};
// Full-width elements remain 32 bits in shared memory, including TF32.
template <unsigned Rank>
__global__ void load(const __grid_constant__ Map map, unsigned *out,
                     unsigned bytes, unsigned swizzle, int x, int y) {
  __shared__ __align__(1024) unsigned tile[256];
  __shared__ __align__(8) unsigned long long barrier;
  const unsigned smem = __cvta_generic_to_shared(tile),
                 bar = __cvta_generic_to_shared(&barrier);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(bar),
        "r"(bytes)
        : "memory");
    if constexpr (Rank == 1)
      asm volatile(
          "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%3}],[%2];" ::"r"(smem),
          "l"(&map), "r"(bar), "r"(x)
          : "memory");
    if constexpr (Rank == 2)
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%3,%4}],[%2];" ::"r"(smem),
          "l"(&map), "r"(bar), "r"(x), "r"(y)
          : "memory");
    if constexpr (Rank == 3)
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%3,%4,%5}],[%2];" ::"r"(smem),
          "l"(&map), "r"(bar), "r"(x), "r"(y), "r"(1)
          : "memory");
    if constexpr (Rank == 4)
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%3,%4,%5,%5}],[%2];" ::"r"(smem),
          "l"(&map), "r"(bar), "r"(x), "r"(y), "r"(1)
          : "memory");
    if constexpr (Rank == 5)
      asm volatile(
          "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{%3,%4,%5,%5,%5}],[%2];" ::"r"(smem),
          "l"(&map), "r"(bar), "r"(x), "r"(y), "r"(1)
          : "memory");
    asm volatile(
        "{.reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],0; @!p bra wait;}" ::"r"(bar)
        : "memory");
  }
  __syncthreads();
  for (unsigned i = threadIdx.x * 4; i < bytes; i += blockDim.x * 4) {
    const unsigned offset =
        swizzle ? i ^ (((((smem + i) >> 7) & (swizzle / 16 - 1))) << 4) : i;
    out[i / 4] = tile[offset / 4];
  }
}
// Hand-specified IEEE input/output pairs exercise both tie parities, signs,
// subnormals, NaNs, infinities and overflow, without using simulator helpers.
constexpr unsigned input_bits[] = {
    0,          0x80000000, 0x3f800000, 0x3f800fff, 0x3f801000,
    0x3f801001, 0x3f803000, 0xbf803000, 0x7f800000, 0xff800000,
    0x7fc12345, 0xffc12345, 1,          0x807fffff, 0x1000,
    0x1001,     0x7fffff,   0x800000,   0x7f7fffff, 0x80001001};
constexpr unsigned tf32_bits[] = {
    0,          0x80000000, 0x3f800000, 0x3f800000, 0x3f800000,
    0x3f802000, 0x3f804000, 0xbf804000, 0x7f800000, 0xff800000,
    0x7fffe000, 0x7fffe000, 0,          0x80800000, 0,
    0x2000,     0x800000,   0x800000,   0x7f800000, 0x80002000};
template <unsigned Rank>
void check(unsigned type, unsigned swizzle, bool nan_fill, bool edge,
           bool captured = false) {
  const unsigned width = swizzle ? swizzle / 4 : 32, rows = Rank == 1 ? 1 : 4;
  const unsigned tile_rows = Rank == 1 ? 1 : 2, count = width * tile_rows;
  const std::array<cuuint64_t, 5> dims{width, rows, 2, 2, 2};
  const std::array<unsigned, 5> box{width, tile_rows, 1, 1, 1},
      estride{1, 1, 1, 1, 1};
  std::array<cuuint64_t, 5> strides{};
  size_t total = width;
  for (unsigned i = 1; i < Rank; ++i) {
    strides[i - 1] = total * 4;
    total *= dims[i];
  }
  std::vector<unsigned> input(total), output(count), expected(count);
  for (size_t i = 0; i < total; ++i) input[i] = input_bits[i % 20];
  unsigned *din, *dout;
  ASSERT_EQ(cudaMalloc(&din, total * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dout, count * 4), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(din, input.data(), total * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  Map map{};
  if (captured) {
    ASSERT_TRUE(type == 11 && (Rank == 2 || Rank == 3) && swizzle == 128 &&
                !nan_fill);
    map.word[0] = (uint64_t)din;
    map.word[1] = Rank == 2 ? 0x276390 : 0x2763a0;
    for (unsigned i = 1; i < Rank; ++i)
      map.word[(i + 2) / 2] |= (strides[i - 1] / 16) << ((i % 2) * 32);
    for (unsigned i = 0; i < Rank; ++i)
      map.word[4 + i / 2] |= (dims[i] - 1) << ((i % 2) * 32);
    map.word[6] |= uint64_t(width - 1) << 56;
    map.word[7] = tile_rows - 1;
    map.word[8] = count * 4;
    map.word[9] = 0x400;
  } else {
    ASSERT_EQ(cuTensorMapEncodeTiled(
                  reinterpret_cast<CUtensorMap *>(&map),
                  static_cast<CUtensorMapDataType>(type), Rank, din,
                  dims.data(), strides.data(), box.data(), estride.data(),
                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                  swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                  : swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                  : swizzle == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                  : CU_TENSOR_MAP_SWIZZLE_NONE,
                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                  nan_fill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                           : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              CUDA_SUCCESS);
  }
  const int x = edge ? -4 : 0, y = edge ? 3 : 1;
  for (unsigned i = 0; i < count; ++i) {
    const int gx = x + i % width, gy = y + i / width;
    if (gx < 0 || gx >= int(width) || (Rank > 1 && gy >= int(rows))) {
      expected[i] = nan_fill ? (type >= 11 ? 0x7fffe000 : 0x7fffffff) : 0;
      continue;
    }
    size_t index = gx;
    if constexpr (Rank > 1) index += gy * width;
    for (unsigned d = 2; d < Rank; ++d) index += strides[d - 1] / 4;
    const unsigned bits = input[index];
    expected[i] = type == 10 ? bits : tf32_bits[index % 20];
    if (type != 11 && (bits & 0x7fffffff) < 0x800000)
      expected[i] = bits & 0x80000000;
  }
  load<Rank><<<1, 32>>>(map, dout, count * 4, swizzle, x, y);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), dout, count * 4, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < count; ++i)
    ASSERT_EQ(output[i], expected[i])
        << "rank=" << Rank << " type=" << type << " i=" << i;
  EXPECT_EQ(cudaFree(din), cudaSuccess);
  EXPECT_EQ(cudaFree(dout), cudaSuccess);
}
template <unsigned Rank>
void canonical() {
  check<Rank>(11, 128, false, false);
  check<Rank>(12, 64, true, true);
  check<Rank>(10, 0, true, true);
}
TEST(TmaFloatLoad, Canonical1D) { canonical<1>(); }
TEST(TmaFloatLoad, Canonical2D) { canonical<2>(); }
TEST(TmaFloatLoad, Canonical3D) { canonical<3>(); }
TEST(TmaFloatLoad, Canonical4D) { canonical<4>(); }
TEST(TmaFloatLoad, Canonical5D) { canonical<5>(); }
TEST(TmaFloatLoad, CapturedTf32) {
  check<2>(11, 128, false, false, true);
  check<3>(11, 128, false, false, true);
}
}  // namespace
