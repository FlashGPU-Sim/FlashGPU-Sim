#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
template <int Shape, int Swizzle = 0, bool ExplicitStride = false, unsigned Issuer = 0>
__global__ void copy_tile(unsigned *out) {
  __shared__ __align__(1024) unsigned input[4096];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned base;
  const unsigned tid = threadIdx.x;
  for (unsigned i = tid; i < 4096; i += 128)
    input[i] = blockIdx.x * 100000 + i + 17;
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (tid < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
    if (tid == 0)
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar)
                   : "memory");
  }
  __syncthreads();
  const unsigned shared = (unsigned)__cvta_generic_to_shared(input);
  unsigned long long desc = (shared >> 4) | (1ull << 46);
  if constexpr (Shape < 2) {
    desc |= (unsigned long long)(Swizzle ? 1 : 128) << 16;
    desc |= (unsigned long long)(Swizzle ? Swizzle / 2 : 8) << 32;
    desc |= (unsigned long long)(Swizzle == 128  ? 2
                                 : Swizzle == 64 ? 4
                                 : Swizzle == 32 ? 6
                                                 : 0)
            << 61;
  }
  if constexpr (ExplicitStride) desc |= 8ull << 32;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (tid == Issuer) {
    if constexpr (Shape == 0)
      asm volatile("tcgen05.cp.cta_group::1.128x256b [%0],%1;" ::"r"(base),
                   "l"(desc)
                   : "memory");
    if constexpr (Shape == 1)
      asm volatile("tcgen05.cp.cta_group::1.128x128b [%0],%1;" ::"r"(base),
                   "l"(desc)
                   : "memory");
    if constexpr (Shape == 2)
      asm volatile(
          "tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0],%1;" ::"r"(base),
          "l"(desc)
          : "memory");
    if constexpr (Shape == 3)
      asm volatile(
          "tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0],%1;" ::"r"(base),
          "l"(desc)
          : "memory");
    if constexpr (Shape == 4)
      asm volatile(
          "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0],%1;" ::"r"(base),
          "l"(desc)
          : "memory");
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(bar)
        : "memory");
  }
  asm volatile(
      "{ .reg .pred p; wait_copy: mbarrier.try_wait.parity.shared::cta.b64 "
      "p,[%0],0; @!p bra wait_copy; }" ::"r"(bar)
      : "memory");
  constexpr unsigned words = Shape == 0 ? 8 : 4;
  for (unsigned c = 0; c < words; c += 4) {
    unsigned a, b, d, e;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(a), "=r"(b), "=r"(d), "=r"(e)
                 : "r"(base + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    const unsigned i = (blockIdx.x * 128 + tid) * words + c;
    out[i] = a;
    out[i + 1] = b;
    out[i + 2] = d;
    out[i + 3] = e;
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,32;" ::"r"(base)
        : "memory");
  }
}
template <int Shape, int Swizzle = 0, bool ExplicitStride = false, unsigned Issuer = 0>
void check_copy() {
  constexpr unsigned words = Shape == 0 ? 8 : 4;
  unsigned *device = nullptr;
  std::vector<unsigned> result(2 * 128 * words);
  ASSERT_EQ(cudaMalloc(&device, result.size() * 4), cudaSuccess);
  copy_tile<Shape, Swizzle, ExplicitStride, Issuer><<<2, 128>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), device, result.size() * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned row = 0; row < 128; ++row)
      for (unsigned col = 0; col < words; ++col) {
        unsigned source_row = Shape == 4   ? row % 32
                              : Shape == 2 ? row % 64
                              : Shape == 3 ? (row / 64) * 32 + row % 32
                                           : row;
        unsigned offset = source_row * 16 + col * 4;
        if constexpr (Shape >= 2 && !ExplicitStride)
          offset = (source_row % 8) * 16 + col * 4;
        if (Shape < 2) {
          offset = Swizzle ? row * Swizzle + col * 4
                           : row * 16 + (col / 4) * 2048 + (col % 4) * 4;
          if (Swizzle) offset ^= ((offset >> 7) & (Swizzle / 16 - 1)) << 4;
        }
        EXPECT_EQ(result[(cta * 128 + row) * words + col],
                  cta * 100000 + offset / 4 + 17)
            << "row=" << row << " col=" << col;
      }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
TEST(Tcgen05Copy, Full128x256) { check_copy<0>(); }
TEST(Tcgen05Copy, Full128x128) { check_copy<1>(); }
TEST(Tcgen05Copy, WarpPairs02_13) { check_copy<2>(); }
TEST(Tcgen05Copy, WarpPairs01_23) { check_copy<3>(); }
TEST(Tcgen05Copy, FourWayBroadcast) { check_copy<4>(); }
TEST(Tcgen05Copy, Swizzle32) { check_copy<0, 32>(); }
TEST(Tcgen05Copy, Swizzle64) { check_copy<0, 64>(); }
TEST(Tcgen05Copy, Swizzle128) { check_copy<0, 128>(); }
TEST(Tcgen05Copy, ExplicitContiguousBroadcastStride) {
  check_copy<2,0,true>(); check_copy<3,0,true>(); check_copy<4,0,true>();
}
TEST(Tcgen05Copy, NonzeroIssuingLane) { check_copy<0, 0, false, 1>(); }
}  // namespace
