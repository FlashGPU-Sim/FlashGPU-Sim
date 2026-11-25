#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include "mma_test_layout.h"

namespace {
template <bool BF16>
__device__ unsigned short encode(float x) {
  if constexpr (BF16) return __bfloat16_as_ushort(__float2bfloat16(x));
  return __half_as_ushort(__float2half(x));
}

// Four independent B buffers. Change backing memory after fill, so treating
// use/lastuse as an ordinary shared read cannot pass the numerical oracle.
template <bool BF16, unsigned N, bool TmemA>
__global__ void ws_collectors(float *out, unsigned long long zero_mask = 0) {
  using A = MmaTestLayout<0, false, 128, 16, 2>;
  using B = MmaTestLayout<0, false, N, 16, 2>;
  __shared__ __align__(128) unsigned char a[A::Storage], b[4][B::Storage];
  __shared__ unsigned base;
  __shared__ unsigned long long bar;
  const unsigned t = threadIdx.x;
  const unsigned bp = __cvta_generic_to_shared(&bar);
  if (t == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bp) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],256;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  for (unsigned i = t; i < 128 * 16; i += 128) {
    const unsigned row = i / 16, k = i % 16;
    *reinterpret_cast<unsigned short *>(a + A::offset(row, k)) =
        encode<BF16>(int(row % 64 % 5) - 2 + int(k % 3));
  }
  __syncthreads();
  if constexpr (TmemA) {
    for (unsigned k = 0; k < 8; ++k) {
      const unsigned offset =
          (t / 8) * 256 + (t % 8) * 16 + (k / 4) * 128 + (k % 4) * 4;
      const unsigned word = *reinterpret_cast<unsigned *>(a + offset);
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(
                       base + 128 + k),
                   "r"(word)
                   : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  for (unsigned round = 0; round < 2; ++round) {
    for (unsigned stage = 0; stage < 2; ++stage) {
      for (unsigned s = 0; s < 4; ++s)
        for (unsigned i = t; i < N * 16; i += 128) {
          const unsigned col = i / 16, k = i % 16;
          const float value = (s + 1) * (col % 3 + 1) + k % 2 + round * 3 +
                              blockIdx.x + stage * 10;
          *reinterpret_cast<unsigned short *>(b[s] + B::offset(col, k)) =
              encode<BF16>(value);
        }
      __syncthreads();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (t == (TmemA ? 32u : 0u)) {
        const unsigned long long ad =
            (__cvta_generic_to_shared(a) >> 4) | (uint64_t(A::Lbo / 16) << 16) |
            (uint64_t(A::Sbo / 16) << 32) | (1ull << 46);
        unsigned long long bd[4];
        for (unsigned s = 0; s < 4; ++s)
          bd[s] = (__cvta_generic_to_shared(b[s]) >> 4) |
                  (uint64_t(B::Lbo / 16) << 16) |
                  (uint64_t(B::Sbo / 16) << 32) | (1ull << 46);
        constexpr unsigned desc = (4u << 24) | ((N / 8) << 17) | 0x10 |
                                  (BF16 ? (1u << 7) | (1u << 10) : 0);
#define ISSUE(S, OP, ACC)                                                \
  if constexpr (TmemA) {                                                 \
    asm volatile("{.reg .pred p; mov.pred p," #ACC                       \
                 "; "                                                    \
                 "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b" #S \
                 "::" OP " [%0],[%1],%2,%3,p,%4;}" ::"r"(base),             \
                 "r"(base + 128), "l"(bd[S]), "r"(desc), "l"(zero_mask)                  \
                 : "memory");                                            \
  } else {                                                               \
    asm volatile("{.reg .pred p; mov.pred p," #ACC                       \
                 "; "                                                    \
                 "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b" #S \
                 "::" OP " [%0],%1,%2,%3,p,%4;}" ::"r"(base),               \
                 "l"(ad), "l"(bd[S]), "r"(desc), "l"(zero_mask)                          \
                 : "memory");                                            \
  }
#define ALL(OP) ISSUE(0, OP, 1) ISSUE(1, OP, 1) ISSUE(2, OP, 1) ISSUE(3, OP, 1)
        if (stage == 0) {
          ISSUE(0, "fill", 0)
          ISSUE(1, "fill", 1) ISSUE(2, "fill", 1) ISSUE(3, "fill", 1)
        } else {
          ALL("use") ALL("lastuse") ALL("discard")
        }
#undef ALL
#undef ISSUE
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(
                bp)
            : "memory");
      }
      asm volatile(
          "{.reg .pred p; retry: mbarrier.try_wait.parity.shared::cta.b64 "
          "p,[%0],%1; @!p bra retry;}" ::"r"(bp),
          "r"(stage)
          : "memory");
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      __syncthreads();
    }
    for (unsigned c = 0; c < N / 2; ++c) {
      unsigned value;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                   : "=r"(value)
                   : "r"(base + c));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      out[((blockIdx.x * 2 + round) * 64 + t % 64) * N + (t / 64) * (N / 2) +
          c] = __uint_as_float(value);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,256;" ::"r"(base)
        : "memory");
  }
}

template <bool BF16, unsigned N, bool TmemA>
void check() {
  float *device;
  std::vector<float> out(2 * 2 * 64 * N);
  ASSERT_EQ(cudaMalloc(&device, out.size() * sizeof(float)), cudaSuccess);
  ws_collectors<BF16, N, TmemA><<<2, 128>>>(device);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), device, out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned round = 0; round < 2; ++round)
      for (unsigned row = 0; row < 64; ++row)
        for (unsigned col = 0; col < N; ++col) {
          float expected = 0;
          for (unsigned slot = 0; slot < 4; ++slot)
            for (unsigned k = 0; k < 16; ++k) {
              const float a = int(row % 5) - 2 + int(k % 3);
              const float b =
                  (slot + 1) * (col % 3 + 1) + k % 2 + round * 3 + cta;
              expected +=
                  a * (4 * b + 10);  // fill + use + lastuse + fresh discard
            }
          ASSERT_EQ(out[((cta * 2 + round) * 64 + row) * N + col], expected)
              << cta << '/' << round << '/' << row << '/' << col;
        }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
TEST(Tcgen05WsCollectors, HalfShared) { check<false, 64, false>(); }
TEST(Tcgen05WsCollectors, BfloatShared) { check<true, 128, false>(); }
TEST(Tcgen05WsCollectors, HalfTmem) { check<false, 128, true>(); }
TEST(Tcgen05WsCollectors, BfloatTmem) { check<true, 64, true>(); }
TEST(Tcgen05WsCollectors, HalfSharedWide) { check<false, 256, false>(); }
TEST(Tcgen05WsCollectors, BfloatTmemWide) { check<true, 256, true>(); }
}  // namespace
