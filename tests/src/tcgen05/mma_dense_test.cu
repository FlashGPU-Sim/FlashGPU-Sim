#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "mma_test_layout.h"
namespace {
enum PatternFlags : unsigned {
  kTransposeA = 1,
  kTransposeB = 2,
  kHalfOutput = 4,
  kUpperDatapaths = 8,
  kNegateB = 16,
  kSwizzle32 = 32,
  kSwizzle64 = 64,
  kSwizzle128 = 96,
  kSwizzle128Atom32 = 128,
  kShiftBase = 256,
  kNegatedAccumulatorPredicate = 512
};
float pattern_a(unsigned row, unsigned k, bool half_output) {
  float value = float((row % 3 + 1 + row / 64) * (k % 2 + 1));
  if (half_output)
    value = __half2float(__float2half(value + float(k % 3) / 64));
  return value;
}
template <unsigned Kind, unsigned M, unsigned N, bool TmemA, unsigned Flags>
struct PatternLayouts {
  static constexpr unsigned Mode = (Flags & 224) == kSwizzle32          ? 6
                                   : (Flags & 224) == kSwizzle64        ? 4
                                   : (Flags & 224) == kSwizzle128       ? 2
                                   : (Flags & 224) == kSwizzle128Atom32 ? 1
                                                                        : 0;
  static constexpr unsigned Prefix = Flags & kShiftBase ? 128 : 0;
  using A = MmaTestLayout < TmemA ? 0 : Mode,
        bool(Flags &kTransposeA), TmemA ? 128 : M, Kind < 2 ? 16 : 32,
        Kind<2 ? 2 : 1, TmemA ? 0 : Prefix>;
  using B = MmaTestLayout < Mode, bool(Flags &kTransposeB), N,
        Kind < 2 ? 16 : 32, Kind<2 ? 2 : 1, Prefix>;
};
template <unsigned Kind, unsigned M, unsigned N = 16, bool WS = false,
          bool TmemA = false, unsigned Flags = 0>
__global__ void dense_patterns(const unsigned char *input, float *output) {
  constexpr unsigned K = Kind < 2 ? 16 : 32;
  constexpr unsigned ARows = TmemA ? 128 : M, Alloc = (WS || TmemA) ? 256 : 32;
  constexpr unsigned Origin = Flags & kUpperDatapaths ? 16 : 0;
  static_assert(!TmemA || !(Flags & kTransposeA));
  static_assert(!Origin || (M == 64 && !WS));
  using Layouts = PatternLayouts<Kind, M, N, TmemA, Flags>;
  using A = typename Layouts::A;
  using B = typename Layouts::B;
  __shared__ __align__(1024) unsigned char a[A::Storage], b[B::Storage];
  __shared__ unsigned base;
  __shared__ unsigned long long bar;
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < A::Storage; i += 128) a[i] = input[i];
  for (unsigned i = t; i < B::Storage; i += 128) b[i] = input[A::Storage + i];
  const unsigned bp = (unsigned)__cvta_generic_to_shared(&bar);
  if (t == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bp) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base)),
        "n"(Alloc)
        : "memory");
  __syncthreads();
  if constexpr (TmemA) {
    // Populate every physical row, with distinct values in the two N halves.
    for (unsigned k = 0; k < 8; ++k) {
      const unsigned offset =
          (t / 8) * 256 + (t % 8) * 16 + (k / 4) * 128 + (k % 4) * 4;
      const unsigned word = *reinterpret_cast<const unsigned *>(a + offset);
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(
                       base + 128 + k),
                   "r"(word)
                   : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  const unsigned destination = base + (Origin << 16);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (t == 0) {
    const unsigned long long ad = (((unsigned)__cvta_generic_to_shared(a) +
                                    (TmemA ? 0 : Layouts::Prefix)) >>
                                   4) |
                                  (uint64_t(A::Lbo / 16) << 16) |
                                  (uint64_t(A::Sbo / 16) << 32) | (1ull << 46) |
                                  (uint64_t(TmemA ? 0 : Layouts::Mode) << 61);
    const unsigned long long bd =
        (((unsigned)__cvta_generic_to_shared(b) + Layouts::Prefix) >> 4) |
        (uint64_t(B::Lbo / 16) << 16) | (uint64_t(B::Sbo / 16) << 32) |
        (1ull << 46) | (uint64_t(Layouts::Mode) << 61);
    constexpr unsigned format = Kind % 2;
    const unsigned desc = ((M / 16) << 24) | ((N / 8) << 17) |
                          (Flags & kHalfOutput ? 0 : 0x10) | (format << 7) |
                          (format << 10) | (1u << 13) |
                          ((Flags & kTransposeA ? 1u : 0u) << 15) |
                          ((Flags & kTransposeB ? 1u : 0u) << 16) |
                          ((Flags & kNegateB ? 1u : 0u) << 14);
#define DENSE_MMA(KIND)                                                      \
  asm volatile(                                                              \
      "{ .reg .pred p; mov.pred p,0; tcgen05.mma.cta_group::1.kind::" KIND   \
      " [%0],%1,%2,%3,p; mov.pred p,1; tcgen05.mma.cta_group::1.kind::" KIND \
      " [%0],%1,%2,%3,p; }" ::"r"(destination),                              \
      "l"(ad), "l"(bd), "r"(desc)                                            \
      : "memory")
    if constexpr (Flags & kNegatedAccumulatorPredicate) {
      static_assert(Kind < 2 && !WS && !TmemA);
      asm volatile(
          "{ .reg .pred p; mov.pred p,1; "
          "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,!p; "
          "mov.pred p,0; "
          "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,!p; }"
          :: "r"(destination), "l"(ad), "l"(bd), "r"(desc) : "memory");
    } else if constexpr (WS) {
      if constexpr (TmemA) {
        asm volatile(
            "{ .reg .pred p; mov.pred p,0; "
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0],[%1],%2,%3,p; "
            "mov.pred p,1; "
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0],[%1],%2,%3,p; }" ::"r"(
                destination),
            "r"(base + 128 + (Origin << 16)), "l"(bd), "r"(desc)
            : "memory");
      } else {
        asm volatile(
            "{ .reg .pred p; mov.pred p,0; "
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0],%1,%2,%3,p; "
            "mov.pred p,1; "
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0],%1,%2,%3,p; }" ::"r"(
                destination),
            "l"(ad), "l"(bd), "r"(desc)
            : "memory");
      }
    } else if constexpr (TmemA) {
      asm volatile(
          "{ .reg .pred p; mov.pred p,0; "
          "tcgen05.mma.cta_group::1.kind::f16 [%0],[%1],%2,%3,p; "
          "mov.pred p,1; "
          "tcgen05.mma.cta_group::1.kind::f16 [%0],[%1],%2,%3,p; }" ::"r"(
              destination),
          "r"(base + 128 + (Origin << 16)), "l"(bd), "r"(desc)
          : "memory");
    } else if constexpr (Kind < 2) {
      DENSE_MMA("f16");
    } else {
      DENSE_MMA("f8f6f4");
    }
#undef DENSE_MMA
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(bp)
        : "memory");
  }
  asm volatile(
      "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
      "p,[%0],0; @!p bra wait; }" ::"r"(bp)
      : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned col = 0; col < (WS ? N / 2 : N); ++col) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                 : "=r"(value)
                 : "r"(base + col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    const float result = Flags & kHalfOutput
                             ? __half2float(__ushort_as_half(value & 0xffff))
                             : __uint_as_float(value);
    if constexpr (WS) {
      output[(blockIdx.x * M + t % 64) * N + (t / 64) * (N / 2) + col] = result;
    } else if (M == 128 || (t % 32 >= Origin && t % 32 < Origin + 16)) {
      const unsigned row = M == 128 ? t : (t / 32) * 16 + t % 32 - Origin;
      output[(blockIdx.x * M + row) * N + col] = result;
    }
  }
  __syncthreads();
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;" ::"r"(base),
        "n"(Alloc)
        : "memory");
  }
}
template <unsigned Kind, unsigned M, unsigned N = 16, bool WS = false,
          bool TmemA = false, unsigned Flags = 0>
void check_dense() {
  constexpr unsigned K = Kind < 2 ? 16 : 32, bytes = Kind < 2 ? 2 : 1;
  constexpr unsigned ARows = TmemA ? 128 : M;
  using Layouts = PatternLayouts<Kind, M, N, TmemA, Flags>;
  using A = typename Layouts::A;
  using B = typename Layouts::B;
  std::vector<unsigned char> input(A::Storage + B::Storage);
  auto fill = [&](unsigned rows, unsigned start, bool is_b) {
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned k = 0; k < K; ++k) {
        float value = is_b ? float(int((r + k) % 5) - 2)
                           : pattern_a(r, k, Flags & kHalfOutput);
        unsigned bits = 0;
        if constexpr (Kind == 0) {
          __half h = __float2half(value);
          std::memcpy(&bits, &h, 2);
        }
        if constexpr (Kind == 1) {
          std::memcpy(&bits, &value, 4);
          bits >>= 16;
        }
        if constexpr (Kind >= 2)
          bits = __nv_cvt_float_to_fp8(value, __NV_SATFINITE,
                                       Kind == 2 ? __NV_E4M3 : __NV_E5M2);
        const unsigned offset =
            start + (is_b ? B::offset(r, k) : A::offset(r, k));
        std::memcpy(input.data() + offset, &bits, bytes);
      }
  };
  fill(ARows, 0, false);
  fill(N, A::Storage, true);
  unsigned char *in = nullptr;
  float *out = nullptr;
  ASSERT_EQ(cudaMalloc(&in, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 2 * M * N * 4), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  dense_patterns<Kind, M, N, WS, TmemA, Flags><<<2, 128>>>(in, out);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> output(2 * M * N);
  ASSERT_EQ(
      cudaMemcpy(output.data(), out, output.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned r = 0; r < M; ++r)
      for (unsigned n = 0; n < N; ++n) {
        float expected = 0;
        const unsigned ar = !TmemA    ? r
                            : WS      ? r + (n / (N / 2)) * 64
                            : M == 64 ? (r / 16) * 32 + r % 16 +
                                            (Flags & kUpperDatapaths ? 16 : 0)
                                      : r;
        float dot = 0;
        for (unsigned k = 0; k < K; ++k)
          dot += (Flags & kNegateB ? 1 : -1) *
                 pattern_a(ar, k, Flags & kHalfOutput) *
                 float(int((n + k) % 5) - 2);
        expected = Flags & kHalfOutput
                       ? __half2float(__float2half(
                             __half2float(__float2half(dot)) + dot))
                       : 2 * dot;
        ASSERT_EQ(output[(cta * M + r) * N + n], expected)
            << Kind << ':' << r << ':' << n;
      }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(Tcgen05Mma, NegatedAccumulatorPredicate) {
  check_dense<0, 64, 16, false, false, kNegatedAccumulatorPredicate>();
  check_dense<1, 128, 16, false, false, kNegatedAccumulatorPredicate>();
}
TEST(Tcgen05Mma, DenseF16PatternsAndAccumulation) {
  check_dense<0, 64>();
  check_dense<0, 128>();
}
TEST(Tcgen05Mma, DenseBf16PatternsAndAccumulation) {
  check_dense<1, 64>();
  check_dense<1, 128>();
}
TEST(Tcgen05Mma, DenseE4M3PatternsAndAccumulation) { check_dense<2, 128>(); }
TEST(Tcgen05Mma, DenseE5M2PatternsAndAccumulation) { check_dense<3, 128>(); }
TEST(Tcgen05Mma, WsSharedF16ShapesAndAccumulation) {
  check_dense<0, 64, 64, true>();
  check_dense<0, 64, 128, true>();
  check_dense<0, 64, 256, true>();
}
TEST(Tcgen05Mma, WsTmemF16ShapesAndAccumulation) {
  check_dense<0, 64, 64, true, true>();
  check_dense<0, 64, 128, true, true>();
  check_dense<0, 64, 256, true, true>();
}
TEST(Tcgen05Mma, WsSharedBf16PatternsAndAccumulation) {
  check_dense<1, 64, 64, true>();
}
TEST(Tcgen05Mma, WsTmemBf16PatternsAndAccumulation) {
  check_dense<1, 64, 128, true, true>();
}
TEST(Tcgen05Mma, OrdinaryTmemF16LayoutsAndAccumulation) {
  check_dense<0, 64, 16, false, true>();
  check_dense<0, 64, 16, false, true, kUpperDatapaths>();
  check_dense<0, 128, 16, false, true>();
}
TEST(Tcgen05Mma, OrdinaryTmemBf16LayoutsAndAccumulation) {
  check_dense<1, 64, 16, false, true>();
  check_dense<1, 128, 16, false, true>();
}
TEST(Tcgen05Mma, F16OutputAndAccumulation) {
  check_dense<0, 64, 16, false, false, kHalfOutput>();
  check_dense<0, 128, 16, false, true, kHalfOutput>();
  check_dense<0, 64, 64, true, true, kHalfOutput>();
}
TEST(Tcgen05Mma, F16SharedTransposeCombinations) {
  check_dense<0, 64, 16, false, false, kTransposeA>();
  check_dense<0, 128, 16, false, false, kTransposeB>();
  check_dense<0, 64, 16, false, false, kTransposeA | kTransposeB | kNegateB>();
}
TEST(Tcgen05Mma, Bf16SharedTransposeCombinations) {
  check_dense<1, 128, 16, false, false, kTransposeA>();
  check_dense<1, 64, 16, false, false, kTransposeB>();
  check_dense<1, 128, 16, false, false, kTransposeA | kTransposeB | kNegateB>();
}
TEST(Tcgen05Mma, TmemATransposeBAndNegation) {
  check_dense<0, 128, 16, false, true, kTransposeB | kNegateB>();
  check_dense<1, 64, 64, true, true, kTransposeB>();
}
TEST(Tcgen05Mma, SwizzledKMajorSharedOperands) {
  check_dense<0, 64, 16, false, false, kSwizzle32>();
  check_dense<1, 128, 16, false, false, kSwizzle64>();
  check_dense<0, 128, 16, false, false, kSwizzle128>();
}
TEST(Tcgen05Mma, SwizzledMnMajorSharedOperands) {
  check_dense<0, 64, 16, false, false,
              kSwizzle32 | kTransposeA | kTransposeB>();
  check_dense<1, 128, 16, false, false,
              kSwizzle64 | kTransposeA | kTransposeB>();
  check_dense<0, 128, 16, false, false,
              kSwizzle128 | kTransposeA | kTransposeB>();
}
TEST(Tcgen05Mma, SwizzledNonPeriodAlignedStart) {
  check_dense<0, 64, 16, false, false, kSwizzle128 | kShiftBase>();
  check_dense<1, 128, 16, false, false,
              kSwizzle64 | kTransposeA | kTransposeB | kShiftBase>();
}
TEST(Tcgen05Mma, MnMajor128BWith32BAtoms) {
  check_dense<0, 64, 16, false, false,
              kSwizzle128Atom32 | kTransposeA | kTransposeB>();
  check_dense<1, 128, 16, false, false,
              kSwizzle128Atom32 | kTransposeA | kTransposeB | kShiftBase>();
}
TEST(Tcgen05Mma, SwizzledTmemAAndWs) {
  check_dense<0, 128, 16, false, true, kSwizzle128>();
  check_dense<1, 64, 64, true, false, kSwizzle64>();
  check_dense<0, 64, 64, true, true, kSwizzle128Atom32 | kTransposeB>();
}
}  // namespace
