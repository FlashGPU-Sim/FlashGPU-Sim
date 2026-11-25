#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "mma_test_layout.h"

namespace {
// Check physical output-lane masks with runtime values so ptxas cannot erase
// the explicit mask operand. Seed D independently of enable-input-d.
template <unsigned Kind, unsigned M, bool TmemA, unsigned Mode = 0>
__global__ void masked_mma(const unsigned char *input, float *output,
                           unsigned m0, unsigned m1, unsigned m2, unsigned m3) {
  constexpr unsigned K = Kind == 2 ? 8 : 16, Bytes = Kind == 2 ? 4 : 2;
  using A = MmaTestLayout<Mode, false, TmemA ? 128 : M, K, Bytes>;
  using B = MmaTestLayout<Mode, false, 16, K, Bytes>;
  __shared__ __align__(1024) unsigned char a[A::Storage], b[B::Storage];
  __shared__ unsigned base;
  __shared__ unsigned long long barrier;
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < A::Storage; i += 128) a[i] = input[i];
  for (unsigned i = t; i < B::Storage; i += 128) b[i] = input[A::Storage + i];
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (!t)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],64;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  for (unsigned col = 0; col < 16; ++col)
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(base + col),
        "r"(__float_as_uint(7.0f))
        : "memory");
  if constexpr (TmemA)
    for (unsigned word = 0; word < 8; ++word) {
      unsigned bits =
          *reinterpret_cast<unsigned *>(a + A::offset(t, word * 4 / Bytes));
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(
                       base + 32 + word),
                   "r"(bits)
                   : "memory");
    }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (!t) {
    const unsigned long long ad = ((unsigned)__cvta_generic_to_shared(a) >> 4) |
                                  (uint64_t(A::Lbo / 16) << 16) |
                                  (uint64_t(A::Sbo / 16) << 32) | (1ull << 46) |
                                  (uint64_t(Mode) << 61);
    const unsigned long long bd = ((unsigned)__cvta_generic_to_shared(b) >> 4) |
                                  (uint64_t(B::Lbo / 16) << 16) |
                                  (uint64_t(B::Sbo / 16) << 32) | (1ull << 46) |
                                  (uint64_t(Mode) << 61);
    const unsigned desc =
        ((M / 16) << 24) | (2u << 17) | 0x10 | (Kind << 7) | (Kind << 10);
#define MASKED(KIND, AOPERAND, CONSTRAINT, AVALUE)                           \
  asm volatile(                                                              \
      "{ .reg .pred p; mov.pred p,0; "                                       \
      "tcgen05.mma.cta_group::1.kind::" KIND " [%0]," AOPERAND               \
      ",%2,%3,{%4,%5,%6,%7},p; "                                             \
      "mov.pred p,1; tcgen05.mma.cta_group::1.kind::" KIND " [%0]," AOPERAND \
      ",%2,%3,{%4,%5,%6,%7},p; }" ::"r"(base),                               \
      CONSTRAINT(AVALUE), "l"(bd), "r"(desc), "r"(m0), "r"(m1), "r"(m2),     \
      "r"(m3)                                                                \
      : "memory")
    if constexpr (Kind == 2) {
      if constexpr (TmemA) {
        MASKED("tf32", "[%1]", "r", base + 32);
      } else {
        MASKED("tf32", "%1", "l", ad);
      }
    } else {
      if constexpr (TmemA) {
        MASKED("f16", "[%1]", "r", base + 32);
      } else {
        MASKED("f16", "%1", "l", ad);
      }
    }
#undef MASKED
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(bar)
        : "memory");
  }
  asm volatile(
      "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
      "p,[%0],0; @!p bra wait; }" ::"r"(bar)
      : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned col = 0; col < 16; ++col) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                 : "=r"(value)
                 : "r"(base + col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    output[t * 16 + col] = __uint_as_float(value);
  }
  __syncthreads();
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,64;" ::"r"(base)
        : "memory");
  }
}

template <unsigned Kind, unsigned M, bool TmemA, unsigned Mode = 0>
void check_mask(unsigned m0, unsigned m1, unsigned m2, unsigned m3) {
  constexpr unsigned K = Kind == 2 ? 8 : 16, Bytes = Kind == 2 ? 4 : 2;
  using A = MmaTestLayout<Mode, false, TmemA ? 128 : M, K, Bytes>;
  using B = MmaTestLayout<Mode, false, 16, K, Bytes>;
  std::vector<unsigned char> input(A::Storage + B::Storage);
  auto fill = [&](unsigned rows, bool is_b) {
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned k = 0; k < K; ++k) {
        float f = is_b ? float(int((row + k) % 5) - 2)
                       : float((row % 7 + 1) * (k % 3 + 1));
        unsigned bits;
        std::memcpy(&bits, &f, 4);
        if constexpr (Kind == 0) bits = __half_as_ushort(__float2half(f));
        if constexpr (Kind == 1) bits >>= 16;
        auto offset = is_b ? A::Storage + B::offset(row, k) : A::offset(row, k);
        std::memcpy(input.data() + offset, &bits, Bytes);
      }
  };
  fill(TmemA ? 128 : M, false);
  fill(16, true);
  unsigned char *in;
  float *out;
  ASSERT_EQ(cudaMalloc(&in, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 128 * 16 * 4), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  masked_mma<Kind, M, TmemA, Mode><<<1, 128>>>(in, out, m0, m1, m2, m3);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> actual(128 * 16);
  ASSERT_EQ(
      cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost),
      cudaSuccess);
  const unsigned masks[] = {m0, m1, m2, m3};
  for (unsigned lane = 0; lane < 128; ++lane)
    for (unsigned col = 0; col < 16; ++col) {
      float expected = 7;
      if ((M == 128 || lane % 32 < 16) &&
          !(masks[lane / 32] & (1u << (lane % 32)))) {
        const unsigned row =
            TmemA || M == 128 ? lane : (lane / 32) * 16 + lane % 32;
        expected = 0;
        for (unsigned k = 0; k < K; ++k)
          expected += 2 * float((row % 7 + 1) * (k % 3 + 1)) *
                      float(int((col + k) % 5) - 2);
      }
      ASSERT_EQ(actual[lane * 16 + col], expected) << lane << ':' << col;
    }
  ASSERT_EQ(cudaFree(in), cudaSuccess);
  ASSERT_EQ(cudaFree(out), cudaSuccess);
}
TEST(Tcgen05Mma, OutputMaskF16PhysicalLanes) {
  check_mask<0, 64, false>(0x80000001, 0x00008002, 0xffffffff, 0);
  check_mask<0, 128, true>(0x80000001, 0x00008002, 0xffffffff, 0);
}
TEST(Tcgen05Mma, OutputMaskBf16PhysicalLanes) {
  check_mask<1, 128, false>(0x80000001, 0x00008002, 0xffffffff, 0);
  check_mask<1, 64, true>(0x80000001, 0x00008002, 0xffffffff, 0);
}
TEST(Tcgen05Mma, Tf32SharedAndMaskedTmem) {
  check_mask<2, 64, false>(0, 0, 0, 0);
  check_mask<2, 128, false>(0x80000001, 0x00008002, 0xffffffff, 0);
  check_mask<2, 64, true>(0x80000001, 0x00008002, 0xffffffff, 0);
}
TEST(Tcgen05Mma, Tf32SharedSwizzles) {
  check_mask<2, 128, false, 6>(0, 0, 0, 0);
  check_mask<2, 64, false, 4>(0x80000001, 0x00008002, 0xffffffff, 0);
  check_mask<2, 128, false, 2>(0, 0, 0, 0);
}
}  // namespace
