#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
namespace {
template <unsigned Kind, unsigned M>
__global__ void dense_patterns(const unsigned char *input, float *output) {
  constexpr unsigned N = 16, K = Kind < 2 ? 16 : 32;
  __shared__ __align__(128) unsigned char a[M * 32], b[N * 32];
  __shared__ unsigned base;
  __shared__ unsigned long long bar;
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < M * 32; i += 128) a[i] = input[i];
  for (unsigned i = t; i < N * 32; i += 128) b[i] = input[M * 32 + i];
  const unsigned bp = (unsigned)__cvta_generic_to_shared(&bar);
  if (t == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bp) : "memory");
  if (t < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],32;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (t == 0) {
    const unsigned long long ad = ((unsigned)__cvta_generic_to_shared(a) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    const unsigned long long bd = ((unsigned)__cvta_generic_to_shared(b) >> 4) |
                                  (8ull << 16) | (16ull << 32) | (1ull << 46);
    constexpr unsigned format = Kind % 2;
    const unsigned desc = ((M / 16) << 24) | ((N / 8) << 17) | 0x10 |
                          (format << 7) | (format << 10) | (1u << 13);
#define DENSE_MMA(KIND)                                                      \
  asm volatile(                                                              \
      "{ .reg .pred p; mov.pred p,0; tcgen05.mma.cta_group::1.kind::" KIND   \
      " [%0],%1,%2,%3,p; mov.pred p,1; tcgen05.mma.cta_group::1.kind::" KIND \
      " [%0],%1,%2,%3,p; }" ::"r"(base),                                     \
      "l"(ad), "l"(bd), "r"(desc)                                            \
      : "memory")
    if constexpr (Kind < 2) {
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
  for (unsigned col = 0; col < N; ++col) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                 : "=r"(value)
                 : "r"(base + col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (M == 128 || t % 32 < 16) {
      const unsigned row = M == 128 ? t : (t / 32) * 16 + t % 32;
      output[(blockIdx.x * M + row) * N + col] = __uint_as_float(value);
    }
  }
  __syncthreads();
  if (t < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,32;" ::"r"(base)
        : "memory");
  }
}
template <unsigned Kind, unsigned M>
void check_dense() {
  constexpr unsigned N = 16, K = Kind < 2 ? 16 : 32, bytes = Kind < 2 ? 2 : 1;
  std::vector<unsigned char> input((M + N) * 32);
  auto fill = [&](unsigned rows, unsigned start, bool is_b) {
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned k = 0; k < K; ++k) {
        float value = is_b ? float(int((r + k) % 5) - 2)
                           : float((r % 3 + 1) * (k % 2 + 1));
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
        unsigned offset = start + (r / 8) * 256 + (r % 8) * 16 +
                          (k * bytes / 16) * 128 + (k * bytes % 16);
        std::memcpy(input.data() + offset, &bits, bytes);
      }
  };
  fill(M, 0, false);
  fill(N, M * 32, true);
  unsigned char *in = nullptr;
  float *out = nullptr;
  ASSERT_EQ(cudaMalloc(&in, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 2 * M * N * 4), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), input.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  dense_patterns<Kind, M><<<2, 128>>>(in, out);
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
        for (unsigned k = 0; k < K; ++k)
          expected -= 2 * float((r % 3 + 1) * (k % 2 + 1)) *
                      float(int((n + k) % 5) - 2);
        ASSERT_EQ(output[(cta * M + r) * N + n], expected)
            << Kind << ':' << r << ':' << n;
      }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
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
}  // namespace
