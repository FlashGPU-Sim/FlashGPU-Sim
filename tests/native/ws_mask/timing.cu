// B200 WS collector/shift timing. Setup, verification and stores are untimed.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/tensor.hpp>
#include <vector>
using namespace cute;
#define CK(expr)                                                 \
  do {                                                           \
    auto e = (expr);                                             \
    if (e != cudaSuccess) {                                      \
      fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); \
      exit(2);                                                   \
    }                                                            \
  } while (0)
__host__ __device__ float av(unsigned r, unsigned k) {
  return (int((r * 3 + k * 5) % 7) - 3) * 0.25f;
}
__host__ __device__ float bv(unsigned c, unsigned k) {
  return (int((c * 5 + k * 3) % 7) - 3) * 0.25f;
}
struct Sample {
  unsigned long long issue, complete;
};
// M64/N128/K16 BF16, K-major. Mode 0 discard, 1 fill, 2 use,
// 3 alternating fill/lastuse. All modes have the same untimed warm fill.
template <unsigned MaxShift, unsigned Mode, unsigned Burst, bool TmemA>
__global__ void ws_mask_timing(float *out, Sample *samples, unsigned rounds,
                               unsigned warmup, unsigned long long mask) {
  constexpr unsigned N = 128, Rows = MaxShift ? 192 : 128;
  using AL =
      decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                             Shape<_64, _64>{}, Step<_1, _2>{}));
  using BL =
      decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                             Shape<Int<Rows>, _64>{}, Step<_1, _2>{}));
  __shared__ __align__(1024) cutlass::bfloat16_t a[64 * 64], b[Rows * 64];
  __shared__ unsigned base;
  __shared__ __align__(8) unsigned long long barrier;
  unsigned tid = threadIdx.x;
  auto at = make_tensor(make_smem_ptr(a), AL{});
  auto bt = make_tensor(make_smem_ptr(b), BL{});
  for (unsigned i = tid; i < 64 * 64; i += 128)
    at(i / 64, i % 64) =
        cutlass::bfloat16_t(i % 64 < 16 ? av(i / 64, i % 64) : 0.f);
  for (unsigned i = tid; i < Rows * 64; i += 128)
    bt(i / 64, i % 64) =
        cutlass::bfloat16_t(i % 64 < 16 ? bv(i / 64, i % 64) : 0.f);
  const unsigned bar = __cvta_generic_to_shared(&barrier);
  if (!tid)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (tid < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],256;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  const unsigned tbase = base;
  if constexpr (TmemA) {
    for (unsigned k = 0; k < 8; ++k) {
      const unsigned w =
          __bfloat16_as_ushort(__float2bfloat16(av(tid, 2 * k))) |
          (unsigned(__bfloat16_as_ushort(__float2bfloat16(av(tid, 2 * k + 1))))
           << 16);
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(tbase + k),
          "r"(w)
          : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint64_t ad = UMMA::make_umma_desc<UMMA::Major::K>(at),
                 bd = UMMA::make_umma_desc<UMMA::Major::K>(bt);
  constexpr unsigned shiftcode = MaxShift == 8    ? 1
                                 : MaxShift == 16 ? 2
                                 : MaxShift == 32 ? 3
                                                  : 0;
  const unsigned desc = (4u << 24) | (16u << 17) | 0x490 | (shiftcode << 30),
                 dst = tbase + 32;
#define MMA(OP, ACC)                                                      \
  if constexpr (TmemA) {                                                  \
    asm volatile(                                                         \
        "{.reg .pred p;setp.ne.u32 "                                      \
        "p,%4,0;tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::" OP \
        " [%0],[%1],%2,%3,p,%5;}" ::"r"(dst),                             \
        "r"(tbase), "l"(bd), "r"(desc), "r"(unsigned(ACC)), "l"(mask)     \
        : "memory");                                                      \
  } else {                                                                \
    asm volatile(                                                         \
        "{.reg .pred p;setp.ne.u32 "                                      \
        "p,%4,0;tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::" OP \
        " [%0],%1,%2,%3,p,%5;}" ::"r"(dst),                               \
        "l"(ad), "l"(bd), "r"(desc), "r"(unsigned(ACC)), "l"(mask)        \
        : "memory");                                                      \
  }
#define COMPLETE()                                                             \
  asm volatile(                                                                \
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(bar) \
      : "memory");                                                             \
  asm volatile(                                                                \
      "{.reg .pred p;wait_done: mbarrier.try_wait.parity.shared::cta.b64 "     \
      "p,[%0],%1;@!p bra wait_done;}" ::"r"(bar),                              \
      "r"(phase)                                                               \
      : "memory");                                                             \
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");              \
  phase ^= 1;
  if (tid < 32 && cute::elect_one_sync()) {
    unsigned phase = 0;
    for (unsigned round = 0; round < rounds + warmup; ++round) {
      MMA("fill", 0)
      COMPLETE()
      unsigned long long start = clock64();
#pragma unroll
      for (unsigned i = 0; i < Burst; ++i) {
        if constexpr (Mode == 0) {
          MMA("discard", i != 0)
        }
        if constexpr (Mode == 1) {
          MMA("fill", i != 0)
        }
        if constexpr (Mode == 2) {
          MMA("use", i != 0)
        }
        if constexpr (Mode == 3) {
          if (i % 2 == 0) {
            MMA("fill", i != 0)
          } else {
            MMA("lastuse", true)
          }
        }
      }
      unsigned long long issued = clock64();
      COMPLETE()
      unsigned long long done = clock64();
      if (round >= warmup)
        samples[round - warmup] = {issued - start, done - start};
    }
  }
#undef COMPLETE
#undef MMA
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (unsigned c = 0; c < N / 2; c += 4) {
    float v[4];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                 : "r"(dst + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    for (unsigned j = 0; j < 4; ++j) out[tid * (N / 2) + c + j] = v[j];
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,256;" ::"r"(tbase)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
}
// Independent host mask oracle consumes spans explicitly.
bool zero_column(uint64_t mask, unsigned c) {
  if (!(mask & (1ull << 39))) return false;
  unsigned segment = c / 64, left = ((mask >> (segment * 8)) & 255) + c % 64;
  bool zero = (mask >> (32 + segment)) & 1;
  for (;;) {
    unsigned size = ((mask >> (zero ? 40 : 48)) & 255) + 1;
    if (left < size) return zero;
    left -= size;
    zero = !zero;
  }
}
template <unsigned Max, unsigned Mode, unsigned Burst, bool TS>
int run(unsigned shift, unsigned pattern, unsigned rounds, unsigned warmup) {
  uint64_t mask = uint64_t(shift) << 56;
  if (pattern == 1)
    mask |= (1ull << 39) | (2ull << 40) | (3ull << 48) | (1ull << 32) | 1 |
            (2ull << 8);
  if (pattern == 2) mask |= (1ull << 39) | (255ull << 40) | (3ull << 32);
  float *d;
  Sample *s;
  CK(cudaMalloc(&d, 8192 * sizeof(float)));
  CK(cudaMalloc(&s, rounds * sizeof(Sample)));
  ws_mask_timing<Max, Mode, Burst, TS><<<1, 128>>>(d, s, rounds, warmup, mask);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> out(8192);
  std::vector<Sample> samples(rounds);
  CK(cudaMemcpy(out.data(), d, 8192 * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(samples.data(), s, rounds * sizeof(Sample),
                cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  for (unsigned row = 0; row < 128; ++row)
    for (unsigned c = 0; c < 64; ++c) {
      const unsigned col = c + (row / 64) * 64;
      float expected = 0;
      if (!zero_column(mask, col))
        for (unsigned k = 0; k < 16; ++k)
          expected += av(TS ? row : row % 64, k) * bv(col + shift, k);
      expected *= Burst ? Burst : 1;
      if (out[row * 64 + c] != expected) {
        if (errors < 4)
          fprintf(stderr, "mismatch row=%u col=%u got=%g expected=%g\n", row,
                  col, out[row * 64 + c], expected);
        ++errors;
      }
    }
  for (unsigned i = 0; i < rounds; ++i)
    printf("SAMPLE,%u,%llu,%llu\n", i, samples[i].issue, samples[i].complete);
  printf(
      "RESULT max_shift=%u shift=%u mode=%u mask=%u tmem_a=%u burst=%u "
      "verification=%s errors=%u\n",
      Max, shift, Mode, pattern, TS, Burst, errors ? "FAIL" : "PASS", errors);
  CK(cudaFree(d));
  CK(cudaFree(s));
  return errors ? 1 : 0;
}
template <unsigned Max, unsigned Mode, bool TS>
int burst_dispatch(unsigned burst, unsigned shift, unsigned mask,
                   unsigned rounds, unsigned warmup) {
#define B(X) \
  case X:    \
    return run<Max, Mode, X, TS>(shift, mask, rounds, warmup);
  switch (burst) { B(0) B(1) B(32) B(128) default : return 2; }
#undef B
}
template <unsigned Max, bool TS>
int mode_dispatch(unsigned mode, unsigned burst, unsigned shift, unsigned mask,
                  unsigned rounds, unsigned warmup) {
#define M(X) \
  case X:    \
    return burst_dispatch<Max, X, TS>(burst, shift, mask, rounds, warmup);
  switch (mode) { M(0) M(1) M(2) M(3) default : return 2; }
#undef M
}
int main(int argc, char **argv) {
  // Positional CLI keeps simulator replay and native capture commands
  // identical.
  if (argc != 8) {
    fprintf(stderr, "usage: %s MAX_SHIFT SHIFT MODE MASK TMEM_A BURST ROUNDS\n",
            argv[0]);
    return 2;
  }
  unsigned v[7];
  for (unsigned i = 0; i < 7; ++i) {
    char *end;
    long n = strtol(argv[i + 1], &end, 10);
    if (*end || n < 0 || n > 65536) return 2;
    v[i] = n;
  }
  if (v[1] > v[0] || v[3] > 2 || v[4] > 1 || !v[6] || v[6] > 1024) return 2;
#define S(X)                                                          \
  case X:                                                             \
    if (v[4])                                                         \
      return mode_dispatch<X, true>(v[2], v[5], v[1], v[3], v[6], 4); \
    else                                                              \
      return mode_dispatch<X, false>(v[2], v[5], v[1], v[3], v[6], 4);
  switch (v[0]) { S(0) S(8) S(16) S(32) default : return 2; }
#undef S
}
