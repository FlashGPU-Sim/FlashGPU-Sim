// Matched MMA-only / memory-only / overlapping workers. No timed TMA.
#include "tmem_ops.cuh"
#ifndef TMEM_BACKGROUND_COLUMN
#define TMEM_BACKGROUND_COLUMN 384
#endif
static_assert(TMEM_BACKGROUND_COLUMN >= 288 && TMEM_BACKGROUND_COLUMN + 64 <= 512);
#ifndef LDS_LIGHT_CHECKSUM
#define LDS_LIGHT_CHECKSUM 0
#endif
#ifndef CONTROLLED_SHARED_REGIONS
#define CONTROLLED_SHARED_REGIONS 0
#endif
#ifndef SHARED_SCRATCH_UPPER
#define SHARED_SCRATCH_UPPER 0
#endif
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
using namespace cute;
#define CK(x)                                                 \
  do {                                                        \
    auto e = (x);                                             \
    if (e != cudaSuccess) {                                   \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); \
      exit(2);                                                \
    }                                                         \
  } while (0)
__host__ __device__ float av(unsigned r, unsigned k) {
  return float(int((r * 3 + k * 5) % 7) - 3) * 0.25f;
}
__host__ __device__ float bv(unsigned c, unsigned k) {
  return float(int((c * 5 + k * 3) % 7) - 3) * 0.25f;
}
struct Sample {
  unsigned long long start, end;
};
template <int N, bool WS, bool TS, unsigned Streams, unsigned Burst, int BG>
__global__ void mixed_probe(float* output, Sample* samples, unsigned rounds, unsigned ratio, unsigned mode,
                         unsigned* checks) {
  constexpr unsigned streams = Streams, burst = Burst;
  constexpr unsigned M = WS ? 64 : 128, PN = WS ? N / 2 : N;
  using AL = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
      Shape<Int<M>, _64>{}, Step<_1, _2>{}));
  using BAtom = std::conditional_t<(WS && !TS && N == 256),
      UMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>,
      UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>>;
  using BL = decltype(tile_to_shape(
      BAtom{},
      Shape<Int<N>, _64>{}, Step<_1, _2>{}));
#if CONTROLLED_SHARED_REGIONS
  extern __shared__ __align__(1024) unsigned char shared_data[];
  auto* a = reinterpret_cast<cutlass::bfloat16_t*>(shared_data);
  auto* b = reinterpret_cast<cutlass::bfloat16_t*>(shared_data + 16384);
#else
  __shared__ __align__(1024) cutlass::bfloat16_t a[M * 64], b[N * 64];
#endif
  auto at = make_tensor(make_smem_ptr(a), AL{});
  auto bt = make_tensor(make_smem_ptr(b), BL{});
  __shared__ unsigned base;
  __shared__ __align__(8) unsigned long long barrier;
#if CONTROLLED_SHARED_REGIONS
  auto* scratch = reinterpret_cast<unsigned*>(shared_data + 65536 + SHARED_SCRATCH_UPPER*131072);
#else
  __shared__ __align__(1024) unsigned scratch[4096];
#endif
  unsigned tid = threadIdx.x;
  for (unsigned i = tid; i < 4096; i += 256)
    scratch[i] = BG == 1 ? ~(0x10000u + i) : 0x10000u + i;
  for (unsigned i = tid; i < M * 64; i += 256)
    at(i / 64, i % 64) =
        cutlass::bfloat16_t(i % 64 < 16 ? av(i / 64, i % 64) : 0.f);
  for (unsigned i = tid; i < N * 64; i += 256)
    bt(i / 64, i % 64) =
        cutlass::bfloat16_t(i % 64 < 16 ? bv(i / 64, i % 64) : 0.f);
  unsigned bar = __cvta_generic_to_shared(&barrier);
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  if (tid < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],512;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  if constexpr (TS) {
    if (tid < 128) {
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
      unsigned v[4];
      for (unsigned j = 0; j < 4; ++j) {
        unsigned k = chunk * 8 + j * 2;
        v[j] = __bfloat16_as_ushort(__float2bfloat16(av(tid, k))) |
               (unsigned(__bfloat16_as_ushort(__float2bfloat16(av(tid, k + 1))))
                << 16);
      }
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};" ::"r"(
              base + chunk * 4),
          "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
          : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
  }
  if (tid < 128) {
    unsigned init[32];
#pragma unroll
    for (unsigned j = 0; j < 32; ++j) init[j] = 0x3f800000u;
    for (unsigned col = 0; col < PN * streams; col += 32)
      tstore<32>(base + 32 + col, init);
#pragma unroll
    for (unsigned j = 0; j < 32; ++j) init[j] = BG == 3 ? ~(0x20000u + tid * 64 + j) : 0x20000u + tid * 64 + j;
    tstore<32>(base + TMEM_BACKGROUND_COLUMN, init);
#pragma unroll
    for (unsigned j = 0; j < 32; ++j)
      init[j] = BG == 3 ? ~(0x20000u + tid * 64 + j + 32) : 0x20000u + tid * 64 + j + 32;
    tstore<32>(base + TMEM_BACKGROUND_COLUMN + 32, init);
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint64_t da = UMMA::make_umma_desc<UMMA::Major::K>(at),
           db = UMMA::make_umma_desc<(WS && !TS && N == 256) ? UMMA::Major::MN : UMMA::Major::K>(bt);
  unsigned idesc = ((M / 16) << 24) | ((N / 8) << 17) | 0x490 | ((WS && !TS && N == 256) << 16);
  // Cache shared state before the asm memory clobbers; elect once so ptxas
  // can retain uniform operands across the whole MMA burst.
  const unsigned tbase = base;
  // Both workers observe the same CTA start barrier. Each uses only its own
  // operands during timing. Global reporting and validation occur after clocks.
  unsigned v[64], checksum = 0;
  unsigned long long worker_start = 0, worker_end = 0;
  __syncthreads();
  // A volatile shared consumer actually waits for a deferred BAR to complete;
  // independent CS2R/STG instructions alone need not do so.
  unsigned begin_gate = *(volatile unsigned*)&base;
  if (begin_gate != tbase) return;
  if ((mode & 1) && tid < 32 && cute::elect_one_sync()) {
    unsigned long long start = clock64();
#pragma unroll 1
    for (unsigned round = 0; round < rounds; ++round) {
#pragma unroll
      for (unsigned op = 0; op < burst; ++op) {
        unsigned stream = op & (streams - 1), d = tbase + 32 + stream * PN,
                 acc = op >= streams;
        if constexpr (WS && TS)
          asm volatile(
              "{.reg .pred p;setp.ne.u32 "
              "p,%4,0;tcgen05.mma.ws.cta_group::1.kind::f16 "
              "[%0],[%1],%2,%3,p;}" ::"r"(d),
              "r"(tbase), "l"(db), "r"(idesc), "r"(acc)
              : "memory");
        else if constexpr (WS)
          asm volatile(
              "{.reg .pred p;setp.ne.u32 "
              "p,%4,0;tcgen05.mma.ws.cta_group::1.kind::f16 "
              "[%0],%1,%2,%3,p;}" ::"r"(d),
              "l"(da), "l"(db), "r"(idesc), "r"(acc)
              : "memory");
        else
          asm volatile(
              "{.reg .pred p;setp.ne.u32 "
              "p,%4,0;tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}" ::
                  "r"(d),
              "l"(da), "l"(db), "r"(idesc), "r"(acc)
              : "memory");
      }
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
          "b64 [%0];" ::"r"(bar)
          : "memory");
      asm volatile(
          "{.reg .pred p;wait_done: mbarrier.try_wait.parity.shared::cta.b64 "
          "p,[%0],%1;@!p bra wait_done;}" ::"r"(bar),
          "r"(round & 1)
          : "memory");
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    unsigned long long end = clock64();
    worker_start = start; worker_end = end;
  }
  if (tid >= 128 && (mode & 2)) {
    unsigned worker = tid - 128, warp = worker / 32, lane = worker % 32;
#pragma unroll
    for (unsigned j = 0; j < 64; ++j)
      v[j] = 0x20000u + worker * 64 + j;
    unsigned long long start = clock64();
#pragma unroll 1
    for (unsigned it = 0; it < rounds * ratio; ++it) {
      if constexpr (BG == 0 || BG == 1) {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          unsigned idx = warp * 1024 + j * 128 + lane * 4;
          unsigned addr = __cvta_generic_to_shared(scratch + idx);
          if constexpr (BG == 0) {
            asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                : "=r"(v[j*4]),"=r"(v[j*4+1]),"=r"(v[j*4+2]),"=r"(v[j*4+3]) : "r"(addr) : "memory");
            if constexpr (!LDS_LIGHT_CHECKSUM) checksum += v[j*4];
          } else {
            unsigned x = 0x10000u + idx;
            asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"
                :: "r"(addr),"r"(x),"r"(x+1),"r"(x+2),"r"(x+3) : "memory");
          }
        }
        if constexpr (BG == 0 && LDS_LIGHT_CHECKSUM) checksum += v[0];
      } else if constexpr (BG == 2) {
        tload<32>(tbase + TMEM_BACKGROUND_COLUMN, v);
        tload<32>(tbase + TMEM_BACKGROUND_COLUMN + 32, v + 32);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        checksum += v[0] + v[32];
      } else if constexpr (BG == 3) {
        tstore<32>(tbase + TMEM_BACKGROUND_COLUMN, v);
        tstore<32>(tbase + TMEM_BACKGROUND_COLUMN + 32, v + 32);
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      } else {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
          asm volatile("add.u32 %0,%0,%1;" : "+r"(checksum) : "r"(worker + 1));
      }
    }
    // Keep every load destination live without adding reductions inside the loop.
    unsigned long long end = clock64();
    worker_start = start; worker_end = end;
  }
  __syncthreads();
  unsigned end_gate = *(volatile unsigned*)&base;
  if (end_gate != tbase) return;
  // All workers have stopped their clocks before any report/verification I/O.
  if (tid == 0) {
    checks[128+128*64] = __cvta_generic_to_shared(a);
    checks[128+128*64+1] = __cvta_generic_to_shared(b);
    checks[128+128*64+2] = __cvta_generic_to_shared(scratch);
  }
  if (tid == 0 && (mode & 1)) samples[0] = {worker_start, worker_end};
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (tid >= 128 && (mode & 2)) {
    unsigned worker = tid - 128;
    if ((worker % 32) == 0) samples[1 + worker/32] = {worker_start, worker_end};
    checks[worker] = checksum;
#pragma unroll
    for (unsigned j = 0; j < (BG < 2 ? 32 : 64); ++j)
      checks[128 + worker * 64 + j] = v[j];
  }
  if (tid >= 128 && (mode & 2)) {
    unsigned worker = tid - 128;
    if constexpr (BG == 1) {
      for (unsigned j = 0; j < 32; ++j)
        checks[128 + worker * 64 + j] =
            scratch[(worker / 32)*1024 + (j/4)*128 + (worker%32)*4 + j%4];
    } else if constexpr (BG == 3) {
      unsigned v[64];
      tload<32>(base + TMEM_BACKGROUND_COLUMN, v); tload<32>(base + TMEM_BACKGROUND_COLUMN + 32, v+32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      for (unsigned j = 0; j < 64; ++j) checks[128 + worker*64+j] = v[j];
    }
  }
  if (tid < 128) {
  for (unsigned stream = 0; stream < streams; ++stream)
    for (unsigned col = 0; col < PN; col += 4) {
      float v[4];
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
                   : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                   : "r"(base + 32 + stream * PN + col));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      for (unsigned j = 0; j < 4; ++j)
        output[(stream * 128 + tid) * PN + col + j] = v[j];
    }
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,512;" ::"r"(base)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
}
template <int N, bool WS, bool TS, unsigned Streams, unsigned Burst, int BG>
int run(unsigned rounds, unsigned ratio, unsigned mode) {
  constexpr unsigned streams = Streams, burst = Burst;
  constexpr unsigned PN = WS ? N / 2 : N;
  std::vector<float> out(streams * 128 * PN);
  std::vector<Sample> samples(5);
  std::vector<unsigned> checks(128 + 128*64 + 3);
  unsigned* dc;
  CK(cudaMalloc(&dc, checks.size()*4));
  CK(cudaMemset(dc, 0, checks.size()*4));
  float* d;
  Sample* s;
  CK(cudaMalloc(&d, out.size() * sizeof(float)));
  CK(cudaMalloc(&s, 5 * sizeof(Sample)));
  CK(cudaMemset(s, 0, 5 * sizeof(Sample)));
  CK(cudaFuncSetAttribute(mixed_probe<N, WS, TS, Streams, Burst, BG>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, CONTROLLED_SHARED_REGIONS ? 224*1024 : 1024));
  mixed_probe<N, WS, TS, Streams, Burst, BG><<<1, 256, CONTROLLED_SHARED_REGIONS ? 224*1024 : 0>>>(d, s, rounds, ratio, mode, dc);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), d, out.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(samples.data(), s, 5 * sizeof(Sample),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(checks.data(), dc, checks.size()*4, cudaMemcpyDeviceToHost));
  unsigned errors = 0;
  for (unsigned stream = 0; stream < streams; ++stream)
    for (unsigned row = 0; row < 128; ++row)
      for (unsigned col = 0; col < PN; ++col) {
        unsigned ar = (WS && !TS) ? row % 64 : row,
                 bc = WS ? (row / 64) * PN + col : col;
        float expected = 0;
        for (unsigned k = 0; k < 16; ++k) expected += av(ar, k) * bv(bc, k);
        expected *= (burst + streams - 1 - stream) / streams;
        if (!(mode & 1)) expected = 1;
        float got = out[(stream * 128 + row) * PN + col];
        if (got != expected) {
          if (errors < 4)
            fprintf(stderr,
                    "mismatch stream=%u row=%u col=%u got=%g expected=%g\n",
                    stream, row, col, got, expected);
          ++errors;
        }
      }
  if (mode & 2) {
    for (unsigned w = 0; w < 128; ++w) {
      unsigned expected_checksum = 0;
      if constexpr (BG == 0) {
        if constexpr (LDS_LIGHT_CHECKSUM)
          expected_checksum = 0x10000u + (w/32)*1024 + (w%32)*4;
        else
          for (unsigned j = 0; j < 8; ++j)
            expected_checksum += 0x10000u + (w/32)*1024 + j*128 + (w%32)*4;
      }
      if constexpr (BG == 2)
        expected_checksum = 2*(0x20000u+w*64)+32;
      if constexpr (BG == 4) expected_checksum = 8*(w+1);
      expected_checksum *= rounds * ratio;
      if (checks[w] != expected_checksum) ++errors;
      for (unsigned j = 0; j < (BG < 2 ? 32 : 64); ++j) {
        unsigned expected = BG < 2 ? 0x10000u + (w/32)*1024 + (j/4)*128 + (w%32)*4 + j%4 : 0x20000u+w*64+j;
        if (checks[128+w*64+j] != expected) ++errors;
      }
    }
  }
  printf("ADDRESS,a=%u,b=%u,scratch=%u\n",checks[128+128*64],checks[128+128*64+1],checks[128+128*64+2]);
  for (unsigned i = 0; i < 5; ++i)
    printf("CLOCK,%u,%llu,%llu\n",i,samples[i].start,samples[i].end);
  printf("RESULT,ws=%u,ts=%u,n=%u,bg=%u,mode=%u,rounds=%u,ratio=%u,mmas=%u,bg_bytes=%llu,verification=%s,errors=%u\n",
      WS, TS, N, BG, mode, rounds, ratio, mode&1 ? rounds*burst : 0,
      (mode&2) && BG<4 ? (unsigned long long)rounds*ratio*(BG<2 ? 16384 : 32768) : 0,
      errors ? "FAIL" : "PASS", errors);
  CK(cudaFree(dc));
  CK(cudaFree(d));
  CK(cudaFree(s));
  return errors ? 1 : 0;
}

template<int BG>
int dispatch(unsigned shape, unsigned rounds, unsigned ratio, unsigned mode) {
  if (shape == 0) return run<128,true,true,1,32,BG>(rounds,ratio,mode);
  if (shape == 1) return run<256,true,false,1,32,BG>(rounds,ratio,mode);
  return run<128,false,false,1,32,BG>(rounds,ratio,mode);
}
int main(int argc,char** argv) {
  if (argc != 6) return 2;
  printf("CONFIG lds_light_checksum=%d,regions=%d,upper=%d\n", LDS_LIGHT_CHECKSUM,CONTROLLED_SHARED_REGIONS,SHARED_SCRATCH_UPPER);
  printf("TMEM background_column=%u\n", unsigned(TMEM_BACKGROUND_COLUMN));
  unsigned shape=atoi(argv[1]), bg=atoi(argv[2]), mode=atoi(argv[3]), rounds=atoi(argv[4]), ratio=atoi(argv[5]);
  if(shape>2 || bg>4 || mode<1 || mode>3 || !rounds || rounds>512 || !ratio || ratio>128) return 2;
  switch(bg) {
    case 0:return dispatch<0>(shape,rounds,ratio,mode);
    case 1:return dispatch<1>(shape,rounds,ratio,mode);
    case 2:return dispatch<2>(shape,rounds,ratio,mode);
    case 3:return dispatch<3>(shape,rounds,ratio,mode);
    case 4:return dispatch<4>(shape,rounds,ratio,mode);
  }
  return 2;
}
