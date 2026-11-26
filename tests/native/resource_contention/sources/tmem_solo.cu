#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tmem_ops.cuh"
#define CK(x)                                                 \
  do {                                                        \
    auto e = (x);                                             \
    if (e != cudaSuccess) {                                   \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); \
      exit(2);                                                \
    }                                                         \
  } while (0)
__device__ __forceinline__ void waitld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void waitst() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__host__ __device__ unsigned pattern(int tid, int col) {
  return (col < 256 ? 256u : 0u) | ((col % 256) << 10) | (unsigned(tid) << 20);
}
__device__ __forceinline__ void group_barrier(int warps);
// Modes: dependent address loads; independent burst loads; stores drained per
// instruction; stores drained per burst; same-thread store/wait/load roundtrip.
template <int W, int D, int MODE>
__global__ void tmem_probe(unsigned* out, unsigned long long* clocks, int n,
                           int warps) {
  __shared__ unsigned base;
  int tid = threadIdx.x, warp = tid / 32;
  if (warp == 0)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],512;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  if (warp < warps) {
    for (int c = 0; c < 512; c += 4) {
      unsigned v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        v[j] = MODE < 2 ? pattern(tid, c + j) : ~pattern(tid, c + j);
      tstore<4>(base + c, v);
    }
    waitst();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  __syncthreads();
  if (warp < warps) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    unsigned v[D][W], acc = 0, offset = 0;
#pragma unroll
    for (int d = 0; d < D; ++d) {
#pragma unroll
      for (int j = 0; j < W; ++j) v[d][j] = pattern(tid, d * W + j);
    }
    unsigned long long begin = clock64();
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
      if constexpr (MODE == 0) {
        tload<W>(base + offset, v[0]);
        waitld();
        offset = v[0][0] & 256u;
      }
      if constexpr (MODE == 1) {
#pragma unroll
        for (int d = 0; d < D; ++d) tload<W>(base + d * W, v[d]);
        waitld();
#pragma unroll
        for (int d = 0; d < D; ++d) acc += v[d][0];
      }
      if constexpr (MODE == 2 || MODE == 3) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          tstore<W>(base + d * W, v[d]);
          if constexpr (MODE == 2) waitst();
        }
        if constexpr (MODE == 3) waitst();
      }
      if constexpr (MODE == 4) {
        // The loaded value feeds the next store, forcing a real roundtrip.
        v[0][0] += 1;
        tstore<W>(base, v[0]);
        waitst();
        tload<W>(base, v[0]);
        waitld();
      }
    }
    // End depends on consumers through subsequent observable output. No in-loop
    // clocks.
    unsigned long long end = clock64();
    // Verification loads/stores must not overlap a slower timing worker.
    group_barrier(warps);
    if (tid % 32 == 0) clocks[warp] = end - begin;
    if constexpr (MODE == 2 || MODE == 3) {
#pragma unroll
      for (int d = 0; d < D; ++d) tload<W>(base + d * W, v[d]);
      waitld();
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
#pragma unroll
      for (int j = 0; j < W; ++j) out[tid * 129 + d * W + j] = v[d][j];
    }
    out[tid * 129 + 128] = MODE == 0 ? offset : acc;
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,512;" ::"r"(base)
        : "memory");
  }
}
// Immediate participant counts avoid the compiler's packed register-count BAR
// encoding; each variant still executes the same count as its active warp set.
__device__ __forceinline__ void group_barrier(int warps) {
  if (warps == 1)
    asm volatile("bar.sync 1,32;" ::: "memory");
  else if (warps == 2)
    asm volatile("bar.sync 1,64;" ::: "memory");
  else
    asm volatile("bar.sync 1,128;" ::: "memory");
}
// Shared modes: named barrier only; scalar dependent address load; vector load;
// vector store; same-thread store/load; pair exchange; CTA exchange.
// Runtime stride is in vector units, providing controlled lane bank conflicts.
template <int V, int MODE>
__global__ void shared_probe(unsigned* out, unsigned long long* clocks, int n,
                             int warps, int stride) {
  extern __shared__ unsigned sm[];
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  unsigned a[8], v[8][V];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int pos = warp * 4096 + i * 512 + lane * stride * V;
    a[i] = (unsigned)__cvta_generic_to_shared(sm + pos);
#pragma unroll
    for (int j = 0; j < V; ++j) {
      sm[pos + j] = MODE == 1
                        ? unsigned(__cvta_generic_to_shared(
                              sm + warp * 4096 + ((lane + 1) % 32) * stride))
                        : (MODE == 3 ? 0u : unsigned(tid * 1000 + i * V + j));
      v[i][j] = tid * 1000 + i * V + j;
    }
  }
  __syncthreads();
  if (warp < warps) {
    unsigned ptr = a[0], acc = 0;
    unsigned long long begin = clock64();
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
      if constexpr (MODE == 0) group_barrier(warps);
      if constexpr (MODE == 1)
        asm volatile("ld.shared.b32 %0,[%0];" : "+r"(ptr)::"memory");
      if constexpr (MODE == 7)
        asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
      if constexpr (MODE == 8)
        asm volatile("bar.sync %0,64;" ::"r"(1 + (warps == 4 ? (warp & 1) : 0))
                     : "memory");
      if constexpr (MODE >= 2 && MODE <= 6) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if constexpr (MODE >= 3) {
            unsigned dst = a[i];
            if constexpr (MODE >= 5)
              dst = (unsigned)__cvta_generic_to_shared(
                  sm + (warp ^ (warps == 4 ? 2 : 1)) * 4096 + i * 512 +
                  lane * stride * V);
            if constexpr (V == 1)
              asm volatile("st.shared.b32 [%0],%1;" ::"r"(dst), "r"(v[i][0])
                           : "memory");
            else
              asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};" ::"r"(dst),
                           "r"(v[i][0]), "r"(v[i][1]), "r"(v[i][2]),
                           "r"(v[i][3])
                           : "memory");
          }
        }
        if constexpr (MODE == 5)
          asm volatile(
              "bar.sync %0,64;" ::"r"(1 + (warps == 4 ? (warp & 1) : 0))
              : "memory");
        if constexpr (MODE == 6) group_barrier(warps);
        if constexpr (MODE != 3) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if constexpr (MODE == 2 || MODE == 4) {
              if constexpr (V == 1)
                asm volatile("ld.volatile.shared.b32 %0,[%1];"
                             : "=r"(v[i][0])
                             : "r"(a[i])
                             : "memory");
              else
                asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                             : "=r"(v[i][0]), "=r"(v[i][1]), "=r"(v[i][2]),
                               "=r"(v[i][3])
                             : "r"(a[i])
                             : "memory");
            } else {
              if constexpr (V == 1)
                asm volatile("ld.shared.b32 %0,[%1];"
                             : "=r"(v[i][0])
                             : "r"(a[i])
                             : "memory");
              else
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                             : "=r"(v[i][0]), "=r"(v[i][1]), "=r"(v[i][2]),
                               "=r"(v[i][3])
                             : "r"(a[i])
                             : "memory");
            }
            acc += v[i][0];
          }
        }
        if constexpr (MODE == 5)
          asm volatile(
              "bar.sync %0,64;" ::"r"(1 + (warps == 4 ? (warp & 1) : 0))
              : "memory");
        if constexpr (MODE == 6) group_barrier(warps);
      }
    }
    unsigned long long end = clock64();
    if (lane == 0) clocks[warp] = end - begin;
    if constexpr (MODE == 3) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < V; ++j)
          v[i][j] = sm[warp * 4096 + i * 512 + lane * stride * V + j];
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < V; ++j) out[tid * 129 + i * V + j] = v[i][j];
    }
    out[tid * 129 + 128] =
        MODE == 1 ? (ptr - (unsigned)__cvta_generic_to_shared(sm)) / 4 : acc;
  }
}
template <int W, int D>
void launch_t(int m, unsigned* o, unsigned long long* c, int n, int w) {
#define T(M)                                     \
  case M:                                        \
    tmem_probe<W, D, M><<<1, 128>>>(o, c, n, w); \
    break;
  switch (m) { T(0) T(1) T(2) T(3) T(4) }
#undef T
}
template <int W>
void dispatch_t(int d, int m, unsigned* o, unsigned long long* c, int n,
                int w) {
  if (d == 1) launch_t<W, 1>(m, o, c, n, w);
  if constexpr (W <= 64) {
    if (d == 2) launch_t<W, 2>(m, o, c, n, w);
  }
  if constexpr (W <= 32) {
    if (d == 4) launch_t<W, 4>(m, o, c, n, w);
  }
}
template <int V>
void launch_s(int m, unsigned* o, unsigned long long* c, int n, int w,
              int stride) {
#define S(M)                                                             \
  case M:                                                                \
    CK(cudaFuncSetAttribute(shared_probe<V, M>,                          \
                            cudaFuncAttributeMaxDynamicSharedMemorySize, \
                            65536));                                     \
    shared_probe<V, M><<<1, 128, 65536>>>(o, c, n, w, stride);           \
    break;
  switch (m) { S(0) S(1) S(2) S(3) S(4) S(5) S(6) S(7) S(8) }
#undef S
}
int main(int argc, char** argv) {
  if (argc != 9) {
    fprintf(stderr,
            "usage: probe family(0=tmem,1=shared) mode width depth warps "
            "stride iterations reps\n");
    return 2;
  }
  int family = atoi(argv[1]), m = atoi(argv[2]), width = atoi(argv[3]),
      depth = atoi(argv[4]), warps = atoi(argv[5]), stride = atoi(argv[6]),
      n = atoi(argv[7]), reps = atoi(argv[8]);
  if (n < 1 || reps < 1 || (warps != 1 && warps != 2 && warps != 4) ||
      (family != 0 && family != 1) || m < 0)
    return 2;
  if (family == 0 &&
      (m > 4 ||
       (width != 1 && width != 2 && width != 4 && width != 8 && width != 16 &&
        width != 32 && width != 64 && width != 128) ||
       (depth != 1 && depth != 2 && depth != 4) || width * depth > 128 ||
       ((m == 0 || m == 4) && depth != 1)))
    return 2;
  if (family == 1 &&
      (m > 8 || (width != 1 && width != 4) || depth != 1 ||
       (stride != 1 && stride != 2 && stride != 4) || (m == 1 && width != 1) ||
       ((m == 5 || m == 6 || m == 8) && warps == 1)))
    return 2;
  unsigned* o;
  unsigned long long* c;
  CK(cudaMalloc(&o, 128 * 129 * 4));
  CK(cudaMalloc(&c, 4 * 8));
  std::vector<unsigned> h(128 * 129);
  int bad = 0;
  for (int r = -1; r < reps; ++r) {
    CK(cudaMemset(o, 0xcd, h.size() * 4));
    if (family == 0) {
      switch (width) {
#define W(X)                                 \
  case X:                                    \
    dispatch_t<X>(depth, m, o, c, n, warps); \
    break;
        W(1) W(2) W(4) W(8) W(16) W(32) W(64) W(128)
#undef W
      }
    } else {
      if (width == 1)
        launch_s<1>(m, o, c, n, warps, stride);
      else
        launch_s<4>(m, o, c, n, warps, stride);
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, h.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long clocks[4];
    CK(cudaMemcpy(clocks, c, 32, cudaMemcpyDeviceToHost));
    for (int tid = 0; tid < warps * 32; ++tid) {
      int count = family == 0 ? width * depth : 8 * width;
      for (int j = 0; j < count; ++j) {
        unsigned expected;
        if (family == 0) {
          int col = j + (m == 0 && n % 2 == 0 ? 256 : 0);
          expected = pattern(tid, col);
          if (m == 4 && j == 0) expected += n;
        } else {
          int peer = tid;
          if ((m == 5 || m == 6) && n % 2)
            peer = tid ^ ((warps == 4 ? 2 : 1) * 32);
          expected = peer * 1000 + j;
        }
        if (!(family == 1 && m == 1) && h[tid * 129 + j] != expected) ++bad;
      }
      unsigned expected = 0;
      if (family == 0) {
        if (m == 0) expected = n % 2 ? 256 : 0;
        if (m == 1)
          for (int d = 0; d < depth; ++d)
            expected += unsigned(n) * pattern(tid, d * width);
      } else if (m == 1)
        expected = (tid / 32) * 4096 + ((tid % 32 + n) % 32) * stride;
      else if (m == 2 || (m >= 4 && m <= 6)) {
        for (int k = 0; k < n; ++k) {
          int peer = tid;
          if ((m == 5 || m == 6) && k % 2 == 0)
            peer = tid ^ ((warps == 4 ? 2 : 1) * 32);
          for (int i = 0; i < 8; ++i) expected += peer * 1000 + i * width;
        }
      }
      if (h[tid * 129 + 128] != expected) ++bad;
    }
    if (r >= 0) {
      printf("RESULT,%d,%d,%d,%d,%d,%d,%d,%d", family, m, width, depth, warps,
             stride, n, r);
      for (int w = 0; w < warps; ++w) printf(",%llu", clocks[w]);
      puts("");
    }
  }
  printf("CHECK,%s,bad=%d\n", bad ? "FAIL" : "PASS", bad);
  CK(cudaFree(o));
  CK(cudaFree(c));
  return bad ? 1 : 0;
}
