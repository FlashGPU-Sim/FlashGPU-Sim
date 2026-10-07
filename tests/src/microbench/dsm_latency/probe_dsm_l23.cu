// FlashGPU-Sim TODO §2.1–2.4 DSM microbenchmarks (single-cluster BW, dep RTT,
// store visibility, all-pairs contention). Called from run_dsm_probes().

#include "common.cuh"

#include <cooperative_groups.h>

#include <vector>

namespace cg = cooperative_groups;

// Forward from probe_dsm.cu helpers — reimplement minimal launch here.
namespace {

constexpr int kMaxCluster = 32;
constexpr int kChaseSteps = 64;
constexpr int kSmemElems = 256;
constexpr int kBwItersSc = 128;

cudaError_t cfg_cluster(const void *fn, int dyn_smem) {
  cudaError_t e =
      cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (e != cudaSuccess) return e;
  if (dyn_smem > 0) {
    e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             dyn_smem);
    if (e != cudaSuccess) return e;
  }
  return cudaFuncSetAttribute(
      fn, cudaFuncAttributeClusterSchedulingPolicyPreference,
      (int)cudaClusterSchedulingPolicySpread);
}

cudaError_t launch_1cluster(const void *fn, void **args, int cs, int block_dim,
                            int dyn_smem) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cs, 1, 1);
  cfg.blockDim = dim3(block_dim, 1, 1);
  cfg.dynamicSmemBytes = static_cast<size_t>(dyn_smem);
  cfg.stream = 0;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cs;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return cudaLaunchKernelExC(&cfg, fn, args);
}

// ---------------------------------------------------------------------------
// 2.1 Single-cluster BW: pattern 0=ring 1=pair 2=bcast
// payload_bytes transferred per CTA per iter (capped by smem).
// cycles_out[0] = clock64 span on rank 0 (optional).
// ---------------------------------------------------------------------------
__global__ void dsm_bw_sc_kernel(int pattern, int iters, int payload_bytes,
                                 uint32_t *sink, uint64_t *cycles_out) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  extern __shared__ __align__(16) uint8_t smem_bytes[];
  const int nwords = payload_bytes / 4;
  for (int i = threadIdx.x; i < nwords; i += blockDim.x) {
    reinterpret_cast<uint32_t *>(smem_bytes)[i] =
        static_cast<uint32_t>(rank * 131u + static_cast<uint32_t>(i));
  }
  cluster.sync();

  int src_rank = 0;
  if (pattern == 0)
    src_rank = (rank + 1) % n;
  else if (pattern == 1) {
    src_rank = rank ^ 1;
    if (src_rank >= n) src_rank = rank;
  } else
    src_rank = 0;

  const uint32_t *remote = reinterpret_cast<const uint32_t *>(
      cluster.map_shared_rank(smem_bytes, src_rank));
  uint32_t acc = 0;

  if (threadIdx.x == 0 && rank == 0 && cycles_out) {
    // barrier so all ranks start together
  }
  cluster.sync();
  const uint64_t t0 = (threadIdx.x == 0 && rank == 0) ? clock64_now() : 0;

#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    for (int i = threadIdx.x; i < nwords; i += blockDim.x) {
      acc += remote[i];
    }
    cluster.sync();
  }

  if (threadIdx.x == 0 && rank == 0 && cycles_out) {
    *cycles_out = clock64_now() - t0;
  }

  __shared__ uint32_t red;
  if (threadIdx.x == 0) red = 0;
  __syncthreads();
  atomicAdd(&red, acc);
  __syncthreads();
  if (threadIdx.x == 0) sink[blockIdx.x] = red;
#else
  if (threadIdx.x == 0) sink[blockIdx.x] = 0;
#endif
}

// ---------------------------------------------------------------------------
// 2.2 Dependent remote load RTT — pointer chase with volatile/asm loads
// ---------------------------------------------------------------------------
__global__ void dsm_dep_load_kernel(uint64_t *out_tot, int n_samples,
                                    int cluster_n) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  if (n != cluster_n) return;

  __shared__ int smem[kSmemElems];
  // Identity cycle chain: smem[i] = (i+1) % N so chase is data-dependent.
  for (int i = threadIdx.x; i < kSmemElems; i += blockDim.x)
    smem[i] = (i + 1) % kSmemElems;
  cluster.sync();

  // Each rank times load to peer (r+1)%n serially is done by only measuring
  // rank 0 → rank 1 for a clean single-link dep RTT; also all off-diag mean.
  if (threadIdx.x != 0) return;

  for (int s = 0; s < n_samples; ++s) {
    // Measure rank r → (r+1)%n for each r; store at [s*n + r]
    for (int r = 0; r < n; ++r) {
      if (rank != r) {
        cluster.sync();
        continue;
      }
      const int dst = (r + 1) % n;
      volatile int *remote =
          cluster.map_shared_rank(reinterpret_cast<int *>(smem), dst);
      int idx = 0;
      // Warm: data-dependent chase
      for (int i = 0; i < kChaseSteps; ++i) idx = remote[idx % kSmemElems];
      asm volatile("" ::: "memory");
      const uint64_t t0 = clock64_now();
#pragma unroll 1
      for (int i = 0; i < kChaseSteps; ++i) {
        idx = remote[idx % kSmemElems];
      }
      const uint64_t t1 = clock64_now();
      out_tot[static_cast<size_t>(s) * n + r] = t1 - t0;
      if (idx == 0x7fffffff) out_tot[0] = 0;
      cluster.sync();
    }
  }
#else
  (void)out_tot;
  (void)n_samples;
  (void)cluster_n;
#endif
}

// ---------------------------------------------------------------------------
// 2.3 Store to peer visible
// ---------------------------------------------------------------------------
__global__ void dsm_store_vis_kernel(uint64_t *out_ns, uint64_t *out_prod_cy,
                                     int n_samples) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  if (n < 2) return;

  __shared__ volatile uint32_t flag;
  __shared__ volatile uint32_t payload;
  __shared__ uint64_t cons_ns;

  if (threadIdx.x == 0) {
    flag = 0;
    payload = 0;
    cons_ns = 0;
  }
  cluster.sync();

  for (int s = 0; s < n_samples; ++s) {
    if (threadIdx.x == 0) {
      flag = 0;
      payload = 0;
      cons_ns = 0;
    }
    cluster.sync();

    uint64_t producer_start_ns = 0;
    if (rank == 0 && threadIdx.x == 0) {
      volatile uint32_t *rflag =
          cluster.map_shared_rank(const_cast<uint32_t *>(&flag), 1);
      volatile uint32_t *rpay =
          cluster.map_shared_rank(const_cast<uint32_t *>(&payload), 1);
      const uint64_t t0 = clock64_now();
      producer_start_ns = globaltimer_now();
      *rpay = 0xA5A5A5A5u;
      // Make store visible to peer (cluster scope)
#if __CUDA_ARCH__ >= 900
      asm volatile("fence.sc.cluster;" ::: "memory");
#endif
      *rflag = 1u;
#if __CUDA_ARCH__ >= 900
      asm volatile("fence.sc.cluster;" ::: "memory");
#endif
      const uint64_t t1 = clock64_now();
      out_prod_cy[s] = t1 - t0;
    } else if (rank == 1 && threadIdx.x == 0) {
      while (flag == 0) {
        asm volatile("" ::: "memory");
      }
      cons_ns = (payload == 0xA5A5A5A5u) ? globaltimer_now() : 0xffffffffull;
    }
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
      uint64_t *remote = cluster.map_shared_rank(&cons_ns, 1);
      const uint64_t visible_ns = *remote;
      out_ns[s] = (visible_ns != 0xffffffffull &&
                   visible_ns >= producer_start_ns)
                      ? (visible_ns - producer_start_ns)
                      : 0xffffffffull;
    }
    cluster.sync();
  }
#else
  (void)out_ns;
  (void)out_prod_cy;
  (void)n_samples;
#endif
}

// ---------------------------------------------------------------------------
// 2.4 Contention: mode 0 = only ranks 0,1; mode 1 = all ranks simultaneous
// ---------------------------------------------------------------------------
__global__ void dsm_contention_kernel(int mode, uint64_t *out_tot, int n_samples,
                                      int cluster_n) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  if (n != cluster_n) return;

  __shared__ int smem[kSmemElems];
  for (int i = threadIdx.x; i < kSmemElems; i += blockDim.x)
    smem[i] = (i + 1) % kSmemElems;
  cluster.sync();

  const int peer = (mode == 0) ? (rank < 2 ? (rank ^ 1) : rank) : (rank ^ 1) % n;
  // For all-pairs ring use (rank+1)%n when mode==2; mode 1 = xor pair
  const int dst = (mode == 2) ? (rank + 1) % n : peer;

  if (threadIdx.x != 0) return;

  volatile int *remote =
      cluster.map_shared_rank(reinterpret_cast<int *>(smem), dst);

  for (int s = 0; s < n_samples; ++s) {
    cluster.sync();
    // Idle ranks in 1-pair mode still sync but don't time
    const bool active = (mode != 0) || (rank < 2);
    int idx = 0;
    if (active) {
      for (int i = 0; i < 8; ++i) idx = remote[idx % kSmemElems];
    }
    cluster.sync();
    uint64_t dt = 0;
    if (active) {
      idx = 0;
      const uint64_t t0 = clock64_now();
#pragma unroll 1
      for (int i = 0; i < kChaseSteps; ++i) {
        idx = remote[idx % kSmemElems];
      }
      dt = clock64_now() - t0;
      if (idx == 0x7fffffff) dt = 0;
    }
    out_tot[static_cast<size_t>(s) * n + rank] = dt;
    cluster.sync();
  }
#else
  (void)mode;
  (void)out_tot;
  (void)n_samples;
  (void)cluster_n;
#endif
}

double median_vec(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return percentile_sorted(v, 0.5);
}

// ---------------------------------------------------------------------------
// Isolated 2-SM peak BW: only ranks (rank_a, rank_b) copy; others idle.
// bi=0: only rank_a reads rank_b (unidirectional).
// bi=1: both ranks read each other (bidirectional).
// No cluster.sync inside the copy loop (needed to saturate the link).
// ---------------------------------------------------------------------------
constexpr int kBw1SmIters = 256;

__global__ void dsm_bw_1sm_kernel(int rank_a, int rank_b, int bi, int iters,
                                  int payload_bytes, uint32_t *smids,
                                  uint32_t *sink, uint64_t *cycles_out) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  extern __shared__ __align__(16) uint8_t smem_bytes[];

  const int nvec = payload_bytes / 16;  // float4 / 16 B
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    reinterpret_cast<uint4 *>(smem_bytes)[i] =
        make_uint4(static_cast<uint32_t>(rank + i), 1u, 2u, 3u);
  }
  if (threadIdx.x == 0) {
    uint32_t id = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    smids[rank] = id;
  }
  cluster.sync();

  const bool is_a = (rank == rank_a);
  const bool is_b = (rank == rank_b);
  const bool active = is_a || (bi && is_b);
  const int src = is_a ? rank_b : rank_a;
  const uint4 *remote = reinterpret_cast<const uint4 *>(
      cluster.map_shared_rank(smem_bytes, src));

  uint32_t acc = 0;
  cluster.sync();
  const uint64_t t0 =
      (threadIdx.x == 0 && is_a && cycles_out) ? clock64_now() : 0;

  if (active) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        const uint4 v = remote[i];
        acc += v.x + v.y + v.z + v.w;
      }
    }
  }

  if (threadIdx.x == 0 && is_a && cycles_out) {
    *cycles_out = clock64_now() - t0;
  }
  cluster.sync();

  __shared__ uint32_t red;
  if (threadIdx.x == 0) red = 0;
  __syncthreads();
  atomicAdd(&red, acc);
  __syncthreads();
  if (threadIdx.x == 0) sink[blockIdx.x] = red;
#else
  if (threadIdx.x == 0) {
    smids[rank_a] = 0;
    sink[blockIdx.x] = 0;
  }
  (void)rank_b;
  (void)bi;
  (void)iters;
  (void)payload_bytes;
  (void)cycles_out;
#endif
}

struct PairBw {
  uint32_t sm_a = 0, sm_b = 0;
  int rank_a = 0, rank_b = 1;
  int order_fwd = -1;
  int d_smid = 0;
  int src = 0;  // 0 = mesh in one cluster, 1 = extra cluster=2 placement
  double uni_gbs = 0, bi_gbs = 0, per_sm_gbs = 0;
  double uni_bpc = 0, bi_bpc = 0, per_sm_bpc = 0;
  double uni_gbs_rev = 0, bi_gbs_rev = 0;
  double uni_bpc_rev = 0, bi_bpc_rev = 0;
  bool ok = false;
  bool ok_rev = false;
};

double mean_dbl(const std::vector<double> &v) {
  if (v.empty()) return 0;
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

bool time_1sm_pair(int cs, int rank_a, int rank_b, int bi, int payload,
                   int block_dim, int iters, uint32_t *d_smids, uint32_t *d_sink,
                   uint64_t *d_cyc, uint32_t *h_smids, double clock64_oh,
                   double *gbs_out, double *bpc_out) {
  int ra = rank_a, rb = rank_b, biv = bi, it = iters, pay = payload;
  void *args[] = {&ra, &rb, &biv, &it, &pay, &d_smids, &d_sink, &d_cyc};
  auto launch = [&]() -> bool {
    cudaError_t e = launch_1cluster((const void *)dsm_bw_1sm_kernel, args, cs,
                                    block_dim, payload);
    if (e != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return true;
  };
  auto warmup = [&]() -> bool {
    if (!launch() || cudaDeviceSynchronize() != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return true;
  };
  if (!warmup()) return false;
  if (!warmup()) return false;

  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  cudaEventRecord(start);
  if (!launch()) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return false;
  }
  cudaEventRecord(stop);
  if (cudaEventSynchronize(stop) != cudaSuccess) {
    (void)cudaGetLastError();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return false;
  }
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  CUDA_CHECK(cudaMemcpy(h_smids, d_smids, sizeof(uint32_t) * cs,
                        cudaMemcpyDeviceToHost));
  uint64_t hcy = 0;
  CUDA_CHECK(
      cudaMemcpy(&hcy, d_cyc, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Bytes *read* by one SM (same for uni and bi per reader).
  const double bytes_one = static_cast<double>(payload) * iters;
  const double bytes_tot = bi ? (2.0 * bytes_one) : bytes_one;
  const double sec = ms * 1e-3;
  if (sec <= 0) return false;
  *gbs_out = (bytes_tot / sec) / (1024.0 * 1024.0 * 1024.0);

  double adj = static_cast<double>(hcy);
  if (adj > clock64_oh) adj -= clock64_oh;
  // BPC for *one SM's* read volume (payload*iters / cycles)
  *bpc_out = (adj > 0) ? (bytes_one / adj) : 0;
  return true;
}

void run_dsm_1sm_peak(const SuiteOptions &opt, MetricSink &sink,
                      int max_cluster) {
  std::printf(
      "\n  --- 2.8 isolated 2-SM peak DSM BW (only one pair active) ---\n");
  std::printf(
      "  note: uni = rank_a reads rank_b; bi = both read each other; "
      "no per-iter cluster.sync\n");
  std::printf(
      "  scale: all unordered distinct-SM rank pairs in one cluster, "
      "then extra cluster=2 placements, then reverse-order remesh\n");

  (void)cfg_cluster((const void *)dsm_bw_1sm_kernel, 32 * 1024);

  int cs = std::min(max_cluster, 16);
  if (cs < 2) cs = 2;

  uint32_t *d_smids = nullptr, *d_sink = nullptr;
  uint64_t *d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_smids, sizeof(uint32_t) * kMaxCluster));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t) * kMaxCluster));
  CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(uint64_t)));
  std::vector<uint32_t> h_smids(kMaxCluster, 0xffffffffu);

  // Discover largest cluster that actually launches
  double dummy_g = 0, dummy_b = 0;
  bool placed = false;
  for (int try_cs = cs; try_cs >= 2; --try_cs) {
    if (time_1sm_pair(try_cs, 0, 1, /*bi=*/0, 8192, 256, kBw1SmIters, d_smids,
                      d_sink, d_cyc, h_smids.data(), opt.clock64_overhead,
                      &dummy_g, &dummy_b)) {
      cs = try_cs;
      placed = true;
      break;
    }
  }
  if (!placed) {
    std::printf("  SKIP 1sm peak: cluster launch failed\n");
    CUDA_CHECK(cudaFree(d_smids));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_cyc));
    return;
  }

  std::printf("  cluster=%d  rank_to_smid:", cs);
  int unique = 0;
  {
    std::vector<uint32_t> seen;
    for (int r = 0; r < cs; ++r) {
      std::printf("  r%d→sm%u", r, h_smids[r]);
      if (std::find(seen.begin(), seen.end(), h_smids[r]) == seen.end()) {
        seen.push_back(h_smids[r]);
        ++unique;
      }
    }
  }
  std::printf("  unique_sms=%d\n", unique);

  // Sweep payload/block on first valid distinct-SM pair to pick saturating cfg
  int best_pay = 16384, best_bs = 256;
  double best_cfg_gbs = 0;
  const int pays[] = {8192, 16384, 32768};
  const int bss[] = {256, 512};
  int sat_ra = 0, sat_rb = 1;
  for (int rb = 1; rb < cs; ++rb) {
    if (h_smids[0] != h_smids[rb]) {
      sat_rb = rb;
      break;
    }
  }
  for (int pay : pays) {
    for (int bs : bss) {
      double g = 0, bpc = 0;
      if (!time_1sm_pair(cs, sat_ra, sat_rb, 1, pay, bs, kBw1SmIters, d_smids,
                         d_sink, d_cyc, h_smids.data(), opt.clock64_overhead,
                         &g, &bpc))
        continue;
      if (h_smids[sat_ra] == h_smids[sat_rb]) continue;
      if (g > best_cfg_gbs) {
        best_cfg_gbs = g;
        best_pay = pay;
        best_bs = bs;
      }
    }
  }
  std::printf(
      "  sat_cfg: payload=%d block=%d  (best bi-agg %.2f GB/s on r%d-r%d)\n",
      best_pay, best_bs, best_cfg_gbs, sat_ra, sat_rb);

  // Clock/thermal settle so the first mesh pairs are not the coldest.
  for (int w = 0; w < 4; ++w) {
    double g = 0, bpc = 0;
    (void)time_1sm_pair(cs, sat_ra, sat_rb, 1, best_pay, best_bs, kBw1SmIters,
                        d_smids, d_sink, d_cyc, h_smids.data(),
                        opt.clock64_overhead, &g, &bpc);
  }

  std::vector<PairBw> pairs;
  auto already = [&](uint32_t a, uint32_t b) {
    for (const auto &q : pairs) {
      if ((q.sm_a == a && q.sm_b == b) || (q.sm_a == b && q.sm_b == a))
        return true;
    }
    return false;
  };

  auto measure_pair = [&](int launch_cs, int ra, int rb, int src) -> bool {
    if (ra >= launch_cs || rb >= launch_cs || ra == rb) return false;
    PairBw p;
    p.rank_a = ra;
    p.rank_b = rb;
    p.src = src;
    double g = 0, bpc = 0;
    if (!time_1sm_pair(launch_cs, ra, rb, /*uni*/ 0, best_pay, best_bs,
                       kBw1SmIters, d_smids, d_sink, d_cyc, h_smids.data(),
                       opt.clock64_overhead, &g, &bpc))
      return false;
    p.sm_a = h_smids[ra];
    p.sm_b = h_smids[rb];
    if (p.sm_a == p.sm_b) {
      std::printf("  skip r%d-r%d: same SM sm%u (not inter-SM DSM)\n", ra, rb,
                  p.sm_a);
      return false;
    }
    if (already(p.sm_a, p.sm_b)) return false;
    p.d_smid = static_cast<int>(p.sm_a > p.sm_b ? p.sm_a - p.sm_b
                                                : p.sm_b - p.sm_a);
    p.uni_gbs = g;
    p.uni_bpc = bpc;
    if (!time_1sm_pair(launch_cs, ra, rb, /*bi*/ 1, best_pay, best_bs,
                       kBw1SmIters, d_smids, d_sink, d_cyc, h_smids.data(),
                       opt.clock64_overhead, &g, &bpc))
      return false;
    p.bi_gbs = g;
    p.per_sm_gbs = g / 2.0;
    p.bi_bpc = bpc;
    p.per_sm_bpc = bpc;
    p.ok = true;
    p.order_fwd = static_cast<int>(pairs.size());
    pairs.push_back(p);
    return true;
  };

  // Full mesh: every unordered rank pair with distinct SMIDs.
  for (int ra = 0; ra < cs; ++ra) {
    for (int rb = ra + 1; rb < cs; ++rb) {
      if (h_smids[ra] == h_smids[rb]) continue;
      (void)measure_pair(cs, ra, rb, /*src=*/0);
    }
  }
  const int n_mesh = static_cast<int>(pairs.size());

  // Extra cluster=2 launches to sample SM pairs outside this cluster.
  const int kCs2Attempts = 32;
  const int kCs2MaxNew = 16;
  int cs2_new = 0;
  for (int attempt = 0; attempt < kCs2Attempts && cs2_new < kCs2MaxNew;
       ++attempt) {
    if (measure_pair(2, 0, 1, /*src=*/1)) ++cs2_new;
  }

  std::printf(
      "  measured %d mesh pairs (cluster=%d) + %d extra cs=2 pairs  "
      "(total %zu)\n",
      n_mesh, cs, cs2_new, pairs.size());

  // Reverse pass: remesh only (same cluster / ranks ⇒ same SMs). Distinguishes
  // warmup (BW follows test order) from topology (BW follows |Δsmid|).
  int n_rev = 0;
  int n_rev_smid_shift = 0;
  for (int i = n_mesh - 1; i >= 0; --i) {
    PairBw &p = pairs[static_cast<size_t>(i)];
    double g = 0, bpc = 0;
    if (!time_1sm_pair(cs, p.rank_a, p.rank_b, /*uni*/ 0, best_pay, best_bs,
                       kBw1SmIters, d_smids, d_sink, d_cyc, h_smids.data(),
                       opt.clock64_overhead, &g, &bpc))
      continue;
    if (h_smids[p.rank_a] != p.sm_a || h_smids[p.rank_b] != p.sm_b) {
      ++n_rev_smid_shift;
      continue;
    }
    p.uni_gbs_rev = g;
    p.uni_bpc_rev = bpc;
    if (!time_1sm_pair(cs, p.rank_a, p.rank_b, /*bi*/ 1, best_pay, best_bs,
                       kBw1SmIters, d_smids, d_sink, d_cyc, h_smids.data(),
                       opt.clock64_overhead, &g, &bpc))
      continue;
    p.bi_gbs_rev = g;
    p.bi_bpc_rev = bpc;
    p.ok_rev = true;
    ++n_rev;
  }
  if (n_rev_smid_shift)
    std::printf("  note: reverse skipped %d pairs (SMID remapped)\n",
                n_rev_smid_shift);

  auto print_row = [](int ord, const PairBw &p, bool rev) {
    const double uni = rev ? p.uni_gbs_rev : p.uni_gbs;
    const double bi = rev ? p.bi_gbs_rev : p.bi_gbs;
    const double bpc = rev ? p.uni_bpc_rev : p.uni_bpc;
    const double per = rev ? (p.bi_gbs_rev * 0.5) : p.per_sm_gbs;
    std::printf("  %3d  %c  r%-2d r%-2d  sm%-4u sm%-4u  %5d  %8.2f %8.2f %8.2f %8.3f\n",
                ord, p.src ? '2' : 'M', p.rank_a, p.rank_b, p.sm_a, p.sm_b,
                p.d_smid, uni, bi, per, bpc);
  };

  std::printf(
      "  --- forward (measurement order)  M=mesh  2=cs2 extra ---\n");
  std::printf("  %3s  %s  %-3s %-3s  %-6s %-6s  %5s  %8s %8s %8s %8s\n",
              "ord", "S", "ra", "rb", "sm_a", "sm_b", "|dSM|", "uni_GB/s",
              "bi_GB/s", "perSM", "uni_BPC");
  for (const auto &p : pairs) {
    if (p.ok) print_row(p.order_fwd, p, /*rev=*/false);
  }

  std::printf("  --- reverse remesh (same pairs, opposite order) ---\n");
  std::printf("  %3s  %s  %-3s %-3s  %-6s %-6s  %5s  %8s %8s %8s %8s\n",
              "ord", "S", "ra", "rb", "sm_a", "sm_b", "|dSM|", "uni_GB/s",
              "bi_GB/s", "perSM", "uni_BPC");
  int rev_ord = 0;
  for (int i = n_mesh - 1; i >= 0; --i) {
    const PairBw &p = pairs[static_cast<size_t>(i)];
    if (!p.ok_rev) continue;
    print_row(rev_ord++, p, /*rev=*/true);
  }

  std::vector<double> per_sm, uni_fwd, uni_rev_in_rev_order;
  double peak_per = 0, peak_uni_bpc = 0, peak_bi_bpc = 0;
  for (const auto &p : pairs) {
    if (!p.ok) continue;
    per_sm.push_back(p.per_sm_gbs);
    uni_fwd.push_back(p.uni_gbs);
    peak_per = std::max(peak_per, p.per_sm_gbs);
    peak_uni_bpc = std::max(peak_uni_bpc, p.uni_bpc);
    peak_bi_bpc = std::max(peak_bi_bpc, p.per_sm_bpc);

    char name[80], notes[192];
    std::snprintf(notes, sizeof(notes),
                  "n_clusters=1; pair=sm%u-sm%u; |dSM|=%d; src=%s; payload=%d; "
                  "block=%d; iters=%d; freq_mhz=%.1f",
                  p.sm_a, p.sm_b, p.d_smid, p.src ? "cs2" : "mesh", best_pay,
                  best_bs, kBw1SmIters, opt.sm_clock_mhz);
    std::snprintf(name, sizeof(name), "dsm_bw_1sm_uni_sm%u_sm%u", p.sm_a,
                  p.sm_b);
    sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", p.uni_gbs, p.uni_gbs,
             p.uni_gbs, "GB/s", notes);
    std::snprintf(name, sizeof(name), "dsm_bw_1sm_bi_sm%u_sm%u", p.sm_a,
                  p.sm_b);
    sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", p.bi_gbs, p.bi_gbs,
             p.bi_gbs, "GB/s", notes);
    std::snprintf(name, sizeof(name), "dsm_bw_1sm_per_sm_sm%u_sm%u", p.sm_a,
                  p.sm_b);
    sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", p.per_sm_gbs,
             p.per_sm_gbs, p.per_sm_gbs, "GB/s", notes);
  }
  for (int i = n_mesh - 1; i >= 0; --i) {
    if (pairs[static_cast<size_t>(i)].ok_rev)
      uni_rev_in_rev_order.push_back(pairs[static_cast<size_t>(i)].uni_gbs_rev);
  }

  const double mean_per = mean_dbl(per_sm);
  double var = 0;
  for (double x : per_sm) var += (x - mean_per) * (x - mean_per);
  if (!per_sm.empty()) var /= static_cast<double>(per_sm.size());
  const double cv = (mean_per > 0) ? (std::sqrt(var) / mean_per) : 0;

  auto half_ratio = [](const std::vector<double> &v) -> double {
    if (v.size() < 4) return 0;
    const size_t mid = v.size() / 2;
    std::vector<double> a(v.begin(), v.begin() + static_cast<long>(mid));
    std::vector<double> b(v.begin() + static_cast<long>(mid), v.end());
    const double ma = mean_dbl(a), mb = mean_dbl(b);
    return (ma > 0) ? (mb / ma) : 0;
  };
  const double order_ratio_fwd = half_ratio(uni_fwd);
  const double order_ratio_rev = half_ratio(uni_rev_in_rev_order);

  std::vector<double> near_uni, far_uni, near_uni_rev, far_uni_rev;
  {
    std::vector<int> ds;
    for (const auto &p : pairs)
      if (p.ok) ds.push_back(p.d_smid);
    std::sort(ds.begin(), ds.end());
    const int med_d = ds.empty() ? 0 : ds[ds.size() / 2];
    for (const auto &p : pairs) {
      if (!p.ok) continue;
      if (p.d_smid <= med_d)
        near_uni.push_back(p.uni_gbs);
      else
        far_uni.push_back(p.uni_gbs);
      if (!p.ok_rev) continue;
      if (p.d_smid <= med_d)
        near_uni_rev.push_back(p.uni_gbs_rev);
      else
        far_uni_rev.push_back(p.uni_gbs_rev);
    }
  }
  const double near_m = mean_dbl(near_uni);
  const double far_m = mean_dbl(far_uni);
  const double dsmid_ratio = (near_m > 0) ? (far_m / near_m) : 0;
  const double near_m_rev = mean_dbl(near_uni_rev);
  const double far_m_rev = mean_dbl(far_uni_rev);
  const double dsmid_ratio_rev =
      (near_m_rev > 0) ? (far_m_rev / near_m_rev) : 0;

  double rel_fwd_rev = 0;
  int n_rel = 0;
  for (const auto &p : pairs) {
    if (!p.ok || !p.ok_rev || p.uni_gbs <= 0) continue;
    rel_fwd_rev += std::fabs(p.uni_gbs_rev - p.uni_gbs) / p.uni_gbs;
    ++n_rel;
  }
  if (n_rel) rel_fwd_rev /= n_rel;

  // Width from peak uni BPC (one reader, no return traffic) — primary.
  const double mhz = opt.sm_clock_mhz > 0 ? opt.sm_clock_mhz : 1785.0;
  const double bpc_from_gbs =
      (mhz > 0) ? (peak_per * 1e9 / (mhz * 1e6)) : 0;
  const double width_b = peak_uni_bpc > 0 ? peak_uni_bpc : bpc_from_gbs;
  const double width_bits = width_b * 8.0;
  int round_bits = 1;
  if (width_bits > 0.5) {
    const double lg = std::log2(width_bits);
    round_bits = 1 << static_cast<int>(std::lround(lg));
    if (round_bits < 8) round_bits = 8;
  }

  std::printf(
      "  pairs=%zu (mesh=%d cs2=%d rev=%d)  perSM_mean=%.2f GB/s  "
      "perSM_peak=%.2f GB/s  CV=%.3f\n",
      pairs.size(), n_mesh, cs2_new, n_rev, mean_per, peak_per, cv);
  std::printf("  peak_uni_bpc=%.3f  peak_bi_perSM_bpc=%.3f  B/cyc\n",
              peak_uni_bpc, peak_bi_bpc);
  std::printf(
      "  DSM datapath width ≈ %.1f B/cyc  (%.0f bits)  rounded=%d bits  "
      "@ %.1f MHz\n",
      width_b, width_bits, round_bits, mhz);
  std::printf(
      "  interpret: width_bytes = peak_uni_BPC; bits = ×8; "
      "use to fit flit payload/lanes; keep legacy dsm_bytes_per_cycle=0\n");

  // Neighbor (|dSM|==1) and per-reader-SM views: hop vs endpoint.
  std::vector<double> nbr_uni;
  std::printf("  --- neighbor pairs (|Δsmid|==1) ---\n");
  for (const auto &p : pairs) {
    if (!p.ok || p.d_smid != 1) continue;
    nbr_uni.push_back(p.uni_gbs);
    std::printf("    sm%-4u-sm%-4u  uni=%6.2f  bi=%6.2f  uni_BPC=%.3f\n",
                p.sm_a, p.sm_b, p.uni_gbs, p.bi_gbs, p.uni_bpc);
  }
  double nbr_mean = mean_dbl(nbr_uni), nbr_var = 0;
  for (double x : nbr_uni) nbr_var += (x - nbr_mean) * (x - nbr_mean);
  if (!nbr_uni.empty()) nbr_var /= static_cast<double>(nbr_uni.size());
  const double nbr_cv = (nbr_mean > 0) ? (std::sqrt(nbr_var) / nbr_mean) : 0;

  struct ReaderAgg {
    uint32_t sm = 0;
    int n = 0;
    double uni_sum = 0, bpc_sum = 0;
  };
  std::vector<ReaderAgg> readers;
  for (const auto &p : pairs) {
    if (!p.ok) continue;
    auto it = std::find_if(readers.begin(), readers.end(),
                           [&](const ReaderAgg &r) { return r.sm == p.sm_a; });
    if (it == readers.end()) {
      readers.push_back({p.sm_a, 1, p.uni_gbs, p.uni_bpc});
    } else {
      ++it->n;
      it->uni_sum += p.uni_gbs;
      it->bpc_sum += p.uni_bpc;
    }
  }
  std::sort(readers.begin(), readers.end(),
            [](const ReaderAgg &a, const ReaderAgg &b) { return a.sm < b.sm; });
  std::printf("  --- grouped by reader SM (rank_a, uni) ---\n");
  std::printf("    %-8s %4s %10s %10s\n", "sm", "n", "mean_uni", "mean_BPC");
  std::vector<double> reader_means;
  for (const auto &r : readers) {
    const double mu = r.uni_sum / std::max(r.n, 1);
    const double mb = r.bpc_sum / std::max(r.n, 1);
    reader_means.push_back(mu);
    std::printf("    sm%-6u %4d %10.2f %10.3f\n", r.sm, r.n, mu, mb);
  }
  const double rmean = mean_dbl(reader_means);
  double rvar = 0;
  for (double x : reader_means) rvar += (x - rmean) * (x - rmean);
  if (!reader_means.empty()) rvar /= static_cast<double>(reader_means.size());
  const double reader_cv = (rmean > 0) ? (std::sqrt(rvar) / rmean) : 0;

  std::printf("  --- variation diagnosis ---\n");
  std::printf(
      "  order: last_half/first_half uni = %.3f (fwd)  %.3f (rev order)\n",
      order_ratio_fwd, order_ratio_rev);
  std::printf(
      "  |dSM|: far/near uni = %.3f (fwd)  %.3f (rev)   "
      "near=%.2f far=%.2f GB/s\n",
      dsmid_ratio, dsmid_ratio_rev, near_m, far_m);
  std::printf("  fwd-vs-rev mean |Δ|/fwd = %.3f  (n=%d remesh pairs)\n",
              rel_fwd_rev, n_rel);
  std::printf("  neighbor |dSM|=1: n=%zu  mean=%.2f GB/s  CV=%.3f\n",
              nbr_uni.size(), nbr_mean, nbr_cv);
  std::printf("  reader-SM group CV=%.3f  (n_readers=%zu)\n", reader_cv,
              readers.size());

  const bool reproducible = (rel_fwd_rev < 0.02 && n_rel >= 4);
  const bool time_warmup =
      (rel_fwd_rev > 0.08 && order_ratio_fwd > 1.15 && order_ratio_rev > 1.15);
  const bool enum_artifact =
      reproducible && order_ratio_fwd > 1.12 && order_ratio_rev < 0.90;
  const bool hop = ((dsmid_ratio > 1.15 || dsmid_ratio < 0.85) &&
                    (dsmid_ratio_rev == 0 || dsmid_ratio_rev > 1.15 ||
                     dsmid_ratio_rev < 0.85) &&
                    nbr_cv < 0.08);
  const bool reader_limited = (reader_cv > 0.12 && nbr_cv > 0.12);
  if (time_warmup && !hop)
    std::printf(
        "  hint: same pair changes with time and later tests are faster "
        "in BOTH passes → warmup/DVFS, not link topology\n");
  else if (reader_limited && reproducible)
    std::printf(
        "  hint: pair BW is deterministic (fwd≈rev) but neighbor pairs "
        "still span a wide range and uni tracks the *reader* SM → "
        "endpoint/SM throughput (clock or LSU), not hop distance. "
        "Do not treat 5090 spread as an H200 NoC map.\n");
  else if (hop && reproducible)
    std::printf(
        "  hint: far vs near |Δsmid| disagrees, neighbors are tight, "
        "fwd≈rev → physical SM–SM topology\n");
  else if (enum_artifact)
    std::printf(
        "  hint: fwd/rev order ratios invert but each pair is identical "
        "→ mesh enumeration follows SM id; not time-warmup\n");
  else
    std::printf(
        "  hint: no single dominant split. Inspect neighbor + reader-SM "
        "tables; CV=%.3f. H200 exclusive for knobs.\n",
        cv);

  sink.add("dsm", "dsm_bw_1sm_peak_gbs", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", peak_per,
           peak_per, peak_per, "GB/s", "best per-SM over isolated pairs");
  sink.add("dsm", "dsm_bw_1sm_pair_cv", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", cv, cv, cv,
           "ratio", "std/mean of per-SM GB/s across pairs");
  sink.add("dsm", "dsm_bw_1sm_n_pairs", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           static_cast<double>(pairs.size()), static_cast<double>(pairs.size()),
           static_cast<double>(pairs.size()), "count",
           "mesh + extra cs=2 unique SM pairs");
  sink.add("dsm", "dsm_bw_1sm_order_ratio_fwd", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           order_ratio_fwd, order_ratio_fwd, order_ratio_fwd, "ratio",
           "last_half/first_half uni GB/s in measurement order");
  sink.add("dsm", "dsm_bw_1sm_order_ratio_rev", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           order_ratio_rev, order_ratio_rev, order_ratio_rev, "ratio",
           "last_half/first_half uni GB/s in reverse remesh order");
  sink.add("dsm", "dsm_bw_1sm_dsmid_far_near", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           dsmid_ratio, dsmid_ratio, dsmid_ratio, "ratio",
           "mean uni far-|dSM| / near-|dSM| (fwd)");
  sink.add("dsm", "dsm_bw_1sm_dsmid_far_near_rev", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           dsmid_ratio_rev, dsmid_ratio_rev, dsmid_ratio_rev, "ratio",
           "mean uni far-|dSM| / near-|dSM| (rev remesh)");
  sink.add("dsm", "dsm_bw_1sm_fwd_rev_rel", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           rel_fwd_rev, rel_fwd_rev, rel_fwd_rev, "ratio",
           "mean |uni_rev-uni_fwd|/uni_fwd");
  sink.add("dsm", "dsm_bw_1sm_nbr_cv", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", nbr_cv,
           nbr_cv, nbr_cv, "ratio", "CV of uni GB/s on |dSM|==1 pairs");
  sink.add("dsm", "dsm_bw_1sm_reader_cv", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           reader_cv, reader_cv, reader_cv, "ratio",
           "CV of per-reader-SM mean uni GB/s");
  sink.add("dsm", "dsm_bpc_1sm_uni", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", peak_uni_bpc,
           peak_uni_bpc, peak_uni_bpc, "bytes/cycle",
           "peak one-SM read BPC (uni)");
  sink.add("dsm", "dsm_bpc_1sm_bi", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", peak_bi_bpc,
           peak_bi_bpc, peak_bi_bpc, "bytes/cycle",
           "peak one-SM BPC under duplex");
  sink.add("dsm", "dsm_datapath_width_bytes", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           width_b, width_b, width_b, "bytes/cycle",
           "primary: peak uni BPC");
  sink.add("dsm", "dsm_datapath_width_bits", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           width_bits, width_bits, width_bits, "bits", "width_bytes * 8");
  sink.add("dsm", "dsm_datapath_width_bits_round", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes",
           static_cast<double>(round_bits), static_cast<double>(round_bits),
           static_cast<double>(round_bits), "bits",
           "nearest power of two");

  CUDA_CHECK(cudaFree(d_smids));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cyc));
}

}  // namespace

// Entry from probe_dsm.cu
void run_dsm_l23_probes(const SuiteOptions &opt, MetricSink &sink,
                        int max_cluster, bool include_bandwidth) {
  std::printf("\n  --- DSM L2–L3 (TODO §2.1–2.4 single-cluster focus) ---\n");

  if (include_bandwidth) {
  // ---- 2.1 Single-cluster BW ----
  std::printf("  --- 2.1 single-cluster DSM bandwidth (n_clusters=1) ---\n");
  (void)cfg_cluster((const void *)dsm_bw_sc_kernel, 32 * 1024);

  const char *pat_names[] = {"ring", "pair", "bcast"};
  const int payloads[] = {256, 1024, 4096, 8192, 16384, 32768};
  const int block_sizes[] = {128, 256, 512};
  const int cluster_sizes[] = {2, 4, 8, 16};

  uint32_t *d_sink = nullptr;
  uint64_t *d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t) * kMaxCluster));
  CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(uint64_t)));

  for (int cs : cluster_sizes) {
    if (cs > max_cluster) continue;
    for (int p = 0; p < 3; ++p) {
      double best_gbs = 0, best_bpc = 0;
      int best_bs = 0, best_pay = 0;
      for (int pay : payloads) {
        for (int bs : block_sizes) {
          if (pay > 48 * 1024) continue;
          int pattern = p, iters = kBwItersSc, payload = pay;
          void *args[] = {&pattern, &iters, &payload, &d_sink, &d_cyc};
          auto ok_launch = [&]() -> bool {
            cudaError_t e =
                launch_1cluster((const void *)dsm_bw_sc_kernel, args, cs, bs,
                                pay);
            if (e != cudaSuccess) {
              (void)cudaGetLastError();
              return false;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
              (void)cudaGetLastError();
              return false;
            }
            return true;
          };
          if (!ok_launch()) continue;
          if (!ok_launch()) continue;

          cudaEvent_t start, stop;
          CUDA_CHECK(cudaEventCreate(&start));
          CUDA_CHECK(cudaEventCreate(&stop));
          CUDA_CHECK(cudaEventRecord(start));
          if (launch_1cluster((const void *)dsm_bw_sc_kernel, args, cs, bs,
                              pay) != cudaSuccess ||
              cudaEventRecord(stop) != cudaSuccess ||
              cudaEventSynchronize(stop) != cudaSuccess) {
            (void)cudaGetLastError();
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            continue;
          }
          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
          CUDA_CHECK(cudaEventDestroy(start));
          CUDA_CHECK(cudaEventDestroy(stop));

          // Bytes: each of cs CTAs reads payload × iters
          const double bytes =
              static_cast<double>(cs) * pay * kBwItersSc;
          const double sec = ms * 1e-3;
          const double gbs =
              (sec > 0) ? (bytes / sec) / (1024.0 * 1024.0 * 1024.0) : 0;

          uint64_t hcy = 0;
          CUDA_CHECK(cudaMemcpy(&hcy, d_cyc, sizeof(uint64_t),
                                cudaMemcpyDeviceToHost));
          double adj = static_cast<double>(hcy);
          if (adj > opt.clock64_overhead) adj -= opt.clock64_overhead;
          // BPC from device cycles on rank0 span covering all iters
          const double bpc = (adj > 0) ? (bytes / adj) : 0;

          if (gbs > best_gbs) {
            best_gbs = gbs;
            best_bpc = bpc;
            best_bs = bs;
            best_pay = pay;
          }

          char name[96], notes[160];
          std::snprintf(name, sizeof(name), "dsm_bw_%s_sc_cs%d_p%d",
                        pat_names[p], cs, pay);
          std::snprintf(notes, sizeof(notes),
                        "n_clusters=1; cluster=%d; payload=%d; block=%d; "
                        "iters=%d; pattern=%s",
                        cs, pay, bs, kBwItersSc, pat_names[p]);
          sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", gbs, gbs, gbs,
                   "GB/s", notes);
        }
      }
      char name[80], notes[160];
      std::snprintf(name, sizeof(name), "dsm_bw_%s_sc_cs%d", pat_names[p], cs);
      std::snprintf(notes, sizeof(notes),
                    "n_clusters=1; best_block=%d best_payload=%d pattern=%s",
                    best_bs, best_pay, pat_names[p]);
      std::printf("  %-44s %10.2f GB/s  bpc_est=%.3f  (%s)\n", name, best_gbs,
                  best_bpc, notes);
      sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", best_gbs, best_gbs,
               best_gbs, "GB/s", notes);
      std::snprintf(name, sizeof(name), "dsm_bpc_%s_est_cs%d", pat_names[p],
                    cs);
      sink.add("dsm", name, "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", best_bpc, best_bpc,
               best_bpc, "bytes/cycle", notes);
    }
  }
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cyc));
  }

  // ---- 2.2 Dependent load RTT ----
  std::printf("  --- 2.2 dependent remote load RTT ---\n");
  int n = std::min(max_cluster, 8);
  if (n < 2) n = 2;
  (void)cfg_cluster((const void *)dsm_dep_load_kernel, 0);
  {
    const size_t elems =
        static_cast<size_t>(opt.samples) * static_cast<size_t>(n);
    uint64_t *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(uint64_t)));
    int ns = opt.samples, cn = n;
    void *args[] = {&d_out, &ns, &cn};
    for (int try_n = n; try_n >= 2; --try_n) {
      cn = try_n;
      ns = opt.samples;
      if (launch_1cluster((const void *)dsm_dep_load_kernel, args, try_n, 32,
                          0) == cudaSuccess &&
          cudaDeviceSynchronize() == cudaSuccess) {
        n = try_n;
        break;
      }
      (void)cudaGetLastError();
    }
    std::vector<uint64_t> h(static_cast<size_t>(opt.samples) * n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out,
                          sizeof(uint64_t) * opt.samples * n,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<double> all;
    std::vector<double> r0r1;
    for (int s = 0; s < opt.samples; ++s) {
      for (int r = 0; r < n; ++r) {
        double adj = static_cast<double>(
                         h[static_cast<size_t>(s) * n + r]) -
                     opt.clock64_overhead;
        if (adj < 0) adj = 0;
        const double per = adj / kChaseSteps;
        all.push_back(per);
        if (r == 0) r0r1.push_back(per);
      }
    }
    const double med = median_vec(all);
    const double med01 = median_vec(r0r1);
    std::printf("  %-44s %10.2f cycles/load  (all ranks→(r+1)%%N dep chase)\n",
                "dsm_remote_load_dep_rtt", med);
    std::printf("  %-44s %10.2f cycles/load  (rank0→1 only)\n",
                "dsm_remote_load_dep_rtt_r0_r1", med01);
    sink.add("dsm", "dsm_remote_load_dep_rtt", "dsm_2x_hop_validate", med, med,
             med, "cycles/load",
             "volatile dependent chase; n_clusters=1");
    sink.add("dsm", "dsm_remote_load_dep_rtt_r0_r1", "dsm_2x_hop_validate",
             med01, med01, med01, "cycles/load", "rank0→rank1");
  }

  // ---- 2.3 Store visibility ----
  std::printf("  --- 2.3 remote store visibility delay ---\n");
  (void)cfg_cluster((const void *)dsm_store_vis_kernel, 0);
  {
    uint64_t *d_ns = nullptr, *d_cy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ns, sizeof(uint64_t) * opt.samples));
    CUDA_CHECK(cudaMalloc(&d_cy, sizeof(uint64_t) * opt.samples));
    CUDA_CHECK(cudaMemset(d_ns, 0, sizeof(uint64_t) * opt.samples));
    CUDA_CHECK(cudaMemset(d_cy, 0, sizeof(uint64_t) * opt.samples));
    int ns = opt.samples;
    void *args[] = {&d_ns, &d_cy, &ns};
    if (launch_1cluster((const void *)dsm_store_vis_kernel, args, 2, 32, 0) ==
            cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess) {
      std::vector<uint64_t> hns(opt.samples), hcy(opt.samples);
      CUDA_CHECK(cudaMemcpy(hns.data(), d_ns, sizeof(uint64_t) * opt.samples,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hcy.data(), d_cy, sizeof(uint64_t) * opt.samples,
                            cudaMemcpyDeviceToHost));
      std::vector<double> vns, vcy;
      for (int i = 0; i < opt.samples; ++i) {
        if (hns[i] == 0xffffffffull) continue;
        vns.push_back(static_cast<double>(hns[i]));
        double adj = static_cast<double>(hcy[i]) - opt.clock64_overhead;
        if (adj < 0) adj = 0;
        vcy.push_back(adj);
      }
      const double med_ns = median_vec(vns);
      const double med_cy = median_vec(vcy);
      // Convert ns → cycles via SM MHz: cycles ≈ ns * 1e-3 * MHz
      const double mhz =
          opt.sm_clock_mhz > 0 ? opt.sm_clock_mhz : 1785.0;
      const double vis_cyc = med_ns * 1e-3 * mhz;
      std::printf("  %-44s %10.2f ns  (~%.1f cycles @ %.0f MHz)\n",
                  "dsm_store_to_peer_visible_ns", med_ns, vis_cyc, mhz);
      std::printf("  %-44s %10.2f cycles  (producer store+fence local)\n",
                  "dsm_store_issue_local_cyc", med_cy);
      sink.add("dsm", "dsm_store_to_peer_visible_ns",
               "gpgpu_dsm_remote_latency", med_ns, med_ns, med_ns, "ns",
               "producer store start to consumer visibility via synchronized globaltimer");
      sink.add("dsm", "dsm_store_to_peer_visible_cyc",
               "gpgpu_dsm_remote_latency", vis_cyc, vis_cyc, vis_cyc, "cycles",
               "from ns * sm_mhz");
      sink.add("dsm", "dsm_store_issue_local_cyc", "gpgpu_dsm_remote_latency",
               med_cy, med_cy, med_cy, "cycles", "producer store+fence clock64");
    } else {
      std::printf("  SKIP store visibility: launch failed\n");
      (void)cudaGetLastError();
    }
    CUDA_CHECK(cudaFree(d_ns));
    CUDA_CHECK(cudaFree(d_cy));
  }

  // ---- 2.4 Contention ----
  std::printf("  --- 2.4 concurrent multi-pair contention ---\n");
  (void)cfg_cluster((const void *)dsm_contention_kernel, 0);
  {
    int cn = std::min(max_cluster, 8);
    if (cn < 2) cn = 2;
    const size_t elems =
        static_cast<size_t>(opt.samples) * static_cast<size_t>(cn);
    uint64_t *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(uint64_t)));

    auto measure_mode = [&](int mode, const char *tag) -> double {
      CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(uint64_t)));
      int md = mode, ns = opt.samples, n = cn;
      void *args[] = {&md, &d_out, &ns, &n};
      for (int try_n = cn; try_n >= 2; --try_n) {
        n = try_n;
        if (launch_1cluster((const void *)dsm_contention_kernel, args, try_n,
                            32, 0) == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess) {
          cn = try_n;
          break;
        }
        (void)cudaGetLastError();
      }
      std::vector<uint64_t> h(static_cast<size_t>(opt.samples) * cn);
      CUDA_CHECK(cudaMemcpy(h.data(), d_out,
                            sizeof(uint64_t) * opt.samples * cn,
                            cudaMemcpyDeviceToHost));
      std::vector<double> v;
      for (int s = 0; s < opt.samples; ++s) {
        for (int r = 0; r < cn; ++r) {
          const uint64_t tot =
              h[static_cast<size_t>(s) * cn + r];
          if (tot == 0 && mode == 0 && r >= 2) continue;
          double adj = static_cast<double>(tot) - opt.clock64_overhead;
          if (adj < 0) adj = 0;
          if (tot > 0) v.push_back(adj / kChaseSteps);
        }
      }
      const double med = median_vec(v);
      std::printf("  %-44s %10.2f cycles/load  (%s cluster=%d)\n", tag, med,
                  tag, cn);
      return med;
    };

    const double lat1 = measure_mode(0, "dsm_pair_lat_1pair");
    const double latA = measure_mode(1, "dsm_pair_lat_allpairs");
    const double ratio = (lat1 > 0) ? (latA / lat1) : 0;
    std::printf("  %-44s %10.3f  (allpairs/1pair)\n",
                "dsm_pair_lat_inflation_ratio", ratio);

    sink.add("dsm", "dsm_pair_lat_1pair", "dsm_contention", lat1, lat1, lat1,
             "cycles/load", "n_clusters=1; only ranks 0,1 active");
    sink.add("dsm", "dsm_pair_lat_allpairs", "dsm_contention", latA, latA, latA,
             "cycles/load", "n_clusters=1; all ranks xor-pair simultaneous");
    sink.add("dsm", "dsm_pair_lat_inflation_ratio", "dsm_contention", ratio,
             ratio, ratio, "ratio", "allpairs/1pair");

    if (include_bandwidth) {
    // All-pairs BW (reuse sc bw pair at this cluster)
    {
      int pay = 16384, bs = 256, p = 1, iters = kBwItersSc;
      uint32_t *ds = nullptr;
      uint64_t *dc = nullptr;
      CUDA_CHECK(cudaMalloc(&ds, sizeof(uint32_t) * cn));
      CUDA_CHECK(cudaMalloc(&dc, sizeof(uint64_t)));
      void *args[] = {&p, &iters, &pay, &ds, &dc};
      (void)cfg_cluster((const void *)dsm_bw_sc_kernel, pay);
      if (launch_1cluster((const void *)dsm_bw_sc_kernel, args, cn, bs, pay) ==
              cudaSuccess &&
          cudaDeviceSynchronize() == cudaSuccess) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        launch_1cluster((const void *)dsm_bw_sc_kernel, args, cn, bs, pay);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        const double bytes =
            static_cast<double>(cn) * pay * kBwItersSc;
        const double gbs =
            (ms > 0) ? (bytes / (ms * 1e-3)) / (1024.0 * 1024.0 * 1024.0) : 0;
        std::printf("  %-44s %10.2f GB/s  (all-pairs pair pattern sc)\n",
                    "dsm_bw_allpairs_sc", gbs);
        char notes[96];
        std::snprintf(notes, sizeof(notes),
                      "n_clusters=1; cluster=%d; payload=%d; pattern=pair", cn,
                      pay);
        sink.add("dsm", "dsm_bw_allpairs_sc", "gpgpu_dsm_flit_payload_bytes / lanes_per_cpc / gx_planes", gbs,
                 gbs, gbs, "GB/s", notes);
      }
      CUDA_CHECK(cudaFree(ds));
      CUDA_CHECK(cudaFree(dc));
    }
    }
    CUDA_CHECK(cudaFree(d_out));
  }

  // Isolated 2-SM peak BW + datapath width (only one pair active)
  if (include_bandwidth) run_dsm_1sm_peak(opt, sink, max_cluster);
}
