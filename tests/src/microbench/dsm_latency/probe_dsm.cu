// DSM / distributed shared memory probes aligned with Luo et al. arXiv:2501.12084 §7
//
// 1) Latency matrix (existing): all-pairs remote smem loads via mapa.
// 2) Bandwidth (new, §7.2): ring / pair / broadcast patterns → GB/s.
// 3) Stride topology (new): latency vs rank stride 1,2,4,... to probe
//    all-to-all vs tree-like SM-to-SM network.

#include "common.cuh"

#include <cooperative_groups.h>

#include <map>
#include <set>
#include <utility>

namespace cg = cooperative_groups;

namespace {

constexpr int kMaxCluster = 32;
constexpr int kSmemElems = 256;
constexpr int kChaseSteps = 64;
constexpr int kBwSmemBytes = 16 * 1024;  // paper uses 16 KB per block
constexpr int kBwIters = 64;

// ---------------------------------------------------------------------------
// Latency matrix (paper §7.1)
// ---------------------------------------------------------------------------

__global__ void dsm_matrix_kernel(uint64_t *latency_tot, uint32_t *smids,
                                  int cluster_n, int n_samples,
                                  int warmup_iters) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  if (n != cluster_n) {
    if (threadIdx.x == 0 && rank == 0) {
      for (int i = 0; i < cluster_n; ++i) smids[i] = 0xffffffffu;
    }
    return;
  }

  __shared__ int smem[kSmemElems];
  __shared__ uint32_t smid_slot;

  for (int i = threadIdx.x; i < kSmemElems; i += blockDim.x) {
    smem[i] = (i + 1) % kSmemElems;
  }
  if (threadIdx.x == 0) {
    smid_slot = smid_now();
    smids[rank] = smid_slot;
  }
  cluster.sync();

  for (int src = 0; src < n; ++src) {
    if (rank == src && threadIdx.x == 0) {
      for (int dst = 0; dst < n; ++dst) {
        int *remote = cluster.map_shared_rank(smem, dst);
        int idx = 0;
        for (int i = 0; i < warmup_iters * kChaseSteps; ++i) {
          idx = remote[idx % kSmemElems];
        }
        (void)idx;
      }
    }
    cluster.sync();
  }

  for (int s = 0; s < n_samples; ++s) {
    for (int src = 0; src < n; ++src) {
      if (rank == src && threadIdx.x == 0) {
        for (int dst = 0; dst < n; ++dst) {
          // mapa OUTSIDE the timer: measure pure remote smem *load* latency,
          // not map_shared_rank / address setup (same isolation as TMA pure).
          int *remote = cluster.map_shared_rank(smem, dst);
          int idx = 0;
          asm volatile("" ::: "memory");
          const uint64_t t0 = clock64_now();
#pragma unroll 1
          for (int i = 0; i < kChaseSteps; ++i) {
            idx = remote[idx];
          }
          const uint64_t t1 = clock64_now();
          latency_tot[static_cast<size_t>(s) * n * n +
                      static_cast<size_t>(src) * n + dst] = t1 - t0;
          if (idx == 0x7fffffff) latency_tot[0] = 0;
        }
      }
      cluster.sync();
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < cluster_n; ++i) smids[i] = 0xffffffffu;
  }
#endif
}

// ---------------------------------------------------------------------------
// Stride topology latency: each rank times load from (rank+stride)%N
// ---------------------------------------------------------------------------

// out[sample * n + rank] = total cycles for kChaseSteps to dest (rank+stride)%n
__global__ void dsm_stride_kernel(uint64_t *out, uint32_t *smids, int cluster_n,
                                  int stride, int n_samples) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());
  if (n != cluster_n) return;

  __shared__ int smem[kSmemElems];
  for (int i = threadIdx.x; i < kSmemElems; i += blockDim.x) {
    smem[i] = (i + 1) % kSmemElems;
  }
  if (threadIdx.x == 0) smids[rank] = smid_now();
  cluster.sync();

  const int dst = (rank + stride) % n;
  // mapa outside timer (pure load chase).
  int *remote = cluster.map_shared_rank(smem, dst);

  // Warmup
  if (threadIdx.x == 0) {
    int idx = 0;
    for (int i = 0; i < kChaseSteps * 4; ++i) idx = remote[idx % kSmemElems];
    (void)idx;
  }
  cluster.sync();

  for (int s = 0; s < n_samples; ++s) {
    if (threadIdx.x == 0) {
      int idx = 0;
      asm volatile("" ::: "memory");
      const uint64_t t0 = clock64_now();
#pragma unroll 1
      for (int i = 0; i < kChaseSteps; ++i) {
        idx = remote[idx];
      }
      const uint64_t t1 = clock64_now();
      out[static_cast<size_t>(s) * n + rank] = t1 - t0;
      if (idx == 0x7fffffff) out[0] = 0;
    }
    cluster.sync();
  }
#else
  (void)out;
  (void)smids;
  (void)cluster_n;
  (void)stride;
  (void)n_samples;
#endif
}

// ---------------------------------------------------------------------------
// Bandwidth kernels (paper §7.2 patterns)
// pattern: 0=ring, 1=pair, 2=broadcast
// ---------------------------------------------------------------------------

__global__ void dsm_bw_kernel(int pattern, int iters, uint32_t *sink) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int n = static_cast<int>(cluster.num_blocks());

  extern __shared__ __align__(16) uint8_t smem_bytes[];
  // Init local smem
  for (int i = threadIdx.x; i < kBwSmemBytes / 4; i += blockDim.x) {
    reinterpret_cast<uint32_t *>(smem_bytes)[i] =
        static_cast<uint32_t>(rank * 131 + i);
  }
  cluster.sync();

  int src_rank = 0;
  if (pattern == 0) {
    src_rank = (rank + 1) % n;  // ring
  } else if (pattern == 1) {
    src_rank = rank ^ 1;  // pair (butterfly step); if n odd, clamp
    if (src_rank >= n) src_rank = rank;
  } else {
    src_rank = 0;  // broadcast from rank 0
  }

  const uint32_t *remote = reinterpret_cast<const uint32_t *>(
      cluster.map_shared_rank(smem_bytes, src_rank));
  uint32_t acc = 0;

  // All threads participate (paper: large block size improves BW).
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    for (int i = threadIdx.x; i < kBwSmemBytes / 4; i += blockDim.x) {
      acc += remote[i];
    }
    // Re-sync so remote smem is stable if someone were writing (read-only here).
    cluster.sync();
  }

  // Reduce acc within CTA so compiler cannot drop loads.
  __shared__ uint32_t red;
  if (threadIdx.x == 0) red = 0;
  __syncthreads();
  atomicAdd(&red, acc);
  __syncthreads();
  if (threadIdx.x == 0) {
    sink[blockIdx.x] = red;
  }
#else
  if (threadIdx.x == 0) sink[blockIdx.x] = 0;
#endif
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

cudaError_t configure_fn(const void *fn, int dyn_smem) {
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

cudaError_t launch_cluster_ex(const void *fn, void **args, int cluster_n,
                              int block_dim, int dyn_smem) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster_n, 1, 1);
  cfg.blockDim = dim3(block_dim, 1, 1);
  cfg.dynamicSmemBytes = static_cast<size_t>(dyn_smem);
  cfg.stream = 0;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster_n;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return cudaLaunchKernelExC(&cfg, fn, args);
}

cudaError_t launch_dsm_matrix(uint64_t *d_lat, uint32_t *d_smids, int n,
                              int n_samples, int warmup_iters) {
  void *args[] = {&d_lat, &d_smids, &n, &n_samples, &warmup_iters};
  return launch_cluster_ex((const void *)dsm_matrix_kernel, args, n, 32, 0);
}

cudaError_t launch_dsm_stride(uint64_t *d_out, uint32_t *d_smids, int n,
                              int stride, int n_samples) {
  void *args[] = {&d_out, &d_smids, &n, &stride, &n_samples};
  return launch_cluster_ex((const void *)dsm_stride_kernel, args, n, 32, 0);
}

int find_max_cluster_size(const void *fn, int dyn_smem) {
  cudaLaunchConfig_t cfg = {};
  cfg.blockDim = dim3(32, 1, 1);
  cfg.dynamicSmemBytes = static_cast<size_t>(dyn_smem);
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  int potential = 0;
  attr.val.clusterDim.x = kMaxCluster;
  cfg.gridDim = dim3(kMaxCluster, 1, 1);
  if (cudaOccupancyMaxPotentialClusterSize(&potential, fn, &cfg) ==
          cudaSuccess &&
      potential > 0) {
    return std::min(potential, kMaxCluster);
  }
  (void)cudaGetLastError();
  for (int n = kMaxCluster; n >= 2; --n) {
    attr.val.clusterDim.x = n;
    cfg.gridDim = dim3(n, 1, 1);
    int active = 0;
    if (cudaOccupancyMaxActiveClusters(&active, fn, &cfg) == cudaSuccess &&
        active >= 1) {
      return n;
    }
    (void)cudaGetLastError();
  }
  return 2;
}

double median_of(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return percentile_sorted(v, 0.5);
}

void print_matrix(const char *title, const std::vector<std::string> &labels,
                  const std::vector<double> &mat, int n) {
  std::printf("  %s\n", title);
  std::printf("  %10s", "src\\dst");
  for (int j = 0; j < n; ++j) std::printf(" %10s", labels[j].c_str());
  std::printf("\n");
  for (int i = 0; i < n; ++i) {
    std::printf("  %10s", labels[i].c_str());
    for (int j = 0; j < n; ++j) {
      std::printf(" %10.1f", mat[static_cast<size_t>(i) * n + j]);
    }
    std::printf("\n");
  }
}

// ---------------------------------------------------------------------------
// Host runners
// ---------------------------------------------------------------------------

int launch_matrix_with_shrink(uint64_t *d_lat, uint32_t *d_smids, int n_want,
                              int samples, int warmup) {
  for (int try_n = n_want; try_n >= 2; --try_n) {
    bool ok = true;
    for (int w = 0; w < std::max(1, warmup); ++w) {
      cudaError_t cerr =
          launch_dsm_matrix(d_lat, d_smids, try_n, /*n_samples=*/1, 2);
      if (cerr != cudaSuccess) {
        ok = false;
        break;
      }
      cerr = cudaDeviceSynchronize();
      if (cerr != cudaSuccess) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      std::printf("  cluster size %d launch failed — trying smaller\n", try_n);
      (void)cudaGetLastError();
      continue;
    }
    // Full measurement
    cudaError_t cerr = launch_dsm_matrix(d_lat, d_smids, try_n, samples, 8);
    if (cerr != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      (void)cudaGetLastError();
      continue;
    }
    return try_n;
  }
  return 0;
}

void run_stride_topology(const SuiteOptions &opt, MetricSink &sink, int n) {
  std::printf("\n  --- stride topology latency (probe all-to-all vs tree) ---\n");
  std::printf(
      "  method: each rank pointer-chases smem of rank (r+stride)%%N\n");
  std::printf(
      "  interpret: flat NoC ⇒ latency ~const vs stride; tree ⇒ grows with "
      "stride\n");

  (void)configure_fn((const void *)dsm_stride_kernel, 0);

  const size_t elems =
      static_cast<size_t>(opt.samples) * static_cast<size_t>(n);
  uint64_t *d_out = nullptr;
  uint32_t *d_smids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_smids, static_cast<size_t>(n) * sizeof(uint32_t)));

  std::vector<double> stride_medians;
  std::vector<int> strides;

  for (int stride = 1; stride <= n / 2; stride *= 2) {
    CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(uint64_t)));
    cudaError_t cerr =
        launch_dsm_stride(d_out, d_smids, n, stride, opt.samples);
    if (cerr != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      std::printf("  SKIP stride=%d: %s\n", stride, cudaGetErrorString(cerr));
      (void)cudaGetLastError();
      continue;
    }
    std::vector<uint64_t> h(elems);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, elems * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // Per-rank median cycles/load, then mean across ranks
    std::vector<double> rank_med;
    for (int r = 0; r < n; ++r) {
      std::vector<double> ps;
      for (int s = 0; s < opt.samples; ++s) {
        double adj = static_cast<double>(
                         h[static_cast<size_t>(s) * n + r]) -
                     opt.clock64_overhead;
        if (adj < 0) adj = 0;
        ps.push_back(adj / kChaseSteps);
      }
      rank_med.push_back(median_of(ps));
    }
    const double med = median_of(rank_med);
    const double mn =
        *std::min_element(rank_med.begin(), rank_med.end());
    const double mx =
        *std::max_element(rank_med.begin(), rank_med.end());

    std::printf("  stride=%-2d  median=%.1f  min=%.1f  max=%.1f  cycles/load\n",
                stride, med, mn, mx);
    char name[64];
    std::snprintf(name, sizeof(name), "dsm_stride_%d_latency", stride);
    char notes[96];
    std::snprintf(notes, sizeof(notes),
                  "cluster=%d; dest=(r+%d)%%N; min=%.1f max=%.1f", n, stride, mn,
                  mx);
    sink.add("dsm", name, "dsm_topology_stride", med, mx, med, "cycles/load",
             notes);
    stride_medians.push_back(med);
    strides.push_back(stride);
  }

  if (stride_medians.size() >= 2) {
    const double s1 = stride_medians.front();
    const double smax = stride_medians.back();
    const double ratio = (s1 > 0) ? smax / s1 : 0;
    std::printf(
        "  stride_latency_ratio (s_max/s_1)=%.3f  "
        "(~1 ⇒ all-to-all/flat; ≫1 ⇒ multi-hop/tree-like)\n",
        ratio);
    sink.add("dsm", "dsm_stride_latency_ratio", "dsm_topology", ratio, ratio,
             ratio, "ratio", "max_stride_lat / stride1_lat");
  }

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_smids));
}

// Launch many clusters so SM-to-SM network is saturated (paper §7.2).
cudaError_t launch_bw_multi(int pattern, int iters, uint32_t *sink, int cs,
                            int block_dim, int n_clusters, int dyn_smem) {
  const int grid = cs * n_clusters;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
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
  void *args[] = {&pattern, &iters, &sink};
  return cudaLaunchKernelExC(&cfg, (const void *)dsm_bw_kernel, args);
}

void run_bandwidth(const SuiteOptions &opt, MetricSink &sink, int max_n) {
  std::printf("\n  --- DSM bandwidth (paper §7.2 ring/pair/broadcast) ---\n");
  std::printf("  smem=%d B/CTA  iters=%d  multi-cluster saturation\n",
              kBwSmemBytes, kBwIters);

  (void)configure_fn((const void *)dsm_bw_kernel, kBwSmemBytes);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  const int nsm = prop.multiProcessorCount;

  const char *pat_names[] = {"ring", "pair", "bcast"};
  const int block_sizes[] = {128, 256, 512};
  const int cluster_sizes[] = {2, 4, 8, 16};

  // sink[blockIdx.x] — size to max grid
  const int max_grid = nsm * 4;
  uint32_t *d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t) * max_grid));

  for (int cs : cluster_sizes) {
    if (cs > max_n) continue;
    // Cover ~4× SM count with whole clusters (paper uses huge residency).
    int n_clusters = std::max(1, (nsm * 4) / cs);
    // Cap to keep launch legal / runtime bounded
    if (n_clusters * cs > max_grid) n_clusters = max_grid / cs;
    if (n_clusters < 1) n_clusters = 1;

    for (int p = 0; p < 3; ++p) {
      double best_gbs = 0;
      int best_bs = 0;
      for (int bs : block_sizes) {
        auto try_once = [&]() -> bool {
          cudaError_t e =
              launch_bw_multi(p, kBwIters, d_sink, cs, bs, n_clusters,
                              kBwSmemBytes);
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
        if (!try_once()) continue;  // warmup attempt
        if (!try_once()) continue;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        if (launch_bw_multi(p, kBwIters, d_sink, cs, bs, n_clusters,
                            kBwSmemBytes) != cudaSuccess ||
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

        // Each CTA reads kBwSmemBytes × iters from a peer (or self for bcast src).
        const int grid = cs * n_clusters;
        const double bytes =
            static_cast<double>(grid) * kBwSmemBytes * kBwIters;
        const double sec = ms * 1e-3;
        const double gbs =
            (sec > 0) ? (bytes / sec) / (1024.0 * 1024.0 * 1024.0) : 0;
        if (gbs > best_gbs) {
          best_gbs = gbs;
          best_bs = bs;
        }
      }
      char name[80];
      std::snprintf(name, sizeof(name), "dsm_bw_%s_cs%d", pat_names[p], cs);
      char notes[128];
      std::snprintf(notes, sizeof(notes),
                    "best_block=%d pattern=%s cluster=%d n_clusters=%d", best_bs,
                    pat_names[p], cs, n_clusters);
      std::printf("  %-44s %10.2f GB/s  (%s)\n", name, best_gbs, notes);
      sink.add("dsm", name, "dsm_bandwidth", best_gbs, best_gbs, best_gbs,
               "GB/s", notes);
    }
  }
  CUDA_CHECK(cudaFree(d_sink));
}

}  // namespace

// Implemented in probe_dsm_l23.cu (TODO §2.1–2.4)
void run_dsm_l23_probes(const SuiteOptions &opt, MetricSink &sink,
                        int max_cluster, bool include_bandwidth);

namespace {

void run_dsm_probes_impl(const SuiteOptions &opt, MetricSink &sink,
                         bool include_bandwidth) {
  std::printf(include_bandwidth
                  ? "\n--- dsm (latency matrix + bandwidth + stride topology) ---\n"
                  : "\n--- dsm_calibration (latency/store/contention only) ---\n");
  print_kernel_source("src/probe_dsm.cu + src/probe_dsm_l23.cu",
                      "H200_profiling (our own cluster DSM kernels)",
                      "clock64 device-side measurements and CUDA-event bandwidth",
                      "dsm_sm_to_sm_latency / dsm_bandwidth");
  std::printf("  methodology: Luo et al. arXiv:2501.12084 §7 + FlashGPU-Sim TODO\n");

  int cluster_launch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                                    opt.device));
  if (!cluster_launch) {
    std::printf("  SKIP: cudaDevAttrClusterLaunch == 0\n");
    sink.add("dsm", "skipped", "dsm_sm_to_sm_latency", 0, 0, 0, "cycles",
             "no cluster launch");
    return;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  if (prop.major < 9) {
    std::printf("  SKIP: need sm_90+ (device %d.%d)\n", prop.major, prop.minor);
    sink.add("dsm", "skipped", "dsm_sm_to_sm_latency", 0, 0, 0, "cycles",
             "cc < 9.0");
    return;
  }

  (void)configure_fn((const void *)dsm_matrix_kernel, 0);

  int n = opt.dsm_cluster_size;
  if (n <= 0) n = find_max_cluster_size((const void *)dsm_matrix_kernel, 0);
  if (n < 2) n = 2;
  if (n > kMaxCluster) n = kMaxCluster;

  std::printf("  requested_cluster_size: %d  chase_steps: %d  samples: %d\n", n,
              kChaseSteps, opt.samples);

  const size_t mat_elems =
      static_cast<size_t>(opt.samples) * static_cast<size_t>(n) * n;
  uint64_t *d_lat = nullptr;
  uint32_t *d_smids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lat, mat_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_smids, static_cast<size_t>(n) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_lat, 0, mat_elems * sizeof(uint64_t)));
  CUDA_CHECK(
      cudaMemset(d_smids, 0xff, static_cast<size_t>(n) * sizeof(uint32_t)));

  int launched_n =
      launch_matrix_with_shrink(d_lat, d_smids, n, opt.samples, opt.warmup);
  if (launched_n < 2) {
    std::printf("  SKIP: could not launch any cluster size >= 2\n");
    sink.add("dsm", "skipped", "dsm_sm_to_sm_latency", 0, 0, 0, "cycles",
             "launch failed");
    CUDA_CHECK(cudaFree(d_lat));
    CUDA_CHECK(cudaFree(d_smids));
    return;
  }
  n = launched_n;
  std::printf("  launched_cluster_size: %d\n", n);

  const size_t full_elems =
      static_cast<size_t>(opt.samples) * static_cast<size_t>(n) * n;
  std::vector<uint64_t> h_lat(full_elems);
  std::vector<uint32_t> h_smids(n);
  CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat, full_elems * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_smids.data(), d_smids, n * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_lat));
  CUDA_CHECK(cudaFree(d_smids));

  if (h_smids[0] == 0xffffffffu) {
    std::printf("  SKIP: invalid smids\n");
    sink.add("dsm", "skipped", "dsm_sm_to_sm_latency", 0, 0, 0, "cycles",
             "invalid smids");
    return;
  }

  std::printf("  rank_to_smid:");
  std::set<uint32_t> unique_sm;
  for (int r = 0; r < n; ++r) {
    std::printf("  r%d→sm%u", r, h_smids[r]);
    unique_sm.insert(h_smids[r]);
  }
  std::printf("\n  unique_sms_in_cluster: %zu / %d ranks\n", unique_sm.size(),
              n);

  std::vector<double> rank_mat(static_cast<size_t>(n) * n, 0.0);
  std::vector<std::string> rank_labels(n);
  double off_diag_sum = 0;
  int off_diag_n = 0;
  double diag_sum = 0;

  for (int src = 0; src < n; ++src) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "r%d/sm%u", src, h_smids[src]);
    rank_labels[src] = buf;
    for (int dst = 0; dst < n; ++dst) {
      std::vector<double> per_sample;
      for (int s = 0; s < opt.samples; ++s) {
        const uint64_t tot =
            h_lat[static_cast<size_t>(s) * n * n +
                  static_cast<size_t>(src) * n + dst];
        double adj = static_cast<double>(tot) - opt.clock64_overhead;
        if (adj < 0) adj = 0;
        per_sample.push_back(adj / static_cast<double>(kChaseSteps));
      }
      const double med = median_of(per_sample);
      rank_mat[static_cast<size_t>(src) * n + dst] = med;
      if (src == dst)
        diag_sum += med;
      else {
        off_diag_sum += med;
        ++off_diag_n;
      }
    }
  }

  print_matrix("rank×rank latency matrix (median cycles/load)", rank_labels,
               rank_mat, n);

  const double local_mean = diag_sum / std::max(1, n);
  const double remote_mean =
      off_diag_n > 0 ? off_diag_sum / off_diag_n : 0.0;
  double remote_min = 1e300, remote_max = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double v = rank_mat[static_cast<size_t>(i) * n + j];
      remote_min = std::min(remote_min, v);
      remote_max = std::max(remote_max, v);
    }
  }
  if (off_diag_n == 0) remote_min = remote_max = 0;

  std::printf("  summary:\n");
  std::printf("    local_smem_mean (diag)     = %.2f cycles/load\n", local_mean);
  std::printf("    remote_smem_mean (offdiag) = %.2f cycles/load\n",
              remote_mean);
  std::printf("    remote_min / remote_max    = %.2f / %.2f\n", remote_min,
              remote_max);
  std::printf("    remote_minus_local         = %.2f cycles/load\n",
              remote_mean - local_mean);

  sink.add("dsm", "dsm_local_smem_mean", "gpgpu_smem_latency", local_mean,
           local_mean, local_mean, "cycles/load", "diagonal mean");
  sink.add("dsm", "dsm_remote_smem_mean", "dsm_sm_to_sm_latency", remote_mean,
           remote_mean, remote_mean, "cycles/load", "off-diagonal mean");
  sink.add("dsm", "dsm_remote_smem_min", "dsm_sm_to_sm_latency", remote_min,
           remote_min, remote_min, "cycles/load", "min off-diagonal");
  sink.add("dsm", "dsm_remote_smem_max", "dsm_sm_to_sm_latency", remote_max,
           remote_max, remote_max, "cycles/load", "max off-diagonal");
  sink.add("dsm", "dsm_remote_minus_local", "dsm_noc_extra_latency",
           remote_mean - local_mean, remote_mean - local_mean,
           remote_mean - local_mean, "cycles/load", "mean remote − local");
  sink.add("dsm", "dsm_cluster_size", "n/a", n, n, n, "ctas",
           "launched cluster width");

  const double hop_est = std::max(0.0, (remote_mean - local_mean) * 0.5);
  sink.add("dsm", "dsm_one_way_hop_est", "gpgpu_dsm_base_latency_cycles",
           hop_est, hop_est, hop_est, "cycles",
           "(dependent remote RTT - local service) / 2");

  if (include_bandwidth) {
    run_stride_topology(opt, sink, n);
    run_bandwidth(opt, sink, n);
  }

  // FlashGPU-Sim TODO §2.1–2.4 (single-cluster BW, dep RTT, store, contention)
  run_dsm_l23_probes(opt, sink, n, include_bandwidth);
}

}  // namespace

void run_dsm_probes(const SuiteOptions &opt, MetricSink &sink) {
  run_dsm_probes_impl(opt, sink, true);
}

void run_dsm_calibration_probes(const SuiteOptions &opt, MetricSink &sink) {
  run_dsm_probes_impl(opt, sink, false);
}
