// Dependent remote-load round trip, store visibility, and all-pairs contention.

#include "common.cuh"

#include <cooperative_groups.h>

#include <vector>

namespace cg = cooperative_groups;

// Forward from probe_dsm.cu helpers — reimplement minimal launch here.
namespace {

constexpr int kChaseSteps = 64;
constexpr int kSmemElems = 256;

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


}  // namespace

// Entry from probe_dsm.cu
void run_dsm_l23_probes(const SuiteOptions &opt, MetricSink &sink,
                        int max_cluster) {
  std::printf("\n  --- dependent load, store visibility, contention ---\n");


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

    CUDA_CHECK(cudaFree(d_out));
  }

}
