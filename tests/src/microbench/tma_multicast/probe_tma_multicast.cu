// TMA multicast probes (extends paper §5 timing discipline).
//
// New tests:
//   3) Issuer vs receiver simultaneity via %globaltimer (device-wide ns).
//   4) Multicast cost sweep over transfer sizes.
// Primary path: .multicast::cluster + ctaMask (legacy shared::cluster optional).

#include "common.cuh"

#include <cooperative_groups.h>

#include <chrono>
#include <thread>

namespace cg = cooperative_groups;

namespace {

constexpr int kMaxCluster = 16;
constexpr int kMaxBytes = 16384;
constexpr int kMbarMaxSpins = 1'000'000;
constexpr float kLaunchTimeoutMs = 15000.f;
// Simultaneity: report raw skew; flag "simultaneous" if max skew < this (ns).
constexpr uint64_t kSimulThresholdNs = 2000;  // 2 µs

#if __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t cluster_nrank() {
  uint32_t n;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n));
  return n;
}

__device__ __forceinline__ void mbar_init(uint64_t *bar, unsigned count) {
  const uint32_t p = smem_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(p), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_expect(uint64_t *bar, unsigned tx) {
  const uint32_t p = smem_u32(bar);
  asm volatile(
      "{\n"
      "  .reg .b64 t;\n"
      "  mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1;\n"
      "}\n" ::"r"(p),
      "r"(tx)
      : "memory");
}

__device__ __forceinline__ int mbar_wait_bounded(uint64_t *bar,
                                                 unsigned parity) {
  const uint32_t p = smem_u32(bar);
  for (int i = 0; i < kMbarMaxSpins; ++i) {
    int ok = 0;
    asm volatile(
        "{\n"
        "  .reg .pred pred;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 pred, [%1], %2;\n"
        "  selp.s32 %0, 1, 0, pred;\n"
        "}\n"
        : "=r"(ok)
        : "r"(p), "r"(parity)
        : "memory");
    if (ok) return 1;
  }
  return 0;
}

// Pure bulk issue: addresses already converted (no cvta inside timed window).
template <int BYTES>
__device__ __forceinline__ void bulk_cta_pure(unsigned long long dst_s,
                                               unsigned long long src_g,
                                               unsigned long long bar_s) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %3, [%2];" ::"l"(dst_s),
      "l"(src_g), "l"(bar_s), "n"(BYTES)
      : "memory");
}

template <int BYTES>
__device__ __forceinline__ void bulk_cluster_mask_pure(uint32_t dst_s,
                                                       unsigned long long src_g,
                                                       uint32_t bar_s,
                                                       uint16_t mask) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1], %3, [%2], %4;" ::"r"(dst_s),
      "l"(src_g), "r"(bar_s), "n"(BYTES), "h"(mask)
      : "memory");
}

#endif  // __CUDA_ARCH__ >= 900

enum class McMode : int { UnicastCta = 0, ClusterMaskFull = 1 };

// Per-launch timing / correctness result collected on rank 0.
struct McTimed {
  uint64_t issue_cycles;       // clock64 issue gap on issuer
  uint64_t e2e_cycles;         // clock64 issuer issue→local done
  uint64_t issuer_done_ns;     // globaltimer at issuer mbar done
  uint64_t receiver_done_ns;   // globaltimer at rank-1 mbar done
  uint64_t skew_max_ns;        // max |recv_done - issuer_done| over dests
  uint32_t payload_match;      // 1 if all dest ranks saw same smem word
  uint32_t ok;
};

__global__ void k_cluster_ping(int *out_n, int expect_n) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  cluster.sync();
  if (threadIdx.x == 0 && cluster.block_rank() == 0) {
    const int n = static_cast<int>(cluster.num_blocks());
    *out_n = (n == expect_n) ? n : -n;
  }
  cluster.sync();
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) *out_n = 0;
#endif
}

// Single sample: unicast or mask multicast with globaltimer simultaneity.
template <int BYTES>
__global__ void k_tma_mc_once(const uint8_t *g, McTimed *out, int mode,
                              uint16_t cta_mask) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint64_t done_ns;     // per-CTA completion globaltimer
  __shared__ uint32_t payload_word;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster_rank());
  const int n = static_cast<int>(cluster_nrank());

  if (threadIdx.x == 0) {
    mbar_init(&bar, /*count=*/1);
    done_ns = 0;
    payload_word = 0;
  }
  __syncthreads();
  mbar_init_fence();
  cluster.sync();

  const bool is_dest =
      (mode == static_cast<int>(McMode::UnicastCta))
          ? (rank == 0)
          : (((cta_mask >> rank) & 1) != 0);

  // Precompute addresses BEFORE any timed region (same discipline as pure TMA).
  unsigned long long dst_s64 = 0, src_g64 = 0, bar_s64 = 0;
  uint32_t dst_s32 = 0, bar_s32 = 0;
  if (threadIdx.x == 0) {
    dst_s64 = 0;
    src_g64 = 0;
    bar_s64 = 0;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s64) : "l"(smem));
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g64) : "l"(g));
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s64) : "l"(&bar));
    dst_s32 = smem_u32(smem);
    bar_s32 = smem_u32(&bar);
  }

  if (threadIdx.x == 0 && is_dest) {
    mbar_expect(&bar, BYTES);
  }
  __syncthreads();
  cluster.sync();

  uint64_t issue_dt = 0, e2e_dt = 0;
  int local_ok = 1;
  constexpr unsigned kParity = 0;

  if (rank == 0 && threadIdx.x == 0) {
    asm volatile("" ::: "memory");
    // issue_dt: ONLY pure bulk (no cvta / expect_tx).
    const uint64_t t0 = clock64_now();
    if (mode == static_cast<int>(McMode::UnicastCta)) {
      bulk_cta_pure<BYTES>(dst_s64, src_g64, bar_s64);
    } else {
      bulk_cluster_mask_pure<BYTES>(dst_s32, src_g64, bar_s32, cta_mask);
    }
    const uint64_t t1 = clock64_now();
    issue_dt = t1 - t0;
    // e2e still covers issue → local mbarrier completion (no cvta).
    if (is_dest) {
      local_ok = mbar_wait_bounded(&bar, kParity);
      done_ns = globaltimer_now();
    }
    e2e_dt = clock64_now() - t0;
    payload_word = *reinterpret_cast<uint32_t *>(smem);
  } else if (threadIdx.x == 0 && is_dest) {
    local_ok = mbar_wait_bounded(&bar, kParity);
    done_ns = globaltimer_now();
    payload_word = *reinterpret_cast<uint32_t *>(smem);
  }

  __syncthreads();
  cluster.sync();

  // Rank 0 gathers completion times + payload from all destination ranks.
  if (rank == 0 && threadIdx.x == 0) {
    out->issue_cycles = issue_dt;
    out->e2e_cycles = e2e_dt;
    out->ok = (local_ok != 0) ? 1u : 0u;
    out->issuer_done_ns = done_ns;
    out->receiver_done_ns = 0;
    out->skew_max_ns = 0;
    out->payload_match = 1;

    const uint32_t ref = payload_word;
    uint64_t issuer_t = done_ns;
    uint64_t max_skew = 0;
    uint64_t rank1_t = 0;

    for (int r = 0; r < n; ++r) {
      const bool dest =
          (mode == static_cast<int>(McMode::UnicastCta))
              ? (r == 0)
              : (((cta_mask >> r) & 1) != 0);
      if (!dest) continue;
      uint64_t *remote_ns = cluster.map_shared_rank(&done_ns, r);
      uint32_t *remote_pay = cluster.map_shared_rank(&payload_word, r);
      const uint64_t t = *remote_ns;
      const uint32_t p = *remote_pay;
      if (p != ref) out->payload_match = 0;
      if (r == 1) rank1_t = t;
      const uint64_t sk = (t >= issuer_t) ? (t - issuer_t) : (issuer_t - t);
      if (sk > max_skew) max_skew = sk;
    }
    out->receiver_done_ns = rank1_t;
    out->skew_max_ns = max_skew;
    if (smem[0] == 0xff) out->ok |= 0u;
  }
  cluster.sync();
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = {};
  }
#endif
}

// Explicit instantiations for size sweep.
#define INST_MC(B)                                                             \
  template __global__ void k_tma_mc_once<B>(const uint8_t *, McTimed *, int,   \
                                            uint16_t)
INST_MC(256);
INST_MC(512);
INST_MC(1024);
INST_MC(2048);
INST_MC(4096);
INST_MC(8192);
INST_MC(12288);
INST_MC(16384);
#undef INST_MC

cudaError_t set_cluster_attrs(const void *fn, int dyn_smem) {
  cudaError_t e = cudaFuncSetAttribute(
      fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
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

cudaError_t launch_cluster(const void *fn, void **argv, int cluster_n,
                           int dyn_smem) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster_n, 1, 1);
  cfg.blockDim = dim3(32, 1, 1);
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
  return cudaLaunchKernelExC(&cfg, fn, argv);
}

bool sync_with_timeout(float timeout_ms, int device, bool *timed_out) {
  if (timed_out) *timed_out = false;
  cudaEvent_t done;
  if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess) {
    (void)cudaGetLastError();
    return cudaDeviceSynchronize() == cudaSuccess;
  }
  if (cudaEventRecord(done, 0) != cudaSuccess) {
    (void)cudaGetLastError();
    cudaEventDestroy(done);
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  while (true) {
    cudaError_t q = cudaEventQuery(done);
    if (q == cudaSuccess) {
      cudaEventDestroy(done);
      return true;
    }
    if (q != cudaErrorNotReady) {
      (void)cudaGetLastError();
      cudaEventDestroy(done);
      return false;
    }
    const float ms = std::chrono::duration<float, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    if (ms > timeout_ms) {
      if (timed_out) *timed_out = true;
      std::printf(
          "  WARN: launch exceeded %.0f ms — skip EventDestroy/DeviceReset "
          "(device %d)\n",
          timeout_ms, device);
      std::fflush(stdout);
      (void)cudaGetLastError();
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

const void *kernel_for_bytes(int bytes) {
  switch (bytes) {
    case 256: return (const void *)k_tma_mc_once<256>;
    case 512: return (const void *)k_tma_mc_once<512>;
    case 1024: return (const void *)k_tma_mc_once<1024>;
    case 2048: return (const void *)k_tma_mc_once<2048>;
    case 4096: return (const void *)k_tma_mc_once<4096>;
    case 8192: return (const void *)k_tma_mc_once<8192>;
    case 12288: return (const void *)k_tma_mc_once<12288>;
    case 16384: return (const void *)k_tma_mc_once<16384>;
    default: return nullptr;
  }
}

uint16_t full_mask(int n) {
  if (n >= 16) return 0xffffu;
  return static_cast<uint16_t>((1u << n) - 1u);
}

int occupancy_max_cluster(const void *fn, int dyn_smem) {
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
      potential >= 2) {
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

bool realloc_bufs(int device, uint8_t **d_g, McTimed **d_out) {
  *d_g = nullptr;
  *d_out = nullptr;
  if (cudaSetDevice(device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (cudaMalloc(d_g, kMaxBytes) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (cudaMalloc(d_out, sizeof(McTimed)) != cudaSuccess) {
    (void)cudaGetLastError();
    (void)cudaFree(*d_g);
    *d_g = nullptr;
    return false;
  }
  // Known pattern for payload match check
  if (cudaMemset(*d_g, 0x3c, kMaxBytes) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

bool try_ping(int cn, int device) {
  if (set_cluster_attrs((const void *)k_cluster_ping, 0) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  int *d_n = nullptr;
  if (cudaMalloc(&d_n, sizeof(int)) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  int expect = cn;
  void *argv[] = {&d_n, &expect};
  if (launch_cluster((const void *)k_cluster_ping, argv, cn, 0) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    cudaFree(d_n);
    return false;
  }
  bool to = false;
  if (!sync_with_timeout(kLaunchTimeoutMs, device, &to)) return false;
  int h = 0;
  if (cudaMemcpy(&h, d_n, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    (void)cudaGetLastError();
    cudaFree(d_n);
    return false;
  }
  cudaFree(d_n);
  return h == cn;
}

bool launch_one_sample(const void *fn, McTimed *d_out, const uint8_t *d_g,
                       int mode, int cn, uint16_t mask, int bytes, int device,
                       McTimed *h_out, bool *did_reset) {
  if (did_reset) *did_reset = false;
  if (set_cluster_attrs(fn, bytes) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (cudaMemset(d_out, 0, sizeof(McTimed)) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  const uint8_t *g = d_g;
  int md = mode;
  uint16_t msk = mask;
  McTimed *out = d_out;
  void *argv[] = {&g, &out, &md, &msk};
  if (launch_cluster(fn, argv, cn, bytes) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  bool to = false;
  if (!sync_with_timeout(kLaunchTimeoutMs, device, &to)) {
    if (did_reset) *did_reset = to;
    return false;
  }
  if (cudaMemcpy(h_out, d_out, sizeof(McTimed), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return h_out->ok != 0;
}

bool measure_mode(const void *fn, uint8_t **d_g, McTimed **d_out, int mode,
                  int cn, uint16_t mask, int bytes, int samples, int warmup,
                  int device, std::vector<McTimed> *results, const char *tag) {
  results->clear();
  results->reserve(samples);
  auto one = [&](McTimed *tmp) -> bool {
    bool reset = false;
    bool ok = launch_one_sample(fn, *d_out, *d_g, mode, cn, mask, bytes, device,
                                tmp, &reset);
    if (reset && !realloc_bufs(device, d_g, d_out)) return false;
    return ok;
  };
  for (int w = 0; w < std::max(1, warmup); ++w) {
    McTimed tmp{};
    if (!one(&tmp)) {
      std::printf("  %s: warmup failed (w=%d)\n", tag, w);
      std::fflush(stdout);
      return false;
    }
  }
  int fails = 0;
  for (int s = 0; s < samples; ++s) {
    McTimed tmp{};
    if (!one(&tmp)) {
      ++fails;
      if (fails >= 3 && results->empty()) return false;
      continue;
    }
    fails = 0;
    results->push_back(tmp);
  }
  return !results->empty();
}

CycleStats stats_u64_field(const std::vector<McTimed> &v,
                           uint64_t McTimed::*field, double overhead) {
  std::vector<uint64_t> raw;
  for (const auto &s : v) {
    if (!s.ok) continue;
    raw.push_back(s.*field);
  }
  return summarize_u64(raw, overhead);
}

CycleStats stats_ns_field(const std::vector<McTimed> &v,
                          uint64_t McTimed::*field) {
  std::vector<uint64_t> raw;
  for (const auto &s : v) {
    if (!s.ok) continue;
    raw.push_back(s.*field);
  }
  return summarize_u64(raw, 0.0);
}

void report(const char *name, const char *knob, const CycleStats &s,
            MetricSink &sink, const char *unit, const char *notes) {
  print_metric_line(name, s, notes);
  sink.add("tma_mc", name, knob, s.median, s.p90, s.mean, unit, notes);
}

}  // namespace

void run_tma_multicast_probes(const SuiteOptions &opt, MetricSink &sink) {
  std::printf("\n--- tma_multicast (simultaneity + size sweep) ---\n");
  print_kernel_source("src/probe_tma_multicast.cu",
                      "TMA multicast size, skew, and fan-out",
                      "clock64 issue/e2e timing; globaltimer destination skew",
                      "TMA multicast latency and simultaneity");
  std::printf(
      "  note: mask multicast primary; skew via %%globaltimer (not clock64)\n");
  std::printf(
      "  note: issue_cycles = pure bulk only (cvta + expect_tx OUTSIDE timer)\n");
  std::printf("  note: simultaneous iff skew_max_ns < %llu\n",
              (unsigned long long)kSimulThresholdNs);
  std::fflush(stdout);

  int cluster_launch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                                    opt.device));
  if (!cluster_launch) {
    std::printf("  SKIP: cluster launch not supported\n");
    sink.add("tma_mc", "skipped", "tma_multicast", 0, 0, 0, "cycles",
             "no cluster launch");
    return;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  if (prop.major < 9) {
    std::printf("  SKIP: need sm_90+ (device %d.%d)\n", prop.major, prop.minor);
    sink.add("tma_mc", "skipped", "tma_multicast", 0, 0, 0, "cycles",
             "cc < 9.0");
    return;
  }

  int max_n = opt.dsm_cluster_size > 0 ? opt.dsm_cluster_size : 0;
  if (max_n < 2) {
    max_n = occupancy_max_cluster((const void *)k_cluster_ping, 0);
  }
  if (max_n < 2) max_n = 2;
  if (max_n > kMaxCluster) max_n = kMaxCluster;

  int cn = 0;
  for (int try_n = max_n; try_n >= 2; --try_n) {
    std::printf("  probing cluster ping n=%d ...\n", try_n);
    std::fflush(stdout);
    if (try_ping(try_n, opt.device)) {
      cn = try_n;
      break;
    }
  }
  if (cn < 2) {
    std::printf("  SKIP: no working cluster size >= 2\n");
    sink.add("tma_mc", "skipped", "tma_multicast", 0, 0, 0, "cycles",
             "cluster ping failed");
    return;
  }
  std::printf("  working_cluster_n: %d\n", cn);
  std::fflush(stdout);

  uint8_t *d_g = nullptr;
  McTimed *d_out = nullptr;
  if (!realloc_bufs(opt.device, &d_g, &d_out)) {
    std::printf("  SKIP: cudaMalloc failed\n");
    return;
  }

  const int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 12288, 16384};
  std::vector<double> fit_bytes, fit_uni, fit_mc;  // for §2.5 slope/BPC

  // ---- Test 4: size sweep (unicast vs mcast) + Test 3: simultaneity ----
  std::printf("  --- size sweep + simultaneity (cluster_n=%d) ---\n", cn);
  for (int bytes : sizes) {
    const void *fn = kernel_for_bytes(bytes);
    if (!fn) continue;

    char tag_u[64], tag_m[64];
    std::snprintf(tag_u, sizeof(tag_u), "unicast_%dB", bytes);
    std::snprintf(tag_m, sizeof(tag_m), "mcast_%dB", bytes);

    std::vector<McTimed> hu, hm;
    if (!measure_mode(fn, &d_g, &d_out, static_cast<int>(McMode::UnicastCta),
                      cn, 0, bytes, opt.samples, opt.warmup, opt.device, &hu,
                      tag_u)) {
      std::printf("  SKIP %s\n", tag_u);
      continue;
    }
    if (!measure_mode(fn, &d_g, &d_out,
                      static_cast<int>(McMode::ClusterMaskFull), cn,
                      full_mask(cn), bytes, opt.samples, opt.warmup, opt.device,
                      &hm, tag_m)) {
      std::printf("  SKIP %s\n", tag_m);
      continue;
    }

    CycleStats u_e2e = stats_u64_field(hu, &McTimed::e2e_cycles, opt.clock64_overhead);
    CycleStats m_e2e = stats_u64_field(hm, &McTimed::e2e_cycles, opt.clock64_overhead);
    CycleStats m_iss = stats_u64_field(hm, &McTimed::issue_cycles, opt.clock64_overhead);
    CycleStats skew = stats_ns_field(hm, &McTimed::skew_max_ns);

    char notes[128];
    std::snprintf(notes, sizeof(notes), "cluster_n=%d bytes=%d ok_u=%zu ok_m=%zu",
                  cn, bytes, hu.size(), hm.size());

    char name[96];
    std::snprintf(name, sizeof(name), "unicast_%dB_e2e", bytes);
    report(name, "TMALoad e2e", u_e2e, sink, "cycles", notes);
    std::snprintf(name, sizeof(name), "mcast_%dB_e2e", bytes);
    report(name, "TMALoad mcast e2e", m_e2e, sink, "cycles", notes);
    std::snprintf(name, sizeof(name), "mcast_%dB_issue", bytes);
    report(name, "tma_mc issue", m_iss, sink, "cycles", notes);

    const double delta = m_e2e.median - u_e2e.median;
    std::snprintf(name, sizeof(name), "mcast_%dB_e2e_minus_unicast", bytes);
    std::printf("  %-44s %10.2f cycles  (mcast − unicast)\n", name, delta);
    sink.add("tma_mc", name, "gpgpu_tma_multicast_latency", delta, delta,
             delta, "cycles", notes);

    // Test 3: simultaneity
    std::snprintf(name, sizeof(name), "mcast_%dB_skew_max_ns", bytes);
    report(name, "tma_mc_simultaneity", skew, sink, "ns",
           "max |recv_done - issuer_done| via globaltimer");
    const double skew_cy =
        (opt.sm_clock_mhz > 0) ? (skew.median * 1e-3 * opt.sm_clock_mhz) : 0;
    std::snprintf(name, sizeof(name), "mcast_%dB_skew_max_cycles_est", bytes);
    std::printf("  %-44s %10.2f cycles  (from ns × SM MHz)\n", name, skew_cy);
    sink.add("tma_mc", name, "tma_mc_simultaneity", skew_cy, skew_cy, skew_cy,
             "cycles", "approx from globaltimer skew");

    const int simul = (skew.median < static_cast<double>(kSimulThresholdNs)) ? 1 : 0;
    std::snprintf(name, sizeof(name), "mcast_%dB_simultaneous", bytes);
    std::printf("  %-44s %10d  (1 if skew_max_ns < %llu)\n", name, simul,
                (unsigned long long)kSimulThresholdNs);
    sink.add("tma_mc", name, "tma_mc_simultaneity", simul, simul, simul, "bool",
             notes);

    // Payload match rate
    int match = 0;
    for (const auto &s : hm)
      if (s.ok && s.payload_match) ++match;
    const double match_rate =
        hm.empty() ? 0 : 100.0 * match / static_cast<double>(hm.size());
    std::snprintf(name, sizeof(name), "mcast_%dB_payload_match_pct", bytes);
    std::printf("  %-44s %10.1f %%\n", name, match_rate);
    sink.add("tma_mc", name, "tma_mc_correctness", match_rate, match_rate,
             match_rate, "percent", notes);

    if (bytes >= 1024 && bytes <= 16384) {
      fit_bytes.push_back(static_cast<double>(bytes));
      fit_uni.push_back(u_e2e.median);
      fit_mc.push_back(m_e2e.median);
    }
    std::fflush(stdout);
  }

  // End-to-end size slope, then bytes per cycle, for unicast and multicast.
  if (fit_bytes.size() >= 2) {
    auto ls_slope = [](const std::vector<double> &x,
                       const std::vector<double> &y) -> double {
      const int np = static_cast<int>(x.size());
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (int i = 0; i < np; ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
      }
      const double den = np * sxx - sx * sx;
      return (den != 0) ? (np * sxy - sx * sy) / den : 0;
    };
    const double slope_u = ls_slope(fit_bytes, fit_uni);
    const double slope_m = ls_slope(fit_bytes, fit_mc);
    const double per_kib_u = slope_u * 1024.0;
    const double per_kib_m = slope_m * 1024.0;
    const double bpc_u = (slope_u > 1e-12) ? (1.0 / slope_u) : 0;
    const double bpc_m = (slope_m > 1e-12) ? (1.0 / slope_m) : 0;
    std::printf("  --- 2.5 TMA size slope / BPC (LS fit 1K–16K) ---\n");
    std::printf("  %-44s %10.2f cycles/KiB\n", "tma_unicast_e2e_per_kib",
                per_kib_u);
    std::printf("  %-44s %10.2f cycles/KiB\n", "tma_mcast_e2e_per_kib",
                per_kib_m);
    std::printf("  %-44s %10.3f bytes/cycle\n", "tma_unicast_bpc_est", bpc_u);
    std::printf("  %-44s %10.3f bytes/cycle\n", "tma_mcast_bpc_est", bpc_m);
    sink.add("tma_mc", "tma_unicast_e2e_per_kib",
             "gpgpu_tma_load_completion_cycles_per_kib", per_kib_u, per_kib_u,
             per_kib_u,
             "cycles/KiB", "LS fit uni e2e");
    sink.add("tma_mc", "tma_mcast_e2e_per_kib",
             "gpgpu_tma_load_completion_cycles_per_kib", per_kib_m, per_kib_m,
             per_kib_m,
             "cycles/KiB", "LS fit mcast e2e");
    sink.add("tma_mc", "tma_unicast_bpc_est",
             "gpgpu_tma_load_completion_cycles_per_kib", bpc_u, bpc_u, bpc_u,
             "bytes/cycle", "1/slope unicast");
    sink.add("tma_mc", "tma_mcast_bpc_est",
             "gpgpu_tma_load_completion_cycles_per_kib", bpc_m, bpc_m, bpc_m,
             "bytes/cycle", "1/slope mcast; diagnostic, not a fabric knob");
  }

  // Fanout sweep at 4 KB
  std::printf("  --- fanout sweep (4KB mask mcast) ---\n");
  const void *fn4k = kernel_for_bytes(4096);
  for (int fan = 2; fan <= cn; fan *= 2) {
    char tag[64];
    std::snprintf(tag, sizeof(tag), "mcast_4k_fan%d", fan);
    std::vector<McTimed> h;
    if (!measure_mode(fn4k, &d_g, &d_out,
                      static_cast<int>(McMode::ClusterMaskFull), cn,
                      full_mask(fan), 4096, opt.samples,
                      std::max(1, opt.warmup / 2), opt.device, &h, tag)) {
      std::printf("  SKIP %s\n", tag);
      continue;
    }
    CycleStats e2e = stats_u64_field(h, &McTimed::e2e_cycles, opt.clock64_overhead);
    CycleStats skew = stats_ns_field(h, &McTimed::skew_max_ns);
    char notes[96];
    std::snprintf(notes, sizeof(notes),
                  "fanout=%d cluster_n=%d bytes=4096; fixed cluster size", fan,
                  cn);
    char name[96];
    std::snprintf(name, sizeof(name), "%s_e2e", tag);
    report(name, "gpgpu_tma_multicast_latency", e2e, sink, "cycles", notes);
    std::snprintf(name, sizeof(name), "%s_skew_max_ns", tag);
    report(name, "tma_mc_simultaneity", skew, sink, "ns", notes);
  }

  if (d_out) (void)cudaFree(d_out);
  if (d_g) (void)cudaFree(d_g);
  (void)cudaGetLastError();
  std::printf(
      "  interpret: low skew_max_ns ⇒ issuer & receivers complete nearly "
      "together\n");
  std::fflush(stdout);
}
