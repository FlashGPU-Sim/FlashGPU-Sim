// Remote mbarrier arrive. Rank 0 arrives on rank 1's barrier through mapa.
// Rank 1 waits on its own local barrier. try_wait on the mapa address is an
// illegal instruction on Hopper.
//
// Log lines match the L12 section of the H200 mbarrier suite.

#include "common.cuh"
#include "suite_main.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace {

__device__ __forceinline__ int mbar_wait_bounded_local(uint32_t p,
                                                       unsigned parity) {
  constexpr int kL12MaxSpins = 1000000;
#pragma unroll 1
  for (int i = 0; i < kL12MaxSpins; ++i) {
    int ok = 0;
    asm volatile("{\n"
                 ".reg .pred pred;\n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
                 "pred, [%1], %2;\n"
                 "selp.s32 %0, 1, 0, pred;\n"
                 "}\n"
                 : "=r"(ok)
                 : "r"(p), "r"(parity)
                 : "memory");
    if (ok) return 1;
  }
  return 0;
}

__global__ void k_remote_mbarrier(uint64_t *arrive_s, uint64_t *wait_s,
                                  uint64_t *e2e_ns, int n) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  __shared__ __align__(8) unsigned long long bar;
  __shared__ uint64_t done_ns;

  const uint32_t local_p = smem_addr_u32(&bar);
  const int peer_rank = 1;
  uint32_t peer_p = 0;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(peer_p)
               : "r"(local_p), "r"(peer_rank));

  for (int i = 0; i < n; ++i) {
    if (rank == 1 && threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(local_p),
                   "r"(1)
                   : "memory");
      done_ns = 0;
    }
    cluster.sync();

    uint64_t start_ns = 0;
    if (rank == 0 && threadIdx.x == 0) {
      start_ns = globaltimer_now();
      const uint64_t t0 = clock64_now();
      asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(
                       peer_p)
                   : "memory");
      const uint64_t t1 = clock64_now();
      arrive_s[i] = t1 - t0;
    }
    if (rank == 1 && threadIdx.x == 0) {
      const uint64_t t0 = clock64_now();
      const int ok = mbar_wait_bounded_local(local_p, 0);
      const uint64_t t1 = clock64_now();
      wait_s[i] = ok ? (t1 - t0) : 0;
      done_ns = ok ? globaltimer_now() : 0;
    }
    cluster.sync();
    if (rank == 0 && threadIdx.x == 0) {
      uint64_t *remote_done = cluster.map_shared_rank(&done_ns, 1);
      e2e_ns[i] = (*remote_done >= start_ns) ? (*remote_done - start_ns) : 0;
    }
    cluster.sync();
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < n; ++i) {
      arrive_s[i] = wait_s[i] = e2e_ns[i] = 0;
    }
  }
#endif
}

cudaError_t launch_cluster2(const void *fn, void **args, int block_dim) {
  cudaError_t e = cudaFuncSetAttribute(
      fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (e != cudaSuccess) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, 1, 1);
  cfg.blockDim = dim3(block_dim, 1, 1);
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return cudaLaunchKernelExC(&cfg, fn, args);
}

void run_remote_mbarrier(const SuiteOptions &opt, MetricSink &sink) {
  std::printf("\n--- mbarrier (pure, no TMA) ---\n");
  print_kernel_source("src/probe_mbarrier.cu", "remote arrive only",
                      "clock64 around arrive/try_wait; globaltimer remote e2e",
                      "gpgpu_mbarrier_arrive_latency / trywait / remote hop");

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  if (prop.major < 9) {
    std::printf("  SKIP L12 remote mbarrier: need CC >= 9.0 (device %d.%d)\n",
                prop.major, prop.minor);
    sink.add("mbarrier", "mbarrier_remote_arrive",
             "gpgpu_mbarrier_arrive_latency", 0, 0, 0, "cycles",
             "L12 Hopper-only");
    return;
  }

  uint64_t *d_a = nullptr, *d_w = nullptr, *d_e = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(uint64_t) * opt.samples));
  CUDA_CHECK(cudaMalloc(&d_w, sizeof(uint64_t) * opt.samples));
  CUDA_CHECK(cudaMalloc(&d_e, sizeof(uint64_t) * opt.samples));
  void *args[] = {&d_a, &d_w, &d_e, const_cast<int *>(&opt.samples)};
  const void *fn = reinterpret_cast<const void *>(k_remote_mbarrier);
  cudaError_t e = launch_cluster2(fn, args, 32);
  if (e != cudaSuccess) {
    std::printf("  SKIP L12 remote mbarrier: %s\n", cudaGetErrorString(e));
    sink.add("mbarrier", "mbarrier_remote_arrive",
             "gpgpu_mbarrier_arrive_latency", 0, 0, 0, "cycles",
             "L12 launch failed");
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_e));
    return;
  }
  if (!sync_device_timeout(15.f, opt.device, "L12 remote mbarrier")) {
    sink.add("mbarrier", "mbarrier_remote_arrive",
             "gpgpu_mbarrier_arrive_latency", 0, 0, 0, "cycles",
             "L12 sync failed; no DeviceReset (process will exit)");
    return;
  }

  std::vector<uint64_t> ha(opt.samples), hw(opt.samples), he(opt.samples);
  CUDA_CHECK(cudaMemcpy(ha.data(), d_a, sizeof(uint64_t) * opt.samples,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hw.data(), d_w, sizeof(uint64_t) * opt.samples,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(he.data(), d_e, sizeof(uint64_t) * opt.samples,
                        cudaMemcpyDeviceToHost));
  CycleStats sa = summarize_u64(ha, opt.clock64_overhead);
  CycleStats sw = summarize_u64(hw, opt.clock64_overhead);
  CycleStats se = summarize_u64(he, 0.0);
  print_metric_line("mbarrier_remote_arrive", sa,
                    "L12 mapa arrive; producer alive");
  print_metric_line("mbarrier_remote_trywait_local", sw,
                    "L12 owner try_wait after remote arrive");
  print_metric_line("mbarrier_remote_e2e_ns", se,
                    "issuer start to owner completion; synchronized globaltimer");
  sink.add("mbarrier", "mbarrier_remote_arrive",
           "gpgpu_mbarrier_arrive_latency", sa.median, sa.p90, sa.mean,
           "cycles", "L12 remote arrive via mapa");
  sink.add("mbarrier", "mbarrier_remote_trywait_local",
           "gpgpu_mbarrier_trywait_latency", sw.median, sw.p90, sw.mean,
           "cycles",
           "L12 owner local wait; PTX forbids wait on cluster-space bar");
  const double mhz = opt.sm_clock_mhz > 0 ? opt.sm_clock_mhz : 1785.0;
  const double e2e_cycles = se.median * mhz * 1e-3;
  sink.add("mbarrier", "mbarrier_remote_e2e_cycles",
           "gpgpu_mbarrier_remote_hop_latency / DSM fabric", e2e_cycles,
           e2e_cycles, e2e_cycles, "cycles",
           "globaltimer end-to-end ns converted with measured SM MHz");
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_e));
}

}  // namespace

int main(int argc, char **argv) {
  return run_suite(argc, argv, "mbarrier", run_remote_mbarrier);
}
