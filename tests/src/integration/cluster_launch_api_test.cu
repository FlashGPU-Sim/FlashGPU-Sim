// Integration tests for Thread Block Cluster launch APIs:
//   cudaLaunchKernelExC, cudaFuncSetAttribute(RequiredCluster*),
//   co-residency under multi-cluster configs, and ordinary <<<>>> still valid.
//
// Multi-SM cases skip on SM120_RTX5090 (one SM per GPC, DSM off).
// The shipped config that can run them is SM90_H200_CLUSTER132.
// The case that requires exactly one SM per GPC runs on SM120_RTX5090.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

// Simple kernel: each block writes blockIdx.x into out[blockIdx.x].
__global__ void cluster_launch_write_blockidx(int *out, int n) {
  if (threadIdx.x == 0 && blockIdx.x < n) {
    out[blockIdx.x] = (int)blockIdx.x;
  }
}

// Cluster-aware kernel: every block records that it ran. Rank and cluster
// id registers are checked in cluster_sreg_test.cu.
__global__ void cluster_launch_mark_ready(int *ready, int n) {
  if (threadIdx.x == 0 && blockIdx.x < n) {
    atomicAdd(ready, 1);
  }
}

// One-producer peer style: CTA 0 writes a value into a shared flag via global
// (no TMA). Used only to prove both blocks ran under a cluster launch.
__global__ void cluster_launch_two_cta_ping(int *flag) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 0) {
    *flag = 42;
  } else {
    // Spin until producer wrote (cluster co-residency not required for this
    // simple flag; co-residency is validated via OneProducer TMA tests).
    // volatile keeps the load in the loop. A plain load is hoisted, and the
    // remaining self-branch has no path to the kernel exit.
    while (*reinterpret_cast<volatile int *>(flag) != 42) {
    }
  }
}

class ClusterLaunchApiTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(ClusterLaunchApiTest, DeviceAttr_ClusterLaunch) {
  int value = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&value, cudaDevAttrClusterLaunch, 0),
            cudaSuccess);
  EXPECT_EQ(value, 1);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterDim2_Succeeds) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  constexpr int N = 2;
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, N * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, N * sizeof(int)), cudaSuccess);

  int n = N;
  void *args[] = {&d_out, &n};

  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_write_blockidx, dim3(N), dim3(32),
      dim3(2, 1, 1), args);
  ASSERT_EQ(err, cudaSuccess) << "cudaLaunchKernelExC failed: "
                              << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> h(N, -1);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 0);
  EXPECT_EQ(h[1], 1);

  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_GridNotMultipleOfCluster_Fails) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 3 * sizeof(int)), cudaSuccess);
  int n = 3;
  void *args[] = {&d_out, &n};

  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_write_blockidx, dim3(3), dim3(32),
      dim3(2, 1, 1), args);
  // Grid 3 is not a multiple of cluster 2.
  EXPECT_NE(err, cudaSuccess);
  // Clear sticky error for subsequent tests in the same process.
  (void)cudaGetLastError();

  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterLargerThanPhysical_Fails) {
  // On configs with n_cores_per_cluster == 1, cluster size 2 must fail.
  const auto topo = flash_test::read_gpgpusim_topology();
  if (!topo.found_config || topo.n_cores_per_cluster >= 2) {
    SKIP_WITH_REASON(
        "Requires n_cores_per_cluster == 1 to test capacity error");
  }

  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 2 * sizeof(int)), cudaSuccess);
  int n = 2;
  void *args[] = {&d_out, &n};

  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_write_blockidx, dim3(2), dim3(32),
      dim3(2, 1, 1), args);
  EXPECT_NE(err, cudaSuccess);
  (void)cudaGetLastError();
  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_CoResidencyTwoCtasRun) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  int *d_flag = nullptr;
  ASSERT_EQ(cudaMalloc(&d_flag, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_flag, 0, sizeof(int)), cudaSuccess);

  void *args[] = {&d_flag};
  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_two_cta_ping, dim3(2), dim3(32),
      dim3(2, 1, 1), args);
  ASSERT_EQ(err, cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_flag, sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h, 42);
  cudaFree(d_flag);
}

TEST_F(ClusterLaunchApiTest, OrdinaryLaunch_StillWorks) {
  constexpr int N = 2;
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, N * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, N * sizeof(int)), cudaSuccess);

  int n = N;
  cluster_launch_write_blockidx<<<N, 32>>>(d_out, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> h(N, -1);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 0);
  EXPECT_EQ(h[1], 1);
  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, FuncSetAttribute_RequiredClusterDims) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  // Set required cluster dims on the function, then use ordinary <<<>>>.
  ASSERT_EQ(cudaFuncSetAttribute((const void *)cluster_launch_write_blockidx,
                                 cudaFuncAttributeRequiredClusterWidth, 2),
            cudaSuccess);
  ASSERT_EQ(cudaFuncSetAttribute((const void *)cluster_launch_write_blockidx,
                                 cudaFuncAttributeRequiredClusterHeight, 1),
            cudaSuccess);
  ASSERT_EQ(cudaFuncSetAttribute((const void *)cluster_launch_write_blockidx,
                                 cudaFuncAttributeRequiredClusterDepth, 1),
            cudaSuccess);

  constexpr int N = 2;
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, N * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, N * sizeof(int)), cudaSuccess);

  int n = N;
  cluster_launch_write_blockidx<<<N, 32>>>(d_out, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> h(N, -1);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 0);
  EXPECT_EQ(h[1], 1);

  // Reset required dims so other tests are not affected (set to 0 = clear
  // is not standard; re-set width 0 may not clear explicit flag). Leave as-is;
  // subsequent Ex launches still override via attributes.
  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_MultiClusterConfig_TwoCtasComplete) {
  // On multi-cluster topology, cluster launch of size 2 must still complete
  // (co-residency). Without Ex, RR can split CTAs across physical clusters.
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);

  int n = 2;
  void *args[] = {&d_ready, &n};
  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_mark_ready, dim3(2), dim3(32), dim3(2, 1, 1),
      args);
  ASSERT_EQ(err, cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h = 0;
  ASSERT_EQ(
      cudaMemcpy(&h, d_ready, sizeof(int), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(h, 2);
  cudaFree(d_ready);
}

// --- four CTAs in one thread-block cluster ---

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterDim4_Succeeds) {
  // Functional check: TB cluster size 4 issues and all CTAs complete when
  // physical packing m >= 4.
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);

  constexpr int N = 4;
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, N * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, N * sizeof(int)), cudaSuccess);

  int n = N;
  void *args[] = {&d_out, &n};

  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_write_blockidx, dim3(N), dim3(32),
      dim3(4, 1, 1), args);
  ASSERT_EQ(err, cudaSuccess) << "cudaLaunchKernelExC clusterDim=4 failed: "
                              << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> h(N, -1);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h[i], i) << "block " << i;
  }
  cudaFree(d_out);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterDim4_AllCtasMarkReady) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);

  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);

  int n = 4;
  void *args[] = {&d_ready, &n};
  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_mark_ready, dim3(4), dim3(32), dim3(4, 1, 1),
      args);
  ASSERT_EQ(err, cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h = 0;
  ASSERT_EQ(
      cudaMemcpy(&h, d_ready, sizeof(int), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(h, 4);
  cudaFree(d_ready);
}

__device__ __forceinline__ void cluster_sync_ptx() {
  asm volatile("barrier.cluster.arrive;\n");
  asm volatile("barrier.cluster.wait;\n");
}

// Every thread in every CTA of the TB cluster participates, then CTA 0
// of each cluster records that the barrier released.
__global__ void cluster_launch_cluster_sync_mark(int *ready, int n) {
  cluster_sync_ptx();
  if (threadIdx.x == 0 && blockIdx.x < n) {
    atomicAdd(ready, 1);
  }
}

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterDim2_ClusterSync_TwoWaves) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);

  const auto topo = flash_test::read_gpgpusim_topology();
  unsigned n_sms = topo.total_sms ? topo.total_sms
                                  : topo.n_clusters * topo.n_cores_per_cluster;
  if (n_sms < 2 || n_sms > 256)
    n_sms = 8;
  // Even CTA count, more CTAs than SMs, but cheap enough for CI.
  int n = static_cast<int>((n_sms + 2) & ~1u);
  if (n < 8) n = 8;
  if (n > 32) n = 32;

  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);

  void *args[] = {&d_ready, &n};
  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_cluster_sync_mark, dim3(n), dim3(32),
      dim3(2, 1, 1), args);
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_ready, sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h, n);
  cudaFree(d_ready);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_HeteroGpc_ClusterDim2_ManyClustersSync) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  SKIP_IF_NOT_HETERO_GPC();

  const auto topo = flash_test::read_gpgpusim_topology();
  unsigned n_sms = topo.total_sms ? topo.total_sms : 5;
  if (n_sms < 2 || n_sms > 256)
    n_sms = 5;
  int n = static_cast<int>((n_sms * 2) & ~1u);
  if (n < 8) n = 8;
  if (n > 32) n = 32;

  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);

  void *args[] = {&d_ready, &n};
  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_cluster_sync_mark, dim3(n), dim3(32),
      dim3(2, 1, 1), args);
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_ready, sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h, n);
  cudaFree(d_ready);
}

TEST_F(ClusterLaunchApiTest, ExLaunch_ClusterLargerThanPhysical_m_Fails) {
  // On any topology, product(clusterDim) > m must be rejected.
  const auto topo = flash_test::read_gpgpusim_topology();
  if (!topo.found_config) {
    SKIP_WITH_REASON("Could not parse gpgpusim.config topology");
  }
  const unsigned m = topo.n_cores_per_cluster;
  const unsigned bad = m + 1;

  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, bad * sizeof(int)), cudaSuccess);
  int n = static_cast<int>(bad);
  void *args[] = {&d_out, &n};

  cudaError_t err = flash_test::launch_kernel_with_cluster(
      (const void *)cluster_launch_write_blockidx, dim3(bad), dim3(32),
      dim3(bad, 1, 1), args);
  EXPECT_NE(err, cudaSuccess)
      << "cluster size " << bad << " should exceed physical m=" << m;
  (void)cudaGetLastError();
  cudaFree(d_out);
}

__device__ __forceinline__ void cluster_bar_arrive() {
  asm volatile("barrier.cluster.arrive;\n");
}

__device__ __forceinline__ void cluster_bar_wait() {
  asm volatile("barrier.cluster.wait;\n");
}

// Rank 0 arrives and does not wait. Rank 1 arrives and waits. The wait must
// release without rank 0 also waiting.
__global__ void split_arrive_wait_kernel(volatile int *flag) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) flag[0] = 1;
    cluster_bar_arrive();
    if (threadIdx.x == 0) {
      while (flag[1] == 0) {
      }
    }
    return;
  }
  cluster_bar_arrive();
  cluster_bar_wait();
  if (threadIdx.x == 0) flag[1] = 1;
}

// Round 1 is a normal arrive+wait. Round 2 must not release on round 1's
// arrival: rank 1 samples the release flag only after rank 0 has arrived
// for round 2 and before rank 1 arrives again.
__global__ void second_round_needs_new_arrive_kernel(volatile int *flag) {
  cluster_bar_arrive();
  cluster_bar_wait();
  if (blockIdx.x == 0) {
    cluster_bar_arrive();
    if (threadIdx.x == 0) flag[0] = 1;
    cluster_bar_wait();
    if (threadIdx.x == 0) flag[1] = 1;
    return;
  }
  if (threadIdx.x == 0) {
    while (flag[0] == 0) {
    }
    // Rank 0 has arrived for round 2. A sticky round-1 arrival would let
    // its wait finish during this local delay.
    const unsigned long long start = clock64();
    while (clock64() - start < 4000ull) {
    }
    flag[2] = flag[1];
  }
  cluster_bar_arrive();
  cluster_bar_wait();
}

// Rank 1 is parked in wait. Rank 0 never arrives and then exits. The wait
// must release.
__global__ void peer_exit_unblocks_wait_kernel(volatile int *flag) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      const unsigned long long start = clock64();
      while (clock64() - start < 20000ull) {
      }
    }
    return;
  }
  cluster_bar_arrive();
  cluster_bar_wait();
  if (threadIdx.x == 0) flag[0] = 1;
}

TEST_F(ClusterLaunchApiTest, SplitArriveWait_ReleasesWithoutPeerWait) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *flag = nullptr;
  ASSERT_EQ(cudaMalloc(&flag, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(flag, 0, 4 * sizeof(int)), cudaSuccess);
  void *args[] = {&flag};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)split_arrive_wait_kernel, dim3(2), dim3(32),
                dim3(2, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h[2] = {};
  ASSERT_EQ(cudaMemcpy(h, flag, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 1);
  EXPECT_EQ(h[1], 1);
  cudaFree(flag);
}

TEST_F(ClusterLaunchApiTest, SecondRoundWait_NeedsNewArrive) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *flag = nullptr;
  ASSERT_EQ(cudaMalloc(&flag, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(flag, 0, 4 * sizeof(int)), cudaSuccess);
  void *args[] = {&flag};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)second_round_needs_new_arrive_kernel, dim3(2),
                dim3(32), dim3(2, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h[3] = {};
  ASSERT_EQ(cudaMemcpy(h, flag, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 1);
  EXPECT_EQ(h[1], 1);
  EXPECT_EQ(h[2], 0) << "round 2 released before the peer arrived again";
  cudaFree(flag);
}

TEST_F(ClusterLaunchApiTest, PeerExit_UnblocksWaiter) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *flag = nullptr;
  ASSERT_EQ(cudaMalloc(&flag, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(flag, 0, sizeof(int)), cudaSuccess);
  void *args[] = {&flag};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)peer_exit_unblocks_wait_kernel, dim3(2),
                dim3(32), dim3(2, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h = 0;
  ASSERT_EQ(cudaMemcpy(&h, flag, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h, 1);
  cudaFree(flag);
}

// Runtime predicate. A literal-false C++ branch is deleted by nvcc and never
// issues the barrier, so the empty mask would not be tested.
__device__ __forceinline__ void cluster_bar_arrive_if(int cond) {
  asm volatile(
      "{\n\t"
      ".reg .pred Poff;\n\t"
      "setp.ne.s32 Poff, %0, 0;\n\t"
      "@Poff barrier.cluster.arrive;\n\t"
      "}\n" ::"r"(cond));
}

__device__ __forceinline__ void cluster_bar_wait_if(int cond) {
  asm volatile(
      "{\n\t"
      ".reg .pred Poff;\n\t"
      "setp.ne.s32 Poff, %0, 0;\n\t"
      "@Poff barrier.cluster.wait;\n\t"
      "}\n" ::"r"(cond));
}

// Rank 0's arrive is predicated off for every lane, then it delays, sets
// flag[0], and does a real arrive+wait. Rank 1's wait must observe flag[0]
// already set. An empty arrive that still credits the phase releases rank 1
// during the delay, and flag[1] stays 0. Both sides still reach a real
// arrive, so the launch returns either way.
__global__ void pred_off_arrive_kernel(volatile int *flag, int cond) {
  if (blockIdx.x == 0) {
    cluster_bar_arrive_if(cond);
    if (threadIdx.x == 0) {
      const unsigned long long start = clock64();
      while (clock64() - start < 8000ull) {
      }
      flag[0] = 1;
    }
    cluster_bar_arrive();
    cluster_bar_wait();
    return;
  }
  cluster_bar_arrive();
  cluster_bar_wait();
  if (threadIdx.x == 0) flag[1] = flag[0];
}

// An empty wait must not park. The flag store does not depend on that wait.
// A later real barrier checks the warp can still arrive.
__global__ void pred_off_wait_kernel(volatile int *flag, int cond) {
  cluster_bar_wait_if(cond);
  if (threadIdx.x == 0) flag[blockIdx.x] = 1;
  cluster_bar_arrive();
  cluster_bar_wait();
}

TEST_F(ClusterLaunchApiTest, PredOffArrive_DoesNotCreditPhase) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *flag = nullptr;
  ASSERT_EQ(cudaMalloc(&flag, 2 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(flag, 0, 2 * sizeof(int)), cudaSuccess);
  int cond = 0;
  void *args[] = {&flag, &cond};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)pred_off_arrive_kernel, dim3(2), dim3(32),
                dim3(2, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h[2] = {};
  ASSERT_EQ(cudaMemcpy(h, flag, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 1);
  EXPECT_EQ(h[1], 1) << "empty arrive released the peer before the real arrive";
  cudaFree(flag);
}

TEST_F(ClusterLaunchApiTest, PredOffWait_FallsThrough) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *flag = nullptr;
  ASSERT_EQ(cudaMalloc(&flag, 2 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(flag, 0, 2 * sizeof(int)), cudaSuccess);
  int cond = 0;
  void *args[] = {&flag, &cond};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)pred_off_wait_kernel, dim3(2), dim3(32),
                dim3(2, 1, 1), args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h[2] = {};
  ASSERT_EQ(cudaMemcpy(h, flag, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(h[0], 1);
  EXPECT_EQ(h[1], 1);
  cudaFree(flag);
}
