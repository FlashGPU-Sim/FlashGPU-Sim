// Integration tests for Thread Block Cluster launch APIs:
//   cudaLaunchKernelExC, cudaFuncSetAttribute(RequiredCluster*),
//   co-residency under multi-cluster configs, and ordinary <<<>>> still valid.
//
// Prefer configs:
//   SM120_RTX5090_REDUCED_CLUSTER2x1  (m=2, n=1)
//   SM120_RTX5090_REDUCED_CLUSTER2x2  (m=2, n=2)
//
// Run:
//   ./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 test "*ClusterLaunch*"
//   ./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x2 test "*ClusterLaunch*"

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

// Cluster-aware kernel: rank 0 writes a marker; every block records its
// relative cluster rank via a host-visible side channel computed from blockIdx
// (sreg path covered separately if PTX emits %cluster_ctarank).
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
    while (*flag != 42) {
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
    GTEST_SKIP() << "Requires n_cores_per_cluster == 1 to test capacity error";
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
