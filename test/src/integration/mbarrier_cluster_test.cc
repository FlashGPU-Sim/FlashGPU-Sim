// Cluster / remote mbarrier tests (mapa + NoC).
//
// Owner CTA inits a local mbarrier; peer CTA maps the barrier address with
// mapa.u64 and performs arrive / try_wait remotely.
//
// Requires: multi-SM cluster + NoC + mbarrier cluster enable
//   SM90_H200_REDUCED_CLUSTER4x4 (defaults)
//   or SM120 reduced with run-dir NoC overlay:
//     -gpgpu_cluster_noc_enable 1
//     -gpgpu_mbarrier_cluster_enable 1

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

namespace {

__device__ __forceinline__ unsigned long long
mapa_u64_shared(void *local, unsigned rank) {
  unsigned long long out = 0;
  unsigned long long in = reinterpret_cast<unsigned long long>(local);
  asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(out) : "l"(in), "r"(rank));
  return out;
}

__device__ __forceinline__ void mbarrier_init_local(unsigned long long *bar,
                                                    unsigned expected) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(p),
               "r"(expected));
}

__device__ __forceinline__ void mbarrier_arrive_remote(unsigned long long bar_g,
                                                       unsigned count) {
  // Generic address of remote barrier (from mapa.u64).
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"l"(bar_g),
               "r"(count));
}

__device__ __forceinline__ void
mbarrier_try_wait_remote(unsigned long long bar_g, int parity) {
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT_REMOTE:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@!P1 bra.uni LAB_WAIT_REMOTE;\n"
               "}\n" ::"l"(bar_g),
               "r"(parity));
}

// Rank 0: init bar, try_wait for arrive from rank 1.
// Rank 1: mapa rank0 bar, arrive(1).
__global__ void remote_arrive_owner_wait_kernel(int *out) {
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (rank == 0 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    out[2] = 1;
    __threadfence_system();
  }

  if (rank == 1 && tid == 0) {
    while (out[2] == 0) {
    }
    unsigned long long remote_bar = mapa_u64_shared(&bar, 0);
    mbarrier_arrive_remote(remote_bar, 1);
    out[1] = 1;
    __threadfence_system();
  }

  if (rank == 0 && tid == 0) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT_OWNER2:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@!P1 bra.uni LAB_WAIT_OWNER2;\n"
                 "}\n" ::"r"(p),
                 "r"(0));
    out[0] = 42;
  }
}

// Rank 0 arrives locally; rank 1 remote try_waits on rank 0's bar.
__global__ void remote_try_wait_kernel(int *out) {
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (rank == 0 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    out[2] = 1;
    __threadfence_system();
  }

  if (rank == 1 && tid == 0) {
    while (out[2] == 0) {
    }
    unsigned long long remote_bar = mapa_u64_shared(&bar, 0);
    // Start waiting before owner arrives (blocks until remote arrive).
    // Use a second flag so owner knows peer is ready to wait.
    out[3] = 1;
    __threadfence_system();
    mbarrier_try_wait_remote(remote_bar, 0);
    out[1] = 99;
  }

  if (rank == 0 && tid == 0) {
    while (out[3] == 0) {
    }
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(p),
                 "r"(1));
    out[0] = 1;
  }
}

class MbarrierClusterTest : public ::testing::Test {};

TEST_F(MbarrierClusterTest, RemoteArriveUnblocksOwner) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)remote_arrive_owner_wait_kernel, grid, block,
                cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h[4] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);

  EXPECT_EQ(h[0], 42) << "owner try_wait did not complete";
  EXPECT_EQ(h[1], 1) << "peer arrive did not run";
}

TEST_F(MbarrierClusterTest, RemoteTryWaitSeesLocalArrive) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)remote_try_wait_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h[4] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);

  EXPECT_EQ(h[0], 1) << "owner arrive did not run";
  EXPECT_EQ(h[1], 99) << "remote try_wait did not complete";
}

// phase must depend on remote tx, not just arrive.
//
//   rank 0: init(1) → wait expect_tx → local arrive → try_wait
//   rank 1: remote expect_tx → owner arrives → remote complete_tx only
//
// If expect_tx is a no-op, local arrive satisfies the barrier and try_wait
// returns before complete_tx (peer writes out[1]=-1). If complete_tx is a
// no-op, try_wait hangs. No remote arrive — that would hide both bugs.
__device__ __forceinline__ void
mbarrier_expect_tx_remote(unsigned long long bar_g, unsigned bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"l"(bar_g),
               "r"(bytes));
}

__device__ __forceinline__ void
mbarrier_complete_tx_remote(unsigned long long bar_g, unsigned bytes) {
  asm volatile(
      "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n" ::"l"(bar_g),
      "r"(bytes));
}

__global__ void remote_expect_complete_kernel(int *out) {
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (rank == 0 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    out[2] = 1;  // init published
    __threadfence_system();
    while (out[3] == 0) {
    }
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(p),
                 "r"(1));
    out[4] = 1;  // local arrive done; try_wait must still need complete_tx
    __threadfence_system();
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT_TX:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@!P1 bra.uni LAB_WAIT_TX;\n"
                 "}\n" ::"r"(p),
                 "r"(0));
    out[0] = 77;
    __threadfence_system();
  }

  if (rank == 1 && tid == 0) {
    while (out[2] == 0) {
    }
    unsigned long long remote_bar = mapa_u64_shared(&bar, 0);
    mbarrier_expect_tx_remote(remote_bar, 32);
    out[3] = 1;  // expect_tx issued; owner may now arrive
    __threadfence_system();
    while (out[4] == 0) {
    }
    // Owner has arrived. If expect_tx was a no-op, try_wait is already
    // satisfied and out[0] becomes 77 without complete_tx — wait a bit
    // so that write is visible before we decide.
    for (int i = 0; i < 10000 && out[0] == 0; i++) {
    }
    if (out[0] == 77) {
      out[1] = -1;
      __threadfence_system();
      return;
    }
    mbarrier_complete_tx_remote(remote_bar, 32);
    while (out[0] == 0) {
    }
    out[1] = 1;
    __threadfence_system();
  }
}

TEST_F(MbarrierClusterTest, RemoteExpectAndCompleteTx) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 5 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 5 * sizeof(int)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)remote_expect_complete_kernel, grid, block,
                cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h[5] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);

  EXPECT_NE(h[1], -1) << "try_wait completed after local arrive alone — "
                         "remote expect_tx did not arm tx (no-op)";
  EXPECT_EQ(h[0], 77) << "owner try_wait did not complete after remote "
                         "complete_tx";
  EXPECT_EQ(h[1], 1) << "peer remote expect_tx/complete_tx did not finish";
}

}  // namespace
