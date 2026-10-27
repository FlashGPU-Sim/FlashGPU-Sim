// Cluster / remote mbarrier tests (mapa + NoC).
//
// Owner CTA inits a local mbarrier; peer CTA maps the barrier address with
// mapa.u64 and performs arrive / try_wait remotely.
//
// Requires a GPC with at least two SMs, DSM on, and cluster mbarrier on.
// Those cases skip on SM120_RTX5090. SM90_H200_CLUSTER132 can run them.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// Defined by the simulator libcudart. The test binary links the system
// libcudart, so this stays weak and binds when the simulator library loads.
extern "C" __attribute__((weak)) void flashgpu_set_cluster_fatal_observer(
    void (*observer)(const char *message));

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
// Volatile keeps handshake loads inside the polling loops, as in the finite
// peer-stream test below. It is not a replacement for memory ordering fences.
template <bool Map32>
__global__ void remote_arrive_owner_wait_kernel(volatile int *out) {
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
    if constexpr (Map32) {
      const unsigned local = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
      unsigned remote;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                   : "=r"(remote) : "r"(local));
      asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                   :: "r"(remote) : "memory");
    } else {
      unsigned long long remote_bar = mapa_u64_shared(&bar, 0);
      mbarrier_arrive_remote(remote_bar, 1);
    }
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
__global__ void remote_try_wait_kernel(volatile int *out) {
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
  for (const void *kernel : {(const void *)remote_arrive_owner_wait_kernel<false>,
                            (const void *)remote_arrive_owner_wait_kernel<true>}) {
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                kernel, grid, block,
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

template <bool OmitExpect>
__global__ void remote_expect_complete_kernel(volatile int *out) {
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
    if constexpr (!OmitExpect)
      mbarrier_expect_tx_remote(remote_bar, 32);
    out[3] = 1;  // expect_tx issued; owner may now arrive
    __threadfence_system();
    while (out[4] == 0) {
    }
    // Owner has arrived. If expect_tx was a no-op, try_wait is already
    // satisfied and out[0] becomes 77 without complete_tx — wait a bit
    // so that write is visible before we decide.
    // Bound the observation in device cycles, not 10000 serialized global
    // loads (millions of simulated cycles). This is a functional check,
    // not a measurement of memory throughput or barrier latency.
    // 4096-cycle window; lengthen if testing higher-latency presets.
    const unsigned long long observe_start = clock64();
    while (out[0] == 0 && clock64() - observe_start < 4096) {}
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
  // Also omit expect_tx deliberately: the bounded observation must detect
  // premature completion, not merely make the positive case finish faster.
  for (bool omit_expect : {false, true}) {
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 5 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 5 * sizeof(int)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  const void *kernel = omit_expect
      ? (const void *)remote_expect_complete_kernel<true>
      : (const void *)remote_expect_complete_kernel<false>;
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                kernel, grid, block, cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h[5] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);

  EXPECT_EQ(h[0], 77) << "owner try_wait did not complete";
  EXPECT_EQ(h[1], omit_expect ? -1 : 1)
      << "premature completion detection disagrees with expect_tx injection";
  }
}

// try_wait 4th operand is the PTX timeout / suspend hint in nanoseconds. Dest pred is true
// only if the waited phase completed; false if the hint expires first.
__device__ __forceinline__ unsigned
mbarrier_try_wait_parity_timeout(uint32_t bar_ptr, unsigned parity,
                                 unsigned timeout) {
  unsigned done = 0;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2, %3;\n"
               "selp.u32 %0, 1, 0, P1;\n"
               "}\n"
               : "=r"(done)
               : "r"(bar_ptr), "r"(parity), "r"(timeout));
  return done;
}

__global__ void mbar_timeout_expires_kernel(unsigned *out) {
  __shared__ unsigned long long bar;
  if (threadIdx.x != 0)
    return;
  mbarrier_init_local(&bar, /*expected=*/2);
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  const unsigned long long start = clock64();
  const unsigned done =
      mbarrier_try_wait_parity_timeout(p, /*parity=*/0, /*timeout=*/1000);
  const unsigned long long end = clock64();
  out[0] = done;
  out[1] = static_cast<unsigned>(end - start);
}

__global__ void mbar_timeout_phase_done_kernel(unsigned *out) {
  __shared__ unsigned long long bar;
  if (threadIdx.x != 0)
    return;
  mbarrier_init_local(&bar, /*expected=*/1);
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(p),
               "r"(1));
  out[0] = mbarrier_try_wait_parity_timeout(p, /*parity=*/0, /*timeout=*/64);
}

TEST_F(MbarrierClusterTest, TryWaitTimeoutExpires_PredFalse) {
  unsigned *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 2 * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0xFFu, 2 * sizeof(unsigned)), cudaSuccess);
  dim3 grid(1), block(32), cluster(1, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)mbar_timeout_expires_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned h[2] = {0xFFu, 0};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  EXPECT_EQ(h[0], 0u) << "timed try_wait must set dest pred false when the "
                      "phase never completes";
  // Simulator policy honors the hint; PTX permits hardware to treat it as a
  // hint rather than an exact duration. Only assert this policy in the sim.
  if (getenv("FLASHGPU_SIM_CLOCK_FROM_PROP")) {
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    const unsigned hint_cycles = (prop.clockRate + 999) / 1000;
    EXPECT_GE(h[1], hint_cycles);
    EXPECT_LE(h[1], hint_cycles + 256u);
  }
}

// Bare peer smem spin: no mbarrier interest. Must abort, not hang.
__global__ void bare_peer_spin_kernel(volatile int *out) {
  __shared__ int flag;
  const int rank = blockIdx.x;
  if (threadIdx.x != 0)
    return;
  if (rank == 0) {
    flag = 0;
    out[0] = 1;
    __threadfence_system();
    // Stay resident while the peer spins on this flag. A local clock loop
    // is not a peer-smem spin, so rule 1 does not apply here.
    const unsigned long long start = clock64();
    while (clock64() - start < 20000ull) {
    }
    return;
  }
  while (out[0] == 0) {
  }
  volatile int *peer =
      reinterpret_cast<volatile int *>(mapa_u64_shared(&flag, 0));
  while (*peer == 0) {
  }
  out[1] = 1;
}

std::string g_bare_peer_fatal;

void note_bare_peer_fatal(const char *message) {
  g_bare_peer_fatal = message ? message : "";
}

void run_bare_peer_spin() {
  setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1);
  int *d_out = nullptr;
  if (cudaMalloc(&d_out, 2 * sizeof(int)) != cudaSuccess) {
    g_bare_peer_fatal = "alloc failed";
    return;
  }
  if (cudaMemset(d_out, 0, 2 * sizeof(int)) != cudaSuccess) {
    g_bare_peer_fatal = "memset failed";
    return;
  }
  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  if (flash_test::launch_kernel_with_cluster(
          (const void *)bare_peer_spin_kernel, grid, block, cluster, args) !=
      cudaSuccess) {
    g_bare_peer_fatal = "launch failed";
    return;
  }
  if (cudaDeviceSynchronize() != cudaSuccess &&
      g_bare_peer_fatal.find("rule 1") == std::string::npos)
    g_bare_peer_fatal = "synchronize failed";
  cudaFree(d_out);
}

TEST_F(MbarrierClusterTest, BarePeerSpin_Aborts) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  // The observer checks the abort string. It does not prove abort() teardown.
  g_bare_peer_fatal.clear();
  ASSERT_TRUE(flashgpu_set_cluster_fatal_observer != nullptr);
  flashgpu_set_cluster_fatal_observer(note_bare_peer_fatal);
  struct ClearObserver {
    ~ClearObserver() {
      flashgpu_set_cluster_fatal_observer(nullptr);
      unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
    }
  } clear_observer;
  run_bare_peer_spin();
  EXPECT_NE(g_bare_peer_fatal.find("rule 1"), std::string::npos)
      << g_bare_peer_fatal;
}

__global__ void finite_peer_stream_kernel(volatile int *out) {
  __shared__ int values[64];
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 0) {
    for (int i = 0; i < 64; ++i) values[i] = i + 1;
    __threadfence_system();
    out[0] = 1;
    while (out[1] == 0) {}
  } else {
    while (out[0] == 0) {}
    volatile int *peer = reinterpret_cast<volatile int *>(
        mapa_u64_shared(values, 0));
    int sum = 0;
#pragma unroll 1
    for (int i = 0; i < 64; ++i) sum += peer[i];
    out[1] = sum;
  }
}

TEST_F(MbarrierClusterTest, FinitePeerStreamDoesNotTripWatchdog) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1), 0);
  int *out = nullptr;
  ASSERT_EQ(cudaMalloc(&out, 2 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(out, 0, 2 * sizeof(int)), cudaSuccess);
  void *args[] = {&out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)finite_peer_stream_kernel, dim3(2), dim3(32),
                dim3(2, 1, 1), args), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int result[2]{};
  ASSERT_EQ(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(result[1], 64 * 65 / 2);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
  EXPECT_EQ(unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG"), 0);
}

// Other warps sit at __syncthreads while tid0 is in try_wait that never
// completes. Abort after the watchdog dwell, not at TEST_TIMEOUT.
__global__ void progressing_producer_with_bar_sync_kernel() {
  __shared__ unsigned long long bar;
  if (threadIdx.x == 0) mbarrier_init_local(&bar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
    asm volatile("{ .reg .pred ready; WAIT_PRODUCER: "
                 "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0; "
                 "@!ready bra.uni WAIT_PRODUCER; }" :: "r"(addr));
  } else if (threadIdx.x == 32) {
    const auto begin = clock64();
    while (clock64() - begin < 6000) {}
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr));
  }
  __syncthreads();
}

TEST_F(MbarrierClusterTest, ActiveProducerWithBarSyncDoesNotTripWatchdog) {
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1), 0);
  progressing_producer_with_bar_sync_kernel<<<1, 96>>>();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG"), 0);
}

__global__ void mixed_barsync_trywait_kernel() {
  __shared__ unsigned long long bar;
  if (threadIdx.x == 0)
    mbarrier_init_local(&bar, /*expected=*/2);
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_MIX_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@!P1 bra.uni LAB_MIX_WAIT;\n"
                 "}\n" ::"r"(p),
                 "r"(0));
  }
  __syncthreads();
}

std::string g_mixed_bar_fatal;

void note_mixed_bar_fatal(const char *message) {
  g_mixed_bar_fatal = message ? message : "";
}

void run_mixed_barsync_trywait() {
  setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1);
  dim3 grid(1), block(64), cluster(1, 1, 1);
  void *args[] = {};
  if (flash_test::launch_kernel_with_cluster(
          (const void *)mixed_barsync_trywait_kernel, grid, block, cluster,
          args) != cudaSuccess) {
    g_mixed_bar_fatal = "launch failed";
    return;
  }
  if (cudaDeviceSynchronize() != cudaSuccess &&
      g_mixed_bar_fatal.find("rule 2") == std::string::npos)
    g_mixed_bar_fatal = "synchronize failed";
}

TEST_F(MbarrierClusterTest, BarSyncThenSingleThreadTryWait_Aborts) {
  // Death-test re-exec is SIGKILLed in the 7 GiB CI cgroup, and abort()
  // inside the OpenMP core loop kills this process before gtest records it.
  g_mixed_bar_fatal.clear();
  ASSERT_TRUE(flashgpu_set_cluster_fatal_observer != nullptr);
  flashgpu_set_cluster_fatal_observer(note_mixed_bar_fatal);
  struct ClearObserver {
    ~ClearObserver() {
      flashgpu_set_cluster_fatal_observer(nullptr);
      unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
    }
  } clear_observer;
  run_mixed_barsync_trywait();
  EXPECT_NE(g_mixed_bar_fatal.find("rule 2"), std::string::npos)
      << g_mixed_bar_fatal;
}

// Full-warp try_wait after syncthreads is not the mixed hang.
__global__ void full_warp_trywait_after_sync_kernel(int *out) {
  __shared__ unsigned long long bar;
  if (threadIdx.x == 0)
    mbarrier_init_local(&bar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(p),
                 "r"(1));
  }
  __syncthreads();
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_FULL_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@!P1 bra.uni LAB_FULL_WAIT;\n"
               "}\n" ::"r"(p),
               "r"(0));
  out[threadIdx.x] = 1;
}

// One peer load, then a long *local* tight loop. Must not abort: the peer
// arm expires after a hop-scale quiet window.
__global__ void peer_then_local_tight_loop_kernel(volatile int *out) {
  __shared__ int flag;
  __shared__ volatile int acc;
  const int rank = blockIdx.x;
  if (threadIdx.x != 0)
    return;
  if (rank == 0) {
    flag = 7;
    out[0] = 1;
    __threadfence_system();
    while (out[1] == 0) {
    }
    return;
  }
  while (out[0] == 0) {
  }
  volatile int *peer =
      reinterpret_cast<volatile int *>(mapa_u64_shared(&flag, 0));
  const int v = *peer;
  acc = 0;
  for (int i = 0; i < 768; i++)
    acc += i;
  out[1] = acc + v;
}

TEST_F(MbarrierClusterTest, PeerThenLocalTightLoop_Ok) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "512", 1), 0);
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 2 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 2 * sizeof(int)), cudaSuccess);
  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)peer_then_local_tight_loop_kernel, grid, block,
                cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
  int h[2] = {0, 0};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  EXPECT_EQ(h[0], 1);
  EXPECT_EQ(h[1], 768 * 767 / 2 + 7);
}

TEST_F(MbarrierClusterTest, FullWarpTryWaitAfterSync_Ok) {
  int *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 32 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 32 * sizeof(int)), cudaSuccess);
  dim3 grid(1), block(32), cluster(1, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)full_warp_trywait_after_sync_kernel, grid, block,
                cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h[32] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  for (int i = 0; i < 32; i++)
    EXPECT_EQ(h[i], 1) << "lane " << i;
}

TEST_F(MbarrierClusterTest, TryWaitTimeoutPhaseDone_PredTrue) {
  unsigned *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, sizeof(unsigned)), cudaSuccess);
  dim3 grid(1), block(32), cluster(1, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)mbar_timeout_phase_done_kernel, grid, block,
                cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  EXPECT_EQ(h, 1u) << "timed try_wait must set dest pred true when the "
                      "waited phase has already completed";
}

__device__ __forceinline__ int read_smid() {
  int value = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ void cluster_phase_arrive_wait() {
  asm volatile("barrier.cluster.arrive;\n");
  asm volatile("barrier.cluster.wait;\n");
}

// One warp. Completes one cluster phase, records %smid, then rank 1 spins on
// rank 0's shared flag. Rule 1 retires that warp.
__global__ void rule1_phase_then_spin_kernel(int *smid) {
  __shared__ int flag;
  const int rank = static_cast<int>(blockIdx.x);
  if (threadIdx.x == 0) flag = 0;
  cluster_phase_arrive_wait();
  if (threadIdx.x == 0) smid[rank] = read_smid();
  if (rank == 0) {
    const unsigned long long start = clock64();
    while (clock64() - start < 20000ull) {
    }
    return;
  }
  volatile int *peer =
      reinterpret_cast<volatile int *>(mapa_u64_shared(&flag, 0));
  while (*peer == 0) {
  }
}

// Two warps. Completes one cluster phase, then rank 1 mixes bar.sync with a
// single-thread try_wait so rule 2 retires both of its warps.
__global__ void rule2_phase_then_mix_kernel(int *smid) {
  __shared__ unsigned long long bar;
  const int rank = static_cast<int>(blockIdx.x);
  cluster_phase_arrive_wait();
  if (threadIdx.x == 0) smid[rank] = read_smid();
  if (rank == 0) return;
  if (threadIdx.x == 0) mbarrier_init_local(&bar, 2);
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_PHASE_MIX:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@!P1 bra.uni LAB_PHASE_MIX;\n"
                 "}\n" ::"r"(p),
                 "r"(0));
  }
  __syncthreads();
}

__global__ void phase_slot_touch_kernel() {}

// One warp on the retired SM plus a peer. A leftover {arrived:1, completed:1}
// makes this wait ask for generation 2.
__global__ void phase_reissue_kernel(int *smid, int *flag) {
  if (threadIdx.x == 0) smid[blockIdx.x] = read_smid();
  cluster_phase_arrive_wait();
  if (threadIdx.x == 0) flag[blockIdx.x] = 1;
}

struct SmSlot {
  unsigned gpc = 0;
  unsigned local = 0;
  unsigned sms = 0;
};

std::vector<unsigned> read_gpc_sms() {
  std::ifstream in("gpgpusim.config");
  std::vector<unsigned> sms;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key) || key != "-gpgpu_gpc_sms") continue;
    unsigned value = 0;
    char sep = 0;
    if (!(iss >> value)) break;
    sms.push_back(value);
    while (iss >> sep >> value) {
      if (sep != ',') break;
      sms.push_back(value);
    }
    break;
  }
  return sms;
}

bool locate_sm(unsigned smid, const std::vector<unsigned> &sms, SmSlot *slot) {
  unsigned base = 0;
  for (unsigned gpc = 0; gpc < sms.size(); ++gpc) {
    if (smid < base + sms[gpc]) {
      slot->gpc = gpc;
      slot->local = smid - base;
      slot->sms = sms[gpc];
      return true;
    }
    base += sms[gpc];
  }
  return false;
}

bool launch_one_cluster(const void *fn, unsigned ctas, unsigned threads,
                        void **args) {
  if (flash_test::launch_kernel_with_cluster(
          fn, dim3(ctas), dim3(threads), dim3(ctas, 1, 1), args) != cudaSuccess) {
    ADD_FAILURE() << "cluster launch failed ctas=" << ctas;
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    ADD_FAILURE() << "cluster synchronize failed ctas=" << ctas;
    return false;
  }
  return true;
}

bool launch_fillers(unsigned count) {
  for (unsigned i = 0; i < count; ++i) {
    if (!launch_one_cluster((const void *)phase_slot_touch_kernel, 2, 32,
                            nullptr))
      return false;
  }
  return true;
}

// Fatal kernel retires rank 1 on some SM. The next visit to that GPC starts
// just after that SM. A cluster of (sms-1) walks every other SM and leaves
// the cursor on the predecessor, so the following cluster-2 starts on the
// retired SM. Fillers rotate the GPC cursor without touching that GPC.
bool expect_same_slot_phase_releases(const char *rule, const void *fatal,
                                    unsigned fatal_threads) {
  const auto topo = flash_test::read_gpgpusim_topology();
  if (!topo.found_config || read_gpc_sms().size() != topo.n_clusters) {
    ADD_FAILURE() << "gpgpusim.config did not list one width per GPC";
    return false;
  }
  const std::vector<unsigned> sms = read_gpc_sms();

  int *smid = nullptr;
  if (cudaMalloc(&smid, 2 * sizeof(int)) != cudaSuccess) {
    ADD_FAILURE() << "smid alloc";
    return false;
  }
  cudaMemset(smid, 0xff, 2 * sizeof(int));
  void *fatal_args[] = {&smid};
  if (!launch_one_cluster(fatal, 2, fatal_threads, fatal_args)) {
    cudaFree(smid);
    return false;
  }
  int retired_pair[2] = {};
  if (cudaMemcpy(retired_pair, smid, sizeof(retired_pair),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    ADD_FAILURE() << "smid copy";
    cudaFree(smid);
    return false;
  }
  const unsigned retired = static_cast<unsigned>(retired_pair[1]);
  SmSlot slot;
  if (!locate_sm(retired, sms, &slot) || slot.sms < 2) {
    ADD_FAILURE() << "retired smid " << retired_pair[1];
    cudaFree(smid);
    return false;
  }

  const unsigned turn = topo.n_clusters - 1;
  if (!launch_fillers(turn) ||
      !launch_one_cluster((const void *)phase_slot_touch_kernel, slot.sms - 1,
                          32, nullptr) ||
      !launch_fillers(turn)) {
    cudaFree(smid);
    return false;
  }

  int *probe_smid = nullptr;
  int *flag = nullptr;
  if (cudaMalloc(&probe_smid, 2 * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&flag, 2 * sizeof(int)) != cudaSuccess) {
    ADD_FAILURE() << "probe alloc";
    cudaFree(smid);
    cudaFree(probe_smid);
    cudaFree(flag);
    return false;
  }
  cudaMemset(probe_smid, 0xff, 2 * sizeof(int));
  cudaMemset(flag, 0, 2 * sizeof(int));
  void *probe_args[] = {&probe_smid, &flag};
  if (!launch_one_cluster((const void *)phase_reissue_kernel, 2, 32,
                          probe_args)) {
    cudaFree(smid);
    cudaFree(probe_smid);
    cudaFree(flag);
    return false;
  }
  int seen[2] = {};
  int done[2] = {};
  if (cudaMemcpy(seen, probe_smid, sizeof(seen), cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      cudaMemcpy(done, flag, sizeof(done), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    ADD_FAILURE() << "probe copy";
    cudaFree(smid);
    cudaFree(probe_smid);
    cudaFree(flag);
    return false;
  }
  const bool match = seen[0] == retired_pair[1] || seen[1] == retired_pair[1];
  std::printf(
      "phase-clear rule=%s retired_smid=%d local=%u/%u probe_smid=%d,%d "
      "match=%d flag=%d,%d\n",
      rule, retired_pair[1], slot.local, slot.sms, seen[0], seen[1],
      match ? 1 : 0, done[0], done[1]);
  std::fflush(stdout);
  EXPECT_TRUE(match) << "probe did not reissue the retired SM";
  EXPECT_EQ(done[0], 1);
  EXPECT_EQ(done[1], 1);
  cudaFree(smid);
  cudaFree(probe_smid);
  cudaFree(flag);
  return match && done[0] == 1 && done[1] == 1;
}

TEST_F(MbarrierClusterTest, Rule1Retire_SameSlotNextPhaseReleases) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  SKIP_IF_CLUSTER_NOC_OFF();
  g_bare_peer_fatal.clear();
  ASSERT_TRUE(flashgpu_set_cluster_fatal_observer != nullptr);
  flashgpu_set_cluster_fatal_observer(note_bare_peer_fatal);
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1), 0);
  struct ClearObserver {
    ~ClearObserver() {
      flashgpu_set_cluster_fatal_observer(nullptr);
      unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
    }
  } clear_observer;
  EXPECT_TRUE(expect_same_slot_phase_releases(
      "1", (const void *)rule1_phase_then_spin_kernel, 32));
  EXPECT_NE(g_bare_peer_fatal.find("rule 1"), std::string::npos)
      << g_bare_peer_fatal;
}

TEST_F(MbarrierClusterTest, Rule2Retire_SameSlotNextPhaseReleases) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  g_mixed_bar_fatal.clear();
  ASSERT_TRUE(flashgpu_set_cluster_fatal_observer != nullptr);
  flashgpu_set_cluster_fatal_observer(note_mixed_bar_fatal);
  ASSERT_EQ(setenv("FLASHGPU_CLUSTER_HANG_WATCHDOG", "256", 1), 0);
  struct ClearObserver {
    ~ClearObserver() {
      flashgpu_set_cluster_fatal_observer(nullptr);
      unsetenv("FLASHGPU_CLUSTER_HANG_WATCHDOG");
    }
  } clear_observer;
  EXPECT_TRUE(expect_same_slot_phase_releases(
      "2", (const void *)rule2_phase_then_mix_kernel, 64));
  EXPECT_NE(g_mixed_bar_fatal.find("rule 2"), std::string::npos)
      << g_mixed_bar_fatal;
}

}  // namespace
