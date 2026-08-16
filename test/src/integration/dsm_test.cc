// Distributed shared memory (DSM) functional tests for intra-cluster NoC.
//
// Prefer multi-SM cluster configs:
//   SM90_H200_REDUCED_CLUSTER4x4  (NoC on by default)
//   SM120_RTX5090_REDUCED_CLUSTER2x1 / 4x4 (enable NoC in run config)
//
//   ./run_tests.sh -c SM90_H200_REDUCED_CLUSTER4x4 test "*Dsm*"
//
// Cross-rank coordination uses mbarrier (not bare smem spin-waits). Under
// PTX functional-first simulation, spinning on peer smem before the peer has
// issued hangs the simulator; mbarrier try_wait is interest-list based.
//
// Important: never mix __syncthreads with a single-thread try_wait in the
// same CTA — other threads waiting at bar.sync deadlock with tid0 in try_wait.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "common/cluster_launch.h"
#include "common/gpgpusim_config_topology.h"

namespace {

// mapa.u64: map generic shared pointer into target CTA rank (returns generic).
// Prefer u64: 32-bit shared form cannot hold full generic windows on large SMs.
__device__ __forceinline__ unsigned long long mapa_u64(void *local,
                                                        unsigned rank) {
  unsigned long long out = 0;
  unsigned long long in = reinterpret_cast<unsigned long long>(local);
  asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(out) : "l"(in), "r"(rank));
  return out;
}

__device__ __forceinline__ uint32_t *mapa_shared_rank(uint32_t *local,
                                                       unsigned rank) {
  return reinterpret_cast<uint32_t *>(mapa_u64(local, rank));
}

__device__ __forceinline__ void mbarrier_init_local(unsigned long long *bar,
                                                    unsigned expected) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(p),
               "r"(expected));
}

__device__ __forceinline__ void mbarrier_arrive_remote(unsigned long long bar_g,
                                                       unsigned count) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"l"(bar_g),
               "r"(count));
}

__device__ __forceinline__ void mbarrier_try_wait_local(unsigned long long *bar,
                                                        int parity) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@!P1 bra.uni LAB_WAIT;\n"
               "}\n" ::"r"(p),
               "r"(parity));
}

// ---------------------------------------------------------------------------
// Self-mapa: single CTA maps its own smem rank 0 (no peer / NoC required).
// ---------------------------------------------------------------------------
__global__ void dsm_self_mapa_kernel(uint32_t *out) {
  __shared__ uint32_t smem[4];
  if (threadIdx.x == 0) {
    smem[0] = 0x11223344u;
    uint32_t *mapped = mapa_shared_rank(smem, /*rank=*/0);
    out[0] = mapped[0];
  }
}

// ---------------------------------------------------------------------------
// Remote store: rank 0 writes into rank 1 smem via mapa; rank 1 waits on a
// local mbarrier that rank 0 arrives remotely after the store.
//
// Only tid0 participates (no bar.sync after try_wait).
// Order:
//   rank1: init bar (expect 1), publish ready flag, try_wait, read smem
//   rank0: wait ready, write out[0]=1 (progress), mapa+store, remote arrive,
//          write out[0]=2 (done)
// ---------------------------------------------------------------------------
__global__ void dsm_peer_store_kernel(uint32_t *out, volatile int *ready) {
  __shared__ uint32_t smem[32];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid == 0)
    smem[0] = 0;
  // No __syncthreads after this — only tid0 continues.

  if (rank == 1 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    ready[0] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
    out[1] = smem[0];
  }

  if (rank == 0 && tid == 0) {
    while (ready[0] == 0) {
    }
    out[0] = 1;  // progress: past ready handshake
    __threadfence_system();
    uint32_t *remote = mapa_shared_rank(smem, /*rank=*/1);
    remote[0] = 0xCAFEBABEu;
    unsigned long long remote_bar = mapa_u64(&bar, /*rank=*/1);
    mbarrier_arrive_remote(remote_bar, 1);
    out[0] = 2;  // done: store + arrive issued
  }
}

// ---------------------------------------------------------------------------
// Remote load: rank 0 fills smem; rank 1 loads via mapa after rank 0 arrives
// on rank 1's mbarrier. Only tid0 on each rank participates.
//
// Rank 0 must stay alive until rank 1 finishes the load — mapa looks up the
// peer CTA by active cluster rank and aborts if that CTA has already exited.
// ---------------------------------------------------------------------------
__global__ void dsm_remote_load_kernel(uint32_t *out, volatile int *ready) {
  __shared__ uint32_t smem[32];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid == 0)
    smem[0] = 0;  // clear any recycled-slot residue

  if (rank == 1 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    ready[0] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
    uint32_t *remote = mapa_shared_rank(smem, /*rank=*/0);
    out[1] = remote[0];
    ready[1] = 1;  // allow producer to exit
    __threadfence_system();
  }

  if (rank == 0 && tid == 0) {
    while (ready[0] == 0) {
    }
    smem[0] = 0xA0000007u;
    // Self-mapa sanity on producer side.
    uint32_t *self = mapa_shared_rank(smem, /*rank=*/0);
    out[0] = self[0];
    unsigned long long remote_bar = mapa_u64(&bar, /*rank=*/1);
    mbarrier_arrive_remote(remote_bar, 1);
    // Stay alive until consumer has completed remote load + mapa.
    while (ready[1] == 0) {
    }
  }
}

class DsmTest : public ::testing::Test {};

TEST_F(DsmTest, SelfMapaLocal) {
  uint32_t *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, sizeof(uint32_t)), cudaSuccess);
  dim3 grid(1), block(32), cluster(1, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_self_mapa_kernel, grid, block, cluster, args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  EXPECT_EQ(h, 0x11223344u);
}

TEST_F(DsmTest, RemoteStoreToPeer_TwoCtas) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  constexpr int kRanks = 2;
  uint32_t *d_out = nullptr;
  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_ready, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(kRanks), block(32), cluster(kRanks, 1, 1);
  void *args[] = {&d_out, &d_ready};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_peer_store_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint32_t> h(kRanks, 0);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, kRanks * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  cudaFree(d_ready);

  EXPECT_GE(h[0], 1u) << "rank0 never passed ready handshake (out[0]=" << h[0]
                      << ")";
  EXPECT_EQ(h[0], 2u) << "rank0 did not finish store+arrive (out[0]=" << h[0]
                      << ")";
  EXPECT_EQ(h[1], 0xCAFEBABEu) << "remote store via mapa not visible on peer";
}

TEST_F(DsmTest, RemoteLoadFromPeer_TwoCtas) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  constexpr int kRanks = 2;
  uint32_t *d_out = nullptr;
  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_ready, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(kRanks), block(32), cluster(kRanks, 1, 1);
  void *args[] = {&d_out, &d_ready};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_remote_load_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint32_t> h(kRanks, 0);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, kRanks * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  cudaFree(d_ready);

  EXPECT_EQ(h[0], 0xA0000007u) << "producer self-mapa / local fill failed";
  EXPECT_EQ(h[1], 0xA0000007u) << "remote load via mapa not visible on peer";
}

// ---------------------------------------------------------------------------
// Remote atom.add: rank 0 mapa's rank 1 smem and atomicAdds; rank 1 waits
// on a local mbarrier then reads the sum. The issuer's own smem must stay
// the local sentinel (not the peer sum).
// ---------------------------------------------------------------------------
__global__ void dsm_remote_atom_kernel(uint32_t *out, volatile int *ready) {
  __shared__ uint32_t smem[32];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid == 0)
    smem[0] = 0;

  if (rank == 1 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    ready[0] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
    out[1] = smem[0];
    ready[1] = 1;
    __threadfence_system();
  }

  if (rank == 0 && tid == 0) {
    while (ready[0] == 0) {
    }
    smem[0] = 0x11111111u;
    uint32_t *remote = mapa_shared_rank(smem, /*rank=*/1);
    atomicAdd(remote, 5u);
    atomicAdd(remote, 13u);
    out[0] = smem[0];
    unsigned long long remote_bar = mapa_u64(&bar, /*rank=*/1);
    mbarrier_arrive_remote(remote_bar, 1);
    while (ready[1] == 0) {
    }
  }
}

TEST_F(DsmTest, RemoteAtomicAdd_TwoCtas) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  constexpr int kRanks = 2;
  uint32_t *d_out = nullptr;
  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_ready, 4 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, 4 * sizeof(int)), cudaSuccess);

  dim3 grid(kRanks), block(32), cluster(kRanks, 1, 1);
  void *args[] = {&d_out, &d_ready};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_remote_atom_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint32_t> h(kRanks, 0);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, kRanks * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  cudaFree(d_ready);

  EXPECT_EQ(h[0], 0x11111111u)
      << "remote atom.add mutated the issuer's local smem (local alias)";
  EXPECT_EQ(h[1], 18u) << "peer smem did not see atomicAdd 5+13 via mapa";
}

// ---------------------------------------------------------------------------
// clusterDim=4 — rank 0 stores a distinct word into each peer.
// multi-peer fan-out is the same kernel (1 producer, 3 consumers).
// ---------------------------------------------------------------------------
__global__ void dsm_cluster4_fanout_kernel(uint32_t *out, volatile int *ready) {
  __shared__ uint32_t smem[8];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid == 0)
    smem[0] = 0;

  if (rank != 0 && tid == 0) {
    mbarrier_init_local(&bar, 1);
    ready[rank] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
    out[rank] = smem[0];
    ready[rank + 4] = 1;
    __threadfence_system();
  }

  if (rank == 0 && tid == 0) {
    while (ready[1] == 0 || ready[2] == 0 || ready[3] == 0) {
    }
    out[0] = 1;
    __threadfence_system();
    for (unsigned r = 1; r < 4; r++) {
      uint32_t *remote = mapa_shared_rank(smem, r);
      remote[0] = 0xC0FFE000u + r;
      unsigned long long remote_bar = mapa_u64(&bar, r);
      mbarrier_arrive_remote(remote_bar, 1);
    }
    out[0] = 2;
    while (ready[5] == 0 || ready[6] == 0 || ready[7] == 0) {
    }
  }
}

TEST_F(DsmTest, RemoteStoreCluster4_Fanout) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(4);
  constexpr int kRanks = 4;
  uint32_t *d_out = nullptr;
  int *d_ready = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_ready, 8 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, kRanks * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, 8 * sizeof(int)), cudaSuccess);

  dim3 grid(kRanks), block(32), cluster(kRanks, 1, 1);
  void *args[] = {&d_out, &d_ready};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_cluster4_fanout_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint32_t> h(kRanks, 0);
  ASSERT_EQ(cudaMemcpy(h.data(), d_out, kRanks * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  cudaFree(d_ready);

  EXPECT_EQ(h[0], 2u);
  EXPECT_EQ(h[1], 0xC0FFE001u);
  EXPECT_EQ(h[2], 0xC0FFE002u);
  EXPECT_EQ(h[3], 0xC0FFE003u);
}

// ---------------------------------------------------------------------------
// first cluster writes MAGIC via mapa and exits; a second launch on
// the same ranks must not see a late NoC deliver of MAGIC after it clears
// smem (drop_messages_to_cta + NULL smem on recycled slots).
// ---------------------------------------------------------------------------
__global__ void dsm_stale_writer_kernel(volatile int *ready) {
  __shared__ uint32_t smem[4];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid != 0)
    return;
  smem[0] = 0;
  if (rank == 1) {
    mbarrier_init_local(&bar, 1);
    ready[0] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
  } else {
    while (ready[0] == 0) {
    }
    uint32_t *remote = mapa_shared_rank(smem, 1);
    remote[0] = 0xDEADBEEFu;
    unsigned long long remote_bar = mapa_u64(&bar, 1);
    mbarrier_arrive_remote(remote_bar, 1);
  }
}

__global__ void dsm_stale_occupant_kernel(uint32_t *out) {
  __shared__ uint32_t smem[4];
  if (threadIdx.x == 0 && blockIdx.x == 1) {
    smem[0] = 0x11111111u;
    out[0] = smem[0];
  }
}

TEST_F(DsmTest, DropOnCtaExit_NoStalePayload) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  int *d_ready = nullptr;
  uint32_t *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ready, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_ready, 0, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, sizeof(uint32_t)), cudaSuccess);

  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args1[] = {&d_ready};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_stale_writer_kernel, grid, block, cluster,
                args1),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  void *args2[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_stale_occupant_kernel, grid, block, cluster,
                args2),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t h = 0;
  ASSERT_EQ(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_ready);
  cudaFree(d_out);
  EXPECT_EQ(h, 0x11111111u) << "recycled CTA smem was overwritten by a stale "
                               "NoC DSM_STORE (expected drop_messages_to_cta)";
}

// ---------------------------------------------------------------------------
// mapa.u32 cannot hold the 64-bit generic shared window
// (SHARED_GENERIC_START is > 2^32). u64 self-mapa still works.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned mapa_u32(void *local, unsigned rank) {
  unsigned out = 0;
  unsigned in = static_cast<unsigned>(reinterpret_cast<unsigned long long>(local));
  asm volatile("mapa.u32 %0, %1, %2;\n" : "=r"(out) : "r"(in), "r"(rank));
  return out;
}

__global__ void dsm_mapa_width_kernel(unsigned long long *out) {
  __shared__ uint32_t smem[4];
  if (threadIdx.x == 0) {
    smem[0] = 0xA5A5A5A5u;
    unsigned long long u64 = mapa_u64(smem, /*rank=*/0);
    unsigned u32 = mapa_u32(smem, /*rank=*/0);
    out[0] = u64;
    out[1] = static_cast<unsigned long long>(u32);
    out[2] = *reinterpret_cast<uint32_t *>(u64);
  }
}

TEST_F(DsmTest, MapaU64WorksU32TruncatesGenericWindow) {
  unsigned long long *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, 3 * sizeof(unsigned long long)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_out, 0, 3 * sizeof(unsigned long long)), cudaSuccess);
  dim3 grid(1), block(32), cluster(1, 1, 1);
  void *args[] = {&d_out};
  ASSERT_EQ(flash_test::launch_kernel_with_cluster(
                (const void *)dsm_mapa_width_kernel, grid, block, cluster,
                args),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned long long h[3] = {};
  ASSERT_EQ(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_out);
  EXPECT_EQ(h[2], 0xA5A5A5A5ull) << "mapa.u64 self-map must read local smem";
  EXPECT_NE(h[0], h[1]) << "mapa.u32 should truncate the 64-bit generic window "
                           "(SHARED_GENERIC_START > 2^32); do not use u32";
  EXPECT_GT(h[0], 0xffffffffull);
}

// Rank 0 arrives on rank 1's mbarrier and exits. Rank 1 then maps rank 0.
// Mapping a cluster rank whose CTA is gone must abort, not alias local smem.
// Handshake is mbarrier (a global spin before the peer has issued hangs).
__global__ void dsm_mapa_after_peer_exit_kernel(volatile int *ready) {
  __shared__ uint32_t smem[4];
  __shared__ unsigned long long bar;
  const int rank = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid != 0)
    return;

  if (rank == 1) {
    mbarrier_init_local(&bar, 1);
    ready[0] = 1;
    __threadfence_system();
    mbarrier_try_wait_local(&bar, 0);
    (void)mapa_u64(smem, /*rank=*/0);
    return;
  }

  while (ready[0] == 0) {
  }
  unsigned long long remote_bar = mapa_u64(&bar, /*rank=*/1);
  mbarrier_arrive_remote(remote_bar, 1);
}

[[noreturn]] void run_mapa_after_peer_exit() {
  int *d_ready = nullptr;
  if (cudaMalloc(&d_ready, sizeof(int)) != cudaSuccess)
    _exit(2);
  if (cudaMemset(d_ready, 0, sizeof(int)) != cudaSuccess)
    _exit(3);
  dim3 grid(2), block(32), cluster(2, 1, 1);
  void *args[] = {&d_ready};
  if (flash_test::launch_kernel_with_cluster(
          (const void *)dsm_mapa_after_peer_exit_kernel, grid, block, cluster,
          args) != cudaSuccess)
    _exit(4);
  cudaDeviceSynchronize();
  std::fprintf(stderr, "ERROR: mapa of exited rank did not abort\n");
  _exit(0);
}

TEST_F(DsmTest, MapaAfterProducerExit_FailsLoud) {
  SKIP_IF_N_CORES_PER_CLUSTER_LT(2);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(run_mapa_after_peer_exit(), "not an active CTA");
}

}  // namespace
