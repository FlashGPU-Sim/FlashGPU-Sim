// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory dsm_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

#pragma once

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"

namespace dsm {

namespace cg = cooperative_groups;

constexpr int kMaxClusterRanks = 16;
constexpr int kMaxTileBytes = 96 * 1024;
constexpr int kSourceBytes = kMaxTileBytes;
constexpr int kDestinationBytes = kMaxTileBytes;
// One 200-KiB allocation prevents a second resident CTA on an SM.  Cluster
// ranks must therefore occupy distinct SMs, which the host validates.
constexpr int kDynamicSharedBytes = 200 * 1024;
constexpr int kThreads = 1024;

enum class Pattern : int {
  PairUnidirectional = 0,
  PairBidirectional = 1,
  Ring = 2,
  PackedRing = 3,
};

constexpr int kMixedLoadStoreSameDirection = 0;
constexpr int kMixedLoadStoreOppositeDirection = 1;
constexpr int kMixedLoadTmaSameDirection = 2;
constexpr int kMixedLoadTmaOppositeDirection = 3;

struct TimingResult {
  unsigned long long clock_begin;
  unsigned long long clock_end;
  unsigned long long ns_begin;
  unsigned long long ns_end;
  unsigned long long checksum;
  unsigned int smid;
  int rank;
  int target_rank;
  int source_rank;
  int active;
};

__device__ __forceinline__ uint32_t source_word(int rank, int word) {
  return (0x9e3779b9u * static_cast<uint32_t>(rank + 1)) ^
         (0x85ebca6bu * static_cast<uint32_t>(word + 17));
}

__device__ __forceinline__ uint32_t store_word(int rank, int component) {
  return (0x6a09e667u * static_cast<uint32_t>(rank + 1)) ^
         (0xbb67ae85u + 0x9e3779b9u * static_cast<uint32_t>(component + 1));
}

__device__ __forceinline__ unsigned int read_smid() {
  unsigned int value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ void read_timers(unsigned long long &clock,
                                             unsigned long long &ns) {
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "mov.u64 %1, %%globaltimer;\n\t"
               "}"
               : "=l"(clock), "=l"(ns)
               :
               : "memory");
}

// The artificial dependency prevents the final DSM load from remaining in
// flight past the end timestamp.
__device__ __forceinline__ void
read_timers_after(unsigned long long dependency, unsigned long long &clock,
                  unsigned long long &ns) {
  unsigned long long consumed;
  asm volatile("{\n\t"
               "add.u64 %0, %3, 0;\n\t"
               "mov.u64 %1, %%clock64;\n\t"
               "mov.u64 %2, %%globaltimer;\n\t"
               "}"
               : "=l"(consumed), "=l"(clock), "=l"(ns)
               : "l"(dependency)
               : "memory");
  asm volatile("" : : "l"(consumed) : "memory");
}

__device__ __forceinline__ uint32_t shared_addr(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned long long
load_remote_4(uint32_t address) {
  uint32_t value;
  asm volatile("ld.volatile.shared::cluster.u32 %0, [%1];"
               : "=r"(value)
               : "r"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ unsigned long long
load_local_4(uint32_t address) {
  uint32_t value;
  asm volatile("ld.volatile.shared::cta.u32 %0, [%1];"
               : "=r"(value)
               : "r"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ unsigned long long
load_remote_16(uint32_t address) {
  uint32_t x, y, z, w;
  asm volatile("ld.volatile.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "r"(address)
               : "memory");
  return static_cast<unsigned long long>(x) + y + z + w;
}

__device__ __forceinline__ unsigned long long
load_local_16(uint32_t address) {
  uint32_t x, y, z, w;
  asm volatile("ld.volatile.shared::cta.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "r"(address)
               : "memory");
  return static_cast<unsigned long long>(x) + y + z + w;
}

__device__ __forceinline__ void store_remote_16(uint32_t address, uint32_t x,
                                                 uint32_t y, uint32_t z,
                                                 uint32_t w) {
  asm volatile("st.volatile.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(address), "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}

__device__ __forceinline__ bool pattern_target(Pattern pattern, int rank,
                                                int cluster_size, int param,
                                                int &target) {
  switch (pattern) {
  case Pattern::PairUnidirectional:
    target = 1;
    return rank == 0 && cluster_size >= 2;
  case Pattern::PairBidirectional:
    target = rank ^ 1;
    return rank < 2;
  case Pattern::Ring:
    target = (rank + param) % cluster_size;
    return target != rank;
  case Pattern::PackedRing: {
    const int group_size = param >> 16;
    const int stride = param & 0xffff;
    const int base = (rank / group_size) * group_size;
    target = base + (rank - base + stride) % group_size;
    return group_size > 1 && target != rank;
  }
  }
  target = rank;
  return false;
}

__device__ __forceinline__ void init_reductions(
    unsigned long long *clock_min, unsigned long long *clock_max,
    unsigned long long *ns_min, unsigned long long *ns_max,
    unsigned long long *checksum) {
  if (threadIdx.x == 0) {
    *clock_min = ~0ull;
    *clock_max = 0;
    *ns_min = ~0ull;
    *ns_max = 0;
    *checksum = 0;
  }
}

template <bool Remote>
__device__ __forceinline__ unsigned long long
run_load_body(uint32_t base, int active_warps, int epoch_bytes) {
  const int logical_thread = threadIdx.x;
  const int logical_threads = active_warps * 32;
  if (logical_thread >= logical_threads || logical_thread * 16 >= epoch_bytes) {
    return 0;
  }

  const int stride = logical_threads * 16;
  int offset = logical_thread * 16;
  unsigned long long sum = 0;
  while (offset < epoch_bytes) {
    unsigned long long values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int item = offset + i * stride;
      values[i] = item < epoch_bytes
                      ? (Remote ? load_remote_16(base + item)
                                : load_local_16(base + item))
                      : 0;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sum += values[i];
    }
    offset += 4 * stride;
  }
  return sum;
}

__device__ __forceinline__ unsigned long long
run_local_pressure(uint32_t base, int pressure_warps, int pressure_mode) {
  constexpr int kPressureBytes = 64 * 1024;
  constexpr int kPressureRounds = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= pressure_warps || pressure_mode == 0) {
    return 0;
  }

  unsigned long long sum = 0;
  if (pressure_mode == 1) {
    const int logical_thread = warp * 32 + lane;
    const int logical_threads = pressure_warps * 32;
    for (int round = 0; round < kPressureRounds; ++round) {
      for (int offset = logical_thread * 16; offset < kPressureBytes;
           offset += logical_threads * 16) {
        sum += load_local_16(base + offset);
      }
    }
  } else {
    const int instructions = kPressureBytes / (pressure_warps * 32 * 16);
    for (int round = 0; round < kPressureRounds; ++round) {
      for (int i = 0; i < instructions; ++i) {
        const int offset =
            (warp * 512 + i * 4096 + lane * 128) & (kPressureBytes - 16);
        sum += load_local_16(base + offset);
      }
    }
  }
  return sum;
}

__global__ void load_kernel(TimingResult *results, Pattern pattern, int param,
                            int active_warps, int pressure_mode,
                            int pressure_warps, int epoch_bytes) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int cluster_size = cluster.num_blocks();
  auto *source = reinterpret_cast<uint32_t *>(storage);

  __shared__ unsigned long long clock_min, clock_max, ns_min, ns_max;
  __shared__ unsigned long long checksum, pressure_sink;
  init_reductions(&clock_min, &clock_max, &ns_min, &ns_max, &checksum);
  if (threadIdx.x == 0) {
    pressure_sink = 0;
  }
  for (int word = threadIdx.x; word < kSourceBytes / 4;
       word += blockDim.x) {
    source[word] = source_word(rank, word);
  }
  __syncthreads();

  int target = rank;
  const bool active = pattern_target(pattern, rank, cluster_size, param, target);
  auto *remote_source = cluster.map_shared_rank(source, target);
  const uint32_t remote_base = shared_addr(remote_source);
  cluster.sync();

  if (active && threadIdx.x < active_warps * 32 &&
      threadIdx.x * 16 < epoch_bytes) {
    unsigned long long begin_clock, end_clock, begin_ns, end_ns;
    read_timers(begin_clock, begin_ns);
    const auto local_sum =
        run_load_body<true>(remote_base, active_warps, epoch_bytes);
    read_timers_after(local_sum, end_clock, end_ns);
    atomicMin(&clock_min, begin_clock);
    atomicMax(&clock_max, end_clock);
    atomicMin(&ns_min, begin_ns);
    atomicMax(&ns_max, end_ns);
    atomicAdd(&checksum, local_sum);
  }

  bool is_remote_source = false;
  if (!active && pressure_mode != 0) {
    for (int reader = 0; reader < cluster_size; ++reader) {
      int reader_target = -1;
      if (pattern_target(pattern, reader, cluster_size, param, reader_target) &&
          reader_target == rank) {
        is_remote_source = true;
      }
    }
  }
  if (is_remote_source) {
    const auto value = run_local_pressure(shared_addr(source), pressure_warps,
                                          pressure_mode);
    if (threadIdx.x < pressure_warps * 32) {
      atomicAdd(&pressure_sink, value);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    TimingResult value{};
    value.smid = read_smid();
    value.rank = rank;
    value.target_rank = active ? target : -1;
    value.source_rank = active ? target : -1;
    value.active = active;
    value.checksum = checksum;
    if (active) {
      value.clock_begin = clock_min;
      value.clock_end = clock_max;
      value.ns_begin = ns_min;
      value.ns_end = ns_max;
    }
    results[rank] = value;
  }
  asm volatile("" : : "l"(pressure_sink) : "memory");
  cluster.sync();
}

__device__ __forceinline__ void fence_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

__device__ __forceinline__ void tma_shared_to_remote(
    uint32_t destination, uint32_t source, uint32_t bytes,
    uint32_t remote_barrier) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :
      : "r"(destination), "r"(source), "r"(bytes), "r"(remote_barrier)
      : "memory");
}

__device__ __forceinline__ void wait_barrier_acquire(
    unsigned long long *barrier, unsigned parity) {
  const uint32_t address = shared_addr(barrier);
  asm volatile("{\n\t"
               ".reg .pred done;\n"
               "WAIT_%=: \n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
               "done, [%0], %1;\n\t"
               "@!done bra.uni WAIT_%=;\n\t"
               "}"
               :
               : "r"(address), "r"(parity)
               : "memory");
}

__global__ void tma_kernel(TimingResult *results, Pattern pattern, int param,
                           int chunk_bytes, int epoch_bytes) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int cluster_size = cluster.num_blocks();
  auto *source = storage;
  auto *destination = storage + kSourceBytes;

  __shared__ __align__(8) unsigned long long completion;
  __shared__ unsigned long long checksum;
  __shared__ unsigned long long begin_clock, end_clock, begin_ns, end_ns;

  for (int word = threadIdx.x; word < kSourceBytes / 4;
       word += blockDim.x) {
    reinterpret_cast<uint32_t *>(source)[word] = source_word(rank, word);
  }
  for (int word = threadIdx.x; word < kDestinationBytes / 4;
       word += blockDim.x) {
    reinterpret_cast<uint32_t *>(destination)[word] = 0;
  }
  if (threadIdx.x == 0) {
    mbarrier_init(&completion, 1);
    checksum = 0;
    begin_clock = end_clock = begin_ns = end_ns = 0;
  }
  __syncthreads();

  int source_rank = -1;
  int ignored = -1;
  const bool active_destination =
      pattern_target(pattern, rank, cluster_size, param, ignored);
  if (active_destination) {
    source_rank = ignored;
  }
  if (threadIdx.x == 0) {
    if (active_destination) {
      mbarrier_arrive_expect_tx(&completion, epoch_bytes);
    }
    fence_async_shared_cta();
  }
  cluster.sync();

  if (threadIdx.x == 0 && active_destination) {
    read_timers(begin_clock, begin_ns);
  }
  if (threadIdx.x == 0) {
    for (int destination_rank = 0; destination_rank < cluster_size;
         ++destination_rank) {
      int wanted_source = -1;
      if (!pattern_target(pattern, destination_rank, cluster_size, param,
                          wanted_source) ||
          wanted_source != rank) {
        continue;
      }
      auto *remote_destination =
          cluster.map_shared_rank(destination, destination_rank);
      auto *remote_completion =
          cluster.map_shared_rank(&completion, destination_rank);
      const uint32_t destination_address = shared_addr(remote_destination);
      const uint32_t source_address = shared_addr(source);
      const uint32_t barrier_address = shared_addr(remote_completion);
      for (int offset = 0; offset < epoch_bytes; offset += chunk_bytes) {
        const int remaining = epoch_bytes - offset;
        const int bytes = chunk_bytes < remaining ? chunk_bytes : remaining;
        tma_shared_to_remote(destination_address + offset,
                             source_address + offset, bytes, barrier_address);
      }
    }
    if (active_destination) {
      wait_barrier_acquire(&completion, 0);
      read_timers(end_clock, end_ns);
    }
  }

  __syncthreads();
  if (active_destination) {
    unsigned long long local_sum = 0;
    const auto *words = reinterpret_cast<const uint32_t *>(destination);
    for (int word = threadIdx.x; word < epoch_bytes / 4;
         word += blockDim.x) {
      local_sum += words[word];
    }
    atomicAdd(&checksum, local_sum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    TimingResult value{};
    value.smid = read_smid();
    value.rank = rank;
    value.target_rank = active_destination ? rank : -1;
    value.source_rank = active_destination ? source_rank : -1;
    value.active = active_destination;
    value.checksum = checksum;
    if (active_destination) {
      value.clock_begin = begin_clock;
      value.clock_end = end_clock;
      value.ns_begin = begin_ns;
      value.ns_end = end_ns;
    }
    results[rank] = value;
  }
  cluster.sync();
}

__global__ void store_kernel(TimingResult *results, Pattern pattern, int param,
                             int active_warps, int epoch_bytes) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int cluster_size = cluster.num_blocks();

  __shared__ unsigned long long clock_min, clock_max, ns_min, ns_max;
  __shared__ unsigned long long checksum;
  init_reductions(&clock_min, &clock_max, &ns_min, &ns_max, &checksum);
  for (int word = threadIdx.x; word < epoch_bytes / 4;
       word += blockDim.x) {
    reinterpret_cast<uint32_t *>(storage)[word] = 0;
  }
  __syncthreads();

  int target = rank;
  const bool active = pattern_target(pattern, rank, cluster_size, param, target);
  auto *remote_destination = cluster.map_shared_rank(storage, target);
  const uint32_t destination = shared_addr(remote_destination);
  const bool participates = active && threadIdx.x < active_warps * 32 &&
                            threadIdx.x * 16 < epoch_bytes;
  const uint32_t x = store_word(rank, 0);
  const uint32_t y = store_word(rank, 1);
  const uint32_t z = store_word(rank, 2);
  const uint32_t w = store_word(rank, 3);

  cluster.sync();
  unsigned long long begin_clock = 0, end_clock = 0, begin_ns = 0, end_ns = 0;
  if (participates) {
    read_timers(begin_clock, begin_ns);
    const int logical_threads = active_warps * 32;
    for (int offset = threadIdx.x * 16; offset < epoch_bytes;
         offset += logical_threads * 16) {
      store_remote_16(destination + offset, x, y, z, w);
    }
  }
  // The one timed rendezvous is the completion/visibility point.  Its fixed
  // cost is removed by the payload-size fit.
  cluster.sync();
  if (participates) {
    read_timers(end_clock, end_ns);
    atomicMin(&clock_min, begin_clock);
    atomicMax(&clock_max, end_clock);
    atomicMin(&ns_min, begin_ns);
    atomicMax(&ns_max, end_ns);
  }
  __syncthreads();

  if (active) {
    unsigned long long local_sum = 0;
    for (int word = threadIdx.x; word < epoch_bytes / 4;
         word += blockDim.x) {
      local_sum += load_remote_4(destination + word * 4);
    }
    atomicAdd(&checksum, local_sum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    TimingResult value{};
    value.smid = read_smid();
    value.rank = rank;
    value.target_rank = active ? target : -1;
    value.source_rank = active ? rank : -1;
    value.active = active;
    value.checksum = checksum;
    if (active) {
      value.clock_begin = clock_min;
      value.clock_end = clock_max;
      value.ns_begin = ns_min;
      value.ns_end = ns_max;
    }
    results[rank] = value;
  }
  cluster.sync();
}

// Rank 0 always reads rank 1.  The write payload is either rank 1 -> rank 0
// (same direction as the read response) or rank 0 -> rank 1 (opposite).
__global__ void mixed_kernel(TimingResult *results, int mixed_kind,
                             int epoch_bytes, int operation_warps,
                             int tma_chunk_bytes) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  auto *source = storage;
  auto *destination = storage + kSourceBytes;

  const bool use_tma = mixed_kind >= kMixedLoadTmaSameDirection;
  const bool same_direction = (mixed_kind & 1) == 0;
  const int write_source_rank = same_direction ? 1 : 0;
  const int write_target_rank = same_direction ? 0 : 1;

  __shared__ __align__(8) unsigned long long completion;
  __shared__ unsigned long long begin_clock, end_clock, begin_ns, end_ns;
  __shared__ unsigned long long load_checksum, write_checksum;
  for (int word = threadIdx.x; word < kSourceBytes / 4;
       word += blockDim.x) {
    reinterpret_cast<uint32_t *>(source)[word] = source_word(rank, word);
  }
  for (int word = threadIdx.x; word < kDestinationBytes / 4;
       word += blockDim.x) {
    reinterpret_cast<uint32_t *>(destination)[word] = 0;
  }
  if (threadIdx.x == 0) {
    mbarrier_init(&completion, 1);
    begin_clock = end_clock = begin_ns = end_ns = 0;
    load_checksum = write_checksum = 0;
    if (use_tma && rank == write_target_rank) {
      mbarrier_arrive_expect_tx(&completion, epoch_bytes);
    }
    fence_async_shared_cta();
  }
  __syncthreads();
  cluster.sync();

  if (threadIdx.x == 0) {
    read_timers(begin_clock, begin_ns);
  }
  __syncthreads();

  if (use_tma && rank == write_source_rank && threadIdx.x == 0) {
    auto *remote_destination =
        cluster.map_shared_rank(destination, write_target_rank);
    auto *remote_completion =
        cluster.map_shared_rank(&completion, write_target_rank);
    const uint32_t destination_address = shared_addr(remote_destination);
    const uint32_t source_address = shared_addr(source);
    const uint32_t barrier_address = shared_addr(remote_completion);
    for (int offset = 0; offset < epoch_bytes; offset += tma_chunk_bytes) {
      const int remaining = epoch_bytes - offset;
      const int bytes =
          tma_chunk_bytes < remaining ? tma_chunk_bytes : remaining;
      tma_shared_to_remote(destination_address + offset,
                           source_address + offset, bytes, barrier_address);
    }
  }

  unsigned long long local_load_sum = 0;
  if (rank == 0) {
    auto *remote_source = cluster.map_shared_rank(source, 1);
    local_load_sum = run_load_body<true>(shared_addr(remote_source),
                                         operation_warps, epoch_bytes);
  }

  if (!use_tma && rank == write_source_rank) {
    auto *remote_destination =
        cluster.map_shared_rank(destination, write_target_rank);
    const uint32_t destination_address = shared_addr(remote_destination);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Opposite-direction traffic shares rank 0 between load and store, so use
    // disjoint warp pools.  Same-direction traffic issues from rank 1.
    const int first_store_warp = same_direction ? 0 : operation_warps;
    const int store_warp = warp - first_store_warp;
    if (store_warp >= 0 && store_warp < operation_warps) {
      const int logical_thread = store_warp * 32 + lane;
      const int logical_threads = operation_warps * 32;
      const uint32_t x = store_word(rank, 0);
      const uint32_t y = store_word(rank, 1);
      const uint32_t z = store_word(rank, 2);
      const uint32_t w = store_word(rank, 3);
      for (int offset = logical_thread * 16; offset < epoch_bytes;
           offset += logical_threads * 16) {
        store_remote_16(destination_address + offset, x, y, z, w);
      }
    }
  }

  if (use_tma && rank == write_target_rank && threadIdx.x == 0) {
    wait_barrier_acquire(&completion, 0);
  }
  cluster.sync();
  if (threadIdx.x == 0) {
    read_timers(end_clock, end_ns);
  }
  __syncthreads();

  if (rank == 0 && (threadIdx.x >> 5) < operation_warps) {
    atomicAdd(&load_checksum, local_load_sum);
  }
  if (rank == 1) {
    auto *written_destination =
        write_target_rank == rank
            ? destination
            : cluster.map_shared_rank(destination, write_target_rank);
    const uint32_t written_address = shared_addr(written_destination);
    unsigned long long local_write_sum = 0;
    for (int word = threadIdx.x; word < epoch_bytes / 4;
         word += blockDim.x) {
      local_write_sum += write_target_rank == rank
                             ? load_local_4(written_address + word * 4)
                             : load_remote_4(written_address + word * 4);
    }
    atomicAdd(&write_checksum, local_write_sum);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    TimingResult value{};
    value.smid = read_smid();
    value.rank = rank;
    value.active = 1;
    value.clock_begin = begin_clock;
    value.clock_end = end_clock;
    value.ns_begin = begin_ns;
    value.ns_end = end_ns;
    if (rank == 0) {
      value.source_rank = 1;
      value.target_rank = 1;
      value.checksum = load_checksum;
    } else {
      value.source_rank = write_source_rank;
      value.target_rank = write_target_rank;
      value.checksum = write_checksum;
    }
    results[rank] = value;
  }
  cluster.sync();
}

} // namespace dsm
