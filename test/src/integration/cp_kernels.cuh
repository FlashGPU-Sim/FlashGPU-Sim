#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ---- Common PTX helpers ----
__device__ inline void mbarrier_init(unsigned long long *bar_addr,
                                     unsigned expected_arrivals) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_addr));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(expected_arrivals));
}

__device__ inline void mbarrier_arrive(unsigned long long *bar_addr) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"l"(bar_s));
}

__device__ inline void mbarrier_arrive_expect_tx(unsigned long long *bar_addr,
                                                 unsigned expected_tx) {
  void const *const ptr = bar_addr;
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(expected_tx));
}

__device__ inline void wait(unsigned long long *bar_addr,
                            unsigned expected_parity) {
  void const *const ptr = bar_addr;
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(expected_parity));
}

// Common constants
constexpr int PRODUCER_THREAD = 0;
constexpr int CONSUMER_THREAD = 32;
constexpr int PRODUCER_WARP = 0;  // Warp 0 (threads 0-31)
constexpr int CONSUMER_WARP = 32; // Warp 1 (threads 32-63)

// Integer sequence utilities
template <int... Is> struct integer_sequence {};

// TMA bulk copy - shared between kernels
template <int bytes>
__device__ inline void cp_async_bulk(void *smem_dst, const void *global_src,
                                     unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));

  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], %3, [%2];" ::"l"(dst_s),
               "l"(src_g), "l"(bar_s), "n"(bytes));
}

template <int start, int N, int... Is>
struct make_integer_sequence
    : make_integer_sequence<start, N - 1, N - 1, Is...> {
  static_assert(N > start, "N must be > start");
};

template <int start, int... Is>
struct make_integer_sequence<start, start, Is...> : integer_sequence<Is...> {};

// Helper to create power-of-two sequences
template <int Start, int End, int Current = Start, int... Is>
struct make_power_of_two_sequence {
  using type = typename make_power_of_two_sequence<Start, End, Current * 2,
                                                   Is..., Current>::type;
};

template <int Start, int End, int... Is>
struct make_power_of_two_sequence<Start, End, End, Is...> {
  using type = integer_sequence<Is..., End>;
};

template <int Start, int End>
using power_of_two_sequence =
    typename make_power_of_two_sequence<Start, End>::type;

// Copy method enumeration
enum class CP_METHOD { NORMAL_LOAD = 0, CP_ASYNC = 1, TMA = 2 };

// cp.async 16-byte copy (cache-global)
__device__ inline void cp_async_16(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"l"(dst_s),
               "l"(src_g));
}

// cp.async 8-byte copy (cache-all, since cg doesn't support 8B)
__device__ inline void cp_async_8(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;" ::"l"(dst_s),
               "l"(src_g));
}

// cp.async 4-byte copy (cache-all, since cg doesn't support 4B)
__device__ inline void cp_async_4(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;" ::"l"(dst_s),
               "l"(src_g));
}

// Wait for cp.async operations to complete
template <int group_size = 0> __device__ inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(group_size));
}

// Commit cp.async group - explicitly commit pending cp.async operations
__device__ inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

//=============================================================================
// TMA Store (Bulk) Operations - shared to global memory
//=============================================================================

// TMA bulk store - from shared memory to global memory
// PTX syntax: cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size;
template <int bytes>
__device__ inline void cp_async_bulk_store(void *global_dst,
                                           const void *smem_src) {
  unsigned long long dst_g, src_s;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(dst_g) : "l"(global_dst));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(src_s) : "l"(smem_src));
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_g),
      "l"(src_s), "n"(bytes));
}

// Commit bulk group - commits pending TMA store operations as a group
__device__ inline void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

// Fence for async proxy - ensures writes to shared memory are visible to TMA
// MUST be called before TMA store operations (cp.async.bulk.global.shared)
// to ensure the shared memory writes are ordered before the TMA reads them
__device__ inline void fence_proxy_async() {
  asm volatile("fence.proxy.async;");
}

// Wait for bulk groups to complete
// N = number of most recent groups that are allowed to be incomplete
// wait_group(0) = wait for all groups to complete
// wait_group(1) = allow 1 incomplete group, etc.
template <int N = 0> __device__ inline void cp_async_bulk_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}

//=============================================================================
// TMA Store Kernel - Tests bulk group commit/wait mechanism
//=============================================================================

/**
 * TMA Store Kernel with Bulk Group Synchronization
 *
 * This kernel tests the bulk group mechanism for TMA store operations:
 * 1. Load data from global memory to shared memory (using normal loads)
 * 2. Modify the data in shared memory
 * 3. Store data from shared memory to global memory using TMA bulk store
 * 4. Use commit_group/wait_group to synchronize store completion
 *
 * Template parameters:
 *   NUM_GROUPS: Number of bulk groups to create per warp
 *   CHUNK_BYTES: Size of each chunk in bytes
 *   PIPELINE_DEPTH: Number of groups allowed to be in-flight (for wait_group)
 */
template <int NUM_GROUPS, int CHUNK_BYTES, int PIPELINE_DEPTH = 0>
__global__ void tma_store_kernel(const uint32_t *__restrict__ src,
                                 uint32_t *__restrict__ dst,
                                 size_t num_elements) {
  extern __shared__ __align__(16) uint8_t smem[];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;

  // Each warp processes NUM_GROUPS chunks
  const int chunks_per_warp = NUM_GROUPS;
  const int elements_per_chunk = CHUNK_BYTES / sizeof(uint32_t);
  const int warp_start_element = global_warp_id * chunks_per_warp * elements_per_chunk;

  // Shared memory layout: each warp gets its own buffer
  uint8_t *warp_smem = smem + warp_id * CHUNK_BYTES;

  // Only lane 0 performs TMA operations
  if (lane_id == 0) {
    for (int group = 0; group < NUM_GROUPS; group++) {
      int chunk_start = warp_start_element + group * elements_per_chunk;

      // Skip if out of bounds
      if (chunk_start >= num_elements) break;

      // Step 1: Load data from global to shared (normal load)
      uint32_t *smem_u32 = reinterpret_cast<uint32_t *>(warp_smem);
      for (int i = 0; i < elements_per_chunk && (chunk_start + i) < num_elements; i++) {
        smem_u32[i] = src[chunk_start + i];
      }

      // Step 2: Modify data in shared memory (add 1 to each element)
      for (int i = 0; i < elements_per_chunk && (chunk_start + i) < num_elements; i++) {
        smem_u32[i] += 1;
      }

      // Step 3: Fence before TMA store to ensure shared memory writes
      // are visible to the async proxy (TMA hardware)
      fence_proxy_async();

      // Step 4: TMA store from shared to global
      uint32_t *dst_chunk = dst + chunk_start;
      cp_async_bulk_store<CHUNK_BYTES>(dst_chunk, warp_smem);

      // Step 5: Commit the bulk group
      cp_async_bulk_commit_group();

      // Step 6: If pipeline depth is set, wait with allowance
      if constexpr (PIPELINE_DEPTH > 0) {
        if (group >= PIPELINE_DEPTH) {
          cp_async_bulk_wait_group<PIPELINE_DEPTH>();
        }
      }
    }

    // Final wait: ensure all groups are complete
    cp_async_bulk_wait_group<0>();
  }

  __syncthreads();
}

/**
 * TMA Store Kernel with Multiple Groups and Out-of-Order Testing
 *
 * This kernel creates multiple bulk groups and tests the sequential
 * completion semantics of wait_group.
 *
 * NOTE: Since we reuse the same shared memory buffer for each chunk,
 * we must wait for each async store to complete before loading the
 * next chunk. This ensures the data is fully transferred before
 * the shared memory is overwritten.
 */
template <int CHUNK_BYTES>
__global__ void tma_store_multi_group_kernel(const uint32_t *__restrict__ src,
                                             uint32_t *__restrict__ dst,
                                             int num_chunks) {
  extern __shared__ __align__(16) uint8_t smem[];

  const int lane_id = threadIdx.x % 32;
  const int elements_per_chunk = CHUNK_BYTES / sizeof(uint32_t);

  // Single warp kernel for simplicity
  if (threadIdx.x < 32) {
    if (lane_id == 0) {
      // Process all chunks, creating one group per chunk
      for (int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * elements_per_chunk;

        // Load to shared memory
        uint32_t *smem_u32 = reinterpret_cast<uint32_t *>(smem);
        for (int i = 0; i < elements_per_chunk; i++) {
          smem_u32[i] = src[chunk_start + i];
        }

        // Increment each element
        for (int i = 0; i < elements_per_chunk; i++) {
          smem_u32[i] += (chunk + 1);  // Add chunk index + 1
        }

        // CRITICAL: Fence before TMA store to ensure shared memory writes
        // are visible to the async proxy (TMA hardware)
        fence_proxy_async();

        // TMA store
        cp_async_bulk_store<CHUNK_BYTES>(dst + chunk_start, smem);

        // Commit this chunk as a group
        cp_async_bulk_commit_group();

        // CRITICAL: Wait for this group to complete before reusing smem
        // Since we have only one shared memory buffer, we must ensure
        // the async store is done before overwriting the buffer
        cp_async_bulk_wait_group<0>();
      }
    }
  }
}

/**
 * TMA Store Kernel with Pipelined Groups
 *
 * Tests wait_group(N) where N > 0, allowing multiple in-flight groups.
 *
 * With NUM_SLOTS shared memory buffers, we can have up to NUM_SLOTS-1
 * groups in flight. Before reusing a slot, we must ensure the group
 * that was using that slot has completed.
 *
 * Example with NUM_SLOTS=3 (MAX_IN_FLIGHT=2):
 *   chunk 0: use slot 0, commit group 0
 *   chunk 1: use slot 1, commit group 1
 *   chunk 2: use slot 2, commit group 2
 *   chunk 3: wait until only 2 groups pending (group 0 done), reuse slot 0
 *   ...
 */
template <int CHUNK_BYTES, int MAX_IN_FLIGHT>
__global__ void tma_store_pipelined_kernel(const uint32_t *__restrict__ src,
                                           uint32_t *__restrict__ dst,
                                           int num_chunks) {
  extern __shared__ __align__(16) uint8_t smem[];

  const int lane_id = threadIdx.x % 32;
  const int elements_per_chunk = CHUNK_BYTES / sizeof(uint32_t);

  // Use multiple shared memory slots for pipelining
  // Need NUM_SLOTS = MAX_IN_FLIGHT + 1 to allow MAX_IN_FLIGHT groups in flight
  constexpr int NUM_SLOTS = MAX_IN_FLIGHT + 1;

  if (threadIdx.x < 32 && lane_id == 0) {
    for (int chunk = 0; chunk < num_chunks; chunk++) {
      int slot = chunk % NUM_SLOTS;
      uint8_t *slot_smem = smem + slot * CHUNK_BYTES;
      int chunk_start = chunk * elements_per_chunk;

      // Before reusing a slot, wait for the group that was using it to complete
      // This happens when chunk >= NUM_SLOTS (i.e., we've cycled through all slots)
      // wait_group<N> waits until at most N groups are still pending
      // To ensure slot is free, we need the group (chunk - NUM_SLOTS) to be done
      // After chunk commits, there are (chunk+1) groups total
      // We need groups 0..(chunk-NUM_SLOTS) to be done, leaving NUM_SLOTS-1 pending
      if (chunk >= NUM_SLOTS) {
        // Wait until at most (NUM_SLOTS - 1) groups are pending
        // This ensures the oldest group (using the slot we're about to reuse) is done
        cp_async_bulk_wait_group<NUM_SLOTS - 1>();
      }

      // Load to shared memory slot
      uint32_t *smem_u32 = reinterpret_cast<uint32_t *>(slot_smem);
      for (int i = 0; i < elements_per_chunk; i++) {
        smem_u32[i] = src[chunk_start + i] * 2;  // Double the value
      }

      // CRITICAL: Fence before TMA store to ensure shared memory writes
      // are visible to the async proxy (TMA hardware)
      fence_proxy_async();

      // TMA store
      cp_async_bulk_store<CHUNK_BYTES>(dst + chunk_start, slot_smem);

      // Commit as a group
      cp_async_bulk_commit_group();
    }

    // Final wait for all groups
    cp_async_bulk_wait_group<0>();
  }
}

template <int Stages, int MyInitStages>
__device__ inline void
epilogue_wait_and_signal(int stage_idx, int warp_id, int lane_id,
                         int NUM_PRODUCER_WARPS, unsigned long long *fwd_bar,
                         integer_sequence<>) {}

template <int Stages, int MyInitStages, int I, int... Is>
__device__ inline void
epilogue_wait_and_signal(int stage_idx, int warp_id, int lane_id,
                         int NUM_PRODUCER_WARPS, unsigned long long *fwd_bar,
                         integer_sequence<I, Is...>) {
  int prev_stage_idx = stage_idx - MyInitStages + I;
  int prev_real_stage_idx = prev_stage_idx * NUM_PRODUCER_WARPS + warp_id;
  if (prev_stage_idx >= 0) {
    const int slot = prev_real_stage_idx % Stages;
    cp_async_wait_group<MyInitStages - 1 - I>();
    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive(&fwd_bar[slot]);
    }
  }
  epilogue_wait_and_signal<Stages, MyInitStages>(stage_idx, warp_id, lane_id,
                                                 NUM_PRODUCER_WARPS, fwd_bar,
                                                 integer_sequence<Is...>{});
}

template <int CHUNK_BYTES, CP_METHOD METHOD = CP_METHOD::CP_ASYNC>
__device__ inline void cp_chunk(uint8_t *dst_slot, const uint8_t *src_chunk,
                                int lane_id,
                                unsigned long long *bar_addr = nullptr) {
  // Each thread in the warp copies a portion of the chunk
  if constexpr (METHOD == CP_METHOD::CP_ASYNC) {
    // Notice that the minimal transfer size of cp.async.cg is 16 bytes.
    // ! We assume CHUNK_BYTES is divisble by cp_async_cg_bytes
    constexpr int cp_async_cg_bytes = 16;
    static_assert(CHUNK_BYTES % cp_async_cg_bytes == 0,
                  "CHUNK_BYTES must be divisible by cp_async_cg_bytes");
    for (int b = lane_id * cp_async_cg_bytes; b < CHUNK_BYTES;
         b += 32 * cp_async_cg_bytes) {

      const uint8_t *src_ptr = src_chunk + b;
      uint8_t *dst_ptr = dst_slot + b;
      cp_async_16(dst_ptr, src_ptr);
    }
  } else if constexpr (METHOD == CP_METHOD::TMA) {
    // TMA copy - only one thread per warp issues the TMA copy
    if (lane_id == 0) {
      cp_async_bulk<CHUNK_BYTES>(dst_slot, src_chunk, bar_addr);
    }
  } else {
    // Use normal load instruction with float4 (16 bytes)
    constexpr int load_bytes = 16;
    static_assert(CHUNK_BYTES % load_bytes == 0,
                  "CHUNK_BYTES must be divisible by load_bytes");
    for (int b = lane_id * load_bytes; b < CHUNK_BYTES; b += 32 * load_bytes) {

      const float4 *src_ptr = reinterpret_cast<const float4 *>(src_chunk + b);
      float4 *dst_ptr = reinterpret_cast<float4 *>(dst_slot + b);
      *dst_ptr = *src_ptr;
    }
  }
}

// ---- cp Kernel with Multiple Producer Warps ----
template <int Stages, int CHUNK_BYTES, int REPEAT, int NUM_PRODUCER_WARPS = 1,
          int NUM_CONSUMER_WARPS = 1, CP_METHOD METHOD = CP_METHOD::CP_ASYNC>
__global__ void cp_bw_kernel(const uint8_t *__restrict__ src,
                             unsigned long long *__restrict__ sink,
                             size_t total_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];

  __shared__ unsigned long long fwd_bar[Stages];
  __shared__ unsigned long long bwd_bar[Stages];
  __shared__ unsigned long long final_sum[NUM_CONSUMER_WARPS];

  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      mbarrier_init(&fwd_bar[s], 1);
      mbarrier_init(&bwd_bar[s], 1); // Only consumer signals backward
    }
  }

  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int total_chunks = total_bytes / CHUNK_BYTES;
  const int producer_warp_chunk_idx_start = blockIdx.x + warp_id * gridDim.x;
  const int producer_warp_chunk_idx_step = NUM_PRODUCER_WARPS * gridDim.x;
  /**
   * Calculate how many chunks this warp will process.
   */
  const int producer_stages =
      (total_chunks * REPEAT + producer_warp_chunk_idx_step - 1 -
       producer_warp_chunk_idx_start) /
      producer_warp_chunk_idx_step;

  const int consumer_warp_id = warp_id - NUM_PRODUCER_WARPS;
  const int consumer_warp_chunk_idx_start =
      blockIdx.x + consumer_warp_id * gridDim.x;
  const int consumer_warp_chunk_idx_step = NUM_CONSUMER_WARPS * gridDim.x;
  const int consumer_stages =
      (total_chunks * REPEAT + consumer_warp_chunk_idx_step - 1 -
       consumer_warp_chunk_idx_start) /
      consumer_warp_chunk_idx_step;

  constexpr int NUM_INITIAL_STAGES = Stages / NUM_PRODUCER_WARPS;

  static_assert(Stages >= NUM_PRODUCER_WARPS,
                "Stages must be at least NUM_PRODUCER_WARPS");
  static_assert(Stages % NUM_PRODUCER_WARPS == 0,
                "Stages must be divisible by NUM_PRODUCER_WARPS");

  // Multiple Producer warps: each warp handles different chunks
  if (warp_id < NUM_PRODUCER_WARPS) { // Producer warps

    if constexpr (METHOD == CP_METHOD::TMA) {
      // TMA requires commit before issue
      if (lane_id == 0) {

        // ! Canonical loop variable (starting from 0) improves performance!
        // ! I don't know why...
        for (int stage_idx = 0; stage_idx < producer_stages; stage_idx++) {
          auto real_stage_idx = stage_idx * NUM_PRODUCER_WARPS + warp_id;
          auto real_chunk_idx = stage_idx * producer_warp_chunk_idx_step +
                                producer_warp_chunk_idx_start;

          const int slot = real_stage_idx % Stages;
          uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
          const uint8_t *src_chunk =
              src + (real_chunk_idx % total_chunks) * CHUNK_BYTES;

          unsigned want_free = (real_stage_idx % (Stages * 2)) < Stages;
          wait(&bwd_bar[slot], want_free);

          mbarrier_arrive_expect_tx(&fwd_bar[slot], CHUNK_BYTES);
          cp_async_bulk<CHUNK_BYTES>(dst_slot, src_chunk, &fwd_bar[slot]);
        }
      }
    } else if constexpr (METHOD == CP_METHOD::CP_ASYNC && false) {
      // cp.async method with pipeline.
      // Surprisingly this is slower than the non-pipelined version below.
      // I guess the overhead of wait_group and commit_group is not
      // fully amortized by pipelining.
      // Keep here for reference.
      for (int stage_idx = 0; stage_idx < producer_stages; stage_idx++) {

        if (stage_idx >= NUM_INITIAL_STAGES) {
          // Wait and signal for previous stages.
          int prev_real_stage_idx =
              (stage_idx - NUM_INITIAL_STAGES) * NUM_PRODUCER_WARPS + warp_id;
          const int slot = prev_real_stage_idx % Stages;
          cp_async_wait_group<NUM_INITIAL_STAGES - 1>();
          __syncwarp();
          if (lane_id == 0) {
            mbarrier_arrive(&fwd_bar[slot]);
          }
        }

        auto real_stage_idx = stage_idx * NUM_PRODUCER_WARPS + warp_id;
        auto real_chunk_idx = stage_idx * producer_warp_chunk_idx_step +
                              producer_warp_chunk_idx_start;

        const int slot = real_stage_idx % Stages;
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
        const uint8_t *src_chunk =
            src + (real_chunk_idx % total_chunks) * CHUNK_BYTES;

        unsigned want_free = (real_stage_idx % (Stages * 2)) < Stages;
        wait(&bwd_bar[slot], want_free);

        cp_chunk<CHUNK_BYTES, METHOD>(dst_slot, src_chunk, lane_id,
                                      &fwd_bar[slot]);

        cp_async_commit_group();
      }

      // Epilogue: wait and signal for the remaining stages
      epilogue_wait_and_signal<Stages, NUM_INITIAL_STAGES>(
          producer_stages, warp_id, lane_id, NUM_PRODUCER_WARPS, fwd_bar,
          make_integer_sequence<0, NUM_INITIAL_STAGES>{});

    } else {
      // Non-TMA methods

      // Each producer warp processes chunks with stride

      for (int stage_idx = 0; stage_idx < producer_stages; stage_idx++) {
        auto real_stage_idx = stage_idx * NUM_PRODUCER_WARPS + warp_id;
        auto real_chunk_idx = stage_idx * producer_warp_chunk_idx_step +
                              producer_warp_chunk_idx_start;

        const int slot = real_stage_idx % Stages;
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
        const uint8_t *src_chunk =
            src + (real_chunk_idx % total_chunks) * CHUNK_BYTES;

        unsigned want_free = (real_stage_idx % (Stages * 2)) < Stages;
        wait(&bwd_bar[slot], want_free);

        cp_chunk<CHUNK_BYTES, METHOD>(dst_slot, src_chunk, lane_id,
                                      &fwd_bar[slot]);
        if constexpr (METHOD == CP_METHOD::CP_ASYNC) {
          // Wait for this warp's cp.async operations to complete
          cp_async_commit_group();
          cp_async_wait_group<0>();
        }
        __syncwarp();
        if (lane_id == 0) {
          mbarrier_arrive(&fwd_bar[slot]);
        }
      }
    }
  }

  // Consumer warp: wait for data ready, then consume and free slot
  else if (warp_id >= NUM_PRODUCER_WARPS) {

    if (lane_id == 0) { // Only one thread for simplicity

      unsigned long long sum = 0;

      for (int stage_idx = 0; stage_idx < consumer_stages; stage_idx++) {
        auto real_stage_idx = stage_idx * NUM_CONSUMER_WARPS + consumer_warp_id;

        // for (int i = blockIdx.x, stage_idx = 0; i < total_chunks * REPEAT;
        //      i += gridDim.x, stage_idx++) {
        //   auto real_stage_idx = stage_idx;

        const int slot = real_stage_idx % Stages;
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;

        unsigned want_ready = (real_stage_idx % (Stages * 2)) >= Stages;
        wait(&fwd_bar[slot], want_ready);

        auto value = *reinterpret_cast<const uint32_t *>(dst_slot);
        sum += value;

        mbarrier_arrive(&bwd_bar[slot]);
      }
      // printf("block %d warp %d consumer_stages %d done.\n", blockIdx.x,
      //        warp_id, consumer_stages);

      // printf("Block %d, chunks %d my chunks %d\n", blockIdx.x,
      //        int(total_chunks), int(stage_idx));
      final_sum[consumer_warp_id] = sum;
    }
  }
  __syncthreads();
  if (lane_id == 0 && warp_id == NUM_PRODUCER_WARPS) {
    // Only one consumer warp writes back the result
    unsigned long long ret = 0;
    for (int i = 0; i < NUM_CONSUMER_WARPS; ++i) {
      ret += final_sum[i];
    }
    sink[blockIdx.x] = ret;
  }
}