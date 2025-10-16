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