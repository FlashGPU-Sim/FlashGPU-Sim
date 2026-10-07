// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory tma_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

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
