// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory tma_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

// nvcc -std=c++17 -arch=sm_120 -O3 bandwidth_test.cu
#include "benchmark_framework.cuh"
#include "kernel_wrappers.cuh"

#ifndef BENCH_TOTAL_MIB
#define BENCH_TOTAL_MIB 64
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 16
#endif

int main() {
  const size_t total_bytes = size_t(BENCH_TOTAL_MIB) * 1024 * 1024;
  constexpr int repeat = BENCH_REPEAT;

#if defined(BENCH_L2_MODE)
  constexpr const char *benchmark_mode = "L2-resident";
#elif defined(BENCH_HBM_MODE)
  constexpr const char *benchmark_mode = "HBM-streaming";
#else
  constexpr const char *benchmark_mode = "legacy-mixed";
#endif

  int l2_bytes = 0;
  int persisting_l2_bytes = 0;
  check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0),
        "get L2 cache size");
  check(cudaDeviceGetAttribute(&persisting_l2_bytes,
                               cudaDevAttrMaxPersistingL2CacheSize, 0),
        "get persisting L2 cache size");
  printf("Benchmark mode: %s\n", benchmark_mode);
  printf("Working set: %d MiB | Repeat: %d | Logical bytes/launch: %zu\n",
         BENCH_TOTAL_MIB, repeat, total_bytes * repeat);
  printf("Device L2: %.2f MiB | Max persisting L2: %.2f MiB\n",
         l2_bytes / double(1 << 20), persisting_l2_bytes / double(1 << 20));
#if defined(BENCH_FORCE_CG)
  printf("Compiler-generated global loads: cache-global (L2 only)\n");
#endif
#if BENCH_FLUSH_L2_MIB > 0
  printf("L2 eviction before each timed launch: %d MiB (outside timing)\n",
         BENCH_FLUSH_L2_MIB);
#endif

  // Define all power of two sequences at the beginning to avoid duplicate declarations
  using power_of_two_sequence_2 = power_of_two_sequence<2, 32>;
  using power_of_two_sequence_4 = power_of_two_sequence<4, 32>;
  using power_of_two_sequence_8 = power_of_two_sequence<8, 32>;
  using power_of_two_sequence_16 = power_of_two_sequence<16, 32>;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());
  // const dim3 grid(1);

#if !defined(ONLY_CP_ASYNC) && !defined(ONLY_NORMAL_LOAD)
  // TMA benchmarks using unified cp_bw_kernel (64 threads: 1 producer + 1
  // consumer warps)
  const dim3 tma_1_block(64);
  printf("\n=== TMA Bandwidth Test (1 Producer Warp) ===\n");
  TMAKernelWrapper<1, 1> tma_wrapper_1;
  BandwidthBenchmark tma_bench_1(test_data, grid, tma_1_block, tma_wrapper_1,
                                 "TMA + 1 producer warp");

  tma_bench_1.run_all_stages<256, repeat>();
  tma_bench_1.run_all_stages<512, repeat>();
  tma_bench_1.run_all_stages<1024, repeat>();
  tma_bench_1.run_all_stages<2048, repeat>();
  tma_bench_1.run_all_stages<4096, repeat>();
  tma_bench_1.run_all_stages<8192, repeat>();
  tma_bench_1.run_all_stages<16384, repeat>();

  // TMA benchmarks with 2 producer warps (128 threads: 2 producer + 2
  // consumer warps)
  const dim3 tma_2_block(128);
  printf("\n=== TMA Bandwidth Test (2 Producer Warps) ===\n");
  TMAKernelWrapper<2, 2> tma_wrapper_2;
  BandwidthBenchmark tma_bench_2(test_data, grid, tma_2_block, tma_wrapper_2,
                                 "TMA + 2 producer warps");

  tma_bench_2.run_all_stages<256, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<512, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<1024, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<2048, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<4096, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<8192, repeat, power_of_two_sequence_2>();
  tma_bench_2.run_all_stages<16384, repeat, power_of_two_sequence_2>();

  // TMA benchmarks with 4 producer warps (256 threads: 4 producer + 4
  // consumer warps)
  const dim3 tma_4_block(256);
  printf("\n=== TMA Bandwidth Test (4 Producer Warps) ===\n");
  TMAKernelWrapper<4, 4> tma_wrapper_4;
  BandwidthBenchmark tma_bench_4(test_data, grid, tma_4_block, tma_wrapper_4,
                                 "TMA + 4 producer warps");

  tma_bench_4.run_all_stages<256, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<512, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<1024, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<2048, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<4096, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<8192, repeat, power_of_two_sequence_4>();
  tma_bench_4.run_all_stages<16384, repeat, power_of_two_sequence_4>();

  // TMA benchmarks with 8 producer warps (512 threads: 8 producer + 8
  // consumer warps)
  const dim3 tma_8_block(512);
  printf("\n=== TMA Bandwidth Test (8 Producer Warps) ===\n");
  TMAKernelWrapper<8, 8> tma_wrapper_8;
  BandwidthBenchmark tma_bench_8(test_data, grid, tma_8_block, tma_wrapper_8,
                                 "TMA + 8 producer warps");

  tma_bench_8.run_all_stages<256, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<512, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<1024, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<2048, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<4096, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<8192, repeat, power_of_two_sequence_8>();
  tma_bench_8.run_all_stages<16384, repeat, power_of_two_sequence_8>();

  // TMA benchmarks with 16 producer warps (1024 threads: 16 producer + 16
  // consumer warps)
  const dim3 tma_16_block(1024);
  printf("\n=== TMA Bandwidth Test (16 Producer Warps) ===\n");
  TMAKernelWrapper<16, 16> tma_wrapper_16;
  BandwidthBenchmark tma_bench_16(test_data, grid, tma_16_block, tma_wrapper_16,
                                  "TMA + 16 producer warps");

  tma_bench_16.run_all_stages<256, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<512, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<1024, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<2048, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<4096, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<8192, repeat, power_of_two_sequence_16>();
  tma_bench_16.run_all_stages<16384, repeat, power_of_two_sequence_16>();
#endif

#if !defined(ONLY_TMA) && !defined(ONLY_NORMAL_LOAD)
  // cp.async benchmarks with 1 producer warp (64 threads: 1 producer + 1
  // consumer warps)
  const dim3 cp_async_1_block(64);
  printf("\n=== cp.async Bandwidth Test (1 Producer Warp) ===\n");
  CPKernelWrapper<1, 1, CP_METHOD::CP_ASYNC> cp_async_wrapper_1;
  BandwidthBenchmark cp_async_bench_1(test_data, grid, cp_async_1_block,
                                      cp_async_wrapper_1,
                                      "cp.async + 1 producer warp");

  cp_async_bench_1.run_all_stages<256, repeat>();
  cp_async_bench_1.run_all_stages<512, repeat>();
  cp_async_bench_1.run_all_stages<1024, repeat>();
  cp_async_bench_1.run_all_stages<2048, repeat>();
  cp_async_bench_1.run_all_stages<4096, repeat>();
  cp_async_bench_1.run_all_stages<8192, repeat>();
  cp_async_bench_1.run_all_stages<16384, repeat>();

  // cp.async benchmarks with 2 producer warps (128 threads: 2 producer + 2
  // consumer warps)
  const dim3 cp_async_2_block(128);
  printf("\n=== cp.async Bandwidth Test (2 Producer Warps) ===\n");
  CPKernelWrapper<2, 2, CP_METHOD::CP_ASYNC> cp_async_wrapper_2;
  BandwidthBenchmark cp_async_bench_2(test_data, grid, cp_async_2_block,
                                      cp_async_wrapper_2,
                                      "cp.async + 2 producer warps");

  cp_async_bench_2.run_all_stages<256, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<512, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<1024, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<2048, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<4096, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<8192, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<16384, repeat, power_of_two_sequence_2>();

  // cp.async benchmarks with 4 producer warps (256 threads: 4 producer + 4
  // consumer warps)
  const dim3 cp_async_4_block(256);
  printf("\n=== cp.async Bandwidth Test (4 Producer Warps) ===\n");
  CPKernelWrapper<4, 4, CP_METHOD::CP_ASYNC> cp_async_wrapper_4;
  BandwidthBenchmark cp_async_bench_4(test_data, grid, cp_async_4_block,
                                      cp_async_wrapper_4,
                                      "cp.async + 4 producer warps");

  cp_async_bench_4.run_all_stages<256, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<512, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<1024, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<2048, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<4096, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<8192, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<16384, repeat, power_of_two_sequence_4>();

  // cp.async benchmarks with 8 producer warps (512 threads: 8 producer + 8
  // consumer warps)
  const dim3 cp_async_8_block(512);
  printf("\n=== cp.async Bandwidth Test (8 Producer Warps) ===\n");
  CPKernelWrapper<8, 8, CP_METHOD::CP_ASYNC> cp_async_wrapper_8;
  BandwidthBenchmark cp_async_bench_8(test_data, grid, cp_async_8_block,
                                      cp_async_wrapper_8,
                                      "cp.async + 8 producer warps");

  cp_async_bench_8.run_all_stages<256, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<512, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<1024, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<2048, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<4096, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<8192, repeat, power_of_two_sequence_8>();
  cp_async_bench_8.run_all_stages<16384, repeat, power_of_two_sequence_8>();

  // cp.async benchmarks with 16 producer warps (1024 threads: 16 producer + 16
  // consumer warps)
  const dim3 cp_async_16_block(1024);
  printf("\n=== cp.async Bandwidth Test (16 Producer Warps) ===\n");
  CPKernelWrapper<16, 16, CP_METHOD::CP_ASYNC> cp_async_wrapper_16;
  BandwidthBenchmark cp_async_bench_16(test_data, grid, cp_async_16_block,
                                       cp_async_wrapper_16,
                                       "cp.async + 16 producer warps");

  cp_async_bench_16.run_all_stages<256, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<512, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<1024, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<2048, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<4096, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<8192, repeat, power_of_two_sequence_16>();
  cp_async_bench_16.run_all_stages<16384, repeat, power_of_two_sequence_16>();
#endif

#if !defined(ONLY_TMA) && !defined(ONLY_CP_ASYNC)
  // Normal load benchmarks with 1 producer warp (64 threads: 1 producer + 1
  // consumer warps)
  const dim3 normal_load_1_block(64);
  printf("\n=== Normal Load Bandwidth Test (1 Producer Warp) ===\n");
  NormalLoadKernelWrapper<1, 1> normal_load_wrapper_1;
  BandwidthBenchmark normal_load_bench_1(test_data, grid, normal_load_1_block,
                                         normal_load_wrapper_1,
                                         "Normal load + 1 producer warp");

  normal_load_bench_1.run_all_stages<256, repeat>();
  normal_load_bench_1.run_all_stages<512, repeat>();
  normal_load_bench_1.run_all_stages<1024, repeat>();
  normal_load_bench_1.run_all_stages<2048, repeat>();
  normal_load_bench_1.run_all_stages<4096, repeat>();
  normal_load_bench_1.run_all_stages<8192, repeat>();
  normal_load_bench_1.run_all_stages<16384, repeat>();

  // Normal load benchmarks with 2 producer warps (128 threads: 2 producer + 2
  // consumer warps)
  const dim3 normal_load_2_block(128);
  printf("\n=== Normal Load Bandwidth Test (2 Producer Warps) ===\n");
  NormalLoadKernelWrapper<2, 2> normal_load_wrapper_2;
  BandwidthBenchmark normal_load_bench_2(test_data, grid, normal_load_2_block,
                                         normal_load_wrapper_2,
                                         "Normal load + 2 producer warps");

  normal_load_bench_2.run_all_stages<256, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<512, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<1024, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<2048, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<4096, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<8192, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<16384, repeat, power_of_two_sequence_2>();

  // Normal load benchmarks with 4 producer warps (256 threads: 4 producer + 4
  // consumer warps)
  const dim3 normal_load_4_block(256);
  printf("\n=== Normal Load Bandwidth Test (4 Producer Warps) ===\n");
  NormalLoadKernelWrapper<4, 4> normal_load_wrapper_4;
  BandwidthBenchmark normal_load_bench_4(test_data, grid, normal_load_4_block,
                                         normal_load_wrapper_4,
                                         "Normal load + 4 producer warps");

  normal_load_bench_4.run_all_stages<256, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<512, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<1024, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<2048, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<4096, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<8192, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<16384, repeat, power_of_two_sequence_4>();

  // Normal load benchmarks with 8 producer warps (512 threads: 8 producer + 8
  // consumer warps)
  const dim3 normal_load_8_block(512);
  printf("\n=== Normal Load Bandwidth Test (8 Producer Warps) ===\n");
  NormalLoadKernelWrapper<8, 8> normal_load_wrapper_8;
  BandwidthBenchmark normal_load_bench_8(test_data, grid, normal_load_8_block,
                                         normal_load_wrapper_8,
                                         "Normal load + 8 producer warps");

  normal_load_bench_8.run_all_stages<256, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<512, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<1024, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<2048, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<4096, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<8192, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<16384, repeat, power_of_two_sequence_8>();

  // Normal load benchmarks with 16 producer warps (1024 threads: 16 producer +
  // 16 consumer warps)
  const dim3 normal_load_16_block(1024);
  printf("\n=== Normal Load Bandwidth Test (16 Producer Warps) ===\n");
  NormalLoadKernelWrapper<16, 16> normal_load_wrapper_16;
  BandwidthBenchmark normal_load_bench_16(test_data, grid, normal_load_16_block,
                                          normal_load_wrapper_16,
                                          "Normal load + 16 producer warps");

  normal_load_bench_16.run_all_stages<256, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<512, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<1024, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<2048, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<4096, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<8192, repeat, power_of_two_sequence_16>();
  normal_load_bench_16
      .run_all_stages<16384, repeat, power_of_two_sequence_16>();
#endif

  return 0;
}
