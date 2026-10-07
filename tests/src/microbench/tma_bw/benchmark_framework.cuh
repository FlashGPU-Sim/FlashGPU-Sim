// Source: https://github.com/seanzw/random
// commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
// directory tma_bw
// Blog: https://seanzw.github.io/posts/gpu_dsm_bw/

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "kernels_common.cuh"

#ifndef BENCH_REALLOCATE_EACH_CONFIG
#define BENCH_REALLOCATE_EACH_CONFIG 1
#endif

#ifndef BENCH_FLUSH_L2_MIB
#define BENCH_FLUSH_L2_MIB 0
#endif

#if BENCH_FLUSH_L2_MIB > 0
__global__ void flush_l2_kernel(const uint4 *__restrict__ data,
                                size_t num_vectors,
                                unsigned long long *__restrict__ sinks) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  unsigned long long sum = 0;
  for (size_t i = thread; i < num_vectors; i += stride) {
    const uint4 value = data[i];
    sum += value.x + value.y + value.z + value.w;
  }
  // Every thread stores its reduction so none of the eviction loads can be
  // removed as dead code.
  sinks[thread] = sum;
}
#endif

// Error checking utility
static void check(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Test data setup
class TestData {
private:
  uint8_t *d_src;
  unsigned long long *d_sink;
  size_t total_bytes;
  int num_sms;
#if BENCH_FLUSH_L2_MIB > 0
  uint4 *d_cache_flush;
  unsigned long long *d_cache_flush_sinks;
  int cache_flush_blocks;
  static constexpr int cache_flush_threads = 256;
#endif

public:
  uint32_t gen_src_value(int i) {
    return i * 131u + 17u;
    // return 2;
  }

  TestData(size_t bytes) : total_bytes(bytes) {
    // Query GPU info
    check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
          "get SM count");
    printf("Detected %d SMs on GPU\n", num_sms);

    // Allocate GPU memory
    check(cudaMalloc(&d_src, total_bytes), "malloc d_src");
    check(cudaMalloc(&d_sink, sizeof(unsigned long long) * num_sms),
          "malloc d_sink");

#if BENCH_FLUSH_L2_MIB > 0
    const size_t cache_flush_bytes =
        size_t(BENCH_FLUSH_L2_MIB) * 1024 * 1024;
    cache_flush_blocks = num_sms * 4;
    check(cudaMalloc(&d_cache_flush, cache_flush_bytes),
          "malloc L2 eviction buffer");
    check(cudaMalloc(&d_cache_flush_sinks,
                     sizeof(unsigned long long) * cache_flush_blocks *
                         cache_flush_threads),
          "malloc L2 eviction sinks");
    check(cudaMemset(d_cache_flush, 1, cache_flush_bytes),
          "initialize L2 eviction buffer");
#endif

    // Initialize data
    std::vector<uint8_t> h_src(total_bytes);
    for (int i = 0; i < total_bytes / 4; ++i) {
      auto p = reinterpret_cast<uint32_t *>(h_src.data() + i * 4);
      *p = gen_src_value(i);
    }
    check(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice),
          "H2D");
  }

  void realloc() {
    cudaFree(d_src);
    cudaFree(d_sink);
    check(cudaMalloc(&d_src, total_bytes), "malloc d_src");
    check(cudaMalloc(&d_sink, sizeof(unsigned long long) * num_sms),
          "malloc d_sink");
    // Initialize data
    std::vector<uint8_t> h_src(total_bytes);
    for (int i = 0; i < total_bytes / 4; ++i) {
      auto p = reinterpret_cast<uint32_t *>(h_src.data() + i * 4);
      *p = gen_src_value(i);
    }
    check(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice),
          "H2D");
  }

  ~TestData() {
    cudaFree(d_src);
    cudaFree(d_sink);
#if BENCH_FLUSH_L2_MIB > 0
    cudaFree(d_cache_flush);
    cudaFree(d_cache_flush_sinks);
#endif
  }

  void flush_l2() {
#if BENCH_FLUSH_L2_MIB > 0
    const size_t cache_flush_bytes =
        size_t(BENCH_FLUSH_L2_MIB) * 1024 * 1024;
    flush_l2_kernel<<<cache_flush_blocks, cache_flush_threads>>>(
        d_cache_flush, cache_flush_bytes / sizeof(uint4),
        d_cache_flush_sinks);
    check(cudaGetLastError(), "L2 flush launch");
    check(cudaDeviceSynchronize(), "L2 flush sync");
#endif
  }

  const uint8_t *get_src() const { return d_src; }
  unsigned long long *get_sink() const { return d_sink; }
  size_t get_total_bytes() const { return total_bytes; }
  int get_num_sms() const { return num_sms; }
};

// Benchmark framework
template <typename KernelFunc> class BandwidthBenchmark {
protected:
  TestData &data;
  dim3 grid, block;
  KernelFunc kernel_func;
  const char *kernel_name;

public:
  BandwidthBenchmark(TestData &data, dim3 g, dim3 b, KernelFunc func,
                     const char *name)
      : data(data), grid(g), block(b), kernel_func(func), kernel_name(name) {}

  void check_result(int repeat, int chunk_bytes) {
    // Sum up the first element from each chunk.
    const size_t total_chunks = data.get_total_bytes() / chunk_bytes;
    unsigned long long expected_chunk_sum = 0;

    for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
      size_t byte_offset = chunk_idx * chunk_bytes;
      // Calculate the first 4 bytes (32-bit int) for this chunk
      size_t i = byte_offset / 4;
      uint32_t first_int = data.gen_src_value(i);
      expected_chunk_sum += first_int;
    }

    // Multiply this with repeat, this should be the expected sum.
    unsigned long long expected_total = expected_chunk_sum * repeat;

    // Compare with sink[0] value.
    std::vector<unsigned long long> h_sink(data.get_num_sms());
    check(cudaMemcpy(h_sink.data(), data.get_sink(),
                     sizeof(unsigned long long) * data.get_num_sms(),
                     cudaMemcpyDeviceToHost),
          "D2H sink");

    unsigned long long actual_sum = 0;
    for (int i = 0; i < data.get_num_sms(); ++i) {
      actual_sum += h_sink[i];
    }

    // If mismatch, print error and abort.
    if (actual_sum != expected_total) {
      printf("ERROR: Result mismatch!\n");
      printf("  Expected: %llu\n", expected_total);
      printf("  Actual:   %llu\n", actual_sum);
      printf("  Repeat:   %d\n", repeat);
      printf("  Chunk size: %d bytes\n", chunk_bytes);
      printf("  Total chunks: %zu\n", total_chunks);
      std::exit(1);
    }
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void run_single_config(int warmup_iters = 1, int num_iters = 10) {
    const size_t shmem_bytes = Stages * CHUNK_BYTES;

    int max_shmem = 0;
    cudaDeviceGetAttribute(&max_shmem,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

    if (shmem_bytes + 4096 > max_shmem) {
      printf("Stages=%2d | Chunk=%4d | Out of SHMEM\n", Stages, CHUNK_BYTES);
      return;
    }

    kernel_func.template set_shmem_size<Stages, CHUNK_BYTES, REPEAT>(
        shmem_bytes + 4096);

#if BENCH_REALLOCATE_EACH_CONFIG
    // Reallocate data before warming the deliberately cache-resident mode.
    data.realloc();
#endif

    // Warm-up
    for (int warmup = 0; warmup < warmup_iters; ++warmup) {
      kernel_func.template launch<Stages, CHUNK_BYTES, REPEAT>(
          grid, block, shmem_bytes, data.get_src(), data.get_sink(),
          data.get_total_bytes());
      check(cudaGetLastError(), "warm-up launch");
      check(cudaDeviceSynchronize(), "warm-up sync");
    }

    // Timed run
    float total_ms = 0.0f;
    cudaEvent_t start, stop;

    for (int iter = 0; iter < num_iters; ++iter) {
#if BENCH_FLUSH_L2_MIB > 0
      // Evict the source working set before each measurement. This kernel and
      // its synchronization are deliberately outside the timed interval.
      data.flush_l2();
#endif
      check(cudaEventCreate(&start), "event create start");
      check(cudaEventCreate(&stop), "event create stop");
      check(cudaEventRecord(start), "record start");

      kernel_func.template launch<Stages, CHUNK_BYTES, REPEAT>(
          grid, block, shmem_bytes, data.get_src(), data.get_sink(),
          data.get_total_bytes());

      check(cudaGetLastError(), "timed launch");
      check(cudaEventRecord(stop), "record stop");
      check(cudaEventSynchronize(stop), "event sync");

      float ms = 0.0f;
      check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
      total_ms += ms;

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    float avg_ms = total_ms / num_iters;
    double seconds = avg_ms / 1e3;
    double data_size = (double)data.get_total_bytes() * REPEAT;
    double gbps = data_size / seconds / 1e9;
    check_result(REPEAT, CHUNK_BYTES);
    printf("Passed | Stages=%2d | Chunk=%4d | Warmup=%d | Iters=%d | Time=%.3f "
           "ms | Data=%.2f MiB | BW=%.2f GB/s\n",
           Stages, CHUNK_BYTES, warmup_iters, num_iters, avg_ms,
           data_size / (1 << 20), gbps);
  }

  // Helper to run single stage from integer sequence
  template <int CHUNK_BYTES, int Stage, int REPEAT>
  void run_stage(int warmup_iters = 1, int num_iters = 10) {
    run_single_config<Stage, CHUNK_BYTES, REPEAT>(warmup_iters, num_iters);
  }

  // Recursive template to run all stages in sequence
  template <int CHUNK_BYTES, int REPEAT>
  void run_stages_impl(integer_sequence<>, int warmup_iters = 1,
                       int num_iters = 10) {}

  template <int CHUNK_BYTES, int REPEAT, int Stage, int... Stages>
  void run_stages_impl(integer_sequence<Stage, Stages...>, int warmup_iters = 1,
                       int num_iters = 10) {
    run_stage<CHUNK_BYTES, Stage, REPEAT>(warmup_iters, num_iters);
    run_stages_impl<CHUNK_BYTES, REPEAT>(integer_sequence<Stages...>{},
                                         warmup_iters, num_iters);
  }

  template <int CHUNK_BYTES, int REPEAT,
            typename StageSequence = power_of_two_sequence<1, 32>>
  void run_all_stages(StageSequence seq = {}, int warmup_iters = 1,
                      int num_iters = 10) {
    printf("%s | CHUNK=%d B | total=%zu B | repeat=%d | warmup=%d | iters=%d\n",
           kernel_name, CHUNK_BYTES, data.get_total_bytes(), REPEAT,
           warmup_iters, num_iters);
    run_stages_impl<CHUNK_BYTES, REPEAT>(seq, warmup_iters, num_iters);
  }
};
