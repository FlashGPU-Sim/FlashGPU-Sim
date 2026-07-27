// TMA-related clock64() microbenchmarks.
//
// Purpose: isolate cycle undercount between real RTX 5090 and GPGPU-Sim by
// measuring device-side SM cycles around individual TMA-related sequences:
//
//   1. clock64 overhead
//   2. mbarrier arrive_expect_tx(0) + try_wait  (sync only, no data)
//   3. TMA bulk load end-to-end (issue + mbarrier wait) for several sizes
//   4. TMA bulk store + commit_group + wait_group(0)
//   5. Ordinary global->shared load baseline (same byte sizes, no TMA)
//
// Methodology:
//   - Single CTA, single warpleader issues TMA; all lanes participate in wait
//   - Use %clock64 via clock64() / inline asm for SM-cycle timestamps
//   - Warmup iterations, then median over measured samples
//   - Subtract measured clock64 overhead from reported medians
//
// Run:
//   # Real GPU (clean shell)
//   ./run_tests.sh bench "TMALatency*"
//
//   # Simulator
//   source setup.sh && source setup_environment
//   ./run_tests.sh -c SM120_RTX5090_REDUCED bench "TMALatency*"
//
// CSV artifacts land in test/run/<config>/ when invoked via run_tests.sh.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/integration/cp_kernels.cuh"

namespace {

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct BenchOptions {
  int warmup = 3;
  int measured = 21;
  bool subtract_clock_overhead = true;
};

struct CycleSummary {
  double min = 0;
  double median = 0;
  double p90 = 0;
  double max = 0;
  int n = 0;
};

bool running_in_simulator() {
  if (const char *e = std::getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) {
    if (e[0] != '\0') return true;
  }
  if (const char *ld = std::getenv("LD_LIBRARY_PATH")) {
    std::string s(ld);
    if (s.find("gpgpu-sim") != std::string::npos) return true;
    if (const char *root = std::getenv("GPGPUSIM_ROOT")) {
      if (s.find(std::string(root) + "/lib/") != std::string::npos) return true;
    }
  }
  return false;
}

double percentile(std::vector<uint64_t> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  q = std::clamp(q, 0.0, 1.0);
  size_t i = static_cast<size_t>(q * (v.size() - 1));
  return static_cast<double>(v[i]);
}

CycleSummary summarize(const std::vector<uint64_t> &samples) {
  CycleSummary s;
  if (samples.empty()) return s;
  std::vector<uint64_t> v = samples;
  std::sort(v.begin(), v.end());
  s.n = static_cast<int>(v.size());
  s.min = static_cast<double>(v.front());
  s.max = static_cast<double>(v.back());
  s.median = percentile(v, 0.5);
  s.p90 = percentile(v, 0.9);
  return s;
}

void print_summary(const char *name, const char *desc, const CycleSummary &s,
                   uint64_t clock_oh) {
  printf("\n=== %s ===\n", name);
  printf("  %s\n", desc);
  printf("  samples=%d  clock64_overhead=%llu\n", s.n,
         (unsigned long long)clock_oh);
  printf("  cycles: min=%.0f  median=%.0f  p90=%.0f  max=%.0f\n", s.min,
         s.median, s.p90, s.max);
}

void write_csv(const char *path, const char *name, const char *desc,
               const CycleSummary &s, uint64_t clock_oh,
               const char *sequence) {
  std::ofstream out(path);
  out << "case,description,sequence,samples,clock64_overhead,"
         "min_cycles,median_cycles,p90_cycles,max_cycles,mode\n";
  out << name << ",\"" << desc << "\",\"" << sequence << "\"," << s.n << ","
      << clock_oh << "," << s.min << "," << s.median << "," << s.p90 << ","
      << s.max << ","
      << (running_in_simulator() ? "simulator" : "native") << "\n";
  printf("  CSV -> %s\n", path);
}

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------

__global__ void k_clock_overhead(uint64_t *d_start, uint64_t *d_end) {
  if (threadIdx.x == 0) {
    uint64_t a, b;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(a)::"memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(b)::"memory");
    d_start[0] = a;
    d_end[0] = b;
  }
}

// mbarrier-only: arrive.expect_tx(0) then try_wait (no TMA data movement)
__global__ void k_mbarrier_only(uint64_t *d_start, uint64_t *d_end,
                                uint32_t *d_ok) {
  __shared__ __align__(8) unsigned long long bar;
  if (threadIdx.x == 0) {
    mbarrier_init(&bar, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
    mbarrier_arrive_expect_tx(&bar, 0);
    wait(&bar, /*parity=*/0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    d_start[0] = t0;
    d_end[0] = t1;
    d_ok[0] = 1;
  }
}

// TMA bulk load GMEM -> SMEM with mbarrier completion, timed end-to-end.
// BYTES must be multiple of 16 and power-of-two friendly (template).
template <int BYTES>
__global__ void k_tma_load_e2e(const uint8_t *__restrict__ g_src,
                               uint64_t *d_start, uint64_t *d_end,
                               uint32_t *d_ok) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ __align__(8) unsigned long long bar;

  if (threadIdx.x == 0) {
    mbarrier_init(&bar, 1);
  }
  __syncthreads();

  // Elect leader = lane 0 of warp 0
  if (threadIdx.x == 0) {
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");

    // Register expected transaction bytes, then issue TMA bulk load.
    mbarrier_arrive_expect_tx(&bar, BYTES);
    cp_async_bulk<BYTES>(smem, g_src, &bar);
    wait(&bar, /*parity=*/0);

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    d_start[0] = t0;
    d_end[0] = t1;

    // Touch first word so compiler keeps the transfer live.
    d_ok[0] = *reinterpret_cast<uint32_t *>(smem);
  }
  __syncthreads();
}

// TMA bulk store SMEM -> GMEM with commit_group + wait_group(0).
template <int BYTES>
__global__ void k_tma_store_e2e(uint8_t *__restrict__ g_dst, uint64_t *d_start,
                                uint64_t *d_end, uint32_t *d_ok) {
  extern __shared__ __align__(16) uint8_t smem[];

  // Fill shared from registers (cheap setup, not timed).
  for (int i = threadIdx.x; i < BYTES / 4; i += blockDim.x) {
    reinterpret_cast<uint32_t *>(smem)[i] = static_cast<uint32_t>(i + 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Ensure SMEM writes visible to async proxy before TMA store.
    fence_proxy_async();

    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");

    cp_async_bulk_store<BYTES>(g_dst, smem);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    d_start[0] = t0;
    d_end[0] = t1;
    d_ok[0] = 1;
  }
}

// Baseline: scalar global loads of BYTES into shared (no TMA).
// Uses 32-bit loads by all threads of the CTA.
template <int BYTES>
__global__ void k_gmem_load_baseline(const uint8_t *__restrict__ g_src,
                                     uint64_t *d_start, uint64_t *d_end,
                                     uint32_t *d_ok) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int nwords = BYTES / 4;

  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
    // Leader does the whole copy for a clean single-thread latency comparison
    // against TMA (also single-thread issued).
    const uint32_t *src = reinterpret_cast<const uint32_t *>(g_src);
    uint32_t *dst = reinterpret_cast<uint32_t *>(smem);
    for (int i = 0; i < nwords; ++i) {
      dst[i] = src[i];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    d_start[0] = t0;
    d_end[0] = t1;
    d_ok[0] = dst[0];
  }
  __syncthreads();
}

// Repeated TMA loads to same buffer (after first fills L2): warm latency.
// Uses two mbarriers in alternation so we never re-init the same address
// (FlashGPU-Sim rejects mbarrier.init collisions).
template <int BYTES>
__global__ void k_tma_load_warm_repeat(const uint8_t *__restrict__ g_src,
                                       uint64_t *d_start, uint64_t *d_end,
                                       uint32_t *d_ok, int repeats) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ __align__(8) unsigned long long bar0;
  __shared__ __align__(8) unsigned long long bar1;

  if (threadIdx.x == 0) {
    mbarrier_init(&bar0, 1);
    mbarrier_init(&bar1, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Cold prime on bar0 (not timed)
    mbarrier_arrive_expect_tx(&bar0, BYTES);
    cp_async_bulk<BYTES>(smem, g_src, &bar0);
    wait(&bar0, /*parity=*/0);

    // Timed warm loads: alternate bar0/bar1 and flip parity after each
    // completion so we never re-init (sim forbids mbarrier init collision).
    unsigned p0 = 1; // bar0 completed phase 0 during prime
    unsigned p1 = 0; // bar1 unused
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
    for (int r = 0; r < repeats; ++r) {
      if ((r % 2) == 0) {
        mbarrier_arrive_expect_tx(&bar1, BYTES);
        cp_async_bulk<BYTES>(smem, g_src, &bar1);
        wait(&bar1, p1);
        p1 ^= 1u;
      } else {
        mbarrier_arrive_expect_tx(&bar0, BYTES);
        cp_async_bulk<BYTES>(smem, g_src, &bar0);
        wait(&bar0, p0);
        p0 ^= 1u;
      }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
    d_start[0] = t0;
    d_end[0] = t1;
    d_ok[0] = *reinterpret_cast<uint32_t *>(smem);
  }
  __syncthreads();
}

// -----------------------------------------------------------------------------
// Fixture
// -----------------------------------------------------------------------------

class TMALatencyTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaMalloc(&d_start_, sizeof(uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_end_, sizeof(uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ok_, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf_, kMaxBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst_, kMaxBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_buf_, 0xAB, kMaxBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dst_, 0, kMaxBytes), cudaSuccess);

    // Prefer more iterations on real GPU; fewer under sim (slow).
    if (running_in_simulator()) {
      opts_.warmup = 1;
      opts_.measured = 5;
    } else {
      opts_.warmup = 5;
      opts_.measured = 51;
    }
  }

  void TearDown() override {
    cudaFree(d_start_);
    cudaFree(d_end_);
    cudaFree(d_ok_);
    cudaFree(d_buf_);
    cudaFree(d_dst_);
  }

  uint64_t read_delta() {
    uint64_t s = 0, e = 0;
    EXPECT_EQ(cudaMemcpy(&s, d_start_, sizeof(uint64_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(&e, d_end_, sizeof(uint64_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    // Handle wrap (unlikely for short kernels)
    return (e >= s) ? (e - s) : 0;
  }

  uint64_t measure_clock_overhead() {
    std::vector<uint64_t> samples;
    samples.reserve(opts_.measured);
    for (int i = 0; i < opts_.warmup + opts_.measured; ++i) {
      k_clock_overhead<<<1, 32>>>(d_start_, d_end_);
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      if (i >= opts_.warmup) samples.push_back(read_delta());
    }
    auto s = summarize(samples);
    return static_cast<uint64_t>(s.median);
  }

  template <typename LaunchFn>
  CycleSummary measure(LaunchFn &&launch, uint64_t clock_oh) {
    std::vector<uint64_t> samples;
    samples.reserve(opts_.measured);
    for (int i = 0; i < opts_.warmup + opts_.measured; ++i) {
      launch();
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
          << "kernel failed: " << cudaGetErrorString(cudaGetLastError());
      if (i >= opts_.warmup) {
        uint64_t d = read_delta();
        if (opts_.subtract_clock_overhead && d >= clock_oh) d -= clock_oh;
        samples.push_back(d);
      }
    }
    return summarize(samples);
  }

  static constexpr int kMaxBytes = 16384;
  uint64_t *d_start_ = nullptr;
  uint64_t *d_end_ = nullptr;
  uint32_t *d_ok_ = nullptr;
  uint8_t *d_buf_ = nullptr;
  uint8_t *d_dst_ = nullptr;
  BenchOptions opts_;
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_F(TMALatencyTest, ClockOverhead) {
  const uint64_t oh = measure_clock_overhead();
  printf("\n=== ClockOverhead ===\n  median clock64();clock64() = %llu cycles\n",
         (unsigned long long)oh);
  EXPECT_GT(oh, 0u);
  // Sanity: overhead should be small on both platforms
  EXPECT_LT(oh, 200u);
  write_csv("TMALatencyTest.ClockOverhead.csv", "ClockOverhead",
            "empty clock64 pair", summarize({oh}), oh, "clock64 clock64");
}

TEST_F(TMALatencyTest, MBarrierOnly) {
  const uint64_t oh = measure_clock_overhead();
  auto s = measure(
      [&] {
        k_mbarrier_only<<<1, 32>>>(d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("MBarrierOnly",
                "arrive.expect_tx(0) + try_wait.parity (no TMA data)", s, oh);
  write_csv("TMALatencyTest.MBarrierOnly.csv", "MBarrierOnly",
            "arrive.expect_tx(0) + try_wait", s, oh,
            "clock64 arrive.expect_tx(0) try_wait clock64");
  // Should complete; value depends on mbarrier model (sim: ~29+32 config)
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, TMALoad256B) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 256;
  auto s = measure(
      [&] {
        k_tma_load_e2e<B><<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("TMALoad256B", "TMA bulk load 256B GMEM->SMEM e2e", s, oh);
  write_csv("TMALatencyTest.TMALoad256B.csv", "TMALoad256B",
            "TMA bulk load 256B end-to-end", s, oh,
            "clock64 arrive.expect_tx TMA_load try_wait clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, TMALoad1KB) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 1024;
  auto s = measure(
      [&] {
        k_tma_load_e2e<B><<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("TMALoad1KB", "TMA bulk load 1KB GMEM->SMEM e2e", s, oh);
  write_csv("TMALatencyTest.TMALoad1KB.csv", "TMALoad1KB",
            "TMA bulk load 1024B end-to-end", s, oh,
            "clock64 arrive.expect_tx TMA_load try_wait clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, TMALoad4KB) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 4096;
  auto s = measure(
      [&] {
        k_tma_load_e2e<B><<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("TMALoad4KB", "TMA bulk load 4KB GMEM->SMEM e2e", s, oh);
  write_csv("TMALatencyTest.TMALoad4KB.csv", "TMALoad4KB",
            "TMA bulk load 4096B end-to-end", s, oh,
            "clock64 arrive.expect_tx TMA_load try_wait clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, TMALoad16KB) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 16384;
  auto s = measure(
      [&] {
        k_tma_load_e2e<B><<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("TMALoad16KB", "TMA bulk load 16KB GMEM->SMEM e2e", s, oh);
  write_csv("TMALatencyTest.TMALoad16KB.csv", "TMALoad16KB",
            "TMA bulk load 16384B end-to-end", s, oh,
            "clock64 arrive.expect_tx TMA_load try_wait clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, TMALoad4KBWarm) {
  // Time 8 back-to-back loads after a cold prime; report cycles per load.
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 4096;
  constexpr int REPS = 8;
  auto s = measure(
      [&] {
        k_tma_load_warm_repeat<B>
            <<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_, REPS);
      },
      oh);
  // Convert total to per-load
  CycleSummary per = s;
  if (s.n > 0) {
    per.min /= REPS;
    per.median /= REPS;
    per.p90 /= REPS;
    per.max /= REPS;
  }
  print_summary("TMALoad4KBWarm",
                "TMA bulk load 4KB x8 after prime (per-load median)", per, oh);
  write_csv("TMALatencyTest.TMALoad4KBWarm.csv", "TMALoad4KBWarm",
            "warm TMA load 4KB per-transfer (8 reps total / 8)", per, oh,
            "prime; clock64 {init arrive TMA wait}x8 clock64 / 8");
  EXPECT_GT(per.median, 0.0);
}

TEST_F(TMALatencyTest, TMAStore4KB) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 4096;
  auto s = measure(
      [&] {
        k_tma_store_e2e<B><<<1, 32, B>>>(d_dst_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("TMAStore4KB",
                "TMA bulk store 4KB + commit_group + wait_group(0)", s, oh);
  write_csv("TMALatencyTest.TMAStore4KB.csv", "TMAStore4KB",
            "TMA store 4KB + commit + wait_group(0)", s, oh,
            "clock64 TMA_store commit_group wait_group(0) clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, GmemLoadBaseline4KB) {
  const uint64_t oh = measure_clock_overhead();
  constexpr int B = 4096;
  auto s = measure(
      [&] {
        k_gmem_load_baseline<B>
            <<<1, 32, B>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  print_summary("GmemLoadBaseline4KB",
                "scalar ld.global 4KB by leader thread (no TMA)", s, oh);
  write_csv("TMALatencyTest.GmemLoadBaseline4KB.csv", "GmemLoadBaseline4KB",
            "scalar global load 4KB baseline", s, oh,
            "clock64 for i: smem[i]=gmem[i] clock64");
  EXPECT_GT(s.median, 0.0);
}

TEST_F(TMALatencyTest, SummaryTable) {
  // Run the key cases once more and print a comparison table for the log.
  // Useful when capturing simulator stdout as the primary artifact.
  const uint64_t oh = measure_clock_overhead();

  auto mbar = measure(
      [&] { k_mbarrier_only<<<1, 32>>>(d_start_, d_end_, d_ok_); }, oh);
  auto tma1k = measure(
      [&] {
        k_tma_load_e2e<1024><<<1, 32, 1024>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  auto tma4k = measure(
      [&] {
        k_tma_load_e2e<4096><<<1, 32, 4096>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  auto tma16k = measure(
      [&] {
        k_tma_load_e2e<16384>
            <<<1, 32, 16384>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);
  auto store4k = measure(
      [&] {
        k_tma_store_e2e<4096><<<1, 32, 4096>>>(d_dst_, d_start_, d_end_, d_ok_);
      },
      oh);
  auto gmem4k = measure(
      [&] {
        k_gmem_load_baseline<4096>
            <<<1, 32, 4096>>>(d_buf_, d_start_, d_end_, d_ok_);
      },
      oh);

  printf("\n");
  printf("================================================================\n");
  printf(" TMA clock64() microbench summary  mode=%s\n",
         running_in_simulator() ? "SIMULATOR" : "NATIVE_GPU");
  printf(" clock64_overhead_median = %llu\n", (unsigned long long)oh);
  printf("----------------------------------------------------------------\n");
  printf(" %-24s %10s %10s %10s\n", "case", "median", "p90", "max");
  printf(" %-24s %10.0f %10.0f %10.0f\n", "MBarrierOnly", mbar.median,
         mbar.p90, mbar.max);
  printf(" %-24s %10.0f %10.0f %10.0f\n", "TMALoad1KB", tma1k.median,
         tma1k.p90, tma1k.max);
  printf(" %-24s %10.0f %10.0f %10.0f\n", "TMALoad4KB", tma4k.median,
         tma4k.p90, tma4k.max);
  printf(" %-24s %10.0f %10.0f %10.0f\n", "TMALoad16KB", tma16k.median,
         tma16k.p90, tma16k.max);
  printf(" %-24s %10.0f %10.0f %10.0f\n", "TMAStore4KB", store4k.median,
         store4k.p90, store4k.max);
  printf(" %-24s %10.0f %10.0f %10.0f\n", "GmemLoadBaseline4KB", gmem4k.median,
         gmem4k.p90, gmem4k.max);
  printf("================================================================\n");
  printf(" Config knobs (sim) for reference:\n");
  printf("   ptx_opcode_latency_tma=32  mbarrier_arrive=29  trywait=32\n");
  printf("   gpgpu_l2_rop_latency=260   dram_latency=254\n");
  printf(" Interpretation:\n");
  printf("   TMALoad - MBarrierOnly ≈ TMA data-path latency\n");
  printf("   TMALoad size scaling ≈ AGU issue + memory bandwidth cost\n");
  printf("   GmemLoad vs TMALoad   ≈ bulk TMA advantage / modeling gap\n");
  printf("================================================================\n");

  // Always succeeds — this is a report test.
  SUCCEED();
}

} // namespace
