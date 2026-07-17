#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kAccRegs = 88;
constexpr int kNFragments = 11;
constexpr int kRegsPerN16 = 8;
constexpr int kSharedWords = 4096;
constexpr uint32_t kLeadingByteOffset = 16;
constexpr uint32_t kStrideByteOffset = 1024;
constexpr uint32_t kSwizzleMode128B = 1;

struct ChainSample {
  uint64_t issue_cycles;
  uint64_t wait_cycles;
  uint64_t total_cycles;
  uint32_t wgmma_count;
  uint32_t block_id;
};

struct ChainResult {
  std::string name;
  int k_tiles;
  int n_fragments;
  int ops_per_round;
  int blocks;
  int rounds;
  uint64_t median_issue_cycles;
  uint64_t median_wait_cycles;
  uint64_t median_total_cycles;
  double issue_cycles_per_wgmma;
  double total_cycles_per_wgmma;
};

__device__ __forceinline__ uint64_t read_clock64() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const *ptr) {
  uint32_t smem_ptr = 0;
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
               "cvt.u32.u64 %0, smem_ptr; }\n"
               : "=r"(smem_ptr)
               : "l"(ptr));
  return smem_ptr;
}

__device__ __forceinline__ uint64_t make_gmma_swizzle_desc(
    void const *ptr, uint32_t leading_byte_offset, uint32_t stride_byte_offset,
    uint32_t swizzle_mode) {
  uint32_t smem_addr = smem_ptr_to_uint(ptr);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  desc |= static_cast<uint64_t>(swizzle_mode & 0x3) << 62;
  return desc;
}

template <int Slot>
__device__ __forceinline__ void issue_m64n16k16_ss(uint64_t desc_a,
                                                   uint64_t desc_b,
                                                   float (&d)[kAccRegs]) {
  static_assert(Slot >= 0 && Slot < kNFragments);
  constexpr int base = Slot * kRegsPerN16;
  int32_t scale_d = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[base + 0]), "+f"(d[base + 1]), "+f"(d[base + 2]),
        "+f"(d[base + 3]), "+f"(d[base + 4]), "+f"(d[base + 5]),
        "+f"(d[base + 6]), "+f"(d[base + 7])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

template <int Slot>
__device__ __forceinline__ void issue_m64n16k16_rs(
    const uint32_t (&a_regs)[4], uint64_t desc_b, float (&d)[kAccRegs]) {
  static_assert(Slot >= 0 && Slot < kNFragments);
  constexpr int base = Slot * kRegsPerN16;
  int32_t scale_d = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %12, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, %13, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[base + 0]), "+f"(d[base + 1]), "+f"(d[base + 2]),
        "+f"(d[base + 3]), "+f"(d[base + 4]), "+f"(d[base + 5]),
        "+f"(d[base + 6]), "+f"(d[base + 7])
      : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
        "r"(scale_d), "l"(desc_b));
}

__device__ __forceinline__ void issue_11_n16_ss(uint64_t desc_a,
                                                const uint64_t (&desc_b)[11],
                                                float (&d)[kAccRegs]) {
  issue_m64n16k16_ss<0>(desc_a, desc_b[0], d);
  issue_m64n16k16_ss<1>(desc_a, desc_b[1], d);
  issue_m64n16k16_ss<2>(desc_a, desc_b[2], d);
  issue_m64n16k16_ss<3>(desc_a, desc_b[3], d);
  issue_m64n16k16_ss<4>(desc_a, desc_b[4], d);
  issue_m64n16k16_ss<5>(desc_a, desc_b[5], d);
  issue_m64n16k16_ss<6>(desc_a, desc_b[6], d);
  issue_m64n16k16_ss<7>(desc_a, desc_b[7], d);
  issue_m64n16k16_ss<8>(desc_a, desc_b[8], d);
  issue_m64n16k16_ss<9>(desc_a, desc_b[9], d);
  issue_m64n16k16_ss<10>(desc_a, desc_b[10], d);
}

template <int I, int Count, int KGroups>
__device__ __forceinline__ void issue_count_n16_ss(
    const uint64_t (&desc_a)[KGroups], const uint64_t (&desc_b)[kNFragments],
    float (&d)[kAccRegs]) {
  if constexpr (I < Count) {
    constexpr int slot = I % kNFragments;
    constexpr int k_group = I / kNFragments;
    issue_m64n16k16_ss<slot>(desc_a[k_group], desc_b[slot], d);
    issue_count_n16_ss<I + 1, Count>(desc_a, desc_b, d);
  }
}

template <int I, int Count>
__device__ __forceinline__ void issue_count_n16_rs(
    const uint64_t (&desc_b)[kNFragments], const uint32_t (&a_regs)[4],
    float (&d)[kAccRegs]) {
  if constexpr (I < Count) {
    constexpr int slot = I % kNFragments;
    issue_m64n16k16_rs<slot>(a_regs, desc_b[slot], d);
    issue_count_n16_rs<I + 1, Count>(desc_b, a_regs, d);
  }
}

template <int KTiles>
__global__ void n16_chain_kernel(ChainSample *samples, float *sink_out,
                                 int rounds) {
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_a[kSharedWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedWords];

  for (int i = threadIdx.x; i < kSharedWords; i += blockDim.x) {
    smem_a[i] = 0x3c003c00u;
    smem_b[i] = 0x3c003c00u;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  uint64_t desc_a[KTiles];
#pragma unroll
  for (int k = 0; k < KTiles; ++k) {
    desc_a[k] = make_gmma_swizzle_desc(
        smem_a + k * 64, kLeadingByteOffset, kStrideByteOffset,
        kSwizzleMode128B);
  }

  uint64_t desc_b[kNFragments];
#pragma unroll
  for (int n = 0; n < kNFragments; ++n) {
    desc_b[n] = make_gmma_swizzle_desc(
        smem_b + n * 64, kLeadingByteOffset, kStrideByteOffset,
        kSwizzleMode128B);
  }

  float d[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    d[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
  }

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
#pragma unroll
    for (int k = 0; k < KTiles; ++k) {
      issue_11_n16_ss(desc_a[k], desc_b, d);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();

  float sink = 0.0f;
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    sink += d[i];
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count =
        static_cast<uint32_t>(rounds * KTiles * kNFragments);
    samples[blockIdx.x].block_id = blockIdx.x;
    sink_out[blockIdx.x] = sink;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    samples[0] = {};
    sink_out[0] = 0.0f;
  }
#endif
}

template <int Count>
__global__ void n16_count_kernel(ChainSample *samples, float *sink_out,
                                 int rounds) {
  static_assert(Count >= 1 && Count <= 128);
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  constexpr int kGroups = (Count + kNFragments - 1) / kNFragments;
  __shared__ __align__(128) uint32_t smem_a[kSharedWords];
  __shared__ __align__(128) uint32_t smem_b[kSharedWords];

  for (int i = threadIdx.x; i < kSharedWords; i += blockDim.x) {
    smem_a[i] = 0x3c003c00u;
    smem_b[i] = 0x3c003c00u;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  uint64_t desc_a[kGroups];
#pragma unroll
  for (int k = 0; k < kGroups; ++k) {
    desc_a[k] = make_gmma_swizzle_desc(
        smem_a + k * 64, kLeadingByteOffset, kStrideByteOffset,
        kSwizzleMode128B);
  }

  uint64_t desc_b[kNFragments];
#pragma unroll
  for (int n = 0; n < kNFragments; ++n) {
    desc_b[n] = make_gmma_swizzle_desc(
        smem_b + n * 64, kLeadingByteOffset, kStrideByteOffset,
        kSwizzleMode128B);
  }

  float d[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    d[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
  }

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
    issue_count_n16_ss<0, Count>(desc_a, desc_b, d);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();

  float sink = 0.0f;
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    sink += d[i];
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count = static_cast<uint32_t>(rounds * Count);
    samples[blockIdx.x].block_id = blockIdx.x;
    sink_out[blockIdx.x] = sink;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    samples[0] = {};
    sink_out[0] = 0.0f;
  }
#endif
}

template <int Count>
__global__ void n16_count_rs_kernel(ChainSample *samples, float *sink_out,
                                    int rounds) {
  static_assert(Count >= 1 && Count <= 128);
#if __CUDA_ARCH__ >= 900
  if (threadIdx.x >= kWarpgroupThreads) return;

  __shared__ __align__(128) uint32_t smem_b[kSharedWords];

  for (int i = threadIdx.x; i < kSharedWords; i += blockDim.x) {
    smem_b[i] = 0x3c003c00u;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  uint64_t desc_b[kNFragments];
#pragma unroll
  for (int n = 0; n < kNFragments; ++n) {
    desc_b[n] = make_gmma_swizzle_desc(
        smem_b + n * 64, kLeadingByteOffset, kStrideByteOffset,
        kSwizzleMode128B);
  }

  uint32_t a_regs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_regs[i] = 0x3c003c00u ^ static_cast<uint32_t>((threadIdx.x + i) & 0xf);
  }

  float d[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    d[i] = static_cast<float>((threadIdx.x & 7) + i) * 0.001f;
  }

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  __syncthreads();

  uint64_t issue_cycles = 0;
  uint64_t wait_cycles = 0;
  const uint64_t total_start = read_clock64();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const uint64_t issue_start = read_clock64();
    issue_count_n16_rs<0, Count>(desc_b, a_regs, d);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    const uint64_t wait_start = read_clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const uint64_t wait_end = read_clock64();
    issue_cycles += wait_start - issue_start;
    wait_cycles += wait_end - wait_start;
  }
  const uint64_t total_end = read_clock64();

  float sink = 0.0f;
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    sink += d[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sink += static_cast<float>(a_regs[i] & 0xff) * 0.0001f;
  }

  if (threadIdx.x == 0) {
    samples[blockIdx.x].issue_cycles = issue_cycles;
    samples[blockIdx.x].wait_cycles = wait_cycles;
    samples[blockIdx.x].total_cycles = total_end - total_start;
    samples[blockIdx.x].wgmma_count = static_cast<uint32_t>(rounds * Count);
    samples[blockIdx.x].block_id = blockIdx.x;
    sink_out[blockIdx.x] = sink;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    samples[0] = {};
    sink_out[0] = 0.0f;
  }
#endif
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

int get_env_int(const char *name, int default_value) {
  const char *text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') return default_value;
  return std::atoi(text);
}

std::string get_env_string(const char *name, const char *default_value) {
  const char *text = std::getenv(name);
  return text == nullptr || text[0] == '\0' ? std::string(default_value)
                                            : std::string(text);
}

uint64_t median_u64(std::vector<uint64_t> values) {
  if (values.empty()) return 0;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  return values[mid];
}

template <int KTiles>
ChainResult run_case(const std::string &name, cudaDeviceProp prop, int blocks,
                     int rounds, const std::string &sample_path) {
  ChainSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, sizeof(ChainSample) * blocks),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, sizeof(float) * blocks), "cudaMalloc sink");
  check_cuda(cudaMemset(d_samples, 0, sizeof(ChainSample) * blocks),
             "cudaMemset samples");

  n16_chain_kernel<KTiles><<<blocks, kWarpgroupThreads>>>(d_samples, d_sink,
                                                          rounds);
  check_cuda(cudaGetLastError(), "n16_chain_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "n16_chain_kernel sync");

  std::vector<ChainSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples, sizeof(ChainSample) * blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy samples");
  cudaFree(d_samples);
  cudaFree(d_sink);

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> total_values;
  issue_values.reserve(samples.size());
  wait_values.reserve(samples.size());
  total_values.reserve(samples.size());
  for (const ChainSample &sample : samples) {
    if (sample.wgmma_count == 0) continue;
    issue_values.push_back(sample.issue_cycles);
    wait_values.push_back(sample.wait_cycles);
    total_values.push_back(sample.total_cycles);
  }

  if (!sample_path.empty()) {
    std::ofstream out(sample_path);
    out << "case,block_id,wgmma_count,issue_cycles,wait_cycles,total_cycles\n";
    for (const ChainSample &sample : samples) {
      out << name << "," << sample.block_id << "," << sample.wgmma_count
          << "," << sample.issue_cycles << "," << sample.wait_cycles << ","
          << sample.total_cycles << "\n";
    }
  }

  ChainResult result{};
  result.name = name;
  result.k_tiles = KTiles;
  result.n_fragments = kNFragments;
  result.ops_per_round = KTiles * kNFragments;
  result.blocks = blocks;
  result.rounds = rounds;
  result.median_issue_cycles = median_u64(issue_values);
  result.median_wait_cycles = median_u64(wait_values);
  result.median_total_cycles = median_u64(total_values);
  const double wgmma_count =
      static_cast<double>(rounds * KTiles * kNFragments);
  result.issue_cycles_per_wgmma =
      wgmma_count == 0.0 ? 0.0
                         : static_cast<double>(result.median_issue_cycles) /
                               wgmma_count;
  result.total_cycles_per_wgmma =
      wgmma_count == 0.0 ? 0.0
                         : static_cast<double>(result.median_total_cycles) /
                               wgmma_count;
  (void)prop;
  return result;
}

template <int Count>
ChainResult run_count_case(const std::string &name, cudaDeviceProp prop,
                           int blocks, int rounds,
                           const std::string &sample_path, bool rs_mode) {
  ChainSample *d_samples = nullptr;
  float *d_sink = nullptr;
  check_cuda(cudaMalloc(&d_samples, sizeof(ChainSample) * blocks),
             "cudaMalloc samples");
  check_cuda(cudaMalloc(&d_sink, sizeof(float) * blocks), "cudaMalloc sink");
  check_cuda(cudaMemset(d_samples, 0, sizeof(ChainSample) * blocks),
             "cudaMemset samples");

  if (rs_mode) {
    n16_count_rs_kernel<Count><<<blocks, kWarpgroupThreads>>>(d_samples,
                                                              d_sink, rounds);
    check_cuda(cudaGetLastError(), "n16_count_rs_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "n16_count_rs_kernel sync");
  } else {
    n16_count_kernel<Count><<<blocks, kWarpgroupThreads>>>(d_samples, d_sink,
                                                           rounds);
    check_cuda(cudaGetLastError(), "n16_count_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "n16_count_kernel sync");
  }

  std::vector<ChainSample> samples(blocks);
  check_cuda(cudaMemcpy(samples.data(), d_samples, sizeof(ChainSample) * blocks,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy samples");
  cudaFree(d_samples);
  cudaFree(d_sink);

  std::vector<uint64_t> issue_values;
  std::vector<uint64_t> wait_values;
  std::vector<uint64_t> total_values;
  issue_values.reserve(samples.size());
  wait_values.reserve(samples.size());
  total_values.reserve(samples.size());
  for (const ChainSample &sample : samples) {
    if (sample.wgmma_count == 0) continue;
    issue_values.push_back(sample.issue_cycles);
    wait_values.push_back(sample.wait_cycles);
    total_values.push_back(sample.total_cycles);
  }

  if (!sample_path.empty()) {
    std::ofstream out(sample_path);
    out << "case,block_id,wgmma_count,issue_cycles,wait_cycles,total_cycles\n";
    for (const ChainSample &sample : samples) {
      out << name << "," << sample.block_id << "," << sample.wgmma_count
          << "," << sample.issue_cycles << "," << sample.wait_cycles << ","
          << sample.total_cycles << "\n";
    }
  }

  ChainResult result{};
  result.name = name;
  result.k_tiles = (Count + kNFragments - 1) / kNFragments;
  result.n_fragments = kNFragments;
  result.ops_per_round = Count;
  result.blocks = blocks;
  result.rounds = rounds;
  result.median_issue_cycles = median_u64(issue_values);
  result.median_wait_cycles = median_u64(wait_values);
  result.median_total_cycles = median_u64(total_values);
  const double wgmma_count = static_cast<double>(rounds * Count);
  result.issue_cycles_per_wgmma =
      wgmma_count == 0.0 ? 0.0
                         : static_cast<double>(result.median_issue_cycles) /
                               wgmma_count;
  result.total_cycles_per_wgmma =
      wgmma_count == 0.0 ? 0.0
                         : static_cast<double>(result.median_total_cycles) /
                               wgmma_count;
  (void)prop;
  return result;
}

template <int Count>
ChainResult run_count_dispatch(int selected_count, cudaDeviceProp prop,
                               int blocks, int rounds,
                               const std::string &sample_path, bool rs_mode) {
  if constexpr (Count > 128) {
    throw std::runtime_error("WGMMA_N16_COUNT_SWEEP count out of range");
  } else {
    if (selected_count == Count) {
      return run_count_case<Count>("count_" + std::to_string(Count), prop,
                                   blocks, rounds, sample_path, rs_mode);
    }
    return run_count_dispatch<Count + 1>(selected_count, prop, blocks, rounds,
                                         sample_path, rs_mode);
  }
}

void write_summary_csv(const std::string &path,
                       const std::vector<ChainResult> &results) {
  std::ofstream out(path);
  out << "case,k_tiles,n_fragments,ops_per_round,blocks,rounds,"
         "median_issue_cycles,median_wait_cycles,median_total_cycles,"
         "issue_cycles_per_wgmma,total_cycles_per_wgmma\n";
  for (const ChainResult &r : results) {
    out << r.name << "," << r.k_tiles << "," << r.n_fragments << ","
        << r.ops_per_round << "," << r.blocks << "," << r.rounds
        << "," << r.median_issue_cycles << "," << r.median_wait_cycles << ","
        << r.median_total_cycles << "," << r.issue_cycles_per_wgmma << ","
        << r.total_cycles_per_wgmma << "\n";
  }
}

void print_summary(const std::vector<ChainResult> &results) {
  printf("case,k_tiles,ops_per_round,blocks,rounds,issue,wait,total,"
         "issue_per_wgmma,total_per_wgmma\n");
  for (const ChainResult &r : results) {
    printf("%s,%d,%d,%d,%d,%llu,%llu,%llu,%.3f,%.3f\n", r.name.c_str(),
           r.k_tiles, r.ops_per_round, r.blocks, r.rounds,
           static_cast<unsigned long long>(r.median_issue_cycles),
           static_cast<unsigned long long>(r.median_wait_cycles),
           static_cast<unsigned long long>(r.median_total_cycles),
           r.issue_cycles_per_wgmma, r.total_cycles_per_wgmma);
  }
}

void require_hopper_or_newer() {
  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  ASSERT_GE(prop.major, 9)
      << "WGMMA requires Hopper or newer, got cc " << prop.major << "."
      << prop.minor;
}

}  // namespace

TEST(WgmmaN16ChainBench, Selected) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  const int blocks = get_env_int("WGMMA_N16_CHAIN_BLOCKS",
                                 std::max(1, prop.multiProcessorCount));
  const int rounds = get_env_int("WGMMA_N16_CHAIN_ROUNDS", 16);
  const std::string selected =
      get_env_string("WGMMA_N16_CHAIN_SELECTED", "all");
  const std::string prefix = get_env_string(
      "WGMMA_N16_CHAIN_OUT_PREFIX", "WgmmaN16ChainBench.Selected");

  std::vector<ChainResult> results;
  auto append_case = [&](const std::string &name) {
    const std::string samples_path = prefix + "." + name + ".samples.csv";
    if (name == "chain11_k1") {
      results.push_back(run_case<1>(name, prop, blocks, rounds, samples_path));
    } else if (name == "chain11_k2") {
      results.push_back(run_case<2>(name, prop, blocks, rounds, samples_path));
    } else if (name == "chain11_k4") {
      results.push_back(run_case<4>(name, prop, blocks, rounds, samples_path));
    } else if (name == "chain11_k8") {
      results.push_back(run_case<8>(name, prop, blocks, rounds, samples_path));
    } else {
      throw std::runtime_error(
          "unsupported WGMMA_N16_CHAIN_SELECTED=" + name +
          " (expected all or chain11_k{1,2,4,8})");
    }
  };

  if (selected == "all") {
    append_case("chain11_k1");
    append_case("chain11_k2");
    append_case("chain11_k4");
    append_case("chain11_k8");
  } else {
    append_case(selected);
  }

  print_summary(results);
  write_summary_csv(prefix + ".summary.csv", results);
}

TEST(WgmmaN16CountSweepBench, Selected) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  int begin = get_env_int("WGMMA_N16_COUNT_SWEEP_BEGIN", 1);
  int end = get_env_int("WGMMA_N16_COUNT_SWEEP_END", 128);
  const int selected_count = get_env_int("WGMMA_N16_COUNT_SWEEP_SELECTED", 0);
  if (selected_count > 0) {
    begin = selected_count;
    end = selected_count;
  }
  ASSERT_GE(begin, 1);
  ASSERT_LE(begin, 128);
  ASSERT_GE(end, begin);
  ASSERT_LE(end, 128);

  const int blocks = get_env_int("WGMMA_N16_COUNT_SWEEP_BLOCKS", 1);
  const int rounds = get_env_int("WGMMA_N16_COUNT_SWEEP_ROUNDS", 256);
  const int write_samples =
      get_env_int("WGMMA_N16_COUNT_SWEEP_WRITE_SAMPLES", 0);
  const std::string prefix = get_env_string(
      "WGMMA_N16_COUNT_SWEEP_OUT_PREFIX", "WgmmaN16CountSweepBench.Selected");

  std::vector<ChainResult> results;
  for (int count = begin; count <= end; ++count) {
    const std::string name = "count_" + std::to_string(count);
    const std::string samples_path =
        write_samples ? prefix + "." + name + ".samples.csv" : std::string();
    results.push_back(
        run_count_dispatch<1>(count, prop, blocks, rounds, samples_path,
                              false));
  }

  print_summary(results);
  write_summary_csv(prefix + ".summary.csv", results);
}

TEST(WgmmaN16CountSweepRsBench, Selected) {
  require_hopper_or_newer();

  int device = 0;
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);

  int begin = get_env_int("WGMMA_N16_RS_COUNT_SWEEP_BEGIN", 1);
  int end = get_env_int("WGMMA_N16_RS_COUNT_SWEEP_END", 128);
  const int selected_count =
      get_env_int("WGMMA_N16_RS_COUNT_SWEEP_SELECTED", 0);
  if (selected_count > 0) {
    begin = selected_count;
    end = selected_count;
  }
  ASSERT_GE(begin, 1);
  ASSERT_LE(begin, 128);
  ASSERT_GE(end, begin);
  ASSERT_LE(end, 128);

  const int blocks = get_env_int("WGMMA_N16_RS_COUNT_SWEEP_BLOCKS", 1);
  const int rounds = get_env_int("WGMMA_N16_RS_COUNT_SWEEP_ROUNDS", 256);
  const int write_samples =
      get_env_int("WGMMA_N16_RS_COUNT_SWEEP_WRITE_SAMPLES", 0);
  const std::string prefix = get_env_string(
      "WGMMA_N16_RS_COUNT_SWEEP_OUT_PREFIX",
      "WgmmaN16CountSweepRsBench.Selected");

  std::vector<ChainResult> results;
  for (int count = begin; count <= end; ++count) {
    const std::string name = "rs_count_" + std::to_string(count);
    const std::string samples_path =
        write_samples ? prefix + "." + name + ".samples.csv" : std::string();
    results.push_back(
        run_count_dispatch<1>(count, prop, blocks, rounds, samples_path,
                              true));
    results.back().name = name;
  }

  print_summary(results);
  write_summary_csv(prefix + ".summary.csv", results);
}
