#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kCpAsyncBytes = 16;

enum BenchMode {
  kModeEmpty = 0,
  kModeIssue = 1,
  kModeComplete = 2,
  kModeGroupWait = 3,
  kModeDepth = 4,
  kModeFa2Issue = 5,
  kModeFa2Complete = 6,
};

enum LocalityMode {
  kLocalityStride = 0,
  kLocalityHot = 1,
  kLocalityBlockHot = 2,
};

struct Options {
  int device = 0;
  int blocks = 132;
  int threads = 128;
  int iters = 4096;
  int warmup = 256;
  int tiles = 4096;
  int groups = 16;
  int group_size = 8;
  int fa2_stages = 8;
  int active_warps = 1;
  int active_lanes = 32;
  int smem_slots = 64;
  int global_slots = 0;
  int compute_gap = 0;
  BenchMode mode = kModeFa2Issue;
  LocalityMode locality = kLocalityBlockHot;
  bool same_address = false;
  std::string csv_path;
};

#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                           \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

__host__ __device__ inline uintptr_t align_up(uintptr_t value,
                                              uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void* ptr) {
  uint64_t addr = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"(ptr));
  return addr;
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ void cp_async_cg_16_l2_128B(void* smem_dst,
                                                       const void* global_src) {
  const uint32_t dst = smem_addr_u32(smem_dst);
  const uint64_t src = gmem_addr_u64(global_src);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
               :
               : "r"(dst), "l"(src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t burn_int_gap(int count, uint32_t x) {
#pragma unroll 1
  for (int i = 0; i < count; ++i) {
    x = x * 1664525u + 1013904223u;
  }
  return x;
}

__device__ __forceinline__ size_t copy_offset(int warp_id, int slot, int lane,
                                              int slots) {
  const int linear = (warp_id * slots + slot) * kWarpSize + lane;
  return static_cast<size_t>(linear) * kCpAsyncBytes;
}

__device__ __forceinline__ void issue_one_cp_async(
    uint8_t* smem_base, const uint8_t* gmem_tile, int warp_id, int lane,
    int active_lanes, int copy_index, int smem_slots, int smem_slot_mask,
    int global_slots, int global_slot_mask, bool same_address) {
  if (lane >= active_lanes) {
    return;
  }
  const int smem_slot = copy_index & smem_slot_mask;
  const int global_slot = same_address ? 0 : (copy_index & global_slot_mask);
  uint8_t* smem_dst =
      smem_base + copy_offset(warp_id, smem_slot, lane, smem_slots);
  const uint8_t* global_src =
      gmem_tile + copy_offset(warp_id, global_slot, lane, global_slots);
  cp_async_cg_16_l2_128B(smem_dst, global_src);
}

template <int Count>
__device__ __forceinline__ void issue_group_fixed(
    uint8_t* smem_base, const uint8_t* gmem_tile, int warp_id, int lane,
    int active_lanes, int copy_base, int smem_slots, int smem_slot_mask,
    int global_slots, int global_slot_mask, bool same_address) {
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    issue_one_cp_async(smem_base, gmem_tile, warp_id, lane, active_lanes,
                       copy_base + i, smem_slots, smem_slot_mask, global_slots,
                       global_slot_mask, same_address);
  }
}

__device__ __forceinline__ void issue_group_runtime(
    uint8_t* smem_base, const uint8_t* gmem_tile, int warp_id, int lane,
    int active_lanes, int copy_base, int group_size, int smem_slots,
    int smem_slot_mask, int global_slots, int global_slot_mask,
    bool same_address) {
#pragma unroll 1
  for (int i = 0; i < group_size; ++i) {
    issue_one_cp_async(smem_base, gmem_tile, warp_id, lane, active_lanes,
                       copy_base + i, smem_slots, smem_slot_mask, global_slots,
                       global_slot_mask, same_address);
  }
}

__device__ __forceinline__ void issue_group(
    uint8_t* smem_base, const uint8_t* gmem_tile, int warp_id, int lane,
    int active_lanes, int copy_base, int group_size, int smem_slots,
    int smem_slot_mask, int global_slots, int global_slot_mask,
    bool same_address) {
  switch (group_size) {
    case 1:
      issue_group_fixed<1>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                           copy_base, smem_slots, smem_slot_mask, global_slots,
                           global_slot_mask, same_address);
      break;
    case 2:
      issue_group_fixed<2>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                           copy_base, smem_slots, smem_slot_mask, global_slots,
                           global_slot_mask, same_address);
      break;
    case 4:
      issue_group_fixed<4>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                           copy_base, smem_slots, smem_slot_mask, global_slots,
                           global_slot_mask, same_address);
      break;
    case 8:
      issue_group_fixed<8>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                           copy_base, smem_slots, smem_slot_mask, global_slots,
                           global_slot_mask, same_address);
      break;
    case 16:
      issue_group_fixed<16>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                            copy_base, smem_slots, smem_slot_mask, global_slots,
                            global_slot_mask, same_address);
      break;
    case 24:
      issue_group_fixed<24>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                            copy_base, smem_slots, smem_slot_mask, global_slots,
                            global_slot_mask, same_address);
      break;
    case 32:
      issue_group_fixed<32>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                            copy_base, smem_slots, smem_slot_mask, global_slots,
                            global_slot_mask, same_address);
      break;
    default:
      issue_group_runtime(smem_base, gmem_tile, warp_id, lane, active_lanes,
                          copy_base, group_size, smem_slots, smem_slot_mask,
                          global_slots, global_slot_mask, same_address);
      break;
  }
}

__device__ __forceinline__ uint64_t choose_tile(int iter, int tiles,
                                                int tile_mask,
                                                LocalityMode locality) {
  (void)tiles;
  if (locality == kLocalityHot) {
    return 0;
  }
  if (locality == kLocalityBlockHot) {
    return static_cast<uint64_t>(blockIdx.x & tile_mask);
  }
  return static_cast<uint64_t>((iter * gridDim.x + blockIdx.x) & tile_mask);
}

__device__ __forceinline__ uint32_t run_generic_chain(
    uint8_t* smem_base, const uint8_t* gmem_tile, BenchMode mode, int warp_id,
    int lane, int active_lanes, int groups, int group_size, int smem_slots,
    int smem_slot_mask, int global_slots, int global_slot_mask, int compute_gap,
    bool same_address, uint32_t sink) {
  int copy_index = 0;
#pragma unroll 1
  for (int g = 0; g < groups; ++g) {
    issue_group(smem_base, gmem_tile, warp_id, lane, active_lanes, copy_index,
                group_size, smem_slots, smem_slot_mask, global_slots,
                global_slot_mask, same_address);
    copy_index += group_size;
    cp_async_commit_group();
    sink = burn_int_gap(compute_gap, sink + static_cast<uint32_t>(g));
    if (mode == kModeGroupWait) {
      cp_async_wait_all();
    }
  }
  if (mode == kModeComplete) {
    cp_async_wait_all();
  }
  return sink;
}

__device__ __forceinline__ uint32_t run_fa2_chain(
    uint8_t* smem_base, const uint8_t* gmem_tile, BenchMode mode, int warp_id,
    int lane, int active_lanes, int fa2_stages, int smem_slots,
    int smem_slot_mask, int global_slots, int global_slot_mask, int compute_gap,
    bool same_address, uint32_t sink) {
  int copy_index = 0;
  issue_group_fixed<24>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                        copy_index, smem_slots, smem_slot_mask, global_slots,
                        global_slot_mask, same_address);
  copy_index += 24;
  cp_async_commit_group();
  if (mode == kModeFa2Complete) {
    cp_async_wait_all();
  }

#pragma unroll 1
  for (int stage = 0; stage < fa2_stages; ++stage) {
    issue_group_fixed<8>(smem_base, gmem_tile, warp_id, lane, active_lanes,
                         copy_index, smem_slots, smem_slot_mask, global_slots,
                         global_slot_mask, same_address);
    copy_index += 8;
    cp_async_commit_group();
    sink = burn_int_gap(compute_gap, sink + static_cast<uint32_t>(stage));
    if (mode == kModeFa2Complete) {
      cp_async_wait_all();
    }
  }
  return sink;
}

__global__ void cp_async_latency_kernel(
    const uint8_t* global_data, uint64_t* samples, uint32_t* block_sinks,
    int iters, int warmup, int tiles, size_t tile_bytes, int global_slots,
    int tile_mask, int global_slot_mask, int groups, int group_size,
    int fa2_stages, int active_warps, int active_lanes, int smem_slots,
    int smem_slot_mask, int compute_gap, int mode_i, int locality_i,
    int same_address_i) {
  extern __shared__ uint8_t smem_raw[];

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const BenchMode mode = static_cast<BenchMode>(mode_i);
  const LocalityMode locality = static_cast<LocalityMode>(locality_i);
  const bool same_address = same_address_i != 0;
  uint8_t* smem_base = reinterpret_cast<uint8_t*>(
      align_up(reinterpret_cast<uintptr_t>(smem_raw), 128));
  uint32_t sink = static_cast<uint32_t>(tid + blockIdx.x * 131u);
  const int total_iters = warmup + iters;

  for (int iter = 0; iter < total_iters; ++iter) {
    __syncthreads();
    const uint64_t tile = choose_tile(iter, tiles, tile_mask, locality);
    const uint8_t* gmem_tile = global_data + tile * tile_bytes;

    uint64_t start = 0;
    if (tid == 0) {
      start = clock64_now();
    }

    if (warp_id < active_warps) {
      if (mode == kModeEmpty) {
        sink = burn_int_gap(compute_gap, sink + static_cast<uint32_t>(iter));
      } else if (mode == kModeFa2Issue || mode == kModeFa2Complete) {
        sink = run_fa2_chain(smem_base, gmem_tile, mode, warp_id, lane,
                             active_lanes, fa2_stages, smem_slots,
                             smem_slot_mask, global_slots, global_slot_mask,
                             compute_gap, same_address, sink);
      } else {
        sink = run_generic_chain(smem_base, gmem_tile, mode, warp_id, lane,
                                 active_lanes, groups, group_size, smem_slots,
                                 smem_slot_mask, global_slots, global_slot_mask,
                                 compute_gap, same_address, sink);
      }
    }

    if (tid == 0) {
      const uint64_t end = clock64_now();
      if (iter >= warmup) {
        samples[blockIdx.x * iters + (iter - warmup)] = end - start;
      }
    }

    if (warp_id < active_warps) {
      cp_async_wait_all();
    }
    __syncthreads();

    if (tid == 0) {
      sink += *reinterpret_cast<volatile uint32_t*>(smem_base);
    }
  }

  if (tid == 0) {
    block_sinks[blockIdx.x] = sink;
  }
}

const char* mode_name(BenchMode mode) {
  switch (mode) {
    case kModeEmpty:
      return "empty";
    case kModeIssue:
      return "issue";
    case kModeComplete:
      return "complete";
    case kModeGroupWait:
      return "group-wait";
    case kModeDepth:
      return "depth";
    case kModeFa2Issue:
      return "fa2-issue";
    case kModeFa2Complete:
      return "fa2-complete";
  }
  return "unknown";
}

const char* locality_name(LocalityMode mode) {
  switch (mode) {
    case kLocalityStride:
      return "stride";
    case kLocalityHot:
      return "hot";
    case kLocalityBlockHot:
      return "block-hot";
  }
  return "unknown";
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device N] [--blocks N] [--threads N] [--iters N]\n"
      "          [--warmup N] [--tiles N]\n"
      "          [--mode empty|issue|complete|group-wait|depth|fa2-issue|"
      "fa2-complete]\n"
      "          [--groups N] [--group-size N] [--fa2-stages N]\n"
      "          [--active-warps N] [--active-lanes N]\n"
      "          [--smem-slots N] [--global-slots N] [--compute-gap N]\n"
      "          [--same-address] [--hot|--block-hot|--stride] [--csv path]\n",
      argv0);
}

bool parse_int_arg(const char* arg, const char* name, int* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = std::atoi(arg + n + 1);
    return true;
  }
  return false;
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else if (parse_int_arg(arg, "--device", &opts.device) ||
               parse_int_arg(arg, "--blocks", &opts.blocks) ||
               parse_int_arg(arg, "--threads", &opts.threads) ||
               parse_int_arg(arg, "--iters", &opts.iters) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--tiles", &opts.tiles) ||
               parse_int_arg(arg, "--groups", &opts.groups) ||
               parse_int_arg(arg, "--group-size", &opts.group_size) ||
               parse_int_arg(arg, "--fa2-stages", &opts.fa2_stages) ||
               parse_int_arg(arg, "--active-warps", &opts.active_warps) ||
               parse_int_arg(arg, "--active-lanes", &opts.active_lanes) ||
               parse_int_arg(arg, "--smem-slots", &opts.smem_slots) ||
               parse_int_arg(arg, "--global-slots", &opts.global_slots) ||
               parse_int_arg(arg, "--compute-gap", &opts.compute_gap)) {
      continue;
    } else if (std::strncmp(arg, "--mode=", 7) == 0) {
      const char* value = arg + 7;
      if (std::strcmp(value, "empty") == 0) {
        opts.mode = kModeEmpty;
      } else if (std::strcmp(value, "issue") == 0) {
        opts.mode = kModeIssue;
      } else if (std::strcmp(value, "complete") == 0) {
        opts.mode = kModeComplete;
      } else if (std::strcmp(value, "group-wait") == 0) {
        opts.mode = kModeGroupWait;
      } else if (std::strcmp(value, "depth") == 0) {
        opts.mode = kModeDepth;
      } else if (std::strcmp(value, "fa2-issue") == 0) {
        opts.mode = kModeFa2Issue;
      } else if (std::strcmp(value, "fa2-complete") == 0) {
        opts.mode = kModeFa2Complete;
      } else {
        std::fprintf(stderr, "Unknown mode: %s\n", value);
        std::exit(1);
      }
    } else if (std::strcmp(arg, "--hot") == 0) {
      opts.locality = kLocalityHot;
    } else if (std::strcmp(arg, "--block-hot") == 0) {
      opts.locality = kLocalityBlockHot;
    } else if (std::strcmp(arg, "--stride") == 0) {
      opts.locality = kLocalityStride;
    } else if (std::strcmp(arg, "--same-address") == 0) {
      opts.same_address = true;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.blocks <= 0 || opts.threads <= 0 || opts.iters <= 0 ||
      opts.warmup < 0 || opts.tiles <= 0 || opts.groups <= 0 ||
      opts.group_size <= 0 || opts.fa2_stages < 0 || opts.active_warps <= 0 ||
      opts.active_lanes <= 0 || opts.smem_slots <= 0 || opts.global_slots < 0 ||
      opts.compute_gap < 0) {
    std::fprintf(stderr,
                 "numeric args must be positive except warmup/fa2-stages/"
                 "global-slots/compute-gap may be zero\n");
    std::exit(1);
  }
  if (opts.threads % kWarpSize != 0) {
    std::fprintf(stderr, "threads must be a multiple of %d\n", kWarpSize);
    std::exit(1);
  }
  if (opts.active_warps > opts.threads / kWarpSize) {
    std::fprintf(stderr, "active-warps cannot exceed threads/32\n");
    std::exit(1);
  }
  if (opts.active_lanes > kWarpSize) {
    std::fprintf(stderr, "active-lanes cannot exceed %d\n", kWarpSize);
    std::exit(1);
  }
  return opts;
}

uint64_t cp_async_per_warp_iter(const Options& opts) {
  if (opts.mode == kModeEmpty) {
    return 0;
  }
  if (opts.mode == kModeFa2Issue || opts.mode == kModeFa2Complete) {
    return static_cast<uint64_t>(24 + opts.fa2_stages * 8);
  }
  return static_cast<uint64_t>(opts.groups) *
         static_cast<uint64_t>(opts.group_size);
}

uint64_t commit_groups_per_warp_iter(const Options& opts) {
  if (opts.mode == kModeEmpty) {
    return 0;
  }
  if (opts.mode == kModeFa2Issue || opts.mode == kModeFa2Complete) {
    return static_cast<uint64_t>(1 + opts.fa2_stages);
  }
  return static_cast<uint64_t>(opts.groups);
}

bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int next_power_of_two(uint64_t value) {
  uint64_t power = 1;
  while (power < value) {
    power <<= 1;
  }
  if (power > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr, "slot count is too large\n");
    std::exit(1);
  }
  return static_cast<int>(power);
}

int default_global_slots(const Options& opts) {
  const uint64_t cp = cp_async_per_warp_iter(opts);
  const uint64_t slots =
      std::max<uint64_t>(static_cast<uint64_t>(opts.smem_slots), cp);
  return next_power_of_two(slots);
}

size_t dynamic_smem_bytes(int threads, int smem_slots) {
  const int warps = threads / kWarpSize;
  uintptr_t p = 0;
  p = align_up(p, 128);
  p += static_cast<uintptr_t>(warps) * static_cast<uintptr_t>(smem_slots) *
       kWarpSize * kCpAsyncBytes;
  return align_up(p, 1024);
}

size_t tile_bytes(int active_warps, int global_slots) {
  const uintptr_t bytes = static_cast<uintptr_t>(active_warps) *
                          static_cast<uintptr_t>(global_slots) * kWarpSize *
                          kCpAsyncBytes;
  return align_up(bytes, 128);
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double q) {
  if (sorted.empty()) {
    return 0;
  }
  const double pos = q * static_cast<double>(sorted.size() - 1);
  return sorted[static_cast<size_t>(std::llround(pos))];
}

double mean_cycles(const std::vector<uint64_t>& values) {
  const double sum =
      std::accumulate(values.begin(), values.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  return sum / static_cast<double>(values.size());
}

void print_stats(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double mean = mean_cycles(sorted);
  double sq = 0.0;
  for (uint64_t v : sorted) {
    const double d = static_cast<double>(v) - mean;
    sq += d * d;
  }
  const double stdev = std::sqrt(sq / static_cast<double>(sorted.size()));

  std::printf(
      "samples=%zu mean=%.2f stdev=%.2f min=%llu p01=%llu p05=%llu "
      "p50=%llu p90=%llu p95=%llu p99=%llu p999=%llu max=%llu\n",
      sorted.size(), mean, stdev,
      static_cast<unsigned long long>(sorted.front()),
      static_cast<unsigned long long>(percentile(sorted, 0.01)),
      static_cast<unsigned long long>(percentile(sorted, 0.05)),
      static_cast<unsigned long long>(percentile(sorted, 0.50)),
      static_cast<unsigned long long>(percentile(sorted, 0.90)),
      static_cast<unsigned long long>(percentile(sorted, 0.95)),
      static_cast<unsigned long long>(percentile(sorted, 0.99)),
      static_cast<unsigned long long>(percentile(sorted, 0.999)),
      static_cast<unsigned long long>(sorted.back()));
}

void write_csv(const std::string& path, const std::vector<uint64_t>& values,
               int blocks, int iters) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "block,iter,cycles\n";
  for (int b = 0; b < blocks; ++b) {
    for (int i = 0; i < iters; ++i) {
      out << b << "," << i << "," << values[b * iters + i] << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opts = parse_options(argc, argv);
  if (opts.global_slots == 0) {
    opts.global_slots = default_global_slots(opts);
  }
  if (!is_power_of_two(opts.tiles) || !is_power_of_two(opts.smem_slots) ||
      !is_power_of_two(opts.global_slots)) {
    std::fprintf(stderr,
                 "tiles, smem-slots, and global-slots must be powers of two "
                 "so the benchmark address ring compiles to bit masks\n");
    std::exit(1);
  }

  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate,
                                    opts.device));

  const size_t smem_bytes = dynamic_smem_bytes(opts.threads, opts.smem_slots);
  const size_t per_tile_bytes =
      tile_bytes(opts.active_warps, opts.global_slots);
  const int tile_mask = opts.tiles - 1;
  const int smem_slot_mask = opts.smem_slots - 1;
  const int global_slot_mask = opts.global_slots - 1;
  const size_t global_bytes =
      per_tile_bytes * static_cast<size_t>(opts.tiles) + 128;
  const size_t sample_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.iters);

  CUDA_CHECK(cudaFuncSetAttribute(cp_async_latency_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(
      cp_async_latency_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));

  uint8_t* d_data = nullptr;
  uint64_t* d_samples = nullptr;
  uint32_t* d_sinks = nullptr;

  CUDA_CHECK(cudaMalloc(&d_data, global_bytes));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_data, 0x5a, global_bytes));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sinks, 0,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));

  const uint64_t cp_per_warp = cp_async_per_warp_iter(opts);
  const uint64_t commits_per_warp = commit_groups_per_warp_iter(opts);
  const uint64_t cp_per_block =
      cp_per_warp * static_cast<uint64_t>(opts.active_warps);
  const uint64_t active_bytes_per_iter =
      cp_per_block * static_cast<uint64_t>(opts.active_lanes) * kCpAsyncBytes;
  const uint64_t measured_data_bytes = active_bytes_per_iter *
                                       static_cast<uint64_t>(opts.blocks) *
                                       static_cast<uint64_t>(opts.iters);

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "mode=%s locality=%s blocks=%d threads=%d iters=%d warmup=%d tiles=%d "
      "groups=%d group_size=%d fa2_stages=%d active_warps=%d "
      "active_lanes=%d smem_slots=%d global_slots=%d compute_gap=%d "
      "same_address=%d\n",
      mode_name(opts.mode), locality_name(opts.locality), opts.blocks,
      opts.threads, opts.iters, opts.warmup, opts.tiles, opts.groups,
      opts.group_size, opts.fa2_stages, opts.active_warps, opts.active_lanes,
      opts.smem_slots, opts.global_slots, opts.compute_gap,
      opts.same_address ? 1 : 0);
  std::printf(
      "smem=%zu bytes tile_bytes=%zu global_bytes=%zu cp_async_per_warp=%llu "
      "commit_groups_per_warp=%llu active_bytes_per_iter=%llu "
      "measured_data_bytes=%llu\n",
      smem_bytes, per_tile_bytes, global_bytes,
      static_cast<unsigned long long>(cp_per_warp),
      static_cast<unsigned long long>(commits_per_warp),
      static_cast<unsigned long long>(active_bytes_per_iter),
      static_cast<unsigned long long>(measured_data_bytes));
  if (opts.mode == kModeIssue || opts.mode == kModeDepth ||
      opts.mode == kModeFa2Issue) {
    std::printf(
        "timing=issue region; final cp.async.wait_group 0 is outside "
        "the measured interval\n");
  } else if (opts.mode == kModeComplete || opts.mode == kModeFa2Complete) {
    std::printf("timing=issue+commit+completion wait in measured interval\n");
  } else if (opts.mode == kModeGroupWait) {
    std::printf(
        "timing=issue+commit+wait after every group in measured "
        "interval\n");
  }

  cp_async_latency_kernel<<<opts.blocks, opts.threads, smem_bytes>>>(
      d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.tiles,
      per_tile_bytes, opts.global_slots, tile_mask, global_slot_mask,
      opts.groups, opts.group_size, opts.fa2_stages, opts.active_warps,
      opts.active_lanes, opts.smem_slots, smem_slot_mask, opts.compute_gap,
      static_cast<int>(opts.mode), static_cast<int>(opts.locality),
      opts.same_address ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(opts.blocks);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        sample_count * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  print_stats(samples);
  const double mean = mean_cycles(samples);
  if (cp_per_warp > 0 && mean > 0.0) {
    std::printf("mean_cycles_per_cp_async_per_warp=%.4f\n",
                mean / static_cast<double>(cp_per_warp));
  }
  if (commits_per_warp > 0 && mean > 0.0) {
    std::printf("mean_cycles_per_commit_group_per_warp=%.4f\n",
                mean / static_cast<double>(commits_per_warp));
  }
  if (mean > 0.0) {
    const double block_inst_per_cycle =
        static_cast<double>(cp_per_block) / mean;
    const double block_bytes_per_cycle =
        static_cast<double>(active_bytes_per_iter) / mean;
    const double block_gib_per_sec = block_bytes_per_cycle *
                                     static_cast<double>(clock_rate_khz) *
                                     1000.0 / (1024.0 * 1024.0 * 1024.0);
    std::printf(
        "block_cp_async_inst_per_cycle=%.4f block_bytes_per_cycle=%.2f "
        "block_gib_per_sec=%.2f\n",
        block_inst_per_cycle, block_bytes_per_cycle, block_gib_per_sec);
  }

  const uint64_t sink_sum =
      std::accumulate(sinks.begin(), sinks.end(), uint64_t{0});
  std::printf("sink=%llu\n", static_cast<unsigned long long>(sink_sum));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.blocks, opts.iters);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
