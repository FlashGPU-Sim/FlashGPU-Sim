#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kLoadBytes = 16;
constexpr int kCpAsyncGroup = 8;
constexpr int kDefaultThreads = 256;

enum class Pattern {
  kStream,
  kStride,
  kPair,
  kFixedPair,
};

enum class AccessOp {
  kLdg,
  kCpAsync,
};

enum class AllocMode {
  kCudaMalloc,
  kVmm,
};

struct Options {
  int device = 0;
  int blocks = 528;
  int threads = kDefaultThreads;
  int iters = 64;
  int warmup = 0;
  int repeat = 1;
  size_t data_bytes = 512ULL * 1024 * 1024;
  size_t tile_bytes = 48ULL * 1024;
  size_t stride_bytes = 128;
  size_t pair_delta_bytes = 4096;
  size_t base_offset = 0;
  size_t smem_bytes = 32768;
  Pattern pattern = Pattern::kStream;
  AccessOp op = AccessOp::kLdg;
  AllocMode alloc_mode = AllocMode::kCudaMalloc;
  bool event_timing = false;
  std::string csv_path;
  std::string batch_mode;
  std::string batch_csv_path;
  std::string pair_csv_path;
  int bit_first = 7;
  int bit_last = 20;
  int batch_random_pairs = 0;
  unsigned batch_seed = 1;
  size_t batch_line_count = 16384;
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

#define CU_CHECK(expr)                                                    \
  do {                                                                    \
    CUresult err__ = (expr);                                              \
    if (err__ != CUDA_SUCCESS) {                                          \
      const char* name__ = nullptr;                                       \
      const char* str__ = nullptr;                                        \
      cuGetErrorName(err__, &name__);                                     \
      cuGetErrorString(err__, &str__);                                    \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s (%s)\n", __FILE__, \
                   __LINE__, name__ ? name__ : "unknown",                \
                   str__ ? str__ : "unknown");                           \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint4 ld_global_cg_u4(const void* ptr) {
  uint4 value;
  const uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_global(ptr));
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(addr)
               : "memory");
  return value;
}

__device__ __forceinline__ uint32_t fold_u4(uint4 value) {
  uint32_t sink = value.x;
  sink ^= value.y;
  sink += value.z;
  sink ^= value.w;
  return sink;
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void* ptr) {
  uint64_t out = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(out) : "l"(ptr));
  return out;
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr,
                                               uint64_t gmem_addr) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                   smem_addr),
               "l"(gmem_addr)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t wrap_addr(uint64_t offset,
                                              uint64_t data_bytes) {
  const uint64_t max_offset = data_bytes - kLoadBytes;
  if (offset <= max_offset) return offset;
  return offset % max_offset;
}

__device__ __forceinline__ uint64_t compute_offset(Pattern pattern,
                                                   uint64_t base_offset,
                                                   uint64_t data_bytes,
                                                   uint64_t tile_bytes,
                                                   uint64_t stride_bytes,
                                                   uint64_t pair_delta_bytes,
                                                   uint64_t iter,
                                                   uint64_t repeat_count,
                                                   uint64_t repeat_idx,
                                                   uint64_t load_idx) {
  const uint64_t tile_id =
      (iter * static_cast<uint64_t>(gridDim.x) + blockIdx.x);
  const uint64_t repeated_tile = tile_id * repeat_count + repeat_idx;
  uint64_t offset = base_offset;
  if (pattern == Pattern::kStream) {
    offset += repeated_tile * tile_bytes + load_idx * kLoadBytes;
  } else if (pattern == Pattern::kStride) {
    const uint64_t logical =
        repeated_tile * (tile_bytes / kLoadBytes) + load_idx;
    offset += logical * stride_bytes;
  } else if (pattern == Pattern::kPair) {
    const uint64_t logical =
        repeated_tile * (tile_bytes / kLoadBytes) + load_idx;
    offset += (logical >> 1) * stride_bytes;
    if (logical & 1ULL) offset += pair_delta_bytes;
  } else {
    if (load_idx & 1ULL) offset += pair_delta_bytes;
  }
  return wrap_addr(offset, data_bytes);
}

template <AccessOp Op>
__global__ void interleave_probe_kernel(
    const uint8_t* __restrict__ data, uint64_t* __restrict__ samples,
    uint32_t* __restrict__ thread_sinks, int iters, int warmup, int repeat,
    uint64_t data_bytes, uint64_t tile_bytes, uint64_t stride_bytes,
    uint64_t pair_delta_bytes, uint64_t base_offset, int pattern_i,
    uint64_t smem_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x;
  const Pattern pattern = static_cast<Pattern>(pattern_i);
  const uint64_t logical_loads = tile_bytes / kLoadBytes;
  const int total_iters = warmup + iters;
  uint32_t sink = static_cast<uint32_t>(blockIdx.x * blockDim.x + tid);

  for (int i = 0; i < total_iters; ++i) {
    __syncthreads();
    uint64_t start = 0;
    if (tid == 0) start = clock64_now();

    for (int r = 0; r < repeat; ++r) {
      if constexpr (Op == AccessOp::kLdg) {
        for (uint64_t load = tid; load < logical_loads; load += blockDim.x) {
          const uint64_t offset =
              compute_offset(pattern, base_offset, data_bytes, tile_bytes,
                             stride_bytes, pair_delta_bytes, i, repeat, r,
                             load);
          sink ^= fold_u4(ld_global_cg_u4(data + offset));
        }
      } else {
        int pending = 0;
        const uint64_t thread_slot_base =
            (static_cast<uint64_t>(tid) * kCpAsyncGroup * kLoadBytes) %
            smem_bytes;
        for (uint64_t load = tid; load < logical_loads; load += blockDim.x) {
          const uint64_t offset =
              compute_offset(pattern, base_offset, data_bytes, tile_bytes,
                             stride_bytes, pair_delta_bytes, i, repeat, r,
                             load);
          const uint64_t smem_offset =
              (thread_slot_base +
               static_cast<uint64_t>(pending) * kLoadBytes) %
              smem_bytes;
          cp_async_cg_16(smem_addr_u32(smem + smem_offset),
                         gmem_addr_u64(data + offset));
          ++pending;
          if (pending == kCpAsyncGroup) {
            cp_async_commit_group();
            cp_async_wait_all();
            sink ^= *reinterpret_cast<volatile uint32_t*>(smem + thread_slot_base);
            pending = 0;
          }
        }
        if (pending != 0) {
          cp_async_commit_group();
          cp_async_wait_all();
          sink ^= *reinterpret_cast<volatile uint32_t*>(smem + thread_slot_base);
        }
      }
    }

    __syncthreads();
    if (tid == 0 && i >= warmup) {
      const uint64_t end = clock64_now();
      samples[blockIdx.x * iters + (i - warmup)] = end - start;
    }
  }

  thread_sinks[blockIdx.x * blockDim.x + tid] = sink;
}

const char* pattern_name(Pattern pattern) {
  switch (pattern) {
    case Pattern::kStream:
      return "stream";
    case Pattern::kStride:
      return "stride";
    case Pattern::kPair:
      return "pair";
    case Pattern::kFixedPair:
      return "fixed_pair";
  }
  return "unknown";
}

const char* op_name(AccessOp op) {
  switch (op) {
    case AccessOp::kLdg:
      return "ldg";
    case AccessOp::kCpAsync:
      return "cp_async";
  }
  return "unknown";
}

const char* alloc_mode_name(AllocMode mode) {
  switch (mode) {
    case AllocMode::kCudaMalloc:
      return "cudaMalloc";
    case AllocMode::kVmm:
      return "vmm";
  }
  return "unknown";
}

size_t align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct DeviceAllocation {
  uint8_t* ptr = nullptr;
  size_t bytes = 0;
  AllocMode mode = AllocMode::kCudaMalloc;
  CUdeviceptr vmm_addr = 0;
  CUmemGenericAllocationHandle vmm_handle = 0;
  bool has_vmm_handle = false;
};

DeviceAllocation allocate_device_data(const Options& opts) {
  DeviceAllocation allocation;
  allocation.mode = opts.alloc_mode;
  allocation.bytes = opts.data_bytes;
  if (opts.alloc_mode == AllocMode::kCudaMalloc) {
    CUDA_CHECK(cudaMalloc(&allocation.ptr, allocation.bytes));
    return allocation;
  }

  CU_CHECK(cuInit(0));
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = opts.device;
  size_t min_granularity = 0;
  size_t recommended_granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(
      &min_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CU_CHECK(cuMemGetAllocationGranularity(
      &recommended_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  const size_t granularity =
      recommended_granularity ? recommended_granularity : min_granularity;
  allocation.bytes = align_up(opts.data_bytes, granularity);

  CU_CHECK(cuMemAddressReserve(&allocation.vmm_addr, allocation.bytes,
                               granularity, 0, 0));
  CU_CHECK(cuMemCreate(&allocation.vmm_handle, allocation.bytes, &prop, 0));
  allocation.has_vmm_handle = true;
  CU_CHECK(cuMemMap(allocation.vmm_addr, allocation.bytes, 0,
                    allocation.vmm_handle, 0));
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = opts.device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(allocation.vmm_addr, allocation.bytes, &access, 1));
  allocation.ptr = reinterpret_cast<uint8_t*>(allocation.vmm_addr);
  std::printf("vmm_min_granularity=%zu vmm_recommended_granularity=%zu "
              "vmm_allocated_bytes=%zu\n",
              min_granularity, recommended_granularity, allocation.bytes);
  return allocation;
}

void free_device_data(DeviceAllocation* allocation) {
  if (allocation->ptr == nullptr) return;
  if (allocation->mode == AllocMode::kCudaMalloc) {
    CUDA_CHECK(cudaFree(allocation->ptr));
  } else {
    CU_CHECK(cuMemUnmap(allocation->vmm_addr, allocation->bytes));
    if (allocation->has_vmm_handle) {
      CU_CHECK(cuMemRelease(allocation->vmm_handle));
    }
    CU_CHECK(cuMemAddressFree(allocation->vmm_addr, allocation->bytes));
  }
  *allocation = DeviceAllocation{};
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device=N] [--op=ldg|cp_async]\n"
      "          [--pattern=stream|stride|pair|fixed_pair]\n"
      "          [--blocks=N] [--threads=N] [--iters=N] [--warmup=N]\n"
      "          [--repeat=N] [--data-bytes=N] [--tile-bytes=N]\n"
      "          [--stride-bytes=N] [--pair-delta-bytes=N]\n"
      "          [--base-offset=N] [--smem-bytes=N]\n"
      "          [--alloc=cuda|vmm] [--event] [--csv=path]\n"
      "          [--batch=base_mask|derivative|random|all]\n"
      "          [--batch-csv=path] [--pair-csv=path]\n"
      "          [--bit-first=N] [--bit-last=N]\n"
      "          [--batch-line-count=N] [--batch-random-pairs=N]\n"
      "          [--batch-seed=N]\n\n"
      "Examples:\n"
      "  %s --op=ldg --pattern=stream --data-bytes=536870912 --tile-bytes=49152\n"
      "  %s --op=ldg --pattern=stride --stride-bytes=4096\n"
      "  %s --op=ldg --pattern=fixed_pair --pair-delta-bytes=65536\n"
      "  %s --op=cp_async --pattern=stream --smem-bytes=32768\n",
      argv0, argv0, argv0, argv0, argv0);
}

bool parse_int_arg(const char* arg, const char* name, int* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = std::atoi(arg + n + 1);
    return true;
  }
  return false;
}

bool parse_size_arg(const char* arg, const char* name, size_t* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(arg + n + 1, &end, 0);
    if (end == arg + n + 1) {
      std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
      std::exit(1);
    }
    if (*end == 'k' || *end == 'K') {
      parsed *= 1024ULL;
      ++end;
    } else if (*end == 'm' || *end == 'M') {
      parsed *= 1024ULL * 1024ULL;
      ++end;
    } else if (*end == 'g' || *end == 'G') {
      parsed *= 1024ULL * 1024ULL * 1024ULL;
      ++end;
    }
    if (*end != '\0') {
      std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
      std::exit(1);
    }
    *value = static_cast<size_t>(parsed);
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
               parse_int_arg(arg, "--repeat", &opts.repeat)) {
      continue;
    } else if (parse_size_arg(arg, "--data-bytes", &opts.data_bytes) ||
               parse_size_arg(arg, "--tile-bytes", &opts.tile_bytes) ||
               parse_size_arg(arg, "--stride-bytes", &opts.stride_bytes) ||
	               parse_size_arg(arg, "--pair-delta-bytes",
	                              &opts.pair_delta_bytes) ||
	               parse_size_arg(arg, "--base-offset", &opts.base_offset) ||
	               parse_size_arg(arg, "--smem-bytes", &opts.smem_bytes) ||
	               parse_size_arg(arg, "--batch-line-count",
	                              &opts.batch_line_count)) {
	      continue;
	    } else if (parse_int_arg(arg, "--bit-first", &opts.bit_first) ||
	               parse_int_arg(arg, "--bit-last", &opts.bit_last) ||
	               parse_int_arg(arg, "--batch-random-pairs",
	                             &opts.batch_random_pairs)) {
	      continue;
	    } else if (std::strncmp(arg, "--batch-seed=", 13) == 0) {
	      opts.batch_seed =
	          static_cast<unsigned>(std::strtoul(arg + 13, nullptr, 0));
	    } else if (std::strcmp(arg, "--op=ldg") == 0) {
	      opts.op = AccessOp::kLdg;
    } else if (std::strcmp(arg, "--op=cp_async") == 0 ||
               std::strcmp(arg, "--op=cp.async") == 0) {
      opts.op = AccessOp::kCpAsync;
    } else if (std::strcmp(arg, "--pattern=stream") == 0) {
      opts.pattern = Pattern::kStream;
    } else if (std::strcmp(arg, "--pattern=stride") == 0) {
      opts.pattern = Pattern::kStride;
    } else if (std::strcmp(arg, "--pattern=pair") == 0) {
      opts.pattern = Pattern::kPair;
    } else if (std::strcmp(arg, "--pattern=fixed_pair") == 0 ||
               std::strcmp(arg, "--pattern=fixed-pair") == 0) {
      opts.pattern = Pattern::kFixedPair;
    } else if (std::strcmp(arg, "--alloc=cuda") == 0 ||
               std::strcmp(arg, "--alloc=cudaMalloc") == 0 ||
               std::strcmp(arg, "--alloc=cudamalloc") == 0) {
      opts.alloc_mode = AllocMode::kCudaMalloc;
    } else if (std::strcmp(arg, "--alloc=vmm") == 0) {
      opts.alloc_mode = AllocMode::kVmm;
	    } else if (std::strcmp(arg, "--event") == 0) {
	      opts.event_timing = true;
	    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
	      opts.csv_path = arg + 6;
	    } else if (std::strncmp(arg, "--batch=", 8) == 0) {
	      opts.batch_mode = arg + 8;
	    } else if (std::strncmp(arg, "--batch-csv=", 12) == 0) {
	      opts.batch_csv_path = arg + 12;
	    } else if (std::strncmp(arg, "--pair-csv=", 11) == 0) {
	      opts.pair_csv_path = arg + 11;
	    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  opts.data_bytes -= opts.data_bytes % kLoadBytes;
  opts.tile_bytes -= opts.tile_bytes % kLoadBytes;
  opts.stride_bytes -= opts.stride_bytes % kLoadBytes;
  opts.pair_delta_bytes -= opts.pair_delta_bytes % kLoadBytes;
  opts.base_offset -= opts.base_offset % kLoadBytes;
  opts.smem_bytes -= opts.smem_bytes % kLoadBytes;
  if (opts.blocks <= 0 || opts.threads <= 0 || opts.iters <= 0 ||
      opts.warmup < 0 || opts.repeat <= 0 || opts.data_bytes < 2 * kLoadBytes ||
      opts.tile_bytes < kLoadBytes || opts.stride_bytes == 0 ||
      opts.smem_bytes < static_cast<size_t>(opts.threads) * kCpAsyncGroup *
                            kLoadBytes) {
    std::fprintf(stderr,
                 "invalid options: positive blocks/threads/iters/repeat, "
                 "data >= 32B, tile >= 16B, stride > 0, and smem >= "
                 "threads*%d*16 are required\n",
                 kCpAsyncGroup);
    std::exit(1);
  }
  if (opts.threads % 32 != 0 || opts.threads > 1024) {
    std::fprintf(stderr,
                 "threads must be a positive multiple of 32 and <= 1024\n");
    std::exit(1);
  }
  return opts;
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string field;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      in_quotes = !in_quotes;
    } else if (c == ',' && !in_quotes) {
      out.push_back(field);
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  out.push_back(field);
  return out;
}

int find_column(const std::vector<std::string>& header, const char* name) {
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == name) return static_cast<int>(i);
  }
  return -1;
}

std::string get_csv_field(const std::vector<std::string>& row, int col,
                          const char* fallback = "") {
  if (col < 0 || static_cast<size_t>(col) >= row.size()) return fallback;
  return row[col];
}

uint64_t parse_u64_field(const std::vector<std::string>& row, int col,
                         uint64_t fallback = 0) {
  if (col < 0 || static_cast<size_t>(col) >= row.size() || row[col].empty()) {
    return fallback;
  }
  return std::strtoull(row[col].c_str(), nullptr, 0);
}

int parse_i32_field(const std::vector<std::string>& row, int col,
                    int fallback = -1) {
  if (col < 0 || static_cast<size_t>(col) >= row.size() || row[col].empty()) {
    return fallback;
  }
  return std::atoi(row[col].c_str());
}

struct PairSpec {
  std::string kind;
  std::string tag;
  int bit = -1;
  uint64_t page_left = 0;
  uint64_t page_right = 0;
  uint64_t line_left = 0;
  uint64_t line_right = 0;
  uint64_t left_offset = 0;
  uint64_t right_offset = 0;
};

std::vector<PairSpec> read_pair_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "Failed to open --pair-csv path: %s\n",
                 path.c_str());
    std::exit(1);
  }

  std::string line;
  if (!std::getline(in, line)) return {};
  const std::vector<std::string> header = split_csv_line(line);
  const int kind_col = find_column(header, "kind");
  const int tag_col = find_column(header, "tag");
  const int bit_col = find_column(header, "bit");
  const int page_left_col = find_column(header, "page_left");
  const int page_right_col = find_column(header, "page_right");
  const int line_left_col = find_column(header, "line_left");
  const int line_right_col = find_column(header, "line_right");
  const int left_offset_col = find_column(header, "left_offset");
  const int right_offset_col = find_column(header, "right_offset");

  if ((left_offset_col < 0 || right_offset_col < 0) &&
      (page_left_col < 0 || page_right_col < 0 || line_left_col < 0 ||
       line_right_col < 0)) {
    std::fprintf(stderr,
                 "--pair-csv needs left_offset/right_offset or "
                 "page_left/page_right/line_left/line_right columns\n");
    std::exit(1);
  }

  std::vector<PairSpec> specs;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> row = split_csv_line(line);
    PairSpec spec;
    spec.kind = get_csv_field(row, kind_col, "pair");
    spec.tag = get_csv_field(row, tag_col, "");
    spec.bit = parse_i32_field(row, bit_col, -1);
    spec.page_left = parse_u64_field(row, page_left_col, 0);
    spec.page_right = parse_u64_field(row, page_right_col, 0);
    spec.line_left = parse_u64_field(row, line_left_col, 0);
    spec.line_right = parse_u64_field(row, line_right_col, 0);
    if (left_offset_col >= 0 && right_offset_col >= 0) {
      spec.left_offset = parse_u64_field(row, left_offset_col, 0);
      spec.right_offset = parse_u64_field(row, right_offset_col, 0);
    } else {
      constexpr uint64_t kPageBytes = 2ULL * 1024ULL * 1024ULL;
      constexpr uint64_t kLineBytes = 128ULL;
      spec.left_offset = spec.page_left * kPageBytes + spec.line_left * kLineBytes;
      spec.right_offset =
          spec.page_right * kPageBytes + spec.line_right * kLineBytes;
    }
    specs.push_back(spec);
  }
  return specs;
}

float run_probe_once(const Options& opts, uint8_t* d_data, uint64_t* d_samples,
                     uint32_t* d_sinks, cudaEvent_t start, cudaEvent_t stop,
                     size_t base_offset, size_t pair_delta_bytes) {
  const int pattern_i = static_cast<int>(Pattern::kFixedPair);
  CUDA_CHECK(cudaEventRecord(start));
  if (opts.op == AccessOp::kLdg) {
    interleave_probe_kernel<AccessOp::kLdg>
        <<<opts.blocks, opts.threads, opts.smem_bytes>>>(
            d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.repeat,
            opts.data_bytes, opts.tile_bytes, opts.stride_bytes,
            pair_delta_bytes, base_offset, pattern_i, opts.smem_bytes);
  } else {
    interleave_probe_kernel<AccessOp::kCpAsync>
        <<<opts.blocks, opts.threads, opts.smem_bytes>>>(
            d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.repeat,
            opts.data_bytes, opts.tile_bytes, opts.stride_bytes,
            pair_delta_bytes, base_offset, pattern_i, opts.smem_bytes);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return elapsed_ms;
}

void run_pair_csv_batch(const Options& opts, uint8_t* d_data,
                        uint64_t* d_samples, uint32_t* d_sinks) {
  if (opts.batch_csv_path.empty()) {
    std::fprintf(stderr, "--batch-csv is required with --pair-csv\n");
    std::exit(1);
  }
  const std::vector<PairSpec> specs = read_pair_csv(opts.pair_csv_path);
  if (specs.empty()) {
    std::fprintf(stderr, "--pair-csv has no pair rows: %s\n",
                 opts.pair_csv_path.c_str());
    std::exit(1);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::ofstream out(opts.batch_csv_path);
  out << "launch_index,kind,tag,bit,page_left,page_right,line_left,line_right,"
         "left_offset,right_offset,base_offset,pair_delta_bytes,event_ms\n";

  for (size_t i = 0; i < specs.size(); ++i) {
    const PairSpec& spec = specs[i];
    if (spec.left_offset + kLoadBytes > opts.data_bytes ||
        spec.right_offset + kLoadBytes > opts.data_bytes) {
      std::fprintf(stderr,
                   "pair row %zu is out of range: left=%llu right=%llu "
                   "data_bytes=%zu\n",
                   i, static_cast<unsigned long long>(spec.left_offset),
                   static_cast<unsigned long long>(spec.right_offset),
                   opts.data_bytes);
      std::exit(1);
    }
    const uint64_t base_offset = std::min(spec.left_offset, spec.right_offset);
    const uint64_t pair_delta_bytes =
        spec.left_offset > spec.right_offset ? spec.left_offset - spec.right_offset
                                             : spec.right_offset - spec.left_offset;
    const float elapsed_ms =
        run_probe_once(opts, d_data, d_samples, d_sinks, start, stop,
                       base_offset, pair_delta_bytes);
    out << i << "," << spec.kind << "," << spec.tag << "," << spec.bit << ","
        << spec.page_left << "," << spec.page_right << "," << spec.line_left
        << "," << spec.line_right << "," << spec.left_offset << ","
        << spec.right_offset << "," << base_offset << ","
        << pair_delta_bytes << "," << elapsed_ms << "\n";
    if (((i + 1) & 0xffULL) == 0) {
      std::fprintf(stderr, "pair_csv_progress=%zu/%zu\n", i + 1,
                   specs.size());
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::fprintf(stderr, "pair_csv_total=%zu\n", specs.size());
}

void run_fixed_pair_batch(const Options& opts, uint8_t* d_data,
                          uint64_t* d_samples, uint32_t* d_sinks) {
  if (opts.batch_csv_path.empty()) {
    std::fprintf(stderr, "--batch-csv is required for --batch\n");
    std::exit(1);
  }
  if (opts.batch_line_count == 0 ||
      opts.batch_line_count * 128ULL > opts.data_bytes) {
    std::fprintf(stderr,
                 "Invalid --batch-line-count=%zu for data_bytes=%zu\n",
                 opts.batch_line_count, opts.data_bytes);
    std::exit(1);
  }
  if (opts.bit_first < 7 || opts.bit_last < opts.bit_first ||
      opts.bit_last > 62) {
    std::fprintf(stderr, "Invalid bit range [%d, %d]\n", opts.bit_first,
                 opts.bit_last);
    std::exit(1);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::ofstream out(opts.batch_csv_path);
  out << "kind,bit,left_line,right_line,left_offset,right_offset,base_offset,"
         "pair_delta_bytes,event_ms\n";

  size_t launched = 0;
  auto progress = [&]() {
    ++launched;
    if ((launched & 0x3fffULL) == 0) {
      std::fprintf(stderr, "batch_progress=%zu\n", launched);
    }
  };
  auto run_pair = [&](const char* kind, int bit, size_t left_line,
                      size_t right_line) {
    const size_t left_offset = left_line * 128ULL;
    const size_t right_offset = right_line * 128ULL;
    const size_t base_offset = std::min(left_offset, right_offset);
    const size_t pair_delta_bytes =
        left_offset > right_offset ? left_offset - right_offset
                                   : right_offset - left_offset;
    const float elapsed_ms =
        run_probe_once(opts, d_data, d_samples, d_sinks, start, stop,
                       base_offset, pair_delta_bytes);
    out << kind << "," << bit << "," << left_line << "," << right_line << ","
        << left_offset << "," << right_offset << "," << base_offset << ","
        << pair_delta_bytes << "," << elapsed_ms << "\n";
    progress();
  };

  if (opts.batch_mode == "base_mask" || opts.batch_mode == "all") {
    for (size_t line = 1; line < opts.batch_line_count; ++line) {
      run_pair("base_mask", -1, 0, line);
    }
  }

  if (opts.batch_mode == "derivative" || opts.batch_mode == "all") {
    for (int bit = opts.bit_first; bit <= opts.bit_last; ++bit) {
      const unsigned line_bit = static_cast<unsigned>(bit - 7);
      const size_t delta_line = 1ULL << line_bit;
      if (delta_line >= opts.batch_line_count) continue;
      for (size_t line = 0; line < opts.batch_line_count; ++line) {
        if ((line & delta_line) != 0) continue;
        const size_t other = line ^ delta_line;
        if (other >= opts.batch_line_count) continue;
        run_pair("derivative", bit, line, other);
      }
    }
  }

  if ((opts.batch_mode == "random" || opts.batch_mode == "all") &&
      opts.batch_random_pairs > 0) {
    std::mt19937_64 rng(opts.batch_seed);
    std::uniform_int_distribution<size_t> dist(0, opts.batch_line_count - 1);
    for (int i = 0; i < opts.batch_random_pairs; ++i) {
      size_t left = dist(rng);
      size_t right = dist(rng);
      if (left == right) right = (right + 1) % opts.batch_line_count;
      run_pair("random", -1, left, right);
    }
  }

  if (opts.batch_mode != "base_mask" && opts.batch_mode != "derivative" &&
      opts.batch_mode != "random" && opts.batch_mode != "all") {
    std::fprintf(stderr, "Unknown --batch mode: %s\n", opts.batch_mode.c_str());
    std::exit(1);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::fprintf(stderr, "batch_total=%zu\n", launched);
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double q) {
  if (sorted.empty()) return 0;
  const double pos = q * static_cast<double>(sorted.size() - 1);
  return sorted[static_cast<size_t>(std::llround(pos))];
}

void print_stats(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double sum =
      std::accumulate(sorted.begin(), sorted.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  const double mean = sum / static_cast<double>(sorted.size());
  double sq = 0.0;
  for (uint64_t v : sorted) {
    const double d = static_cast<double>(v) - mean;
    sq += d * d;
  }
  const double stdev = std::sqrt(sq / static_cast<double>(sorted.size()));
  std::printf(
      "samples=%zu mean=%.2f stdev=%.2f min=%llu p01=%llu p05=%llu p50=%llu "
      "p90=%llu p95=%llu p99=%llu p999=%llu max=%llu\n",
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
  const Options opts = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, opts.device));

  DeviceAllocation data_allocation = allocate_device_data(opts);
  uint8_t* d_data = data_allocation.ptr;
  uint64_t* d_samples = nullptr;
  uint32_t* d_sinks = nullptr;
  const size_t sample_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.iters);
  const size_t sink_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.threads);
  const size_t logical_loads = opts.tile_bytes / kLoadBytes;
  const size_t measured_bytes = logical_loads * kLoadBytes *
                                static_cast<size_t>(opts.repeat) *
                                static_cast<size_t>(opts.blocks) *
                                static_cast<size_t>(opts.iters);
  const size_t kernel_bytes = logical_loads * kLoadBytes *
                              static_cast<size_t>(opts.repeat) *
                              static_cast<size_t>(opts.blocks) *
                              static_cast<size_t>(opts.iters + opts.warmup);

  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sinks, sink_count * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_data, 0x5a, data_allocation.bytes));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_sinks, 0, sink_count * sizeof(uint32_t)));

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "case=l2-hbm-interleave op=%s pattern=%s blocks=%d threads=%d iters=%d "
      "warmup=%d repeat=%d data_bytes=%zu tile_bytes=%zu logical_loads=%zu "
      "stride_bytes=%zu pair_delta_bytes=%zu base_offset=%zu smem_bytes=%zu "
      "alloc=%s allocated_bytes=%zu measured_bytes=%zu kernel_bytes=%zu "
      "data_ptr=%p\n",
      op_name(opts.op), pattern_name(opts.pattern), opts.blocks, opts.threads,
      opts.iters, opts.warmup, opts.repeat, opts.data_bytes, opts.tile_bytes,
      logical_loads, opts.stride_bytes, opts.pair_delta_bytes, opts.base_offset,
      opts.smem_bytes, alloc_mode_name(opts.alloc_mode), data_allocation.bytes,
      measured_bytes, kernel_bytes, static_cast<void*>(d_data));

  if (!opts.batch_mode.empty()) {
    std::printf(
        "batch_mode=%s batch_csv=%s bit_first=%d bit_last=%d "
        "batch_line_count=%zu batch_random_pairs=%d batch_seed=%u\n",
        opts.batch_mode.c_str(), opts.batch_csv_path.c_str(), opts.bit_first,
        opts.bit_last, opts.batch_line_count, opts.batch_random_pairs,
        opts.batch_seed);
    run_fixed_pair_batch(opts, d_data, d_samples, d_sinks);
    free_device_data(&data_allocation);
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_sinks));
    return 0;
  }

  if (!opts.pair_csv_path.empty()) {
    std::printf("pair_csv=%s batch_csv=%s\n", opts.pair_csv_path.c_str(),
                opts.batch_csv_path.c_str());
    run_pair_csv_batch(opts, d_data, d_samples, d_sinks);
    free_device_data(&data_allocation);
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_sinks));
    return 0;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (opts.event_timing) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
  }

  const int pattern_i = static_cast<int>(opts.pattern);
  if (opts.op == AccessOp::kLdg) {
    interleave_probe_kernel<AccessOp::kLdg>
        <<<opts.blocks, opts.threads, opts.smem_bytes>>>(
            d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.repeat,
            opts.data_bytes, opts.tile_bytes, opts.stride_bytes,
            opts.pair_delta_bytes, opts.base_offset, pattern_i, opts.smem_bytes);
  } else {
    interleave_probe_kernel<AccessOp::kCpAsync>
        <<<opts.blocks, opts.threads, opts.smem_bytes>>>(
            d_data, d_samples, d_sinks, opts.iters, opts.warmup, opts.repeat,
            opts.data_bytes, opts.tile_bytes, opts.stride_bytes,
            opts.pair_delta_bytes, opts.base_offset, pattern_i, opts.smem_bytes);
  }
  CUDA_CHECK(cudaGetLastError());

  if (opts.event_timing) {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms > 0.0f) {
      const double measured_gbs =
          static_cast<double>(measured_bytes) / (elapsed_ms * 1.0e6);
      const double kernel_gbs =
          static_cast<double>(kernel_bytes) / (elapsed_ms * 1.0e6);
      std::printf("event_elapsed_ms=%.6f measured_GBps=%.3f kernel_GBps=%.3f\n",
                  elapsed_ms, measured_gbs, kernel_gbs);
    }
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<uint64_t> samples(sample_count);
  std::vector<uint32_t> sinks(sink_count);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        sample_count * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks, sink_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  const uint64_t sink_sum =
      std::accumulate(sinks.begin(), sinks.end(), uint64_t{0});
  print_stats(samples);
  std::printf("sink=%llu\n", static_cast<unsigned long long>(sink_sum));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.blocks, opts.iters);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  if (opts.event_timing) {
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  free_device_data(&data_allocation);
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
