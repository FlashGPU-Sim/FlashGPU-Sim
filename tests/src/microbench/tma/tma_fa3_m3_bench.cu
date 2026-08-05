#include <cuda_runtime.h>
#include <cuda.h>

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

constexpr int kThreads = 32;
constexpr int kHalfBytes = 2;
constexpr int kDefaultL2Bytes = 32 * 1024 * 1024;

enum LoadKind {
  kLoadKV = 0,
  kLoadK = 1,
  kLoadV = 2,
};

struct Options {
  int device = 0;
  int blocks = 128;
  int iters = 8;
  int warmup = 0;
  int batch = 64;
  int heads = 16;
  int seqlen = 512;
  int head_dim = 128;
  int head_dim_v = 128;
  int block_m = 128;
  int block_n = 128;
  int m_block = 3;
  int group_nblocks = 2;
  int l2_bytes = kDefaultL2Bytes;
  int repeat = 1;
  bool causal = true;
  bool sweep_hb = true;
  bool explicit_cluster = false;
  LoadKind load_kind = kLoadKV;
  std::string csv_path;
};

struct Sample {
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  uint64_t global_start = 0;
  uint64_t global_end = 0;
  uint32_t smid = 0;
  uint32_t bidh = 0;
  uint32_t bidb = 0;
  uint32_t n_blocks = 0;
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

#define CU_CHECK(expr)                                                       \
  do {                                                                       \
    CUresult err__ = (expr);                                                 \
    if (err__ != CUDA_SUCCESS) {                                             \
      const char* name__ = nullptr;                                          \
      const char* text__ = nullptr;                                          \
      cuGetErrorName(err__, &name__);                                        \
      cuGetErrorString(err__, &text__);                                      \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s: %s\n", __FILE__,    \
                   __LINE__, name__ ? name__ : "unknown",                   \
                   text__ ? text__ : "unknown");                            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__host__ __device__ inline uintptr_t align_up(uintptr_t value,
                                              uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__host__ __device__ inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__host__ __device__ inline int pow2_floor(int value) {
  int result = 1;
  while (result <= value / 2) result <<= 1;
  return result;
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint64_t globaltimer_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint32_t smid_now() {
  uint32_t value = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value)::"memory");
  return value;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(smem_addr_u32(bar)), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(smem_addr_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar,
                                                     uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "waitLoop%=: \n"
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
      "@!complete bra.uni waitLoop%=;\n"
      "}\n"
      :
      : "r"(smem_addr_u32(bar)), "r"(parity)
      : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
               :
               : "r"(smem_addr_u32(bar))
               : "memory");
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_cta(
    uint8_t* smem_dst, uint8_t* global_tmap, int32_t coord0, int32_t coord1,
    int32_t coord2, int32_t coord3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :
      : "r"(smem_addr_u32(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(global_tmap)), "r"(coord0), "r"(coord1),
        "r"(coord2), "r"(coord3), "r"(smem_addr_u32(bar))
      : "memory");
}

__device__ inline void map_first_round_hb(int tile_idx, int iter, int heads,
                                          int batch, int swizzle,
                                          int total_hb, int sweep_hb,
                                          int* bidh, int* bidb) {
  const int first_section_hb = total_hb < swizzle ? total_hb : swizzle;
  int hb = tile_idx % first_section_hb;
  if (sweep_hb) {
    const int section_base = (iter * first_section_hb) % total_hb;
    hb = (section_base + hb) % total_hb;
  }
  *bidb = hb / heads;
  *bidh = hb - *bidb * heads;
  if (*bidb >= batch) {
    *bidb = batch - 1;
    *bidh = heads - 1;
  }
}

__global__ void tma_fa3_m3_kernel(
    const uint16_t* k, const uint16_t* v, uint8_t* global_tmaps,
    Sample* samples, uint32_t* block_sinks, int iters, int warmup, int batch,
    int heads, int seqlen, int head_dim, int head_dim_v, int block_m,
    int block_n, int m_block, int group_nblocks, int swizzle, int causal,
    int sweep_hb, int repeat, int load_kind_i) {
  extern __shared__ uint8_t smem[];

  const int tid = threadIdx.x;
  const LoadKind load_kind = static_cast<LoadKind>(load_kind_i);
  const bool load_k = load_kind == kLoadKV || load_kind == kLoadK;
  const bool load_v = load_kind == kLoadKV || load_kind == kLoadV;
  const int chunk_dim = 64;
  const int d_chunks = ceil_div(head_dim, chunk_dim);
  const int chunk_bytes = block_n * chunk_dim * kHalfBytes;
  const int tile_bytes = d_chunks * chunk_bytes;
  const int total_hb = batch * heads;
  const int n_total = ceil_div(seqlen, block_n);
  const int causal_n_max =
      ceil_div((m_block + 1) * block_m, block_n) < n_total
          ? ceil_div((m_block + 1) * block_m, block_n)
          : n_total;
  const int n_blocks = causal ? causal_n_max : n_total;

  uintptr_t p = reinterpret_cast<uintptr_t>(smem);
  uint8_t* dst_k = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_k) +
      static_cast<uintptr_t>(group_nblocks) * tile_bytes;
  uint8_t* dst_v = reinterpret_cast<uint8_t*>(align_up(p, 1024));
  p = reinterpret_cast<uintptr_t>(dst_v) +
      static_cast<uintptr_t>(group_nblocks) * tile_bytes;
  uint64_t* bar = reinterpret_cast<uint64_t*>(align_up(p, 8));

  uint8_t* global_tmap_k = global_tmaps;
  uint8_t* global_tmap_v = global_tmap_k + 128;

  uint32_t sink = 0;
  const int total_iters = warmup + iters;
  for (int i = 0; i < total_iters; ++i) {
    int bidh = 0;
    int bidb = 0;
    map_first_round_hb(static_cast<int>(blockIdx.x), i, heads, batch, swizzle,
                       total_hb, sweep_hb, &bidh, &bidb);

    uint64_t clock_start = 0;
    uint64_t clock_end = 0;
    uint64_t global_start = 0;
    uint64_t global_end = 0;
    if (tid == 0) {
      clock_start = clock64_now();
      global_start = globaltimer_now();
    }

    for (int rep = 0; rep < repeat; ++rep) {
      for (int group_hi = n_blocks - 1; group_hi >= 0;
           group_hi -= group_nblocks) {
        const int count =
            group_hi + 1 < group_nblocks ? group_hi + 1 : group_nblocks;
        const uint32_t expected_bytes =
            static_cast<uint32_t>(count * tile_bytes *
                                  (load_k && load_v ? 2 : 1));
        if (tid == 0) {
          mbarrier_init(bar, 1);
          mbarrier_arrive_expect_tx(bar, expected_bytes);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

          for (int slot = 0; slot < count; ++slot) {
            const int n_block = group_hi - slot;
            const int coord1 = n_block * block_n;
            for (int d_chunk = 0; d_chunk < d_chunks; ++d_chunk) {
              const int coord0 = d_chunk * chunk_dim;
              const int dst_offset = slot * tile_bytes + d_chunk * chunk_bytes;
              if (load_k) {
                cp_async_bulk_tensor_4d_cta(dst_k + dst_offset, global_tmap_k,
                                            coord0, coord1, bidh, bidb, bar);
              }
              if (load_v) {
                cp_async_bulk_tensor_4d_cta(dst_v + dst_offset, global_tmap_v,
                                            coord0, coord1, bidh, bidb, bar);
              }
            }
          }

          mbarrier_wait_parity(bar, 0);
          if (load_k) sink += *reinterpret_cast<volatile uint16_t*>(dst_k);
          if (load_v) sink += *reinterpret_cast<volatile uint16_t*>(dst_v);
          mbarrier_inval(bar);
        }
      }
    }

    if (tid == 0) {
      clock_end = clock64_now();
      global_end = globaltimer_now();
      if (i >= warmup) {
        const int out_iter = i - warmup;
        Sample& sample =
            samples[static_cast<size_t>(blockIdx.x) * iters + out_iter];
        sample.clock_start = clock_start;
        sample.clock_end = clock_end;
        sample.global_start = global_start;
        sample.global_end = global_end;
        sample.smid = smid_now();
        sample.bidh = static_cast<uint32_t>(bidh);
        sample.bidb = static_cast<uint32_t>(bidb);
        sample.n_blocks = static_cast<uint32_t>(n_blocks);
      }
    }
  }

  if (tid == 0) {
    block_sinks[blockIdx.x] = sink;
  }
}

const char* load_kind_name(LoadKind kind) {
  switch (kind) {
    case kLoadKV:
      return "kv";
    case kLoadK:
      return "k";
    case kLoadV:
      return "v";
  }
  return "unknown";
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device N] [--blocks N] [--iters N] [--warmup N]\n"
      "          [--batch N] [--heads N] [--seqlen N] [--head-dim N]\n"
      "          [--head-dim-v N] [--block-m N] [--block-n N]\n"
      "          [--m-block N] [--group-nblocks N] [--repeat N]\n"
      "          [--causal 0|1] [--sweep-hb 0|1]\n"
      "          [--explicit-cluster 0|1] [--load kv|k|v]\n"
      "          [--l2-bytes N] [--csv path]\n",
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
               parse_int_arg(arg, "--iters", &opts.iters) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--batch", &opts.batch) ||
               parse_int_arg(arg, "--heads", &opts.heads) ||
               parse_int_arg(arg, "--seqlen", &opts.seqlen) ||
               parse_int_arg(arg, "--head-dim", &opts.head_dim) ||
               parse_int_arg(arg, "--head-dim-v", &opts.head_dim_v) ||
               parse_int_arg(arg, "--block-m", &opts.block_m) ||
               parse_int_arg(arg, "--block-n", &opts.block_n) ||
               parse_int_arg(arg, "--m-block", &opts.m_block) ||
               parse_int_arg(arg, "--group-nblocks", &opts.group_nblocks) ||
               parse_int_arg(arg, "--repeat", &opts.repeat) ||
               parse_int_arg(arg, "--l2-bytes", &opts.l2_bytes)) {
      continue;
    } else if (std::strncmp(arg, "--causal=", 9) == 0) {
      opts.causal = std::atoi(arg + 9) != 0;
    } else if (std::strncmp(arg, "--sweep-hb=", 11) == 0) {
      opts.sweep_hb = std::atoi(arg + 11) != 0;
    } else if (std::strncmp(arg, "--explicit-cluster=", 19) == 0) {
      opts.explicit_cluster = std::atoi(arg + 19) != 0;
    } else if (std::strncmp(arg, "--load=", 7) == 0) {
      const char* value = arg + 7;
      if (std::strcmp(value, "kv") == 0) {
        opts.load_kind = kLoadKV;
      } else if (std::strcmp(value, "k") == 0) {
        opts.load_kind = kLoadK;
      } else if (std::strcmp(value, "v") == 0) {
        opts.load_kind = kLoadV;
      } else {
        std::fprintf(stderr, "Unknown load kind: %s\n", value);
        std::exit(1);
      }
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.blocks <= 0 || opts.iters <= 0 || opts.warmup < 0 ||
      opts.batch <= 0 || opts.heads <= 0 || opts.seqlen <= 0 ||
      opts.head_dim <= 0 || opts.head_dim_v <= 0 || opts.block_m <= 0 ||
      opts.block_n <= 0 || opts.m_block < 0 || opts.group_nblocks <= 0 ||
      opts.repeat <= 0 || opts.l2_bytes <= 0) {
    std::fprintf(stderr, "invalid non-positive option\n");
    std::exit(1);
  }
  if (opts.head_dim != opts.head_dim_v) {
    std::fprintf(stderr,
                 "this focused bench currently expects head_dim == head_dim_v\n");
    std::exit(1);
  }
  return opts;
}

int fa3_l2_swizzle(const Options& opts) {
  const long long size_one_kv_head =
      static_cast<long long>(opts.seqlen) *
      static_cast<long long>(opts.head_dim + opts.head_dim_v) * kHalfBytes;
  if (size_one_kv_head <= 0 || opts.l2_bytes < size_one_kv_head) {
    return 1;
  }
  return pow2_floor(static_cast<int>(opts.l2_bytes / size_one_kv_head));
}

size_t dynamic_smem_bytes(const Options& opts) {
  const int chunk_dim = 64;
  const int d_chunks = ceil_div(opts.head_dim, chunk_dim);
  const size_t chunk_bytes =
      static_cast<size_t>(opts.block_n) * chunk_dim * kHalfBytes;
  const size_t tile_bytes =
      static_cast<size_t>(d_chunks) * chunk_bytes;
  uintptr_t p = 0;
  p = align_up(p, 1024) + tile_bytes * opts.group_nblocks;
  p = align_up(p, 1024) + tile_bytes * opts.group_nblocks;
  p = align_up(p, 8) + sizeof(uint64_t);
  return align_up(p, 1024);
}

void encode_tensormap_4d_bshd(CUtensorMap* tmap, void* base,
                              const Options& opts) {
  const uint64_t global_dim[4] = {
      static_cast<uint64_t>(opts.head_dim),
      static_cast<uint64_t>(opts.seqlen),
      static_cast<uint64_t>(opts.heads),
      static_cast<uint64_t>(opts.batch),
  };
  const uint64_t global_stride[3] = {
      static_cast<uint64_t>(opts.heads) *
          static_cast<uint64_t>(opts.head_dim) * kHalfBytes,
      static_cast<uint64_t>(opts.head_dim) * kHalfBytes,
      static_cast<uint64_t>(opts.seqlen) *
          static_cast<uint64_t>(opts.heads) *
          static_cast<uint64_t>(opts.head_dim) * kHalfBytes,
  };
  const uint32_t box_dim[4] = {
      64u,
      static_cast<uint32_t>(opts.block_n),
      1u,
      1u,
  };
  const uint32_t element_stride[4] = {1u, 1u, 1u, 1u};
  CU_CHECK(cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, base, global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double q) {
  if (sorted.empty()) return 0;
  const double pos = q * static_cast<double>(sorted.size() - 1);
  return sorted[static_cast<size_t>(std::llround(pos))];
}

void print_stats(const char* label, const std::vector<uint64_t>& values) {
  std::vector<uint64_t> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double sum =
      std::accumulate(sorted.begin(), sorted.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; });
  const double mean = sum / static_cast<double>(sorted.size());
  std::printf(
      "%s samples=%zu mean=%.2f min=%llu p50=%llu p90=%llu p95=%llu "
      "p99=%llu max=%llu\n",
      label, sorted.size(), mean,
      static_cast<unsigned long long>(sorted.front()),
      static_cast<unsigned long long>(percentile(sorted, 0.50)),
      static_cast<unsigned long long>(percentile(sorted, 0.90)),
      static_cast<unsigned long long>(percentile(sorted, 0.95)),
      static_cast<unsigned long long>(percentile(sorted, 0.99)),
      static_cast<unsigned long long>(sorted.back()));
}

void write_csv(const std::string& path, const std::vector<Sample>& samples,
               int blocks, int iters) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "block,iter,smid,bidh,bidb,n_blocks,clock_start,clock_end,"
         "clock_delta,global_start_ns,global_end_ns,global_delta_ns\n";
  for (int b = 0; b < blocks; ++b) {
    for (int i = 0; i < iters; ++i) {
      const Sample& s = samples[static_cast<size_t>(b) * iters + i];
      out << b << "," << i << "," << s.smid << "," << s.bidh << ","
          << s.bidb << "," << s.n_blocks << "," << s.clock_start << ","
          << s.clock_end << "," << (s.clock_end - s.clock_start) << ","
          << s.global_start << "," << s.global_end << ","
          << (s.global_end - s.global_start) << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));
  CU_CHECK(cuInit(0));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate,
                                    opts.device));

  const int swizzle = fa3_l2_swizzle(opts);
  const size_t smem_bytes = dynamic_smem_bytes(opts);
  CUDA_CHECK(cudaFuncSetAttribute(tma_fa3_m3_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(tma_fa3_m3_kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));

  const size_t elems =
      static_cast<size_t>(opts.batch) * opts.seqlen * opts.heads *
      opts.head_dim;
  const size_t sample_count =
      static_cast<size_t>(opts.blocks) * static_cast<size_t>(opts.iters);

  uint16_t* d_k = nullptr;
  uint16_t* d_v = nullptr;
  uint8_t* d_tmaps = nullptr;
  Sample* d_samples = nullptr;
  uint32_t* d_sinks = nullptr;

  CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_tmaps, 256));
  CUDA_CHECK(cudaMalloc(&d_samples, sample_count * sizeof(Sample)));
  CUDA_CHECK(cudaMalloc(&d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(d_k, 0x3c, elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemset(d_v, 0x5a, elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemset(d_tmaps, 0, 256));
  CUDA_CHECK(cudaMemset(d_samples, 0, sample_count * sizeof(Sample)));
  CUDA_CHECK(cudaMemset(d_sinks, 0,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t)));

  alignas(64) CUtensorMap h_tmaps[2];
  std::memset(h_tmaps, 0, sizeof(h_tmaps));
  encode_tensormap_4d_bshd(&h_tmaps[0], d_k, opts);
  encode_tensormap_4d_bshd(&h_tmaps[1], d_v, opts);
  CUDA_CHECK(cudaMemcpy(d_tmaps, h_tmaps, sizeof(h_tmaps),
                        cudaMemcpyHostToDevice));

  const int n_total = ceil_div(opts.seqlen, opts.block_n);
  const int n_causal =
      std::min(n_total, ceil_div((opts.m_block + 1) * opts.block_m,
                                 opts.block_n));
  const int n_blocks = opts.causal ? n_causal : n_total;
  const int chunk_dim = 64;
  const int d_chunks = ceil_div(opts.head_dim, chunk_dim);
  const uint64_t chunk_bytes =
      static_cast<uint64_t>(opts.block_n) * chunk_dim * kHalfBytes;
  const uint64_t tile_bytes =
      static_cast<uint64_t>(d_chunks) * chunk_bytes;
  const uint64_t bytes_per_task =
      tile_bytes * static_cast<uint64_t>(n_blocks) *
      static_cast<uint64_t>(opts.load_kind == kLoadKV ? 2 : 1) *
      static_cast<uint64_t>(opts.repeat);

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "shape=H%dD%dB%dS%d causal=%d block_m=%d block_n=%d m_block=%d "
      "n_blocks=%d swizzle=%d blocks=%d threads=%d iters=%d warmup=%d "
      "group_nblocks=%d repeat=%d sweep_hb=%d explicit_cluster=%d "
      "cluster=1x1x1 load=%s smem=%zu\n",
      opts.heads, opts.head_dim, opts.batch, opts.seqlen, opts.causal ? 1 : 0,
      opts.block_m, opts.block_n, opts.m_block, n_blocks, swizzle, opts.blocks,
      kThreads, opts.iters, opts.warmup, opts.group_nblocks, opts.repeat,
      opts.sweep_hb ? 1 : 0, opts.explicit_cluster ? 1 : 0,
      load_kind_name(opts.load_kind), smem_bytes);
  std::printf("tile_bytes=%llu bytes_per_task=%llu measured_bytes=%llu\n",
              static_cast<unsigned long long>(tile_bytes),
              static_cast<unsigned long long>(bytes_per_task),
              static_cast<unsigned long long>(
                  bytes_per_task * static_cast<uint64_t>(opts.blocks) *
                  static_cast<uint64_t>(opts.iters)));

  if (opts.explicit_cluster) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 1;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(opts.blocks);
    launch_config.blockDim = dim3(kThreads);
    launch_config.dynamicSmemBytes = smem_bytes;
    launch_config.stream = nullptr;
    launch_config.attrs = &attr;
    launch_config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &launch_config, tma_fa3_m3_kernel, d_k, d_v, d_tmaps, d_samples,
        d_sinks, opts.iters, opts.warmup, opts.batch, opts.heads, opts.seqlen,
        opts.head_dim, opts.head_dim_v, opts.block_m, opts.block_n,
        opts.m_block, opts.group_nblocks, swizzle, opts.causal ? 1 : 0,
        opts.sweep_hb ? 1 : 0, opts.repeat, static_cast<int>(opts.load_kind)));
  } else {
    tma_fa3_m3_kernel<<<opts.blocks, kThreads, smem_bytes>>>(
        d_k, d_v, d_tmaps, d_samples, d_sinks, opts.iters, opts.warmup,
        opts.batch, opts.heads, opts.seqlen, opts.head_dim, opts.head_dim_v,
        opts.block_m, opts.block_n, opts.m_block, opts.group_nblocks, swizzle,
        opts.causal ? 1 : 0, opts.sweep_hb ? 1 : 0, opts.repeat,
        static_cast<int>(opts.load_kind));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Sample> samples(sample_count);
  std::vector<uint32_t> sinks(opts.blocks);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples,
                        sample_count * sizeof(Sample),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sinks.data(), d_sinks,
                        static_cast<size_t>(opts.blocks) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  std::vector<uint64_t> clock_delta;
  std::vector<uint64_t> global_delta;
  clock_delta.reserve(sample_count);
  global_delta.reserve(sample_count);
  for (const Sample& s : samples) {
    clock_delta.push_back(s.clock_end - s.clock_start);
    global_delta.push_back(s.global_end - s.global_start);
  }
  print_stats("clock_cycles", clock_delta);
  print_stats("globaltimer_ns", global_delta);

  const double mean_cycles =
      std::accumulate(clock_delta.begin(), clock_delta.end(), 0.0,
                      [](double a, uint64_t b) { return a + b; }) /
      static_cast<double>(clock_delta.size());
  const double aggregate_bytes_per_cycle =
      mean_cycles > 0.0 ? static_cast<double>(bytes_per_task) *
                              static_cast<double>(opts.blocks) / mean_cycles
                        : 0.0;
  const double aggregate_gib_per_sec =
      aggregate_bytes_per_cycle * static_cast<double>(clock_rate_khz) * 1000.0 /
      (1024.0 * 1024.0 * 1024.0);
  const uint64_t sink_sum =
      std::accumulate(sinks.begin(), sinks.end(), uint64_t{0});
  std::printf("mean_aggregate_tma_bw_gib_s=%.2f bytes_per_cycle=%.2f\n",
              aggregate_gib_per_sec, aggregate_bytes_per_cycle);
  std::printf("sink=%llu\n", static_cast<unsigned long long>(sink_sum));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.blocks, opts.iters);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_tmaps));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_sinks));
  return 0;
}
