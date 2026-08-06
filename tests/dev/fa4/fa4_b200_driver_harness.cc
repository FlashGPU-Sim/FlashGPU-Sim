#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {

struct Options {
  const char *module_path = nullptr;
  const char *kernel_name = nullptr;
  bool load_only = false;
  bool legacy_coord_pointers = false;
  bool head_dim_v_set = false;
  std::uint32_t batch = 1;
  std::uint32_t seqlen_q = 128;
  std::uint32_t seqlen_k = 128;
  std::uint32_t heads = 2;
  std::uint32_t head_dim = 64;
  std::uint32_t head_dim_v = 64;
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t block_x = 512;
  std::uint32_t dynamic_smem = 231424;
  std::string dtype = "fp16";
};

void checkCuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: cudaError=%d\n", what,
                 static_cast<int>(status));
    std::exit(1);
  }
}

void checkCu(CUresult status, const char *what) {
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed: CUresult=%d\n", what,
                 static_cast<int>(status));
    std::exit(1);
  }
}

[[noreturn]] void usage(const char *argv0) {
  std::fprintf(
      stderr,
      "usage: %s <module.fatbin> <kernel> [options]\n"
      "\n"
      "options:\n"
      "  load-only                  Resolve the kernel but do not launch it\n"
      "  --batch N                  Batch size (default: 1)\n"
      "  --seqlen-q N               Query sequence length (default: 128)\n"
      "  --seqlen-k N               Key/value sequence length (default: 128)\n"
      "  --heads N                  Number of heads (default: 2)\n"
      "  --head-dim N               Q/K head dimension (default: 64)\n"
      "  --head-dim-v N             V/O head dimension (default: head-dim)\n"
      "  --grid-x/y/z N             Override CUDA grid dimensions\n"
      "  --block-x N                CUDA block x dimension (default: 512)\n"
      "  --dynamic-smem N           Dynamic shared memory bytes (default: "
      "231424)\n"
      "  --dtype fp16|bf16          Tensor map element type (default: fp16)\n"
      "  --legacy-coord-pointers    Fill coord tensors with raw pointers, "
      "matching\n"
      "                             the first bring-up harness\n",
      argv0);
  std::exit(2);
}

std::uint32_t parseU32(const char *value, const char *name) {
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (*value == '\0' || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return static_cast<std::uint32_t>(parsed);
}

void parseArgs(int argc, char **argv, Options *opts) {
  if (argc < 3) usage(argv[0]);
  opts->module_path = argv[1];
  opts->kernel_name = argv[2];

  for (int i = 3; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "load-only" || arg == "--load-only") {
      opts->load_only = true;
    } else if (arg == "--batch") {
      opts->batch = parseU32(need_value("--batch"), "--batch");
    } else if (arg == "--seqlen-q") {
      opts->seqlen_q = parseU32(need_value("--seqlen-q"), "--seqlen-q");
    } else if (arg == "--seqlen-k") {
      opts->seqlen_k = parseU32(need_value("--seqlen-k"), "--seqlen-k");
    } else if (arg == "--heads") {
      opts->heads = parseU32(need_value("--heads"), "--heads");
    } else if (arg == "--head-dim") {
      opts->head_dim = parseU32(need_value("--head-dim"), "--head-dim");
    } else if (arg == "--head-dim-v") {
      opts->head_dim_v = parseU32(need_value("--head-dim-v"), "--head-dim-v");
      opts->head_dim_v_set = true;
    } else if (arg == "--grid-x") {
      opts->grid_x = parseU32(need_value("--grid-x"), "--grid-x");
    } else if (arg == "--grid-y") {
      opts->grid_y = parseU32(need_value("--grid-y"), "--grid-y");
    } else if (arg == "--grid-z") {
      opts->grid_z = parseU32(need_value("--grid-z"), "--grid-z");
    } else if (arg == "--block-x") {
      opts->block_x = parseU32(need_value("--block-x"), "--block-x");
    } else if (arg == "--dynamic-smem") {
      opts->dynamic_smem =
          parseU32(need_value("--dynamic-smem"), "--dynamic-smem");
    } else if (arg == "--dtype") {
      opts->dtype = need_value("--dtype");
      if (opts->dtype != "fp16" && opts->dtype != "bf16") {
        std::fprintf(stderr, "invalid --dtype: %s\n", opts->dtype.c_str());
        std::exit(2);
      }
    } else if (arg == "--legacy-coord-pointers") {
      opts->legacy_coord_pointers = true;
    } else {
      usage(argv[0]);
    }
  }

  if (!opts->head_dim_v_set) opts->head_dim_v = opts->head_dim;

  if (opts->batch == 0 || opts->seqlen_q == 0 || opts->seqlen_k == 0 ||
      opts->heads == 0 || opts->head_dim == 0 || opts->head_dim_v == 0 ||
      opts->grid_x == 0 || opts->grid_y == 0 || opts->grid_z == 0 ||
      opts->block_x == 0) {
    std::fprintf(stderr, "shape and launch dimensions must be non-zero\n");
    std::exit(2);
  }
}

std::size_t checkedMul(std::size_t a, std::size_t b, const char *name) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    std::fprintf(stderr, "size overflow while computing %s\n", name);
    std::exit(2);
  }
  return a * b;
}

std::uint32_t roundUp(std::uint32_t value, std::uint32_t align) {
  return ((value + align - 1) / align) * align;
}

void fillCoordTensor(unsigned char (&dst)[16], const void *ptr,
                     bool legacy_coord_pointers) {
  std::memset(dst, 0, sizeof(dst));
  if (legacy_coord_pointers) {
    const auto value = reinterpret_cast<std::uint64_t>(ptr);
    std::memcpy(dst, &value, sizeof(value));
  }
}

CUtensorMapDataType tensorMapDataType(const Options &opts) {
  return opts.dtype == "bf16" ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                              : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}

void encodeTensorMap(CUtensorMap *map, void *base, std::uint32_t head_dim,
                     std::uint32_t seqlen, std::uint32_t heads,
                     std::uint32_t batch, CUtensorMapDataType dtype) {
  const cuuint64_t global_dim[4] = {head_dim, seqlen, heads, batch};
  const cuuint64_t global_strides[3] = {
      static_cast<cuuint64_t>(head_dim) * sizeof(std::uint16_t),
      static_cast<cuuint64_t>(head_dim) * seqlen * sizeof(std::uint16_t),
      static_cast<cuuint64_t>(head_dim) * seqlen * heads *
          sizeof(std::uint16_t),
  };
  const cuuint32_t box_dim[4] = {head_dim, 128, 1, 1};
  const cuuint32_t element_strides[4] = {1, 1, 1, 1};
  checkCu(cuTensorMapEncodeTiled(
              map, dtype, 4, base, global_dim, global_strides, box_dim,
              element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
              CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled");
}

std::size_t tensorBytes(std::uint32_t batch, std::uint32_t seqlen,
                        std::uint32_t heads, std::uint32_t head_dim) {
  std::size_t elements = batch;
  elements = checkedMul(elements, seqlen, "tensor elements");
  elements = checkedMul(elements, heads, "tensor elements");
  elements = checkedMul(elements, head_dim, "tensor elements");
  return checkedMul(elements, sizeof(std::uint16_t), "tensor bytes");
}

void checkDevicePrefixPattern(const void *device_ptr, std::size_t bytes,
                              unsigned char expected, const char *name) {
  constexpr std::size_t kMaxCheckBytes = 4096;
  const std::size_t check_bytes = std::min(bytes, kMaxCheckBytes);
  std::array<unsigned char, kMaxCheckBytes> host{};
  checkCuda(
      cudaMemcpy(host.data(), device_ptr, check_bytes, cudaMemcpyDeviceToHost),
      name);
  for (std::size_t i = 0; i < check_bytes; ++i) {
    if (host[i] != expected) {
      std::fprintf(stderr,
                   "%s canary mismatch at byte %zu: expected 0x%02x, got "
                   "0x%02x\n",
                   name, i, static_cast<unsigned>(expected),
                   static_cast<unsigned>(host[i]));
      std::exit(1);
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  Options opts;
  parseArgs(argc, argv, &opts);

  checkCuda(cudaSetDevice(0), "cudaSetDevice");

  CUmodule module = nullptr;
  checkCu(cuModuleLoad(&module, opts.module_path), "cuModuleLoad");
  std::printf("loaded module %s\n", opts.module_path);

  CUfunction kernel = nullptr;
  checkCu(cuModuleGetFunction(&kernel, module, opts.kernel_name),
          "cuModuleGetFunction");
  std::printf("resolved kernel %s\n", opts.kernel_name);

  if (opts.load_only) {
    std::printf("load-only mode complete\n");
    return 0;
  }

  const std::uint32_t padded_q = std::max(roundUp(opts.seqlen_q, 128), 128u);
  const std::uint32_t padded_k = std::max(roundUp(opts.seqlen_k, 128), 128u);
  const std::size_t q_bytes =
      tensorBytes(opts.batch, padded_q, opts.heads, opts.head_dim);
  const std::size_t k_bytes =
      tensorBytes(opts.batch, padded_k, opts.heads, opts.head_dim);
  const std::size_t v_bytes =
      tensorBytes(opts.batch, padded_k, opts.heads, opts.head_dim_v);
  const std::size_t o_bytes =
      tensorBytes(opts.batch, padded_q, opts.heads, opts.head_dim_v);

  void *q = nullptr;
  void *k = nullptr;
  void *v = nullptr;
  void *o = nullptr;
  constexpr unsigned char kQFill = 0x00;
  constexpr unsigned char kKFill = 0x38;
  constexpr unsigned char kVFill = 0x34;
  constexpr unsigned char kOFill = 0x00;
  checkCuda(cudaMalloc(&q, q_bytes), "cudaMalloc(q)");
  checkCuda(cudaMalloc(&k, k_bytes), "cudaMalloc(k)");
  checkCuda(cudaMalloc(&v, v_bytes), "cudaMalloc(v)");
  checkCuda(cudaMalloc(&o, o_bytes), "cudaMalloc(o)");
  checkCuda(cudaMemset(q, kQFill, q_bytes), "cudaMemset(q)");
  checkCuda(cudaMemset(k, kKFill, k_bytes), "cudaMemset(k)");
  checkCuda(cudaMemset(v, kVFill, v_bytes), "cudaMemset(v)");
  checkCuda(cudaMemset(o, kOFill, o_bytes), "cudaMemset(o)");

  alignas(64) unsigned char tensor_q[16];
  alignas(64) unsigned char tensor_k[16];
  alignas(64) unsigned char tensor_v[16];
  alignas(64) unsigned char tensor_o[16];
  fillCoordTensor(tensor_q, q, opts.legacy_coord_pointers);
  fillCoordTensor(tensor_k, k, opts.legacy_coord_pointers);
  fillCoordTensor(tensor_v, v, opts.legacy_coord_pointers);
  fillCoordTensor(tensor_o, o, opts.legacy_coord_pointers);

  const CUtensorMapDataType dtype = tensorMapDataType(opts);
  alignas(64) CUtensorMap tma_q;
  alignas(64) CUtensorMap tma_k;
  alignas(64) CUtensorMap tma_v;
  alignas(64) CUtensorMap tma_o;
  encodeTensorMap(&tma_q, q, opts.head_dim, padded_q, opts.heads, opts.batch,
                  dtype);
  encodeTensorMap(&tma_k, k, opts.head_dim, padded_k, opts.heads, opts.batch,
                  dtype);
  encodeTensorMap(&tma_v, v, opts.head_dim_v, padded_k, opts.heads, opts.batch,
                  dtype);
  encodeTensorMap(&tma_o, o, opts.head_dim_v, padded_q, opts.heads, opts.batch,
                  dtype);

  const float scale =
      1.4426950408889634f / std::sqrt(static_cast<float>(opts.head_dim));
  unsigned char tiny0[3] = {};
  unsigned char tiny1[3] = {};
  std::uint32_t u32_seqlen_q = opts.seqlen_q;
  std::uint32_t u32_seqlen_k = opts.seqlen_k;
  std::uint32_t u32_head_dim = opts.head_dim;
  std::uint32_t u32_heads = opts.heads;
  std::uint32_t u32_batch = opts.batch;
  std::uint32_t u32_stride = 0;
  std::uint32_t u32_grid_x = opts.grid_x;
  std::uint32_t u32_0 = 0;
  unsigned char vec12_0[12] = {};
  unsigned char vec12_1[12] = {};
  unsigned char vec12_2[12] = {};
  unsigned char vec12_3[12] = {};
  unsigned char vec12_4[12] = {};

  // ABI from the generated MLIR host launch: coord q/k/v/o, then TMA q/k/v/o.
  void *params[] = {
      tensor_q,      tensor_k,      tensor_v,
      tensor_o,      &tma_q,        &tma_k,
      &tma_v,        &tma_o,        const_cast<float *>(&scale),
      tiny0,         tiny1,         &u32_seqlen_q,
      &u32_seqlen_k, &u32_head_dim, &u32_heads,
      &u32_batch,    &u32_stride,   vec12_0,
      vec12_1,       vec12_2,       vec12_3,
      &u32_grid_x,   vec12_4,       &u32_0,
  };

  std::printf(
      "launch shape batch=%u seqlen_q=%u seqlen_k=%u heads=%u head_dim=%u "
      "head_dim_v=%u grid=(%u,%u,%u) block=(%u,1,1) smem=%u dtype=%s\n",
      opts.batch, opts.seqlen_q, opts.seqlen_k, opts.heads, opts.head_dim,
      opts.head_dim_v, opts.grid_x, opts.grid_y, opts.grid_z, opts.block_x,
      opts.dynamic_smem, opts.dtype.c_str());

  checkCu(cuLaunchKernel(kernel, opts.grid_x, opts.grid_y, opts.grid_z,
                         opts.block_x, 1, 1, opts.dynamic_smem, nullptr, params,
                         nullptr),
          "cuLaunchKernel");
  std::printf("launched kernel\n");

  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  checkDevicePrefixPattern(k, k_bytes, kKFill, "K input canary");
  checkDevicePrefixPattern(v, v_bytes, kVFill, "V input canary");
  std::printf("synchronized\n");
  return 0;
}
