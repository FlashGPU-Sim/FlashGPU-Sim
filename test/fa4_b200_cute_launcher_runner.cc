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
#include <vector>

#include "fa4_b200_launcher.h"

#ifndef FA4_B200_RUNNER_BWD
#define FA4_B200_RUNNER_BWD 0
#endif

namespace {

struct Options {
  bool load_only = false;
  bool head_dim_v_set = false;
  bool causal = true;
  std::uint32_t batch = 1;
  std::uint32_t seqlen_q = 128;
  std::uint32_t seqlen_k = 128;
  std::uint32_t heads = 2;
  std::uint32_t head_dim = 64;
  std::uint32_t head_dim_v = 64;
  std::string dtype = "fp16";
};

void checkCuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: cudaError=%d\n", what,
                 static_cast<int>(status));
    std::exit(1);
  }
}

[[noreturn]] void usage(const char *argv0) {
  std::fprintf(
      stderr,
      "usage: %s [options]\n"
      "\n"
      "options:\n"
      "  load-only                  Load the generated CUDA library only\n"
      "  --batch N                  Batch size (default: 1)\n"
      "  --seqlen-q N               Query sequence length (default: 128)\n"
      "  --seqlen-k N               Key/value sequence length (default: 128)\n"
      "  --heads N                  Number of heads (default: 2)\n"
      "  --head-dim N               Q/K head dimension (default: 64)\n"
      "  --head-dim-v N             V/O head dimension (default: head-dim)\n"
      "  --dtype fp16|bf16          Tensor dtype compiled into the artifact\n"
      "  --causal                   Fill metadata for a causal artifact\n"
      "  --non-causal               Fill metadata for a non-causal artifact\n",
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
  for (int i = 1; i < argc; ++i) {
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
    } else if (arg == "--dtype") {
      opts->dtype = need_value("--dtype");
      if (opts->dtype != "fp16" && opts->dtype != "bf16") {
        std::fprintf(stderr, "invalid --dtype: %s\n", opts->dtype.c_str());
        std::exit(2);
      }
    } else if (arg == "--causal") {
      opts->causal = true;
    } else if (arg == "--non-causal") {
      opts->causal = false;
    } else {
      usage(argv[0]);
    }
  }

  if (!opts->head_dim_v_set) opts->head_dim_v = opts->head_dim;

  if (opts->batch == 0 || opts->seqlen_q == 0 || opts->seqlen_k == 0 ||
      opts->heads == 0 || opts->head_dim == 0 || opts->head_dim_v == 0) {
    std::fprintf(stderr, "shape dimensions must be non-zero\n");
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

std::size_t tensorBytes(std::uint32_t batch, std::uint32_t seqlen,
                        std::uint32_t heads, std::uint32_t head_dim) {
  std::size_t elements = batch;
  elements = checkedMul(elements, seqlen, "tensor elements");
  elements = checkedMul(elements, heads, "tensor elements");
  elements = checkedMul(elements, head_dim, "tensor elements");
  return checkedMul(elements, sizeof(std::uint16_t), "tensor bytes");
}

std::size_t tensor3BytesF32(std::uint32_t dim0, std::uint32_t dim1,
                            std::uint32_t dim2) {
  std::size_t elements = dim0;
  elements = checkedMul(elements, dim1, "tensor3 elements");
  elements = checkedMul(elements, dim2, "tensor3 elements");
  return checkedMul(elements, sizeof(float), "tensor3 bytes");
}

template <typename Tensor>
void fillTensorDescriptor(void *data, std::uint32_t batch, std::uint32_t seqlen,
                          std::uint32_t heads, std::uint32_t head_dim,
                          Tensor *tensor) {
  tensor->data = data;
  tensor->dynamic_shapes[0] = static_cast<std::int32_t>(batch);
  tensor->dynamic_shapes[1] = static_cast<std::int32_t>(seqlen);
  tensor->dynamic_shapes[2] = static_cast<std::int32_t>(heads);
  tensor->dynamic_shapes[3] = static_cast<std::int32_t>(head_dim);
  tensor->dynamic_strides[0] =
      static_cast<std::int64_t>(seqlen) * heads * head_dim;
  tensor->dynamic_strides[1] = static_cast<std::int64_t>(heads) * head_dim;
  tensor->dynamic_strides[2] = head_dim;
}

template <typename Tensor>
void fillTensor3Descriptor(void *data, std::uint32_t dim0, std::uint32_t dim1,
                           std::uint32_t dim2, Tensor *tensor) {
  tensor->data = data;
  tensor->dynamic_shapes[0] = static_cast<std::int32_t>(dim0);
  tensor->dynamic_shapes[1] = static_cast<std::int32_t>(dim1);
  tensor->dynamic_shapes[2] = static_cast<std::int32_t>(dim2);
  tensor->dynamic_strides[0] = static_cast<std::int64_t>(dim1) * dim2;
  tensor->dynamic_strides[1] = dim2;
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

std::uint16_t oneBits(const std::string &dtype) {
  if (dtype == "fp16") return 0x3c00u;
  if (dtype == "bf16") return 0x3f80u;
  std::fprintf(stderr, "unsupported dtype for oneBits: %s\n", dtype.c_str());
  std::exit(2);
}

void fillDeviceU16(void *device_ptr, std::size_t bytes, std::uint16_t value,
                   const char *name) {
  if (bytes % sizeof(std::uint16_t) != 0) {
    std::fprintf(stderr, "%s byte size is not u16 aligned: %zu\n", name, bytes);
    std::exit(2);
  }
  std::vector<std::uint16_t> host(bytes / sizeof(std::uint16_t), value);
  checkCuda(cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice),
            name);
}

void fillDeviceF32(void *device_ptr, std::size_t bytes, float value,
                   const char *name) {
  if (bytes % sizeof(float) != 0) {
    std::fprintf(stderr, "%s byte size is not float aligned: %zu\n", name,
                 bytes);
    std::exit(2);
  }
  std::vector<float> host(bytes / sizeof(float), value);
  checkCuda(cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice),
            name);
}

void fillLse(void *device_ptr, std::uint32_t batch, std::uint32_t heads,
             std::uint32_t seqlen_q, std::uint32_t seqlen_k, bool causal) {
  std::vector<float> host(static_cast<std::size_t>(batch) * heads * seqlen_q);
  for (std::uint32_t b = 0; b < batch; ++b) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      for (std::uint32_t q = 0; q < seqlen_q; ++q) {
        std::uint32_t visible = seqlen_k;
        if (causal) visible = std::min<std::uint32_t>(seqlen_k, q + 1);
        if (visible == 0) visible = 1;
        const std::size_t offset =
            (static_cast<std::size_t>(b) * heads + h) * seqlen_q + q;
        host[offset] = std::log(static_cast<float>(visible));
      }
    }
  }
  checkCuda(cudaMemcpy(device_ptr, host.data(),
                       host.size() * sizeof(float), cudaMemcpyHostToDevice),
            "fill(lse)");
}

void checkDeviceU16All(const void *device_ptr, std::size_t bytes,
                       std::uint16_t expected, const char *name) {
  if (bytes % sizeof(std::uint16_t) != 0) {
    std::fprintf(stderr, "%s byte size is not u16 aligned: %zu\n", name, bytes);
    std::exit(2);
  }
  std::vector<std::uint16_t> host(bytes / sizeof(std::uint16_t));
  checkCuda(cudaMemcpy(host.data(), device_ptr, bytes, cudaMemcpyDeviceToHost),
            name);
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (host[i] != expected) {
      std::fprintf(stderr,
                   "%s mismatch at element %zu: expected 0x%04x, got 0x%04x\n",
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

  fa4_b200_launcher_Kernel_Module_t module{};
  fa4_b200_launcher_Kernel_Module_Load(&module);
  std::printf("loaded generated FA4 launcher module\n");
  if (opts.load_only) {
    std::printf("load-only mode complete\n");
    return 0;
  }

  const std::size_t q_bytes =
      tensorBytes(opts.batch, opts.seqlen_q, opts.heads, opts.head_dim);
  const std::size_t k_bytes =
      tensorBytes(opts.batch, opts.seqlen_k, opts.heads, opts.head_dim);
  const std::size_t v_bytes =
      tensorBytes(opts.batch, opts.seqlen_k, opts.heads, opts.head_dim_v);
  const std::size_t o_bytes =
      tensorBytes(opts.batch, opts.seqlen_q, opts.heads, opts.head_dim_v);

  void *q = nullptr;
  void *k = nullptr;
  void *v = nullptr;
  void *o = nullptr;
  void *d_o = nullptr;
  void *lse = nullptr;
  void *d_psum = nullptr;
  void *d_qaccum = nullptr;
  void *d_k = nullptr;
  void *d_v = nullptr;
  checkCuda(cudaMalloc(&q, q_bytes), "cudaMalloc(q)");
  checkCuda(cudaMalloc(&k, k_bytes), "cudaMalloc(k)");
  checkCuda(cudaMalloc(&v, v_bytes), "cudaMalloc(v)");
#if FA4_B200_RUNNER_BWD
  const std::size_t lse_bytes =
      tensor3BytesF32(opts.batch, opts.heads, opts.seqlen_q);
  checkCuda(cudaMalloc(&d_o, o_bytes), "cudaMalloc(dO)");
  checkCuda(cudaMalloc(&lse, lse_bytes), "cudaMalloc(lse)");
  checkCuda(cudaMalloc(&d_psum, lse_bytes), "cudaMalloc(dPsum)");
  checkCuda(cudaMalloc(&d_qaccum, lse_bytes), "cudaMalloc(dQaccum)");
  checkCuda(cudaMalloc(&d_k, k_bytes), "cudaMalloc(dK)");
  checkCuda(cudaMalloc(&d_v, v_bytes), "cudaMalloc(dV)");
#else
  checkCuda(cudaMalloc(&o, o_bytes), "cudaMalloc(o)");
#endif

  const std::uint16_t one = oneBits(opts.dtype);
  fillDeviceU16(q, q_bytes, 0, "fill(q)");
  fillDeviceU16(k, k_bytes, 0, "fill(k)");
  fillDeviceU16(v, v_bytes, one, "fill(v)");
#if FA4_B200_RUNNER_BWD
  fillDeviceU16(d_o, o_bytes, one, "fill(dO)");
  fillLse(lse, opts.batch, opts.heads, opts.seqlen_q, opts.seqlen_k,
          opts.causal);
  fillDeviceF32(d_psum, lse_bytes, 0.0f, "fill(dPsum)");
  fillDeviceF32(d_qaccum, lse_bytes, 0.0f, "fill(dQaccum)");
  fillDeviceU16(d_k, k_bytes, 0, "fill(dK)");
  fillDeviceU16(d_v, v_bytes, 0, "fill(dV)");
#else
  fillDeviceU16(o, o_bytes, 0, "fill(o)");
#endif

  fa4_b200_launcher_Tensor_mQ_t mQ{};
  fa4_b200_launcher_Tensor_mK_t mK{};
  fa4_b200_launcher_Tensor_mV_t mV{};
  fillTensorDescriptor(q, opts.batch, opts.seqlen_q, opts.heads, opts.head_dim,
                       &mQ);
  fillTensorDescriptor(k, opts.batch, opts.seqlen_k, opts.heads, opts.head_dim,
                       &mK);
  fillTensorDescriptor(v, opts.batch, opts.seqlen_k, opts.heads,
                       opts.head_dim_v, &mV);
#if FA4_B200_RUNNER_BWD
  fa4_b200_launcher_Tensor_mdO_t mdO{};
  fa4_b200_launcher_Tensor_mLSE_t mLSE{};
  fa4_b200_launcher_Tensor_mdPsum_t mdPsum{};
  fa4_b200_launcher_Tensor_mdQaccum_t mdQaccum{};
  fa4_b200_launcher_Tensor_mdK_t mdK{};
  fa4_b200_launcher_Tensor_mdV_t mdV{};
  fillTensorDescriptor(d_o, opts.batch, opts.seqlen_q, opts.heads,
                       opts.head_dim_v, &mdO);
  fillTensor3Descriptor(lse, opts.batch, opts.heads, opts.seqlen_q, &mLSE);
  fillTensor3Descriptor(d_psum, opts.batch, opts.heads, opts.seqlen_q,
                        &mdPsum);
  fillTensor3Descriptor(d_qaccum, opts.batch, opts.heads, opts.seqlen_q,
                        &mdQaccum);
  fillTensorDescriptor(d_k, opts.batch, opts.seqlen_k, opts.heads,
                       opts.head_dim, &mdK);
  fillTensorDescriptor(d_v, opts.batch, opts.seqlen_k, opts.heads,
                       opts.head_dim_v, &mdV);
#else
  fa4_b200_launcher_Tensor_mO_t mO{};
  fillTensorDescriptor(o, opts.batch, opts.seqlen_q, opts.heads,
                       opts.head_dim_v, &mO);
#endif

  const float scale = 1.0f / std::sqrt(static_cast<float>(opts.head_dim));
  std::printf(
      "launch generated FA4 %s wrapper shape batch=%u seqlen_q=%u "
      "seqlen_k=%u heads=%u head_dim=%u head_dim_v=%u dtype=%s %s\n",
#if FA4_B200_RUNNER_BWD
      "backward",
#else
      "forward",
#endif
      opts.batch, opts.seqlen_q, opts.seqlen_k, opts.heads, opts.head_dim,
      opts.head_dim_v, opts.dtype.c_str(),
      opts.causal ? "causal" : "non-causal");

#if FA4_B200_RUNNER_BWD
  const int ret = cute_dsl_fa4_b200_launcher_wrapper(
      &module, &mQ, &mK, &mV, &mdO, &mLSE, &mdPsum, &mdQaccum, &mdK, &mdV,
      scale, 0);
#else
  const int ret =
      cute_dsl_fa4_b200_launcher_wrapper(&module, &mQ, &mK, &mV, &mO, scale, 0);
#endif
  if (ret != 0) {
    std::fprintf(stderr, "generated FA4 wrapper returned %d\n", ret);
    return 1;
  }

  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  checkDeviceU16All(k, k_bytes, 0, "K input");
  checkDeviceU16All(v, v_bytes, one, "V input");
#if FA4_B200_RUNNER_BWD
  checkDeviceU16All(q, q_bytes, 0, "Q input");
  checkDeviceU16All(d_o, o_bytes, one, "dO input");
  std::printf("synchronized backward smoke check passed\n");
#else
  checkDeviceU16All(o, o_bytes, one, "O output");
  std::printf("synchronized and numeric check passed\n");
#endif
  return 0;
}
