#pragma once

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace fa3_hopper_test {

struct Fa3PrefillCase {
  const char *name;
  int batch;
  int seqlen;
  int heads;
  int head_dim;
  bool causal;
};

#define FA3_PREFILL_CASE_LIST(X)                    \
  X(H32D64FullB64S512, 64, 512, 32, 64, false)      \
  X(H32D64FullB32S1024, 32, 1024, 32, 64, false)    \
  X(H32D64FullB16S2048, 16, 2048, 32, 64, false)    \
  X(H32D64FullB8S4096, 8, 4096, 32, 64, false)      \
  X(H32D64FullB4S8192, 4, 8192, 32, 64, false)      \
  X(H32D64CausalB64S512, 64, 512, 32, 64, true)     \
  X(H32D64CausalB32S1024, 32, 1024, 32, 64, true)   \
  X(H32D64CausalB16S2048, 16, 2048, 32, 64, true)   \
  X(H32D64CausalB8S4096, 8, 4096, 32, 64, true)     \
  X(H32D64CausalB4S8192, 4, 8192, 32, 64, true)     \
  X(H16D128FullB64S512, 64, 512, 16, 128, false)    \
  X(H16D128FullB32S1024, 32, 1024, 16, 128, false)  \
  X(H16D128FullB16S2048, 16, 2048, 16, 128, false)  \
  X(H16D128FullB8S4096, 8, 4096, 16, 128, false)    \
  X(H16D128FullB4S8192, 4, 8192, 16, 128, false)    \
  X(H16D128CausalB64S512, 64, 512, 16, 128, true)   \
  X(H16D128CausalB32S1024, 32, 1024, 16, 128, true) \
  X(H16D128CausalB16S2048, 16, 2048, 16, 128, true) \
  X(H16D128CausalB8S4096, 8, 4096, 16, 128, true)   \
  X(H16D128CausalB4S8192, 4, 8192, 16, 128, true)

static constexpr int kFa3PrefillCaseCount = 20;

#define FA3_PREFILL_CASE_ENTRY(name, batch, seqlen, heads, head_dim, causal) \
  Fa3PrefillCase{#name, batch, seqlen, heads, head_dim, causal},
static constexpr Fa3PrefillCase kFa3PrefillCases[] = {
    FA3_PREFILL_CASE_LIST(FA3_PREFILL_CASE_ENTRY)};
#undef FA3_PREFILL_CASE_ENTRY

static_assert(sizeof(kFa3PrefillCases) / sizeof(kFa3PrefillCases[0]) ==
                  kFa3PrefillCaseCount,
              "FA3 prefill case list must contain 20 cases");

struct Fa3RunResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  float output0 = 0.0f;
  float lse0 = 0.0f;
};

inline bool is_valid_fa3_prefill_case(const Fa3PrefillCase &cfg) {
  return cfg.batch > 0 && cfg.seqlen > 0 &&
         cfg.batch * cfg.seqlen == 32 * 1024 &&
         ((cfg.heads == 32 && cfg.head_dim == 64) ||
          (cfg.heads == 16 && cfg.head_dim == 128));
}

inline void fill_half(std::vector<cutlass::half_t> &x, float scale) {
  for (size_t i = 0; i < x.size(); ++i) {
    float v = std::sin(float(i % 251) * 0.013f) * scale;
    x[i] = cutlass::half_t(v);
  }
}

template <int HeadDim, bool IsCausal>
inline Fa3RunResult run_fa3_prefill_fp16_typed(const Fa3PrefillCase &cfg) {
  constexpr int D = HeadDim;
  constexpr int DV = HeadDim;
  const int B = cfg.batch;
  const int M = cfg.seqlen;
  const int N = cfg.seqlen;
  const int H = cfg.heads;
  const int H_KV = cfg.heads;

  const size_t q_elems = size_t(B) * M * H * D;
  const size_t k_elems = size_t(B) * N * H_KV * D;
  const size_t v_elems = size_t(B) * N * H_KV * DV;
  const size_t o_elems = size_t(B) * M * H * DV;
  const size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;
  int *d_tile_count_semaphore = nullptr;

  Fa3RunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);
    free_if_needed(d_tile_count_semaphore);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  if (!is_valid_fa3_prefill_case(cfg) || cfg.head_dim != HeadDim ||
      cfg.causal != IsCausal) {
    return finish(cudaErrorInvalidValue, "Fa3PrefillCase");
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_tile_count_semaphore, sizeof(int)));
  }

  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_q, 0, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_k, 0, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_v, 0, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
  if constexpr (IsCausal) {
    FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_tile_count_semaphore, 0, sizeof(int)));
  }

  Flash_fwd_params params = {};
  params.q_ptr = d_q;
  params.k_ptr = d_k;
  params.v_ptr = d_v;
  params.o_ptr = d_o;
  params.softmax_lse_ptr = d_lse;

  params.q_batch_stride = M * H * D;
  params.k_batch_stride = N * H_KV * D;
  params.v_batch_stride = N * H_KV * DV;
  params.o_batch_stride = M * H * DV;

  params.q_row_stride = H * D;
  params.k_row_stride = H_KV * D;
  params.v_row_stride = H_KV * DV;
  params.o_row_stride = H * DV;

  params.q_head_stride = D;
  params.k_head_stride = D;
  params.v_head_stride = DV;
  params.o_head_stride = DV;
  params.v_dim_stride = 1;

  params.b = B;
  params.b_k = B;
  params.h = H;
  params.h_k = H_KV;
  params.seqlen_q = M;
  params.seqlen_k = N;
  params.seqlen_knew = 0;
  params.total_q = B * M;
  params.total_k = B * N;
  params.total_knew = 0;
  params.d = D;
  params.dv = DV;
  params.seqlen_q_rounded = M;
  params.seqlen_k_rounded = N;
  params.d_rounded = D;
  params.dv_rounded = DV;
  params.rotary_dim = 0;

  params.scale_softmax = 1.0f / std::sqrt(float(D));
  params.softcap = 0.0f;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.attention_chunk = 0;

  params.is_bf16 = false;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = IsCausal;
  params.is_local = false;
  params.is_rotary_interleaved = false;

  params.num_splits = 1;
  params.pack_gqa = false;
  params.tile_count_semaphore = d_tile_count_semaphore;
  params.tile_count_semaphore_offset = 0;
  params.skip_scheduler_metadata_computation = true;
  params.varlen_sort_batches = true;
  params.head_swizzle = IsCausal;
  params.prepare_varlen_pdl = false;
  params.arch = 90;
  params.num_sm = 0;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, D, DV, 1, cutlass::half_t, cutlass::half_t,
                IsCausal, false, false, false, false,
                false, false, false, false, false>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  cutlass::half_t output0;
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&output0, d_o, sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(&result.lse0, d_lse, sizeof(float),
                                      cudaMemcpyDeviceToHost));
  result.output0 = float(output0);

  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

inline Fa3RunResult run_fa3_prefill_fp16(const Fa3PrefillCase &cfg) {
  if (cfg.head_dim == 64 && !cfg.causal) {
    return run_fa3_prefill_fp16_typed<64, false>(cfg);
  }
  if (cfg.head_dim == 64 && cfg.causal) {
    return run_fa3_prefill_fp16_typed<64, true>(cfg);
  }
  if (cfg.head_dim == 128 && !cfg.causal) {
    return run_fa3_prefill_fp16_typed<128, false>(cfg);
  }
  if (cfg.head_dim == 128 && cfg.causal) {
    return run_fa3_prefill_fp16_typed<128, true>(cfg);
  }

  Fa3RunResult result;
  result.error = cudaErrorInvalidValue;
  result.where = "Fa3PrefillCase";
  return result;
}

inline Fa3RunResult run_fa3_fwd_hdim128_fp16() {
  constexpr int B = 9;
  constexpr int M = 64;
  constexpr int N = 128;
  constexpr int H = 6;
  constexpr int H_KV = 6;
  constexpr int D = 128;
  constexpr int DV = 128;

  constexpr size_t q_elems = size_t(B) * M * H * D;
  constexpr size_t k_elems = size_t(B) * N * H_KV * D;
  constexpr size_t v_elems = size_t(B) * N * H_KV * DV;
  constexpr size_t o_elems = size_t(B) * M * H * DV;
  constexpr size_t lse_elems = size_t(B) * H * M;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;

  Fa3RunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      cudaError_t free_error = cudaFree(ptr);
      if (cleanup_error == cudaSuccess) cleanup_error = free_error;
    };
    free_if_needed(d_q);
    free_if_needed(d_k);
    free_if_needed(d_v);
    free_if_needed(d_o);
    free_if_needed(d_lse);

    if (error != cudaSuccess) {
      result.error = error;
      result.where = where;
    } else if (cleanup_error != cudaSuccess) {
      result.error = cleanup_error;
      result.where = "cudaFree";
    } else {
      result.error = cudaSuccess;
      result.where = "success";
    }
    return result;
  };

#define FA3_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  FA3_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  std::vector<cutlass::half_t> h_q(q_elems);
  std::vector<cutlass::half_t> h_k(k_elems);
  std::vector<cutlass::half_t> h_v(v_elems);
  std::vector<cutlass::half_t> h_o(o_elems);
  std::vector<float> h_lse(lse_elems);

  fill_half(h_q, 0.25f);
  fill_half(h_k, 0.20f);
  fill_half(h_v, 0.30f);

  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));

  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(),
                                      q_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(),
                                      k_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(),
                                      v_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyHostToDevice));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

  Flash_fwd_params params = {};
  params.q_ptr = d_q;
  params.k_ptr = d_k;
  params.v_ptr = d_v;
  params.o_ptr = d_o;
  params.softmax_lse_ptr = d_lse;

  params.q_batch_stride = M * H * D;
  params.k_batch_stride = N * H_KV * D;
  params.v_batch_stride = N * H_KV * DV;
  params.o_batch_stride = M * H * DV;

  params.q_row_stride = H * D;
  params.k_row_stride = H_KV * D;
  params.v_row_stride = H_KV * DV;
  params.o_row_stride = H * DV;

  params.q_head_stride = D;
  params.k_head_stride = D;
  params.v_head_stride = DV;
  params.o_head_stride = DV;
  params.v_dim_stride = 1;

  params.b = B;
  params.b_k = B;
  params.h = H;
  params.h_k = H_KV;
  params.seqlen_q = M;
  params.seqlen_k = N;
  params.seqlen_knew = 0;
  params.total_q = B * M;
  params.total_k = B * N;
  params.total_knew = 0;
  params.d = D;
  params.dv = DV;
  params.seqlen_q_rounded = M;
  params.seqlen_k_rounded = N;
  params.d_rounded = D;
  params.dv_rounded = DV;
  params.rotary_dim = 0;

  params.scale_softmax = 1.0f / std::sqrt(float(D));
  params.softcap = 0.0f;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.attention_chunk = 0;

  params.is_bf16 = false;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = false;
  params.is_local = false;
  params.is_rotary_interleaved = false;

  params.num_splits = 1;
  params.pack_gqa = false;
  params.skip_scheduler_metadata_computation = true;
  params.varlen_sort_batches = false;
  params.head_swizzle = false;
  params.prepare_varlen_pdl = false;
  params.arch = 90;
  params.num_sm = 0;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, 128, 128, 1, cutlass::half_t, cutlass::half_t,
                false, false, false, false, false,
                false, false, false, false, false>(params, stream);

  FA3_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_o.data(), d_o,
                                      o_elems * sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA3_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_lse.data(), d_lse,
                                      lse_elems * sizeof(float),
                                      cudaMemcpyDeviceToHost));

  result.output0 = float(h_o[0]);
  result.lse0 = h_lse[0];
  return finish(cudaSuccess, "success");

#undef FA3_RETURN_IF_CUDA_ERROR
}

}  // namespace fa3_hopper_test
