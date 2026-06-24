#pragma once

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstddef>
#include <limits>
#include <vector>

#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace fa2_hopper_test {

struct Fa2PrefillCase {
  const char *name;
  int batch;
  int seqlen;
  int heads;
  int head_dim;
  bool causal;
};

#if !defined(FA2_PREFILL_ENABLE_H32D64_FULL) && \
    !defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL) && \
    !defined(FA2_PREFILL_ENABLE_H16D128_FULL) && \
    !defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
#define FA2_PREFILL_ENABLE_H32D64_FULL
#define FA2_PREFILL_ENABLE_H32D64_CAUSAL
#define FA2_PREFILL_ENABLE_H16D128_FULL
#define FA2_PREFILL_ENABLE_H16D128_CAUSAL
#endif

#define FA2_PREFILL_H32D64_FULL_CASE_LIST(X)        \
  X(H32D64FullB64S512, 64, 512, 32, 64, false)      \
  X(H32D64FullB32S1024, 32, 1024, 32, 64, false)    \
  X(H32D64FullB16S2048, 16, 2048, 32, 64, false)    \
  X(H32D64FullB8S4096, 8, 4096, 32, 64, false)      \
  X(H32D64FullB4S8192, 4, 8192, 32, 64, false)

#define FA2_PREFILL_H32D64_CAUSAL_CASE_LIST(X)      \
  X(H32D64CausalB64S512, 64, 512, 32, 64, true)     \
  X(H32D64CausalB32S1024, 32, 1024, 32, 64, true)   \
  X(H32D64CausalB16S2048, 16, 2048, 32, 64, true)   \
  X(H32D64CausalB8S4096, 8, 4096, 32, 64, true)     \
  X(H32D64CausalB4S8192, 4, 8192, 32, 64, true)

#define FA2_PREFILL_H16D128_FULL_CASE_LIST(X)       \
  X(H16D128FullB64S512, 64, 512, 16, 128, false)    \
  X(H16D128FullB32S1024, 32, 1024, 16, 128, false)  \
  X(H16D128FullB16S2048, 16, 2048, 16, 128, false)  \
  X(H16D128FullB8S4096, 8, 4096, 16, 128, false)    \
  X(H16D128FullB4S8192, 4, 8192, 16, 128, false)

#define FA2_PREFILL_H16D128_CAUSAL_CASE_LIST(X)     \
  X(H16D128CausalB64S512, 64, 512, 16, 128, true)   \
  X(H16D128CausalB32S1024, 32, 1024, 16, 128, true) \
  X(H16D128CausalB16S2048, 16, 2048, 16, 128, true) \
  X(H16D128CausalB8S4096, 8, 4096, 16, 128, true)   \
  X(H16D128CausalB4S8192, 4, 8192, 16, 128, true)

#define FA2_PREFILL_CASE_LIST(X)                    \
  FA2_PREFILL_H32D64_FULL_CASE_LIST(X)              \
  FA2_PREFILL_H32D64_CAUSAL_CASE_LIST(X)            \
  FA2_PREFILL_H16D128_FULL_CASE_LIST(X)             \
  FA2_PREFILL_H16D128_CAUSAL_CASE_LIST(X)

static constexpr int kFa2PrefillCaseCount = 20;

#define FA2_PREFILL_SMOKE_H32D64_FULL_CASE_LIST(X)  \
  X(H32D64FullB2S128, 2, 128, 32, 64, false)

#define FA2_PREFILL_SMOKE_H32D64_CAUSAL_CASE_LIST(X) \
  X(H32D64CausalB2S128, 2, 128, 32, 64, true)

#define FA2_PREFILL_SMOKE_H16D128_FULL_CASE_LIST(X) \
  X(H16D128FullB2S128, 2, 128, 16, 128, false)

#define FA2_PREFILL_SMOKE_H16D128_CAUSAL_CASE_LIST(X) \
  X(H16D128CausalB2S128, 2, 128, 16, 128, true)

#define FA2_PREFILL_SMOKE_CASE_LIST(X)              \
  FA2_PREFILL_SMOKE_H32D64_FULL_CASE_LIST(X)        \
  FA2_PREFILL_SMOKE_H32D64_CAUSAL_CASE_LIST(X)      \
  FA2_PREFILL_SMOKE_H16D128_FULL_CASE_LIST(X)       \
  FA2_PREFILL_SMOKE_H16D128_CAUSAL_CASE_LIST(X)

static constexpr int kFa2PrefillSmokeCaseCount = 4;

#define FA2_PREFILL_SMALL_H32D64_FULL_CASE_LIST(X)  \
  X(H32D64FullB32S256, 32, 256, 32, 64, false)

#define FA2_PREFILL_SMALL_H32D64_CAUSAL_CASE_LIST(X) \
  X(H32D64CausalB32S256, 32, 256, 32, 64, true)

#define FA2_PREFILL_SMALL_H16D128_FULL_CASE_LIST(X) \
  X(H16D128FullB32S256, 32, 256, 16, 128, false)

#define FA2_PREFILL_SMALL_H16D128_CAUSAL_CASE_LIST(X) \
  X(H16D128CausalB32S256, 32, 256, 16, 128, true)

#define FA2_PREFILL_SMALL_CASE_LIST(X)              \
  FA2_PREFILL_SMALL_H32D64_FULL_CASE_LIST(X)        \
  FA2_PREFILL_SMALL_H32D64_CAUSAL_CASE_LIST(X)      \
  FA2_PREFILL_SMALL_H16D128_FULL_CASE_LIST(X)       \
  FA2_PREFILL_SMALL_H16D128_CAUSAL_CASE_LIST(X)

static constexpr int kFa2PrefillSmallCaseCount = 4;

#define FA2_PREFILL_MEDIUM_H32D64_FULL_CASE_LIST(X) \
  X(H32D64FullB16S512, 16, 512, 32, 64, false)

#define FA2_PREFILL_MEDIUM_H32D64_CAUSAL_CASE_LIST(X) \
  X(H32D64CausalB16S512, 16, 512, 32, 64, true)

#define FA2_PREFILL_MEDIUM_H16D128_FULL_CASE_LIST(X) \
  X(H16D128FullB16S512, 16, 512, 16, 128, false)

#define FA2_PREFILL_MEDIUM_H16D128_CAUSAL_CASE_LIST(X) \
  X(H16D128CausalB16S512, 16, 512, 16, 128, true)

#define FA2_PREFILL_MEDIUM_CASE_LIST(X)             \
  FA2_PREFILL_MEDIUM_H32D64_FULL_CASE_LIST(X)       \
  FA2_PREFILL_MEDIUM_H32D64_CAUSAL_CASE_LIST(X)     \
  FA2_PREFILL_MEDIUM_H16D128_FULL_CASE_LIST(X)      \
  FA2_PREFILL_MEDIUM_H16D128_CAUSAL_CASE_LIST(X)

static constexpr int kFa2PrefillMediumCaseCount = 4;

#define FA2_PREFILL_SENSITIVITY_CASE_LIST(X)       \
  X(H1D128FullB1S256, 1, 256, 1, 128, false)

static constexpr int kFa2PrefillSensitivityCaseCount = 1;

#define FA2_PREFILL_SENSITIVITY_H1D128_FULL_CASE_LIST(X) \
  X(H1D128FullB1S256, 1, 256, 1, 128, false)             \
  X(H1D128FullB1S512, 1, 512, 1, 128, false)             \
  X(H1D128FullB1S1024, 1, 1024, 1, 128, false)           \
  X(H1D128FullB1S2048, 1, 2048, 1, 128, false)           \
  X(H1D128FullB1S4096, 1, 4096, 1, 128, false)

#define FA2_PREFILL_SENSITIVITY_H1D128_CAUSAL_CASE_LIST(X) \
  X(H1D128CausalB1S256, 1, 256, 1, 128, true)              \
  X(H1D128CausalB1S512, 1, 512, 1, 128, true)              \
  X(H1D128CausalB1S1024, 1, 1024, 1, 128, true)            \
  X(H1D128CausalB1S2048, 1, 2048, 1, 128, true)            \
  X(H1D128CausalB1S4096, 1, 4096, 1, 128, true)

#define FA2_PREFILL_SENSITIVITY_H1D128_CASE_LIST(X)     \
  FA2_PREFILL_SENSITIVITY_H1D128_FULL_CASE_LIST(X)      \
  FA2_PREFILL_SENSITIVITY_H1D128_CAUSAL_CASE_LIST(X)

static constexpr int kFa2PrefillSensitivityH1D128CaseCount = 10;

#if defined(FA2_PREFILL_SENSITIVITY_LARGE_D128_FULL_BH_HALF)
#define FA2_PREFILL_SENSITIVITY_LARGE_D128_FULL_CASE_LIST(X) \
  X(H8D128FullB32S512, 32, 512, 8, 128, false)
#else
#define FA2_PREFILL_SENSITIVITY_LARGE_D128_FULL_CASE_LIST(X) \
  X(H16D128FullB64S512, 64, 512, 16, 128, false)
#endif

static constexpr int kFa2PrefillSensitivityLargeD128FullCaseCount = 1;

#define FA2_PREFILL_CASE_ENTRY(name, batch, seqlen, heads, head_dim, causal) \
  Fa2PrefillCase{#name, batch, seqlen, heads, head_dim, causal},
static constexpr Fa2PrefillCase kFa2PrefillCases[] = {
    FA2_PREFILL_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillSmokeCases[] = {
    FA2_PREFILL_SMOKE_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillSmallCases[] = {
    FA2_PREFILL_SMALL_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillMediumCases[] = {
    FA2_PREFILL_MEDIUM_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillSensitivityCases[] = {
    FA2_PREFILL_SENSITIVITY_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillSensitivityH1D128Cases[] = {
    FA2_PREFILL_SENSITIVITY_H1D128_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
static constexpr Fa2PrefillCase kFa2PrefillSensitivityLargeD128FullCases[] = {
    FA2_PREFILL_SENSITIVITY_LARGE_D128_FULL_CASE_LIST(FA2_PREFILL_CASE_ENTRY)};
#undef FA2_PREFILL_CASE_ENTRY

static_assert(sizeof(kFa2PrefillCases) / sizeof(kFa2PrefillCases[0]) ==
                  kFa2PrefillCaseCount,
              "FA2 prefill case list must contain 20 cases");
static_assert(sizeof(kFa2PrefillSmokeCases) /
                      sizeof(kFa2PrefillSmokeCases[0]) ==
                  kFa2PrefillSmokeCaseCount,
              "FA2 prefill smoke case list must contain 4 cases");
static_assert(sizeof(kFa2PrefillSmallCases) /
                      sizeof(kFa2PrefillSmallCases[0]) ==
                  kFa2PrefillSmallCaseCount,
              "FA2 prefill small case list must contain 4 cases");
static_assert(sizeof(kFa2PrefillMediumCases) /
                      sizeof(kFa2PrefillMediumCases[0]) ==
                  kFa2PrefillMediumCaseCount,
              "FA2 prefill medium case list must contain 4 cases");
static_assert(sizeof(kFa2PrefillSensitivityCases) /
                      sizeof(kFa2PrefillSensitivityCases[0]) ==
                  kFa2PrefillSensitivityCaseCount,
              "FA2 prefill sensitivity case list must contain 1 case");
static_assert(sizeof(kFa2PrefillSensitivityH1D128Cases) /
                      sizeof(kFa2PrefillSensitivityH1D128Cases[0]) ==
                  kFa2PrefillSensitivityH1D128CaseCount,
              "FA2 prefill H1D128 sensitivity case list must contain 10 cases");
static_assert(sizeof(kFa2PrefillSensitivityLargeD128FullCases) /
                      sizeof(kFa2PrefillSensitivityLargeD128FullCases[0]) ==
                  kFa2PrefillSensitivityLargeD128FullCaseCount,
              "FA2 prefill large D128 full sensitivity case list must contain 1 case");

struct Fa2RunResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  float output0 = 0.0f;
  float lse0 = 0.0f;
  bool reference_checked = false;
  float output0_ref = 0.0f;
  float lse0_ref = 0.0f;
  float max_output_abs_error = 0.0f;
  float max_lse_abs_error = 0.0f;
  size_t max_output_abs_error_index = 0;
  size_t max_lse_abs_error_index = 0;
};

inline Fa2RunResult make_fa2_invalid_result(const char *where) {
  Fa2RunResult result;
  result.error = cudaErrorInvalidValue;
  result.where = where;
  return result;
}

inline bool is_supported_fa2_prefill_case(const Fa2PrefillCase &cfg) {
  return cfg.batch > 0 && cfg.seqlen > 0 && cfg.heads > 0 &&
         ((cfg.heads == 32 && cfg.head_dim == 64) ||
          (cfg.heads == 16 && cfg.head_dim == 128));
}

inline bool is_valid_fa2_prefill_case(const Fa2PrefillCase &cfg) {
  return is_supported_fa2_prefill_case(cfg) &&
         cfg.batch * cfg.seqlen == 32 * 1024;
}

inline bool is_valid_fa2_prefill_smoke_case(const Fa2PrefillCase &cfg) {
  return is_supported_fa2_prefill_case(cfg) &&
         cfg.batch == 2 && cfg.seqlen == 128;
}

inline bool is_valid_fa2_prefill_tuning_case(const Fa2PrefillCase &cfg) {
  return is_supported_fa2_prefill_case(cfg) && cfg.seqlen % 128 == 0;
}

inline bool is_valid_fa2_prefill_sensitivity_case(
    const Fa2PrefillCase &cfg) {
  return cfg.batch == 1 && cfg.seqlen == 256 && cfg.heads == 1 &&
         cfg.head_dim == 128 && !cfg.causal;
}

inline bool is_valid_fa2_prefill_sensitivity_h1d128_case(
    const Fa2PrefillCase &cfg) {
  return cfg.batch == 1 && cfg.heads == 1 && cfg.head_dim == 128 &&
         cfg.seqlen >= 256 && cfg.seqlen <= 4096 && cfg.seqlen % 128 == 0;
}

inline bool is_valid_fa2_prefill_sensitivity_large_d128_full_case(
    const Fa2PrefillCase &cfg) {
  return cfg.seqlen == 512 && cfg.head_dim == 128 && !cfg.causal &&
         ((cfg.batch == 64 && cfg.heads == 16) ||
          (cfg.batch == 32 && cfg.heads == 8));
}

inline void fill_half(std::vector<cutlass::half_t> &x, float scale) {
  for (size_t i = 0; i < x.size(); ++i) {
    float v = std::sin(float(i % 251) * 0.013f) * scale;
    x[i] = cutlass::half_t(v);
  }
}

#ifndef __CUDA_ARCH__
inline size_t fa2_qkv_index(int batch_idx, int row, int head, int dim,
                            int seqlen, int heads, int head_dim) {
  return ((size_t(batch_idx) * seqlen + row) * heads + head) * head_dim + dim;
}

inline size_t fa2_lse_index(int batch_idx, int head, int row, int seqlen,
                            int heads) {
  return (size_t(batch_idx) * heads + head) * seqlen + row;
}

struct Fa2ReferenceStats {
  float output0_ref = 0.0f;
  float lse0_ref = 0.0f;
  float max_output_abs_error = 0.0f;
  float max_lse_abs_error = 0.0f;
  size_t max_output_abs_error_index = 0;
  size_t max_lse_abs_error_index = 0;
};

template <int HeadDim, bool IsCausal>
inline Fa2ReferenceStats compute_fa2_reference_errors(
    const std::vector<cutlass::half_t> &q,
    const std::vector<cutlass::half_t> &k,
    const std::vector<cutlass::half_t> &v,
    const std::vector<cutlass::half_t> &actual_o,
    const std::vector<float> &actual_lse, int batch, int seqlen_q,
    int seqlen_k, int heads) {
  constexpr int D = HeadDim;
  const float scale = 1.0f / std::sqrt(float(D));
  std::vector<float> scores(seqlen_k);
  Fa2ReferenceStats stats;

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int m = 0; m < seqlen_q; ++m) {
        const int valid_k = IsCausal ? std::min(m + 1, seqlen_k) : seqlen_k;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < valid_k; ++n) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            const size_t q_idx =
                fa2_qkv_index(b, m, h, d, seqlen_q, heads, D);
            const size_t k_idx =
                fa2_qkv_index(b, n, h, d, seqlen_k, heads, D);
            dot += float(q[q_idx]) * float(k[k_idx]);
          }
          scores[n] = dot * scale;
          max_score = std::max(max_score, scores[n]);
        }

        float sum_exp = 0.0f;
        for (int n = 0; n < valid_k; ++n) {
          scores[n] = std::exp(scores[n] - max_score);
          sum_exp += scores[n];
        }
        const float inv_sum = 1.0f / sum_exp;
        const float ref_lse = max_score + std::log(sum_exp);
        const size_t lse_idx = fa2_lse_index(b, h, m, seqlen_q, heads);
        if (lse_idx == 0) stats.lse0_ref = ref_lse;
        const float lse_error = std::fabs(actual_lse[lse_idx] - ref_lse);
        if (lse_error > stats.max_lse_abs_error) {
          stats.max_lse_abs_error = lse_error;
          stats.max_lse_abs_error_index = lse_idx;
        }

        for (int d = 0; d < D; ++d) {
          float ref = 0.0f;
          for (int n = 0; n < valid_k; ++n) {
            const size_t v_idx =
                fa2_qkv_index(b, n, h, d, seqlen_k, heads, D);
            ref += scores[n] * inv_sum * float(v[v_idx]);
          }
          const size_t o_idx = fa2_qkv_index(b, m, h, d, seqlen_q, heads, D);
          if (o_idx == 0) stats.output0_ref = ref;
          const float output_error = std::fabs(float(actual_o[o_idx]) - ref);
          if (output_error > stats.max_output_abs_error) {
            stats.max_output_abs_error = output_error;
            stats.max_output_abs_error_index = o_idx;
          }
        }
      }
    }
  }

  return stats;
}
#endif  // !defined(__CUDA_ARCH__)

template <int HeadDim, bool IsCausal>
inline void run_fa2_fwd_kernel(flash::Flash_fwd_params &params,
                               cudaStream_t stream) {
  if constexpr (HeadDim == 64) {
    flash::run_mha_fwd_hdim64<cutlass::half_t, IsCausal>(params, stream);
  } else {
    static_assert(HeadDim == 128);
    flash::run_mha_fwd_hdim128<cutlass::half_t, IsCausal>(params, stream);
  }
}

inline void populate_fa2_params(flash::Flash_fwd_params &params,
                                cutlass::half_t *d_q, cutlass::half_t *d_k,
                                cutlass::half_t *d_v, cutlass::half_t *d_o,
                                float *d_lse, int batch, int seqlen_q,
                                int seqlen_k, int heads, int head_dim,
                                bool causal) {
  const int head_dim_rounded = head_dim;
  params = {};
  params.q_ptr = d_q;
  params.k_ptr = d_k;
  params.v_ptr = d_v;
  params.o_ptr = d_o;
  params.softmax_lse_ptr = d_lse;

  params.q_batch_stride = seqlen_q * heads * head_dim;
  params.k_batch_stride = seqlen_k * heads * head_dim;
  params.v_batch_stride = seqlen_k * heads * head_dim;
  params.o_batch_stride = seqlen_q * heads * head_dim;

  params.q_row_stride = heads * head_dim;
  params.k_row_stride = heads * head_dim;
  params.v_row_stride = heads * head_dim;
  params.o_row_stride = heads * head_dim;

  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.b = batch;
  params.h = heads;
  params.h_k = heads;
  params.h_h_k_ratio = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_knew = 0;
  params.total_q = batch * seqlen_q;
  params.d = head_dim;
  params.seqlen_q_rounded = seqlen_q;
  params.seqlen_k_rounded = seqlen_k;
  params.d_rounded = head_dim_rounded;
  params.rotary_dim = 0;

  params.scale_softmax = 1.0f / std::sqrt(float(head_dim));
  params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = causal ? 0 : -1;
  params.softcap = 0.0f;

  params.philox_args = at::PhiloxCudaState(0, 0);
  params.rng_state = nullptr;
  params.is_bf16 = false;
  params.is_causal = causal;
  params.is_seqlens_k_cumulative = true;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.unpadded_lse = false;
  params.seqlenq_ngroups_swapped = false;
}

template <int HeadDim, bool IsCausal>
inline Fa2RunResult run_fa2_fwd_fp16_typed(int batch, int seqlen_q,
                                           int seqlen_k, int heads,
                                           bool initialize_inputs,
                                           bool validate_reference = false) {
  const int D = HeadDim;
  const int H = heads;
  const int H_KV = heads;

  const size_t q_elems = size_t(batch) * seqlen_q * H * D;
  const size_t k_elems = size_t(batch) * seqlen_k * H_KV * D;
  const size_t v_elems = size_t(batch) * seqlen_k * H_KV * D;
  const size_t o_elems = size_t(batch) * seqlen_q * H * D;
  const size_t lse_elems = size_t(batch) * H * seqlen_q;

  cutlass::half_t *d_q = nullptr;
  cutlass::half_t *d_k = nullptr;
  cutlass::half_t *d_v = nullptr;
  cutlass::half_t *d_o = nullptr;
  float *d_lse = nullptr;
#if defined(FA2_FWD_CTA_PROFILE)
  flash::Flash_fwd_cta_profile_record *d_cta_profile = nullptr;
  const char *cta_profile_csv_path = std::getenv("FA2_CTA_PROFILE_CSV");
  const int cta_profile_count =
      ((seqlen_q + 127) / 128) * batch * heads;
#endif
  std::vector<cutlass::half_t> h_q;
  std::vector<cutlass::half_t> h_k;
  std::vector<cutlass::half_t> h_v;

  Fa2RunResult result;
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
#if defined(FA2_FWD_CTA_PROFILE)
    free_if_needed(d_cta_profile);
#endif

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

#define FA2_RETURN_IF_CUDA_ERROR(expr)                           \
  do {                                                           \
    cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr); \
  } while (0)

  FA2_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(&d_lse, lse_elems * sizeof(float)));

  if (initialize_inputs || validate_reference) {
    h_q.resize(q_elems);
    h_k.resize(k_elems);
    h_v.resize(v_elems);
    if (initialize_inputs) {
      fill_half(h_q, 0.25f);
      fill_half(h_k, 0.20f);
      fill_half(h_v, 0.30f);
    }
  }

  if (initialize_inputs) {
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(),
                                        q_elems * sizeof(cutlass::half_t),
                                        cudaMemcpyHostToDevice));
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_k, h_k.data(),
                                        k_elems * sizeof(cutlass::half_t),
                                        cudaMemcpyHostToDevice));
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(),
                                        v_elems * sizeof(cutlass::half_t),
                                        cudaMemcpyHostToDevice));
  } else {
    FA2_RETURN_IF_CUDA_ERROR(
        cudaMemset(d_q, 0, q_elems * sizeof(cutlass::half_t)));
    FA2_RETURN_IF_CUDA_ERROR(
        cudaMemset(d_k, 0, k_elems * sizeof(cutlass::half_t)));
    FA2_RETURN_IF_CUDA_ERROR(
        cudaMemset(d_v, 0, v_elems * sizeof(cutlass::half_t)));
  }
  FA2_RETURN_IF_CUDA_ERROR(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA2_RETURN_IF_CUDA_ERROR(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

  flash::Flash_fwd_params params;
  populate_fa2_params(params, d_q, d_k, d_v, d_o, d_lse, batch, seqlen_q,
                      seqlen_k, heads, D, IsCausal);

#if defined(FA2_FWD_CTA_PROFILE)
  if (cta_profile_csv_path != nullptr && cta_profile_csv_path[0] != '\0') {
    FA2_RETURN_IF_CUDA_ERROR(cudaMalloc(
        &d_cta_profile,
        size_t(cta_profile_count) *
            sizeof(flash::Flash_fwd_cta_profile_record)));
    FA2_RETURN_IF_CUDA_ERROR(cudaMemset(
        d_cta_profile, 0,
        size_t(cta_profile_count) *
            sizeof(flash::Flash_fwd_cta_profile_record)));
    params.cta_profile_ptr = d_cta_profile;
    params.cta_profile_capacity = cta_profile_count;
  }
#endif

  cudaStream_t stream = nullptr;
  run_fa2_fwd_kernel<HeadDim, IsCausal>(params, stream);

  FA2_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA2_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
#if defined(FA2_FWD_CTA_PROFILE)
  if (d_cta_profile != nullptr) {
    std::vector<flash::Flash_fwd_cta_profile_record> h_cta_profile(
        cta_profile_count);
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(
        h_cta_profile.data(), d_cta_profile,
        size_t(cta_profile_count) *
            sizeof(flash::Flash_fwd_cta_profile_record),
        cudaMemcpyDeviceToHost));
    std::ofstream out(cta_profile_csv_path);
    out << "cta_linear,cta_x,cta_y,cta_z,start_smid,end_smid,"
           "start_globaltimer,end_globaltimer,duration_globaltimer\n";
    for (int i = 0; i < cta_profile_count; ++i) {
      const auto &r = h_cta_profile[i];
      const uint64_t duration =
          r.end_globaltimer >= r.start_globaltimer
              ? r.end_globaltimer - r.start_globaltimer
              : 0;
      out << i << "," << r.cta_x << "," << r.cta_y << "," << r.cta_z
          << "," << r.start_smid << "," << r.end_smid << ","
          << r.start_globaltimer << "," << r.end_globaltimer << ","
          << duration << "\n";
    }
  }
#endif
  cutlass::half_t output0;
  FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(&output0, d_o, sizeof(cutlass::half_t),
                                      cudaMemcpyDeviceToHost));
  FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(&result.lse0, d_lse, sizeof(float),
                                      cudaMemcpyDeviceToHost));
  result.output0 = float(output0);

  if (validate_reference) {
#ifndef __CUDA_ARCH__
    std::vector<cutlass::half_t> h_o(o_elems);
    std::vector<float> h_lse(lse_elems);
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_o.data(), d_o,
                                        o_elems * sizeof(cutlass::half_t),
                                        cudaMemcpyDeviceToHost));
    FA2_RETURN_IF_CUDA_ERROR(cudaMemcpy(h_lse.data(), d_lse,
                                        lse_elems * sizeof(float),
                                        cudaMemcpyDeviceToHost));
    const Fa2ReferenceStats stats =
        compute_fa2_reference_errors<HeadDim, IsCausal>(
            h_q, h_k, h_v, h_o, h_lse, batch, seqlen_q, seqlen_k, heads);
    result.reference_checked = true;
    result.output0_ref = stats.output0_ref;
    result.lse0_ref = stats.lse0_ref;
    result.max_output_abs_error = stats.max_output_abs_error;
    result.max_lse_abs_error = stats.max_lse_abs_error;
    result.max_output_abs_error_index = stats.max_output_abs_error_index;
    result.max_lse_abs_error_index = stats.max_lse_abs_error_index;
#endif
  }

  return finish(cudaSuccess, "success");

#undef FA2_RETURN_IF_CUDA_ERROR
}

inline Fa2RunResult run_fa2_prefill_fp16(const Fa2PrefillCase &cfg,
                                         bool initialize_inputs = false,
                                         bool validate_reference = false) {
  if (!is_supported_fa2_prefill_case(cfg)) {
    return make_fa2_invalid_result("Fa2PrefillCase");
  }

#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
  if (cfg.head_dim == 64 && !cfg.causal) {
    return run_fa2_fwd_fp16_typed<64, false>(cfg.batch, cfg.seqlen,
                                             cfg.seqlen, cfg.heads,
                                             initialize_inputs,
                                             validate_reference);
  }
#endif
#if defined(FA2_PREFILL_ENABLE_H32D64_CAUSAL)
  if (cfg.head_dim == 64 && cfg.causal) {
    return run_fa2_fwd_fp16_typed<64, true>(cfg.batch, cfg.seqlen,
                                            cfg.seqlen, cfg.heads,
                                            initialize_inputs,
                                            validate_reference);
  }
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_FULL)
  if (cfg.head_dim == 128 && !cfg.causal) {
    return run_fa2_fwd_fp16_typed<128, false>(cfg.batch, cfg.seqlen,
                                              cfg.seqlen, cfg.heads,
                                              initialize_inputs,
                                              validate_reference);
  }
#endif
#if defined(FA2_PREFILL_ENABLE_H16D128_CAUSAL)
  if (cfg.head_dim == 128 && cfg.causal) {
    return run_fa2_fwd_fp16_typed<128, true>(cfg.batch, cfg.seqlen,
                                             cfg.seqlen, cfg.heads,
                                             initialize_inputs,
                                             validate_reference);
  }
#endif
  return make_fa2_invalid_result("Fa2PrefillVariantDisabled");
}

#if defined(FA2_PREFILL_GROUP_SENSITIVITY)
inline Fa2RunResult run_fa2_sensitivity_fp16(const Fa2PrefillCase &cfg) {
  if (!is_valid_fa2_prefill_sensitivity_case(cfg)) {
    return make_fa2_invalid_result("Fa2SensitivityCase");
  }

  return run_fa2_fwd_fp16_typed<128, false>(cfg.batch, cfg.seqlen,
                                            cfg.seqlen, cfg.heads, false);
}
#endif

#if defined(FA2_PREFILL_GROUP_SENSITIVITY_H1D128)
inline Fa2RunResult run_fa2_sensitivity_h1d128_fp16(
    const Fa2PrefillCase &cfg) {
  if (!is_valid_fa2_prefill_sensitivity_h1d128_case(cfg)) {
    return make_fa2_invalid_result("Fa2SensitivityH1D128Case");
  }

  if (cfg.causal) {
    return run_fa2_fwd_fp16_typed<128, true>(cfg.batch, cfg.seqlen,
                                             cfg.seqlen, cfg.heads, false);
  }
  return run_fa2_fwd_fp16_typed<128, false>(cfg.batch, cfg.seqlen,
                                            cfg.seqlen, cfg.heads, false);
}
#endif

#if defined(FA2_PREFILL_GROUP_SENSITIVITY_LARGE_D128_FULL)
inline Fa2RunResult run_fa2_sensitivity_large_d128_full_fp16(
    const Fa2PrefillCase &cfg) {
  if (!is_valid_fa2_prefill_sensitivity_large_d128_full_case(cfg)) {
    return make_fa2_invalid_result("Fa2SensitivityLargeD128FullCase");
  }

  return run_fa2_fwd_fp16_typed<128, false>(cfg.batch, cfg.seqlen,
                                            cfg.seqlen, cfg.heads, false);
}
#endif

inline Fa2RunResult run_fa2_fwd_smoke_fp16() {
#if defined(FA2_PREFILL_ENABLE_H32D64_FULL)
  return run_fa2_fwd_fp16_typed<64, false>(1, 128, 128, 2, true, true);
#else
  return make_fa2_invalid_result("Fa2FixedSmokeVariantDisabled");
#endif
}

}  // namespace fa2_hopper_test
