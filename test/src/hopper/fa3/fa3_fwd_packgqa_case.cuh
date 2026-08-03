#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "fa3_fwd_hdim128_fp16_case.cuh"

namespace fa3_hopper_test {

struct Fa3PackGqaRunResult {
  cudaError_t error = cudaSuccess;
  const char *where = "success";
  bool reference_checked = false;
  float output0 = 0.0f;
  float output0_ref = 0.0f;
  float lse0 = 0.0f;
  float lse0_ref = 0.0f;
  float max_output_abs_error = 0.0f;
  float max_lse_abs_error = 0.0f;
  size_t max_output_abs_error_index = 0;
  size_t max_lse_abs_error_index = 0;
};

inline size_t fa3_packgqa_qo_index(int batch_idx, int row, int head,
                                   int dim, int seqlen, int heads,
                                   int head_dim) {
  return ((size_t(batch_idx) * seqlen + row) * heads + head) * head_dim +
         dim;
}

inline size_t fa3_packgqa_kv_index(int batch_idx, int row, int kv_head,
                                   int dim, int seqlen, int heads_kv,
                                   int head_dim) {
  return ((size_t(batch_idx) * seqlen + row) * heads_kv + kv_head) *
             head_dim +
         dim;
}

inline size_t fa3_packgqa_lse_index(int batch_idx, int head, int row,
                                    int seqlen, int heads) {
  return (size_t(batch_idx) * heads + head) * seqlen + row;
}

struct Fa3PackGqaReferenceStats {
  float output0_ref = 0.0f;
  float lse0_ref = 0.0f;
  float max_output_abs_error = 0.0f;
  float max_lse_abs_error = 0.0f;
  size_t max_output_abs_error_index = 0;
  size_t max_lse_abs_error_index = 0;
};

template <int HeadDim>
inline Fa3PackGqaReferenceStats compute_fa3_packgqa_reference_errors(
    const std::vector<cutlass::half_t> &q,
    const std::vector<cutlass::half_t> &k,
    const std::vector<cutlass::half_t> &v,
    const std::vector<cutlass::half_t> &actual_o,
    const std::vector<float> &actual_lse, int batch, int seqlen_q,
    int seqlen_k, int heads, int heads_kv) {
  constexpr int D = HeadDim;
  const int qheads_per_kv_head = heads / heads_kv;
  const float scale = 1.0f / std::sqrt(float(D));
  std::vector<float> scores(seqlen_k);
  Fa3PackGqaReferenceStats stats;

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kv_head = h / qheads_per_kv_head;
      for (int m = 0; m < seqlen_q; ++m) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < seqlen_k; ++n) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            const size_t q_idx = fa3_packgqa_qo_index(
                b, m, h, d, seqlen_q, heads, D);
            const size_t k_idx = fa3_packgqa_kv_index(
                b, n, kv_head, d, seqlen_k, heads_kv, D);
            dot += float(q[q_idx]) * float(k[k_idx]);
          }
          scores[n] = dot * scale;
          max_score = std::max(max_score, scores[n]);
        }

        float sum_exp = 0.0f;
        for (int n = 0; n < seqlen_k; ++n) {
          scores[n] = std::exp(scores[n] - max_score);
          sum_exp += scores[n];
        }
        const float inv_sum = 1.0f / sum_exp;
        const float ref_lse = max_score + std::log(sum_exp);
        const size_t lse_idx =
            fa3_packgqa_lse_index(b, h, m, seqlen_q, heads);
        if (lse_idx == 0) stats.lse0_ref = ref_lse;
        float lse_error = std::fabs(actual_lse[lse_idx] - ref_lse);
        if (!std::isfinite(lse_error)) {
          lse_error = std::numeric_limits<float>::infinity();
        }
        if (lse_error > stats.max_lse_abs_error) {
          stats.max_lse_abs_error = lse_error;
          stats.max_lse_abs_error_index = lse_idx;
        }

        for (int d = 0; d < D; ++d) {
          float ref = 0.0f;
          for (int n = 0; n < seqlen_k; ++n) {
            const size_t v_idx = fa3_packgqa_kv_index(
                b, n, kv_head, d, seqlen_k, heads_kv, D);
            ref += scores[n] * inv_sum * float(v[v_idx]);
          }
          const size_t o_idx = fa3_packgqa_qo_index(
              b, m, h, d, seqlen_q, heads, D);
          if (o_idx == 0) stats.output0_ref = ref;
          float output_error = std::fabs(float(actual_o[o_idx]) - ref);
          if (!std::isfinite(output_error)) {
            output_error = std::numeric_limits<float>::infinity();
          }
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

inline Fa3PackGqaRunResult run_fa3_fwd_packgqa_hdim128_fp16() {
  constexpr int B = 1;
  constexpr int M = 32;
  constexpr int N = 32;
  constexpr int H = 4;
  constexpr int H_KV = 1;
  constexpr int D = 128;
  constexpr int DV = 128;
  static_assert(H % H_KV == 0);
  static_assert(M * (H / H_KV) == 128,
                "PackGQA smoke case should occupy exactly one M tile");

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

  Fa3PackGqaRunResult result;
  auto finish = [&](cudaError_t error, const char *where) {
    cudaError_t cleanup_error = cudaSuccess;
    auto free_if_needed = [&](auto *ptr) {
      if (ptr == nullptr) return;
      const cudaError_t free_error = cudaFree(ptr);
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

#define FA3_PACKGQA_RETURN_IF_CUDA_ERROR(expr)                         \
  do {                                                                 \
    const cudaError_t status__ = (expr);                               \
    if (status__ != cudaSuccess) return finish(status__, #expr);       \
  } while (0)

  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaSetDevice(0));

  std::vector<cutlass::half_t> h_q(q_elems);
  std::vector<cutlass::half_t> h_k(k_elems);
  std::vector<cutlass::half_t> h_v(v_elems);
  std::vector<cutlass::half_t> h_o(o_elems);
  std::vector<float> h_lse(lse_elems);
  fill_half(h_q, 0.25f);
  fill_half(h_k, 0.20f);
  fill_half(h_v, 0.30f);

  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMalloc(&d_q, q_elems * sizeof(cutlass::half_t)));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMalloc(&d_k, k_elems * sizeof(cutlass::half_t)));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMalloc(&d_v, v_elems * sizeof(cutlass::half_t)));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMalloc(&d_o, o_elems * sizeof(cutlass::half_t)));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMalloc(&d_lse, lse_elems * sizeof(float)));

  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      d_q, h_q.data(), q_elems * sizeof(cutlass::half_t),
      cudaMemcpyHostToDevice));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      d_k, h_k.data(), k_elems * sizeof(cutlass::half_t),
      cudaMemcpyHostToDevice));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      d_v, h_v.data(), v_elems * sizeof(cutlass::half_t),
      cudaMemcpyHostToDevice));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(
      cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

  Flash_fwd_params params = {};
  set_fa3_prefill_base_params(params, d_q, d_k, d_v, d_o, d_lse, B, M, N,
                              H, H_KV, D, DV, M, N, false);
  params.pack_gqa = true;
  params.varlen_sort_batches = false;
  params.head_swizzle = false;

  cudaStream_t stream = nullptr;
  run_flash_fwd<90, D, DV, 1, cutlass::half_t, cutlass::half_t,
                false, false, false, false, false,
                false, false, true, false, false>(params, stream);

  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize());
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      h_o.data(), d_o, o_elems * sizeof(cutlass::half_t),
      cudaMemcpyDeviceToHost));
  FA3_PACKGQA_RETURN_IF_CUDA_ERROR(cudaMemcpy(
      h_lse.data(), d_lse, lse_elems * sizeof(float),
      cudaMemcpyDeviceToHost));

  const Fa3PackGqaReferenceStats stats =
      compute_fa3_packgqa_reference_errors<D>(h_q, h_k, h_v, h_o, h_lse, B,
                                               M, N, H, H_KV);
  result.reference_checked = true;
  result.output0 = float(h_o[0]);
  result.output0_ref = stats.output0_ref;
  result.lse0 = h_lse[0];
  result.lse0_ref = stats.lse0_ref;
  result.max_output_abs_error = stats.max_output_abs_error;
  result.max_lse_abs_error = stats.max_lse_abs_error;
  result.max_output_abs_error_index = stats.max_output_abs_error_index;
  result.max_lse_abs_error_index = stats.max_lse_abs_error_index;
  return finish(cudaSuccess, "success");

#undef FA3_PACKGQA_RETURN_IF_CUDA_ERROR
}

}  // namespace fa3_hopper_test
