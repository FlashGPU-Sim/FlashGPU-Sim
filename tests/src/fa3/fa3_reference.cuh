#pragma once

#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace fa3_hopper_test {

struct Fa3TensorComparison {
  float worst_error_ratio = 0.0f;
  float abs_error = 0.0f;
  float allowed_error = 0.0f;
  float actual = 0.0f;
  float expected = 0.0f;
  size_t index = 0;
};

struct Fa3ForwardReference {
  std::vector<float> output;
  std::vector<cutlass::half_t> output_half;
  std::vector<float> lse;
};

struct Fa3BackwardReference {
  std::vector<float> dq;
  std::vector<float> dk;
  std::vector<float> dv;
};

inline size_t fa3_qkv_index(int batch_idx, int row, int head, int dim,
                            int seqlen, int heads, int head_dim) {
  return ((size_t(batch_idx) * seqlen + row) * heads + head) * head_dim + dim;
}

inline size_t fa3_lse_index(int batch_idx, int head, int row, int seqlen,
                            int heads) {
  return (size_t(batch_idx) * heads + head) * seqlen + row;
}

template <typename Actual>
inline Fa3TensorComparison compare_fa3_tensor(
    const std::vector<Actual> &actual, const std::vector<float> &expected,
    float abs_tolerance, float rel_tolerance) {
  Fa3TensorComparison comparison;
  if (actual.size() != expected.size()) {
    comparison.worst_error_ratio = std::numeric_limits<float>::infinity();
    return comparison;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    const float actual_value = float(actual[i]);
    const float expected_value = expected[i];
    const float abs_error = std::fabs(actual_value - expected_value);
    const float allowed_error =
        abs_tolerance + rel_tolerance * std::fabs(expected_value);
    float error_ratio = abs_error / allowed_error;
    if (!std::isfinite(error_ratio)) {
      error_ratio = std::numeric_limits<float>::infinity();
    }
    if (i == 0 || error_ratio > comparison.worst_error_ratio) {
      comparison.worst_error_ratio = error_ratio;
      comparison.abs_error = abs_error;
      comparison.allowed_error = allowed_error;
      comparison.actual = actual_value;
      comparison.expected = expected_value;
      comparison.index = i;
    }
  }
  return comparison;
}

template <int HeadDim, bool IsCausal>
inline Fa3ForwardReference compute_fa3_forward_reference(
    const std::vector<cutlass::half_t> &q,
    const std::vector<cutlass::half_t> &k,
    const std::vector<cutlass::half_t> &v, int batch, int seqlen_q,
    int seqlen_k, int heads) {
  constexpr int D = HeadDim;
  const float scale = 1.0f / std::sqrt(float(D));
  const size_t output_elems = size_t(batch) * seqlen_q * heads * D;
  const size_t lse_elems = size_t(batch) * heads * seqlen_q;

  Fa3ForwardReference reference;
  reference.output.resize(output_elems);
  reference.output_half.resize(output_elems);
  reference.lse.resize(lse_elems);
  std::vector<float> probabilities(seqlen_k);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int m = 0; m < seqlen_q; ++m) {
        const int valid_k = IsCausal ? std::min(m + 1, seqlen_k) : seqlen_k;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < valid_k; ++n) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            dot += float(q[fa3_qkv_index(b, m, h, d, seqlen_q, heads, D)]) *
                   float(k[fa3_qkv_index(b, n, h, d, seqlen_k, heads, D)]);
          }
          probabilities[n] = dot * scale;
          max_score = std::max(max_score, probabilities[n]);
        }

        float sum_exp = 0.0f;
        for (int n = 0; n < valid_k; ++n) {
          probabilities[n] = std::exp(probabilities[n] - max_score);
          sum_exp += probabilities[n];
        }
        const float inv_sum = 1.0f / sum_exp;
        reference.lse[fa3_lse_index(b, h, m, seqlen_q, heads)] =
            max_score + std::log(sum_exp);

        for (int d = 0; d < D; ++d) {
          float output = 0.0f;
          for (int n = 0; n < valid_k; ++n) {
            output += probabilities[n] * inv_sum *
                      float(v[fa3_qkv_index(b, n, h, d, seqlen_k, heads, D)]);
          }
          const size_t index = fa3_qkv_index(b, m, h, d, seqlen_q, heads, D);
          reference.output[index] = output;
          reference.output_half[index] = cutlass::half_t(output);
        }
      }
    }
  }
  return reference;
}

template <int HeadDim, bool IsCausal>
inline Fa3BackwardReference compute_fa3_backward_reference(
    const std::vector<cutlass::half_t> &q,
    const std::vector<cutlass::half_t> &k,
    const std::vector<cutlass::half_t> &v,
    const std::vector<cutlass::half_t> &output,
    const std::vector<cutlass::half_t> &dout, int batch, int seqlen_q,
    int seqlen_k, int heads) {
  constexpr int D = HeadDim;
  const float scale = 1.0f / std::sqrt(float(D));

  Fa3BackwardReference reference;
  reference.dq.assign(q.size(), 0.0f);
  reference.dk.assign(k.size(), 0.0f);
  reference.dv.assign(v.size(), 0.0f);
  std::vector<float> probabilities(seqlen_k);
  std::vector<float> dp(seqlen_k);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int m = 0; m < seqlen_q; ++m) {
        const int valid_k = IsCausal ? std::min(m + 1, seqlen_k) : seqlen_k;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < valid_k; ++n) {
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            dot += float(q[fa3_qkv_index(b, m, h, d, seqlen_q, heads, D)]) *
                   float(k[fa3_qkv_index(b, n, h, d, seqlen_k, heads, D)]);
          }
          probabilities[n] = dot * scale;
          max_score = std::max(max_score, probabilities[n]);
        }

        float sum_exp = 0.0f;
        for (int n = 0; n < valid_k; ++n) {
          probabilities[n] = std::exp(probabilities[n] - max_score);
          sum_exp += probabilities[n];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (int n = 0; n < valid_k; ++n) {
          probabilities[n] *= inv_sum;
        }

        float delta = 0.0f;
        for (int d = 0; d < D; ++d) {
          const size_t output_index =
              fa3_qkv_index(b, m, h, d, seqlen_q, heads, D);
          delta += float(dout[output_index]) * float(output[output_index]);
        }

        for (int n = 0; n < valid_k; ++n) {
          float dp_value = 0.0f;
          for (int d = 0; d < D; ++d) {
            const size_t output_index =
                fa3_qkv_index(b, m, h, d, seqlen_q, heads, D);
            const size_t value_index =
                fa3_qkv_index(b, n, h, d, seqlen_k, heads, D);
            dp_value += float(dout[output_index]) * float(v[value_index]);
            reference.dv[value_index] +=
                probabilities[n] * float(dout[output_index]);
          }
          dp[n] = dp_value;
        }

        for (int n = 0; n < valid_k; ++n) {
          const float ds = probabilities[n] * (dp[n] - delta) * scale;
          for (int d = 0; d < D; ++d) {
            const size_t q_index =
                fa3_qkv_index(b, m, h, d, seqlen_q, heads, D);
            const size_t k_index =
                fa3_qkv_index(b, n, h, d, seqlen_k, heads, D);
            reference.dq[q_index] += ds * float(k[k_index]);
            reference.dk[k_index] += ds * float(q[q_index]);
          }
        }
      }
    }
  }
  return reference;
}

}  // namespace fa3_hopper_test
