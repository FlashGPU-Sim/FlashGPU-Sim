// Fixed FlashAttention-3 Hopper forward case.
//
// Case:
//   B = 9, M = 64, N = 128
//   H = 6, H_KV = 6
//   D = 128, DV = 128
//   dtype = fp16
//   causal = false, local = false, softcap = 0
//   split-KV = false, paged-KV = false, pack-GQA = false
//
// This is meant for offline compilation / PTX inspection. It launches an
// SM90a/Hopper kernel and is not expected to run on non-Hopper GPUs.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace {

#define CHECK_CUDA_RT(expr)                                                   \
    do {                                                                      \
        cudaError_t status__ = (expr);                                        \
        if (status__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                  \
                         cudaGetErrorString(status__));                       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

template <typename T>
T *device_alloc(size_t count) {
    T *ptr = nullptr;
    CHECK_CUDA_RT(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

void fill_half(std::vector<cutlass::half_t> &x, float scale) {
    for (size_t i = 0; i < x.size(); ++i) {
        float v = std::sin(float(i % 251) * 0.013f) * scale;
        x[i] = cutlass::half_t(v);
    }
}

} // namespace

int main() {
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

    CHECK_CUDA_RT(cudaSetDevice(0));

    std::vector<cutlass::half_t> h_q(q_elems);
    std::vector<cutlass::half_t> h_k(k_elems);
    std::vector<cutlass::half_t> h_v(v_elems);
    std::vector<cutlass::half_t> h_o(o_elems);
    std::vector<float> h_lse(lse_elems);

    fill_half(h_q, 0.25f);
    fill_half(h_k, 0.20f);
    fill_half(h_v, 0.30f);

    cutlass::half_t *d_q = device_alloc<cutlass::half_t>(q_elems);
    cutlass::half_t *d_k = device_alloc<cutlass::half_t>(k_elems);
    cutlass::half_t *d_v = device_alloc<cutlass::half_t>(v_elems);
    cutlass::half_t *d_o = device_alloc<cutlass::half_t>(o_elems);
    float *d_lse = device_alloc<float>(lse_elems);

    CHECK_CUDA_RT(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(cutlass::half_t),
                             cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemcpy(d_k, h_k.data(), k_elems * sizeof(cutlass::half_t),
                             cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemcpy(d_v, h_v.data(), v_elems * sizeof(cutlass::half_t),
                             cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemset(d_o, 0, o_elems * sizeof(cutlass::half_t)));
    CHECK_CUDA_RT(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));

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

    CHECK_CUDA_RT(cudaDeviceSynchronize());
    CHECK_CUDA_RT(cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(cutlass::half_t),
                             cudaMemcpyDeviceToHost));
    CHECK_CUDA_RT(cudaMemcpy(h_lse.data(), d_lse, lse_elems * sizeof(float),
                             cudaMemcpyDeviceToHost));

    std::printf("o[0] ~= %.6f, lse[0] = %.6f\n",
                float(h_o[0]), h_lse[0]);

    CHECK_CUDA_RT(cudaFree(d_q));
    CHECK_CUDA_RT(cudaFree(d_k));
    CHECK_CUDA_RT(cudaFree(d_v));
    CHECK_CUDA_RT(cudaFree(d_o));
    CHECK_CUDA_RT(cudaFree(d_lse));
    return 0;
}
