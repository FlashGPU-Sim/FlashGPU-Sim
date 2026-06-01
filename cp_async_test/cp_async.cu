// cp.async (Ampere LDGSTS) functional test for FlashGPU-Sim.
// Uses inline PTX so the exact cp.async forms are exercised:
//   - cp.async.ca.shared.global [smem],[gmem],16      (full 16B copy)
//   - cp.async.ca.shared.global [smem],[gmem],16,src  (src-size zero-fill)
//   - @p cp.async.ca.shared.global ... (ignore-src predicate)
//   - cp.async.commit_group ; cp.async.wait_group 0 ;
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned smem_u32(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// Test 1: each thread copies 16 bytes (float4) global->shared->global.
__global__ void cpasync_copy16(const float *in, float *out, int n) {
  extern __shared__ float s[];
  int t = threadIdx.x;
  if (t * 4 + 3 < n) {
    unsigned sa = smem_u32(&s[t * 4]);
    const float *ga = &in[t * 4];
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sa),
                 "l"(ga));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
  }
  __syncthreads();
  if (t * 4 + 3 < n)
    for (int i = 0; i < 4; ++i) out[t * 4 + i] = s[t * 4 + i];
}

// Test 2: copy cp-size=16 but src-size=8 -> upper 8 bytes (2 floats) zero-filled.
__global__ void cpasync_srcsize(const float *in, float *out) {
  __shared__ float s[4];
  if (threadIdx.x == 0) {
    unsigned sa = smem_u32(&s[0]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 8;\n" ::"r"(sa),
                 "l"(in));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int i = 0; i < 4; ++i) out[i] = s[i];
}

// Test 3: instruction-level predication. @p false -> the cp.async is SKIPPED
// (NOT zero-filled); shared keeps its prior value. We pre-init shared to a
// sentinel and check it is preserved when the predicate is false.
__global__ void cpasync_pred(const float *in, float *out, int do_copy) {
  __shared__ float s[4];
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i) s[i] = -1.0f;  // sentinel
    unsigned sa = smem_u32(&s[0]);
    int p = do_copy;
    asm volatile(
        "{\n\t"
        " .reg .pred %%q;\n\t"
        " setp.ne.s32 %%q, %2, 0;\n\t"
        " @%%q cp.async.ca.shared.global [%0], [%1], 16;\n\t"
        "}\n" ::"r"(sa),
        "l"(in), "r"(p));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int i = 0; i < 4; ++i) out[i] = s[i];
}

int main() {
  const int N = 256;
  size_t bytes = N * sizeof(float);
  float *hin = (float *)malloc(bytes), *hout = (float *)malloc(bytes);
  for (int i = 0; i < N; ++i) hin[i] = (float)(i + 1);

  float *din, *dout;
  cudaMalloc(&din, bytes);
  cudaMalloc(&dout, bytes);
  cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice);

  int errors = 0;

  // Test 1: identity copy.
  cudaMemset(dout, 0, bytes);
  cpasync_copy16<<<1, N / 4, bytes>>>(din, dout, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i)
    if (hout[i] != hin[i]) errors++;
  printf("[T1 copy16]     errors=%d  out[5]=%.1f (expect 6.0)\n", errors,
         hout[5]);

  // Test 2: src-size=8 -> s[0],s[1] from in, s[2],s[3] zero.
  cudaMemset(dout, 0, bytes);
  cpasync_srcsize<<<1, 32>>>(din, dout);
  cudaDeviceSynchronize();
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  int e2 = 0;
  if (hout[0] != hin[0] || hout[1] != hin[1]) e2++;
  if (hout[2] != 0.0f || hout[3] != 0.0f) e2++;
  printf("[T2 srcsize8]   errors=%d  out={%.1f,%.1f,%.1f,%.1f} (expect "
         "1,2,0,0)\n",
         e2, hout[0], hout[1], hout[2], hout[3]);
  errors += e2;

  // Test 3a: predicate true -> copies. 3b: predicate false -> skip (sentinel).
  cudaMemset(dout, 0, bytes);
  cpasync_pred<<<1, 32>>>(din, dout, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  int e3a = (hout[0] == hin[0] && hout[3] == hin[3]) ? 0 : 1;

  cudaMemset(dout, 0, bytes);
  cpasync_pred<<<1, 32>>>(din, dout, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  int e3b = (hout[0] == -1.0f && hout[3] == -1.0f) ? 0 : 1;  // sentinel kept
  printf("[T3 predication] errors=%d (true-copy=%d, false-skip=%d)\n",
         e3a + e3b, e3a, e3b);
  errors += e3a + e3b;

  printf(errors == 0 ? "CP_ASYNC TEST PASSED\n" : "CP_ASYNC TEST FAILED\n");
  cudaFree(din);
  cudaFree(dout);
  free(hin);
  free(hout);
  return errors == 0 ? 0 : 1;
}
