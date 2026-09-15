// A deliberately small, fixed-shape GEMM used to close the first real SASS
// functional-execution path.  The kernel is launched as a single thread; the
// point of this fixture is ISA correctness, not throughput.
extern "C" __global__ void sass_gemm_2x2x4(const float *__restrict__ a,
                                           const float *__restrict__ b,
                                           float *__restrict__ c) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        acc += a[m * 4 + k] * b[k * 2 + n];
      c[m * 2 + n] = acc;
    }
  }
}
