#include <stdint.h>

// One warp computes a complete 16x8x8 FP16 GEMM tile.  Fragment placement
// follows the PTX mma.sync m16n8k8 contract; the compiled fixture is consumed
// only through its SM120 SASS image.
extern "C" __global__ void sass_mma_gemm_m16n8k8(const uint16_t *__restrict__ a,
                                                 const uint16_t *__restrict__ b,
                                                 float *__restrict__ d) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned group = lane >> 2;
  const unsigned thread = lane & 3;
  const unsigned column0 = thread * 2;

  const uint16_t a_values[4] = {
      a[group * 8 + column0], a[group * 8 + column0 + 1],
      a[(group + 8) * 8 + column0], a[(group + 8) * 8 + column0 + 1]};
  const uint16_t b_values[2] = {b[group * 8 + column0],
                                b[group * 8 + column0 + 1]};
  const unsigned a0 = static_cast<unsigned>(a_values[0]) |
                      (static_cast<unsigned>(a_values[1]) << 16);
  const unsigned a1 = static_cast<unsigned>(a_values[2]) |
                      (static_cast<unsigned>(a_values[3]) << 16);
  const unsigned b0 = static_cast<unsigned>(b_values[0]) |
                      (static_cast<unsigned>(b_values[1]) << 16);
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(b0));

  d[group * 8 + column0] = d0;
  d[group * 8 + column0 + 1] = d1;
  d[(group + 8) * 8 + column0] = d2;
  d[(group + 8) * 8 + column0 + 1] = d3;
}
