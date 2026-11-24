#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
__global__ void fragments(unsigned *out) {
  __shared__ unsigned base;
  const unsigned lane = threadIdx.x;
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],32;" ::"r"(
          (unsigned)__cvta_generic_to_shared(&base))
      : "memory");
  __syncthreads();
  for (unsigned col = 0; col < 32; col += 4) {
    const unsigned v = (blockIdx.x + 1) * 100000 + lane * 100 + col;
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};" ::"r"(base +
                                                                          col),
        "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3)
        : "memory");
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#define READ_SHAPE(SHAPE, REPEAT, INDEX, EXTRA)                      \
  do {                                                               \
    unsigned a, b, c, d;                                             \
    asm volatile("tcgen05.ld.sync.aligned." SHAPE "." REPEAT         \
                 ".b32 {%0,%1,%2,%3},[%4]" EXTRA ";"                 \
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)                \
                 : "r"(base));                                       \
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");     \
    const unsigned idx = ((blockIdx.x * 6 + INDEX) * 32 + lane) * 4; \
    out[idx] = a;                                                    \
    out[idx + 1] = b;                                                \
    out[idx + 2] = c;                                                \
    out[idx + 3] = d;                                                \
  } while (0)
  READ_SHAPE("32x32b", "x4", 0, "");
  READ_SHAPE("16x64b", "x4", 1, "");
  READ_SHAPE("16x128b", "x2", 2, "");
  READ_SHAPE("16x256b", "x1", 3, "");
  READ_SHAPE("16x32bx2", "x4", 4, ",16");
  const unsigned v = (blockIdx.x + 1) * 200000 + lane * 100;
  asm volatile(
      "tcgen05.st.sync.aligned.16x32bx2.x4.b32 [%0],16,{%1,%2,%3,%4};" ::"r"(
          base),
      "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3)
      : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  READ_SHAPE("32x32b", "x4", 5, "");
#undef READ_SHAPE
  __syncthreads();
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                   : "memory");
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,32;" ::"r"(base)
      : "memory");
}
TEST(Tcgen05Fragments, CrossShapeLoadsAndHalfSplitStores) {
  unsigned *device = nullptr;
  std::vector<unsigned> output(2 * 6 * 32 * 4);
  ASSERT_EQ(cudaMalloc(&device, output.size() * 4), cudaSuccess);
  fragments<<<2, 32>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device, output.size() * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned shape = 0; shape < 6; ++shape)
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned r = 0; r < 4; ++r) {
          unsigned row = lane, col = r;
          if (shape == 1) {
            row = lane / 4 + (lane % 2) * 8;
            col = lane % 4 / 2 + r * 2;
          }
          if (shape == 2) {
            row = lane / 4 + (r % 2) * 8;
            col = lane % 4 + (r / 2) * 4;
          }
          if (shape == 3) {
            row = lane / 4 + (r / 2) * 8;
            col = (lane % 4) * 2 + r % 2;
          }
          if (shape == 4) {
            row = lane % 16;
            col = r + (lane >= 16 ? 16 : 0);
          }
          unsigned expected = (cta + 1) * 100000 + row * 100 + col;
          if (shape == 5 && lane < 16)
            expected = (cta + 1) * 200000 + lane * 100 + r;
          EXPECT_EQ(output[((cta * 6 + shape) * 32 + lane) * 4 + r], expected)
              << "cta=" << cta << " shape=" << shape << " lane=" << lane
              << " reg=" << r;
        }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
}  // namespace
