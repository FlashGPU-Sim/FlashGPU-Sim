#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#define TMEM_R16(P)                                                          \
  #P "0," #P "1," #P "2," #P "3," #P "4," #P "5," #P "6," #P "7," #P "8," #P \
     "9," #P "10," #P "11," #P "12," #P "13," #P "14," #P "15"
#define TMEM_R128                                               \
  TMEM_R16(a)                                                   \
  "," TMEM_R16(b) "," TMEM_R16(c) "," TMEM_R16(d) "," TMEM_R16( \
      e) "," TMEM_R16(f) "," TMEM_R16(g) "," TMEM_R16(h)
namespace {
__global__ void wide_transfer(unsigned *out) {
  __shared__ unsigned base;
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],256;" ::"r"(
          (unsigned)__cvta_generic_to_shared(&base))
      : "memory");
  __syncthreads();
  for (unsigned col = 0; col < 128; ++col) {
    const unsigned value = 10000 * blockIdx.x + 128 * threadIdx.x + col;
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(base + col),
        "r"(value)
        : "memory");
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile(
      "{ .reg .b32 a<16>,b<16>,c<16>,d<16>,e<16>,f<16>,g<16>,h<16>;\n"
      "tcgen05.ld.sync.aligned.32x32b.x128.b32 {" TMEM_R128
      "},[%0];\n"
      "tcgen05.wait::ld.sync.aligned;\n"
      "tcgen05.st.sync.aligned.32x32b.x128.b32 [%1],{" TMEM_R128
      "};\n"
      "tcgen05.wait::st.sync.aligned; }" ::"r"(base),
      "r"(base + 128)
      : "memory");
  for (unsigned col = 0; col < 128; ++col) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                 : "=r"(value)
                 : "r"(base + 128 + col));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    out[(blockIdx.x * 32 + threadIdx.x) * 128 + col] = value;
  }
  __syncthreads();
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                   : "memory");
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,256;" ::"r"(base)
      : "memory");
}
TEST(Tcgen05Fragments, Full128RegisterTransferPreservesEveryColumn) {
  unsigned *device = nullptr;
  std::vector<unsigned> output(2 * 32 * 128);
  ASSERT_EQ(cudaMalloc(&device, output.size() * 4), cudaSuccess);
  wide_transfer<<<2, 32>>>(device);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device, output.size() * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned col = 0; col < 128; ++col)
        EXPECT_EQ(output[(cta * 32 + lane) * 128 + col],
                  cta * 10000 + lane * 128 + col)
            << cta << ":" << lane << ":" << col;
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
}  // namespace
#undef TMEM_R128
#undef TMEM_R16
