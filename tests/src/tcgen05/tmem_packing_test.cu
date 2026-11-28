#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
// Four warps deliberately use the same TMEM base. Each must address its own
// 32-datapath partition; distinct low/high halfwords catch accidental aliases.
template <bool Split>
__global__ void packing(unsigned *out) {
  __shared__ unsigned base;
  const unsigned tid = threadIdx.x;
  if (tid < 32)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],64;" ::
            "r"((unsigned)__cvta_generic_to_shared(&base))
        : "memory");
  __syncthreads();
  const unsigned lo = 100 + blockIdx.x * 1024 + tid;
  const unsigned v0 = lo | ((lo + 256) << 16);
  const unsigned v1 = (lo + 1) | ((lo + 257) << 16);
  const unsigned v2 = (lo + 2) | ((lo + 258) << 16);
  const unsigned v3 = (lo + 3) | ((lo + 259) << 16);
  if constexpr (Split)
    asm volatile(
        "tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 "
        "[%0],16,{%1,%2,%3,%4};" ::"r"(base),
        "r"(v0), "r"(v1), "r"(v2), "r"(v3)
        : "memory");
  else
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 "
        "[%0],{%1,%2,%3,%4};" ::"r"(base),
        "r"(v0), "r"(v1), "r"(v2), "r"(v3)
        : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  unsigned a, b, c, d;
  // Plain reads independently check unpacking, not just a symmetric roundtrip.
  const unsigned read_base = base;
  if constexpr (Split)
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x4.b32 {%0,%1,%2,%3},[%4],16;"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(read_base));
  else
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "r"(read_base));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  const unsigned i = (blockIdx.x * blockDim.x + tid) * 8;
  out[i] = a;
  out[i + 1] = b;
  out[i + 2] = c;
  out[i + 3] = d;
  if constexpr (Split)
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 "
        "{%0,%1,%2,%3},[%4],16;"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(base));
  else
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0,%1,%2,%3},[%4];"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(base));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  out[i + 4] = a;
  out[i + 5] = b;
  out[i + 6] = c;
  out[i + 7] = d;
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,64;" ::"r"(base)
        : "memory");
  }
}

template <bool Split>
void check_packing() {
  unsigned *device = nullptr;
  std::vector<unsigned> output(2 * 128 * 8);
  ASSERT_EQ(cudaMalloc(&device, output.size() * 4), cudaSuccess);
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    packing<Split><<<2, 128>>>(device);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), device, output.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (unsigned cta = 0; cta < 2; ++cta)
      for (unsigned tid = 0; tid < 128; ++tid) {
        const unsigned lo = 100 + cta * 1024 + tid;
        const unsigned i = (cta * 128 + tid) * 8;
        for (unsigned j = 0; j < 4; ++j) {
          EXPECT_EQ(output[i + j], lo + j / 2 + (j % 2) * 256)
              << cta << ":" << tid << ":" << j;
          EXPECT_EQ(output[i + 4 + j], (lo + j) | ((lo + j + 256) << 16))
              << cta << ":" << tid << ":" << j;
        }
      }
  }
  EXPECT_EQ(cudaFree(device), cudaSuccess);
}
TEST(Tcgen05Fragments, PackUnpackFourWarps) { check_packing<false>(); }
TEST(Tcgen05Fragments, HalfSplitPackUnpackFourWarps) { check_packing<true>(); }
}  // namespace
