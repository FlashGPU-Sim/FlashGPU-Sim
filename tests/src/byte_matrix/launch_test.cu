#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
__host__ __device__ unsigned char payload(unsigned cta, unsigned warp, unsigned matrix, unsigned lane, unsigned byte) {
  return 1 + ((cta * 23 + warp * 47 + matrix * 71 + lane * 4 + byte) % 251);
}
template <unsigned Count, unsigned Address>
__global__ void store_bytes(unsigned *out) {
  __shared__ __align__(16) unsigned char tile[2][4 * 256];
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (unsigned i = lane; i < 1024; i += 32) tile[warp][i] = 0xac;
  __syncthreads();
  unsigned values[4]{};
  #pragma unroll
  for (unsigned m = 0; m < Count; ++m)
    #pragma unroll
    for (unsigned b = 0; b < 4; ++b)
      values[m] |= unsigned(payload(blockIdx.x, warp, m, lane, b)) << (8*b);
  auto *ptr = &tile[warp][(lane / 8) * 256 + ((lane % 8 * 3) % 8) * 32];
  const unsigned addr = __cvta_generic_to_shared(ptr);
  if constexpr (Count == 1 && Address == 0)
    asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared::cta.b8 [%0],{%1};"
      :: "r"(addr), "r"(values[0]) : "memory");
  if constexpr (Count == 2 && Address == 0)
    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared::cta.b8 [%0],{%1, %2};"
      :: "r"(addr), "r"(values[0]), "r"(values[1]) : "memory");
  if constexpr (Count == 4 && Address == 0)
    asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared::cta.b8 [%0],{%1, %2, %3, %4};"
      :: "r"(addr), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]) : "memory");
  if constexpr (Count == 1 && Address == 1)
    asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0],{%1};"
      :: "l"(uint64_t(addr)), "r"(values[0]) : "memory");
  if constexpr (Count == 2 && Address == 1)
    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0],{%1, %2};"
      :: "l"(uint64_t(addr)), "r"(values[0]), "r"(values[1]) : "memory");
  if constexpr (Count == 4 && Address == 1)
    asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0],{%1, %2, %3, %4};"
      :: "l"(uint64_t(addr)), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]) : "memory");
  if constexpr (Count == 1 && Address == 2)
    asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.b8 [%0],{%1};"
      :: "l"(reinterpret_cast<uint64_t>(ptr)), "r"(values[0]) : "memory");
  if constexpr (Count == 2 && Address == 2)
    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.b8 [%0],{%1, %2};"
      :: "l"(reinterpret_cast<uint64_t>(ptr)), "r"(values[0]), "r"(values[1]) : "memory");
  if constexpr (Count == 4 && Address == 2)
    asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.b8 [%0],{%1, %2, %3, %4};"
      :: "l"(reinterpret_cast<uint64_t>(ptr)), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]) : "memory");
  __syncthreads();
  const auto *words = reinterpret_cast<const unsigned *>(tile[warp]);
  for (unsigned i = lane; i < 256; i += 32) out[(blockIdx.x*2+warp)*256+i] = words[i];
}
template<unsigned Count, unsigned Address> void check() {
  unsigned char *device;
  ASSERT_EQ(cudaMalloc(&device, 4096), cudaSuccess);
  store_bytes<Count,Address><<<2,64>>>(reinterpret_cast<unsigned *>(device));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned char> actual(4096), expected(4096,0xac);
  ASSERT_EQ(cudaMemcpy(actual.data(),device,actual.size(),cudaMemcpyDeviceToHost),cudaSuccess);
  // Independent inverse mapping: reconstruct each logical matrix column/row.
  for (unsigned cta=0; cta<2; ++cta)
    for(unsigned warp=0; warp<2; ++warp)
      for(unsigned m=0; m<Count; ++m)
        for(unsigned col=0; col<8; ++col)
          for(unsigned row=0; row<16; ++row)
            expected[(cta*2+warp)*1024+m*256+(col*3%8)*32+row] =
                payload(cta,warp,m,(row%8)*4+col/2,(row/8)*2+col%2);
  for(unsigned i=0;i<actual.size();++i) ASSERT_EQ(actual[i],expected[i]) << "byte=" << i;
  EXPECT_EQ(cudaFree(device),cudaSuccess);
}
TEST(ByteMatrixLaunch, X1Shared32) { check<1,0>(); }
TEST(ByteMatrixLaunch, X2Shared32) { check<2,0>(); }
TEST(ByteMatrixLaunch, X4Shared32) { check<4,0>(); }
TEST(ByteMatrixLaunch, X1Shared64) { check<1,1>(); }
TEST(ByteMatrixLaunch, X2Shared64) { check<2,1>(); }
TEST(ByteMatrixLaunch, X4Shared64) { check<4,1>(); }
TEST(ByteMatrixLaunch, X1Generic64) { check<1,2>(); }
TEST(ByteMatrixLaunch, X2Generic64) { check<2,2>(); }
TEST(ByteMatrixLaunch, X4Generic64) { check<4,2>(); }
} // namespace
