#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>

namespace {
template <int Form>
__global__ void bulk_zero(unsigned *out, const unsigned *counts) {
  __shared__ __align__(16) unsigned data[256];
  for (unsigned phase = 0; phase < 4; ++phase) {
    for (unsigned i = threadIdx.x; i < 256; i += 32)
      data[i] = 0x12340000 + blockIdx.x * 1024 + phase * 256 + i;
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned address = __cvta_generic_to_shared(data + 2);
      unsigned bytes = counts[phase];
      if constexpr (Form == 0)
        asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;" ::
                     "r"(address), "r"(bytes) : "memory");
      if constexpr (Form == 1)
        asm volatile("st.bulk.shared::cta [%0], %1, 0;" ::
                     "r"(address), "l"((unsigned long long)bytes) : "memory");
      if constexpr (Form == 2)
        asm volatile("st.bulk.weak [%0], %1, 0;" ::
                     "l"(data + 2), "r"(bytes) : "memory");
      if constexpr (Form == 3)
        asm volatile("st.bulk [%0], %1, 0;" ::
                     "l"(data + 2), "l"((unsigned long long)bytes) : "memory");
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 256; i += 32)
      out[(blockIdx.x * 4 + phase) * 256 + i] = data[i];
    __syncthreads();
  }
}
template <int Form> void check_bulk() {
  unsigned *out, *counts;
  const std::array<unsigned, 4> sizes{0, 8, 264, 1008};
  std::array<unsigned, 2048> result{};
  ASSERT_EQ(cudaMalloc(&out, sizeof(result)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&counts, sizeof(sizes)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(counts, sizes.data(), sizeof(sizes), cudaMemcpyHostToDevice), cudaSuccess);
  bulk_zero<Form><<<2, 32>>>(out, counts);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), out, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned cta = 0; cta < 2; ++cta)
    for (unsigned phase = 0; phase < 4; ++phase)
      for (unsigned i = 0; i < 256; ++i) {
        unsigned expected = i >= 2 && i < 2 + sizes[phase] / 4 ? 0 :
                            0x12340000 + cta * 1024 + phase * 256 + i;
        ASSERT_EQ(result[(cta * 4 + phase) * 256 + i], expected)
            << cta << '/' << phase << '/' << i;
      }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
  EXPECT_EQ(cudaFree(counts), cudaSuccess);
}
TEST(SharedBulkLaunch, WeakShared32) { check_bulk<0>(); }
TEST(SharedBulkLaunch, Shared64) { check_bulk<1>(); }
TEST(SharedBulkLaunch, WeakGeneric32) { check_bulk<2>(); }
TEST(SharedBulkLaunch, Generic64) { check_bulk<3>(); }

template <int Form>
__global__ void hinted_load(const unsigned long long *input, unsigned long long *out) {
  const unsigned i = (blockIdx.x * 32 + threadIdx.x) * 4;
  unsigned long long a,b,c,d;
#define LOAD_HINT(S) asm volatile(S " {%0,%1,%2,%3}, [%4];" \
  : "=l"(a), "=l"(b), "=l"(c), "=l"(d) : "l"(input + i) : "memory")
  if constexpr (Form == 0) { LOAD_HINT("ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v4.u64"); }
  if constexpr (Form == 1) { LOAD_HINT("ld.global.L1::no_allocate.L2::64B.v4.u64"); }
  if constexpr (Form == 2) { LOAD_HINT("ld.global.cg.L2::128B.v4.u64"); }
#undef LOAD_HINT
  out[i]=a; out[i+1]=b; out[i+2]=c; out[i+3]=d;
}
template<int Form> void check_load() {
  std::array<unsigned long long,256> data{}, result{};
  for (unsigned i=0;i<data.size();++i) data[i]=0x9876543212340000ULL+i*12345;
  unsigned long long *input, *out;
  ASSERT_EQ(cudaMalloc(&input,sizeof(data)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out,sizeof(data)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(input,data.data(),sizeof(data),cudaMemcpyHostToDevice),cudaSuccess);
  hinted_load<Form><<<2,32>>>(input,out);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(),out,sizeof(result),cudaMemcpyDeviceToHost),cudaSuccess);
  EXPECT_EQ(result,data);
  EXPECT_EQ(cudaFree(input),cudaSuccess);
  EXPECT_EQ(cudaFree(out),cudaSuccess);
}
TEST(CacheHintLaunch, NoncoherentNoAllocate256) { check_load<0>(); }
TEST(CacheHintLaunch, CacheAllNoAllocate64) { check_load<1>(); }
TEST(CacheHintLaunch, CacheGlobal128) { check_load<2>(); }
} // namespace

namespace {
__global__ void load128_store8(const unsigned *input, unsigned *out,
                              unsigned char *bytes) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned a,b,c,d;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(input + i*4) : "memory");
  asm volatile("st.global.v4.u32 [%0],{%1,%2,%3,%4};" :: "l"(out+i*4),
               "r"(a),"r"(b),"r"(c),"r"(d) : "memory");
  if ((i & 1) == 0)
    asm volatile("st.global.u8 [%0],%1;" :: "l"(bytes+i*2+1),"r"(a) : "memory");
}
TEST(CacheHintLaunch, NoAllocate128AndPredicatedByteStore) {
  std::array<unsigned,256> input{},actual{};
  for(unsigned i=0;i<input.size();++i) input[i]=0xabc00000u+i*37+11;
  unsigned *in,*out;
  unsigned char *bytes;
  ASSERT_EQ(cudaMalloc(&in,sizeof(input)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out,sizeof(actual)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&bytes,128),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in,input.data(),sizeof(input),cudaMemcpyHostToDevice),cudaSuccess);
  ASSERT_EQ(cudaMemset(bytes,0xa5,128),cudaSuccess);
  load128_store8<<<2,32>>>(in,out,bytes);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(),out,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  EXPECT_EQ(actual,input);
  std::array<unsigned char,128> result;
  ASSERT_EQ(cudaMemcpy(result.data(),bytes,result.size(),cudaMemcpyDeviceToHost),cudaSuccess);
  for(unsigned i=0;i<64;++i) {
    EXPECT_EQ(result[i*2],0xa5);
    EXPECT_EQ(result[i*2+1],i%2==0 ? (input[i*4]&255) : 0xa5);
  }
  EXPECT_EQ(cudaFree(in),cudaSuccess);
  EXPECT_EQ(cudaFree(out),cudaSuccess);
  EXPECT_EQ(cudaFree(bytes),cudaSuccess);
}
}
