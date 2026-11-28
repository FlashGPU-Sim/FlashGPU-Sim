#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
namespace {
template <bool System>
__global__ void atomic64_return(unsigned long long *counter,
                                unsigned long long *old) {
  unsigned long long value;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  // Keep the increment runtime-dependent so ptxas cannot replace the test
  // with an aggregated warp operation and unrelated uniform instructions.
  const unsigned long long increment =
      static_cast<volatile unsigned long long *>(old)[index];
  if constexpr (System)
    asm volatile("atom.relaxed.sys.global.add.u64 %0,[%1],%2;"
                 : "=l"(value)
                 : "l"(counter), "l"(increment)
                 : "memory");
  else
    asm volatile("atom.relaxed.gpu.global.add.u64 %0,[%1],%2;"
                 : "=l"(value)
                 : "l"(counter), "l"(increment)
                 : "memory");
  old[index] = value;
}
template <bool System>
void check_atomic64() {
  unsigned long long *counter, *old;
  ASSERT_EQ(cudaMalloc(&counter, 24), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&old, 128 * 8), cudaSuccess);
  std::array<unsigned long long, 3> values{
      0x123456789abcdef0ull, 0xfffffff000000000ull, 0xabcdef1234567890ull};
  const auto initial = values;
  std::vector<unsigned long long> increments(128, 0x100000003ull);
  ASSERT_EQ(cudaMemcpy(old, increments.data(), 128 * 8, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(counter, values.data(), 24, cudaMemcpyHostToDevice),
            cudaSuccess);
  atomic64_return<System><<<4, 32>>>(counter + 1, old);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(values.data(), counter, 24, cudaMemcpyDeviceToHost),
            cudaSuccess);
  std::vector<unsigned long long> returned(128), expected(128);
  ASSERT_EQ(cudaMemcpy(returned.data(), old, 128 * 8, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 128; ++i)
    expected[i] = initial[1] + i * 0x100000003ull;
  std::sort(returned.begin(), returned.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(returned, expected);
  EXPECT_EQ(values[0], initial[0]);
  EXPECT_EQ(values[1], initial[1] + 128 * 0x100000003ull);
  EXPECT_EQ(values[2], initial[2]);
  EXPECT_EQ(cudaFree(counter), cudaSuccess);
  EXPECT_EQ(cudaFree(old), cudaSuccess);
}
TEST(GlobalReduce, Atomic64GpuReturnedValues) { check_atomic64<false>(); }
TEST(GlobalReduce, Atomic64SysReturnedValues) { check_atomic64<true>(); }

template <unsigned Mode>
__global__ void reduce(unsigned long long *wide, unsigned *narrow) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (Mode == 0) {
    const unsigned value = i + 1;
    asm volatile("red.global.add.u32 [%0],%1;" ::"l"(narrow), "r"(value)
                 : "memory");
  } else if constexpr (Mode == 1) {
    const unsigned long long value = (1ull << 33) + i + 1;
    asm volatile("red.global.add.u64 [%0],%1;" ::"l"(wide), "l"(value)
                 : "memory");
  } else if constexpr (Mode == 2) {
    const unsigned long long value = 1ull << (i % 64);
    if (i % 3)
      asm volatile("red.global.xor.b64 [%0],%1;" ::"l"(wide), "l"(value)
                   : "memory");
  } else if (threadIdx.x == 0) {
    // Exactly one publishing thread per CTA, as used by MegaMHC.
    if constexpr (Mode == 3)
      asm volatile("red.async.release.gpu.global.add.u64 [%0],1;" ::"l"(wide)
                   : "memory");
    if constexpr (Mode == 4)
      asm volatile("red.async.release.sys.global.add.u64 [%0],1;" ::"l"(wide)
                   : "memory");
    if constexpr (Mode == 5)
      asm volatile("red.async.release.gpu.global.add.u32 [%0],1;" ::"l"(narrow)
                   : "memory");
    if constexpr (Mode == 6)
      asm volatile("red.async.release.sys.global.add.s32 [%0],-1;" ::"l"(narrow)
                   : "memory");
  }
}
template <unsigned Mode>
void check() {
  unsigned long long *wide;
  unsigned *narrow;
  ASSERT_EQ(cudaMalloc(&wide, 3 * sizeof(*wide)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&narrow, 3 * sizeof(*narrow)), cudaSuccess);
  std::array<unsigned long long, 3> w{
      0x123456789abcdef0ull, 0xfffffffffffffff0ull, 0xfedcba9876543210ull};
  std::array<unsigned, 3> n{0x12345678, 0xfffffff0, 0x87654321};
  auto ew = w;
  auto en = n;
  ASSERT_EQ(cudaMemcpy(wide, w.data(), sizeof(w), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(narrow, n.data(), sizeof(n), cudaMemcpyHostToDevice),
            cudaSuccess);
  reduce<Mode><<<4, 32>>>(wide + 1, narrow + 1);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(w.data(), wide, sizeof(w), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(n.data(), narrow, sizeof(n), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 128; ++i) {
    if constexpr (Mode == 0) en[1] += i + 1;
    if constexpr (Mode == 1) ew[1] += (1ull << 33) + i + 1;
    if constexpr (Mode == 2)
      if (i % 3) ew[1] ^= 1ull << (i % 64);
  }
  if constexpr (Mode == 3 || Mode == 4) ew[1] += 4;
  if constexpr (Mode == 5) en[1] += 4;
  if constexpr (Mode == 6) en[1] -= 4;
  EXPECT_EQ(w, ew);
  EXPECT_EQ(n, en);
  EXPECT_EQ(cudaFree(wide), cudaSuccess);
  EXPECT_EQ(cudaFree(narrow), cudaSuccess);
}
TEST(GlobalReduce, Add32) { check<0>(); }
TEST(GlobalReduce, Add64) { check<1>(); }
TEST(GlobalReduce, Xor64Predicated) { check<2>(); }
TEST(GlobalReduce, AsyncGpu64) { check<3>(); }
TEST(GlobalReduce, AsyncSys64) { check<4>(); }
TEST(GlobalReduce, AsyncGpu32) { check<5>(); }
TEST(GlobalReduce, AsyncSysSigned32) { check<6>(); }
}  // namespace
namespace {
__device__ __noinline__ unsigned long long publish_then_overwrite(
    unsigned long long *counter, unsigned long long *elapsed) {
  unsigned long long result, begin, end;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin)::"memory");
  asm volatile(
      "{.reg .u64 value; mov.u64 value,7; red.async.release.gpu.global.add.u64 "
      "[%1],value; mov.u64 value,99; mov.u64 %0,value;}"
      : "=l"(result)
      : "l"(counter)
      : "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
  *elapsed = end - begin;
  return result;
}
__global__ void async_return(unsigned long long *counter,
                             unsigned long long *out) {
  if (threadIdx.x == 0) out[0] = publish_then_overwrite(counter, out + 1);
}
TEST(GlobalReduce, AsyncCapturesOperandsAcrossReturnAndExit) {
  unsigned long long *counter, *out;
  ASSERT_EQ(cudaMalloc(&counter, 8), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 16), cudaSuccess);
  ASSERT_EQ(cudaMemset(counter, 0, 8), cudaSuccess);
  async_return<<<1, 32>>>(counter, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned long long count, values[2];
  ASSERT_EQ(cudaMemcpy(&count, counter, 8, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(values, out, 16, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(count, 7u);
  EXPECT_EQ(values[0], 99u);
  // With the pinned SM100 config, even an L2 response takes 260 cycles.
  // Continuing before that distinguishes nonblocking issue from atomic waiting.
  EXPECT_LT(values[1], 200u) << "async reduction blocked the issuing warp";
  EXPECT_EQ(cudaFree(counter), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
#include <fstream>
#include <sstream>
namespace {
__global__ void ordered_reduce(unsigned long long *counter,
                               unsigned long long *out) {
  __shared__ unsigned value;
  if (threadIdx.x == 0) {
    unsigned long long begin, end;
    const unsigned at = __cvta_generic_to_shared(&value);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin)::"memory");
    asm volatile("st.shared.u32 [%0],42;" ::"r"(at) : "memory");
    asm volatile("red.async.release.gpu.global.add.u64 [%0],1;" ::"l"(counter)
                 : "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");
    out[0] = end - begin;
    out[1] = value;
  }
}
TEST(GlobalReduce, ReleaseWaitsForPriorSharedStore) {
  unsigned delay = 0;
  std::ifstream config("gpgpusim.config");
  for (std::string line; std::getline(config, line);) {
    std::istringstream fields(line);
    std::string name;
    fields >> name;
    if (name == "-gpgpu_smem_store_visibility_latency") fields >> delay;
  }
  ASSERT_GE(delay, 512u)
      << "requires the global-reductions CI visibility config";
  unsigned long long *counter, *out;
  ASSERT_EQ(cudaMalloc(&counter, 8), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 16), cudaSuccess);
  ASSERT_EQ(cudaMemset(counter, 0, 8), cudaSuccess);
  ordered_reduce<<<1, 32>>>(counter, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned long long count, values[2];
  ASSERT_EQ(cudaMemcpy(&count, counter, 8, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(values, out, 16, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(values[1], 42u);
  EXPECT_GE(values[0], delay);
  EXPECT_EQ(cudaFree(counter), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
