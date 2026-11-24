#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
struct Input {
  unsigned mask, base;
  int offset;
};

__global__ void dynamic_search(const Input *in, unsigned *out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned result;
  asm volatile("fns.b32 %0,%1,%2,%3;"
               : "=r"(result)
               : "r"(in[i].mask), "r"(in[i].base), "r"(in[i].offset));
  out[i] = result;
}

// Independent reference: enumerate positions, then index the collected list.
unsigned reference(const Input &v) {
  std::vector<unsigned> positions;
  for (unsigned i = 0; i < 32; ++i)
    if ((v.mask >> i) & 1u) positions.push_back(i);
  if (!v.offset) return ((v.mask >> v.base) & 1u) ? v.base : UINT32_MAX;
  std::vector<unsigned> eligible;
  if (v.offset > 0) {
    for (auto i : positions)
      if (i >= v.base) eligible.push_back(i);
  } else {
    for (auto p = positions.rbegin(); p != positions.rend(); ++p)
      if (*p <= v.base) eligible.push_back(*p);
  }
  const uint64_t rank = v.offset < 0 ? -int64_t(v.offset) : int64_t(v.offset);
  return rank <= eligible.size() ? eligible[rank - 1] : UINT32_MAX;
}

void check(const std::vector<Input> &input) {
  ASSERT_EQ(input.size() % 128, 0u);
  Input *in = nullptr;
  unsigned *out = nullptr;
  ASSERT_EQ(cudaMalloc(&in, input.size() * sizeof(Input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, input.size() * sizeof(unsigned)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), input.size() * sizeof(Input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  dynamic_search<<<input.size() / 128, 128>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> actual(input.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), out, input.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (size_t i = 0; i < input.size(); ++i)
    EXPECT_EQ(actual[i], reference(input[i]))
        << "i=" << i << " mask=" << input[i].mask << " base=" << input[i].base
        << " offset=" << input[i].offset;
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}

TEST(BitSearchLaunch, DynamicMixedDirectionsAndBoundaries) {
  std::vector<Input> input;
  for (unsigned mask : {0u, UINT32_MAX, 0xaaaaaaaau, 0x55555555u, 1u,
                        0x80000000u, 0x81234567u, 0x00ff0000u})
    for (unsigned base = 0; base < 32; ++base)
      for (int offset = -33; offset <= 33; ++offset)
        input.push_back({mask, base, offset});
  check(input);
}
TEST(BitSearchLaunch, ExtremeSignedOffsets) {
  std::vector<Input> input;
  for (unsigned mask : {0u, UINT32_MAX, 0x80000001u, 0xabcdef01u})
    for (unsigned base = 0; base < 32; ++base)
      for (int offset : {-std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(), -1024, 1024})
        input.push_back({mask, base, offset});
  check(input);
}

// CUDA 13.3 lowers INT_MIN to a sequence whose signed abs overflows. Keep
// this ISA boundary checked on PTX; see docs/features/blackwell-bit-search.md.
TEST(BitSearchPtxOnly, ExtremeNegativeOffset) {
  std::vector<Input> input;
  for (unsigned mask : {0u, UINT32_MAX, 0x80000001u, 0xabcdef01u})
    for (unsigned base = 0; base < 32; ++base)
      input.push_back({mask, base, std::numeric_limits<int>::min()});
  check(input);
}

__global__ void immediate_predicated_search(unsigned *out) {
  unsigned value = 0xaaaaaaaau, tid = threadIdx.x;
  asm volatile("{ .reg .pred p; setp.ne.u32 p,%1,0; @p fns.b32 %0,%0,3,-1; }"
               : "+r"(value)
               : "r"(tid));
  out[tid * 3] = value;
  asm volatile("fns.b32 %0,0xaaaaaaaa,2,1;" : "=r"(value));
  out[tid * 3 + 1] = value;
  asm volatile("fns.b32 %0,0xaaaaaaaa,2,0;" : "=r"(value));
  out[tid * 3 + 2] = value;
}
TEST(BitSearchLaunch, ImmediatePredicationAndAlias) {
  unsigned *out;
  ASSERT_EQ(cudaMalloc(&out, 128 * 3 * sizeof(unsigned)), cudaSuccess);
  immediate_predicated_search<<<1, 128>>>(out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  unsigned actual[128 * 3];
  ASSERT_EQ(cudaMemcpy(actual, out, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 128; ++i) {
    EXPECT_EQ(actual[3 * i], i ? 3u : 0xaaaaaaaau);
    EXPECT_EQ(actual[3 * i + 1], 3u);
    EXPECT_EQ(actual[3 * i + 2], UINT32_MAX);
  }
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
