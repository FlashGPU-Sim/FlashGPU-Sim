#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
template <unsigned Mode>
__global__ void shuffle_alias(const unsigned *input, unsigned *out) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned lane = threadIdx.x & 31, warp = tid / 32;
  const unsigned masks[] = {0, 0x1800, 0x1c00, 0x1000};
  const unsigned clamp = Mode == 0 ? 0 : 31;
  const unsigned config = masks[warp & 3] | clamp;
  const unsigned delta = (warp / 4) & 31;
  unsigned r = input[tid], p;
#define SHUFFLE(M)                                                              \
  asm volatile("{.reg .pred p; shfl.sync." M ".b32 %0|p,%0,%2,%3,0xffffffff; " \
               "selp.u32 %1,1,0,p;}"                                         \
               : "+r"(r), "=r"(p) : "r"(delta), "r"(config));
  if constexpr (Mode == 0) { SHUFFLE("up") }
  if constexpr (Mode == 1) { SHUFFLE("down") }
  if constexpr (Mode == 2) { SHUFFLE("bfly") }
  if constexpr (Mode == 3) { SHUFFLE("idx") }
#undef SHUFFLE
  out[tid * 2] = r;
  out[tid * 2 + 1] = p;
}
template <unsigned Mode> void check_shuffle() {
  constexpr unsigned count = 4096;
  std::vector<unsigned> input(count), output(count * 2);
  for (unsigned i = 0; i < count; ++i) input[i] = 0x54321000u + 37 * i;
  unsigned *in, *out;
  ASSERT_EQ(cudaMalloc(&in, count * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, count * 8), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), count * 4, cudaMemcpyHostToDevice), cudaSuccess);
  shuffle_alias<Mode><<<16, 256>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), out, count * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  const int widths[] = {32, 8, 4, 16};
  for (unsigned tid = 0; tid < count; ++tid) {
    const int lane = tid % 32, warp = tid / 32;
    const int width = widths[warp % 4], start = lane / width * width;
    const int delta = (warp / 4) % 32;
    const int source = Mode == 0 ? lane - delta : Mode == 1 ? lane + delta :
                       Mode == 2 ? lane ^ delta : start + delta % width;
    const bool valid = Mode == 0 ? source >= start : source < start + width;
    const unsigned expected = input[tid - lane + (valid ? source : lane)];
    ASSERT_EQ(output[tid * 2], expected) << "tid=" << tid;
    ASSERT_EQ(output[tid * 2 + 1], unsigned(valid)) << "tid=" << tid;
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
TEST(WarpShuffle, UpAliasedSourceAndSegments) { check_shuffle<0>(); }
TEST(WarpShuffle, DownAliasedSourceAndSegments) { check_shuffle<1>(); }
TEST(WarpShuffle, ButterflyAliasedSourceAndSegments) { check_shuffle<2>(); }
TEST(WarpShuffle, IndexAliasedSourceAndSegments) { check_shuffle<3>(); }

__global__ void shuffle_predicated_loop(const unsigned *in, unsigned *out) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned r = in[tid], p = 0, odd = threadIdx.x & 1;
#pragma unroll 1
  for (unsigned i = 0; i < 3; ++i) {
    asm volatile("{.reg .pred p; setp.ne.u32 p,%2,0; "
                 "@p shfl.sync.up.b32 %0|p,%0,2,0,0xaaaaaaaa; "
                 "selp.u32 %1,1,0,p;}"
                 : "+r"(r), "=r"(p) : "r"(odd));
    r += i + 1;
  }
  out[tid * 2] = r;
  out[tid * 2 + 1] = p;
}
TEST(WarpShuffle, PredicatedAliasingAndRepeatedInstruction) {
  constexpr unsigned count = 256;
  std::vector<unsigned> input(count), result(count * 2), expected(count);
  for (unsigned i = 0; i < count; ++i) input[i] = expected[i] = 37 * i;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    const auto prior = expected;
    for (unsigned i = 0; i < count; ++i) {
      const unsigned lane = i % 32;
      expected[i] = prior[i - (lane > 1 && (lane & 1) ? 2 : 0)] + iteration + 1;
    }
  }
  unsigned *in, *out;
  ASSERT_EQ(cudaMalloc(&in, count * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, count * 8), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), count * 4, cudaMemcpyHostToDevice), cudaSuccess);
  shuffle_predicated_loop<<<2, 128>>>(in, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(result.data(), out, count * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < count; ++i) {
    EXPECT_EQ(result[2 * i], expected[i]) << i;
    EXPECT_EQ(result[2 * i + 1], unsigned((i & 1) && (i % 32 > 1))) << i;
  }
  EXPECT_EQ(cudaFree(in), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
}
}  // namespace
