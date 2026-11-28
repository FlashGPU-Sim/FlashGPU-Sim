#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

template <unsigned Barrier>
__device__ unsigned reduce_group(unsigned local, unsigned limit) {
  unsigned count, all, any;
  asm volatile(
      "{ .reg .pred p, a, o;\n"
      "setp.lt.u32 p, %3, %4;\n"
      "bar.red.popc.u32 %0, %5, 64, !p;\n"
      "bar.red.and.pred a, %5, 64, !p;\n"
      "bar.red.or.pred o, %5, 64, !p;\n"
      "selp.u32 %1, 1, 0, a;\n"
      "selp.u32 %2, 1, 0, o;\n"
      "}"
      : "=r"(count), "=r"(all), "=r"(any)
      : "r"(local), "r"(limit), "n"(Barrier) : "memory");
  return count + 128 * all + 256 * any;
}

__global__ void cta_group_reduce(unsigned *out) {
  const unsigned group = threadIdx.x / 64;
  for (unsigned round = 0; round < 4; ++round) {
    const unsigned limit = round == 0 ? 0 : round == 1 ? 1 : round == 2 ? 64
        : 3 + blockIdx.x + group;
    const unsigned local = threadIdx.x % 64;
    const unsigned value = group == 0 ? reduce_group<1>(local, limit)
                                      : reduce_group<2>(local, limit);
    out[(blockIdx.x * 4 + round) * 128 + threadIdx.x] = value;
  }
}

TEST(CtaReduction, IndependentExplicitCountGroupsAndNegation) {
  unsigned *dst = nullptr;
  std::vector<unsigned> out(3 * 4 * 128);
  ASSERT_EQ(cudaMalloc(&dst, out.size() * sizeof(unsigned)), cudaSuccess);
  cta_group_reduce<<<3, 128>>>(dst);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dst, out.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < out.size(); ++i) {
    const unsigned round = (i / 128) % 4;
    const unsigned block = i / (128 * 4), group = (i % 128) / 64;
    const unsigned limit = round == 0 ? 0 : round == 1 ? 1 : round == 2 ? 64
        : 3 + block + group;
    const unsigned expected = 64 - limit + 128 * (limit == 0) + 256 * (limit < 64);
    EXPECT_EQ(out[i], expected) << "output index " << i;
  }
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}

__global__ void cta_implicit_negated_reduce(unsigned *out) {
  const unsigned limit = blockIdx.x == 0 ? 0 : blockIdx.x == 1 ? 64 : 7;
  unsigned count, all, any;
  asm volatile(
      "{ .reg .pred p, a, o;\n"
      "setp.lt.u32 p, %3, %4;\n"
      "bar.red.popc.u32 %0, 0, !p;\n"
      "bar.red.and.pred a, 0, !p;\n"
      "bar.red.or.pred o, 0, !p;\n"
      "selp.u32 %1, 1, 0, a;\n"
      "selp.u32 %2, 1, 0, o;\n"
      "}"
      : "=r"(count), "=r"(all), "=r"(any)
      : "r"(threadIdx.x), "r"(limit) : "memory");
  out[blockIdx.x * 64 + threadIdx.x] = count + 128 * all + 256 * any;
}

TEST(CtaReduction, ImplicitCountNegatedPredicate) {
  unsigned *dst = nullptr;
  std::vector<unsigned> out(3 * 64);
  ASSERT_EQ(cudaMalloc(&dst, out.size() * sizeof(unsigned)), cudaSuccess);
  cta_implicit_negated_reduce<<<3, 64>>>(dst);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out.data(), dst, out.size() * sizeof(unsigned),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < out.size(); ++i)
    EXPECT_EQ(out[i], i < 64 ? 448u : i < 128 ? 0u : 313u) << i;
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
}
