#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

__global__ void unpack_with_discarded_components(uint16_t *output,
                                                 uint64_t packed) {
  uint16_t word0 = 0;
  uint16_t word2 = 0;
  asm volatile("mov.b64 {%0, _, %1, _}, %2;"
               : "=h"(word0), "=h"(word2)
               : "l"(packed));
  output[0] = word0;
  output[1] = word2;
}

__global__ void bit_type_predicate_width(uint32_t *output) {
  uint32_t branch_iterations = 0;
  const uint32_t minus_four = 0xfffffffcu;
  asm volatile(
      "{\n\t"
      ".reg .b32 counter;\n\t"
      ".reg .b32 guard;\n\t"
      ".reg .pred keep_looping;\n\t"
      ".reg .pred under_limit;\n\t"
      "mov.b32 counter, %1;\n\t"
      "mov.u32 guard, 0;\n\t"
      "bit_width_loop_%=:\n\t"
      "add.s32 counter, counter, 4;\n\t"
      "add.u32 guard, guard, 1;\n\t"
      "setp.ne.b32 keep_looping, counter, 0;\n\t"
      "@!keep_looping bra bit_width_done_%=;\n\t"
      "setp.lt.u32 under_limit, guard, 3;\n\t"
      "@under_limit bra bit_width_loop_%=;\n\t"
      "bit_width_done_%=:\n\t"
      "mov.u32 %0, guard;\n\t"
      "}\n"
      : "=r"(branch_iterations)
      : "r"(minus_four));

  uint16_t predicate16 = 0;
  const uint16_t minus_one = 0xffffu;
  asm volatile(
      "{\n\t"
      ".reg .b16 sum;\n\t"
      ".reg .pred nonzero;\n\t"
      "add.s16 sum, %1, 1;\n\t"
      "setp.ne.b16 nonzero, sum, 0;\n\t"
      "selp.u16 %0, 1, 0, nonzero;\n\t"
      "}\n"
      : "=h"(predicate16)
      : "h"(minus_one));

  output[0] = branch_iterations;
  output[1] = predicate16;
}

TEST(VectorOperandIntegrationTest, DiscardsPlaceholderComponents) {
  constexpr uint64_t kPacked = 0x1122334455667788ULL;
  constexpr uint16_t kExpected[] = {0x7788, 0x3344};

  uint16_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  unpack_with_discarded_components<<<1, 1>>>(output, kPacked);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint16_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], kExpected[0]);
  EXPECT_EQ(actual[1], kExpected[1]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(BitTypePredicateIntegrationTest, ComparesOnlyTheDeclaredBitWidth) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 2 * sizeof(*output)), cudaSuccess);

  bit_type_predicate_width<<<1, 1>>>(output);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], 1u);
  EXPECT_EQ(actual[1], 0u);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

}  // namespace
