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

__global__ void consume_compiler_pack_alias_chain(uint64_t *output,
                                                  uint32_t low,
                                                  uint32_t high) {
  uint64_t packed = 0;
  uint32_t unpacked_low = 0;
  uint32_t unpacked_high = 0;
  uint64_t incremented = 0;
  uint64_t repacked = 0;
  asm volatile("mov.b64 %0, {%1, %2};"
               : "=l"(packed)
               : "r"(low), "r"(high));
  asm volatile("mov.b64 {%0, %1}, %2;"
               : "=r"(unpacked_low), "=r"(unpacked_high)
               : "l"(packed));
  asm volatile("add.u64 %0, %1, 1;" : "=l"(incremented) : "l"(packed));
  asm volatile("mov.b64 %0, {%1, %2};"
               : "=l"(repacked)
               : "r"(unpacked_low), "r"(unpacked_high));
  output[0] = repacked;
  output[1] = incremented;
}

__global__ void consume_compiler_pack_across_barrier(uint64_t *output,
                                                     uint32_t low,
                                                     uint32_t high) {
  uint32_t source_low = 0;
  uint32_t source_high = 0;
  asm volatile("mov.u32 %0, %2;\n\t"
               "mov.u32 %1, %3;"
               : "=r"(source_low), "=r"(source_high)
               : "r"(low), "r"(high));
  __syncthreads();

  uint64_t packed = 0;
  uint64_t incremented = 0;
  asm volatile("mov.b64 %0, {%1, %2};"
               : "=l"(packed)
               : "r"(source_low), "r"(source_high));
  asm volatile("add.u64 %0, %1, 1;" : "=l"(incremented) : "l"(packed));
  output[threadIdx.x] = incremented;
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

__global__ void compiler_shift_add_mad(uint32_t *output, uint32_t x,
                                       uint32_t fraction) {
  uint32_t result = 0;
  asm volatile(
      "{\n\t"
      ".reg .s32 copied_x, copied_fraction, shifted_x, combined;\n\t"
      "mov.b32 copied_x, %1;\n\t"
      "mov.b32 copied_fraction, %2;\n\t"
      "shl.b32 shifted_x, copied_x, 23;\n\t"
      "add.s32 combined, shifted_x, copied_fraction;\n\t"
      "mov.b32 %0, combined;\n\t"
      "}\n"
      : "=r"(result)
      : "r"(x), "r"(fraction));
  output[0] = result;
}

__global__ void compiler_sign_bit_shift_add_mad(uint32_t *output, uint32_t x,
                                                uint32_t fraction) {
  uint32_t result = 0;
  asm volatile(
      "{\n\t"
      ".reg .s32 copied_x, copied_fraction, shifted_x, combined;\n\t"
      "mov.b32 copied_x, %1;\n\t"
      "mov.b32 copied_fraction, %2;\n\t"
      "shl.b32 shifted_x, copied_x, 31;\n\t"
      "add.s32 combined, shifted_x, copied_fraction;\n\t"
      "mov.b32 %0, combined;\n\t"
      "}\n"
      : "=r"(result)
      : "r"(x), "r"(fraction));
  output[0] = result;
}

__global__ void compiler_shift_add_with_extra_use(uint32_t *output,
                                                  uint32_t x,
                                                  uint32_t fraction) {
  uint32_t result = 0;
  uint32_t copied = 0;
  asm volatile(
      "{\n\t"
      ".reg .s32 copied_x, copied_fraction, shifted_x, combined;\n\t"
      "mov.b32 copied_x, %2;\n\t"
      "mov.b32 copied_fraction, %3;\n\t"
      "shl.b32 shifted_x, copied_x, 23;\n\t"
      "add.s32 combined, shifted_x, copied_fraction;\n\t"
      "mov.b32 %0, combined;\n\t"
      "mov.b32 %1, copied_x;\n\t"
      "}\n"
      : "=r"(result), "=r"(copied)
      : "r"(x), "r"(fraction));
  output[0] = result;
  output[1] = copied;
}

__global__ void compiler_scalar_copy(uint32_t *output, uint32_t input) {
  uint32_t result = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 alias;\n\t"
      "mov.b32 alias, %1;\n\t"
      "add.u32 %0, alias, 7;\n\t"
      "}\n"
      : "=r"(result)
      : "r"(input));
  output[0] = result;
}

__global__ void compiler_scalar_copy_with_source_redefinition(
    uint32_t *output, uint32_t input) {
  uint32_t result = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 source, alias;\n\t"
      "mov.b32 source, %1;\n\t"
      "mov.b32 alias, source;\n\t"
      "add.u32 source, source, 1;\n\t"
      "add.u32 %0, alias, source;\n\t"
      "}\n"
      : "=r"(result)
      : "r"(input));
  output[0] = result;
}

__global__ void compiler_packed_f32x2_literal(uint64_t *output,
                                              uint64_t packed_two) {
  uint64_t added = 0;
  uint64_t subtracted = 0;
  uint64_t fused = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n\t"
      ".reg .b64 packed_one, packed_two_literal;\n\t"
      "mov.b64 packed_one, 4575657222473777152;\n\t"
      "add.rm.f32x2 %0, %3, packed_one;\n\t"
      "sub.rn.f32x2 %1, %3, packed_one;\n\t"
      "mov.b64 packed_two_literal, 4611686019501129728;\n\t"
      "fma.rn.f32x2 %2, packed_one, packed_two_literal, packed_one;\n\t"
      "}\n"
      : "=l"(added), "=l"(subtracted), "=l"(fused)
      : "l"(packed_two));
#else
  added = 0x4040000040400000ULL;
  subtracted = 0x3f8000003f800000ULL;
  fused = 0x4040000040400000ULL;
#endif
  output[0] = added;
  output[1] = subtracted;
  output[2] = fused;
}

__global__ void compiler_packed_f32x2_literal_with_other_use(
    uint64_t *output, uint64_t packed_two) {
  uint64_t added = 0;
  uint64_t copied = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n\t"
      ".reg .b64 packed_one;\n\t"
      "mov.b64 packed_one, 4575657222473777152;\n\t"
      "add.rm.f32x2 %0, %2, packed_one;\n\t"
      "mov.b64 %1, packed_one;\n\t"
      "}\n"
      : "=l"(added), "=l"(copied)
      : "l"(packed_two));
#else
  added = 0x4040000040400000ULL;
  copied = 0x3f8000003f800000ULL;
#endif
  output[0] = added;
  output[1] = copied;
}

__global__ void compiler_broadcast_pack(uint64_t *output,
                                        uint32_t scalar_bits) {
  [[maybe_unused]] uint64_t broadcast = 0;
  uint64_t added = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n\t"
      "mov.b64 %0, {%2, %2};\n\t"
      "add.rn.f32x2 %1, %0, %0;\n\t"
      "}\n"
      : "=l"(broadcast), "=l"(added)
      : "r"(scalar_bits));
#else
  added = 0x4040000040400000ULL;
#endif
  output[0] = added;
}

__global__ void compiler_negated_multiply(float *output, float a, float b) {
  float result = 0.0f;
  asm volatile(
      "{\n\t"
      ".reg .f32 product, zero;\n\t"
      "mul.f32 product, %1, %2;\n\t"
      "mov.b32 zero, 0f00000000;\n\t"
      "sub.f32 %0, zero, product;\n\t"
      "}\n"
      : "=f"(result)
      : "f"(a), "f"(b));
  output[0] = result;
}

__global__ void compiler_negated_multiply_with_extra_use(float *output,
                                                         float a, float b) {
  float result = 0.0f;
  float product_copy = 0.0f;
  asm volatile(
      "{\n\t"
      ".reg .f32 product, zero;\n\t"
      "mul.f32 product, %2, %3;\n\t"
      "mov.b32 zero, 0f00000000;\n\t"
      "sub.f32 %0, zero, product;\n\t"
      "mov.b32 %1, product;\n\t"
      "}\n"
      : "=f"(result), "=f"(product_copy)
      : "f"(a), "f"(b));
  output[0] = result;
  output[1] = product_copy;
}

__global__ void compiler_predicate_not_guard(uint32_t *output,
                                             uint32_t input) {
  uint32_t guarded_by_inverted = 0x11111111u;
  uint32_t guarded_by_double_inverted = 0x33333333u;
  asm volatile(
      "{\n\t"
      ".reg .pred source0, inverted0, source1, inverted1;\n\t"
      "setp.ne.u32 source0, %2, 0;\n\t"
      "not.pred inverted0, source0;\n\t"
      "@inverted0 mov.u32 %0, 0x22222222;\n\t"
      "setp.ne.u32 source1, %2, 0;\n\t"
      "not.pred inverted1, source1;\n\t"
      "@!inverted1 mov.u32 %1, 0x44444444;\n\t"
      "}\n"
      : "+r"(guarded_by_inverted), "+r"(guarded_by_double_inverted)
      : "r"(input));
  output[0] = guarded_by_inverted;
  output[1] = guarded_by_double_inverted;
}

__global__ void compiler_predicate_byte_extract(uint32_t *output,
                                                uint32_t input) {
  uint32_t result0 = 0;
  uint32_t result1 = 0;
  uint32_t result2 = 0;
  uint32_t result3 = 0;
  uint32_t result4 = 0;
  uint32_t result5 = 0;
  uint32_t result6 = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 t0, t1, t2, t3, t4, t5, t6;\n\t"
      ".reg .pred p0, p1, p2, p3, p4, p5, p6;\n\t"
      "and.b32 t0, %7, 1;\n\t"
      "setp.ne.b32 p0, t0, 0;\n\t"
      "selp.b32 %0, 11, 101, p0;\n\t"
      "and.b32 t1, %7, 2;\n\t"
      "setp.eq.s32 p1, t1, 0;\n\t"
      "selp.b32 %1, 12, 102, p1;\n\t"
      "and.b32 t2, %7, 4;\n\t"
      "setp.eq.s32 p2, t2, 0;\n\t"
      "selp.b32 %2, 13, 103, p2;\n\t"
      "and.b32 t3, %7, 8;\n\t"
      "setp.eq.s32 p3, t3, 0;\n\t"
      "selp.b32 %3, 14, 104, p3;\n\t"
      "and.b32 t4, %7, 16;\n\t"
      "setp.eq.s32 p4, t4, 0;\n\t"
      "selp.b32 %4, 15, 105, p4;\n\t"
      "and.b32 t5, %7, 32;\n\t"
      "setp.eq.s32 p5, t5, 0;\n\t"
      "selp.b32 %5, 16, 106, p5;\n\t"
      "and.b32 t6, %7, 64;\n\t"
      "setp.eq.s32 p6, t6, 0;\n\t"
      "selp.b32 %6, 17, 107, p6;\n\t"
      "}\n"
      : "=r"(result0), "=r"(result1), "=r"(result2), "=r"(result3),
        "=r"(result4), "=r"(result5), "=r"(result6)
      : "r"(input));
  output[0] = result0;
  output[1] = result1;
  output[2] = result2;
  output[3] = result3;
  output[4] = result4;
  output[5] = result5;
  output[6] = result6;
}

__global__ void compiler_predicate_byte_extract_with_extra_temp_use(
    uint32_t *output, uint32_t input) {
  uint32_t result0 = 0;
  uint32_t result1 = 0;
  uint32_t result2 = 0;
  uint32_t result3 = 0;
  uint32_t result4 = 0;
  uint32_t result5 = 0;
  uint32_t result6 = 0;
  uint32_t copied_temp = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 t0, t1, t2, t3, t4, t5, t6;\n\t"
      ".reg .pred p0, p1, p2, p3, p4, p5, p6;\n\t"
      "and.b32 t0, %8, 1;\n\t"
      "setp.ne.b32 p0, t0, 0;\n\t"
      "selp.b32 %0, 11, 101, p0;\n\t"
      "and.b32 t1, %8, 2;\n\t"
      "setp.ne.b32 p1, t1, 0;\n\t"
      "selp.b32 %1, 12, 102, p1;\n\t"
      "and.b32 t2, %8, 4;\n\t"
      "setp.ne.b32 p2, t2, 0;\n\t"
      "selp.b32 %2, 13, 103, p2;\n\t"
      "and.b32 t3, %8, 8;\n\t"
      "setp.ne.b32 p3, t3, 0;\n\t"
      "selp.b32 %3, 14, 104, p3;\n\t"
      "and.b32 t4, %8, 16;\n\t"
      "setp.ne.b32 p4, t4, 0;\n\t"
      "selp.b32 %4, 15, 105, p4;\n\t"
      "and.b32 t5, %8, 32;\n\t"
      "setp.ne.b32 p5, t5, 0;\n\t"
      "selp.b32 %5, 16, 106, p5;\n\t"
      "and.b32 t6, %8, 64;\n\t"
      "setp.ne.b32 p6, t6, 0;\n\t"
      "selp.b32 %6, 17, 107, p6;\n\t"
      "mov.b32 %7, t0;\n\t"
      "}\n"
      : "=r"(result0), "=r"(result1), "=r"(result2), "=r"(result3),
        "=r"(result4), "=r"(result5), "=r"(result6), "=r"(copied_temp)
      : "r"(input));
  output[0] = result0;
  output[1] = result1;
  output[2] = result2;
  output[3] = result3;
  output[4] = result4;
  output[5] = result5;
  output[6] = result6;
  output[7] = copied_temp;
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

TEST(PtxReorderPackIntegrationTest, PreservesNestedCompilerAliases) {
  constexpr uint32_t kLow = 0xffffffffu;
  constexpr uint32_t kHigh = 0x89abcdefu;
  constexpr uint64_t kPacked =
      (static_cast<uint64_t>(kHigh) << 32) | kLow;
  constexpr uint64_t kExpected[] = {kPacked, kPacked + 1};

  uint64_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  consume_compiler_pack_alias_chain<<<1, 1>>>(output, kLow, kHigh);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], kExpected[0]);
  EXPECT_EQ(actual[1], kExpected[1]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderPackIntegrationTest, PreservesPackAcrossPriorBarrier) {
  constexpr uint32_t kLow = 0x76543210u;
  constexpr uint32_t kHigh = 0xfedcba98u;
  constexpr uint64_t kPacked =
      (static_cast<uint64_t>(kHigh) << 32) | kLow;
  constexpr uint64_t kExpected = kPacked + 1;

  uint64_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  consume_compiler_pack_across_barrier<<<1, 1>>>(output, kLow, kHigh);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpected);

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

TEST(PtxReorderCompilerPatternIntegrationTest, FusesShiftAddIntoMad) {
  constexpr uint32_t kX = 0xffffffe1u;
  constexpr uint32_t kFraction = 0x00754321u;
  constexpr uint32_t kExpected = (kX << 23) + kFraction;

  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  compiler_shift_add_mad<<<1, 1>>>(output, kX, kFraction);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpected);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest, FusesSignBitShiftIntoMad) {
  constexpr uint32_t kX = 0x80000001u;
  constexpr uint32_t kFraction = 0x76543210u;
  constexpr uint32_t kExpected = (kX << 31) + kFraction;

  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  compiler_sign_bit_shift_add_mad<<<1, 1>>>(output, kX, kFraction);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpected);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     PreservesExternallyUsedIntermediate) {
  constexpr uint32_t kX = 0x10203040u;
  constexpr uint32_t kFraction = 0x00543210u;
  constexpr uint32_t kExpected[] = {(kX << 23) + kFraction, kX};

  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  compiler_shift_add_with_extra_use<<<1, 1>>>(output, kX, kFraction);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], kExpected[0]);
  EXPECT_EQ(actual[1], kExpected[1]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest, EliminatesScalarCopy) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  compiler_scalar_copy<<<1, 1>>>(output, 35);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, 42u);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     RejectsScalarCopyWithSourceRedefinition) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  compiler_scalar_copy_with_source_redefinition<<<1, 1>>>(output, 20);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, 41u);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     PropagatesPackedF32x2Literals) {
  constexpr uint64_t kPackedTwo = 0x4000000040000000ULL;
  constexpr uint64_t kExpected[] = {
      0x4040000040400000ULL,
      0x3f8000003f800000ULL,
      0x4040000040400000ULL,
  };

  uint64_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  compiler_packed_f32x2_literal<<<1, 1>>>(output, kPackedTwo);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual[3] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 3; ++i) EXPECT_EQ(actual[i], kExpected[i]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     RejectsPackedLiteralWithUnsupportedUse) {
  constexpr uint64_t kPackedTwo = 0x4000000040000000ULL;
  constexpr uint64_t kExpected[] = {
      0x4040000040400000ULL,
      0x3f8000003f800000ULL,
  };

  uint64_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  compiler_packed_f32x2_literal_with_other_use<<<1, 1>>>(output, kPackedTwo);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i = 0; i < 2; ++i) EXPECT_EQ(actual[i], kExpected[i]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     FusesScalarBroadcastPack) {
  constexpr uint32_t kPackedOnePointFive = 0x3fc00000u;
  constexpr uint64_t kExpected = 0x4040000040400000ULL;

  uint64_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);

  compiler_broadcast_pack<<<1, 1>>>(output, kPackedOnePointFive);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, kExpected);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest, FusesNegatedMultiply) {
  float *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);
  // Compare bits: EXPECT_FLOAT_EQ alone treats +0 and -0 as equal.
  const float inputs[] = {1.5f, 0.0f, -0.0f, -1.5f};
  const uint32_t expected[] = {0xc0400000u, 0u, 0u, 0x40400000u};
  for (unsigned i = 0; i < 4; ++i) {
    compiler_negated_multiply<<<1, 1>>>(output, inputs[i], 2.0f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t actual = 0;
    ASSERT_EQ(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual, expected[i]);
  }
  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     RejectsNegatedMultiplyWithExtraUse) {
  constexpr float kExpected[] = {-3.0f, 3.0f};
  float *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, sizeof(kExpected)), cudaSuccess);

  compiler_negated_multiply_with_extra_use<<<1, 1>>>(output, 1.5f, 2.0f);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_FLOAT_EQ(actual[0], kExpected[0]);
  EXPECT_FLOAT_EQ(actual[1], kExpected[1]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     FoldsPredicateNotIntoConsumerGuard) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 2 * sizeof(*output)), cudaSuccess);

  compiler_predicate_not_guard<<<1, 1>>>(output, 0);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t actual[2] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], 0x22222222u);
  EXPECT_EQ(actual[1], 0x33333333u);

  compiler_predicate_not_guard<<<1, 1>>>(output, 1);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], 0x11111111u);
  EXPECT_EQ(actual[1], 0x44444444u);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     ExtractsOneByteIntoSevenPredicates) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 7 * sizeof(*output)), cudaSuccess);

  compiler_predicate_byte_extract<<<1, 1>>>(output, 0);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t actual[7] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  const uint32_t expected_zero[7] = {101, 12, 13, 14, 15, 16, 17};
  for (unsigned i = 0; i < 7; ++i)
    EXPECT_EQ(actual[i], expected_zero[i]);

  compiler_predicate_byte_extract<<<1, 1>>>(output, 0x7f);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  const uint32_t expected_set[7] = {11, 102, 103, 104, 105, 106, 107};
  for (unsigned i = 0; i < 7; ++i)
    EXPECT_EQ(actual[i], expected_set[i]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

TEST(PtxReorderCompilerPatternIntegrationTest,
     RejectsPredicateByteExtractWithExtraTemporaryUse) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 8 * sizeof(*output)), cudaSuccess);

  compiler_predicate_byte_extract_with_extra_temp_use<<<1, 1>>>(output, 0x7f);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t actual[8] = {};
  ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  const uint32_t expected[8] = {11, 12, 13, 14, 15, 16, 17, 1};
  for (unsigned i = 0; i < 8; ++i) EXPECT_EQ(actual[i], expected[i]);

  EXPECT_EQ(cudaFree(output), cudaSuccess);
}

}  // namespace
