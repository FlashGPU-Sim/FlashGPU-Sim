#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "gpgpu-sim/flash/tensormap.h"
#include "gpgpu-sim/flash/tma_reduction.h"

namespace {

TEST(TmaTensorReductionTest, ValidatesDocumentedOperationTypePairs) {
  EXPECT_TRUE(
      tma_tensor_reduction_supported(tma_reduction_op_t::ADD, TMA_DTYPE_F32));
  EXPECT_TRUE(tma_tensor_reduction_supported(tma_reduction_op_t::ADD,
                                             TMA_DTYPE_F32_FTZ));
  EXPECT_TRUE(
      tma_tensor_reduction_supported(tma_reduction_op_t::ADD, TMA_DTYPE_F16));
  EXPECT_TRUE(
      tma_tensor_reduction_supported(tma_reduction_op_t::ADD, TMA_DTYPE_BF16));
  EXPECT_TRUE(
      tma_tensor_reduction_supported(tma_reduction_op_t::MIN, TMA_DTYPE_S64));
  EXPECT_TRUE(
      tma_tensor_reduction_supported(tma_reduction_op_t::INC, TMA_DTYPE_U32));
  EXPECT_TRUE(tma_tensor_reduction_supported(tma_reduction_op_t::BIT_XOR,
                                             TMA_DTYPE_U64));

  EXPECT_FALSE(
      tma_tensor_reduction_supported(tma_reduction_op_t::MIN, TMA_DTYPE_F32));
  EXPECT_FALSE(
      tma_tensor_reduction_supported(tma_reduction_op_t::ADD, TMA_DTYPE_S64));
  EXPECT_FALSE(
      tma_tensor_reduction_supported(tma_reduction_op_t::INC, TMA_DTYPE_S32));
  EXPECT_FALSE(
      tma_tensor_reduction_supported(tma_reduction_op_t::ADD, TMA_DTYPE_F64));
}

TEST(TmaTensorReductionTest, AppliesIntegerOperations) {
  std::array<uint32_t, 4> dst = {4, 0, std::numeric_limits<uint32_t>::max(),
                                 0xf0f0u};
  const std::array<uint32_t, 4> src = {7, 0, 2, 0x0ff0u};

  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_U32, dst.data(),
                             src.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<uint32_t, 4>{11, 0, 1, 0x100e0u}));

  apply_tma_tensor_reduction(tma_reduction_op_t::BIT_XOR, TMA_DTYPE_U32,
                             dst.data(), src.data(), sizeof(dst));
  EXPECT_EQ(dst[3], 0x10f10u);
}

TEST(TmaTensorReductionTest, AppliesSignedMinAndMax) {
  std::array<int32_t, 4> dst = {-9, 12, -3, 8};
  const std::array<int32_t, 4> src = {-4, 7, -11, 20};

  apply_tma_tensor_reduction(tma_reduction_op_t::MIN, TMA_DTYPE_S32, dst.data(),
                             src.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<int32_t, 4>{-9, 7, -11, 8}));

  apply_tma_tensor_reduction(tma_reduction_op_t::MAX, TMA_DTYPE_S32, dst.data(),
                             src.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<int32_t, 4>{-4, 7, -11, 20}));
}

TEST(TmaTensorReductionTest, AppliesIncAndDec) {
  std::array<uint32_t, 4> dst = {0, 2, 5, 7};
  const std::array<uint32_t, 4> limit = {3, 3, 3, 7};

  apply_tma_tensor_reduction(tma_reduction_op_t::INC, TMA_DTYPE_U32, dst.data(),
                             limit.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<uint32_t, 4>{1, 3, 0, 0}));

  apply_tma_tensor_reduction(tma_reduction_op_t::DEC, TMA_DTYPE_U32, dst.data(),
                             limit.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<uint32_t, 4>{0, 2, 3, 7}));
}

TEST(TmaTensorReductionTest, AppliesFloatingPointAdd) {
  std::array<float, 4> dst = {1.0f, -2.0f, 3.5f, 0.25f};
  const std::array<float, 4> src = {0.5f, 4.0f, -1.5f, 0.75f};

  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_F32, dst.data(),
                             src.data(), sizeof(dst));
  EXPECT_EQ(dst, (std::array<float, 4>{1.5f, 2.0f, 2.0f, 1.0f}));
}

TEST(TmaTensorReductionTest, DistinguishesF32AndF32Ftz) {
  constexpr uint32_t kSmallestSubnormal = 1;
  std::array<uint32_t, 1> regular_dst = {0};
  std::array<uint32_t, 1> ftz_dst = {0};
  const std::array<uint32_t, 1> src = {kSmallestSubnormal};

  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_F32,
                             regular_dst.data(), src.data(), sizeof(src));
  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_F32_FTZ,
                             ftz_dst.data(), src.data(), sizeof(src));

  EXPECT_EQ(regular_dst[0], kSmallestSubnormal);
  EXPECT_EQ(ftz_dst[0], 0u);
}

TEST(TmaTensorReductionTest, AppliesF16AndBf16Operations) {
  std::array<uint16_t, 2> f16_dst = {0x3c00u, 0x4000u};  // 1, 2
  const std::array<uint16_t, 2> f16_src = {0x4000u, 0x3c00u};
  apply_tma_tensor_reduction(tma_reduction_op_t::ADD, TMA_DTYPE_F16,
                             f16_dst.data(), f16_src.data(), sizeof(f16_dst));
  EXPECT_EQ(f16_dst, (std::array<uint16_t, 2>{0x4200u, 0x4200u}));  // 3, 3

  std::array<uint16_t, 2> bf16_dst = {0x3f80u, 0x4040u};        // 1, 3
  const std::array<uint16_t, 2> bf16_src = {0x4000u, 0x3f00u};  // 2, 0.5
  apply_tma_tensor_reduction(tma_reduction_op_t::MAX, TMA_DTYPE_BF16,
                             bf16_dst.data(), bf16_src.data(),
                             sizeof(bf16_dst));
  EXPECT_EQ(bf16_dst, (std::array<uint16_t, 2>{0x4000u, 0x4040u}));
}

TEST(TmaTensorReductionTest, RejectsUnsupportedCombination) {
  float dst = 1.0f;
  const float src = 2.0f;
  EXPECT_THROW(
      apply_tma_tensor_reduction(tma_reduction_op_t::MIN, TMA_DTYPE_F32, &dst,
                                 &src, sizeof(dst)),
      std::invalid_argument);
}

}  // namespace
