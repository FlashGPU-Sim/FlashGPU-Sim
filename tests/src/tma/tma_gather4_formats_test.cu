#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <vector>
#include "cp_kernels.cuh"
namespace {
template <bool Cta>
__global__ void gather_formats(const CUtensorMap *map, unsigned *out, int x,
                               unsigned row_bytes) {
  __shared__ __align__(1024) unsigned storage[512];
  __shared__ __align__(8) unsigned long long barrier;
  unsigned *tile = storage + 32;  // Nonzero swizzle phase.
  if (threadIdx.x == 0) mbarrier_init(&barrier, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(&barrier, 4 * row_bytes);
    unsigned dst = __cvta_generic_to_shared(tile),
             bar = __cvta_generic_to_shared(&barrier);
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [%0, "
        "{%1,%2,%3,%4,%5}];" ::"l"(map),
        "r"(x), "r"(5), "r"(-1), "r"(5), "r"(8)
        : "memory");
    if constexpr (Cta)
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::"
          "complete_tx::bytes.L2::cache_hint [%0], [%1, {%2,%3,%4,%5,%6}], "
          "[%7], %8;" ::"r"(dst),
          "l"(map), "r"(x), "r"(5), "r"(-1), "r"(5), "r"(8), "r"(bar),
          "l"(0x1000000000000000ull)
          : "memory");
    else
      asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global."
          "mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4,%5,%6}], [%7];" ::
              "r"(dst),
          "l"(map), "r"(x), "r"(5), "r"(-1), "r"(5), "r"(8), "r"(bar)
          : "memory");
  }
  wait(&barrier, 0);
#pragma unroll
  for (unsigned i = 0; i < 4; ++i)
    out[threadIdx.x + i * 32] = tile[threadIdx.x + i * 32];
}
TEST(TmaGather4Formats, RawFloatSwizzleStrideAndOob) {
  const unsigned widths[] = {1, 2, 4, 4, 8, 8, 2, 4, 8, 2, 4, 4, 4};
  const CUtensorMapDataType types[] = {
      CU_TENSOR_MAP_DATA_TYPE_UINT8,       CU_TENSOR_MAP_DATA_TYPE_UINT16,
      CU_TENSOR_MAP_DATA_TYPE_UINT32,      CU_TENSOR_MAP_DATA_TYPE_INT32,
      CU_TENSOR_MAP_DATA_TYPE_UINT64,      CU_TENSOR_MAP_DATA_TYPE_INT64,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT64,     CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
      CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ};
  unsigned char *input;
  unsigned *out;
  CUtensorMap *dm;
  ASSERT_EQ(cudaMalloc(&input, 8192), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 512), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dm, sizeof(CUtensorMap)), cudaSuccess);
  std::vector<unsigned char> data(8192);
  // Exact float values keep raw transport and TF32 conversion references
  // separate.
  const uint32_t vals[] = {0x3f801000u, 0x3f803000u, 0x80000001u, 0x7f800000u};
  for (unsigned i = 0; i < data.size(); i += 4)
    std::memcpy(data.data() + i, &vals[(i / 4) % 4], 4);
  ASSERT_EQ(cudaMemcpy(input, data.data(), data.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  for (unsigned type = 0; type < 13; ++type)
    for (unsigned swizzle = 0; swizzle < 7; ++swizzle) {
      SCOPED_TRACE(::testing::Message()
                   << "type=" << type << " swizzle=" << swizzle);
      unsigned width = widths[type],
               row_bytes = swizzle == 1   ? 32
                           : swizzle == 2 ? 64
                                          : 128,
               columns = row_bytes / width;
      bool nan = type >= 6 && (swizzle & 1);
      cuuint64_t dims[] = {1024 / width, 8}, strides[] = {1024};
      cuuint32_t box[] = {columns, 1}, steps[] = {1, 8};
      alignas(128) CUtensorMap map{};
      ASSERT_EQ(cuTensorMapEncodeTiled(
                    &map, types[type], 2, input, dims, strides, box, steps,
                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                    static_cast<CUtensorMapSwizzle>(swizzle),
                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    nan ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                        : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                CUDA_SUCCESS);
      ASSERT_EQ(cudaMemcpy(dm, &map, sizeof(map), cudaMemcpyHostToDevice),
                cudaSuccess);
      if (swizzle & 1)
        gather_formats<true><<<1, 32>>>(dm, out, 0, row_bytes);
      else
        gather_formats<false><<<1, 32>>>(dm, out, 0, row_bytes);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::array<unsigned char, 512> result{};
      ASSERT_EQ(cudaMemcpy(result.data(), out, 512, cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (unsigned row = 0; row < 4; ++row)
        for (unsigned col = 0; col < columns; ++col) {
          std::array<unsigned char, 8> expected{};
          if (row == 0 || row == 2) {
            std::memcpy(expected.data(), data.data() + 5 * 1024 + col * width,
                        width);
            if (type >= 10) {
              uint32_t word = vals[col % 4];
              if (col % 4 == 2)
                word = 0x80000000u;
              else if (type >= 11 && col % 4 == 0)
                word = 0x3f800000u;
              else if (type >= 11 && col % 4 == 1)
                word = 0x3f804000u;
              std::memcpy(expected.data(), &word, 4);
            }
          } else if (nan) {
            for (unsigned b = 0; b < width; ++b) expected[b] = 0xff;
            expected[width - 1] = 0x7f;
            if (type >= 11) {
              expected[0] = 0;
              expected[1] = 0xe0;
            }
          }
          for (unsigned b = 0; b < width; ++b) {
            unsigned offset = row * row_bytes + col * width + b;
            unsigned line = (offset + 128) / 128;
            unsigned physical =
                offset ^
                (swizzle <= 3   ? (swizzle ? (line % (1u << swizzle)) * 16 : 0)
                 : swizzle <= 5 ? (line % 4) * 32
                                : (line % 2) * 64);
            if (swizzle == 5) physical ^= (line % 2) * 8;
            EXPECT_EQ(result[physical], expected[b])
                << row << "," << col << "," << b;
          }
        }
    }
  EXPECT_EQ(cudaFree(dm), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
}

TEST(TmaGather4Formats, PackedSubbyteAndPadding) {
  unsigned char *input;
  unsigned *out;
  CUtensorMap *dm;
  ASSERT_EQ(cudaMalloc(&input, 8192), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, 512), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dm, sizeof(CUtensorMap)), cudaSuccess);
  for (unsigned type = 13; type <= 15; ++type)
    for (unsigned swizzle = 0; swizzle < 7; ++swizzle) {
      if (type >= 14 && swizzle != 0 && swizzle != 3 && swizzle != 4) continue;
      SCOPED_TRACE(::testing::Message()
                   << "type=" << type << " swizzle=" << swizzle);
      const unsigned bits = type == 15 ? 6 : 4;
      const unsigned cols = type >= 14     ? 128
                            : swizzle == 1 ? 64
                            : swizzle == 2 ? 128
                                           : 256;
      const unsigned row_bytes = cols * bits / 8,
                     shared_row = type >= 14 ? cols : row_bytes;
      std::vector<unsigned char> data(8192);
      for (unsigned row = 0; row < 8; ++row)
        for (unsigned x = 0; x < 512; ++x) {
          const unsigned value = (row * 13 + x) & ((1u << bits) - 1);
          for (unsigned bit = 0; bit < bits; ++bit)
            data[row * 1024 + (x * bits + bit) / 8] |=
                ((value >> bit) & 1u) << ((x * bits + bit) % 8);
        }
      ASSERT_EQ(cudaMemcpy(input, data.data(), 8192, cudaMemcpyHostToDevice),
                cudaSuccess);
      alignas(128) CUtensorMap map{};
      cuuint64_t dims[] = {512, 8}, strides[] = {1024};
      cuuint32_t box[] = {cols, 1}, steps[] = {1, 1};
      ASSERT_EQ(cuTensorMapEncodeTiled(
                    &map, static_cast<CUtensorMapDataType>(type), 2, input,
                    dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    static_cast<CUtensorMapSwizzle>(swizzle),
                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                CUDA_SUCCESS);
      ASSERT_EQ(cudaMemcpy(dm, &map, sizeof(map), cudaMemcpyHostToDevice),
                cudaSuccess);
      gather_formats<true><<<1, 32>>>(dm, out, 128, row_bytes);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::array<unsigned char, 512> result{};
      ASSERT_EQ(cudaMemcpy(result.data(), out, 512, cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (unsigned row = 0; row < 4; ++row)
        for (unsigned col = 0; col < cols; ++col) {
          unsigned actual = 0;
          for (unsigned bit = 0; bit < bits; ++bit) {
            const unsigned logical =
                row * shared_row * 8 +
                (type >= 14 ? (col / 16) * 128 + (col % 16) * bits
                            : col * bits) +
                bit;
            const unsigned byte = logical / 8, line = (byte + 128) / 128;
            unsigned physical =
                byte ^
                (swizzle <= 3   ? (swizzle ? (line % (1u << swizzle)) * 16 : 0)
                 : swizzle <= 5 ? (line % 4) * 32
                                : (line % 2) * 64);
            if (swizzle == 5) physical ^= (line % 2) * 8;
            actual |= ((result[physical] >> (logical % 8)) & 1u) << bit;
          }
          const unsigned expected =
              row == 0 || row == 2 ? (5 * 13 + 128 + col) & ((1u << bits) - 1)
                                   : 0;
          EXPECT_EQ(actual, expected) << row << "," << col;
        }
    }
  EXPECT_EQ(cudaFree(dm), cudaSuccess);
  EXPECT_EQ(cudaFree(out), cudaSuccess);
  EXPECT_EQ(cudaFree(input), cudaSuccess);
}
}  // namespace
