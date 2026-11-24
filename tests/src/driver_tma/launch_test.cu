#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

// Test-only opaque descriptor construction from the retained Driver layout.
// No simulator decoder or address helper is used by this fixture or oracle.
struct alignas(64) DriverMap {
  uint64_t words[16]{};
};
template <int Rank, bool Store>
__global__ void driver_map_copy(const __grid_constant__ DriverMap source,
                                const __grid_constant__ DriverMap destination,
                                unsigned char *out, unsigned bytes,
                                unsigned swizzle) {
  __shared__ __align__(1024) unsigned char tile[1024];
  __shared__ __align__(8) unsigned long long barrier;
  const unsigned smem = (unsigned)__cvta_generic_to_shared(tile);
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(bar),
        "r"(bytes)
        : "memory");
    if constexpr (Rank == 2)
      asm volatile(
          "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{x,y}],[%2]; }" ::"r"(smem),
          "l"(&source), "r"(bar)
          : "memory");
    if constexpr (Rank == 3)
      asm volatile(
          "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
          "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{x,y,y}],[%2]; }" ::"r"(smem),
          "l"(&source), "r"(bar)
          : "memory");
    if constexpr (Rank == 4)
      asm volatile(
          "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
          "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{x,y,y,y}],[%2]; }" ::"r"(smem),
          "l"(&source), "r"(bar)
          : "memory");
    if constexpr (Rank == 5)
      asm volatile(
          "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
          "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0],[%1,{x,y,y,y,y}],[%2]; }" ::"r"(smem),
          "l"(&source), "r"(bar)
          : "memory");
    asm volatile(
        "{.reg .pred p; loop: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],0; @!p bra loop;}" ::"r"(bar)
        : "memory");
    if constexpr (Store) {
      // Store the unchanged shared tile into a different global tensor.
      if constexpr (Rank == 2)
        asm volatile(
            "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0,{x,y}],[%1]; }" ::"l"(&destination),
            "r"(smem)
            : "memory");
      if constexpr (Rank == 4)
        asm volatile(
            "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
            "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
            "[%0,{x,y,y,y}],[%1]; }" ::"l"(&destination),
            "r"(smem)
            : "memory");
      if constexpr (Rank == 5)
        asm volatile(
            "{ .reg .b32 x,y; mov.u32 x,0; mov.u32 y,1; "
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
            "[%0,{x,y,y,y,y}],[%1]; }" ::"l"(&destination),
            "r"(smem)
            : "memory");
      asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::
                       : "memory");
    }
  }
  __syncthreads();
  if constexpr (!Store) {
    for (unsigned i = threadIdx.x * 4; i < bytes; i += blockDim.x * 4) {
      const unsigned physical =
          swizzle ? i ^ ((((i >> 7) + (smem >> 7)) & ((swizzle / 16) - 1)) << 4)
                  : i;
      *reinterpret_cast<unsigned *>(out + i) =
          *reinterpret_cast<unsigned *>(tile + physical);
    }
  }
}

template <int Rank, bool Store = false>
void check_driver(uint32_t control, unsigned elem, unsigned swizzle) {
  const unsigned width = swizzle ? swizzle / elem : 16;
  const unsigned rows = 4;
  std::array<unsigned, 5> dims{width, rows, 2, 2, 2}, box{width, 2, 1, 1, 1};
  size_t total = elem;
  for (unsigned i = 0; i < Rank; ++i) total *= dims[i];
  const unsigned tile_bytes = width * 2 * elem;
  std::vector<unsigned char> input(total), output(Store ? total : tile_bytes),
      expected(output.size(), 0xa5);
  for (size_t i = 0; i < total; ++i) input[i] = (i * 37 + (i >> 5) + 11) % 251;
  unsigned char *in, *out;
  ASSERT_EQ(cudaMalloc(&in, total), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&out, output.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(in, input.data(), total, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(out, 0xa5, output.size()), cudaSuccess);
  DriverMap src{}, dst{};
  src.words[0] = (uint64_t)in;
  src.words[1] = control;
  size_t stride = width * elem;
  for (unsigned i = 1; i < Rank; ++i) {
    unsigned word = (i + 2) / 2, shift = (i % 2) * 32;
    src.words[word] |= (uint64_t(stride / 16) << shift);
    stride *= dims[i];
  }
  for (unsigned i = 0; i < Rank; ++i)
    src.words[4 + i / 2] |= uint64_t(dims[i] - 1) << ((i % 2) * 32);
  src.words[6] |= uint64_t(box[0] - 1) << 56;
  for (unsigned i = 1; i < Rank; ++i)
    src.words[7] |= uint64_t(box[i] - 1) << (8 * (i - 1));
  src.words[8] = tile_bytes;
  src.words[9] = swizzle ? swizzle * 8 : 0x10;
  dst = src;
  dst.words[0] = (uint64_t)out;
  size_t origin = 0, multiplier = width * elem;
  for (unsigned i = 1; i < Rank; ++i) {
    origin += multiplier;
    multiplier *= dims[i];
  }
  for (unsigned i = 0; i < tile_bytes; ++i)
    expected[Store ? origin + i : i] = input[origin + i];
  driver_map_copy<Rank, Store><<<1, 32>>>(src, dst, out, tile_bytes, swizzle);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(output.data(), out, output.size(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(output, expected);
  ASSERT_EQ(cudaFree(in), cudaSuccess);
  ASSERT_EQ(cudaFree(out), cudaSuccess);
}
TEST(DriverTma, Dense2D) {
  check_driver<2>(0x240010, 1, 0);
  check_driver<2>(0x266010, 1, 128);
  check_driver<2>(0x264510, 2, 64);
  check_driver<2>(0x60390, 4, 0);
}
TEST(DriverTma, Dense3D) {
  check_driver<3>(0x266520, 2, 128);
  check_driver<3>(0x2663a0, 4, 128);
}
TEST(DriverTma, Dense4D) { check_driver<4>(0x266030, 1, 128); }
TEST(DriverTma, Dense5D) {
  check_driver<5>(0x266540, 2, 128);
  check_driver<5>(0x2663c0, 4, 128);
}
TEST(DriverTma, StoreRoundtrip) {
  check_driver<2, true>(0x264510, 2, 64);
  check_driver<4, true>(0x266030, 1, 128);
  check_driver<5, true>(0x2663c0, 4, 128);
}
