#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

template <bool Shared>
__global__ void load_bytes(const int8_t *input, int64_t *output) {
  __shared__ int8_t shared_input[32];
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t signed32;
  int64_t signed64;
  uint32_t unsigned32;
  uint64_t unsigned64;
  if constexpr (Shared) {
    shared_input[threadIdx.x] = input[index];
    __syncthreads();
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_input + threadIdx.x));
    asm volatile("ld.shared.s8 %0, [%1];" : "=r"(signed32) : "r"(address));
    asm volatile("ld.shared.s8 %0, [%1];" : "=l"(signed64) : "r"(address));
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(unsigned32) : "r"(address));
    asm volatile("ld.shared.u8 %0, [%1];" : "=l"(unsigned64) : "r"(address));
  } else {
    const int8_t *address = input + index;
    asm volatile("ld.global.s8 %0, [%1];" : "=r"(signed32) : "l"(address));
    asm volatile("ld.global.s8 %0, [%1];" : "=l"(signed64) : "l"(address));
    asm volatile("ld.global.u8 %0, [%1];" : "=r"(unsigned32) : "l"(address));
    asm volatile("ld.global.u8 %0, [%1];" : "=l"(unsigned64) : "l"(address));
  }
  output[4 * index] = signed32;
  output[4 * index + 1] = signed64;
  output[4 * index + 2] = unsigned32;
  output[4 * index + 3] = unsigned64;
}

template <bool Shared>
void check_byte_extension() {
  std::array<int8_t, 256> input;
  for (int i = 0; i < 256; ++i) input[i] = static_cast<int8_t>(i - 128);
  std::array<int64_t, 256 * 4> output{};
  int8_t *device_input = nullptr;
  int64_t *device_output = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_input, sizeof(input)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_output, sizeof(output)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_input, input.data(), sizeof(input),
                                    cudaMemcpyHostToDevice));
  load_bytes<Shared><<<8, 32>>>(device_input, device_output);
  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), device_output, sizeof(output),
                                    cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaFree(device_output));
  ASSERT_EQ(cudaSuccess, cudaFree(device_input));
  for (unsigned i = 0; i < input.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(static_cast<int64_t>(input[i]), output[4 * i]);
    EXPECT_EQ(static_cast<int64_t>(input[i]), output[4 * i + 1]);
    EXPECT_EQ(static_cast<uint8_t>(input[i]), output[4 * i + 2]);
    EXPECT_EQ(static_cast<uint8_t>(input[i]), output[4 * i + 3]);
  }
}

TEST(PtxSignedByteLoad, GlobalInto32And64BitDestinations) {
  check_byte_extension<false>();
}

TEST(PtxSignedByteLoad, SharedInto32And64BitDestinations) {
  check_byte_extension<true>();
}

}  // namespace
