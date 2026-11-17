#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "cp_kernels.cuh"
#include "ptx/mbarrier.cuh"

// Commit and wait on a bulk group that contains no TMA stores. This kernel is
// local to the bulk-group test translation unit so including cp_kernels.cuh in
// another test binary does not define a second global entry point.
__global__ void empty_tma_bulk_group_kernel(uint32_t *completion) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
    *completion = 1;
  }
}

//=============================================================================
// TMA Store with Bulk Group Tests
//=============================================================================

// Test class for TMA Store with Bulk Group functionality
class TmaBulkGroupStoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use smaller data for store tests
    num_elements = 4096;  // 16KB of uint32_t data
    data_size_bytes = num_elements * sizeof(uint32_t);

    // Allocate and initialize host input
    h_input.resize(num_elements);
    h_output.resize(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
      h_input[i] = static_cast<uint32_t>(i + 1);
    }
  }

  // Test data
  size_t num_elements;
  size_t data_size_bytes;
  std::vector<uint32_t> h_input;
  std::vector<uint32_t> h_output;
};

// Test: Basic TMA Store with single bulk group
TEST_F(TmaBulkGroupStoreTest, BasicTMAStore) {
  constexpr int CHUNK_BYTES = 256;  // 64 uint32_t elements per chunk
  constexpr int NUM_GROUPS = 1;

  uint32_t *d_input = nullptr;
  uint32_t *d_output = nullptr;

  ASSERT_EQ(cudaMalloc(&d_input, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, data_size_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), data_size_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, data_size_bytes), cudaSuccess);

  // Calculate grid/block dimensions
  const int threads_per_block = 32;  // Single warp
  const int elements_per_warp = NUM_GROUPS * (CHUNK_BYTES / sizeof(uint32_t));
  const int num_warps_needed = (num_elements + elements_per_warp - 1) / elements_per_warp;
  const int blocks = num_warps_needed;

  const size_t shared_mem_size = CHUNK_BYTES;

  tma_store_kernel<NUM_GROUPS, CHUNK_BYTES>
      <<<blocks, threads_per_block, shared_mem_size>>>(d_input, d_output, num_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

  // Copy result back
  ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, data_size_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify: each element should be input + 1
  int errors = 0;
  for (size_t i = 0; i < num_elements; ++i) {
    uint32_t expected = h_input[i] + 1;
    if (h_output[i] != expected) {
      if (errors < 10) {
        printf("Mismatch at %zu: expected %u, got %u\n", i, expected, h_output[i]);
      }
      errors++;
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_input);
  cudaFree(d_output);
}

// Test: TMA Store with multiple groups per warp
TEST_F(TmaBulkGroupStoreTest, MultiGroupTMAStore) {
  constexpr int CHUNK_BYTES = 128;  // 32 uint32_t elements per chunk
  const int num_chunks = 8;
  const int elements_per_chunk = CHUNK_BYTES / sizeof(uint32_t);
  const size_t test_elements = num_chunks * elements_per_chunk;

  std::vector<uint32_t> test_input(test_elements);
  std::vector<uint32_t> test_output(test_elements);

  for (size_t i = 0; i < test_elements; ++i) {
    test_input[i] = static_cast<uint32_t>(i * 10);
  }

  uint32_t *d_input = nullptr;
  uint32_t *d_output = nullptr;
  size_t test_bytes = test_elements * sizeof(uint32_t);

  ASSERT_EQ(cudaMalloc(&d_input, test_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, test_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, test_input.data(), test_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, test_bytes), cudaSuccess);

  const size_t shared_mem_size = CHUNK_BYTES;

  tma_store_multi_group_kernel<CHUNK_BYTES>
      <<<1, 32, shared_mem_size>>>(d_input, d_output, num_chunks);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

  ASSERT_EQ(cudaMemcpy(test_output.data(), d_output, test_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify: each chunk's elements should have (chunk_index + 1) added
  int errors = 0;
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    for (int i = 0; i < elements_per_chunk; i++) {
      size_t idx = chunk * elements_per_chunk + i;
      uint32_t expected = test_input[idx] + (chunk + 1);
      if (test_output[idx] != expected) {
        if (errors < 10) {
          printf("Chunk %d, elem %d (idx %zu): expected %u, got %u\n",
                 chunk, i, idx, expected, test_output[idx]);
        }
        errors++;
      }
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_input);
  cudaFree(d_output);
}

// Test: Pipelined TMA Store with wait_group(N)
TEST_F(TmaBulkGroupStoreTest, PipelinedTMAStore) {
  constexpr int CHUNK_BYTES = 128;
  constexpr int MAX_IN_FLIGHT = 2;  // Allow 2 groups in flight
  const int num_chunks = 6;
  const int elements_per_chunk = CHUNK_BYTES / sizeof(uint32_t);
  const size_t test_elements = num_chunks * elements_per_chunk;

  std::vector<uint32_t> test_input(test_elements);
  std::vector<uint32_t> test_output(test_elements);

  for (size_t i = 0; i < test_elements; ++i) {
    test_input[i] = static_cast<uint32_t>(i + 100);
  }

  uint32_t *d_input = nullptr;
  uint32_t *d_output = nullptr;
  size_t test_bytes = test_elements * sizeof(uint32_t);

  ASSERT_EQ(cudaMalloc(&d_input, test_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, test_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, test_input.data(), test_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, test_bytes), cudaSuccess);

  // Need enough shared memory for multiple slots
  const size_t shared_mem_size = (MAX_IN_FLIGHT + 1) * CHUNK_BYTES;

  tma_store_pipelined_kernel<CHUNK_BYTES, MAX_IN_FLIGHT>
      <<<1, 32, shared_mem_size>>>(d_input, d_output, num_chunks);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

  ASSERT_EQ(cudaMemcpy(test_output.data(), d_output, test_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify: each element should be doubled
  int errors = 0;
  for (size_t i = 0; i < test_elements; ++i) {
    uint32_t expected = test_input[i] * 2;
    if (test_output[i] != expected) {
      if (errors < 10) {
        printf("Index %zu: expected %u, got %u\n", i, expected, test_output[i]);
      }
      errors++;
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_input);
  cudaFree(d_output);
}

// Test: Large scale TMA Store with many groups
TEST_F(TmaBulkGroupStoreTest, LargeScaleTMAStore) {
  constexpr int CHUNK_BYTES = 256;
  constexpr int NUM_GROUPS = 16;

  // Use larger data
  const size_t large_elements = 65536;  // 256KB
  const size_t large_bytes = large_elements * sizeof(uint32_t);

  std::vector<uint32_t> large_input(large_elements);
  std::vector<uint32_t> large_output(large_elements);

  for (size_t i = 0; i < large_elements; ++i) {
    large_input[i] = static_cast<uint32_t>(i % 1000);
  }

  uint32_t *d_input = nullptr;
  uint32_t *d_output = nullptr;

  ASSERT_EQ(cudaMalloc(&d_input, large_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, large_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_input, large_input.data(), large_bytes,
                       cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_output, 0, large_bytes), cudaSuccess);

  const int threads_per_block = 128;  // 4 warps per block
  const int elements_per_warp = NUM_GROUPS * (CHUNK_BYTES / sizeof(uint32_t));
  const int warps_per_block = threads_per_block / 32;
  const int elements_per_block = warps_per_block * elements_per_warp;
  const int blocks = (large_elements + elements_per_block - 1) / elements_per_block;

  const size_t shared_mem_size = warps_per_block * CHUNK_BYTES;

  tma_store_kernel<NUM_GROUPS, CHUNK_BYTES>
      <<<blocks, threads_per_block, shared_mem_size>>>(d_input, d_output, large_elements);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

  ASSERT_EQ(cudaMemcpy(large_output.data(), d_output, large_bytes,
                       cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify
  int errors = 0;
  for (size_t i = 0; i < large_elements; ++i) {
    uint32_t expected = large_input[i] + 1;
    if (large_output[i] != expected) {
      if (errors < 10) {
        printf("Index %zu: expected %u, got %u\n", i, expected, large_output[i]);
      }
      errors++;
    }
  }
  EXPECT_EQ(errors, 0) << "Total mismatches: " << errors;

  cudaFree(d_input);
  cudaFree(d_output);
}

// Test: commit and wait on a bulk group with no TMA stores.
TEST_F(TmaBulkGroupStoreTest, EmptyBulkGroupCompletes) {
  uint32_t *device_completion = nullptr;
  ASSERT_EQ(cudaMalloc(&device_completion, sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_completion, 0, sizeof(uint32_t)), cudaSuccess);

  empty_tma_bulk_group_kernel<<<1, 1>>>(device_completion);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

  uint32_t completion = 0;
  ASSERT_EQ(cudaMemcpy(&completion, device_completion, sizeof(completion),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(completion, 1u);

  EXPECT_EQ(cudaFree(device_completion), cudaSuccess);
}

namespace {

using flashgpu::test::ptx::mbarrier_arrive_expect_tx;
using flashgpu::test::ptx::mbarrier_init;
using flashgpu::test::ptx::mbarrier_inval;
using flashgpu::test::ptx::mbarrier_wait_parity;
using flashgpu::test::ptx::smem_u64_addr;

constexpr unsigned kChunkBytes = 4096;
constexpr unsigned kCopiesPerPhase = 4;
constexpr unsigned kPhaseBytes = kChunkBytes * kCopiesPerPhase;
constexpr unsigned kPhases = 2;
constexpr unsigned kWordsPerPhase = kPhaseBytes / sizeof(uint32_t);
constexpr unsigned kTotalWords = kPhases * kWordsPerPhase;
constexpr unsigned kSlotPressureTransactions = 17;
constexpr unsigned kSlotPressureChunkBytes = 2816;
constexpr unsigned kSlotPressureBytes =
    kSlotPressureTransactions * kSlotPressureChunkBytes;
constexpr unsigned kSlotPressureWords = kSlotPressureBytes / sizeof(uint32_t);

__device__ __forceinline__ uint64_t global_address(const void* pointer) {
  uint64_t address = 0;
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(address) : "l"(pointer));
  return address;
}

__device__ __forceinline__ void linear_bulk_g2s(void* destination,
                                                const void* source,
                                                uint64_t* barrier) {
  const uint64_t destination_s = smem_u64_addr(destination);
  const uint64_t source_g = global_address(source);
  const uint64_t barrier_s = smem_u64_addr(barrier);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "l"(destination_s), "l"(source_g), "n"(kChunkBytes),
        "l"(barrier_s)
      : "memory");
}

__device__ __forceinline__ void slot_pressure_bulk_g2s(void* destination,
                                                       const void* source,
                                                       uint64_t* barrier) {
  const uint64_t destination_s = smem_u64_addr(destination);
  const uint64_t source_g = global_address(source);
  const uint64_t barrier_s = smem_u64_addr(barrier);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "l"(destination_s), "l"(source_g), "n"(kSlotPressureChunkBytes),
        "l"(barrier_s)
      : "memory");
}

__global__ void tma_transaction_slot_pressure_kernel(const uint32_t* input,
                                                     uint64_t* output_sum) {
  extern __shared__ __align__(16) uint8_t shared[];
  auto* barrier = reinterpret_cast<uint64_t*>(shared + kSlotPressureBytes);

  if (threadIdx.x == 0) {
    mbarrier_init(barrier, 1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbarrier_arrive_expect_tx(barrier, kSlotPressureBytes);
  }
  __syncthreads();

  const unsigned warp = threadIdx.x / warpSize;
  const unsigned lane = threadIdx.x % warpSize;
  if (lane == 0 && warp < kSlotPressureTransactions) {
    const unsigned byte_offset = warp * kSlotPressureChunkBytes;
    slot_pressure_bulk_g2s(shared + byte_offset,
                           input + byte_offset / sizeof(uint32_t), barrier);
  }
  __syncthreads();

  if (threadIdx.x == 0) mbarrier_wait_parity(barrier, 0);
  __syncthreads();

  if (threadIdx.x == 0) {
    const auto* words = reinterpret_cast<const uint32_t*>(shared);
    uint64_t sum = 0;
    for (unsigned word = 0; word < kSlotPressureWords; ++word) {
      sum += words[word];
    }
    *output_sum = sum;
    mbarrier_inval(barrier);
  }
}

__global__ void linear_bulk_two_phase_kernel(const uint32_t* input,
                                              uint64_t* phase_sums) {
  extern __shared__ __align__(16) uint8_t shared[];
  if (threadIdx.x != 0) return;

  auto* barrier = reinterpret_cast<uint64_t*>(shared + kPhaseBytes);
  mbarrier_init(barrier, 1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

  for (unsigned phase = 0; phase < kPhases; ++phase) {
    mbarrier_arrive_expect_tx(barrier, kPhaseBytes);
    for (unsigned copy = 0; copy < kCopiesPerPhase; ++copy) {
      linear_bulk_g2s(shared + copy * kChunkBytes,
                      input + phase * kWordsPerPhase +
                          copy * (kChunkBytes / sizeof(uint32_t)),
                      barrier);
    }

    mbarrier_wait_parity(barrier, phase & 1);
    asm volatile("" ::: "memory");

    const auto* words = reinterpret_cast<const uint32_t*>(shared);
    uint64_t sum = 0;
    for (unsigned word = 0; word < kWordsPerPhase; ++word) {
      sum += words[word];
    }
    phase_sums[phase] = sum;
  }

  mbarrier_inval(barrier);
}

TEST(TmaLinearBulkMultiTxBarrierTest,
     AggregatesFourTransactionsAndReusesBarrierAcrossPhases) {
  std::array<uint32_t, kTotalWords> input{};
  for (unsigned i = 0; i < input.size(); ++i) input[i] = i + 1;

  std::array<uint64_t, kPhases> expected{};
  for (unsigned phase = 0; phase < kPhases; ++phase) {
    expected[phase] = std::accumulate(
        input.begin() + phase * kWordsPerPhase,
        input.begin() + (phase + 1) * kWordsPerPhase, uint64_t{0});
  }

  uint32_t* device_input = nullptr;
  uint64_t* device_sums = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_sums, sizeof(expected)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), sizeof(input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  constexpr unsigned kSharedBytes = kPhaseBytes + sizeof(uint64_t);
  linear_bulk_two_phase_kernel<<<1, 1, kSharedBytes>>>(device_input,
                                                       device_sums);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::array<uint64_t, kPhases> actual{};
  ASSERT_EQ(cudaMemcpy(actual.data(), device_sums, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, expected);

  EXPECT_EQ(cudaFree(device_sums), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

TEST(TmaTransactionSlotTest, SeventeenthTransactionMakesForwardProgress) {
  std::array<uint32_t, kSlotPressureWords> input{};
  for (unsigned i = 0; i < input.size(); ++i) input[i] = i + 1;
  const uint64_t expected =
      std::accumulate(input.begin(), input.end(), uint64_t{0});

  uint32_t* device_input = nullptr;
  uint64_t* device_sum = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, sizeof(input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_sum, sizeof(uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, input.data(), sizeof(input),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  constexpr unsigned kThreads = kSlotPressureTransactions * 32;
  constexpr unsigned kSharedBytes = kSlotPressureBytes + sizeof(uint64_t);
  tma_transaction_slot_pressure_kernel<<<1, kThreads, kSharedBytes>>>(
      device_input, device_sum);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint64_t actual = 0;
  ASSERT_EQ(cudaMemcpy(&actual, device_sum, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, expected);

  EXPECT_EQ(cudaFree(device_sum), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

}  // namespace
