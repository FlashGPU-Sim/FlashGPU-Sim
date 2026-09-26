#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "cp_kernels.cuh"

// Template parameters for configurable test
template <int STAGES = 2, int NUM_PRODUCERS = 1, int NUM_CONSUMERS = 1,
          int CHUNK_SIZE = 256>
struct TMAConfig {
  static constexpr int stages = STAGES;
  static constexpr int num_producers = NUM_PRODUCERS;
  static constexpr int num_consumers = NUM_CONSUMERS;
  static constexpr int chunk_size = CHUNK_SIZE;
  static constexpr int total_warps = num_producers + num_consumers;
};

// Test class for the linear-bulk TMA producer-consumer pipeline
class TmaProducerConsumerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test with 1MB of data
    num_elements = 262144; // 1MB / 4 bytes per float
    data_size_bytes = num_elements * sizeof(float);

    // Allocate host memory
    h_input.resize(num_elements);

    // Initialize with random data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(1, 10);

    for (size_t i = 0; i < num_elements; ++i) {
      h_input[i] = dis(gen);
      //   h_input[i] = 2;
    }
  }

  // CUDA error checking helpers
  cudaError_t checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s - %s\n", message,
              cudaGetErrorString(error));
    }
    return error;
  }

  bool cudaSafeMalloc(void **ptr, size_t size) {
    cudaError_t error = cudaMalloc(ptr, size);
    return checkCudaError(error, "cudaMalloc") == cudaSuccess;
  }

  bool cudaSafeMemcpy(void *dst, const void *src, size_t size,
                      cudaMemcpyKind kind) {
    cudaError_t error = cudaMemcpy(dst, src, size, kind);
    return checkCudaError(error, "cudaMemcpy") == cudaSuccess;
  }

  void cudaSafeFree(void *ptr) {
    if (ptr) {
      cudaError_t error = cudaFree(ptr);
      checkCudaError(error, "cudaFree");
    }
  }

  template <typename Config, CP_METHOD Method = CP_METHOD::TMA>
  bool runTMATest() {
    const size_t chunk_size_bytes = Config::chunk_size * sizeof(float);
    const size_t total_bytes = num_elements * sizeof(float);
    const int repeat = 1; // Single pass for correctness testing

    // Allocate device memory
    uint8_t *d_input = nullptr;
    unsigned long long *d_output = nullptr;

    // Copy input data to device (cast to uint8_t)
    EXPECT_TRUE(cudaSafeMalloc((void **)&d_input, data_size_bytes));
    EXPECT_TRUE(cudaSafeMemcpy(d_input, h_input.data(), data_size_bytes,
                               cudaMemcpyHostToDevice));

    // Calculate shared memory size needed
    const size_t shared_mem_size = Config::stages * chunk_size_bytes;

    // Launch kernel with enough threads for all warps
    const int threads_per_block = Config::total_warps * 32;
    int num_sms = 0;
    auto error =
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    EXPECT_EQ(error, cudaSuccess)
        << "Failed to get SM count: " << cudaGetErrorString(error);
    printf("Detected %d SMs on GPU\n", num_sms);
    const int blocks = num_sms;
    // const int blocks = 1;
    EXPECT_TRUE(cudaSafeMalloc((void **)&d_output,
                               sizeof(unsigned long long) * blocks));

    cp_bw_kernel<Config::stages, chunk_size_bytes, repeat,
                 Config::num_producers, Config::num_consumers, Method>
        <<<blocks, threads_per_block, shared_mem_size>>>(d_input, d_output,
                                                         total_bytes);

    cudaError_t kernel_error = cudaGetLastError();
    EXPECT_EQ(kernel_error, cudaSuccess)
        << "Kernel launch failed: " << cudaGetErrorString(kernel_error);

    cudaError_t sync_error = cudaDeviceSynchronize();
    EXPECT_EQ(sync_error, cudaSuccess)
        << "Kernel execution failed: " << cudaGetErrorString(sync_error);

    // Copy result back to host
    std::vector<unsigned long long> h_output(blocks);
    EXPECT_TRUE(cudaSafeMemcpy(h_output.data(), d_output,
                               sizeof(unsigned long long) * blocks,
                               cudaMemcpyDeviceToHost));

    // Sum all the block results.
    unsigned long long h_result = 0;
    for (const auto &val : h_output) {
      h_result += val;
    }

    // The cp_bw_kernel sums the first uint32_t of each chunk
    // For verification, we need to compute the expected result differently
    unsigned long long expected_result = computeExpectedSum<Config>();

    // Verify exact match for unsigned long long results
    EXPECT_EQ(h_result, expected_result)
        << "TMA test failed. Expected: " << expected_result
        << ", Got: " << h_result << ", Method: " << static_cast<int>(Method)
        << ", Config: stages=" << Config::stages
        << ", producers=" << Config::num_producers
        << ", consumers=" << Config::num_consumers
        << ", chunk_size=" << Config::chunk_size;

    // Cleanup
    cudaSafeFree(d_input);
    cudaSafeFree(d_output);

    return sync_error == cudaSuccess && kernel_error == cudaSuccess;
  }

  // Compute expected result based on how cp_bw_kernel processes data.
  // With repeat=1, each unique chunk is processed exactly once across all blocks.
  // The kernel sums the first uint32_t of each chunk's data.
  template <typename Config>
  unsigned long long computeExpectedSum() {
    const size_t chunk_size_bytes = Config::chunk_size * sizeof(float);
    const size_t total_chunks = data_size_bytes / chunk_size_bytes;

    unsigned long long expected_sum = 0;
    const uint32_t *data_as_uint32 =
        reinterpret_cast<const uint32_t *>(h_input.data());

    // Each chunk is processed exactly once. Sum the first uint32 of each chunk.
    for (size_t chunk = 0; chunk < total_chunks; ++chunk) {
      size_t offset_in_uint32 = (chunk * chunk_size_bytes) / sizeof(uint32_t);
      if (offset_in_uint32 < num_elements) {
        expected_sum += data_as_uint32[offset_in_uint32];
      }
    }

    return expected_sum;
  }

  // Test data
  size_t num_elements;
  size_t data_size_bytes;
  std::vector<int> h_input;
};

// Basic TMA test with minimal configuration
// Note: TMA requires at least 2 stages due to barrier parity synchronization
TEST_F(TmaProducerConsumerTest, BasicTwoStageProducerConsumer) {
  using Config = TMAConfig<2, 1, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multi-stage test
TEST_F(TmaProducerConsumerTest, MultiStageProducerConsumer) {
  using Config = TMAConfig<4, 1, 1, 512>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multiple producers test
TEST_F(TmaProducerConsumerTest, MultipleProducers) {
  using Config = TMAConfig<2, 2, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multiple consumers test
TEST_F(TmaProducerConsumerTest, MultipleConsumers) {
  using Config = TMAConfig<2, 1, 2, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Complex configuration with multiple producers and consumers
TEST_F(TmaProducerConsumerTest, MultipleProducersAndConsumers) {
  using Config =
      TMAConfig<4, 2, 2, 512>; // 4 stages is divisible by 2 producers
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Large chunk size test
TEST_F(TmaProducerConsumerTest, LargeChunkSize) {
  using Config = TMAConfig<2, 1, 1, 1024>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Stress test with many stages and workers
TEST_F(TmaProducerConsumerTest, StressTest) {
  using Config =
      TMAConfig<6, 3, 3, 256>; // 6 stages is divisible by 3 producers
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Test different copy methods
// Not supported yet in the simulator
// TEST_F(TmaProducerConsumerTest, CPAsyncMethod) {
//   using Config = TMAConfig<2, 1, 1, 256>;
//   bool result = runTMATest<Config, CP_METHOD::CP_ASYNC>();
//   ASSERT_TRUE(result);
// }
