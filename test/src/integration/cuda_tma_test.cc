#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
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

// Test class for CUDA TMA functionality
class CudaTMATest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test with 1MB of data
    num_elements = 262144; // 1MB / 4 bytes per float
    data_size_bytes = num_elements * sizeof(float);

    // Allocate host memory
    h_input.resize(num_elements);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(1, 10);

    for (size_t i = 0; i < num_elements; ++i) {
      h_input[i] = dis(gen);
      //   h_input[i] = 2;
    }
  }

  void TearDown() override {
    // Cleanup handled by vectors and RAII
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

    auto start_time = std::chrono::high_resolution_clock::now();

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

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

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
        << ", chunk_size=" << Config::chunk_size
        << ", Duration: " << duration.count() << " μs";

    // Cleanup
    cudaSafeFree(d_input);
    cudaSafeFree(d_output);

    return sync_error == cudaSuccess && kernel_error == cudaSuccess;
  }

  // Compute expected result based on how cp_bw_kernel processes data
  template <typename Config> unsigned long long computeExpectedSum() {
    const size_t chunk_size_bytes = Config::chunk_size * sizeof(float);
    const size_t total_chunks = data_size_bytes / chunk_size_bytes;

    unsigned long long expected_sum = 0;
    const uint32_t *data_as_uint32 =
        reinterpret_cast<const uint32_t *>(h_input.data());

    // cp_bw_kernel sums the first uint32_t of each chunk
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

  static constexpr float TOLERANCE = 1e-5f;
};

// Basic TMA test with minimal configuration
TEST_F(CudaTMATest, BasicSingleStageProducerConsumer) {
  using Config = TMAConfig<1, 1, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multi-stage test
TEST_F(CudaTMATest, MultiStageProducerConsumer) {
  using Config = TMAConfig<4, 1, 1, 512>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multiple producers test
TEST_F(CudaTMATest, MultipleProducers) {
  using Config = TMAConfig<2, 2, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Multiple consumers test
TEST_F(CudaTMATest, MultipleConsumers) {
  using Config = TMAConfig<2, 1, 2, 256>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Complex configuration with multiple producers and consumers
TEST_F(CudaTMATest, MultipleProducersAndConsumers) {
  using Config =
      TMAConfig<4, 2, 2, 512>; // 4 stages is divisible by 2 producers
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Large chunk size test
TEST_F(CudaTMATest, LargeChunkSize) {
  using Config = TMAConfig<2, 1, 1, 1024>;
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Stress test with many stages and workers
TEST_F(CudaTMATest, StressTest) {
  using Config =
      TMAConfig<6, 3, 3, 256>; // 6 stages is divisible by 3 producers
  bool result = runTMATest<Config, CP_METHOD::TMA>();
  ASSERT_TRUE(result);
}

// Test different copy methods
TEST_F(CudaTMATest, CPAsyncMethod) {
  using Config = TMAConfig<2, 1, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::CP_ASYNC>();
  ASSERT_TRUE(result);
}

TEST_F(CudaTMATest, NormalLoadMethod) {
  using Config = TMAConfig<2, 1, 1, 256>;
  bool result = runTMATest<Config, CP_METHOD::NORMAL_LOAD>();
  ASSERT_TRUE(result);
}

TEST_F(CudaTMATest, NormalLoad4Producer4Consumer) {
  using Config = TMAConfig<4, 4, 4, 256>;
  bool result = runTMATest<Config, CP_METHOD::NORMAL_LOAD>();
  ASSERT_TRUE(result);
}

// Performance comparison test
TEST_F(CudaTMATest, PerformanceComparison) {
  const int num_iterations = 5; // Reduced for faster testing

  // Test different configurations and methods
  struct TestConfig {
    const char *name;
    std::function<bool()> test_func;
  };

  std::vector<TestConfig> configs = {
      {"TMA Single Stage",
       [this]() {
         return runTMATest<TMAConfig<1, 1, 1, 256>, CP_METHOD::TMA>();
       }},
      {"TMA Multi Stage",
       [this]() {
         return runTMATest<TMAConfig<4, 1, 1, 256>, CP_METHOD::TMA>();
       }},
      {"TMA Multi Producer",
       [this]() {
         return runTMATest<TMAConfig<2, 2, 1, 256>, CP_METHOD::TMA>();
       }},
      {"TMA Multi Consumer",
       [this]() {
         return runTMATest<TMAConfig<2, 1, 2, 256>, CP_METHOD::TMA>();
       }},
      {"CP_ASYNC Method",
       [this]() {
         return runTMATest<TMAConfig<2, 1, 1, 256>, CP_METHOD::CP_ASYNC>();
       }},
      {"Normal Load Method", [this]() {
         return runTMATest<TMAConfig<2, 1, 1, 256>, CP_METHOD::NORMAL_LOAD>();
       }}};

  for (const auto &config : configs) {
    auto start = std::chrono::high_resolution_clock::now();

    bool success = true;
    for (int i = 0; i < num_iterations; ++i) {
      success &= config.test_func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_TRUE(success) << "Performance test failed for config: "
                         << config.name;

    std::cout << config.name
              << " average time: " << (duration.count() / num_iterations)
              << " μs" << std::endl;
  }
}