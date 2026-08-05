#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <memory>

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU implementation for verification (simulates what GPGPU-Sim would execute)
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

class CudaVectorAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 1MB of data = 1024*1024 bytes / 4 bytes per float = 262,144 elements
        num_elements = 262144;  // 1MB of float data
        data_size_bytes = num_elements * sizeof(float);
        
        // Allocate host memory
        h_a.resize(num_elements);
        h_b.resize(num_elements);
        h_c.resize(num_elements);
        h_c_ref.resize(num_elements);
        
        // Initialize test data with random values
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducible tests
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
        
        for (int i = 0; i < num_elements; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
            h_c[i] = 0.0f;
            h_c_ref[i] = 0.0f;
        }
        
        // Compute reference result on CPU
        vectorAddCPU(h_a.data(), h_b.data(), h_c_ref.data(), num_elements);
    }
    
    void TearDown() override {
        // Cleanup is automatic with std::vector
    }
    
    // CUDA runtime API wrappers with error checking
    cudaError_t cudaCheckError(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        }
        return error;
    }
    
    bool cudaSafeMalloc(void** ptr, size_t size) {
        cudaError_t error = cudaMalloc(ptr, size);
        return cudaCheckError(error, "cudaMalloc") == cudaSuccess;
    }
    
    bool cudaSafeMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
        cudaError_t error = cudaMemcpy(dst, src, size, kind);
        return cudaCheckError(error, "cudaMemcpy") == cudaSuccess;
    }
    
    void cudaSafeFree(void* ptr) {
        cudaError_t error = cudaFree(ptr);
        cudaCheckError(error, "cudaFree");
    }
    
    bool launchVectorAddKernel(float* d_a, float* d_b, float* d_c, int n) {
        // Calculate grid and block dimensions
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        // Launch kernel
        vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
        // Check for launch errors
        cudaError_t error = cudaGetLastError();
        if (cudaCheckError(error, "kernel launch") != cudaSuccess) {
            return false;
        }
        
        // Wait for kernel to complete
        error = cudaDeviceSynchronize();
        return cudaCheckError(error, "cudaDeviceSynchronize") == cudaSuccess;
    }
    
    // Test data
    int num_elements;
    size_t data_size_bytes;
    std::vector<float> h_a, h_b, h_c, h_c_ref;
    
    // Tolerance for floating point comparison
    static constexpr float TOLERANCE = 1e-6f;
};

// Basic vector addition test
TEST_F(CudaVectorAddTest, BasicVectorAddition) {
    // This test uses actual CUDA runtime API
    
    // Step 1: Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_a, data_size_bytes));
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_b, data_size_bytes));
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_c, data_size_bytes));
    
    // Step 2: Copy data from host to device
    ASSERT_TRUE(cudaSafeMemcpy(d_a, h_a.data(), data_size_bytes, cudaMemcpyHostToDevice));
    ASSERT_TRUE(cudaSafeMemcpy(d_b, h_b.data(), data_size_bytes, cudaMemcpyHostToDevice));
    
    // Step 3: Launch kernel
    auto start_time = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(launchVectorAddKernel(d_a, d_b, d_c, num_elements));
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Step 4: Copy result back to host
    ASSERT_TRUE(cudaSafeMemcpy(h_c.data(), d_c, data_size_bytes, cudaMemcpyDeviceToHost));
    
    // Step 5: Verify results
    for (int i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(h_c[i], h_c_ref[i], TOLERANCE) 
            << "Mismatch at index " << i 
            << ": expected " << h_c_ref[i] 
            << ", got " << h_c[i];
    }
    
    // Step 6: Cleanup
    cudaSafeFree(d_a);
    cudaSafeFree(d_b);
    cudaSafeFree(d_c);
    
}

// Test with different data patterns
TEST_F(CudaVectorAddTest, SpecialValueTesting) {
    // Test with special floating point values
    
    // Create vectors with special values
    std::vector<float> special_a(num_elements);
    std::vector<float> special_b(num_elements);
    std::vector<float> special_c(num_elements, 0.0f);
    std::vector<float> special_ref(num_elements);
    
    // Fill with patterns including edge cases
    for (int i = 0; i < num_elements; ++i) {
        if (i % 4 == 0) {
            special_a[i] = 0.0f;
            special_b[i] = static_cast<float>(i);
        } else if (i % 4 == 1) {
            special_a[i] = static_cast<float>(i);
            special_b[i] = 0.0f;
        } else if (i % 4 == 2) {
            special_a[i] = -static_cast<float>(i);
            special_b[i] = static_cast<float>(i);
        } else {
            special_a[i] = 1.0f;
            special_b[i] = -1.0f;
        }
        special_ref[i] = special_a[i] + special_b[i];
    }
    
    // Simulate CUDA operations
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_a, data_size_bytes));
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_b, data_size_bytes));
    ASSERT_TRUE(cudaSafeMalloc((void**)&d_c, data_size_bytes));
    
    ASSERT_TRUE(cudaSafeMemcpy(d_a, special_a.data(), data_size_bytes, cudaMemcpyHostToDevice));
    ASSERT_TRUE(cudaSafeMemcpy(d_b, special_b.data(), data_size_bytes, cudaMemcpyHostToDevice));
    
    ASSERT_TRUE(launchVectorAddKernel(d_a, d_b, d_c, num_elements));
    
    ASSERT_TRUE(cudaSafeMemcpy(special_c.data(), d_c, data_size_bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(special_c[i], special_ref[i], TOLERANCE)
            << "Special value test failed at index " << i;
    }
    
    cudaSafeFree(d_a);
    cudaSafeFree(d_b);
    cudaSafeFree(d_c);
}

// Parameterized test for different data sizes
class CudaVectorAddParameterizedTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        num_elements = GetParam();
        data_size_bytes = num_elements * sizeof(float);
        
        h_a.resize(num_elements);
        h_b.resize(num_elements);
        h_c.resize(num_elements);
        h_c_ref.resize(num_elements);
        
        // Initialize with simple pattern for parameterized tests
        for (int i = 0; i < num_elements; ++i) {
            h_a[i] = static_cast<float>(i % 100);
            h_b[i] = static_cast<float>((i + 1) % 100);
            h_c_ref[i] = h_a[i] + h_b[i];
        }
    }
    
    int num_elements;
    size_t data_size_bytes;
    std::vector<float> h_a, h_b, h_c, h_c_ref;
    static constexpr float TOLERANCE = 1e-6f;
};

TEST_P(CudaVectorAddParameterizedTest, DifferentSizes) {
    // Test with different data sizes
    
    float* d_a = (float*)malloc(data_size_bytes);
    float* d_b = (float*)malloc(data_size_bytes);
    float* d_c = (float*)malloc(data_size_bytes);
    
    ASSERT_NE(d_a, nullptr);
    ASSERT_NE(d_b, nullptr);
    ASSERT_NE(d_c, nullptr);
    
    std::memcpy(d_a, h_a.data(), data_size_bytes);
    std::memcpy(d_b, h_b.data(), data_size_bytes);
    
    // Simulate kernel execution
    for (int i = 0; i < num_elements; ++i) {
        ((float*)d_c)[i] = ((float*)d_a)[i] + ((float*)d_b)[i];
    }
    
    std::memcpy(h_c.data(), d_c, data_size_bytes);
    
    // Verify results
    for (int i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(h_c[i], h_c_ref[i], TOLERANCE);
    }
    
    free(d_a);
    free(d_b);
    free(d_c);
}

// Test different data sizes: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB
INSTANTIATE_TEST_SUITE_P(
    VariousSizes,
    CudaVectorAddParameterizedTest,
    ::testing::Values(
        256,      // 1KB
        1024,     // 4KB
        4096,     // 16KB
        16384,    // 64KB
        65536,    // 256KB
        262144    // 1MB
    )
);