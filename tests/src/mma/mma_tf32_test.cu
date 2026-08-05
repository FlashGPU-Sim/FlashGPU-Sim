// CUDA kernel with inline PTX for mma.sync.aligned.m16n8k4 with TF32 inputs
// Tests TF32 inputs → F32 output accumulation
// TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (19 bits total)
// M16N8K4 is supported on sm90 (Hopper)

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

// TF32 rounding helper (CPU-side)
// TF32 has 10-bit mantissa, F32 has 23-bit mantissa
// Round by truncating lower 13 bits of F32 mantissa
inline float tf32_round(float f32) {
    uint32_t u32 = *reinterpret_cast<uint32_t*>(&f32);
    // TF32: keep sign (1 bit) + exponent (8 bits) + mantissa (10 bits)
    // Truncate lower 13 bits of F32's 23-bit mantissa
    u32 &= 0xFFFFE000;  // Clear lower 13 bits
    return *reinterpret_cast<float*>(&u32);
}

// Test fixture for TF32 MMA M16N8K4 integration tests
class MMATF32M16N8K4IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 4;

    float* h_A;  // TF32 (stored as F32, rounded to TF32 precision)
    float* h_B;  // TF32
    float* h_C;  // F32
    float* h_D;  // F32
    float* h_D_ref; // Reference result for validation

    float *d_A, *d_B, *d_C, *d_D;

    void SetUp() override {
        // Allocate host memory
        h_A = new float[M * K];
        h_B = new float[K * N];
        h_C = new float[M * N];
        h_D = new float[M * N];
        h_D_ref = new float[M * N];

        // Allocate device memory
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_D, M * N * sizeof(float));
    }

    void TearDown() override {
        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_D;
        delete[] h_D_ref;

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
    }

    // Compute reference result on CPU with TF32 rounding
    void compute_reference() {
        // Compute D = A * B + C with TF32 precision
        // A is row-major: A[m][k] = A[m * K + k]
        // B is column-major: B[k][n] = B[n * K + k]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    // Apply TF32 rounding to inputs
                    float a_tf32 = tf32_round(h_A[i * K + k]);
                    float b_tf32 = tf32_round(h_B[j * K + k]);
                    sum += a_tf32 * b_tf32;
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    // Run MMA kernel
    void run_mma_kernel();
};

// Simple kernel using mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
__global__ void mma_m16n8k4_tf32_kernel(
    const float* A,  // 16x4 TF32 matrix (row-major, stored as F32)
    const float* B,  // 4x8 TF32 matrix (column-major, stored as F32)
    const float* C,  // 16x8 F32 accumulator
    float* D         // 16x8 F32 output
) {
    // Only use first warp for simplicity
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Load fragments into registers
    // For m16n8k4 TF32: A needs 2 registers, B needs 1 register, C/D need 4 registers each
    // TF32 A and B use b32 (uint32_t), C/D use f32
    uint32_t A_frag[2];
    uint32_t B_frag[1];
    float C_frag[4];
    float D_frag[4];

    // Fragment distribution for M16N8K4
    // groupID = lane_id >> 2 (range 0-7)
    // threadID_in_group = lane_id % 4 (range 0-3)
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;

    // Load A fragments (16×4 row-major)
    // Each thread loads 2 TF32 values in one column
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col = threadID_in_group;

    A_frag[0] = *reinterpret_cast<const uint32_t*>(&A[a_row0 * 4 + a_col]);
    A_frag[1] = *reinterpret_cast<const uint32_t*>(&A[a_row1 * 4 + a_col]);

    // Load B fragments (4×8 column-major)
    // Each thread loads 1 TF32 value
    int b_row = threadID_in_group;
    int b_col = groupID;
    B_frag[0] = *reinterpret_cast<const uint32_t*>(&B[b_col * 4 + b_row]);

    // Load C fragments (16×8 row-major output)
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;
    C_frag[0] = C[c_row0 * 8 + c_col0];
    C_frag[1] = C[c_row0 * 8 + c_col1];
    C_frag[2] = C[c_row1 * 8 + c_col0];
    C_frag[3] = C[c_row1 * 8 + c_col1];

    // Execute MMA instruction using inline PTX
    // TF32 uses b32 registers (r constraint) for A and B
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "           // D output (4 F32 registers)
        "{%4, %5}, "                   // A input (2 b32 registers)
        "{%6}, "                       // B input (1 b32 register)
        "{%7, %8, %9, %10};\n"         // C accumulator (4 F32 registers)
        : "=f"(D_frag[0]), "=f"(D_frag[1]), "=f"(D_frag[2]), "=f"(D_frag[3])
        : "r"(A_frag[0]), "r"(A_frag[1]),
          "r"(B_frag[0]),
          "f"(C_frag[0]), "f"(C_frag[1]), "f"(C_frag[2]), "f"(C_frag[3])
    );

    // Store D fragments
    D[c_row0 * 8 + c_col0] = D_frag[0];
    D[c_row0 * 8 + c_col1] = D_frag[1];
    D[c_row1 * 8 + c_col0] = D_frag[2];
    D[c_row1 * 8 + c_col1] = D_frag[3];
}

void MMATF32M16N8K4IntegrationTest::run_mma_kernel() {
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 32 threads = 1 warp)
    mma_m16n8k4_tf32_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy result back
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Test 1: All ones test
TEST_F(MMATF32M16N8K4IntegrationTest, AllOnesTest) {
    // Initialize with 1.0
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: 4.0 = sum of 1*1 for K=4)
    // TF32 tolerance: 1e-6f for exact values
    // Small integers like 4.0 are exactly representable in TF32
    float tolerance = 1e-6f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 4.0f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 2: Zero matrix test
TEST_F(MMATF32M16N8K4IntegrationTest, ZeroMatrixTest) {
    // Initialize with zeros
    for (int i = 0; i < M * K; i++) h_A[i] = 0.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 0.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results
    // TF32 tolerance: 1e-6f for zero test (exact representation)
    float tolerance = 1e-6f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 0.0f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 3: Precision test - verify TF32 truncation
TEST_F(MMATF32M16N8K4IntegrationTest, PrecisionTest) {
    // Use values that show TF32 precision loss
    // TF32 has 10-bit mantissa (vs F32's 23-bit)
    // Use values that differ in lower 13 bits
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f + (i * 0.0001f);  // Small increments
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f + (i * 0.0001f);
    }
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference with TF32 rounding
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // TF32 tolerance: 1e-3f for precision test
    // Accounts for 10-bit mantissa truncation (vs F32's 23-bit)
    // Values with fine precision differences test rounding behavior
    float tolerance = 1e-3f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 4: Random values test
TEST_F(MMATF32M16N8K4IntegrationTest, RandomValuesTest) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = dist(gen);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = dist(gen);
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen);
    }

    // Compute reference with TF32 rounding
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // TF32 tolerance: 1e-2f for random values accounts for:
    // - 10-bit mantissa precision (vs F32's 23-bit)
    // - Accumulation errors over K=4 multiply-adds
    // - Larger input range [-10, 10] amplifying rounding errors
    float tolerance = 1e-2f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// ========== M16N8K8 TF32 MMA Tests ==========

// Test fixture for TF32 MMA M16N8K8 integration tests
class MMATF32M16N8K8IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 8;

    float* h_A;  // TF32 (stored as F32, rounded to TF32 precision)
    float* h_B;  // TF32
    float* h_C;  // F32
    float* h_D;  // F32
    float* h_D_ref; // Reference result for validation

    float *d_A, *d_B, *d_C, *d_D;

    void SetUp() override {
        // Allocate host memory
        h_A = new float[M * K];
        h_B = new float[K * N];
        h_C = new float[M * N];
        h_D = new float[M * N];
        h_D_ref = new float[M * N];

        // Allocate device memory
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_D, M * N * sizeof(float));
    }

    void TearDown() override {
        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_D;
        delete[] h_D_ref;

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
    }

    // Compute reference result on CPU with TF32 rounding
    void compute_reference() {
        // Compute D = A * B + C with TF32 precision
        // A is row-major: A[m][k] = A[m * K + k]
        // B is column-major: B[k][n] = B[n * K + k]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    // Apply TF32 rounding to inputs
                    float a_tf32 = tf32_round(h_A[i * K + k]);
                    float b_tf32 = tf32_round(h_B[j * K + k]);
                    sum += a_tf32 * b_tf32;
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    // Run MMA kernel
    void run_mma_kernel();
};

// Kernel using mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
__global__ void mma_m16n8k8_tf32_kernel(
    const float* A,  // 16x8 TF32 matrix (row-major, stored as F32)
    const float* B,  // 8x8 TF32 matrix (column-major, stored as F32)
    const float* C,  // 16x8 F32 accumulator
    float* D         // 16x8 F32 output
) {
    // Only use first warp for simplicity
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Load fragments into registers
    // For m16n8k8 TF32: A needs 4 registers, B needs 2 registers, C/D need 4 registers each
    uint32_t A_frag[4];
    uint32_t B_frag[2];
    float C_frag[4];
    float D_frag[4];

    // Fragment distribution for M16N8K8
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;

    // Load A fragments (16×8 row-major)
    // Each thread loads 4 TF32 values
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col0 = threadID_in_group;
    int a_col1 = threadID_in_group + 4;

    A_frag[0] = *reinterpret_cast<const uint32_t*>(&A[a_row0 * 8 + a_col0]);
    A_frag[1] = *reinterpret_cast<const uint32_t*>(&A[a_row1 * 8 + a_col0]);
    A_frag[2] = *reinterpret_cast<const uint32_t*>(&A[a_row0 * 8 + a_col1]);
    A_frag[3] = *reinterpret_cast<const uint32_t*>(&A[a_row1 * 8 + a_col1]);

    // Load B fragments (8×8 column-major)
    // Each thread loads 2 TF32 values
    int b_row0 = threadID_in_group;
    int b_row1 = threadID_in_group + 4;
    int b_col = groupID;
    B_frag[0] = *reinterpret_cast<const uint32_t*>(&B[b_col * 8 + b_row0]);
    B_frag[1] = *reinterpret_cast<const uint32_t*>(&B[b_col * 8 + b_row1]);

    // Load C fragments (16×8 row-major output)
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;
    C_frag[0] = C[c_row0 * 8 + c_col0];
    C_frag[1] = C[c_row0 * 8 + c_col1];
    C_frag[2] = C[c_row1 * 8 + c_col0];
    C_frag[3] = C[c_row1 * 8 + c_col1];

    // Execute MMA instruction using inline PTX
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "           // D output (4 F32 registers)
        "{%4, %5, %6, %7}, "           // A input (4 b32 registers)
        "{%8, %9}, "                   // B input (2 b32 registers)
        "{%10, %11, %12, %13};\n"      // C accumulator (4 F32 registers)
        : "=f"(D_frag[0]), "=f"(D_frag[1]), "=f"(D_frag[2]), "=f"(D_frag[3])
        : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
          "r"(B_frag[0]), "r"(B_frag[1]),
          "f"(C_frag[0]), "f"(C_frag[1]), "f"(C_frag[2]), "f"(C_frag[3])
    );

    // Store D fragments
    D[c_row0 * 8 + c_col0] = D_frag[0];
    D[c_row0 * 8 + c_col1] = D_frag[1];
    D[c_row1 * 8 + c_col0] = D_frag[2];
    D[c_row1 * 8 + c_col1] = D_frag[3];
}

void MMATF32M16N8K8IntegrationTest::run_mma_kernel() {
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 32 threads = 1 warp)
    mma_m16n8k8_tf32_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy result back
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Test 1: All ones test
TEST_F(MMATF32M16N8K8IntegrationTest, AllOnesTest) {
    // Initialize with 1.0
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: 8.0 = sum of 1*1 for K=8)
    float tolerance = 1e-6f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 8.0f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 2: Zero matrix test
TEST_F(MMATF32M16N8K8IntegrationTest, ZeroMatrixTest) {
    // Initialize with zeros
    for (int i = 0; i < M * K; i++) h_A[i] = 0.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 0.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results
    float tolerance = 1e-6f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 0.0f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 3: Precision test - verify TF32 truncation
TEST_F(MMATF32M16N8K8IntegrationTest, PrecisionTest) {
    // Use values that show TF32 precision loss
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f + (i * 0.0001f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f + (i * 0.0001f);
    }
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference with TF32 rounding
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // TF32 tolerance for precision test with K=8
    // Accounts for 10-bit mantissa truncation (vs F32's 23-bit)
    // K=8 accumulation may amplify rounding differences
    float tolerance = 1e-4f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 4: Random values test
TEST_F(MMATF32M16N8K8IntegrationTest, RandomValuesTest) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = dist(gen);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = dist(gen);
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen);
    }

    // Compute reference with TF32 rounding
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // TF32 tolerance for random values with K=8
    // Increased tolerance accounts for:
    // - 10-bit mantissa precision (vs FP32's 23-bit)
    // - K=8 accumulation errors (8 multiply-adds per output element)
    // - Larger input range [-10, 10] amplifying rounding errors
    float tolerance = 1e-4f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}
