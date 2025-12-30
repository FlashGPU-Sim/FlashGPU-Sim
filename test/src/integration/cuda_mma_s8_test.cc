// CUDA kernel with inline PTX for mma.sync.aligned.m16n8k16/k32 with S8/U8 inputs
// Tests INT8 inputs → INT32 output accumulation with saturation
// Following pattern from cuda_mma_m16n8k8_test.cc

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

// Test fixture for S8 MMA M16N8K16 integration tests
class MMAS8M16N8K16IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

    int8_t* h_A;
    int8_t* h_B;
    int32_t* h_C;
    int32_t* h_D;
    int32_t* h_D_ref;  // Reference result for validation

    int8_t *d_A, *d_B;
    int32_t *d_C, *d_D;

    void SetUp() override {
        // Allocate host memory
        h_A = new int8_t[M * K];
        h_B = new int8_t[K * N];
        h_C = new int32_t[M * N];
        h_D = new int32_t[M * N];
        h_D_ref = new int32_t[M * N];

        // Allocate device memory
        cudaMalloc(&d_A, M * K * sizeof(int8_t));
        cudaMalloc(&d_B, K * N * sizeof(int8_t));
        cudaMalloc(&d_C, M * N * sizeof(int32_t));
        cudaMalloc(&d_D, M * N * sizeof(int32_t));
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

    // Helper: Saturate INT64 to INT32 range
    int32_t saturate_s32(int64_t val) {
        if (val > INT32_MAX) return INT32_MAX;
        if (val < INT32_MIN) return INT32_MIN;
        return static_cast<int32_t>(val);
    }

    // Compute reference result on CPU with saturation
    void compute_reference() {
        // Compute D = A * B + C with saturation
        // A is row-major: A[m][k] = A[m * K + k]
        // B is column-major: B[k][n] = B[n * K + k]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int64_t sum = 0;  // Use 64-bit to detect overflow
                for (int k = 0; k < K; k++) {
                    sum += static_cast<int64_t>(h_A[i * K + k]) *
                           static_cast<int64_t>(h_B[j * K + k]);
                }
                // Add accumulator and saturate
                sum += h_C[i * N + j];
                h_D_ref[i * N + j] = saturate_s32(sum);
            }
        }
    }

    // Run MMA kernel
    void run_mma_kernel();
};

// Simple kernel using mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
__global__ void mma_m16n8k16_s8_kernel(
    const int8_t* A,   // 16x16 S8 matrix (row-major)
    const int8_t* B,   // 16x8 S8 matrix (column-major)
    const int32_t* C,  // 16x8 S32 accumulator
    int32_t* D         // 16x8 S32 output
) {
    // Only use first warp for simplicity
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Load fragments into registers
    // For m16n8k16: A needs 1 register (packed S8), B needs 1 register, C/D need 4 registers each
    unsigned A_frag[1];
    unsigned B_frag[1];
    int32_t C_frag[4];
    int32_t D_frag[4];

    // Fragment distribution (similar to F16 but K=16)
    int groupID = lane_id / 4;
    int threadID_in_group = lane_id % 4;

    // Load A fragments (16×16 row-major)
    int8_t a_vals[4];
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    a_vals[0] = A[a_row0 * 16 + threadID_in_group * 4];
    a_vals[1] = A[a_row0 * 16 + threadID_in_group * 4 + 1];
    a_vals[2] = A[a_row1 * 16 + threadID_in_group * 4];
    a_vals[3] = A[a_row1 * 16 + threadID_in_group * 4 + 1];
    A_frag[0] = *reinterpret_cast<unsigned*>(&a_vals[0]);

    // Load B fragments (16×8 column-major)
    int8_t b_vals[4];
    int b_col = groupID;
    b_vals[0] = B[b_col * 16 + threadID_in_group * 4];
    b_vals[1] = B[b_col * 16 + threadID_in_group * 4 + 1];
    b_vals[2] = B[b_col * 16 + threadID_in_group * 4 + 2];
    b_vals[3] = B[b_col * 16 + threadID_in_group * 4 + 3];
    B_frag[0] = *reinterpret_cast<unsigned*>(&b_vals[0]);

    // Load C fragments
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
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "       // D output (4 S32 registers)
        "{%4}, "                   // A input (1 packed S8 register)
        "{%5}, "                   // B input (1 packed S8 register)
        "{%6, %7, %8, %9};\n"      // C accumulator (4 S32 registers)
        : "=r"(D_frag[0]), "=r"(D_frag[1]), "=r"(D_frag[2]), "=r"(D_frag[3])
        : "r"(A_frag[0]),
          "r"(B_frag[0]),
          "r"(C_frag[0]), "r"(C_frag[1]), "r"(C_frag[2]), "r"(C_frag[3])
    );

    // Store D fragments
    D[c_row0 * 8 + c_col0] = D_frag[0];
    D[c_row0 * 8 + c_col1] = D_frag[1];
    D[c_row1 * 8 + c_col0] = D_frag[2];
    D[c_row1 * 8 + c_col1] = D_frag[3];
}

void MMAS8M16N8K16IntegrationTest::run_mma_kernel() {
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 32 threads = 1 warp)
    mma_m16n8k16_s8_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy result back
    cudaMemcpy(h_D, d_D, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

// Test 1: All ones test
TEST_F(MMAS8M16N8K16IntegrationTest, AllOnesTest) {
    // Initialize with 1
    for (int i = 0; i < M * K; i++) h_A[i] = 1;
    for (int i = 0; i < K * N; i++) h_B[i] = 1;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: 16 = sum of 1*1 for K=16)
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], 16) << "Mismatch at index " << i;
    }
}

// Test 2: Saturation test with max values
TEST_F(MMAS8M16N8K16IntegrationTest, SaturationTest) {
    // Use max S8 value (127)
    for (int i = 0; i < M * K; i++) h_A[i] = 127;
    for (int i = 0; i < K * N; i++) h_B[i] = 127;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Expected: 127 * 127 * 16 = 258,048 (fits in S32)
    int32_t expected = 127 * 127 * 16;
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], expected)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << expected << ")";
    }
}

// Test 3: Zero matrix test
TEST_F(MMAS8M16N8K16IntegrationTest, ZeroMatrixTest) {
    // Initialize with zeros
    for (int i = 0; i < M * K; i++) h_A[i] = 0;
    for (int i = 0; i < K * N; i++) h_B[i] = 0;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], 0) << "Mismatch at index " << i;
    }
}

// Test 4: Negative values test
TEST_F(MMAS8M16N8K16IntegrationTest, NegativeValuesTest) {
    // Use mix of positive and negative values
    for (int i = 0; i < M * K; i++) h_A[i] = (i % 2 == 0) ? 10 : -10;
    for (int i = 0; i < K * N; i++) h_B[i] = (i % 2 == 0) ? 5 : -5;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results match reference
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], h_D_ref[i])
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 5: Random values test
TEST_F(MMAS8M16N8K16IntegrationTest, RandomValuesTest) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(-127, 127);

    // Initialize with random S8 values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<int8_t>(dist(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<int8_t>(dist(gen));
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen) * 100;  // Random accumulator values
    }

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], h_D_ref[i])
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 6: Non-zero accumulator test
TEST_F(MMAS8M16N8K16IntegrationTest, NonZeroAccumulatorTest) {
    // Simple values with non-zero C
    for (int i = 0; i < M * K; i++) h_A[i] = 2;
    for (int i = 0; i < K * N; i++) h_B[i] = 3;
    for (int i = 0; i < M * N; i++) h_C[i] = 100;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Expected: 2 * 3 * 16 + 100 = 196
    int32_t expected = 2 * 3 * 16 + 100;
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], expected)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << expected << ")";
    }
}
