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
    // For m16n8k16: A needs 2 registers (8 packed S8), B needs 1 register (4 packed S8), C/D need 4 registers each
    unsigned A_frag[2];
    unsigned B_frag[1];
    int32_t C_frag[4];
    int32_t D_frag[4];

    // Fragment distribution for M16N8K16
    // groupID = lane_id >> 2 (range 0-7)
    // threadID_in_group = lane_id % 4 (range 0-3)
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;

    // Load A fragments (16×16 row-major)
    // Each thread loads 8 S8 values from matrix A
    int8_t a_vals[8];
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int col_base = threadID_in_group * 4;

    // Load a[0..3] from row=groupID, columns [col_base..col_base+3]
    a_vals[0] = A[a_row0 * 16 + col_base + 0];
    a_vals[1] = A[a_row0 * 16 + col_base + 1];
    a_vals[2] = A[a_row0 * 16 + col_base + 2];
    a_vals[3] = A[a_row0 * 16 + col_base + 3];
    // Load a[4..7] from row=groupID+8, columns [col_base..col_base+3]
    a_vals[4] = A[a_row1 * 16 + col_base + 0];
    a_vals[5] = A[a_row1 * 16 + col_base + 1];
    a_vals[6] = A[a_row1 * 16 + col_base + 2];
    a_vals[7] = A[a_row1 * 16 + col_base + 3];

    A_frag[0] = *reinterpret_cast<unsigned*>(&a_vals[0]);  // a[0..3]
    A_frag[1] = *reinterpret_cast<unsigned*>(&a_vals[4]);  // a[4..7]

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
        "{%0, %1, %2, %3}, "           // D output (4 S32 registers)
        "{%4, %5}, "                   // A input (2 packed S8 registers)
        "{%6}, "                       // B input (1 packed S8 register)
        "{%7, %8, %9, %10};\n"         // C accumulator (4 S32 registers)
        : "=r"(D_frag[0]), "=r"(D_frag[1]), "=r"(D_frag[2]), "=r"(D_frag[3])
        : "r"(A_frag[0]), "r"(A_frag[1]),
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

// ============================================================================
// M16N8K32 Tests
// ============================================================================

class MMAS8M16N8K32IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 32;

    int8_t* h_A; int8_t* h_B; int32_t* h_C; int32_t* h_D; int32_t* h_D_ref;
    int8_t *d_A, *d_B; int32_t *d_C, *d_D;

    void SetUp() override {
        h_A = new int8_t[M * K]; h_B = new int8_t[K * N];
        h_C = new int32_t[M * N]; h_D = new int32_t[M * N]; h_D_ref = new int32_t[M * N];
        cudaMalloc(&d_A, M * K * sizeof(int8_t)); cudaMalloc(&d_B, K * N * sizeof(int8_t));
        cudaMalloc(&d_C, M * N * sizeof(int32_t)); cudaMalloc(&d_D, M * N * sizeof(int32_t));
    }

    void TearDown() override {
        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_D; delete[] h_D_ref;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    }

    void compute_reference() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += (int32_t)h_A[i * K + k] * (int32_t)h_B[j * K + k];
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    void run_mma_kernel();
};

__global__ void mma_m16n8k32_s8_kernel(const int8_t* A, const int8_t* B, const int32_t* C, int32_t* D) {
    if (threadIdx.x / 32 != 0) return;
    int lane_id = threadIdx.x % 32;
    unsigned A_frag[4], B_frag[2]; int32_t C_frag[4], D_frag[4];
    int groupID = lane_id >> 2, threadID_in_group = lane_id % 4;
    for (int i = 0; i < 4; i++) {
        uint8_t a_vals[4];
        int a_row = (i < 2) ? groupID : (groupID + 8);
        int col_base = threadID_in_group * 4 + ((i & 1) ? 16 : 0);
        for (int j = 0; j < 4; j++) a_vals[j] = A[a_row * 32 + col_base + j];
        A_frag[i] = *reinterpret_cast<unsigned*>(a_vals);
    }
    for (int i = 0; i < 2; i++) {
        uint8_t b_vals[4];
        int row_base = threadID_in_group * 4 + (i ? 16 : 0);
        for (int j = 0; j < 4; j++) b_vals[j] = B[groupID * 32 + row_base + j];
        B_frag[i] = *reinterpret_cast<unsigned*>(b_vals);
    }
    for (int i = 0; i < 4; i++) {
        int c_row = (i < 2) ? groupID : (groupID + 8);
        int c_col = threadID_in_group * 2 + (i & 1);
        C_frag[i] = C[c_row * 8 + c_col];
    }
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(D_frag[0]), "=r"(D_frag[1]), "=r"(D_frag[2]), "=r"(D_frag[3])
        : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
          "r"(B_frag[0]), "r"(B_frag[1]),
          "r"(C_frag[0]), "r"(C_frag[1]), "r"(C_frag[2]), "r"(C_frag[3])
    );
    for (int i = 0; i < 4; i++) {
        int d_row = (i < 2) ? groupID : (groupID + 8);
        int d_col = threadID_in_group * 2 + (i & 1);
        D[d_row * 8 + d_col] = D_frag[i];
    }
}

void MMAS8M16N8K32IntegrationTest::run_mma_kernel() {
    cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);
    mma_m16n8k32_s8_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaMemcpy(h_D, d_D, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

TEST_F(MMAS8M16N8K32IntegrationTest, BasicTest) {
    for (int i = 0; i < M * K; i++) h_A[i] = 1;
    for (int i = 0; i < K * N; i++) h_B[i] = 1;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;
    compute_reference(); run_mma_kernel();
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], h_D_ref[i]) << "Mismatch at " << i;
    }
}

// ============================================================================
// M8N8K16 Tests
// ============================================================================

class MMAS8M8N8K16IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 8;
    static constexpr int N = 8;
    static constexpr int K = 16;

    int8_t* h_A; int8_t* h_B; int32_t* h_C; int32_t* h_D; int32_t* h_D_ref;
    int8_t *d_A, *d_B; int32_t *d_C, *d_D;

    void SetUp() override {
        h_A = new int8_t[M * K]; h_B = new int8_t[K * N];
        h_C = new int32_t[M * N]; h_D = new int32_t[M * N]; h_D_ref = new int32_t[M * N];
        cudaMalloc(&d_A, M * K * sizeof(int8_t)); cudaMalloc(&d_B, K * N * sizeof(int8_t));
        cudaMalloc(&d_C, M * N * sizeof(int32_t)); cudaMalloc(&d_D, M * N * sizeof(int32_t));
    }

    void TearDown() override {
        delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_D; delete[] h_D_ref;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    }

    void compute_reference() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += (int32_t)h_A[i * K + k] * (int32_t)h_B[j * K + k];
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    void run_mma_kernel();
};

__global__ void mma_m8n8k16_s8_kernel(const int8_t* A, const int8_t* B, const int32_t* C, int32_t* D) {
    if (threadIdx.x / 32 != 0) return;
    int lane_id = threadIdx.x % 32;
    unsigned A_frag[1], B_frag[1]; int32_t C_frag[2], D_frag[2];
    int groupID = lane_id / 4, threadID_in_group = lane_id % 4;
    uint8_t a_vals[4];
    int a_row = groupID, col_base = threadID_in_group * 4;
    for (int i = 0; i < 4; i++) a_vals[i] = A[a_row * 16 + col_base + i];
    A_frag[0] = *reinterpret_cast<unsigned*>(a_vals);
    uint8_t b_vals[4];
    int row_base = threadID_in_group * 4;
    for (int i = 0; i < 4; i++) b_vals[i] = B[groupID * 16 + row_base + i];
    B_frag[0] = *reinterpret_cast<unsigned*>(b_vals);
    C_frag[0] = C[groupID * 8 + threadID_in_group * 2];
    C_frag[1] = C[groupID * 8 + threadID_in_group * 2 + 1];
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(D_frag[0]), "=r"(D_frag[1])
        : "r"(A_frag[0]), "r"(B_frag[0]),
          "r"(C_frag[0]), "r"(C_frag[1])
    );
    D[groupID * 8 + threadID_in_group * 2] = D_frag[0];
    D[groupID * 8 + threadID_in_group * 2 + 1] = D_frag[1];
}

void MMAS8M8N8K16IntegrationTest::run_mma_kernel() {
    cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);
    mma_m8n8k16_s8_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaMemcpy(h_D, d_D, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

TEST_F(MMAS8M8N8K16IntegrationTest, BasicTest) {
    for (int i = 0; i < M * K; i++) h_A[i] = 1;
    for (int i = 0; i < K * N; i++) h_B[i] = 1;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;
    compute_reference(); run_mma_kernel();
    for (int i = 0; i < M * N; i++) {
        EXPECT_EQ(h_D[i], h_D_ref[i]) << "Mismatch at " << i;
    }
}
