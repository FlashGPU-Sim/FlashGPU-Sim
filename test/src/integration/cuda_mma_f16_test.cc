// CUDA kernel with inline PTX for mma.sync.aligned.m16n8k8
// Based on NVIDIA CUTLASS official implementation style
// Tests F16 input → F32 output accumulation

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cmath>

// Test fixture for F16 MMA M16N8K8 integration tests
class MMAF16M16N8K8IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 8;

    uint16_t* h_A;
    uint16_t* h_B;
    float* h_C;
    float* h_D;
    float* h_D_ref;  // Reference result for validation

    uint16_t *d_A, *d_B;
    float *d_C, *d_D;

    void SetUp() override {
        // Allocate host memory
        h_A = new uint16_t[M * K];
        h_B = new uint16_t[K * N];
        h_C = new float[M * N];
        h_D = new float[M * N];
        h_D_ref = new float[M * N];

        // Allocate device memory
        cudaMalloc(&d_A, M * K * sizeof(uint16_t));
        cudaMalloc(&d_B, K * N * sizeof(uint16_t));
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

    // Helper: Convert F32 to F16
    uint16_t f32_to_f16(float f32) {
        uint32_t bits;
        memcpy(&bits, &f32, sizeof(float));

        uint32_t sign = (bits >> 31) & 0x1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t frac = bits & 0x7FFFFF;

        // Handle special cases
        if (exp > 15) {
            // Overflow to infinity
            return (sign << 15) | 0x7C00;
        }
        if (exp < -14) {
            // Underflow to zero or denormal
            if (exp < -24) return (sign << 15);
            // Denormal
            uint32_t denorm_frac = (0x800000 | frac) >> (-14 - exp);
            return (sign << 15) | (denorm_frac >> 13);
        }

        // Normalized F16
        uint32_t f16_exp = exp + 15;
        uint32_t f16_frac = frac >> 13;
        return (sign << 15) | (f16_exp << 10) | f16_frac;
    }

    // Helper: Convert F16 to F32
    float f16_to_f32(uint16_t f16) {
        uint32_t sign = (f16 >> 15) & 0x1;
        uint32_t exp = (f16 >> 10) & 0x1F;
        uint32_t frac = f16 & 0x3FF;

        if (exp == 0) {
            if (frac == 0) return sign ? -0.0f : 0.0f;
            // Denormal
            float result = frac / 1024.0f / 16384.0f;
            return sign ? -result : result;
        } else if (exp == 31) {
            // Inf or NaN
            return frac ? NAN : (sign ? -INFINITY : INFINITY);
        }

        // Normalized
        uint32_t f32_exp = exp - 15 + 127;
        uint32_t f32_frac = frac << 13;
        uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;

        float result;
        memcpy(&result, &f32_bits, sizeof(float));
        return result;
    }

    // Compute reference result on CPU
    void compute_reference() {
        // Convert F16 inputs to F32 for computation
        float A_f32[M * K];
        float B_f32[K * N];

        for (int i = 0; i < M * K; i++) {
            A_f32[i] = f16_to_f32(h_A[i]);
        }
        for (int i = 0; i < K * N; i++) {
            B_f32[i] = f16_to_f32(h_B[i]);
        }

        // Compute D = A * B + C
        // A is row-major: A[m][k] = A[m * K + k]
        // B is column-major: B[k][n] = B[n * K + k]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A_f32[i * K + k] * B_f32[j * K + k];  // B is column-major!
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    // Run MMA kernel
    void run_mma_kernel();
};

// Simple kernel using mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
// Each warp computes one 16x8 output tile
__global__ void mma_m16n8k8_f16_kernel(
    const uint16_t* A,  // 16x8 F16 matrix (row-major)
    const uint16_t* B,  // 8x8 F16 matrix (column-major)
    const float* C,     // 16x8 F32 accumulator
    float* D            // 16x8 F32 output
) {
    // Only use first warp for simplicity
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Load fragments into registers
    // For m16n8k8: A needs 2 registers (packed F16), B needs 1 register, C/D need 4 registers each
    unsigned A_frag[2];
    unsigned B_frag[1];
    float C_frag[4];
    float D_frag[4];

    // Proper NVIDIA tensor core fragment distribution
    // groupID = lane_id / 4 (0-7), threadID_in_group = lane_id % 4 (0-3)
    int groupID = lane_id / 4;
    int threadID_in_group = lane_id % 4;

    // Each thread produces 4 output elements at specific positions:
    // d[0]: row=groupID,   col=threadID_in_group*2
    // d[1]: row=groupID,   col=threadID_in_group*2+1
    // d[2]: row=groupID+8, col=threadID_in_group*2
    // d[3]: row=groupID+8, col=threadID_in_group*2+1

    // Load A fragments (16×8 row-major, A is M×K)
    // Official formula: row = groupID for a0,a1; groupID+8 for a2,a3
    //                   col = threadID_in_group * 2 + (i & 0x1) where i = {0,1,2,3}
    uint16_t a_vals[4];
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col0 = threadID_in_group * 2;
    int a_col1 = threadID_in_group * 2 + 1;
    // A is row-major: A[row][col] = A[row * K + col] where K=8
    a_vals[0] = A[a_row0 * 8 + a_col0];  // a0: row=groupID, col=threadID_in_group*2
    a_vals[1] = A[a_row0 * 8 + a_col1];  // a1: row=groupID, col=threadID_in_group*2+1
    a_vals[2] = A[a_row1 * 8 + a_col0];  // a2: row=groupID+8, col=threadID_in_group*2
    a_vals[3] = A[a_row1 * 8 + a_col1];  // a3: row=groupID+8, col=threadID_in_group*2+1
    A_frag[0] = *reinterpret_cast<unsigned*>(&a_vals[0]);
    A_frag[1] = *reinterpret_cast<unsigned*>(&a_vals[2]);

    // Load B fragments (8×8 column-major K×N, B is stored as B[n*K+k])
    // Official formula: row = (threadID_in_group * 2) + i, col = groupID, where i = {0, 1}
    // So thread gets B[row0][col] and B[row1][col] where row0=threadID_in_group*2, row1=row0+1
    uint16_t b_vals[2];
    int b_row0 = threadID_in_group * 2;
    int b_row1 = threadID_in_group * 2 + 1;
    int b_col = groupID;
    // B is column-major: B[n][k] stored as B[n*K + k]
    b_vals[0] = B[b_col * 8 + b_row0];  // B[col][row0]
    b_vals[1] = B[b_col * 8 + b_row1];  // B[col][row1]
    B_frag[0] = *reinterpret_cast<unsigned*>(&b_vals[0]);

    // Load C fragments - same positions as D output
    int c_row0 = groupID;
    int c_row1 = groupID + 8;
    int c_col0 = threadID_in_group * 2;
    int c_col1 = threadID_in_group * 2 + 1;
    C_frag[0] = C[c_row0 * 8 + c_col0];
    C_frag[1] = C[c_row0 * 8 + c_col1];
    C_frag[2] = C[c_row1 * 8 + c_col0];
    C_frag[3] = C[c_row1 * 8 + c_col1];

    // Execute MMA instruction using inline PTX
    // Syntax: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    // Based on NVIDIA CUTLASS mma_sm75.h
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "       // D output (4 F32 registers)
        "{%4, %5}, "               // A input (2 packed F16 registers)
        "{%6}, "                   // B input (1 packed F16 register)
        "{%7, %8, %9, %10};\n"     // C accumulator (4 F32 registers)
        : "=f"(D_frag[0]), "=f"(D_frag[1]), "=f"(D_frag[2]), "=f"(D_frag[3])
        : "r"(A_frag[0]), "r"(A_frag[1]),
          "r"(B_frag[0]),
          "f"(C_frag[0]), "f"(C_frag[1]), "f"(C_frag[2]), "f"(C_frag[3])
    );

    // Store D fragments - same positions as C
    D[c_row0 * 8 + c_col0] = D_frag[0];
    D[c_row0 * 8 + c_col1] = D_frag[1];
    D[c_row1 * 8 + c_col0] = D_frag[2];
    D[c_row1 * 8 + c_col1] = D_frag[3];
}

void MMAF16M16N8K8IntegrationTest::run_mma_kernel() {
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 32 threads = 1 warp)
    mma_m16n8k8_f16_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy result back
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Test 1: Simple case with all 1.0 values
TEST_F(MMAF16M16N8K8IntegrationTest, AllOnesTest) {
    // Initialize with 1.0 in F16 = 0x3C00
    for (int i = 0; i < M * K; i++) h_A[i] = 0x3C00;  // 1.0
    for (int i = 0; i < K * N; i++) h_B[i] = 0x3C00;  // 1.0
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: 8.0 = sum of 1.0*1.0 for K=8)
    for (int i = 0; i < M * N; i++) {
        EXPECT_FLOAT_EQ(h_D[i], 8.0f) << "Mismatch at index " << i;
    }
}

// Test 2: Zero matrix test
TEST_F(MMAF16M16N8K8IntegrationTest, ZeroMatrixTest) {
    // Initialize with zeros
    for (int i = 0; i < M * K; i++) h_A[i] = 0x0000;  // 0.0
    for (int i = 0; i < K * N; i++) h_B[i] = 0x0000;  // 0.0
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: all zeros)
    for (int i = 0; i < M * N; i++) {
        EXPECT_FLOAT_EQ(h_D[i], 0.0f) << "Mismatch at index " << i;
    }
}

// Test 3: Random values test
TEST_F(MMAF16M16N8K8IntegrationTest, RandomValuesTest) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize with random F16 values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = f32_to_f16(dist(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = f32_to_f16(dist(gen));
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen);
    }

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results with tolerance for FP16 precision
    for (int i = 0; i < M * N; i++) {
        // F16 dynamic tolerance: max(0.01, |expected| * 1%)
        // - Absolute: 0.01 for small values
        // - Relative: 1% of expected for large values
        // F16 has 10-bit mantissa, requiring larger tolerance than F32
        float tolerance = std::max(0.01f, std::abs(h_D_ref[i]) * 0.01f);
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 4: Random values with larger range
TEST_F(MMAF16M16N8K8IntegrationTest, RandomValuesLargeRangeTest) {
    // Random number generator with larger range
    std::random_device rd;
    std::mt19937 gen(123);  // Different seed
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Initialize with random F16 values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = f32_to_f16(dist(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = f32_to_f16(dist(gen));
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen);
    }

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results with larger tolerance for FP16 precision and larger values
    for (int i = 0; i < M * N; i++) {
        // F16 dynamic tolerance: max(0.1, |expected| * 2%)
        // Larger than previous test due to:
        // - Wider input range [-10, 10] amplifies rounding errors
        // - Accumulation errors over K=8 multiply-adds
        // - F16's 10-bit mantissa precision limits
        float tolerance = std::max(0.1f, std::abs(h_D_ref[i]) * 0.02f);
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test 5: Mixed positive and negative values
TEST_F(MMAF16M16N8K8IntegrationTest, MixedSignTest) {
    // Initialize with alternating positive/negative values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (i % 2 == 0) ? 0x3C00 : 0xBC00;  // +1.0 or -1.0
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (i % 2 == 0) ? 0x4000 : 0xC000;  // +2.0 or -2.0
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], 0.1f)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}

// Test fixture for F16 MMA M8N8K4 integration tests
// Per PTX ISA: F16 M8N8K4 computes 4 separate MMA operations
// This test validates ALL 4 separate computation outputs
class MMAF16M8N8K4IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 8;
    static constexpr int N = 8;
    static constexpr int K = 4;
    static constexpr int NUM_COMP = 4;

    // Host arrays: 4 separate matrices for each of the 4 computations
    uint16_t* h_A;  // [NUM_COMP][M][K] = 4 × 8×4 = 128 elements
    uint16_t* h_B;  // [NUM_COMP][K][N] = 4 × 4×8 = 128 elements
    float* h_C;     // [NUM_COMP][M][N] = 4 × 8×8 = 256 elements
    float* h_D;     // [NUM_COMP][M][N] = 4 × 8×8 = 256 elements
    float* h_D_ref; // [NUM_COMP][M][N] = 4 × 8×8 = 256 elements

    uint16_t *d_A, *d_B;
    float *d_C, *d_D;

    void SetUp() override {
        h_A = new uint16_t[NUM_COMP * M * K];
        h_B = new uint16_t[NUM_COMP * K * N];
        h_C = new float[NUM_COMP * M * N];
        h_D = new float[NUM_COMP * M * N];
        h_D_ref = new float[NUM_COMP * M * N];

        cudaMalloc(&d_A, NUM_COMP * M * K * sizeof(uint16_t));
        cudaMalloc(&d_B, NUM_COMP * K * N * sizeof(uint16_t));
        cudaMalloc(&d_C, NUM_COMP * M * N * sizeof(float));
        cudaMalloc(&d_D, NUM_COMP * M * N * sizeof(float));
    }

    void TearDown() override {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_D;
        delete[] h_D_ref;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
    }

    uint16_t f32_to_f16(float f32) {
        uint32_t bits;
        memcpy(&bits, &f32, sizeof(float));
        uint32_t sign = (bits >> 31) & 0x1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t frac = bits & 0x7FFFFF;

        if (exp > 15) return (sign << 15) | 0x7C00;
        if (exp < -14) {
            if (exp < -24) return (sign << 15);
            uint32_t denorm_frac = (0x800000 | frac) >> (-14 - exp);
            return (sign << 15) | (denorm_frac >> 13);
        }

        uint32_t f16_exp = exp + 15;
        uint32_t f16_frac = frac >> 13;
        return (sign << 15) | (f16_exp << 10) | f16_frac;
    }

    float f16_to_f32(uint16_t f16) {
        uint32_t sign = (f16 >> 15) & 0x1;
        uint32_t exp = (f16 >> 10) & 0x1F;
        uint32_t frac = f16 & 0x3FF;

        if (exp == 0) {
            if (frac == 0) return sign ? -0.0f : 0.0f;
            float result = frac / 1024.0f / 16384.0f;
            return sign ? -result : result;
        } else if (exp == 31) {
            return frac ? NAN : (sign ? -INFINITY : INFINITY);
        }

        uint32_t f32_exp = exp - 15 + 127;
        uint32_t f32_frac = frac << 13;
        uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;

        float result;
        memcpy(&result, &f32_bits, sizeof(float));
        return result;
    }

    void compute_reference() {
        // Compute reference for all 4 separate computations
        for (int comp = 0; comp < NUM_COMP; comp++) {
            float A_f32[M * K];
            float B_f32[K * N];

            int offset_A = comp * M * K;
            int offset_B = comp * K * N;
            int offset_C = comp * M * N;
            int offset_D = comp * M * N;

            for (int i = 0; i < M * K; i++) {
                A_f32[i] = f16_to_f32(h_A[offset_A + i]);
            }
            for (int i = 0; i < K * N; i++) {
                B_f32[i] = f16_to_f32(h_B[offset_B + i]);
            }

            // D[comp] = A[comp] * B[comp] + C[comp] (A is row-major, B is column-major)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A_f32[i * K + k] * B_f32[j * K + k];
                    }
                    h_D_ref[offset_D + i * N + j] = sum + h_C[offset_C + i * N + j];
                }
            }
        }
    }

    void run_mma_kernel();
};

// Kernel for F16 M8N8K4 with 4-computation decomposition
// This kernel validates that each of the 4 computations loads from separate arrays and produces separate outputs
__global__ void mma_m8n8k4_f16_kernel(
    const uint16_t* A,  // [4][8][4] F16 matrices (row-major)
    const uint16_t* B,  // [4][4][8] F16 matrices (column-major)
    const float* C,     // [4][8][8] F32 accumulators
    float* D            // [4][8][8] F32 outputs
) {
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Determine which of the 4 MMA computations this thread participates in
    int computation_idx = (lane_id % 16) / 4;  // 0-3

    // Compute offsets for this computation's matrices
    constexpr int M = 8, N = 8, K = 4;
    int offset_A = computation_idx * M * K;
    int offset_B = computation_idx * K * N;
    int offset_C = computation_idx * M * N;
    int offset_D = computation_idx * M * N;

    // Fragment distribution per corrected plan:
    // Each thread holds: 4 F16 values in A (2 regs), 4 F16 values in B (2 regs), 8 F32 values in C/D (8 regs)
    unsigned A_frag[2];
    unsigned B_frag[2];
    float C_frag[8];
    float D_frag[8];

    // Load A fragments from this computation's matrix (row layout):
    // row = lane_id % 4          if lane_id < 16
    //       (lane_id % 4) + 4    otherwise
    // col = i                    for a[i] where i = {0,1,2,3}
    uint16_t a_vals[4];
    int a_row_base = (lane_id % 4) + (lane_id >= 16 ? 4 : 0);
    for (int i = 0; i < 4; i++) {
        a_vals[i] = A[offset_A + a_row_base * 4 + i];
    }
    A_frag[0] = *reinterpret_cast<unsigned*>(&a_vals[0]);
    A_frag[1] = *reinterpret_cast<unsigned*>(&a_vals[2]);

    // Load B fragments from this computation's matrix (col layout):
    // row = i                    for b[i] where i = {0,1,2,3}
    // col = lane_id % 4          if lane_id < 16
    //       (lane_id % 4) + 4    otherwise
    uint16_t b_vals[4];
    int b_col_base = (lane_id % 4) + (lane_id >= 16 ? 4 : 0);
    for (int i = 0; i < 4; i++) {
        b_vals[i] = B[offset_B + b_col_base * 4 + i];
    }
    B_frag[0] = *reinterpret_cast<unsigned*>(&b_vals[0]);
    B_frag[1] = *reinterpret_cast<unsigned*>(&b_vals[2]);

    // Load C fragments from this computation's accumulator (accumulator distribution):
    // row = X                    if lane_id < 16
    //       X + 4                otherwise
    //       where X = (lane_id & 0b1) + (i & 0b10)  for c[i] where i = {0,...,7}
    // col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1)
    for (int i = 0; i < 8; i++) {
        int X = (lane_id & 0b1) + (i & 0b10);
        int row = X + (lane_id >= 16 ? 4 : 0);
        int col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1);
        C_frag[i] = C[offset_C + row * 8 + col];
    }

    // Execute MMA instruction for this computation
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "   // D output (8 F32 registers)
        "{%8, %9}, "                           // A input (2 packed F16 registers)
        "{%10, %11}, "                         // B input (2 packed F16 registers)
        "{%12, %13, %14, %15, %16, %17, %18, %19};\n"  // C accumulator (8 F32 registers)
        : "=f"(D_frag[0]), "=f"(D_frag[1]), "=f"(D_frag[2]), "=f"(D_frag[3]),
          "=f"(D_frag[4]), "=f"(D_frag[5]), "=f"(D_frag[6]), "=f"(D_frag[7])
        : "r"(A_frag[0]), "r"(A_frag[1]),
          "r"(B_frag[0]), "r"(B_frag[1]),
          "f"(C_frag[0]), "f"(C_frag[1]), "f"(C_frag[2]), "f"(C_frag[3]),
          "f"(C_frag[4]), "f"(C_frag[5]), "f"(C_frag[6]), "f"(C_frag[7])
    );

    // Store D fragments to this computation's output (using same distribution as C)
    for (int i = 0; i < 8; i++) {
        int X = (lane_id & 0b1) + (i & 0b10);
        int row = X + (lane_id >= 16 ? 4 : 0);
        int col = (i & 0b100) + (lane_id & 0b10) + (i & 0b1);
        D[offset_D + row * 8 + col] = D_frag[i];
    }
}

void MMAF16M8N8K4IntegrationTest::run_mma_kernel() {
    // Copy all 4 computation matrices to device
    cudaMemcpy(d_A, h_A, NUM_COMP * M * K * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, NUM_COMP * K * N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, NUM_COMP * M * N * sizeof(float), cudaMemcpyHostToDevice);

    mma_m8n8k4_f16_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy all 4 computation results back to host
    cudaMemcpy(h_D, d_D, NUM_COMP * M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Test 1: All ones test for M8N8K4 - validates all 4 separate computations
TEST_F(MMAF16M8N8K4IntegrationTest, AllOnesTest) {
    // Initialize all 4 computations with ones (they should all produce the same result)
    for (int comp = 0; comp < NUM_COMP; comp++) {
        for (int i = 0; i < M * K; i++) h_A[comp * M * K + i] = 0x3C00;  // 1.0
        for (int i = 0; i < K * N; i++) h_B[comp * K * N + i] = 0x3C00;  // 1.0
        for (int i = 0; i < M * N; i++) h_C[comp * M * N + i] = 0.0f;
    }

    compute_reference();
    run_mma_kernel();

    // Validate all 4 computation outputs
    // Expected: 4.0 = sum of 1.0*1.0 for K=4
    for (int comp = 0; comp < NUM_COMP; comp++) {
        for (int i = 0; i < M * N; i++) {
            int idx = comp * M * N + i;
            EXPECT_FLOAT_EQ(h_D[idx], 4.0f)
                << "Mismatch in computation " << comp << " at index " << i;
        }
    }
}

// Test 2: Random values test for M8N8K4 - uses DIFFERENT values for each computation
TEST_F(MMAF16M8N8K4IntegrationTest, RandomValuesTest) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize each of the 4 computations with different random values
    for (int comp = 0; comp < NUM_COMP; comp++) {
        for (int i = 0; i < M * K; i++) {
            h_A[comp * M * K + i] = f32_to_f16(dist(gen));
        }
        for (int i = 0; i < K * N; i++) {
            h_B[comp * K * N + i] = f32_to_f16(dist(gen));
        }
        for (int i = 0; i < M * N; i++) {
            h_C[comp * M * N + i] = dist(gen);
        }
    }

    compute_reference();
    run_mma_kernel();

    // Validate all 4 computation outputs independently
    for (int comp = 0; comp < NUM_COMP; comp++) {
        int failures = 0;
        for (int i = 0; i < M * N; i++) {
            int idx = comp * M * N + i;
            float tolerance = std::max(0.01f, std::abs(h_D_ref[idx]) * 0.01f);
            if (std::abs(h_D[idx] - h_D_ref[idx]) > tolerance) {
                if (failures == 0) {
                    // Debug: Print first 4 values from each computation
                    printf("\n=== DEBUG: Computation %d first mismatch at index %d ===\n", comp, i);
                    for (int c = 0; c < NUM_COMP; c++) {
                        printf("Comp %d first 4 D actual: ", c);
                        for (int k = 0; k < 4; k++) {
                            printf("%.4f ", h_D[c * M * N + k]);
                        }
                        printf(" | expected: ");
                        for (int k = 0; k < 4; k++) {
                            printf("%.4f ", h_D_ref[c * M * N + k]);
                        }
                        printf("\n");
                    }
                }
                failures++;
            }
            EXPECT_NEAR(h_D[idx], h_D_ref[idx], tolerance)
                << "Mismatch in computation " << comp << " at index " << i
                << " (got: " << h_D[idx] << ", expected: " << h_D_ref[idx] << ")";
        }
    }
}
