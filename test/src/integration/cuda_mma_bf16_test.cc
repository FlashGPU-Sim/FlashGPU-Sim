// CUDA kernel with inline PTX for mma.sync.aligned.m16n8k8 with BF16 inputs
// Tests BF16 inputs → F32 output accumulation
// Following pattern from cuda_mma_m16n8k8_test.cc

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

// BF16 conversion helpers (CPU-side)
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t u32 = static_cast<uint32_t>(bf16) << 16;
    return *reinterpret_cast<float*>(&u32);
}

inline uint16_t f32_to_bf16(float f32) {
    uint32_t u32 = *reinterpret_cast<uint32_t*>(&f32);
    // Round to nearest even
    uint32_t rounding_bias = 0x00007FFF + ((u32 >> 16) & 1);
    return static_cast<uint16_t>((u32 + rounding_bias) >> 16);
}

// Test fixture for BF16 MMA M16N8K8 integration tests
class MMABF16M16N8K8IntegrationTest : public ::testing::Test {
protected:
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 8;

    uint16_t* h_A;  // BF16
    uint16_t* h_B;  // BF16
    float* h_C;     // F32
    float* h_D;     // F32
    float* h_D_ref; // Reference result for validation

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

    // Compute reference result on CPU
    void compute_reference() {
        // Compute D = A * B + C
        // A is row-major: A[m][k] = A[m * K + k]
        // B is column-major: B[k][n] = B[n * K + k]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float a = bf16_to_f32(h_A[i * K + k]);
                    float b = bf16_to_f32(h_B[j * K + k]);
                    sum += a * b;
                }
                h_D_ref[i * N + j] = sum + h_C[i * N + j];
            }
        }
    }

    // Run MMA kernel
    void run_mma_kernel();
};

// Simple kernel using mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32
__global__ void mma_m16n8k8_bf16_kernel(
    const uint16_t* A,  // 16x8 BF16 matrix (row-major)
    const uint16_t* B,  // 8x8 BF16 matrix (column-major)
    const float* C,     // 16x8 F32 accumulator
    float* D            // 16x8 F32 output
) {
    // Only use first warp for simplicity
    int warp_id = threadIdx.x / 32;
    if (warp_id != 0) return;

    int lane_id = threadIdx.x % 32;

    // Load fragments into registers
    // For m16n8k8 BF16: A needs 2 registers, B needs 1 register, C/D need 4 registers each
    unsigned A_frag[2];
    unsigned B_frag[1];
    float C_frag[4];
    float D_frag[4];

    // Fragment distribution for M16N8K8 (same as F16)
    // groupID = lane_id >> 2 (range 0-7)
    // threadID_in_group = lane_id % 4 (range 0-3)
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;

    // Load A fragments (16×8 row-major)
    // Each thread loads 4 BF16 values (2 per row for 2 rows)
    uint16_t a_vals[4];
    int a_row0 = groupID;
    int a_row1 = groupID + 8;
    int a_col0 = threadID_in_group * 2;
    int a_col1 = threadID_in_group * 2 + 1;

    a_vals[0] = A[a_row0 * 8 + a_col0];
    a_vals[1] = A[a_row0 * 8 + a_col1];
    a_vals[2] = A[a_row1 * 8 + a_col0];
    a_vals[3] = A[a_row1 * 8 + a_col1];

    A_frag[0] = *reinterpret_cast<unsigned*>(&a_vals[0]);  // a[0,1]
    A_frag[1] = *reinterpret_cast<unsigned*>(&a_vals[2]);  // a[2,3]

    // Load B fragments (8×8 column-major)
    uint16_t b_vals[2];
    int b_col = groupID;
    b_vals[0] = B[b_col * 8 + threadID_in_group * 2];
    b_vals[1] = B[b_col * 8 + threadID_in_group * 2 + 1];
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
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "           // D output (4 F32 registers)
        "{%4, %5}, "                   // A input (2 packed BF16 registers)
        "{%6}, "                       // B input (1 packed BF16 register)
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

void MMABF16M16N8K8IntegrationTest::run_mma_kernel() {
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 32 threads = 1 warp)
    mma_m16n8k8_bf16_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);

    // Copy result back
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Test 1: All ones test
TEST_F(MMABF16M16N8K8IntegrationTest, AllOnesTest) {
    // Initialize with 1.0
    for (int i = 0; i < M * K; i++) h_A[i] = f32_to_bf16(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = f32_to_bf16(1.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results (expected: 8.0 = sum of 1*1 for K=8)
    // Use tolerance for BF16 precision (7 bits mantissa)
    float tolerance = 1e-2f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 8.0f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 2: Zero matrix test
TEST_F(MMABF16M16N8K8IntegrationTest, ZeroMatrixTest) {
    // Initialize with zeros
    for (int i = 0; i < M * K; i++) h_A[i] = f32_to_bf16(0.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = f32_to_bf16(0.0f);
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

// Test 3: Precision comparison vs F32
TEST_F(MMABF16M16N8K8IntegrationTest, PrecisionTest) {
    // Use values that show BF16 precision differences (7-bit mantissa)
    // BF16 can represent powers of 2 exactly
    for (int i = 0; i < M * K; i++) h_A[i] = f32_to_bf16(0.125f);  // 2^-3
    for (int i = 0; i < K * N; i++) h_B[i] = f32_to_bf16(0.25f);   // 2^-2
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Expected: 0.125 * 0.25 * 8 = 0.25
    // BF16 tolerance is larger than F16 (7 bits vs 10 bits mantissa)
    float tolerance = 5e-3f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], 0.25f, tolerance) << "Mismatch at index " << i;
    }
}

// Test 4: Random values test
TEST_F(MMABF16M16N8K8IntegrationTest, RandomValuesTest) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize with random BF16 values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = f32_to_bf16(dist(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = f32_to_bf16(dist(gen));
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = dist(gen);
    }

    // Compute reference
    compute_reference();

    // Run MMA kernel
    run_mma_kernel();

    // Verify results with BF16 tolerance
    float tolerance = 1e-2f;
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_D[i], h_D_ref[i], tolerance)
            << "Mismatch at index " << i
            << " (got: " << h_D[i] << ", expected: " << h_D_ref[i] << ")";
    }
}
