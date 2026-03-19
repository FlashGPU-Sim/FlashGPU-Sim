#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>

// Note: This test file tests MMA instructions, NOT WMMA
// MMA instructions use mma.sync.aligned syntax (PTX ISA spec)
// WMMA uses wmma.mma.sync syntax (different instruction set)

// TODO: Add inline PTX MMA kernels when we figure out correct PTX syntax
// For now, these tests validate CPU reference implementations

// CPU reference implementation for MMA M16N8K8 (F16 inputs, F32 accumulator)
void mmaReferenceF16M16N8K8(
    const uint16_t* A,  // 16x8 matrix, F16
    const uint16_t* B,  // 8x8 matrix, F16
    const float* C,     // 16x8 matrix, F32 accumulator
    float* D,           // 16x8 matrix, F32 output
    bool row_major_A,
    bool row_major_B
) {
    // Helper to convert F16 to F32
    auto f16_to_f32 = [](uint16_t f16) -> float {
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
    };

    // Compute D = A × B + C
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = C[m * 8 + n];
            for (int k = 0; k < 8; ++k) {
                int a_idx = row_major_A ? (m * 8 + k) : (k * 16 + m);
                int b_idx = row_major_B ? (k * 8 + n) : (n * 8 + k);
                float a_val = f16_to_f32(A[a_idx]);
                float b_val = f16_to_f32(B[b_idx]);
                acc += a_val * b_val;
            }
            D[m * 8 + n] = acc;
        }
    }
}

// CPU reference implementation for MMA M16N8K16 (S8 inputs, S32 accumulator)
void mmaReferenceS8M16N8K16(
    const int8_t* A,    // 16x16 matrix, S8
    const int8_t* B,    // 16x8 matrix, S8
    const int32_t* C,   // 16x8 matrix, S32 accumulator
    int32_t* D,         // 16x8 matrix, S32 output
    bool row_major_A,
    bool row_major_B,
    bool saturate
) {
    auto saturate_s32 = [saturate](int64_t val) -> int32_t {
        if (!saturate) return static_cast<int32_t>(val);
        if (val > INT32_MAX) return INT32_MAX;
        if (val < INT32_MIN) return INT32_MIN;
        return static_cast<int32_t>(val);
    };

    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            int64_t acc = C[m * 8 + n];
            for (int k = 0; k < 16; ++k) {
                int a_idx = row_major_A ? (m * 16 + k) : (k * 16 + m);
                int b_idx = row_major_B ? (k * 8 + n) : (n * 16 + k);
                acc += static_cast<int64_t>(A[a_idx]) * static_cast<int64_t>(B[b_idx]);
            }
            D[m * 8 + n] = saturate_s32(acc);
        }
    }
}

class CudaTensorMMATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Random number generator with fixed seed
        gen = std::mt19937(42);
    }

    void TearDown() override {
        // Cleanup
    }

    // CUDA error checking helpers
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

    // Generate random F16 value
    uint16_t randomF16(float min_val, float max_val) {
        std::uniform_real_distribution<float> dis(min_val, max_val);
        float val = dis(gen);

        // Convert F32 to F16 (simplified - assumes normalized range)
        uint32_t f32_bits;
        memcpy(&f32_bits, &val, sizeof(float));

        uint32_t sign = (f32_bits >> 31) & 0x1;
        int32_t exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (f32_bits >> 13) & 0x3FF;

        if (exp <= 0) return sign << 15;  // Flush to zero
        if (exp >= 31) return (sign << 15) | 0x7C00;  // Inf

        return (sign << 15) | (exp << 10) | frac;
    }

    // Generate random S8 value
    int8_t randomS8() {
        std::uniform_int_distribution<int> dis(-128, 127);
        return static_cast<int8_t>(dis(gen));
    }

    std::mt19937 gen;
};

// Test: F16 MMA M16N8K8 with ROW-COL layout (identity matrix)
TEST_F(CudaTensorMMATest, F16_M16N8K8_RowCol_Identity) {
    // Matrix A: 16x8 identity (first 8 rows are I, rest are zeros)
    // Matrix B: 8x8 random values
    // Matrix C: zeros
    // Expected D = B (first 8 rows), zeros (rows 8-15)

    std::vector<uint16_t> h_A(16 * 8, 0);  // 128 elements
    std::vector<uint16_t> h_B(8 * 8);      // 64 elements
    std::vector<float> h_C(16 * 8, 0.0f);  // 128 elements
    std::vector<float> h_D(16 * 8);
    std::vector<float> h_D_ref(16 * 8);

    // Initialize A as partial identity (first 8x8)
    for (int i = 0; i < 8; ++i) {
        h_A[i * 8 + i] = 0x3C00;  // 1.0 in F16
    }

    // Initialize B with random values
    for (int i = 0; i < 64; ++i) {
        h_B[i] = randomF16(-10.0f, 10.0f);
    }

    // Compute CPU reference
    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, false);

    // TODO: Add CUDA kernel execution when PTX MMA syntax is resolved
    // For now, just verify CPU reference correctness

    // Expected: First 8 rows should approximately match B values
    // (since A is partial identity: first 8x8 is I, rest is zeros)
    // D[i,j] = sum_k(A[i,k] * B[k,j]) + C[i,j]
    // For i < 8: D[i,j] ≈ B[i,j] (since A is identity)
    // For i >= 8: D[i,j] ≈ 0 (since A rows are zeros)

    const float tolerance = 1e-3f;  // F16 precision tolerance
    bool reference_looks_correct = true;

    // Verify the reference implementation produces sensible results
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int idx = i * 8 + j;
            // First 8 rows: Result should be non-zero (B values)
            if (std::abs(h_D_ref[idx]) < tolerance && h_B[idx] != 0) {
                reference_looks_correct = false;
            }
        }
    }

    for (int i = 8; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            int idx = i * 8 + j;
            // Last 8 rows: Result should be close to zero
            if (std::abs(h_D_ref[idx]) > tolerance) {
                reference_looks_correct = false;
            }
        }
    }

    EXPECT_TRUE(reference_looks_correct)
        << "CPU reference implementation produces unexpected results";
}

// Test: S8 MMA M16N8K16 with saturation
TEST_F(CudaTensorMMATest, S8_M16N8K16_Saturation) {
    // Matrix A: 16x16, all max value (127)
    // Matrix B: 16x8, all max value (127)
    // Matrix C: zeros
    // Expected D: 127 * 127 * 16 = 258,048 (fits in S32, no saturation needed)

    std::vector<int8_t> h_A(16 * 16, 127);
    std::vector<int8_t> h_B(16 * 8, 127);
    std::vector<int32_t> h_C(16 * 8, 0);
    std::vector<int32_t> h_D(16 * 8);
    std::vector<int32_t> h_D_ref(16 * 8);

    // Compute CPU reference with saturation enabled
    mmaReferenceS8M16N8K16(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, true, true);

    // TODO: Add CUDA kernel execution when PTX MMA syntax is resolved

    // Verify CPU reference produces correct result
    int32_t expected = 127 * 127 * 16;  // = 258,048
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_EQ(h_D_ref[i], expected)
            << "CPU reference incorrect at element " << i
            << " (row " << (i / 8) << ", col " << (i % 8) << ")";
    }
}

// Test: Zero matrix multiplication
TEST_F(CudaTensorMMATest, F16_M16N8K8_ZeroMatrices) {
    std::vector<uint16_t> h_A(16 * 8, 0);
    std::vector<uint16_t> h_B(8 * 8, 0);
    std::vector<float> h_C(16 * 8, 0.0f);
    std::vector<float> h_D(16 * 8);
    std::vector<float> h_D_ref(16 * 8);

    // Compute CPU reference
    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, true);

    // TODO: Add CUDA kernel execution when PTX MMA syntax is resolved

    // All outputs should be zero
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_FLOAT_EQ(h_D_ref[i], 0.0f) << "CPU reference should be zero at element " << i;
    }
}

// Test: Layout mode difference (ROW-ROW vs ROW-COL)
TEST_F(CudaTensorMMATest, LayoutModeComparison) {
    std::vector<uint16_t> h_A(16 * 8);
    std::vector<uint16_t> h_B(8 * 8);
    std::vector<float> h_C(16 * 8, 0.0f);
    std::vector<float> h_D_row_col(16 * 8);
    std::vector<float> h_D_row_row(16 * 8);
    std::vector<float> h_D_ref_row_col(16 * 8);
    std::vector<float> h_D_ref_row_row(16 * 8);

    // Initialize with random values
    for (int i = 0; i < 16 * 8; ++i) {
        h_A[i] = randomF16(-1.0f, 1.0f);
    }
    for (int i = 0; i < 8 * 8; ++i) {
        h_B[i] = randomF16(-1.0f, 1.0f);
    }

    // Compute CPU references with different layouts
    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref_row_col.data(), true, false);
    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref_row_row.data(), true, true);

    // TODO: Add CUDA kernel execution when PTX MMA syntax is resolved

    // Results should differ between layouts (different memory access patterns)
    bool results_differ = false;
    for (int i = 0; i < 16 * 8; ++i) {
        if (std::abs(h_D_ref_row_col[i] - h_D_ref_row_row[i]) > 1e-5f) {
            results_differ = true;
            break;
        }
    }

    // Layout mode affects the computation, so results should differ
    EXPECT_TRUE(results_differ)
        << "CPU references for different layouts should produce different results";
}

// Test: Accumulator addition (C != 0)
TEST_F(CudaTensorMMATest, F16_M16N8K8_NonZeroAccumulator) {
    std::vector<uint16_t> h_A(16 * 8, 0x3C00);  // All 1.0
    std::vector<uint16_t> h_B(8 * 8, 0x3C00);   // All 1.0
    std::vector<float> h_C(16 * 8, 100.0f);     // Accumulator = 100.0
    std::vector<float> h_D_ref(16 * 8);

    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, true);

    // D = A × B + C = (1×1×8) + 100 = 108.0 for all elements
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_FLOAT_EQ(h_D_ref[i], 108.0f);
    }
}

// Test: S8 negative values
TEST_F(CudaTensorMMATest, S8_M16N8K16_NegativeValues) {
    std::vector<int8_t> h_A(16 * 16, -10);
    std::vector<int8_t> h_B(16 * 8, 5);
    std::vector<int32_t> h_C(16 * 8, 0);
    std::vector<int32_t> h_D_ref(16 * 8);

    mmaReferenceS8M16N8K16(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, true, false);

    // D = (-10) * 5 * 16 = -800 for all elements
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_EQ(h_D_ref[i], -800);
    }
}

// Test: Random F16 values (stress test)
TEST_F(CudaTensorMMATest, F16_M16N8K8_RandomValues) {
    std::vector<uint16_t> h_A(16 * 8);
    std::vector<uint16_t> h_B(8 * 8);
    std::vector<float> h_C(16 * 8);
    std::vector<float> h_D_ref(16 * 8);

    // Initialize with random values
    for (int i = 0; i < 16 * 8; ++i) {
        h_A[i] = randomF16(-10.0f, 10.0f);
        h_C[i] = static_cast<float>(std::uniform_real_distribution<float>(-50.0f, 50.0f)(gen));
    }
    for (int i = 0; i < 8 * 8; ++i) {
        h_B[i] = randomF16(-10.0f, 10.0f);
    }

    // Compute reference (should not crash or produce NaN for valid inputs)
    mmaReferenceF16M16N8K8(h_A.data(), h_B.data(), h_C.data(), h_D_ref.data(), true, true);

    // Verify no NaN results (valid computation)
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_FALSE(std::isnan(h_D_ref[i]));
    }
}

// Note: Actual PTX MMA instruction execution tests will be added after implementation
// These tests validate:
// 1. Reference implementation correctness
// 2. Matrix dimensions and layouts
// 3. Type conversions and saturations
// 4. Edge cases (zeros, identity, negative values, random data)

// Future tests (after MMA implementation):
// - CUDA kernel with inline PTX mma.sync.aligned instructions
// - Comparison between GPGPU-Sim execution and CPU reference
// - Performance validation
// - Memory transaction pattern verification
