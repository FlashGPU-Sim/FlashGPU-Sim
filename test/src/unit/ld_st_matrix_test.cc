#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// ldmatrix instruction loads data from shared memory into registers
// Each thread in a warp participates in loading a matrix fragment
//
// ldmatrix.sync.aligned.xN.m8n8.shared.b16
// - x1: Loads one 8x8 matrix, each thread gets 1 register (32-bit)
//       Threads 0-7 provide row addresses for the single matrix
// - x2: Loads two 8x8 matrices, each thread gets 2 registers
//       Threads 0-7 provide row addresses for matrix 0
//       Threads 8-15 provide row addresses for matrix 1
// - x4: Loads four 8x8 matrices, each thread gets 4 registers
//       Threads 0-7 provide row addresses for matrix 0
//       Threads 8-15 provide row addresses for matrix 1
//       Threads 16-23 provide row addresses for matrix 2
//       Threads 24-31 provide row addresses for matrix 3

// Constants for the test
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = WARP_SIZE; // Single warp, single block
constexpr int MATRIX_DIM = 8;         // 8x8 matrix
constexpr int MATRIX_ELEMENTS =
    MATRIX_DIM * MATRIX_DIM; // 64 elements per matrix

// Template parameter N specifies x1, x2, or x4 variant
// N = number of matrices to load (1, 2, or 4)
template <int N> __global__ void ldmatrix_test_kernel(uint32_t *output) {
  // Shared memory for N matrices (each 8x8 matrix of 16-bit elements = 128
  // bytes) Must be aligned to 16 bytes for ldmatrix
  __shared__ __align__(16) uint16_t smem[MATRIX_ELEMENTS * N];

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;

  // Step 1: Initialize shared memory with known values
  // Total elements = 64 * N, each thread initializes (64 * N / 32) = 2 * N
  // elements
  for (int i = 0; i < 2 * N; ++i) {
    int idx = lane_id * 2 * N + i;
    if (idx < MATRIX_ELEMENTS * N) {
      smem[idx] = static_cast<uint16_t>(idx);
    }
  }

  __syncthreads();

  // Step 2: Each thread calculates its address for ldmatrix
  // For ldmatrix.xN.m8n8:
  // - x1: Only threads 0-7 addresses are used (row 0-7 of single matrix)
  // - x2: Threads 0-7 -> matrix 0, rows 0-7; Threads 8-15 -> matrix 1, rows 0-7
  // - x4: Threads 0-7 -> matrix 0; 8-15 -> matrix 1; 16-23 -> matrix 2; 24-31
  // -> matrix 3
  const int matrix_idx = lane_id / 8; // Which matrix this thread addresses
  const int row = lane_id % 8;        // Row within that matrix

  // Calculate the shared memory address
  // Each matrix starts at matrix_idx * MATRIX_ELEMENTS
  // Each row is 8 elements (16 bytes, aligned)
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
      &smem[matrix_idx * MATRIX_ELEMENTS + row * MATRIX_DIM]));

  // Step 3: Use ldmatrix inline assembly to load from shared memory
  // Use template specialization via if constexpr for different variants
  if constexpr (N == 1) {
    uint32_t reg_value;
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(reg_value)
                 : "r"(smem_addr));
    output[lane_id] = reg_value;
  } else if constexpr (N == 2) {
    uint32_t reg0, reg1;
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(reg0), "=r"(reg1)
                 : "r"(smem_addr));
    output[lane_id * 2] = reg0;
    output[lane_id * 2 + 1] = reg1;
  } else if constexpr (N == 4) {
    uint32_t reg0, reg1, reg2, reg3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "r"(smem_addr));
    output[lane_id * 4] = reg0;
    output[lane_id * 4 + 1] = reg1;
    output[lane_id * 4 + 2] = reg2;
    output[lane_id * 4 + 3] = reg3;
  }
}

// Explicit template instantiations
template __global__ void ldmatrix_test_kernel<1>(uint32_t *output);
template __global__ void ldmatrix_test_kernel<2>(uint32_t *output);
template __global__ void ldmatrix_test_kernel<4>(uint32_t *output);

// Test fixture with template parameter for number of matrices
template <int N> class LdMatrixTestT : public ::testing::Test {
protected:
  static constexpr int NUM_REGS_PER_THREAD = N;
  static constexpr int TOTAL_OUTPUT_SIZE = WARP_SIZE * N;

  void SetUp() override {
    h_output.resize(TOTAL_OUTPUT_SIZE);

    cudaError_t err =
        cudaMalloc(&d_output, TOTAL_OUTPUT_SIZE * sizeof(uint32_t));
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to allocate device memory: " << cudaGetErrorString(err);
  }

  void TearDown() override {
    if (d_output) {
      cudaFree(d_output);
      d_output = nullptr;
    }
  }

  // Run kernel and copy results back to host
  void runKernelAndCopyBack() {
    ldmatrix_test_kernel<N><<<1, BLOCK_SIZE>>>(d_output);

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess)
        << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess)
        << "Kernel execution failed: " << cudaGetErrorString(err);

    err = cudaMemcpy(h_output.data(), d_output,
                     TOTAL_OUTPUT_SIZE * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to copy results to host: " << cudaGetErrorString(err);
  }

  // Compute expected output based on ldmatrix semantics
  void computeExpectedOutput(std::vector<uint32_t> &expected) {
    expected.resize(TOTAL_OUTPUT_SIZE);

    // Initialize the "shared memory" simulation
    std::vector<uint16_t> smem(MATRIX_ELEMENTS * N);
    for (int i = 0; i < MATRIX_ELEMENTS * N; ++i) {
      smem[i] = static_cast<uint16_t>(i);
    }

    // ldmatrix.xN.m8n8 loads N 8x8 matrices of 16-bit elements
    // Each thread gets N 32-bit registers, each containing two 16-bit values
    // The distribution follows the tensor core fragment layout:
    // - Thread (lane_id) gets data from row (lane_id / 4), columns ((lane_id %
    // 4) * 2) and ((lane_id % 4) * 2 + 1)
    for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
      int row = lane_id / 4;
      int col = (lane_id % 4) * 2;

      for (int reg = 0; reg < N; ++reg) {
        int matrix_idx = reg;
        int smem_offset = matrix_idx * MATRIX_ELEMENTS + row * MATRIX_DIM + col;
        uint16_t lo = smem[smem_offset];
        uint16_t hi = smem[smem_offset + 1];

        // Pack two 16-bit values into one 32-bit register (little-endian)
        expected[lane_id * N + reg] =
            static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
    }
  }

  // Compare actual output against expected values
  void compareResults(const std::vector<uint32_t> &expected) {
    for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
      for (int reg = 0; reg < N; ++reg) {
        int idx = lane_id * N + reg;
        uint16_t lo_actual = h_output[idx] & 0xFFFF;
        uint16_t hi_actual = (h_output[idx] >> 16) & 0xFFFF;
        uint16_t lo_expected = expected[idx] & 0xFFFF;
        uint16_t hi_expected = (expected[idx] >> 16) & 0xFFFF;

        EXPECT_EQ(h_output[idx], expected[idx])
            << "X" << N << " Mismatch at thread " << lane_id << " reg " << reg
            << ": expected 0x" << std::hex << expected[idx]
            << " (lo=" << lo_expected << ", hi=" << hi_expected << ")"
            << ", got 0x" << h_output[idx] << " (lo=" << lo_actual
            << ", hi=" << hi_actual << ")" << std::dec;
      }
    }
  }

  // Print loaded values for debugging
  void printResults() {
    printf("\nldmatrix.x%d loaded values:\n", N);
    if (N == 1) {
      printf("Thread | Register Value | Lo (16-bit) | Hi (16-bit)\n");
      printf("-------+----------------+-------------+------------\n");
    } else {
      printf("Thread | Reg | Register Value | Lo (16-bit) | Hi (16-bit)\n");
      printf("-------+-----+----------------+-------------+------------\n");
    }

    for (int i = 0; i < WARP_SIZE; ++i) {
      for (int reg = 0; reg < N; ++reg) {
        int idx = i * N + reg;
        uint16_t lo = h_output[idx] & 0xFFFF;
        uint16_t hi = (h_output[idx] >> 16) & 0xFFFF;
        if (N == 1) {
          printf("  %2d   |   0x%08x   |    %5u    |    %5u\n", i,
                 h_output[idx], lo, hi);
        } else {
          printf("  %2d   |  %d  |   0x%08x   |    %5u    |    %5u\n", i, reg,
                 h_output[idx], lo, hi);
        }
      }
    }
  }

  std::vector<uint32_t> h_output;
  uint32_t *d_output = nullptr;
};

// Type aliases for different test configurations
using LdMatrixX1Test = LdMatrixTestT<1>;
using LdMatrixX2Test = LdMatrixTestT<2>;
using LdMatrixX4Test = LdMatrixTestT<4>;

// ============== X1 Tests ==============
TEST_F(LdMatrixX1Test, BasicLdMatrixX1) {
  runKernelAndCopyBack();

  std::vector<uint32_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

TEST_F(LdMatrixX1Test, PrintLoadedValuesX1) {
  runKernelAndCopyBack();
  printResults();
}

// ============== X2 Tests ==============
TEST_F(LdMatrixX2Test, BasicLdMatrixX2) {
  runKernelAndCopyBack();

  std::vector<uint32_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

TEST_F(LdMatrixX2Test, PrintLoadedValuesX2) {
  runKernelAndCopyBack();
  printResults();
}

// ============== X4 Tests ==============
TEST_F(LdMatrixX4Test, BasicLdMatrixX4) {
  runKernelAndCopyBack();

  std::vector<uint32_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

TEST_F(LdMatrixX4Test, PrintLoadedValuesX4) {
  runKernelAndCopyBack();
  printResults();
}
