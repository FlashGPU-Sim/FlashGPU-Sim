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

// ============== X2 Tests ==============
TEST_F(LdMatrixX2Test, BasicLdMatrixX2) {
  runKernelAndCopyBack();

  std::vector<uint32_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

// ============== X4 Tests ==============
TEST_F(LdMatrixX4Test, BasicLdMatrixX4) {
  runKernelAndCopyBack();

  std::vector<uint32_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

// ============================================================================
// stmatrix tests
// ============================================================================
// stmatrix instruction stores data from registers to shared memory
// Each thread in a warp participates in storing a matrix fragment
//
// stmatrix.sync.aligned.xN.m8n8.shared.b16
// - x1: Stores one 8x8 matrix, each thread provides 1 register (32-bit)
//       Threads 0-7 provide row addresses for the single matrix
// - x2: Stores two 8x8 matrices, each thread provides 2 registers
//       Threads 0-7 provide row addresses for matrix 0
//       Threads 8-15 provide row addresses for matrix 1
// - x4: Stores four 8x8 matrices, each thread provides 4 registers
//       Threads 0-7 provide row addresses for matrix 0
//       Threads 8-15 provide row addresses for matrix 1
//       Threads 16-23 provide row addresses for matrix 2
//       Threads 24-31 provide row addresses for matrix 3

// Template parameter N specifies x1, x2, or x4 variant
// N = number of matrices to store (1, 2, or 4)
template <int N>
__global__ void stmatrix_test_kernel(uint32_t *input, uint16_t *output) {
  // Shared memory for N matrices (each 8x8 matrix of 16-bit elements = 128
  // bytes) Must be aligned to 16 bytes for stmatrix
  __shared__ __align__(16) uint16_t smem[MATRIX_ELEMENTS * N];

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;

  // Step 1: Initialize shared memory to zero
  for (int i = 0; i < 2 * N; ++i) {
    int idx = lane_id * 2 * N + i;
    if (idx < MATRIX_ELEMENTS * N) {
      smem[idx] = 0;
    }
  }

  __syncthreads();

  // Step 2: Each thread calculates its address for stmatrix
  // Same addressing scheme as ldmatrix
  const int matrix_idx = lane_id / 8; // Which matrix this thread addresses
  const int row = lane_id % 8;        // Row within that matrix

  // Calculate the shared memory address
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
      &smem[matrix_idx * MATRIX_ELEMENTS + row * MATRIX_DIM]));

  // Step 3: Load register values from global memory input
  // Each thread loads N registers worth of data
  // The input is organized as: input[lane_id * N + reg]

  // Step 4: Use stmatrix inline assembly to store to shared memory
  if constexpr (N == 1) {
    uint32_t reg_value = input[lane_id];
    asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n"
                 :
                 : "r"(smem_addr), "r"(reg_value));
  } else if constexpr (N == 2) {
    uint32_t reg0 = input[lane_id * 2];
    uint32_t reg1 = input[lane_id * 2 + 1];
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                 :
                 : "r"(smem_addr), "r"(reg0), "r"(reg1));
  } else if constexpr (N == 4) {
    uint32_t reg0 = input[lane_id * 4];
    uint32_t reg1 = input[lane_id * 4 + 1];
    uint32_t reg2 = input[lane_id * 4 + 2];
    uint32_t reg3 = input[lane_id * 4 + 3];
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_addr), "r"(reg0), "r"(reg1), "r"(reg2), "r"(reg3));
  }

  __syncthreads();

  // Step 5: Copy shared memory to global memory output for verification
  for (int i = 0; i < 2 * N; ++i) {
    int idx = lane_id * 2 * N + i;
    if (idx < MATRIX_ELEMENTS * N) {
      output[idx] = smem[idx];
    }
  }
}

// Explicit template instantiations for stmatrix
template __global__ void stmatrix_test_kernel<1>(uint32_t *input,
                                                  uint16_t *output);
template __global__ void stmatrix_test_kernel<2>(uint32_t *input,
                                                  uint16_t *output);
template __global__ void stmatrix_test_kernel<4>(uint32_t *input,
                                                  uint16_t *output);

// Test fixture for stmatrix with template parameter for number of matrices
template <int N> class StMatrixTestT : public ::testing::Test {
protected:
  static constexpr int NUM_REGS_PER_THREAD = N;
  static constexpr int TOTAL_INPUT_SIZE = WARP_SIZE * N;   // 32-bit registers
  static constexpr int TOTAL_OUTPUT_SIZE = MATRIX_ELEMENTS * N; // 16-bit elements

  void SetUp() override {
    h_input.resize(TOTAL_INPUT_SIZE);
    h_output.resize(TOTAL_OUTPUT_SIZE);

    cudaError_t err =
        cudaMalloc(&d_input, TOTAL_INPUT_SIZE * sizeof(uint32_t));
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to allocate device input memory: " << cudaGetErrorString(err);

    err = cudaMalloc(&d_output, TOTAL_OUTPUT_SIZE * sizeof(uint16_t));
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to allocate device output memory: " << cudaGetErrorString(err);
  }

  void TearDown() override {
    if (d_input) {
      cudaFree(d_input);
      d_input = nullptr;
    }
    if (d_output) {
      cudaFree(d_output);
      d_output = nullptr;
    }
  }

  // Prepare input data - each thread gets N registers
  // The data is organized following the tensor core fragment layout
  void prepareInput() {
    // Generate input values that will be stored via stmatrix
    // Each thread's register contains two 16-bit values packed into 32-bit
    // Following the same layout as ldmatrix output:
    // Thread (lane_id) stores data to row (lane_id / 4), columns ((lane_id % 4) * 2) and ((lane_id % 4) * 2 + 1)
    for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
      int row = lane_id / 4;
      int col = (lane_id % 4) * 2;

      for (int reg = 0; reg < N; ++reg) {
        int matrix_idx = reg;
        int smem_offset = matrix_idx * MATRIX_ELEMENTS + row * MATRIX_DIM + col;
        uint16_t lo = static_cast<uint16_t>(smem_offset);
        uint16_t hi = static_cast<uint16_t>(smem_offset + 1);

        // Pack two 16-bit values into one 32-bit register (little-endian)
        h_input[lane_id * N + reg] =
            static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
    }
  }

  // Run kernel and copy results back to host
  void runKernelAndCopyBack() {
    // Copy input to device
    cudaError_t err = cudaMemcpy(d_input, h_input.data(),
                                  TOTAL_INPUT_SIZE * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to copy input to device: " << cudaGetErrorString(err);

    stmatrix_test_kernel<N><<<1, BLOCK_SIZE>>>(d_input, d_output);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess)
        << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess)
        << "Kernel execution failed: " << cudaGetErrorString(err);

    err = cudaMemcpy(h_output.data(), d_output,
                     TOTAL_OUTPUT_SIZE * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to copy results to host: " << cudaGetErrorString(err);
  }

  // Compute expected output in shared memory after stmatrix
  void computeExpectedOutput(std::vector<uint16_t> &expected) {
    expected.resize(TOTAL_OUTPUT_SIZE);

    // stmatrix stores the register data back to shared memory
    // The expected result is that smem[i] = i for all elements
    // (since we prepared input to store value i at position i)
    for (int i = 0; i < TOTAL_OUTPUT_SIZE; ++i) {
      expected[i] = static_cast<uint16_t>(i);
    }
  }

  // Compare actual output against expected values
  void compareResults(const std::vector<uint16_t> &expected) {
    for (int matrix_idx = 0; matrix_idx < N; ++matrix_idx) {
      for (int row = 0; row < MATRIX_DIM; ++row) {
        for (int col = 0; col < MATRIX_DIM; ++col) {
          int idx = matrix_idx * MATRIX_ELEMENTS + row * MATRIX_DIM + col;
          EXPECT_EQ(h_output[idx], expected[idx])
              << "X" << N << " Mismatch at matrix " << matrix_idx 
              << " row " << row << " col " << col
              << ": expected " << expected[idx]
              << ", got " << h_output[idx];
        }
      }
    }
  }

  std::vector<uint32_t> h_input;
  std::vector<uint16_t> h_output;
  uint32_t *d_input = nullptr;
  uint16_t *d_output = nullptr;
};

// Type aliases for different stmatrix test configurations
using StMatrixX1Test = StMatrixTestT<1>;
using StMatrixX2Test = StMatrixTestT<2>;
using StMatrixX4Test = StMatrixTestT<4>;

// ============== StMatrix X1 Tests ==============
TEST_F(StMatrixX1Test, BasicStMatrixX1) {
  prepareInput();
  runKernelAndCopyBack();

  std::vector<uint16_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

// ============== StMatrix X2 Tests ==============
TEST_F(StMatrixX2Test, BasicStMatrixX2) {
  prepareInput();
  runKernelAndCopyBack();

  std::vector<uint16_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

// ============== StMatrix X4 Tests ==============
TEST_F(StMatrixX4Test, BasicStMatrixX4) {
  prepareInput();
  runKernelAndCopyBack();

  std::vector<uint16_t> expected;
  computeExpectedOutput(expected);
  compareResults(expected);
}

// ============================================================================
// Transpose Tests (sanity checks for .trans modifier)
// ============================================================================

__global__ void ldmatrix_m8n8_trans_test_kernel(uint32_t *output) {
  __shared__ __align__(16) uint16_t smem[MATRIX_ELEMENTS];

  const int lane_id = threadIdx.x % WARP_SIZE;

  // Initialize as row-major, will be interpreted as transposed
  for (int i = lane_id; i < MATRIX_ELEMENTS; i += WARP_SIZE) {
    smem[i] = static_cast<uint16_t>(i);
  }
  __syncthreads();

  const int row = lane_id % 8;
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem[row * MATRIX_DIM]));

  uint32_t reg_value;
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
               : "=r"(reg_value)
               : "r"(smem_addr));
  output[lane_id] = reg_value;
}

class LdMatrixM8N8TransTest : public ::testing::Test {
protected:
  void SetUp() override {
    h_output.resize(WARP_SIZE);
    cudaMalloc(&d_output, WARP_SIZE * sizeof(uint32_t));
  }

  void TearDown() override {
    if (d_output)
      cudaFree(d_output);
  }

  void runKernelAndCopyBack() {
    ldmatrix_m8n8_trans_test_kernel<<<1, BLOCK_SIZE>>>(d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, WARP_SIZE * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
  }

  std::vector<uint32_t> h_output;
  uint32_t *d_output = nullptr;
};

TEST_F(LdMatrixM8N8TransTest, TransposeLoadSanity) {
  runKernelAndCopyBack();

  // Verify exact values for transpose load
  // Fragment position: (frag_row, frag_col) = (lane_id / 4, (lane_id % 4) * 2)
  // For transpose: access transposed matrix at (frag_col, frag_row)
  // - Column address comes from lane (col_index = (lane_id % 4))
  // - Lanes 0-3 provide addresses for columns 0-3
  // - Access rows: each row is 8 elements apart
  for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
    int frag_row = lane_id / 4;           // fragment row position (0-7)
    int frag_col_base = (lane_id % 4) * 2; // fragment column position (0,2,4,6)

    int col_index = lane_id % 4;          // which column address (0-3)
    int col_address_lane = col_index;     // lane providing column address

    // For m8n8: 8 rows x 8 cols, smem[i] = i (row-major storage)
    // Transpose accesses: (col, row) instead of (row, col)
    // smem is initialized as smem[row * 8 + col] = row * 8 + col
    // For transpose, we access smem[frag_col * 8 + frag_row]
    uint16_t expected_lo = frag_col_base * 8 + frag_row;
    uint16_t expected_hi = (frag_col_base + 1) * 8 + frag_row;
    uint32_t expected = expected_lo | (expected_hi << 16);

    EXPECT_EQ(h_output[lane_id], expected)
        << "Mismatch at lane " << lane_id
        << " (frag_row=" << frag_row << ", frag_col=" << frag_col_base << ")"
        << ": expected " << std::hex << expected
        << " (lo=" << expected_lo << ", hi=" << expected_hi << ")"
        << ", got " << h_output[lane_id]
        << " (lo=" << (h_output[lane_id] & 0xFFFF)
        << ", hi=" << ((h_output[lane_id] >> 16) & 0xFFFF) << ")" << std::dec;
  }
}

// stmatrix transpose test
__global__ void stmatrix_m8n8_trans_test_kernel(uint16_t *output) {
  __shared__ __align__(16) uint16_t smem[MATRIX_ELEMENTS];

  const int lane_id = threadIdx.x % WARP_SIZE;

  // Initialize shared memory to zeros
  for (int i = lane_id; i < MATRIX_ELEMENTS; i += WARP_SIZE) {
    smem[i] = 0;
  }
  __syncthreads();

  // Each thread computes its register value based on fragment distribution
  // For m8n8: matrix_row_id = lane_id / 4, col_offset = (lane_id % 4) * 2
  int matrix_row_id = lane_id / 4;
  int col_offset = (lane_id % 4) * 2;

  // Pack two elements into a 32-bit register
  // Use values that make it easy to verify transpose
  uint16_t elem_lo = matrix_row_id * 10 + col_offset;
  uint16_t elem_hi = matrix_row_id * 10 + col_offset + 1;
  uint32_t reg_value = elem_lo | (elem_hi << 16);

  // Calculate address for stmatrix (row-based addressing)
  const int row = lane_id % 8;
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem[row * MATRIX_DIM]));

  // Use stmatrix with .trans modifier
  asm volatile("stmatrix.sync.aligned.x1.m8n8.trans.shared.b16 [%0], {%1};\n"
               :
               : "r"(smem_addr), "r"(reg_value));

  __syncthreads();

  // Copy shared memory to output for verification
  for (int i = lane_id; i < MATRIX_ELEMENTS; i += WARP_SIZE) {
    output[i] = smem[i];
  }
}

class StMatrixM8N8TransTest : public ::testing::Test {
protected:
  void SetUp() override {
    h_output.resize(MATRIX_ELEMENTS);
    cudaMalloc(&d_output, MATRIX_ELEMENTS * sizeof(uint16_t));
  }

  void TearDown() override {
    if (d_output)
      cudaFree(d_output);
  }

  void runKernelAndCopyBack() {
    stmatrix_m8n8_trans_test_kernel<<<1, BLOCK_SIZE>>>(d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, MATRIX_ELEMENTS * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
  }

  std::vector<uint16_t> h_output;
  uint16_t *d_output = nullptr;
};

TEST_F(StMatrixM8N8TransTest, TransposeStoreSanity) {
  runKernelAndCopyBack();

  // Verify transpose storage
  // For transpose: thread provides column address, stores to rows
  // Fragment position: (frag_row, frag_col) = (lane_id / 4, (lane_id % 4) * 2)
  // Register contains: elem_lo = frag_row * 10 + frag_col, elem_hi = frag_row * 10 + frag_col + 1
  // Memory access for transpose: (frag_col, frag_row) instead of (frag_row, frag_col)

  // Build expected output matrix
  std::vector<uint16_t> expected(MATRIX_ELEMENTS, 0);
  for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
    int frag_row = lane_id / 4;           // fragment row position (0-7)
    int frag_col_base = (lane_id % 4) * 2; // fragment column position (0,2,4,6)

    // Values from register (set in kernel)
    uint16_t elem_lo = frag_row * 10 + frag_col_base;
    uint16_t elem_hi = frag_row * 10 + frag_col_base + 1;

    // For transpose: store at memory position (frag_col, frag_row)
    // smem[frag_col * 8 + frag_row] for first element
    // smem[(frag_col+1) * 8 + frag_row] for second element
    expected[frag_col_base * MATRIX_DIM + frag_row] = elem_lo;
    expected[(frag_col_base + 1) * MATRIX_DIM + frag_row] = elem_hi;
  }

  // Verify the stored values
  for (int i = 0; i < MATRIX_ELEMENTS; ++i) {
    EXPECT_EQ(h_output[i], expected[i])
        << "Mismatch at index " << i << " (row " << i / MATRIX_DIM
        << ", col " << i % MATRIX_DIM << "): expected " << expected[i]
        << ", got " << h_output[i];
  }
}
