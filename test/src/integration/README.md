# Integration Tests

This folder contains integration tests for GPGPU-Sim that execute full CUDA kernels through the simulator.

## Purpose

Integration tests validate end-to-end functionality by:
- Compiling real CUDA kernels with inline PTX assembly
- Executing kernels through GPGPU-Sim functional simulator
- Comparing results against CPU reference implementations
- Testing complete instruction paths from PTX parsing to execution

## Relationship to Unit Tests

| Test Type | Location | Purpose | Scope |
|-----------|----------|---------|-------|
| **Unit Tests** | `test/src/unit/` | Test individual functions/helpers in isolation | Function-level |
| **Integration Tests** | `test/src/integration/` (this folder) | Test full CUDA kernels through simulator | End-to-end |

## Running Integration Tests

### Prerequisites

Source environment setup before running tests:
```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
```

### Run All Integration Tests

```bash
# From project root
test/run_tests.sh run test --target sm120 --group integration
```

### Run Specific Test Suites

The test runner accepts Google Test filter patterns:

```bash
# Run only F16 MMA tests
test/run_tests.sh run test --target sm120 --group integration '*F16*'

# Run only TF32 MMA tests
test/run_tests.sh run test --target sm120 --group integration '*TF32*'

# Run all MMA tests (F16, BF16, TF32, S8)
test/run_tests.sh run test --target sm120 --group integration '*MMA*'

# Run specific test case
test/run_tests.sh run test --target sm120 --group integration \
  'MMAF16M16N8K8IntegrationTest.AllOnesTest'
```

**Note**: Pass the filter directly after the suite without a
`--gtest_filter` prefix. The runner handles Google Test formatting.

## Test Organization

### Tensor Core MMA Tests

Located in `test/src/integration/mma/` subdirectory:

#### F16 Tests (`mma/cuda_mma_f16_test.cc`)
- Tests F16 (half-precision float) MMA operations with M16N8K8 shape
- Test cases:
  - AllOnesTest: Uniform inputs (all 1.0)
  - ZeroMatrixTest: Zero matrices
  - IdentityMatrixTest: Identity matrix multiplication
  - RandomValuesTest: Random values stress test
  - MixedSignTest: Positive/negative value handling

**Fragment distribution**: Each thread holds 4 F16 values for A, 2 for B, 4 F32 for C/D

#### BF16 Tests (`mma/cuda_mma_bf16_test.cc`)
- Tests BF16 (bfloat16) MMA operations with M16N8K8 shape
- BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
- Wider dynamic range than F16, preferred for ML training

**Fragment distribution**: Same as F16 (4 BF16 for A, 2 for B, 4 F32 for C/D)

#### TF32 Tests (`mma/cuda_mma_tf32_test.cc`)
- Tests TF32 (TensorFloat-32) MMA operations with M16N8K4 shape
- TF32: 1 sign bit, 8 exponent bits, 10 mantissa bits
- Uses `b32` registers for A/B fragments (not `f32`)
- **Important**: M16N8K4 is required for sm90 (Hopper), M16N8K8 only for sm80-89 (Ampere)

**Fragment distribution**: Each thread holds 2 TF32 for A, 1 for B, 4 F32 for C/D

#### S8 Tests (`mma/cuda_mma_s8_test.cc`)
- Tests S8 (signed 8-bit integer) MMA operations with M16N8K16 shape
- Includes saturation tests (clamp to [-128, 127])
- Integer accumulation in S32 format

**Fragment distribution**: Each thread holds 4 S8 for A, 2 for B, 4 S32 for C/D

### Other Integration Tests

- `cuda_tensor_mma_test.cc`: Original CPU reference tests (7 tests)
- `cuda_tma_test.cc`: TMA (Tensor Memory Accelerator) tests

## Test Structure

Each MMA integration test follows this pattern:

```cpp
class MMAF16M16N8K8IntegrationTest : public ::testing::Test {
protected:
    // Test parameters
    static constexpr int M = 16, N = 8, K = 8;

    // Host and device memory
    uint16_t *h_A, *h_B;
    float *h_C, *h_D, *h_D_ref;

    void SetUp() override {
        // Allocate memory
    }

    void TearDown() override {
        // Free memory
    }

    void compute_reference() {
        // CPU reference implementation
    }

    void run_mma_kernel() {
        // Execute MMA kernel through GPGPU-Sim
    }
};

TEST_F(MMAF16M16N8K8IntegrationTest, TestName) {
    // Initialize inputs
    // Compute reference
    // Run MMA kernel
    // Validate results with EXPECT_NEAR
}
```

## Validation Strategy

### Tolerance Values

Different data types require different tolerances due to precision limitations:

```cpp
// F16: 1e-3f - 10-bit mantissa precision
float tolerance = 1e-3f;

// BF16: 1e-2f - 7-bit mantissa precision (less precise than F16)
float tolerance = 1e-2f;

// TF32: 1e-3f to 1e-2f - 10-bit mantissa + accumulation errors
float tolerance = 1e-3f;  // or 1e-2f for random tests

// S8: 0 - Exact integer arithmetic (no tolerance)
EXPECT_EQ(result, expected);
```

### CPU Reference Implementation

Each test computes expected results on CPU:
1. Apply data type rounding (F16, BF16, TF32)
2. Perform matrix multiplication: D = A × B + C
3. Apply saturation if needed (integer types)
4. Compare GPU result against CPU reference

## Expected Test Results

After full implementation (Issue #18):

| Phase | Tests Passing | Status |
|-------|---------------|--------|
| Infrastructure complete | 0/25 | Parser/opcodes only |
| M16N8K8 F16/F32 impl | 16/25 | F16 tests pass |
| All shapes/types impl | 21/25 | F16, BF16, S8 pass |
| TF32 M16N8K4 impl | 25/25 | All tests pass ✅ |

Current status: **25/25 tests passing** (F16: 16, BF16: 5, TF32: 5, S8: 4)

## Troubleshooting

### Test Compilation Errors

If tests fail to compile:
- Verify CUDA_INSTALL_PATH is set: `echo $CUDA_INSTALL_PATH`
- Source environment: `source setup_environment`
- Check CUDA toolkit version supports architecture (sm90 for TF32 M16N8K4)

### Test Execution Failures

If tests compile but fail:
- Check tolerance values match data type precision
- Verify fragment distribution matches PTX ISA specification
- Review CPU reference implementation for correct type rounding
- Enable debug output: Set `gpgpu_ctx->debug_tensorcore = true` in tensor_mma.cc

### All Results Zero

If test results are all zeros:
- Verify shape case is handled in tensor_mma.cc shape switch (lines 406-424)
- Check that shape enum maps correctly from parser to implementation
- Ensure fragment distribution uses correct groupID/threadID formulas

## Adding New MMA Tests

When adding support for new MMA shapes or data types:

1. **Create test file**: `mma/cuda_mma_<type>_test.cc`
2. **Define test fixture**: Inherit from `::testing::Test`
3. **Implement reference**: CPU implementation with proper type rounding
4. **Write test cases**: At least 5 tests (all-ones, zeros, identity, random, edge cases)
5. **Update this README**: Document new test file and expected results

## References

- Test execution: `docs/testing-instructions.md`
- MMA implementation: `src/gpgpu-sim/flash/mma/tensor_mma.{h,cc,md}`
- PTX ISA: [MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
- Google Test: [Testing framework documentation](https://google.github.io/googletest/)
