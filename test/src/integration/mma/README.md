# MMA Integration Tests

End-to-end integration tests for PTX MMA (Matrix Multiply-Accumulate) tensor core instructions.

## Purpose

This directory contains comprehensive integration tests that validate MMA instruction implementations by:
- Executing real CUDA kernels with inline PTX assembly through GPGPU-Sim
- Comparing GPU results against CPU reference implementations
- Testing complete instruction paths from PTX parsing to functional execution

## Test Files

### F16 Tests - `cuda_mma_f16_test.cc`

Tests F16 (IEEE 754 half-precision float) MMA operations with M16N8K8 shape.

**Test cases**:
- `AllOnesTest` - Uniform inputs (all 1.0)
- `ZeroMatrixTest` - Zero matrices
- `IdentityMatrixTest` - Identity matrix multiplication
- `RandomValuesTest` - Random values stress test
- `MixedSignTest` - Positive/negative value handling

**Fragment distribution**: Each thread holds 4 F16 values for A, 2 for B, 4 F32 for C/D

### BF16 Tests - `cuda_mma_bf16_test.cc`

Tests BF16 (bfloat16) MMA operations with M16N8K8 shape.

**Key characteristics**:
- BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
- Wider dynamic range than F16, preferred for ML training
- Same test cases as F16 with adjusted tolerance (1e-2f due to 7-bit mantissa)

**Fragment distribution**: Same as F16 (4 BF16 for A, 2 for B, 4 F32 for C/D)

### TF32 Tests - `cuda_mma_tf32_test.cc`

Tests TF32 (TensorFloat-32) MMA operations with M16N8K4 shape.

**Key characteristics**:
- TF32 format: 1 sign bit, 8 exponent bits, 10 mantissa bits
- Uses `b32` registers for A/B fragments (not `f32`)
- M16N8K4 required for sm_90 (Hopper), M16N8K8 only for sm_80-89 (Ampere)

**Fragment distribution**: Each thread holds 2 TF32 for A, 1 for B, 4 F32 for C/D

### S8 Tests - `cuda_mma_s8_test.cc`

Tests S8 (signed 8-bit integer) MMA operations with M16N8K16 and M16N8K32 shapes.

**Key characteristics**:
- Integer accumulation in S32 format
- Includes saturation tests (clamp to [-128, 127])
- Tests both signed (S8) and unsigned (U8) variants

**Fragment distribution**: Each thread holds 4 S8 for A, 2 for B, 4 S32 for C/D

## Running Tests

Use the full SM120 configuration:

```bash
# Run all MMA tests
./test/run_tests.sh -c SM120_RTX5090 run "*MMA*"

# Run specific data type tests
./test/run_tests.sh -c SM120_RTX5090 run "CudaMmaF16Test*"
./test/run_tests.sh -c SM120_RTX5090 run "CudaMmaBf16Test*"
./test/run_tests.sh -c SM120_RTX5090 run "CudaMmaTf32Test*"
./test/run_tests.sh -c SM120_RTX5090 run "CudaMmaS8Test*"

# Run specific test case
./test/run_tests.sh -c SM120_RTX5090 run "CudaMmaF16Test.AllOnesTest"
```

**Configuration**: `SM120_RTX5090` (default)
- 170 SMs
- 16 memory controllers
- Complete hardware simulation

## Test Structure

Each MMA integration test follows this pattern:

```cpp
class MMATestFixture : public ::testing::Test {
protected:
    // Test parameters (M, N, K dimensions)
    // Host and device memory pointers

    void SetUp() override { /* Allocate memory */ }
    void TearDown() override { /* Free memory */ }
    void compute_reference() { /* CPU reference implementation */ }
    void run_mma_kernel() { /* Execute through GPGPU-Sim */ }
};

TEST_F(MMATestFixture, TestCase) {
    // 1. Initialize inputs
    // 2. Compute CPU reference
    // 3. Run GPU kernel through simulator
    // 4. Validate results with EXPECT_NEAR
}
```

## Validation Strategy

### Tolerance Values

Different data types require different tolerances due to precision limitations:

- **F16**: `1e-3f` - 10-bit mantissa precision
- **BF16**: `1e-2f` - 7-bit mantissa precision (less precise than F16)
- **TF32**: `1e-3f` to `1e-2f` - 10-bit mantissa + accumulation errors
- **S8**: `0` - Exact integer arithmetic (no tolerance)

### CPU Reference Implementation

Each test computes expected results on CPU:
1. Apply data type rounding (F16, BF16, TF32) to simulate precision
2. Perform matrix multiplication: D = A × B + C
3. Apply saturation if needed (integer types)
4. Compare GPU result against CPU reference with appropriate tolerance

## Expected Results

All 30 MMA tests should pass:
- **F16 tests**: 16 test cases
- **BF16 tests**: 5 test cases
- **TF32 tests**: 5 test cases
- **S8 tests**: 4 test cases

**Status**: ✅ All tests passing (verified as of commit 1d88e1c7)

## Adding New MMA Tests

When adding support for new MMA shapes or data types:

1. **Create test file**: `cuda_mma_<type>_test.cc` in this directory
2. **Define test fixture**: Inherit from `::testing::Test`
3. **Implement CPU reference**: With proper type rounding for the data type
4. **Write test cases**: At least 5 tests (all-ones, zeros, identity, random, edge cases)
5. **Update this README**: Document new test file and expected results

## Troubleshooting

### Test Compilation Errors

```bash
# Verify environment setup
source setup.sh && source setup_environment

# Rebuild tests
./test/run_tests.sh build test --target sm120 --group integration
```

### Test Execution Failures

If tests compile but fail:
- Check tolerance values match data type precision
- Verify fragment distribution matches PTX ISA specification
- Review CPU reference implementation for correct type rounding
- Compare against hardware results if available

### All Results Zero

If test results are all zeros:
- Verify MMA shape is handled in dispatcher (src/gpgpu-sim/flash/mma/tensor_mma.cc)
- Check shape enum mapping from parser to implementation
- Ensure fragment distribution uses correct thread-to-element formulas

## References

- Test framework documentation: `../README.md`
- MMA implementation: `src/gpgpu-sim/flash/mma/`
- MMA design documentation: `docs/mma_instructions.md`
- Testing instructions: `docs/testing-instructions.md`
- PTX ISA: [MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
