# Testing Instructions for GPGPU-Sim

## Prerequisites

Before running tests, ensure the environment is properly configured:

```bash
# Source environment setup scripts (REQUIRED before testing)
source setup.sh
source setup_environment
```

**Note:** These commands must be run before `make test` or any compilation commands.

## Building Tests

```
source setup.sh && source setup_environment && ./test/run_tests.sh build
```

**Note:** Never directly invoke the built test binary as it does not have GPU configuration files! Always use `run_tests.sh` as the driver.

**Automatic Library Rebuild:** The test runner automatically detects when GPGPU-Sim library (`lib/gcc-*/libcudart.so`) is out of date compared to source files and rebuilds it with `make FLASH=1 -j` before building tests. This prevents test failures from stale library builds.

## Running Tests

```bash
# Run all tests with default configuration (SM120_RTX5090)
source setup.sh && source setup_environment && ./test/run_tests.sh run

# Run tests with reduced configuration (faster, less memory)
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED run

# List available GPU configurations
source setup.sh && source setup_environment && ./test/run_tests.sh list-configs
```

## GPU Configurations

The test framework supports multiple GPU configurations:

- **SM120_RTX5090** (default): Full RTX 5090 simulation (170 SMs, 16 memory controllers)
  - Use for: Complete validation, performance testing, final verification
  - Resource usage: High memory and time

- **SM120_RTX5090_REDUCED**: Lightweight configuration (1 SM, 1 L2, 1 DDR)
  - Use for: Quick tests, development iterations, continuous integration
  - Resource usage: Low memory and time

**Selecting a configuration:**
```bash
# Explicit config selection
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED run

# Run specific test with config
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED run CudaVectorAdd
```

## Listing Available Tests

List all available test cases using GoogleTest's `--gtest_list_tests`:
```bash
source setup.sh && source setup_environment && ./test/run_tests.sh list
```

This displays the actual test suite and test case names from the compiled test binary.

## Select Test Cases

**Note:** You can pass test name (with regex matches), which will be passed to the test binary as `--gtest_filter`. No need to pass in `--gtest_filter` in the command line of `run_tests.sh`.
```bash
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED run "*MMA*"
```

## Test Status & Configuration Matrix

For a comprehensive view of all test suites, their recommended configurations, and current status, see:
- **[Test Configuration Matrix](test-configuration-matrix.md)** - Complete test suite reference

**Quick reference:**
- MMA tests → Use `SM120_RTX5090_REDUCED` (single-block functionality tests)
- Other tests → Use `SM120_RTX5090` (full validation)

**Excluded tests:** Currently, 2 tests are excluded due to unimplemented features:
- `TMA.CPAsyncMethod` - Uses unimplemented `cp.async` instruction
- `TMA.PerformanceComparison` - Internally calls CPAsyncMethod

These exclusions are hard-coded in `test/run_tests.sh` and are automatically skipped during test runs.

## Test Organization

Tests for MMA (Matrix Multiply-Accumulate) instructions:
- Integration tests: `test/src/integration/mma/cuda_mma_*.cc`
  - F16 tests: `test/src/integration/mma/cuda_mma_f16_test.cc`
  - BF16 tests: `test/src/integration/mma/cuda_mma_bf16_test.cc`
  - TF32 tests: `test/src/integration/mma/cuda_mma_tf32_test.cc`
  - S8/U8 tests: `test/src/integration/mma/cuda_mma_s8_test.cc`

## Expected Test Results (Issue #18 Implementation)

**After Step 1-2 (Infrastructure):** 12/25 tests passing (helpers validated)
**After Step 3 (M16N8K8 F16/F32):** 18/25 tests passing (core functionality)
**After Step 4 (All shapes/types):** 23/25 tests passing (comprehensive coverage)
**After Step 5 (Load/Store):** 25/25 tests passing (complete) ✅

## Build Requirements

- CUDA Toolkit (set CUDA_INSTALL_PATH if needed)
- GCC/G++ compiler
- Make build system

## Troubleshooting

If you see "Install CUDA Toolkit and set CUDA_INSTALL_PATH":
- The build system expects CUDA to be installed
- Set the environment variable: `export CUDA_INSTALL_PATH=/path/to/cuda`
- Or install CUDA Toolkit for your platform

If tests don't exist yet:
- Tests may be in untracked files (check `tests/` directory)
- Tests may need to be created according to test strategy in docs/mma_instructions.md
