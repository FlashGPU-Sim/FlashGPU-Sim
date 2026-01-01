# Testing Instructions for GPGPU-Sim

## Prerequisites

Before running tests, ensure the environment is properly configured:

```bash
# Source environment setup scripts (REQUIRED before testing)
source setup.sh
source setup_environment
```

**Note:** These commands must be run before `make test` or any compilation commands.

## Running Tests

```bash
# Run all tests with default configuration (SM120_RTX5090)
./test/run_tests.sh run

# Run tests with reduced configuration (faster, less memory)
./test/run_tests.sh run --config SM120_RTX5090_REDUCED
./test/run_tests.sh run -c SM120_RTX5090_REDUCED

# List available GPU configurations
./test/run_tests.sh list-configs
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
./test/run_tests.sh run --config SM120_RTX5090_REDUCED

# Run specific test with config
./test/run_tests.sh run -c SM120_RTX5090_REDUCED CudaVectorAdd
```

## Test Organization

Tests for MMA (Matrix Multiply-Accumulate) instructions:
- Unit tests: `tests/src/unit/tensor_mma_test.cc` (if exists)
- Integration tests: `tests/src/integration/cuda_tensor_mma_test.cc` (if exists)

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
