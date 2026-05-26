# Testing Instructions for GPGPU-Sim

## Prerequisites

### Simulator Mode (Default)

Before running tests with GPGPU-Sim simulation, ensure the environment is properly configured:

```bash
# Source environment setup scripts (REQUIRED for simulator mode)
source setup.sh
source setup_environment
```

**Note:** These commands must be run before `./test/run_tests.sh test` or any compilation commands in simulator mode.

### Native GPU Mode

To run tests on real GPU hardware (for test validation), ensure:
- A clean shell environment (no simulator environment variables)
- CUDA-capable GPU installed
- CUDA toolkit installed and available in PATH

**Important:** Do NOT source `setup_environment` before running tests in native GPU mode. The test runner automatically detects a clean environment and skips the simulator build.

## Building Tests

```
source setup.sh && source setup_environment && ./test/run_tests.sh build
```

**Note:** Never directly invoke the built test binary as it does not have GPU configuration files! Always use `run_tests.sh` as the driver.

**Automatic Library Rebuild (Simulator Mode):** When running in simulator mode (after sourcing `setup_environment`), the test runner automatically detects when GPGPU-Sim library is out of date by:
1. Resolving the actual `libcudart.so` path using `find lib -name libcudart.so`
2. Comparing modification times of source files against the library
3. Rebuilding with `make FLASH=1 -j` if needed

This prevents test failures from stale library builds. In native GPU mode, the simulator build is skipped entirely.

## Running Tests

```bash
# Run all tests with default configuration (SM120_RTX5090)
source setup.sh && source setup_environment && ./test/run_tests.sh test

# Run tests with reduced configuration (faster, less memory)
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED test

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
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED test

# Run specific test with config
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED test CudaVectorAdd
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
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED test "*MMA*"
```

## Native GPU Mode (Test Validation)

Run tests on real GPU hardware to validate test correctness without simulator overhead:

```bash
# In a CLEAN shell (no setup_environment sourced)
./test/run_tests.sh build    # Builds tests only (skips simulator build)
./test/run_tests.sh test      # Runs on real GPU
```

**Prerequisites:**
- Clean shell environment (no `GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN` variable)
- No simulator library paths in `LD_LIBRARY_PATH`
- CUDA toolkit and GPU hardware available

**Automatic Detection:** The test runner checks:
1. Whether `GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN` is set
2. Whether `LD_LIBRARY_PATH` contains simulator library paths (e.g., `gpgpu-sim_distribution/lib`)

If both checks pass (clean environment), native GPU mode is activated and the simulator build is skipped.

**Warning:** If you previously sourced `setup_environment` in the current shell, start a new shell session to ensure a clean environment. Unsetting variables manually may leave residual `LD_LIBRARY_PATH` contamination.

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

## TMA (Tensor Memory Accelerator) Testing

TMA instructions (`cp.async.bulk.tensor`) use the Triton-based test workflow to generate PTX kernels and launcher harnesses for simulation.

### TMA Test Requirements

**Stubbed Instructions**: The following instructions are parsed but implemented as NOPs and are **explicitly allowed** in PTX inspection:
- `cp.async.bulk.commit_group` - NOP with debug logging (see FLASH.md:297)
- `cp.async.bulk.wait_group` - NOP with debug logging (see FLASH.md:297)
- `tensormap.cp_fence` - Parsed but not functionally simulated

**Device-Side Tensormap Creation**: All tensormap descriptors must be created on the device side using Triton's `tl.make_tensor_descriptor` within the kernel (not passed as host-side parameters).

**Multi-Dimensional Coverage**: Tests cover 1D-5D tensor operations with corner cases (remainder handling, degenerate dimensions, non-uniform sizes).

### TMA Test Workflow (Triton-based)

1. **Setup Python Environment**:
   ```bash
   python3.12 -m venv test/triton_trace/.venv
   test/triton_trace/.venv/bin/python -m pip install -U pip uv

   UV_CACHE_DIR=/data/wzr/rtl-lib/.uv-cache \
     test/triton_trace/.venv/bin/uv pip install \
     --python test/triton_trace/.venv/bin/python \
     --link-mode hardlink \
     torch triton numpy nvidia-cuda-nvcc
   ```

   Use `nvidia-cuda-nvcc` from the venv when Triton emits PTX 9.1 / `sm_120a`;
   CUDA 12.x `ptxas` cannot assemble those kernels. The validation sweep scripts
   detect the venv CUDA `nvcc` automatically after activating `.venv`.

2. **Modify Test Kernels**: Edit `test/triton_trace/example_tensor_add.py` with test variants

3. **Extract PTX via Triton Tracker**:
   ```bash
   source test/triton_trace/.venv/bin/activate
   python test/triton_trace/examples/example_tensor_add.py
   ```
   Generates launcher artifacts in `test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers/`

4. **Build Launchers**:
   ```bash
   make -f kernel_add_1d_launch2_Makefile  # In launchers directory
   ```

5. **Execute Under GPGPU-Sim**:
   ```bash
   # CRITICAL: Source environment first (otherwise runs on real GPU!)
   source setup.sh && source setup_environment
   cd test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers
   ./kernel_add_1d_launch2
   ```

6. **Validate Output**: Check tensor addition produces correct results

For real-GPU replay or NCU profiling, do not source `setup_environment`. For
GPGPU-Sim runs, require both `Validation PASSED` and simulator counters such as
`gpu_tot_sim_cycle`; validation alone can also happen on the real CUDA runtime.

**TMA Test Status** (issue #31):
- ✅ 1D TMA: PASSED (8,192 elements validated)
- ✅ 2D TMA: PASSED (regression test)
- ✅ 3D TMA: PASSED (262,144 elements validated)
- ✅ 4D TMA: PASSED (1,048,576 elements validated)
- ✅ 5D TMA: PASSED (1,048,576 elements validated)

**Implementation Notes**: Multi-dimensional support (1D-5D) includes PTX parser support for 5-element vectors, L2 cache integration with byte/sector masking, and proper handling of sector-subdivided and late responses.

## CI/CD Testing

### Continuous Integration

The repository includes a GitHub Actions workflow that runs automated tests on pull requests. The CI workflow uses the reduced GPU configuration for fast feedback:

```bash
# CI workflow runs this command
./test/ci/run_ci_tests.sh
```

**CI configuration:**
- Uses `SM120_RTX5090_REDUCED` for resource efficiency
- Runs in a minimal Docker container (`docker/Dockerfile.ci`)
- Automatically triggered on PR approval

### Local Workflow Validation with act

You can test the CI workflow locally using [act](https://nektosact.com):

```bash
# Prerequisites
# 1. Install act: https://nektosact.com/installation/index.html
# 2. Configure .actrc (already included in repository)

# Run the PR test workflow locally
act
```

The `.actrc` file configures Docker socket binding and runner image for local workflow execution.

## Troubleshooting

If you see "Install CUDA Toolkit and set CUDA_INSTALL_PATH":
- The build system expects CUDA to be installed
- Set the environment variable: `export CUDA_INSTALL_PATH=/path/to/cuda`
- Or install CUDA Toolkit for your platform

If tests don't exist yet:
- Tests may be in untracked files (check `tests/` directory)
- Tests may need to be created according to test strategy in docs/mma_instructions.md

**TMA tests run on real GPU instead of simulation**:
- Ensure `setup_environment` is sourced before running launcher executables
- Verify "GPGPU-Sim version" message appears at startup
