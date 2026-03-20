# GPGPU-Sim Test Framework

Unit and integration tests for GPGPU-Sim using Google Test.

## Quick Start

### Simulator Mode (Default)
```bash
cd test
source ../setup.sh && source ../setup_environment  # Required for simulator
./run_tests.sh setup    # One-time setup
./run_tests.sh test     # Build and run all tests with GPGPU-Sim
```

### Native GPU Mode (Test Validation)
```bash
cd test
# In a CLEAN shell (no setup_environment sourced)
./run_tests.sh setup    # One-time setup
./run_tests.sh test     # Build and run tests on real GPU
```

**Note:** The test runner automatically detects simulator vs. native mode based on environment variables (`GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN`) and `LD_LIBRARY_PATH` contents.

## Commands

| Command | Description |
|---------|-------------|
| `./run_tests.sh setup` | Download Google Test |
| `./run_tests.sh build` | Build all tests (auto-creates run directory, skips simulator build in native mode) |
| `./run_tests.sh test` | Run all tests (auto-creates run directory, skips simulator build in native mode) |
| `./run_tests.sh test <pattern>` | Run specific test |
| `./run_tests.sh bench <pattern>` | Run microbenchmark test |
| `./run_tests.sh refresh` | Refresh run directory and configuration |
| `./run_tests.sh list` | List available tests |
| `./run_tests.sh list-configs` | List available GPU configurations |
| `./run_tests.sh clean` | Clean build files |

**Mode Detection:** The `build`, `test`, and `bench` commands automatically detect whether to operate in simulator mode (requires GPGPU-Sim library) or native GPU mode (skips simulator build) based on:
- Presence of `GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN` environment variable
- Absence of simulator library paths in `LD_LIBRARY_PATH`

### GPU Configuration Options

| Command | Description |
|---------|-------------|
| `./run_tests.sh test --config <name>` | Run tests with specific GPU configuration |
| `./run_tests.sh test -c <name>` | Short form of --config |
| `./run_tests.sh list-configs` | Show all available configurations |

## Running Individual Tests

```bash
# Run specific test suite
./run_tests.sh test CudaVectorAdd

# Run specific test case
./run_tests.sh test BasicVectorAddition

# Run with verbose output
./run_tests.sh -v test MBarrierTest
```

## GPU Configurations

The test framework supports multiple GPU configurations for different testing scenarios:

### Available Configurations

- **SM120_RTX5090** (default): Full RTX 5090 configuration with 170 SMs
- **SM120_RTX5090_REDUCED**: Lightweight configuration with 1 SM, 1 L2, 1 DDR (faster for quick tests)

### Using Different Configurations

```bash
# Run with default configuration (SM120_RTX5090)
./run_tests.sh test

# Run with reduced configuration (faster)
./run_tests.sh test --config SM120_RTX5090_REDUCED
./run_tests.sh test -c SM120_RTX5090_REDUCED

# List all available configurations
./run_tests.sh list-configs

# Run specific test with custom config
./run_tests.sh test -c SM120_RTX5090_REDUCED CudaVectorAdd
```

### When to Use Each Configuration

- **SM120_RTX5090**: Full validation, performance testing, comprehensive test runs
- **SM120_RTX5090_REDUCED**: Quick smoke tests, development iterations, CI/CD pipelines

### Configuration Matrix

For detailed test-to-configuration mapping and test status, see:
- **[docs/test-configuration-matrix.md](../docs/test-configuration-matrix.md)** - Complete test suite reference

**Recommended config rule:**
- MMA tests (single-block functionality) → `SM120_RTX5090_REDUCED`
- Other tests (multi-block validation) → `SM120_RTX5090`

### Adding Custom Configurations

1. Create config directory: `configs/YOUR_CONFIG_NAME/`
2. Add required files: `gpgpusim.config` and `config_*.icnt`
3. Configuration will be automatically detected by `./run_tests.sh list-configs`

## Test Organization

```
test/src/
├── unit/              # Unit tests (basic functionality, no simulator component interaction)
│   ├── basic_test.cc           # gtest template
│   ├── bulk_group_test.cc      # Tests bulk_group_manager_t
│   └── host_tensormap_test.cc  # Tests cuTensorMapEncodeTiled
│
├── integration/       # Integration tests (test interaction between GPGPU-Sim components)
│   ├── integration_test.cc     # Integration test template
│   ├── cuda_vector_add_test.cc
│   ├── mbarrier_test.cc
│   ├── cuda_tma_test.cc
│   ├── cuda_ld_st_matrix_test.cc
│   └── mma/                    # MMA integration tests
│
├── standalone/        # Dev tests (decoupled from simulator)
│   ├── tensor_mma_test.cc      # Type conversion / saturation tests
│   ├── tma_swizzle_test.cc     # TMA swizzle pattern verification
│   ├── mbarrier_test.cc        # Mbarrier placeholder tests
│   └── cuda_tensor_mma_test.cc # CPU reference MMA validation
│
└── microbench/        # Performance microbenchmarks (separate binaries)
    ├── mma/
    │   ├── README.md
    │   ├── inst_latency_bench.cc
    │   └── mma_issue_bench.cc
    └── mbarrier/
        ├── README.md
        ├── mbarrier_trywait_latency_bench.cc
        ├── mbarrier_visibility_bench.cc
        └── mbarrier_contention_bench.cc
```

**`unit/`** and **`integration/`** are compiled into `run_all_tests` (via `make test`).
**`standalone/`** dev tests are a separate binary `run_dev_tests` (via `make dev`).
**`microbench/`** tests are separate binaries (via `make bench`).

## Writing Tests

### Basic Test
```cpp
#include <gtest/gtest.h>

TEST(TestSuite, TestName) {
    EXPECT_EQ(expected, actual);
    ASSERT_TRUE(condition);
}
```

### Test Fixture
```cpp
class MyTest : public ::testing::Test {
protected:
    void SetUp() override { /* setup */ }
    void TearDown() override { /* cleanup */ }
};

TEST_F(MyTest, TestName) {
    EXPECT_TRUE(some_condition);
}
```

### CUDA Integration Test
```cpp
// Use real CUDA runtime API
float* d_ptr;
ASSERT_TRUE(cudaSafeMalloc((void**)&d_ptr, size));
ASSERT_TRUE(cudaSafeMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
// Launch kernel and verify results
cudaSafeFree(d_ptr);
```

## Test Environment

Tests run from `test/run/<CONFIG_NAME>/` directory with GPGPU-Sim configuration files. The run directory and configuration are automatically created when building or running tests.

- Default configuration: `SM120_RTX5090`
- Configuration files are copied from `configs/<CONFIG_NAME>/` to the run directory
- Test binaries are shared across all configurations (no duplication)
- Each configuration has its own isolated run directory

## Build System

- Unit tests: Compiled with g++
- Integration tests: Compiled with nvcc for CUDA support
- Automatic dependency management
- Links with CUDA runtime for integration tests

## Requirements

- C++17 compiler
- NVCC (for CUDA tests)
- Google Test (auto-downloaded)
- CUDA runtime library
