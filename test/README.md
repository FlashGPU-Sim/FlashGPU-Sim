# GPGPU-Sim Test Framework

Unit and integration tests for GPGPU-Sim using Google Test.

## Quick Start

```bash
cd test
./run_tests.sh setup    # One-time setup
./run_tests.sh run      # Build and run all tests
```

## Commands

| Command | Description |
|---------|-------------|
| `./run_tests.sh setup` | Download Google Test |
| `./run_tests.sh build` | Build all tests (auto-creates run directory) |
| `./run_tests.sh run` | Run all tests (auto-creates run directory) |
| `./run_tests.sh run <test>` | Run specific test |
| `./run_tests.sh refresh` | Refresh run directory and configuration |
| `./run_tests.sh list` | List available tests |
| `./run_tests.sh list-configs` | List available GPU configurations |
| `./run_tests.sh clean` | Clean build files |

### GPU Configuration Options

| Command | Description |
|---------|-------------|
| `./run_tests.sh run --config <name>` | Run tests with specific GPU configuration |
| `./run_tests.sh run -c <name>` | Short form of --config |
| `./run_tests.sh list-configs` | Show all available configurations |

## Running Individual Tests

```bash
# Run specific test suite
./run_tests.sh run CudaVectorAdd

# Run specific test case
./run_tests.sh run BasicVectorAddition

# Run with verbose output
./run_tests.sh -v run MBarrierTest
```

## GPU Configurations

The test framework supports multiple GPU configurations for different testing scenarios:

### Available Configurations

- **SM120_RTX5090** (default): Full RTX 5090 configuration with 170 SMs
- **SM120_RTX5090_REDUCED**: Lightweight configuration with 1 SM, 1 L2, 1 DDR (faster for quick tests)

### Using Different Configurations

```bash
# Run with default configuration (SM120_RTX5090)
./run_tests.sh run

# Run with reduced configuration (faster)
./run_tests.sh run --config SM120_RTX5090_REDUCED
./run_tests.sh run -c SM120_RTX5090_REDUCED

# List all available configurations
./run_tests.sh list-configs

# Run specific test with custom config
./run_tests.sh run -c SM120_RTX5090_REDUCED CudaVectorAdd
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
test/
├── src/unit/           # Unit tests
│   ├── basic_test.cc
│   └── mbarrier_test.cc
└── src/integration/    # Integration tests
    ├── cuda_vector_add_test.cc
    └── integration_test.cc
```

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