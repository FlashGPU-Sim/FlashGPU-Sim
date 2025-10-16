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
| `./run_tests.sh clean` | Clean build files |

## Running Individual Tests

```bash
# Run specific test suite
./run_tests.sh run CudaVectorAdd

# Run specific test case  
./run_tests.sh run BasicVectorAddition

# Run with verbose output
./run_tests.sh -v run MBarrierTest
```

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

Tests automatically run from `test/run/SM120_RTX5090/` directory with GPGPU-Sim configuration files. The run directory and configuration are automatically created when building or running tests.

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