# GPGPU-Sim Test Framework

Unit and integration tests for GPGPU-Sim using Google Test.

## Quick Start

### Simulator Mode

```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
cd test
./run_tests.sh setup
./run_tests.sh run test --target sm120 --group unit
./run_tests.sh run test --target sm120 --group integration
```

### Native GPU Mode (Test Validation)

```bash
cd test
# Start from a clean shell without setup_environment.
./run_tests.sh setup
./run_tests.sh run test --target sm120 --group integration
```

The runner detects simulator versus native mode from
`GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN` and `LD_LIBRARY_PATH`. It also rejects a
GPU configuration whose compute capability does not match the selected target.

## Commands

| Command | Description |
|---------|-------------|
| `./run_tests.sh setup` | Download Google Test |
| `./run_tests.sh build SUITE ...` | Build one selected target/group |
| `./run_tests.sh run SUITE ... [FILTER]` | Build and run one selected target/group |
| `./run_tests.sh refresh` | Refresh run directory and configuration |
| `./run_tests.sh list` | List the complete suite/target/group hierarchy and modes |
| `./run_tests.sh list-configs` | List available GPU configurations |
| `./run_tests.sh clean` | Clean build files |

The public selection model is:

```text
action -> suite -> target -> group -> optional mode/filter
```

`test`, `microbench`, and `trace` default to target `sm120`; `analysis` always
requires `--target fa2|fa3`. Every target requires `--group`. Use
`./run_tests.sh help` for the accepted groups and examples.

## Example Invocations

```bash
# SM120 correctness
./run_tests.sh run test --target sm120 --group integration CudaVectorAdd

# Hopper instruction and FlashAttention smoke tests
./run_tests.sh run test --target sm90 --group instructions WgmmaF16
./run_tests.sh run test --target sm90 --group fa2-smoke
./run_tests.sh run test --target sm90 --group fa3-smoke

# FA analysis; breakdown/scaling/concurrency require a compile-time mode
./run_tests.sh run analysis --target fa2 --group breakdown --mode only_mma
./run_tests.sh build analysis --target fa3 --group scaling --mode all

# Microbench and trace
./run_tests.sh run microbench --target sm120 --group mbarrier
./run_tests.sh build microbench --target sm90 --group tma
./run_tests.sh run trace --target sm120 --group gpt2 flash_attn

# Explicit compatible configuration
./run_tests.sh -c SM120_RTX5090 run test --target sm120 \
  --group integration CudaVectorAdd
```

Standalone calibration groups such as `microbench/sm120/memory` and
`microbench/sm90/{cp-async,mma,tma}` are build-only in the generic runner.
Use their local Makefiles to supply benchmark-specific runtime arguments.

## GPU Configurations

- **SM120_RTX5090**: default SM120 configuration.
- **SM90_H100**: default Hopper configuration.

Additional configuration directories are discovered automatically when they
contain `gpgpusim.config`. Run `./run_tests.sh list-configs` to list them.

Use `SM120_RTX5090` for SM120 tests and `SM90_H100` for Hopper tests unless
an experiment explicitly requires another architecture-specific configuration.

## Test Organization

The supported runner hierarchy is `suite / target / group / optional mode`:

```text
run_tests.sh
│
├── test
│   ├── sm120
│   │   ├── unit
│   │   │   ├── basic
│   │   │   ├── bulk-group
│   │   │   └── host-tensormap
│   │   └── integration
│   │       ├── basic-integration
│   │       ├── vector-add
│   │       ├── ldmatrix/stmatrix
│   │       ├── TMA
│   │       ├── multidimensional-TMA
│   │       ├── mbarrier
│   │       ├── mbarrier-sanity
│   │       └── MMA: F16/BF16/TF32/S8
│   └── sm90
│       ├── instructions
│       │   ├── named-barrier
│       │   └── WGMMA
│       ├── fa2-smoke
│       └── fa3-smoke
│
├── analysis
│   ├── fa2
│   │   ├── small
│   │   ├── medium
│   │   ├── large
│   │   ├── breakdown                 [mode required]
│   │   ├── scaling                   [mode required]
│   │   └── concurrency               [mode required]
│   └── fa3
│       ├── small
│       ├── medium
│       ├── large
│       ├── breakdown                 [mode required]
│       ├── scaling                   [mode required]
│       └── concurrency               [mode required]
│
├── microbench
│   ├── sm120
│   │   ├── mbarrier
│   │   │   └── try-wait latency/visibility
│   │   ├── mma
│   │   │   ├── instruction-latency
│   │   │   ├── issue/ILP
│   │   │   ├── accept-queue
│   │   │   └── saturation
│   │   └── memory
│   │       ├── global-load-bandwidth
│   │       ├── L2-latency
│   │       ├── L2/HBM-interleave
│   │       └── L2-partition-latency
│   └── sm90
│       ├── cp-async
│       │   ├── latency
│       │   ├── issue-scope
│       │   └── PTX-probe
│       ├── mma
│       │   ├── accept-queue
│       │   └── saturation
│       ├── tma
│       │   ├── completion-latency
│       │   ├── descriptor-setup
│       │   └── FA3-M3
│       └── wgmma
│           ├── async-latency
│           ├── FP16-core-sweep
│           ├── N16-chain
│           ├── RF-bandwidth
│           └── softmax-mix
│
└── trace
    └── sm120
        └── gpt2
            ├── embedding
            ├── GELU
            ├── flash-attention
            ├── layernorm
            ├── residual-add
            └── linear
```

`small`, `medium`, and `large` run directly. Analysis groups marked above
require one compile-time mode; `mode=all` is build-only. The legacy `dev`
suite is not part of the supported hierarchy. `test/triton_trace/` remains an
independent capture and offline-validation tool and is not managed by
`run_tests.sh`.

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

- Unit, integration, and GPU microbench sources are compiled with NVCC
- Google Test support objects and final binaries are built with the host C++ compiler
- Automatic dependency management
- Test binaries link with the CUDA runtime

## Requirements

- C++17 compiler
- NVCC (for CUDA tests)
- Google Test (auto-downloaded)
- CUDA runtime library
