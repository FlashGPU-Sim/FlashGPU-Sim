# Flash GPGPU-Sim: Fast and Accurate Modern GPU Simulator

## Philosophy

Traditional GPGPU-Sim faces three critical limitations:

1. **Missing Modern GPU Features**: Essential Hopper/Blackwell features are unsupported including Tensor Memory Accelerator (TMA), memory barriers (`mbarrier`), warp specialization, tensor memory operations, and thread block clusters.
2. **Extreme Simulation Overhead**: Simulation slowdown exceeds 2,000,000x compared to native execution, making large-scale studies impractical.
3. **Limited Validation Infrastructure**: Absence of comprehensive test suites and continuous validation hampers reliable development and deployment.

Flash GPGPU-Sim addresses these challenges through:

1. **Modern Feature Implementation**: Native support for TMA, `mbarrier`, and other cutting-edge GPU capabilities
2. **Multi-threaded Simulation Engine**: OpenMP-based parallelization targeting significant speedup while maintaining accuracy
3. **Comprehensive Testing Framework**: Integration test suite with validated configurations for continuous development

## Get Started

### Environment Setup

Configure CUDA installation paths:

```bash
export CUDA_INSTALL_PATH=/path/to/cuda     # e.g., /usr/local/cuda-12.0
export CUDA_INSTALL_PATH=/path/to/cuda             # e.g., /usr/local/cuda-12.0
source setup_environment
```

### Building GPGPU-Sim

**Multi-threaded version** (Flash mode, default):
```bash
make -j$(nproc)
```

Flash mode is now **enabled by default** (`FLASH=1`). This provides modern GPU feature support and multi-threaded simulation out of the box.

The Flash mode enables:
- `-DFLASH_GPGPU_SIM`: Core Flash features
- `-DFLASH_GPGPU_SIM_OMP`: OpenMP parallelization
- `-fopenmp`: OpenMP compiler support

### Running Tests

After building, validate installation with the test framework:

```bash
cd test
./run_tests.sh setup                    # One-time Google Test setup
./run_tests.sh run BasicVectorAddition  # Run specific test
./run_tests.sh run CudaTMATest          # Run TMA-specific tests
./run_tests.sh list                     # Show available tests
./run_tests.sh help                     # Show all options
```

### Running Custom Applications

Execute CUDA applications following standard GPGPU-Sim workflow:
1. Compile your CUDA application with dynamic `cudart` linking
2. Source the environment: `source setup_environment`
3. Run your binary directly

**Note**: For Flash mode issues, first verify behavior in single-threaded mode to isolate multi-threading related problems.

### Docker Setup

A Docker-based development environment is provided for consistent builds across different systems.

**Quick Start**:

```bash
# Build Docker image
./docker.sh build

# Enter container shell (environment auto-configured)
./docker.sh shell

# Inside container: build directly (Flash mode is default)
make -j$(nproc)
```

For detailed Docker usage, see [docker/README.md](docker/README.md).

## Roadmap

**Legend**: 
- ✅ **finished**: Implementation complete, requires additional testing
- 🔶 **partial**: Core framework implemented, some features pending  
- 🔄 **wip**: Active development in progress
- ⭕ **init**: Planning phase, not yet started
- 🔄 **long-term**: Ongoing continuous development

### Feature Status

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-threaded Simulation | ✅ **finished** | OpenMP-based SM parallelization |
| Memory Barriers (`mbarrier`) | 🔶 **partial** | Parser support, CTA-level sync, idealized implementation |
| Tensor Memory Accelerator (TMA) | 🔄 **wip** | Basic framework, integration testing available |
| Warp Group Async MMA (`wgmma`) | ⭕ **init** | Specification review phase |
| PyTorch Record/Replay | ⭕ **init** | Workload capture framework |
| Mixed Precision MMA | ⭕ **init** | fp16/bf16/fp8 tensor operations |
| 5th Gen MMA (`tcgen05`) | ⭕ **init** | Hopper tensor core integration |
| L2 Cache Partitioning | ⭕ **init** | Coherence protocol with partitioned L2 |
| Better Logging System | ⭕ **init** | Help debugging |
| NCU like report | ⭕ **init** | Easier understanding of the system bottleneck |
| Hopper/Blackwell Validation | 🔄 **long-term** | Hardware validation framework |
| Extended Test Coverage | 🔄 **long-term** | New workloads and benchmarks |

## Implementation Details

### Multi-Threaded Simulation Architecture

**Design Philosophy**: Parallelizes SM (Streaming Multiprocessor) simulation while maintaining memory subsystem serialization to avoid complex coherence issues.

**Technical Implementation**:
- **OpenMP Integration**: Uses `#pragma omp parallel for` to distribute SM simulation across threads
- **Thread-Local State**: Duplicates statistics (`shader_core_stats`) and PTX instruction objects per thread to eliminate data races
- **Selective Synchronization**: Strategic mutex placement (`pthread_mutex_t`) around shared resources:
  - Global simulation state (`g_sim_lock`)
  - Memory subsystem operations  
  - Statistics aggregation

**Compilation Flags**:
- `FLASH_GPGPU_SIM_OMP`: Enables OpenMP code paths
- `-fopenmp`: Links OpenMP runtime library

**Performance Benefits**: We achieved 4-8x simulation speedup on multi-core systems while preserving simulation accuracy.

### Memory Barrier (`mbarrier`) Support

**Architecture**: Centralized barrier management through `mbarrier_manager_t` class located in `src/gpgpu-sim/flash/mbarrier.{cc,h}`.

**Core Components**:
- **Parser Integration**: Extended PTX lexer (`ptx.l`) and grammar (`ptx.y`) to recognize `mbarrier` instructions
- **Barrier State Machine**: Each barrier tracks:
  - Arrival count vs. expected count
  - Phase-based synchronization (parity bit)
  - TMA transaction coordination (`m_expected_tx_count`, `m_arrived_tx_count`)
  - Per-warp waiting queues

**Current Limitations**:
1. **Idealized Implementation**: Barriers exist in simulator memory rather than GPU shared memory
2. **CTA-Level Only**: No support for cluster-level synchronization.
3. **Warp Granularity**: Single blocked thread stalls entire warp (GPGPU-Sim architectural limitation)

