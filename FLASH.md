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
./run_tests.sh test BasicVectorAddition  # Run specific test
./run_tests.sh test CudaTMATest          # Run TMA-specific tests
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
| Memory Barriers (`mbarrier`) | 🔶 **partial** | Used ops + remote via NoC; `try_wait` timeout dest-pred; objects not smem-backed |
| Tensor Memory Accelerator (TMA) | 🔶 **partial** | Multi-dimensional tensor transfers, tensormap support, integration tests |
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

### Flash Module Architecture

**Location**: `src/gpgpu-sim/flash/`

The flash module implements SM 12.0 (Hopper/Blackwell) modern CUDA features for accurate simulation of cutting-edge GPU capabilities. It provides core infrastructure for advanced synchronization, memory operations, and warp-level primitives.

**File Organization**:
- `tma.{cc,h}`: Tensor Memory Accelerator (TMA) for multi-dimensional tensor transfers
- `mbarrier.{cc,h}`: Memory barrier operations for producer-consumer synchronization
- `elect.{cc,h}`: Leader election primitives for warp specialization
- `ld_st_matrix.{cc,h}`: Matrix load/store operations for efficient tensor core data movement

**Integration Points**:
- PTX parser extensions in `src/cuda-sim/ptx_parser.{y,l}` for instruction recognition
- Instruction handlers in `src/cuda-sim/cuda-sim.cc` for execution dispatch
- Memory subsystem coordination through `mem_fetch_interface` for data transfers

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

### Memory Barrier (`mbarrier`)

**Purpose**: Thread-level memory barriers for producer-consumer synchronization patterns, essential for asynchronous memory operations and warp specialization on Hopper/Blackwell architectures.

**Architecture**: Centralized barrier management through `mbarrier_manager_t` class located in `src/gpgpu-sim/flash/mbarrier.{cc,h}`.

**Core Components**:
- **Parser Integration**: Extended PTX lexer (`ptx.l`) and grammar (`ptx.y`) to recognize `mbarrier` instructions
- **Barrier State Machine**: Each barrier tracks arrival count, TMA transaction coordination, phase-based synchronization, and per-warp waiting queues

**State Structure** (`mbarrier_t`):
```cpp
struct mbarrier_t {
  int m_expected_count;      // Number of threads expected to arrive
  int m_arrived_count;       // Current arrival count
  int m_expected_tx_count;   // Expected TMA transactions
  int m_arrived_tx_count;    // Completed TMA transactions
  int m_phase;               // Parity bit for phase-based synchronization
  std::set<int> m_waiting_warps;  // Warps blocked on this barrier
};
```

**Core Operations**:

```cpp
// Initialize barrier with expected arrival count
void init(gpgpu_sim *gpu, const thread_index_t &thread_index,
          uint64_t addr, int expected_count);

// Arrive at barrier (may release waiting warps if count reached)
std::set<int> arrive(gpgpu_sim *gpu, const thread_index_t &thread_index,
                     uint64_t addr, int arrival_count);

// Wait on barrier with phase/parity check
bool try_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
              uint64_t addr, int parity);

// TMA transaction coordination
void expect_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
               uint64_t addr, int expected_tx_count);
std::set<int> complete_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
                          uint64_t addr, int completed_tx_count);
```

**Synchronization Semantics**:
- Phase-based synchronization allows barrier reuse without re-initialization
- Arrival count accumulates until reaching expected count
- TMA transactions integrate with barrier completion (arrive + tx completion)
- Warps unblock atomically when barrier advances to next phase

**Current Limitations**:
1. **Storage**: Barrier objects live in simulator state, not as 64-bit values in GPU shared memory. Kernels that inspect mbarrier bytes in smem will not see HW-faithful contents.
2. **Cluster wait**: Remote (mapa) arrive / try_wait / expect_tx / complete_tx go through the intra-cluster NoC when `-gpgpu_mbarrier_cluster_enable 1` and `-gpgpu_cluster_noc_enable 1`. `barrier.cluster` / `cluster.sync()` are **not** implemented (see `docs/cluster_noc/programming_model.md`).
3. **Warp granularity**: A blocked thread stalls the entire warp (GPGPU-Sim SIMT).
4. **`try_wait` timeout**: The optional 4th operand is implemented. Dest pred is **true** if the waited phase completed, **false** if the hint expires first. Hint is modeled in sim cycles, not silicon nanoseconds.
5. **Unused PTX**: `test_wait`, `pending_count`, and try_wait without `.parity` are unused in-tree and hard-fail / fail to parse.

### Tensor Memory Accelerator (TMA)

**Purpose**: TMA enables asynchronous multi-dimensional tensor transfers between global and shared memory, essential for high-performance tensor operations on Hopper/Blackwell architectures.

**Core Interfaces**:

```cpp
// Main TMA instruction handler
void handle_tma_inst(const ptx_instruction *pI, ptx_thread_info *thread);

// Tensormap descriptor manipulation (initialize, create, etc.)
void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread);

// TMA unit for asynchronous transfer management
class tma_unit_t {
  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
  void cycle();  // Process pending transfers
  void fill(mem_fetch *mf);  // Handle memory responses
};
```

**Tensormap Descriptor** (`tensormap_descriptor_t`):
- 128-byte aligned structure mirroring CUDA Driver API layout
- Describes multi-dimensional tensor shape (up to 5D: x, y, z, w, v)
- Supports various data types: U8/U16/U32/U64, F16/F32/F64, BF16
- Configurable swizzle modes (none / 32B / 64B / 128B; 96B aborts)
- Out-of-bounds fill modes (zero or NaN)

**Key Operations**:
- `cp.async.bulk.tensor`: Asynchronous tensor copy with mbarrier coordination
- `tensormap.replace`: Modify tensormap descriptor fields
- Address calculation: Handles strides, swizzling, and multi-dimensional indexing

### Elect Instruction

**Purpose**: `elect.sync` implements deterministic leader election for warp specialization patterns (producer-consumer, leader-follower).

**Interface**:
```cpp
void handle_elect_inst(const ptx_instruction *pI, ptx_thread_info *thread);
```

**Instruction Syntax**:
```
elect.sync d|p, membermask
```

**Behavior**:
- Elects the **lowest-numbered active lane** from the membership mask
- Returns elected lane ID in destination register `d`
- Sets predicate `p` to `True` for elected thread, `False` for all others
- Election is **deterministic** and **side-effect free**

**Use Cases**:
- Warp specialization (e.g., one thread handles shared memory allocation)
- Producer-consumer patterns (elect producer from warp)
- Optimized reduction operations (single thread commits result)

### Matrix Load/Store Operations

**Purpose**: `ldmatrix` and `stmatrix` enable efficient data movement between shared memory and registers for tensor core operations, handling the complex lane-to-matrix element mapping automatically.

**Interfaces**:
```cpp
void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst);
void handle_stmatrix_inst(const ptx_instruction *pI, core_t *core,
                          warp_inst_t &inst);
```

**Matrix Layout**:
- **8×8 matrices** distributed across warp lanes
- Each of 32 lanes handles one or more matrix elements
- **Lane mapping**: Lane `i` → Row `i/4`, Columns `(i%4)*2` and `(i%4)*2+1`
- Data type: 16-bit elements (fp16/bf16)

**Implementation Details**:
- Template-based design for code reuse between load/store paths
- Supports multiple matrices per instruction (`.num{1,2,4}`)
- Transpose modes (`.trans`) handled via layout transformation
- Memory coalescing through consecutive lane access patterns

**Example Layout** (single 8×8 matrix):
```
Lane 0-3:   Row 0 (each lane handles 2 consecutive columns)
Lane 4-7:   Row 1
Lane 8-11:  Row 2
...
Lane 28-31: Row 7
```

### Known Limitations

**TMA Subsystem**:
1. **CP.ASYNC sector masking**: Corner cases with non-cacheline-aligned sizes not fully handled
2. **Commit/wait groups**: `cp.async.bulk.commit_group` / `wait_group` are real. Stores join a bulk group; `wait_group N` parks until at most N committed groups remain incomplete. Payload bytes are still written at functional execute (same as TMA loads); the wait is the timing park. `.read` after a tensormap publish with no prior store is an empty wait.
3. **Tensormap**: Used `replace.tile.*` fields and `cp_fenceproxy` are implemented. Unrecognized variants abort. `swizzle_atomicity` only accepts 16B (0).
4. **96B swizzle**: Unused in-tree; named abort (32/64/128B and none work).
5. **Multi-dimensional testing**: 1D–5D coverage is in `docs/testing-instructions.md`
6. **Cluster destination**: `.shared::cluster` is functionally supported; hop timing via NoC when enabled (see Cluster / TMA multicast below)

**Mbarrier Subsystem**:
1. **Storage**: Barrier objects live in simulator state, not as 64-bit GPU shared-memory contents
2. **Remote cluster ops**: mapa’d arrive / try_wait / expect_tx / complete_tx use cluster NoC when `-gpgpu_mbarrier_cluster_enable 1` and NoC is on. Not “CTA-level only.” `barrier.cluster` / `cluster.sync()` remain unimplemented
3. **Warp granularity**: A blocked thread stalls the entire warp
4. **`try_wait` timeout**: Optional 4th operand implemented — dest pred true if the phase completed, false if the hint expires first (hint in sim cycles)
5. **Unused variants**: `test_wait`, `pending_count`, try_wait without `.parity` hard-fail / do not parse
6. **Calibration knobs**: `gpgpu_mbarrier_arrive_latency` / `trywait_latency` are end-to-end TMA+mbarrier fits, not pure HW barrier cost

**Cluster / TMA multicast / DSM / NoC** (see **`docs/cluster_noc/README.md`**):
1. **TB cluster launch**: `cudaLaunchKernelExC` / `__cluster_dims__` co-schedule CTAs onto one physical `simt_core_cluster` (target name: `gpc_t`). Details: `docs/cluster_noc/programming_model.md`.
2. **Peer model**: `.shared::cluster` peers share `cluster_group` / ranks on the same physical cluster / GPC.
3. **Selective multicast**: `.multicast::cluster` + `ctaMask` for data and mbarrier `complete_tx`.
4. **Intra-cluster NoC (today)**: TMA peer data/mbar, DSM, remote mbarrier via delay-line `cluster_noc_t` when `-gpgpu_cluster_noc_enable 1`. Default on for `SM90_H200_REDUCED_CLUSTER16x2`. **Target:** per-GPC flit fabric (`dsm_fabric_t`) in `docs/cluster_noc/dsm_fabric.md`.
5. Prefer `SM120_*_REDUCED_CLUSTER*` for functional tests; H200 reduced (`SM90_H200_REDUCED_CLUSTER16x2`) for functional NoC-on. Published cycle calibration uses `SM90_H200_CLUSTER132` (`docs/cluster_noc/calibration.md`).
6. **`mapa` lifetime**: `mapa` maps only an **active** cluster rank. If the target CTA has exited or the rank was never co-resident, the simulator **aborts** (it does not alias the issuer’s shared memory). Keep the producer CTA alive until consumers finish `mapa`.
7. **DSM `atom`**: CUDA `atomicAdd` on a `mapa` / `map_shared_rank` pointer is a generic PTX `atom.add` against the **owner** CTA’s smem. PTX `red` and `red.async` are unimplemented (`inst_not_implemented`). nvcc 12.8 does not emit `red.shared::cluster` for that C++ path; `red.async` is a separate Hopper mbarrier-completion opcode.
8. **Hang preventers**: a bare peer-smem spin (no mbarrier interest) and a prolonged mix of `bar.sync` + single-thread `try_wait` abort instead of hanging until the test timeout. Not hardware detectors. Parked `try_wait` is exempt (`docs/cluster_noc/programming_model.md`).
9. **Maturity:** Functional track closed except `red`/`red.async`. SM↔SM timing is still a **delay line** (flat hop, job 2046238). Target fabric + scoreboard-complete remote loads: living checklist **`docs/cluster_noc/todos.md`**. Remote DSM stores default to write-on-deliver (`-gpgpu_dsm_store_immediate 0`); NoC-off still writes immediately.

**General**:
- Flash mode multi-threading may have race conditions in certain edge cases
- Validation primarily focused on basic usage patterns; complex corner cases may exhibit deviations from hardware behavior
