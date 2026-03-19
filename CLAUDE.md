# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **FlashGPU-Sim**, a fork of GPGPU-Sim that adds modern GPU features (Hopper/Blackwell SM 12.0) and multi-threaded simulation capabilities. It simulates NVIDIA GPUs at the cycle level for CUDA/OpenCL workloads.

**Key distinction**: Flash mode (`FLASH=1`) is now the default build mode and includes:
- Modern GPU features (TMA, mbarrier, tensor cores)
- OpenMP-based multi-threaded simulation (4-8x speedup)
- Comprehensive test framework

## Build Commands

### Environment Setup (Required First)
```bash
# MUST run before any build/test commands
source setup.sh && source setup_environment
```

**Important**: `setup.sh` sets `CUDA_INSTALL_PATH`, and `setup_environment` configures GPGPU-Sim paths. Both must be sourced before compilation or testing.

### Building
```bash
# Full rebuild with Flash features (default)
source setup.sh && source setup_environment && make FLASH=1 -j

# Clean build
make clean && make FLASH=1 -j
```

**Note**: Flash mode is now default. The `FLASH=1` flag explicitly ensures modern GPU features are enabled.

## Testing Commands

### Build Tests (Simulator Mode)
```bash
source setup.sh && source setup_environment && ./test/run_tests.sh build
```

**Critical**: The test runner automatically rebuilds GPGPU-Sim library if source files are newer than the resolved `libcudart.so` path (found via `find lib -name libcudart.so`). Never bypass this by running test binaries directly.

### Native GPU Mode (Test Validation)
To run tests on real GPU hardware without simulator overhead:

```bash
# In a CLEAN shell (no setup_environment sourced)
./test/run_tests.sh build    # Builds tests only
./test/run_tests.sh test      # Runs on real GPU
```

**Prerequisites**: Clean environment (no `GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN` and no simulator paths in `LD_LIBRARY_PATH`). Start a new shell if you previously sourced `setup_environment`.

### Run Tests
```bash
# Run all tests (slow - use sparingly)
./test/run_tests.sh test

# Run specific test pattern (RECOMMENDED)
./test/run_tests.sh test "*MMA*"              # All MMA tests
./test/run_tests.sh test "cuda_mma_s8_test"   # Specific test suite

# Use reduced configuration for faster iteration
./test/run_tests.sh -c SM120_RTX5090_REDUCED test "*MMA*"

# Run microbenchmarks (separate binaries, not included in 'test')
./test/run_tests.sh bench "*MMAIssue*"         # MMA issue gap benchmarks
./test/run_tests.sh bench "*InstLatency*"      # Instruction latency benchmarks

# List available tests
./test/run_tests.sh list

# List GPU configurations
./test/run_tests.sh list-configs
```

**`test` vs `bench`**: `test` runs verification tests (unit + integration) from `run_all_tests`. `bench` runs microbenchmarks from separate binaries (`build/bin/*_bench`). They are independent — `test` never runs microbenchmarks.

**Test selection**: Use glob patterns passed directly as arguments (internally converted to `--gtest_filter`). Do NOT pass `--gtest_filter` manually.

**GPU configurations**:
- `SM120_RTX5090` (default): Full simulation, high memory/time
- `SM120_RTX5090_REDUCED`: Lightweight (1 SM), fast iteration

### Running Single Tests During Development
```bash
# Pattern: ./test/run_tests.sh -c <config> test "<pattern>"
source setup.sh && source setup_environment && ./test/run_tests.sh -c SM120_RTX5090_REDUCED test "MMAS8M16N8K16*"
```

## Code Architecture

### High-Level Structure

```
src/
├── cuda-sim/          # PTX functional simulation (front-end)
│   ├── ptx.l          # PTX lexer (instruction tokenization)
│   ├── ptx.y          # PTX parser (grammar, IR construction)
│   ├── ptx_ir.{h,cc}  # PTX instruction representation
│   ├── opcodes.h      # Opcode definitions (MMA_OP, TENSOR_MMA_OP, etc.)
│   ├── instructions.cc # Instruction execution handlers
│   └── cuda-sim.cc    # CUDA runtime API simulation
│
├── gpgpu-sim/         # Timing simulation (back-end)
│   ├── shader.{h,cc}  # SM pipeline (shader_core_ctx class)
│   ├── gpu-sim.{h,cc} # GPU-level simulation (gpgpu_sim class)
│   ├── gpu-cache.cc   # Cache hierarchy (L1/L2)
│   ├── dram.cc        # Memory controller
│   └── flash/         # Modern GPU features (Flash extensions)
│       ├── tensor_mma.{h,cc}    # MMA instruction implementation
│       ├── tma.{h,cc}           # Tensor Memory Accelerator
│       ├── mbarrier.{h,cc}      # Memory barriers
│       ├── ld_st_matrix.{h,cc}  # ldmatrix/stmatrix
│       └── elect.{h,cc}         # Warp leader election
│
├── intersim2/         # Interconnect network simulator
└── accelwattch/       # Power model

test/
├── run_tests.sh       # Test driver (ALWAYS use this, not raw binaries)
└── src/integration/   # Integration tests (end-to-end CUDA kernels)
    ├── cuda_mma_f16_test.cc   # F16 MMA tests
    ├── cuda_mma_bf16_test.cc  # BF16 MMA tests
    ├── cuda_mma_tf32_test.cc  # TF32 MMA tests
    └── cuda_mma_s8_test.cc    # S8/U8 MMA tests

configs/               # GPU microarchitecture configurations
docs/                  # Technical documentation
```

### Execution Flow: PTX Instruction → Hardware Simulation

1. **PTX Parsing** (`src/cuda-sim/`):
   - `ptx.l` tokenizes PTX assembly (e.g., `mma.sync.aligned.m16n8k16` → tokens)
   - `ptx.y` parses grammar, creates `ptx_instruction` IR objects
   - `opcodes.h` defines opcode enums (e.g., `TENSOR_MMA_OP`)

2. **Functional Execution** (`src/cuda-sim/instructions.cc`):
   - `ptx_thread_info::ptx_exec_inst()` dispatches based on opcode
   - For MMA: routes to `flash_gpgpu_sim::tensor_mma_impl()`
   - Computes functional result (matrix multiply-accumulate)

3. **Timing Simulation** (`src/gpgpu-sim/shader.cc`):
   - `shader_core_ctx` models SM pipeline stages (fetch, decode, execute, writeback)
   - Tracks instruction latencies, resource conflicts, memory accesses
   - Interacts with cache hierarchy (`gpu-cache.cc`) and memory (`dram.cc`)

### Flash Namespace (`src/gpgpu-sim/flash/`)

**Purpose**: Isolated implementation of modern GPU features to avoid breaking existing GPGPU-Sim code.

**Key Design Principle**: MMA (modern PTX `mma.sync`) is **completely separate** from WMMA (legacy `wmma.*`):
- Different opcodes: `TENSOR_MMA_OP` vs `MMA_OP`
- Different namespaces: `flash_gpgpu_sim::` vs global scope
- Different implementations: `flash/tensor_mma.cc` vs `instructions.cc:mma_impl()`

**MMA Implementation Architecture**:
- **Main dispatcher**: `tensor_mma_impl()` routes by data type
- **Type-specific functions**:
  - `tensor_mma_f16_impl()` - F16/BF16 floating-point
  - `tensor_mma_f16_m8n8k4_impl()` - F16/BF16 M8N8K4 (4-computation architecture)
  - `tensor_mma_tf32_impl()` - TF32 floating-point
  - `tensor_mma_s8_impl()` - S8/U8 M16N8K16/K32
  - `tensor_mma_s8_m8n8k16_impl()` - S8/U8 M8N8K16 (specialized for M=8)
- **Fragment distribution**: Each shape has specific thread-to-matrix-element mapping
- **Load/Store**: `tensor_mma_ld_impl()`, `tensor_mma_st_impl()`

**Adding New MMA Shapes** (typical workflow):
1. Add opcode enum to `src/cuda-sim/opcodes.h` (e.g., `MMA_M8N8K16 = 106`)
2. Add lexer token to `src/cuda-sim/ptx.l` (e.g., `\.m8n8k16`)
3. Add parser mapping in `src/cuda-sim/ptx_ir.cc`
4. Implement function in `src/gpgpu-sim/flash/tensor_mma.cc` with fragment distribution
5. Add test case in `test/src/integration/cuda_mma_*_test.cc`
6. Update documentation in `docs/mma_instructions.md`

### Multi-Threading Architecture (Flash Mode)

**Key Concept**: Parallelizes SM simulation, serializes memory subsystem.

**Implementation**:
- OpenMP `#pragma omp parallel for` across SMs in `gpgpu_sim::cycle()`
- Thread-local statistics (`shader_core_stats`) to avoid data races
- Mutex protection (`g_sim_lock`) for shared state (memory controllers, global stats)
- Compilation flags: `FLASH_GPGPU_SIM_OMP`, `-fopenmp`

**Performance**: 4-8x speedup on multi-core systems while maintaining accuracy.

## Commit Message Format

Follow the tags defined in `docs/git-msg-tags.md`:

```
[tag]: Brief description

Detailed explanation if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Common tags**:
- `[feat]`: New feature
- `[bugfix]`: Bug fix
- `[test]`: Test-only changes (don't use if feat/bugfix also present)
- `[docs]`: Documentation updates
- `[refactor]`: Code refactoring without functionality change
- `[agent.skill]`: Agent skill modifications

**Example**:
```
[feat]: Add S8/U8 MMA support (M16N8K16/K32, M8N8K16) for issue #26

Implements signed/unsigned 8-bit integer MMA shapes with proper
fragment distribution and saturation support.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Key Architectural Concepts

### PTX vs. Hardware ISA
- **PTX**: Virtual ISA (what GPGPU-Sim simulates by default)
- **PTXPlus/SASS**: Native hardware ISA (optional via cuobjdump)
- Most development focuses on PTX functional simulation

### Warp-Level Execution
- 32 threads execute in lockstep (SIMT model)
- MMA instructions are warp-level collectives (all 32 threads participate)
- Fragment distribution maps thread IDs to matrix elements

### Core Classes
- `gpgpu_sim`: Top-level GPU simulation (in `gpu-sim.{h,cc}`)
- `shader_core_ctx`: SM pipeline simulation (in `shader.{h,cc}`)
- `ptx_thread_info`: Per-thread PTX execution state (in `instructions.cc`)
- `core_t`: Abstract hardware model interface (in `abstract_hardware_model.h`)
- `warp_inst_t`: Warp-level instruction representation

### Memory Hierarchy
- **L1 Cache**: Per-SM data cache
- **L2 Cache**: Shared across all SMs
- **DRAM**: Memory controllers with detailed timing models
- **Interconnect**: Network-on-chip via intersim2

## Common Development Patterns

### Adding New PTX Instructions
1. Define opcode in `src/cuda-sim/opcodes.h`
2. Add lexer/parser rules in `src/cuda-sim/ptx.{l,y}`
3. Implement handler in `src/cuda-sim/instructions.cc` or `src/gpgpu-sim/flash/`
4. Add test in `test/src/integration/`
5. Document in `docs/`

### Debugging Functional Simulation
- Use `GPGPU_Context()->debug_printf()` for traced output
- Enable PTX instruction tracing via config options
- Check `ptx_thread_info` state for register values
- Validate against CPU reference implementation in tests

### Working with MMA Instructions
- **Read first**: `docs/mma_instructions.md` for specifications
- **Reference**: `src/gpgpu-sim/flash/tensor_mma.md` for API documentation
- **Fragment distribution**: Derive from PTX ISA documentation or existing implementations
- **Testing strategy**: CPU reference validation + hardware comparison

## Important Notes

### Do NOT
- Run test binaries directly (use `./test/run_tests.sh`)
- Run full test suite frequently (it takes too long - use specific test patterns)
- Modify WMMA code when working on MMA (they're separate instruction families)
- Forget to source `setup.sh && setup_environment` before building/testing

### DO
- Use reduced GPU configs for development iterations
- Run specific test patterns matching your changes
- Add comprehensive tests for new features
- Update documentation when modifying interfaces
- Check existing implementations before adding new code

### Known Limitations
- TMA: Some corner cases not fully handled (see `FLASH.md` for details)
- Mbarrier: Idealized implementation, CTA-level only
- Multi-threading: Potential race conditions in edge cases
- Flash mode: Primarily validated on basic usage patterns

## Documentation References

- **Build instructions**: `docs/build-instructions.md`
- **Testing instructions**: `docs/testing-instructions.md`
- **MMA design**: `docs/mma_instructions.md`
- **Flash features**: `FLASH.md`
- **MMA API**: `src/gpgpu-sim/flash/tensor_mma.md`
- **Flash module overview**: `src/gpgpu-sim/flash/README.md`
- **Original GPGPU-Sim**: `README.md`

## Triton Kernel Trace Testing

This workflow tests GPGPU-Sim against real Triton-generated kernels by comparing simulation results with actual GPU execution.

### Directory Structure

```
test/triton_trace/
├── <test_name>.py                # Triton kernel test script (e.g., example_tma_gemm.py)
├── track_triton_kernels.py       # Kernel tracker utility
└── triton_kernel_tracking/
    └── <test_name>/
        ├── launchers/            # C++ harness, PTX, Makefile
        │   ├── <kernel>_launch1_harness.cu
        │   ├── <kernel>_launch1_kernel.ptx
        │   ├── <kernel>_launch1_Makefile
        │   ├── gpgpusim.config   # Must copy from configs/
        │   └── config_ampere_islip.icnt
        ├── data/                 # Serialized tensors (inputs + expected outputs)
        └── binaries/             # CUBIN/PTX binaries
```

### Step 1: Generate Tracking Files

```bash
# Activate Triton virtual environment (user-specific path)
source <your_triton_venv>/bin/activate

# Run the test script to generate tracking files
python3 test/triton_trace/<test_name>.py
```

**Note**: Triton environment setup is user-specific. Ensure your environment has Triton installed with GPU support.

This creates the `triton_kernel_tracking/<test_name>/` directory with harness, PTX, and test data.

### Step 2: Build the Standalone Launcher

```bash
# Find the generated Makefile and build
make -C test/triton_trace/triton_kernel_tracking/<test_name>/launchers -f <kernel>_launch1_Makefile
```

### Step 3: Copy GPU Config (Required After Regenerating Launchers)

```bash
cp configs/SM120_RTX5090/gpgpusim.config \
   test/triton_trace/triton_kernel_tracking/<test_name>/launchers/

cp configs/SM120_RTX5090/config_ampere_islip.icnt \
   test/triton_trace/triton_kernel_tracking/<test_name>/launchers/
```

**Important**: Must redo this step whenever you regenerate the tracking files (Step 1).

### Step 4: Test on Real GPU (Baseline Validation)

```bash
# In a CLEAN shell (no setup_environment sourced)
test/triton_trace/triton_kernel_tracking/<test_name>/launchers/<kernel>_launch1
```

Expected output: `Validation PASSED`

### Step 5: Test on GPGPU-Sim

```bash
source setup_environment && \
(cd test/triton_trace/triton_kernel_tracking/<test_name>/launchers && ./<kernel>_launch1)
```

**Note**: The executable must run from the `launchers/` directory because GPGPU-Sim looks for `gpgpusim.config` in the current working directory.

### Development Cycle (Modify GPGPU-Sim and Re-test)

```bash
# 1. Modify GPGPU-Sim source code
# 2. Rebuild the simulator
make FLASH=1 -j$(nproc)

# 3. Re-run the test (no need to rebuild the launcher)
source setup_environment && \
(cd test/triton_trace/triton_kernel_tracking/<test_name>/launchers && ./<kernel>_launch1)
```

### Key Files for Debugging

| File | Purpose |
|------|---------|
| `<kernel>_launch1_kernel.ptx` | PTX code being simulated - examine for instruction patterns |
| `<kernel>_launch1_harness.cu` | C++ harness - modify to add debug output or dump intermediate results |
| `data/*.bin` | Input tensors and expected outputs |

## Quick Reference

```bash
# Standard development cycle
source setup.sh && source setup_environment

# Build simulator
make FLASH=1 -j$(nproc)

# Build and run specific tests
./test/run_tests.sh build              # Build all (verif + bench)
./test/run_tests.sh build verif        # Build verification tests only
./test/run_tests.sh build bench        # Build microbenchmarks only
./test/run_tests.sh -c SM120_RTX5090_REDUCED test "MMAS8*"
./test/run_tests.sh bench "*MMAIssue*" # Run microbenchmarks

# Clean rebuild
make clean && make FLASH=1 -j$(nproc)
```
