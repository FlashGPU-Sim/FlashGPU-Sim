# FlashGPU-Sim Development Notes

This document gives contributors a high-level map of the inherited GPGPU-Sim
structure and the FlashGPU-Sim extensions. It also explains how a PTX
instruction moves from source text to functional and timing simulation.
Component documents provide the detailed behavior of individual instructions
and subsystems.

## Development Setup

From the repository root, configure the CUDA and simulator environment before
building or running simulator-backed tests:

```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
make FLASH=1 -j"$(nproc)"
```

## Build System

### Generated Sources

#### PTX Parser

The PTX parser (`src/cuda-sim/ptx.tab.c`, `ptx.tab.h`) is generated from:

- `src/cuda-sim/ptx.l` (lexer)
- `src/cuda-sim/ptx.y` (parser grammar)

The parser is normally regenerated when these files change.

#### Opcode Definitions

Changes to `src/cuda-sim/opcodes.def` affect generated:

- Opcode switch statements in various files
- Instruction dispatcher logic

#### Regeneration Failures

The build normally refreshes generated sources automatically. If changes to
the PTX lexer/parser inputs or opcode definitions leave stale generated code
or cause compilation errors, perform a clean rebuild:

```bash
make clean
make -j"$(nproc)"
```

### Build Artifacts

On Linux, a successful simulator build places the replacement CUDA runtime at
`lib/gcc-*/cuda-*/release/libcudart.so` (or the corresponding `debug`
directory). Intermediate object files use the matching hierarchy under
`build/gcc-*/cuda-*/`.

## Running Tests

Use `test/run_tests.py` rather than invoking generated test binaries directly.
The runner installs the selected GPU configuration, builds the required
test groups, and distinguishes simulator execution from native-GPU validation.

For example, from the repository root:

```bash
./test/run_tests.py setup
./test/run_tests.py run --arch sm120 --test-group integration CudaVectorAdd
```

The selection hierarchy is:

```text
action -> architecture -> test group -> optional profile/mode/filter
```

Use focused test groups and filters while developing. Run the broader relevant
test group before submitting a change. See the
[Test Framework Guide](../test/README.md) for supported groups and examples.

Native-GPU validation must run from a clean shell in which
`setup_environment` has not been sourced. The test guide explains how the
runner distinguishes native and simulator environments.

## Continuous Integration

Pull requests targeting `flash` run
[`test/ci/run_ci_tests.sh`](../test/ci/run_ci_tests.sh) in the CI container.
The current gate covers:

- architecture manifest, test-group build metadata, PTX scheduler, and gtest discovery
  regression checks;
- SM120 unit, integration, barrier, TMA, and MMA test groups with `SM120_RTX5090`;
- SM90 integration, barrier, WGMMA, and FA2 smoke test groups with `SM90_H100`;
- FA3 forward smoke shapes, the fixed-forward integration case, and backward
  smoke shapes with `SM90_H100`; and
- SM120 GPT-2 trace smoke tests.

The workflow carries architecture and test set as independent matrix fields: SM120
core, SM90 core, SM90 FA2, and SM90 FA3. Each job has a 7 GiB container memory
limit and no additional swap allowance. Simulator builds use two jobs by default. FA2
kernel compilation is serial because a single NVCC translation unit approaches
5 GiB of resident memory; FA3 specializations also use an object-level
serialization chain to stay within the same budget.

Build and run logs are written under `test/logs/ci/logs/`, and gtest XML is
written under `test/logs/ci/xml/`. The workflow uploads the complete
`test/logs/ci/` tree even when a gate fails. The
[PR workflow](../.github/workflows/pr-tests.yml) and CI runner are the
authoritative sources for the current matrix layout and test scope.

## Repository Map

- `src/cuda-sim/`: PTX loading, parsing, instruction representation, and
  functional execution.
- `src/gpgpu-sim/`: timing model, SM pipelines, caches, memory controllers,
  interconnect integration, and power interfaces.
- `src/gpgpu-sim/flash/`: modern GPU extensions, including TMA, memory
  barriers, MMA/WGMMA, matrix load/store, PTX scheduling, and register
  allocation support.
- `configs/`: GPU model configurations.
- `test/`: unit tests, integration tests, microbenchmarks, analysis workloads,
  and the public test runner.
- `tools/`: Triton kernel capture and standalone harness-generation tool and
  documentation.
- `test/triton_trace/`: Triton examples and offline validation workloads.
- `docs/`: build, testing, development, and instruction documentation.

### Core Types

- `ptx_instruction`: parsed, static PTX instruction and operand information
  defined under `src/cuda-sim/ptx_ir.*`.
- `ptx_thread_info`: per-thread PTX execution state defined in
  `src/cuda-sim/ptx_sim.h`.
- `warp_inst_t`: a dynamic warp-level instruction used by the timing model.
- `core_t`: the abstract hardware-model interface in
  `src/abstract_hardware_model.h`; Flash instruction implementations use it to
  access participating threads and core state without depending directly on a
  concrete SM implementation.
- `shader_core_ctx`: the concrete SM pipeline model in
  `src/gpgpu-sim/shader.*`.
- `gpgpu_sim`: top-level GPU timing and cycle coordination in
  `src/gpgpu-sim/gpu-sim.*`.

## Simulation Layers

FlashGPU-Sim extends GPGPU-Sim while retaining its separation between
functional execution and the timing model:

```text
PTX source
    |
    v
lexer and parser
    |
    v
static instruction representation (`ptx_instruction`)
    |
    +----------------------+
    |                      |
    v                      v
functional semantics      dynamic warp instruction (`warp_inst_t`)
    |                      |
    |                      v
    |                 SM timing pipelines
    |                      |
    +----------+-----------+
               |
               v
       caches, interconnect, and DRAM
```

Functional execution determines architectural results such as register values,
memory contents, predicates, and synchronization state. Timing execution models
when work can issue and complete, including pipeline occupancy, dependencies,
resource conflicts, cache behavior, interconnect traffic, and DRAM timing.

Correct support for a new instruction may require changes in both paths. A
functionally correct implementation can still have an incomplete timing model,
and a timing opcode without functional semantics cannot validate program
results.

## PTX Front End

The PTX front end is under `src/cuda-sim/`:

- `ptx_loader.*` loads PTX modules.
- `ptx.l` recognizes opcodes, modifiers, identifiers, literals, punctuation,
  and other lexical tokens.
- `ptx.y` combines those tokens according to the PTX grammar.
- `ptx_parser.*` coordinates parser state and constructs instructions.
- `ptx_ir.*` defines the parsed instruction and operand representation.
- `opcodes.def` defines the simulator's opcode entries and their dispatch
  names.

For a source instruction such as `mma.sync.aligned`, the lexer first recognizes
the opcode and modifiers. The parser then checks their arrangement and records
the shape, data types, layouts, predicates, and operands in a
`ptx_instruction`. This is FlashGPU-Sim's internal representation of a parsed
PTX instruction; it is not LLVM IR.

## Functional Execution

Functional execution is primarily implemented under `src/cuda-sim/`.
`ptx_thread_info` represents per-thread PTX execution state, and
`ptx_thread_info::ptx_exec_inst()` dispatches the current instruction to its
functional implementation.

Depending on the instruction family, the implementation may remain in
`src/cuda-sim/instructions.cc` or delegate to a component under
`src/gpgpu-sim/flash/`. Warp- or warp-group-level operations use the thread and
core context to update all participating lanes consistently.

Functional tests should validate results against an independent reference where
possible. Matrix instructions, for example, require both numerical validation
and correct lane-to-fragment mapping.

## Timing Simulation

The timing model is primarily under `src/gpgpu-sim/`:

- `gpu-sim.*` contains the top-level `gpgpu_sim` model and GPU cycle
  coordination.
- `shader.*` contains `shader_core_ctx`, warp state, and the SM pipeline.
- `scoreboard.*` tracks register dependencies.
- `gpu-cache.*` and `l2cache.*` model cache behavior.
- `dram.*` models memory-controller and DRAM timing.
- `src/intersim2/` provides the interconnection network.
- `src/accelwattch/` provides the power-model integration.

`warp_inst_t` represents an instruction moving through the timing model.
`shader_core_ctx` models fetch, decode, issue, execution, and completion while
coordinating scoreboards, execution units, load/store behavior, and statistics.
Memory requests continue through the cache, interconnect, and DRAM models.

## Flash Extensions

Modern GPU extensions live under `src/gpgpu-sim/flash/`. Keeping these
components localized reduces unnecessary changes to the inherited GPGPU-Sim
implementation while still integrating through shared parser, execution, and
timing interfaces.

Important components include:

- `mma/`: PTX `mma.sync` functional implementations and fragment mappings.
- `wgmma/`: warp-group asynchronous matrix operations.
- `tma.*` and `tensormap.*`: Tensor Memory Accelerator operations and tensor
  descriptors.
- `mbarrier.*` and `bulk_group.*`: asynchronous completion and synchronization
  support. See the
  [`mbarrier` implementation notes](../src/gpgpu-sim/flash/mbarrier.md) for the
  modeled state and current scope.
- `ld_st_matrix.*`: matrix-oriented shared-memory load and store operations.
- `elect.*`: warp leader election.
- `ptx_sched/` and `reg_alloc.*`: PTX scheduling and register-allocation
  support.
- `gem5/`: optional gem5 memory-subsystem integration.

### MMA and WMMA

Legacy `wmma.*` and modern `mma.sync` are separate instruction families:

| Area | Legacy WMMA | Modern MMA |
| --- | --- | --- |
| Opcode family | `MMA_OP` and related opcodes | `TENSOR_MMA_OP` and related opcodes |
| Functional implementation | `src/cuda-sim/instructions.cc` | `src/gpgpu-sim/flash/mma/` |
| Main dispatcher | `mma_impl()` | `flash_gpgpu_sim::tensor_mma_impl()` |

Changes for one family should not be applied to the other without checking both
the PTX definition and the existing dispatch path.

## Parallel Simulation

Flash mode uses OpenMP in selected simulation paths. During a core clock cycle,
the simulator distributes `simt_core_cluster::core_cycle()` calls across
OpenMP threads. The local interconnect also advances independent subnets in
parallel.

State that is naturally scoped to one cluster or worker is kept separate where
practical:

- Each SIMT core cluster records `shader_core_stats` independently and the GPU
  aggregates those statistics at reporting and simulation-completion
  boundaries.
- PTX source-line statistics and mutable dynamic PTX instruction objects use
  per-OpenMP-thread storage.
- The shared `gpu_sim_insn` instruction counter uses atomic updates.
- The simulator memory block map uses a shared mutex to protect concurrent
  lookup and allocation.

OpenMP worker-local containers are currently sized for at most
`FLASH_GPGPU_SIM_OMP_MAX_THREADS` (32) threads. Configurations that set
`OMP_NUM_THREADS` above this limit are unsupported.

Changes to global statistics, shared queues, memory requests, or cross-SM state
must therefore be reviewed for ordering and data-race implications. Do not
assume that code inherited from a serial path is safe when called concurrently.

## Adding or Extending a PTX Instruction

PTX instruction support usually crosses several layers:

1. Add or update the opcode definition in `src/cuda-sim/opcodes.def`.
2. Extend token recognition in `src/cuda-sim/ptx.l` and grammar handling in
   `src/cuda-sim/ptx.y` as needed.
3. Preserve instruction modifiers, types, shapes, and operands in the
   `ptx_instruction` representation under `src/cuda-sim/ptx_ir.*`.
4. Implement functional semantics in `src/cuda-sim/` or the appropriate
   `src/gpgpu-sim/flash/` component.
5. Add timing behavior when the instruction has pipeline, latency, resource,
   synchronization, or memory-system effects.
6. Add focused unit or integration coverage under `test/`.
7. Update the relevant instruction or component documentation.

Do not assume that similar PTX spellings share an implementation. For example,
legacy `wmma.*` and modern `mma.sync` use different opcodes and implementation
paths. Start with the existing component documentation and neighboring
implementations.

## Testing Changes

- Prefer the smallest test group and filter that exercise the change during
  development.
- Add a CPU reference or another independent expected result for functional
  instruction tests when practical.
- Test both functional results and timing/resource behavior when a change
  affects both.
- Do not treat a native-GPU validation pass as proof that the simulator was
  used; simulator runs should also report simulator statistics.
- Update tests and documentation when changing public interfaces,
  configurations, or instruction behavior.

## Code Style

The Flash extension uses the repository's LLVM-based formatting rules. See the
[Code Formatting Workflow](workflows/code-formatting.md) before modifying files
under `src/gpgpu-sim/flash/`.

Keep changes focused and avoid unrelated formatting or generated-file churn.

## Further Reading

- Test runner and architecture/test-group hierarchy:
  [Test Framework](../test/README.md)
- Flash extension overview:
  [Flash README](../src/gpgpu-sim/flash/README.md)
- MMA implementation interface:
  [Tensor MMA Interface](../src/gpgpu-sim/flash/mma/tensor_mma.md)
- TMA implementation interface:
  [TMA Documentation](../src/gpgpu-sim/flash/tma.md)
- Triton capture and replay:
  [TritonTrace](../tools/README.md)
- Triton examples and validation:
  [TritonTrace Examples and Validation](../test/triton_trace/README.md)
