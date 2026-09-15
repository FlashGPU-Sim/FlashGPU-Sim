# Experimental SASS frontend

The SASS frontend is **experimental**, supporting selected SM90 (H100) and
SM120 (RTX 5090) instruction forms and workloads. It is execution-driven:
it executes the native code in a cubin, not a dynamic instruction trace and
not a SASS-to-PTX translation. Support for one opcode spelling does not imply
support for every modifier, operand form, datatype or architecture.

## Why a SASS frontend?

PTX is a virtual ISA, not the instruction stream executed by the GPU. Even
when a compiler such as Triton has already scheduled and pipelined its PTX,
`ptxas` still performs target-specific instruction selection, optimization,
register allocation and scheduling before producing SASS. Direct PTX execution
therefore mixes two sources of timing error: the hardware model and differences
between the simulated program and the program running on hardware.

For the workloads motivating this frontend, important examples include:

- **GEMM load/compute scheduling:** the placement of shared-memory matrix loads
  (`LDSM`) relative to tensor instructions (`HMMA`) changes dependency stalls
  and tensor-pipeline utilization. Matching individual instruction latencies
  does not recover the overlap of a differently scheduled native stream.
- **Attention softmax:** the interleaving of exponentials (`MUFU.EX2`), FP32
  arithmetic and shuffles changes the independent work available to hide
  latency and the pressure on shared issue resources such as MIO. Native
  instruction selection and local simplifications can also change the
  instruction count and dependency graph, not just their order.
- **Physical execution details:** register allocation can introduce spills
  and changes the physical operands seen by the register file. Native reuse
  hints and dependency controls are additional inputs absent from the original
  PTX stream. Their effects cannot all be represented by one PTX latency table.

During bring-up we explored PTX reordering, including SASS-guided scheduling,
and some peephole optimizations to reduce this gap. These approaches can improve
particular cases, but reproducing more of `ptxas` inside the simulator is not a
sustainable general solution. Rules must track compiler versions, targets and
code patterns; reordering alone cannot reconstruct all instruction-selection,
register-allocation and control-encoding decisions. Tuning hardware parameters
to compensate can make one kernel look accurate while hiding offsetting errors
that do not transfer to another workload.

The SASS frontend instead executes the compiler's actual output, so calibration
can focus on how hardware executes that stream rather than approximating the
compiler again. It does **not** make timing automatically accurate: native
semantics, scheduling controls, resource contention and memory behavior still
need independent validation. PTX remains useful and supported; SASS provides
an experimental path for workloads where the compiler-to-hardware gap matters.

## Enable SASS execution

Use CUDA 13.3 for the documented SASS workflow (tested with nvdisasm 13.3.73).
A physical GPU is not required for simulation; native reference/NCU collection
does require the target GPU. Workloads must contain a cubin for the selected
architecture and link the shared CUDA runtime, not `libcudart_static.a`.

From the repository root:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-13.3
source setup_environment release
make -j4

OMP_NUM_THREADS=4 python3 tests/run_tests.py run \
  --arch sm90 --group fa3 --profile medium-forward \
  --config SM90_H100_SASS_FRONTEND --sass-timing \
  --gtest-filter 'Fa3PrefillFp16MediumTest.H32D64FullB16S512' \
  --timeout 3600
```

For an SM120 example, use `--arch sm120 --group integration`,
`--config SM120_RTX5090_SASS_FRONTEND --sass-timing` and
`--gtest-filter 'CudaVectorAddTest.BasicVectorAddition'`.

The two dedicated configs select `-gpgpu_execution_frontend sass-timing`.
The original `SM90_H100` and `SM120_RTX5090` remain the PTX configs.
The runner supplies the executable, architecture, Python interpreter and
NVIDIA decoder tool paths. Selecting a SASS config alone does **not** discover
these paths.

For a custom CUDA executable, after building/sourcing the environment above:

```bash
sim_root="$PWD"
export FLASHGPU_SASS_BINARY=/absolute/path/to/application
export FLASHGPU_SASS_ARCH=sm90
export FLASHGPU_SASS_PYTHON="$(command -v python3)"
export FLASHGPU_SASS_NVDISASM="$CUDA_INSTALL_PATH/bin/nvdisasm"
export FLASHGPU_SASS_CUOBJDUMP="$CUDA_INSTALL_PATH/bin/cuobjdump"
export FLASHGPU_SASS_DECODE_TOOL="$sim_root/src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py"

sass_run_dir=$(mktemp -d)
cp -a "$sim_root/configs/SM90_H100_SASS_FRONTEND/." "$sass_run_dir/"
cd "$sass_run_dir"
"$FLASHGPU_SASS_BINARY"
```

Pass the application's normal arguments as needed. For SM120, change both
the architecture and config directory. Check the log for
`mode=execution-driven SASS timing simulation`; do not infer the frontend
from the name of the executable or output directory.

Config selection accepts `environment` (default), `ptx`,
`sass-functional`, and `sass-timing`. An explicit config selection overrides
mode environment variables and runner flags. For functional-only testing,
use the runner's `--sass` with a config whose selector is `environment`,
or set `sass-functional` in a private config copy. Passing `--sass` with a
config explicitly selecting `sass-timing` does not disable timing.

Each process dumps `_sass_<pid>_<sequence>.sassir` in its run directory for
debugging. Repeated launches reuse decoded state within that process;
there is no persistent cross-process cache. Do not commit generated SASSIR.
`--sassir` / `FLASHGPU_SASS_IR` are low-level explicit-image overrides,
not required for normal use. CuBit is not a runtime dependency.

## Implementation and timing boundary

```text
CUDA launch + cubin
    -> nvdisasm JSON / cuobjdump metadata -> SASSIR
    -> per-warp PC, registers, predicates and native functional semantics
    -> common warp_inst_t -> scheduler / scoreboard / pipelines / memory system
```

- **Decode and launch:** official NVIDIA tools provide instruction spelling,
  kernel ABI and resource metadata. The importer also preserves raw 128-bit
  words and typed operands. Architecture-specific code interprets native
  scheduling controls. The CUDA adapter supplies parameters, memory and
  tensor maps; kernel resource usage is independent of a PTX entry object.
- **Execution:** each warp has native R/UR/P/UP state and its own PC.
  Instructions determine branches, addresses and outputs dynamically.
  Handlers live under
  [`flash/sass/functional/`](../src/gpgpu-sim/flash/sass/functional/),
  split into control, data movement, integer, float, memory, MMA, WGMMA,
  TMA and mbarrier modules. Unsupported forms report an error and stop;
  there is no silent PTX fallback or guessed NOP semantics.
- **Backend integration:** [`execution_frontend`](../src/gpgpu-sim/flash/frontend/execution_frontend.h)
  defines thread/CTA lifecycle, fetch, execution, control-flow and completion
  hooks. PTX and SASS use separate adapters. The SASS
  [timing adapter](../src/gpgpu-sim/flash/sass/timing/shader_adapter.cc)
  projects static metadata and dynamic functional results into `warp_inst_t`.
  Hardware latency/issue parameters belong to `instruction_timing_config`,
  not PTX functional state; legacy `-ptx_opcode_*` names remain compatible.
- **Dependencies:** SASS retains native instruction order; PTX reorder is not
  used. Encoded stall/wait controls and backend resource availability jointly
  determine readiness. Explicit native dependencies bypass PTX-style inferred
  RAW reservation/checks to avoid double counting; variable-latency dependency
  barriers are released by backend completion, not simply functional execution.
- **Warpgroup and async operations:** a WGMMA collective creates one timing
  operation while all four participating warps update their native state.
  TMA, mbarrier and deferred CTA-barrier ordering bridge architectural effects
  with backend completion. Functional completion is not a timing shortcut.

This is an ISA/SIMT separation, not a generic NPU abstraction: warps, lanes,
CTAs and the existing scheduler remain GPU-specific, and adapters retain
privileged shader access. A non-SIMT architecture needs its own execution
and scheduling model. See the
[timing contract](../src/gpgpu-sim/flash/sass/TIMING_CONTRACT.md) and
[parameter reference](../src/gpgpu-sim/flash/sass/PARAMETERS.md) for details.

## Current coverage and limitations

Coverage below means implemented/tested **forms**, not complete ISA support.

| Area | Implemented scope |
| --- | --- |
| Scalar and control | Register/uniform moves, special registers and clocks, integer arithmetic/logic, FP32 arithmetic/conversions, EX2, shuffles/votes, branches/reconvergence, validated call/return paths and CTA barriers |
| Memory | Global, shared, constant and per-thread local accesses; selected atomic forms; SM120 LDGSTS async-copy and zero-fill forms |
| Matrix operations | Selected HMMA forms and LDSM/STSM normal/transposed fragments; SM90 GMMA forms used by FA3 and focused WGMMA tests |
| Async synchronization | mbarrier phases/arrival/transaction counts, waits, bulk groups and async-proxy ordering used by supported kernels |
| Tensor transfers | Validated TMA descriptor/opcode subsets: loads across ranks 1–5, stores across ranks 1–2, selected 2D reduction and swizzled layouts; not arbitrary tensor-map encodings |
| End-to-end focus | SM120 scalar/tensor GEMM and attention bring-up; SM90 FP16 FA3 forward medium correctness and medium/large timing regression |

Medium FA3 is the maintained full-output correctness/cycle gate. Large FA3
tests successful execution and timing, **not full O/LSE numerical correctness**.
Earlier instruction/workload bring-up is not a promise that every PTX test
or arbitrary CUDA binary runs through SASS. Backward kernels and additional
datatypes have individual support but are not covered by the forward tables
below. SM100 and arbitrary future SASS encodings are not validated here.

Native control fields are reverse-engineered and opcode-dependent. Functional
support does not imply precise timing calibration for every instruction mix.
For example, `ATOMS.ADD` updates shared memory functionally but currently uses
an L2-read proxy for timing; this is not a claim that hardware shared atomics
access L2. Cache/memory, TMA overlap and instruction-fetch models also remain
approximations. The dedicated H100 config is a validation configuration, not
proof of H100's internal SoC organization.

## H100 FA3 forward results

Snapshot: **2026-09-15**. FP16, full = non-causal; B=batch, S=sequence length,
H=head count, D=head dimension. Both datasets use CUDA 13.3 and
`SM90_H100_SASS_FRONTEND` with a 1500 MHz simulated core, CCD=2 and cold-cache
validation (`-gpgpu_perf_sim_memcpy 0`). Hardware collection targeted 1500 MHz
and used NCU cold-cache replay.

Compare **kernel duration**, not unrelated SM active/elapsed cycle counters:

```text
sim_us = sim_cycles / 1500
error = (sim_us - ncu_us) / ncu_us * 100%
```

Positive error means the simulator is slower. These are end-to-end forward
kernel measurements, not isolated steady-state phases or host wall time.

### Medium: four cases after frontend-interface separation

Execution code corresponds to `64505db0`; all four cases compare every O/LSE
element against a CPU reference and pass. Hardware durations use the standard
end-to-end collection, **not** the older, separate medium-variant NCU values
in the frozen regression CSV. Do not mix those hardware references.
The medium numerical-gate executable was rebuilt; NCU uses the retained
standard executable listed for large below, not a new native collection of
that rebuilt medium binary. The error column is therefore a comparison to
the retained same-shape standard measurement, not a same-binary recalibration.

| B | S | H | D | Attention | Sim cycles | Sim µs | NCU µs | Error |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 16 | 512 | 32 | 64 | causal | 122689 | 81.79 | 93.09 | -12.14% |
| 16 | 512 | 32 | 64 | full | 178429 | 118.95 | 124.93 | -4.78% |
| 16 | 512 | 16 | 128 | causal | 97900 | 65.27 | 66.98 | -2.56% |
| 16 | 512 | 16 | 128 | full | 116150 | 77.43 | 75.84 | +2.10% |

Maximum hardware error is **12.14%** (D64 causal), so these data do not establish
a universal 10% accuracy bound. In contrast, the maximum simulator-to-simulator
cycle drift against the frozen medium baseline is **0.1042%**, below the
0.5% CI regression gate. Small regression drift is not hardware accuracy.

### Large: twenty cases from the frozen post-rebase + ATOMS run

This queue used a library frozen **before** the later frontend-interface
refactor. It therefore does not validate the newest commit on large; medium
provides that newer regression. All twenty cases completed successfully,
including recovered runs. Large does not run the full numerical reference.

| B | S | H | D | Attention | Sim cycles | Sim µs | NCU µs | Error |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 64 | 512 | 32 | 64 | causal | 475004 | 316.67 | 336.45 | -5.88% |
| 64 | 512 | 32 | 64 | full | 667970 | 445.31 | 465.92 | -4.42% |
| 64 | 512 | 16 | 128 | causal | 377959 | 251.97 | 244.06 | +3.24% |
| 64 | 512 | 16 | 128 | full | 412963 | 275.31 | 268.26 | +2.63% |
| 32 | 1024 | 32 | 64 | causal | 723550 | 482.37 | 515.30 | -6.39% |
| 32 | 1024 | 32 | 64 | full | 1161411 | 774.27 | 804.06 | -3.70% |
| 32 | 1024 | 16 | 128 | causal | 471689 | 314.46 | 305.12 | +3.06% |
| 32 | 1024 | 16 | 128 | full | 722026 | 481.35 | 457.60 | +5.19% |
| 16 | 2048 | 32 | 64 | causal | 1016587 | 677.72 | 734.40 | -7.72% |
| 16 | 2048 | 32 | 64 | full | 1818194 | 1212.13 | 1240.00 | -2.25% |
| 16 | 2048 | 16 | 128 | causal | 749178 | 499.45 | 470.14 | +6.23% |
| 16 | 2048 | 16 | 128 | full | 1352438 | 901.63 | 842.72 | +6.99% |
| 8 | 4096 | 32 | 64 | causal | 1819256 | 1212.84 | 1300.00 | -6.70% |
| 8 | 4096 | 32 | 64 | full | 3477293 | 2318.20 | 2350.00 | -1.35% |
| 8 | 4096 | 16 | 128 | causal | 1345220 | 896.81 | 841.18 | +6.61% |
| 8 | 4096 | 16 | 128 | full | 2621923 | 1747.95 | 1610.00 | +8.57% |
| 4 | 8192 | 32 | 64 | causal | 3277920 | 2185.28 | 2330.00 | -6.21% |
| 4 | 8192 | 32 | 64 | full | 6491279 | 4327.52 | 4380.00 | -1.20% |
| 4 | 8192 | 16 | 128 | causal | 2579111 | 1719.41 | 1590.00 | +8.14% |
| 4 | 8192 | 16 | 128 | full | 5065339 | 3376.89 | 3110.00 | +8.58% |

All ten non-causal cycle counts match the frozen large baseline exactly.
The maximum absolute causal drift is **0.1492%**; at S8192 the maximum is
**0.0173%**. Maximum absolute hardware error is **8.58%** for this dataset.

### Provenance and reproduction

Do not treat the two simulator generations as one build. Recorded SHA-256s:

| Dataset | Simulator library | Workload executable |
| --- | --- | --- |
| Medium | `445e69102fd86f4e44e5f256ad5e4af2ebcf494973e3edceae81b88cca60608f` | `9bd588fd129afd53bb9dad0e7d176b85950b94df8626b036515f6b3f25caf37a` |
| Large | `e4386397888db99751ff286f0b06f04950c3c80153550cbe961897a27ebfe34d` | `145c11f24bb46bd5d33f5ca5be3085cfaaff01ccf9325cda9017dfa95a54a76e` |

The large log build tag is `b343810a_modified_3.0`; the library hash, not that
modified-tree tag alone, identifies the tested build. Medium was snapshotted
before committing the final interface cleanup; its hash identifies the actual
tested library. Hardware summary SHA-256:
`87c1fc7074c150f75c6b6e18d20512a6667a95636a717ace1939dbdfc4c8d979`.
The local collection is named
`h100_native/e2e_smoke_small_medium_large_1500_20260907`.
Raw generated packages/logs are not checked into the public source tree.

Use the [FA3 regression workflow](../tests/scripts/README.md#fa-3-forward-regression)
to build a package, collect native/NCU results, and run `--profile medium` or
`--profile large`. Keep the exact binary, config, library and hardware summary
together: rebuilding a cubin can change instruction scheduling. The queue
runs four jobs with four physical cores each, in ascending S order.

The [immutable baseline](../tests/baselines/fa3-sass-ccd2-20260914/README.md)
and [medium checker](../tests/ci/check_sass_medium.py) remain unchanged.
Use medium for iterative changes and rerun large for final PR validation;
never refresh the baseline merely to make a regression pass.
