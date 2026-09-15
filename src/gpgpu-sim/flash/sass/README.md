# Experimental SASS execution-driven frontend

Start with the [user guide](../../../../docs/sass-frontend.md) for activation,
current coverage and versioned H100 FA-3 results. This file contains additional
implementation and historical bring-up details.

## Source organization

- `decode/`: static IR import, operand parsing and SM90/SM120 control words.
- `functional/`: instruction semantics, separated by instruction family.
- `runtime/`: launch/ABI/memory adapters and tensor-map interpretation.
- `timing/`: projection onto generic backend instructions and timing runtime.
- `frontend.*`, `ir.h`: shared frontend state and IR interfaces.
- `tools/`: NVIDIA disassembler import and per-run SASSIR dumps.

Directory moves do not change instruction semantics or timing parameters.
Shared hardware resource models remain outside this frontend. Maintained
calibration rationale lives with the [H100 config](../../../../configs/SM90_H100_SASS_FRONTEND/README.md);
end-to-end timing references live in the [frozen baseline](../../../../tests/baselines/fa3-sass-ccd2-20260914/).
Intermediate experiment reports remain in Git history, not this usage guide.
See [PARAMETERS.md](PARAMETERS.md) for added/changed controls, defaults, units
and the selected H100 values; optional experimental knobs are not all calibrated.

This directory is the start of a **static SASS frontend**, not a SASS trace
replayer and not a SASS-to-PTX translator. The intended execution path is:

```text
cubin .text.<kernel> --(nvdisasm JSON, once)--> versioned SASS IR
                                           |
dynamic warp PC --> static instruction fetch --> SASS semantics
                                           |
                         warp_inst_t timing/memory pipeline
```

The cubin is decoded once when the kernel image is loaded. During simulation,
each warp owns a PC and physical architectural state; branches and predicates
choose the next PC. No sequence of dynamically executed instructions is
recorded or replayed.

## What is implemented

- `ir.h` is the simulator-owned interchange IR. It retains the raw 128-bit
  instruction, official mnemonic/operand text, typed operands, decoder
  provenance, and the exact static scheduling control. Legacy SASSIR v1 fields
  remain only for optional CuBit cross-check sassirs.
- `decode/sm90_control.*` and `decode/sm120_control.*` decode their scheduling
  fields. SM90 retains four reuse bits `[125:122]`; SM120 uses the three
  bits `[124:122]` observed by its instruction forms.
- `tools/nvdisasm_to_sassir.py` uses NVIDIA's versioned `--emit-json`
  interface for the canonical mnemonic, predicate, and operand spelling. It
  independently reads the cubin ELF section to retain every raw 128-bit word,
  derives PCs/control fields without parsing human-readable disassembly, and
  obtains the kernel parameter bank/ordinal ABI from official `cuobjdump -elf`
  metadata. SASSIR v3 also records register, static-shared, local-memory, and
  stack resource usage from official `cuobjdump -res-usage`; one sassir may
  contain multiple kernels from the same cubin. Nonzero-start helper functions
  embedded in a kernel text section retain their official instructions and a
  resolved static call-target attribute.
- `decode/operand_parser.*` converts the official operand syntax into typed registers,
  predicates, integer/FP immediates, absolute/negated register sources,
  constant-memory references, descriptor addresses, generic address
  expressions, register sets, and nvdisasm's auxiliary `PR` operand. New
  syntax remains opaque and therefore fails closed rather than acquiring
  guessed semantics.
- Official instruction attributes such as `control-flow`,
  `inter-warp-barrier`, and `subroutine-call` are retained in SASSIR instead of
  being rediscovered from PTX in the timing backend.
- `decode/sassir_decoder.*` loads SM90 and SM120 images and validates their exported
  fields against the architecture-specific raw control word. Executing an
  imported image needs no Python, Rust, CuBit, or JSON parser; automatic import
  invokes Python and the official CUDA tools on first use within each process.
- `frontend.*` provides PC-based static fetch, a physical R/UR/P/UP warp state,
  and an opcode semantic-handler boundary. A handler selects the next PC, so
  this remains execution-driven.
- `functional/functional.*` provides fail-closed SM90 and SM120 functional slices and a
  frontend-neutral constant/global/shared/per-thread-local memory interface.
  The public entry
  point is intentionally small; implementations are grouped into
  `data_movement`, `integer`, `float`, `control`, `memory`, `mbarrier`, `tma`,
  `mma`, and `wgmma` translation units. It currently covers the exact instruction
  forms dynamically required by the checked-in scalar GEMM, tensor-core GEMM,
  Triton attention, and official FlashAttention-2 correctness gates, including
  Hopper register/shared F16/BF16/TF32 `HGMMA.64x8`, FP8 `QGMMA.64x8`, INT8
  `IGMMA.64x8`, B1 `BGMMA.64x8`, shared/shared F16
  `HGMMA.64x64x16.F32` with 128-byte swizzling, `HMMA.1688.F32`,
  `HMMA.16816.F32`, normal/transposed x1/x2/x4
  `LDSM`/`STSM`, FP32-to-FP16 packing, FP32 softmax arithmetic and warp
  shuffles, integer and
  uniform-register operations, local stack spills, divergent control flow,
  CTA barriers, mbarriers, TMA load/store, and SM80-style 16-byte `LDGSTS`
  asynchronous copies.
- `cta_executor` round-robins independent physical warp states through one
  functional CTA. Named barriers can block one warp without stalling the
  driver, and reusable barriers retain an explicit generation. This scheduler
  is a correctness mechanism, not a timing or warp-overlap model.
- `runtime/tensor_map.*` decodes only the differentially validated SM120 tiled-1D F32
  and tiled-2D FP16/F32 plus tiled-3D/4D/5D F32 descriptor subsets and fails
  closed on every other encoding. `UTMALDG` supports all five validated ranks
  while `UTMASTG` supports ranks one and two; their implemented forms handle
  unswizzled and 32/64/128-byte swizzled tiles. The validated
  `UTMAREDG.2D.ADD` form performs the functional F32/F32_FTZ read-modify-write
  through the backend-neutral tensor-reduction helper. `UBLKCP.G.S` performs
  the validated linear shared-to-global bulk copy; bulk commit/wait and fence
  markers complete immediately in this functional model. Launch-time
  `functional_tensor_map_2d` registration remains available as a
  frontend-neutral sideband path.

## Functional semantics layout

The source split follows the PTX ISA 9.3 section 9.7 taxonomy where that maps
cleanly to SASS, while keeping hardware-specific asynchronous mechanisms
separate:

| Module | PTX 9.3 reference | SASS responsibility |
|---|---|---|
| `functional/data_movement.cc` | 9.7.9 | register moves and special-register reads |
| `functional/integer.cc` | 9.7.1, 9.7.6-9.7.8 | integer arithmetic, logic, shifts, comparisons, selection |
| `functional/float.cc` | 9.7.3-9.7.5 | floating-point arithmetic and mixed-precision conversion |
| `functional/control.cc` | 9.7.13 | branches, reconvergence, warp votes, CTA barriers, exit |
| `functional/memory.cc` | 9.7.9-9.7.12 | scalar memory, atomics, and functional fences |
| `functional/mbarrier.cc` | 9.7.12 | split-phase barrier state and waits |
| `functional/tma.cc` | 9.7.9.26.5 | tensor-memory asynchronous copies |
| `functional/mma.cc` | 9.7.14-9.7.15 | matrix-fragment movement and MMA |

`functional/common.cc` owns shared operand, address, register-pair, and FP16
helpers. `functional/internal.h` is private to these translation units;
external users continue to include only `functional/functional.h`.

For executable tests, the primary interface is the runner's automatic strict
SASS mode:

```bash
./tests/run_tests.py run --arch sm120 --group integration \
  --gtest-filter 'CudaVectorAddTest.BasicVectorAddition' --sass
```

At the first ABI/resource query or launch, the runtime pairs the registered
kernel symbol with its exact CUDA fatbin handle, extracts that one cubin from
the selected test executable, and decodes it with the CUDA toolkit's official
`nvdisasm`/`cuobjdump`. The resulting `_sass_<pid>_<sequence>.sassir` is dumped
in the current run directory for debugging; the log maps its absolute path to
the kernel name. Within that process, repeated queries and launches reuse the
path for the registered fatbin/kernel pair. New processes always decode again:
there is no cross-process disk cache or cache directory setting. Generated
dumps are not checked-in fixtures. `FLASHGPU_SASS_DECODE_TOOL` selects the
automatic `tools/dump_kernel_sassir.py` importer (set by the test runner).

For low-level decoder work, a sassir can still be generated explicitly:

```bash
python3 src/gpgpu-sim/flash/sass/tools/nvdisasm_to_sassir.py \
  --nvdisasm "$CUDA_INSTALL_PATH/bin/nvdisasm" \
  kernel.cubin '<mangled-kernel-name>' ['<second-kernel-name>' ...] \
  -o kernel.sassir
```

The older `cubit_to_sassir.py` exporter remains available as an optional
bitfield/round-trip oracle. CuBit is not required to generate or execute the
SASSIR dumps.

Driver modules registered with `cuModuleLoad` retain an absolute source path,
so a later working-directory change does not redirect decoding. Decoding is
still lazy: keep the module file available and unchanged until its kernels
have been decoded. Registration does not snapshot the cubin contents.

The standalone inspector is also a convenient frontend smoke test:

```bash
g++ -std=c++17 -fno-exceptions -Isrc/gpgpu-sim/flash/sass \
  src/gpgpu-sim/flash/sass/decode/{sassir_decoder,operand_parser}.cc \
  src/gpgpu-sim/flash/sass/decode/{sm90_control,sm120_control}.cc \
  src/gpgpu-sim/flash/sass/tools/sassir_inspect.cc \
  -o /tmp/sassir_inspect
/tmp/sassir_inspect kernel.sassir '<mangled-kernel-name>'
```

An unknown instruction is retained with its raw 128 bits, and the tool exits
with status 2. The frontend will load it but refuses to execute it. This is
intentional fail-closed behavior.

Unrecoverable C++ frontend/runtime errors print a diagnostic to stderr and
call `flash_gpgpu_sim::panic` (`abort`/SIGABRT); do not introduce `throw`/`catch`
for these paths. Decoder errors retain the sassir path and line, and semantic
invariant failures retain the opcode and PC. Numeric parsing uses checked
non-throwing conversions. Normal execution statuses (including barrier waits
and unsupported-form results consumed by the runtime) remain explicit values;
the runtime panics on an unsupported execution, with no PTX fallback.
The inspector's unknown-instruction census still returns status 2, but malformed
input now panics instead of returning status 1. Python tools keep their existing
nonzero-exit error handling. This does not disable exceptions globally in the
legacy PTX implementation or third-party libraries.

The primary adapter was exercised with nvdisasm 13.3.73 and records both its
producer string and JSON schema version in SASSIR v4. The sassir remains the
compatibility boundary: nvdisasm and its JSON parser are not linked into the
simulator. See [`COVERAGE.md`](COVERAGE.md) for the real SM120 GEMM census.

## CUDA runtime launch adapter

`runtime/runtime_adapter.*` connects the normal CUDA launch path to the functional
frontend. The runner's `--sass` option selects automatic strict functional
SASS mode. `--sass-timing` adds the execution-driven detailed timing pass to
the same strict functional contract. `FLASHGPU_SASS_IR` and
`--sassir` remain low-level overrides for decoder/frontend debugging.

`gpgpusim.config` can instead select the frontend with
`-gpgpu_execution_frontend sass-timing`. Accepted values are `environment`
(default, preserving existing runner/environment selection), `ptx`,
`sass-functional`, and `sass-timing`. Explicit config selection takes precedence
over the mode requested by the runner or inherited SASS mode variables; invalid
values print a diagnostic and panic during initialization. `ptx` leaves the
existing PTX functional/timing setting in charge.

This option selects execution mode, not decoder tools: automatic SASS decoding
still needs the existing binary/architecture/tool environment supplied
by the runner. An explicit `FLASHGPU_SASS_IR` still selects the debug
image. Selecting a SASS mode in config needs neither `FLASHGPU_SASS_AUTO` nor
`FLASHGPU_SASS_TIMING`. The working `SM90_H100_SASS_FRONTEND` config selects
`sass-timing` explicitly. Archived configs under `tests/baselines/` retain
their original environment selection and byte identity.

In strict mode:

- fatbin registration records CUDA's exact `hostFun`, mangled-device-symbol,
  and fatbin-handle mapping without extracting or parsing PTX. The handle is a
  cache/search hint, not a `cuobjdump --list-elf` image ordinal. Automatic mode
  resolves the exact symbol in the selected executable's cubins and rejects
  conflicting same-named definitions rather than trusting the handle as an index;
- launch arguments are copied by ordinal, size, and offset from the sassir's
  official cubin ABI, including an explicit zero-sized ABI for parameterless
  kernels;
- launch limits and `cudaFuncSetAttribute` use the sassir's official static
  shared-memory usage instead of PTX metadata or a zero-sized assumption;
- grid/block dimensions and architecture-specific SM90 or SM120 launch
  constants are supplied to the SASS state, while simulator global/parameter
  memory and per-CTA shared/local memory are exposed through
  `functional_memory`; and
- every CTA and physical warp executes from SASS PC zero to the real `EXIT`.

An absent kernel, unsupported opcode/form, invalid memory access, deadlocked
CTA, or instruction-limit violation terminates the launch. Once this mode is
selected there is no PTX PDOM analysis or PTX functional fallback.

Multiple filters can be kept as one auditable functional gate with
`tests/run_sass_suite.py`; suite descriptions may also select a runner profile,
compile-time mode, and executable. Kernel/cubin selection is deliberately not
part of the suite: it comes from CUDA registration in the exact executable.
When separate translation units instantiate the same externally named kernel,
automatic decode accepts the duplicate only if their complete generated
sassirs are identical, including resources, ABI, constant banks,
instructions, and scheduling controls. Differing same-named definitions remain
a hard ambiguity error.
The low-level explicit-sassir field is accepted only for focused frontend
debugging.

```bash
./tests/run_sass_suite.py tests/src/wgmma/sass_functional_suite.json
```

## Timing-backend boundary

The shader, WGMMA, TMA, cp.async, and mbarrier timing paths now consume only
`inst_t`/`warp_inst_t` and hardware-facing metadata. They do not cast pipeline
instructions to `ptx_instruction` or reparse PTX operands/options. PTX-specific
translation happens during `ptx_instruction::pre_decode()`; values produced by
PTX functional execution cross the boundary once in
`exec_shader_core_ctx::func_exec_inst()`.

This removes the instruction-type blocker for a SASS frontend. The functional
SASS path now owns physical registers, predicates, special-register inputs,
and frontend-neutral memory access; it does not use `ptx_thread_info`. The CUDA
runtime adapter now supplies real launch and simulator memory state. The
timing adapter projects decoded SASS onto `warp_inst_t`, carries native control
words as explicit dependencies, and obtains execution-pipeline latency and
initiation intervals from the selected GPU configuration. See
[`TIMING_CONTRACT.md`](TIMING_CONTRACT.md) for the exact frontend/backend
contract.

### Timing diagnostics

The issue trace is disabled by default. Set `FGSIM_ISSUE_TRACE_FILE` to enable
it, with optional `FGSIM_ISSUE_TRACE_SM`, `FGSIM_ISSUE_TRACE_WARP`, and
`FGSIM_ISSUE_TRACE_MAX` filters. Besides scheduler issue and stall events, the
trace records instruction UID/native dependency control plus
`OPERANDS_COLLECTED`, `OPERANDS_READ`, `LDST_ISSUE`, `LDST_STALL`, and
`COMPLETE`. `LDST_ISSUE` includes the generated memory-transaction count and
`LDST_STALL` includes the backend memory-stage rejection reason. These events
are intended for bounded single-SM/warp debug windows rather than full-run
profiling.

## Executable correctness gates

The PR functional-test matrix includes four `sm90-sass-medium-*` jobs, one
process each for B16/S512, H32/D64 or H16/D128, causal and non-causal. They use
nonzero deterministic inputs and compare the entire O and LSE tensors with
the CPU reference. `tests/ci/check_sass_medium.py` additionally requires actual
SASS timing execution and at most 0.5% absolute cycle drift from the immutable
medium baseline. Neither gate is advisory; no backward build is needed.

Run a job locally with CUDA 13.3:

```bash
CUDA_INSTALL_PATH=/usr/local/cuda-13.3 CI_JOB=sm90-sass-medium-d128c \
  OMP_NUM_THREADS=4 bash tests/ci/run_ci_tests.sh
```

SASS CI builds `docker/Dockerfile.ci`, then layers the matching CUDA 13.3.73
compiler/disassembler using `docker/Dockerfile.ci-sass`; it does not depend
on the old registry digest. Other CI jobs keep their existing image. This
numerical gate is stronger than the frozen profiling binaries,
which used zero inputs and checked only execution success. After correcting
TMA-store ordering against deferred barriers, all four local medium runs pass
O/LSE checks and the 0.5% gate. Container/hosted CI evidence is tracked separately
in the [FA-3 regression workflow](../../../../tests/scripts/README.md#fa-3-forward-regression).

Two small CUDA 13.3 `sm_120a` cubins are kept under
`tests/src/unit/fixtures`; the unit build generates their nvdisasm SASSIR under
`tests/build/generated/sass-unit/`:

- `sass_gemm_2x2x4` executes a 42-instruction scalar-FFMA path and compares all
  four results against a CPU GEMM;
- `sass_mma_gemm_m16n8k8` executes one converged warp through a real
  `HMMA.1688.F32`, writes a complete 16x8 tile, and compares all 128 results.

Both gates fetch by dynamic SASS PC, require every static instruction and
operand to decode, and fail on any missing semantic handler or memory access.
They have no PTX image, CuBit dependency, or trace/fallback path. Small
driver-API runners next to the fixtures run the exact same cubins on a physical
GPU for cross-checking.

The repository's unchanged `CudaVectorAddTest` is the first end-to-end CUDA
runtime gate. Its ordinary `cudaMalloc`/`cudaMemcpy`,
`vectorAddKernel<<<1024,256>>>`, and `cudaDeviceSynchronize` sequence runs the
32/32 decoded SM120 instructions. Both numerical cases pass after 1,024 CTAs,
8,192 physical warps, and 163,840 dynamic SASS instructions per launch:

```bash
./tests/run_tests.py run --arch sm120 --group integration \
  --gtest-filter 'CudaVectorAddTest.*' \
  --sass
```

The 936-instruction nvdisasm sassir for Triton's real TMA GEMM is a third
execution gate. Four physical warp states run from PC zero to `EXIT` in 2,787
dynamic CTA instructions. A nonuniform exactly representable FP16 input spans
both K stages; all 4,096 elements of the emitted 64x64 tile are compared with
the CPU reference, and the surrounding output sentinel must remain untouched.
No PTX functional state or dynamic trace participates in this check.

The fourth gate is Triton's real tiled GQA attention kernel for
`B=2, S=128, QH=32, KVH=8, D=128`. The complete 3,640-instruction official
sassir is loaded once, then four warp states execute one causal CTA from PC
zero to `EXIT` in 13,003 dynamic CTA instructions. Nonuniform Q/K/V inputs
exercise QK HMMA, scale/mask/max/shuffle/exp2/sum, PV HMMA, local spills, TMA,
and FP16 output. All 8,192 elements in the emitted 64x128 tile must agree with
an independent CPU softmax-attention reference within 0.02 absolute error.

The fifth gate is the official FlashAttention-2 FP16 `D=128`, non-causal
specialization compiled for SM120. Its runtime launch is exactly
`B=1, S=4096, H=16, D=128, full`; one four-warp CTA traverses 64 K/V tiles and
executes 237,328 dynamic instructions before all warps reach `EXIT`. The
2,648-instruction sassir contains 512 `HMMA.16816.F32`, 160 `LDSM`, 56
16-byte `LDGSTS`, and 132 `MUFU.EX2` instructions. Every static instruction
and all 85 distinct opcode strings have registered fail-closed semantics. The
gate compares all 16,384 FP16 output elements and 128 LSE values with an
independent CPU full-attention reference. This kernel is FA-2's SM80-style
cp.async/HMMA implementation on RTX 5090, not a WGMMA/TMA kernel.

The same SASSIR v3 image now also drives the unchanged
`Fa2PrefillFp16SmokeTest.H1D128FullB1S256` through its normal CUDA runtime
launch. Two CTAs and eight warps execute 35,456 dynamic instructions and pass
the GoogleTest's original output and LSE CPU references. Strict SASS device
variable registration does not invoke the PTX symbol-table loader; registered
globals/constants remain unsupported until represented explicitly in a
sassir and functional memory.

Synthetic CTA gates additionally isolate reusable named barriers, reliable
reconvergence, leader election/warp voting, mbarrier phase transitions and
invalidation, registered 2D TMA load/store, `LDSM`/`STSM`, and nonuniform
`HMMA.16816.F32` fragment arithmetic.

The unchanged `SharedMemoryOptinTest` is also closed through a two-kernel
SASSIR v3 image. The failed no-opt-in launch and both function-attribute checks
use the cubin's 1,024-byte static shared-memory resource value. The accepted
75,264-byte dynamic-shared launch executes 18 dynamic SASS instructions,
including packed `HFMA2` constant materialization and byte-wide `STS`/`LDS`,
and reproduces both output sentinels.

All eight unchanged `ldmatrix`/`stmatrix` integration cases run through one
eight-kernel SASSIR v3 image. They cover normal x1/x2/x4 loads and stores plus
transposed x1 load/store, execute 23 to 55 dynamic instructions per launch,
and retain the tests' complete per-element matrix checks. Their support also
closes general byte permutation and 64/128-bit scalar shared-memory transfers.

The eight unchanged `CpAsyncSrcSizeTest` cases run through an eight-kernel
SASSIR v3 image as well. SM120 lowers the PTX source-size operand into the low
four address bits, `.ZFILL`, and a copy-enable predicate; the functional
handler consumes that SASS representation directly. Full, immediate,
per-lane, odd, zero, ignore-source, cache-policy, and combined cases all pass
their 16-byte-per-lane checks in 21 to 33 dynamic instructions.

Timing integration was layered on top of the functional gates:

1. The functional launch adapter imports the launch symbol, parameter ABI,
   cubin resource metadata, grid/CTA state, registered globals/constants, and
   CUDA-created tensor maps.
   Launch metadata remains separate from instruction decoding.
2. The controlled GEMM and attention gates model the live `LEPC`/`CALL` CUDA
   tensor-copy syscall paths and fail closed on an unsupported branch.
3. `exec_shader_core_ctx` adapts validated SASS launch and memory state while
   reusing the issue, operand collector, execution, cache, and interconnect
   machinery.
4. Each static instruction is projected onto frontend-neutral timing fields:
   physical registers, memory space/width, execution pipe, latency, initiation
   interval, and the exact control word.

## Scheduling model

The SASS order and encoded dependency control are architectural inputs. The
frontend never reorders them. The timing bridge models both:

- producer-side stall/yield and the six hardware scoreboard barriers encoded
  in the SASS control word; and
- pipeline/resource availability already modeled by the simulator.

Hopper GMMA is issued once per four-warp group while the SASS adapter advances
and executes every participant's architectural state. Its trailing `gsb`
operand commits the generic WGMMA group. Register-selected Hopper CTA barriers
likewise publish their warp-uniform dynamic id and participant count before the
existing named-barrier timing resource consumes the instruction.

Some SM120 opcode classes overlay parts of the standard control positions.
For that reason the IR always keeps the raw instruction and complete official
mnemonic; the generic bit decoder is a consistency check, not the final
opcode-aware timing classification.

Unknown functional semantics and unknown timing classifications fail closed;
they never silently behave like NOPs or a generic one-cycle ALU instruction.
