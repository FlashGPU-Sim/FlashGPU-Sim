# SM120 decoder coverage

Date: 2026-09-01

Primary decoder: NVIDIA nvdisasm 13.3.73, JSON schema 13.2.0

Historical cross-check: CuBit
`1eb8dc68d977c323fb6a24a60b8b76cc0813bc59`

Input:
`tests/ci/perf/traces/SM120_RTX5090/gemm-m4096-n128-k4096/kernel_tma_gemm_launch1_kernel.cubin`
(`kernel_tma_gemm`)

This is a static cubin decode census. It does not use a dynamic trace.

Names ending in `.sassir` below are historical human-readable labels for the
decoded kernel images used while closing each functional frontier. They are no
longer repository files: suites now regenerate equivalent SASSIR from their
exact executables into ignored content-addressed caches.

| Result | Count |
|---|---:|
| 128-bit instructions | 936 |
| Official opcodes decoded by nvdisasm JSON | 936 (100%) |
| Operands covered by the typed grammar | 936 (100%) |
| Historical CuBit structural decode | 882 (94.2%) |
| Decoded `HMMA.16816.F32` | 32 |
| Decoded `LDS` / `LDSM` | 45 / 16 |
| Instructions with yield bit set | 691 |
| Instructions with a nonzero wait mask | 167 |
| Official `control-flow` attributes | 139 |
| Official `inter-warp-barrier` attributes | 37 |
| Official `subroutine-call` attributes | 8 |

The 54 words historically unknown to CuBit are all decoded by nvdisasm. They
are concentrated in Blackwell TMA/control plumbing rather than the HMMA compute
body:

| Historically CuBit-missing opcode family | Count |
|---|---:|
| `SYNCS` | 10 |
| `LEPC` | 8 |
| `CALL` | 8 |
| `UTMALDG` | 8 |
| `STSM` | 4 |
| `ATOMG` | 3 |
| `DEPBAR` | 3 |
| `CCTL` | 3 |
| `UTMACCTL` | 3 |
| `MEMBAR` | 3 |
| `UTMASTG` | 1 |

Official opcode and typed-operand coverage are therefore no longer the
blockers for this cubin. Executing the whole kernel still requires explicit
functional/timing semantics for shared memory, TMA, call/return, and
synchronization. An instruction with opaque operand syntax or no semantic
handler fails closed.

## Functional frontier

Functional execution is being closed before any SASS-to-`warp_inst_t` timing
projection. The frontend currently registers handlers for 284 exact official
opcode strings, covering 920 of this kernel's 936 static instructions (98.3%).
This is an opcode census, not a claim that the complete kernel executes: every
handler also validates its exact operand form.

The generated real-kernel gate starts at PC zero with four warps and controlled
launch, constant, global, and shared-memory state. All warps reach `EXIT` after
2,787 dynamic CTA SASS instructions. The path constructs and decodes its raw
SM120 tensor maps, executes two K stages through TMA load, mbarrier wait,
`LDSM.16.M88.4`, and `HMMA.16816.F32`, packs and stores the result through
`STSM.16.M88.4` and `UTMASTG.2D`, then invalidates its barriers. The nonuniform
FP16 input has nonzeros in both K stages; every element of the resulting 64x64
tile agrees bit-for-bit with the CPU integer-exact reference, while output
outside the tile retains its sentinel value.

The remaining 16 statically uncovered instructions are eight each of `LEPC`
and `CALL.ABS.NOINC`. They occur only on dormant helper-call
branches for this controlled launch. Missing handlers still fail closed if a
future launch selects one of those paths.

The reusable functional substrate now includes per-lane and uniform source
evaluation, 32/64-bit register pairs, CTA/thread/lane special registers,
single-group address expressions, global/shared/per-thread-local memory,
divergent-path serialization, nested reconvergence, independent CTA warp
state, named-barrier generations with active-warp retirement, logical mbarrier
state, and these target forms:

- `MOV`, `MOV.64`, `UMOV`, `UMOV.64`, uniform-input `R2UR` status, OR, and
  broadcast forms, uniform-predicate packing `UP2UR`, `S2R`, `S2UR`, and
  `CS2R`/`CS2R.32`/`CS2UR.32`, including the flat functional
  `SR_SWINHI` shared aperture, timing-placement-aware `SR_VIRTUALSMID`, and a
  monotonic `SR_GLOBALTIMERLO` value in nanoseconds derived from architectural
  cycles and the configured core frequency during timing execution, with
  dynamic instruction count as the flat-functional fallback,
  plus segmented `SHFL.BFLY` and immediate/register-indexed `SHFL.IDX` warp
  shuffles with clamp/mask predicate behavior;
- `IADD`, `IADD.64`, `IADD3`, `UIADD3`, and `UIADD3.64` with the target carry
  forms;
- integer absolute value, multiply-add including signed/unsigned high halves,
  Hopper carry input, and typed register ones-complement operands,
  register-selected funnel shift, effective-address, population
  count/find-leading-one, predicate compare/select/movement including
  OR-combine and extended high/low ordering forms, and integer/FP32 conversion
  forms;
- `LDS`, `STS`, their U16/64/128-bit forms, and generic scalar/128-bit
  `LD.E`/`ST.E` across the flat shared window and full global pointers;
- `LDCU.U8`/`.128`, U8/U16/scalar/64/128-bit global loads including the
  scalar GPU-strong cache form, 64/128-bit global stores, 128-bit constant
  global loads,
  16-byte `LDGSTS`, and `LDL`/`STL` local-stack accesses plus the exact
  discarded-result
  `ATOMG.E.EXCH.STRONG.GPU` form with implicit 64-bit vector and uniform
  address pairs, returned-old-value `ATOMG.E.ADD.STRONG.GPU`, shared
  `ATOMS.ADD` with returned-old-value or discarded results, and warp-grouped
  `ATOMS.POPC.INC.32` forms;
- `BRA`, `BSSY`/`BSYNC` reconvergent and reliable regions, ordinary and
  reliable `BREAK`, operand-predicate branches, convergence branches,
  warp-uniform any branches, `WARPSYNC.ALL` and collective markers, guarded
  `ELECT`, two-output predicate LUTs, ballot/reduction `VOTEU.ALL`/`ANY`,
  `YIELD`, `NANOSLEEP.SYNCS`, and `NOP`;
- deferred-wait `BAR.SYNC.DEFER_BLOCKING`, nonblocking `BAR.ARV` for full-CTA named
  barriers with immediate or warp-uniform register-selected ids,
  `SYNCS.EXCH.64`, all observed per-lane
  `SYNCS.ARRIVE.TRANS64` arrival/count/expect-tx forms, the paired LDGSTS
  arrival-count/transaction-count forms, parity try-wait,
  invalidate and invalidate-all, including import of the opaque shared-memory
  initialization encoding emitted by CUDA 13.3;
- validated rank-1 through rank-5 `UTMALDG`, rank-1/rank-2/rank-5 `UTMASTG`,
  functional `UTMAPF.L2.4D`,
  rank-2 F32/F32_FTZ `UTMAREDG.ADD`, linear `UBLKCP.G.S`/`UBLKCP.S.G` copies,
  and 32/64/128-byte shared swizzles;
  `LDSM.16.M88.4`, `LDSM.16.MT88.4`, `STSM.16.M88.4`, `HMMA.1688.F32`,
  `HMMA.16816.F32`, their `.BF16` input variants, `HMMA.1684.F32.TF32`,
  `HMMA.1688.F32.TF32`, signed `IMMA.16816.S8.S8`/`IMMA.16832.S8.S8`,
  `F2FP.F16.F32.PACK_AB`, and the selected-half `HADD2.F32` conversion form;
  and
- dependency/cache-control operations whose functional effect is complete
  immediately in the flat functional-memory model.

Control support is not a claim of complete resource modeling: `NANOSLEEP`
does not model the requested sleep duration, and `USETMAXREG` allocation
functionally succeeds without a dynamic register-pool timing model.
`BAR.SYNC.DEFER_BLOCKING` issues an arrival; the frontend waits at subsequent
ordering instructions, rather than blocking immediately at the arrival.

Synthetic gates independently exercise registered tensor-map load/store,
mbarrier transitions, matrix fragment transfers, and nonuniform tensor-core
arithmetic. Raw tensor-map decoding is intentionally limited to the compact
SM120 tiled rank-1 through rank-5 forms validated by device-created
descriptors. The explicit packed descriptor ABI produced by the simulator's
host-side `cuTensorMapEncodeTiled` implementation has separate
rank-1/rank-2/rank-4/rank-5
compatibility decoders, so it cannot be confused with NVIDIA's opaque hardware
encoding. Unknown data types, ranks, strides, or control modes fail closed.
Timing work remains downstream of these correctness gates.

## SM90 WGMMA functional path

The two `wgmma_m64n8k16_f16*_sm90.sassir` generated images are official CUDA 13.3
SM90 images for the unchanged register/shared F16 WGMMA integration kernels.
Nvdisasm decodes and types all 120 instructions in the single-warpgroup image
and all 144 instructions in the three-warpgroup image. Each CTA executes from
PC zero through shared-memory B staging, register A/C fragment construction,
`WARPGROUP.ARRIVE`, `HGMMA.64x8x16.F32`, `WARPGROUP.DEPBAR.LE`, and the real
output stores before every warp reaches `EXIT`. The four-warp image executes
484 dynamic instructions.

The eight single-warpgroup `WgmmaF16M64N8K16IntegrationTest` cases cover
uniform, mixed-sign, random, large-range, nonzero-accumulator, and scale-D-zero
inputs. `ThreeWarpgroupsLaunch12WarpsTest` additionally runs three independent
tiles and descriptor bases concurrently in one 384-thread CTA. Every original
CPU-reference assertion passes in strict SASS mode with no PTX parsing, PDOM
analysis, or functional fallback. The handler decodes Hopper's four-UR `gdesc`
bundle explicitly: the register/shared form uses its second descriptor pair
for B. The multi-warpgroup prologue also gates Hopper's carry-producing
five-operand `IADD3`, `IADD3.X`, and `LEA.HI.X.SX32` address sequence.

The two `wgmma_m64n64k16_f16_ss_*_sm90.sassir` generated images each contain 200/200
officially decoded instructions and 17,408 bytes of static shared memory. They
execute the unchanged FA3-layout integration kernels through shared/shared
`HGMMA.64x64x16.F32`, with 32 FP32 accumulator registers per thread. Both
`KMajorOperandsMatchReference` and `MnMajorBOperandMatchesReference` pass their
original 4,096-element CPU references. This gates both descriptor pairs in the
four-UR bundle, exact inversion of Hopper's base-aware 128-byte shared-memory
swizzle, and the `.tnspB` MN-major B layout. The FA3 forward gate below extends
this to 64x128, 64x176, and 64x192 shared/shared shapes and wide
register/shared accumulation.

`wgmma_m64n8k16_bf16_sm90.sassir` and
`wgmma_m64n8k8_tf32_sm90.sassir` add 112/112 and 104/104 officially decoded
instructions. Their four original random and scale-D-zero cases pass through
`HGMMA.64x8x16.F32.BF16` and `HGMMA.64x8x8.F32.TF32`, respectively. These
gates distinguish packed BF16 pairs from one-TF32-per-register A fragments and
exercise both 16-bit and 32-bit shared B layouts without relaxing the F16
descriptor checks.

The four `wgmma_m64n8k32_fp8_*_sm90.sassir` generated images contain 128/128 decoded
instructions apiece. All E4M3/E5M2 A/B combinations pass the original random
matrix references through their exact `QGMMA.64x8x32.F32` opcode, including a
mixed-format scale-D-zero case. The handler reconstructs four packed FP8 A
values per physical register, decodes byte-wide shared B, and distinguishes
the E4M3 finite-maximum/NaN encoding from E5M2 infinity/NaN semantics.

The four `wgmma_m64n8k32_{s8,u8}_*_sm90.sassir` generated images likewise contain
128/128 decoded instructions each. Their original S8/S8, U8/U8, S8/U8, and
U8/S8 random cases all pass exact integer equality through the corresponding
`IGMMA.64x8x32` opcode. The signedness of A and B is decoded independently,
scale D remains predicate-controlled, and `.SAT` forms clamp the S32
accumulator instead of relying on host overflow behavior.

`wgmma_m64n8k256_b1_and_popc_sm90.sassir` contains 104/104 decoded
instructions. Its random accumulate and scale-D-zero cases reconstruct eight
32-bit words per 256-bit A/B fragment and pass exact S32 equality through
`BGMMA.64x8x256.AND.POPC`. With this gate, all 25 cases currently registered
by the SM90 WGMMA test binary have a generated strict-SASS path and pass their
unchanged assertions.

The complete gate is declared in
`tests/src/wgmma/sass_functional_suite.json` and can be reproduced with:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-13.3
source ./setup_environment
./tests/run_sass_suite.py tests/src/wgmma/sass_functional_suite.json
```

The suite runner checks each filter independently and fails if the aggregate
GTest count differs from the declared 25 tests, so a stale or over-broad
filter cannot silently shrink this coverage claim.

## Closed mbarrier functional path

The two generated SM120 images close the complete barrier group through real
compiled SASS.
The thread-level image contains 1,008 officially decoded static instructions
across 12 kernels and passes all 13 `MBarrierThreadLevelTest` cases. The sanity
image contains 544 instructions across four kernels and passes `Phase`,
`TryWait`, `Arrive`, and `TMA`. Every launch enters at PC zero, executes the
actual CUDA launch dimensions, and reaches `EXIT` without PTX registration,
PTX PDOM analysis, or functional fallback.

These gates cover lane-divergent initialization and arrivals, predication,
arrival counts, expect-tx, multi-phase reuse, invalidate, per-lane shared
atomics, warp synchronization, two-warp producer/consumer behavior, and named
CTA barrier reuse after a warp retires, device-side 1D/F32 tensor-map
construction, a 64-byte TMA load, transaction completion, and element-wise
tile verification.

`named_barrier_sm90.sassir` adds the two Hopper-only barrier cases in a 256
instruction official image. Both two-warp kernels execute their delayed
producer/waiter paths through nonblocking `BAR.ARV`, explicit-count
`BAR.SYNC.DEFER_BLOCKING`, guarded `NANOSLEEP.SYNCS`, both Hopper phase-check
forms, and `SYNCS.ARRIVE.TRANS64.A1T0`. The mixed case additionally performs
its 256-byte shared-to-global `UBLKCP.G.S`; SM90 registers only the reusable
bulk-copy slice rather than SM120 tensor-map opcodes. The source now correctly
restricts the release arrival to lane zero because each mbarrier is initialized
with an expected arrival count of one.

The SM120 and SM90 declarative suites under `tests/src/barrier` enforce their
17-test and 2-test totals, respectively. Together they close all 19 barrier
cases registered across the two architecture groups through strict SASS.

## Closed TMA tensor-map functional paths

The generated `host_tensormap_1d_sm120.sassir` image closes all three
unchanged `HostTensorMapTest` launches. The official CUDA 13.3 image has 552
decoded static instructions across its 64/128/256-element tile
specializations. The tests use separate host-created rank-1 FP32 load/store
tensor maps, add one across every thread, and store through `UTMASTG.1D`.
Their 14 CTAs and 68 warps execute 6,938 dynamic SASS instructions in total;
all 2,176 outputs match the original assertions. Registration defers PTX,
PDOM analysis is skipped, and no functional fallback is available. The exact
binary passes all three cases on the physical RTX 5090.

This gate also distinguishes the simulator's documented 128-byte host tensor
map ABI from the compact device-created SM120 encoding and covers
`PLOP3.LUT`, guarded `ELECT`, `BRA.U.ANY`, unsigned round-up integer-to-float,
unsigned integer min/max, and whole-cache invalidation on a real path.

The companion `multidim_1d_sm120.sassir` closes all three `TMA1DTest` cases
through their device-created compact tensor maps. The single 160-instruction
kernel specialization runs with 1024, 8192, and 256 elements for a combined
74 CTAs, 296 warps, and 27,158 dynamic SASS instructions. Its warp-wide
descriptor copy exercises both implicit 64-bit components of
`ATOMG.E.EXCH.STRONG.GPU`; all 9,472 random FP32 outputs satisfy the original
tolerance checks. The same three cases pass on the physical RTX 5090.

`multidim_2d_sm120.sassir` closes the three `TMA2DTest` cases with a
device-created rank-2 FP32 tensor map and a 16-by-16 tile. Its 192 static
instructions run 64-by-64, 128-by-32, and 256-by-256 matrices for a combined
288 CTAs, 2,304 warps, and 228,096 dynamic SASS instructions. This gate
differentially validates the FP32 `0x0390` compact format alongside the
existing FP16 `0x0310` form; all 73,728 output elements pass the original
checks in strict SASS and on the physical RTX 5090.

`multidim_3d_sm120.sassir` closes all four `TMA3DTest` cases with a
device-created rank-3 FP32 tensor map and a 16-by-16-by-16 tile. Its 408
static instructions include a separately reported 29-instruction CUDA u16
division helper in the same ELF text section. The real path executes
`CALL.REL.NOINC` and returns through `RET.REL.NODEC`; its conversion sequence
also validates U16-to-FP32 and explicit round-toward-zero forms. The four
launches total 18 CTAs, 72 warps, and 85,518 dynamic SASS instructions, and all
69,888 random FP32 outputs pass on strict SASS and the physical RTX 5090.

`multidim_4d_sm120.sassir` closes all three `TMA4DTest` cases with a
device-created rank-4 FP32 tensor map and an 8-by-8-by-8-by-8 tile. Its 400
static instructions include a 30-instruction internal u16 division helper.
The 8-cube, mixed 16-by-8-by-8-by-4, and 16-by-16-by-8-by-8 launches total 7
CTAs, 28 warps, and 29,575 dynamic SASS instructions. Their changing outer
strides validate all three encoded global strides, and all 24,576 random FP32
outputs pass in strict SASS and on the physical RTX 5090.

`multidim_5d_sm120.sassir` closes all three `TMA5DTest` cases with a
device-created rank-5 FP32 tensor map and a 4-by-4-by-4-by-4-by-4 tile. The
448-instruction image needs no internal helper. Its 1/2/4-CTA launches total
28 warps and 11,508 dynamic SASS instructions; the changing shapes validate
all four encoded outer strides. All 6,144 random FP32 outputs pass in strict
SASS and on the physical RTX 5090.

`tensor_reduce_sm120.sassir` closes both `TmaTensorReduceTest` cases through
the 48-instruction `tensor_reduce_add_kernel`. Three one-warp launches execute
111 dynamic SASS instructions and 96 element-wise shared-to-global reductions.
The first case verifies all 32 ordinary FP32 sums; the second differentially
checks that `FLOAT32` preserves the smallest subnormal while `FLOAT32_FTZ`
flushes it to zero. The same binary passes both cases on the physical RTX 5090.
The SASS functional path reuses the backend-neutral tensor-reduction primitive;
no timing behavior is inferred from this gate.

`bulk_store_sm120.sassir` closes all five `BulkGroupIntegrationTest` cases
through four official kernel images totaling 624 static instructions. Their 35
CTAs and 83 warps execute 1,783,913 dynamic SASS instructions and verify 70,208
32-bit outputs. This gate covers linear `UBLKCP.G.S` shared-to-global copies,
16-byte size units, functional commit/wait markers, uniform comparison and
R2UR status forms, negative descriptor displacements, and 32-bit shared-address
wraparound when a pipelined slot is reused. The same five cases pass on the
physical RTX 5090. These are functional correctness claims only; bulk-group
overlap and completion timing remain unmodeled here.

`cp_bw_sm120.sassir` closes all ten `CudaTMATest` cases through ten official
kernel specializations totaling 6,360 static instructions. The nine
standalone launches total 1,530 CTAs, 5,440 warps, and 1,612,328 dynamic SASS
instructions; `PerformanceComparison` contributes 25 more launches and
4,632,830 instructions. The path covers two to six pipeline stages, multiple
producer and consumer warps, 1/2/4 KiB chunks, and an ordinary global-load
control path. It also validates the SM120 launch-header `NCTAID` offsets,
linear global-to-shared `UBLKCP.S.G`, ordinary reconvergence-region `BREAK`,
64-bit global stores, 128-bit global loads, signed `IMAD.HI`, and the
`MUFU.RCP` integer-division refinement sequence generated by CUDA 13.3. Every
launch exactly matches the original host aggregate, and all ten cases pass on
the physical RTX 5090. Copy/barrier overlap and the test's host wall-clock
measurements remain outside this functional claim.

`tma_edge_sm120.sassir` closes the remaining `TmaEdgeCasesTest` launch in the
repository's PTX functional definition. Its one CTA and one warp execute 108
dynamic SASS instructions and produce `{0, 0, 11, 22}` from a four-element
rank-1 U32 tile starting at coordinate `-2`. The gate adds the validated
device-created U32 descriptor format `0x0100` and exercises lower-bound OOB
zero fill. The source was also corrected so the `.sync.aligned` 128-byte
tensor-map copy runs with all 32 lanes; CUDA 13.3 lowers it to one 32-bit
`ATOMG` per lane. Together, the generated images now give every one of the 37
SM120 `tma` gtests a strict execution-driven SASS gate.

`tests/src/tma/sass_functional_suite.json` binds all ten test families to the
corresponding generated images and enforces that aggregate 37-test count.

There is an unresolved hardware differential for this one edge case. The PTX
simulator and SASS frontend pass, but the corrected cubin returns
`cudaErrorIllegalInstruction` (715) for coordinate `-2` on the local RTX 5090
with driver 610.57.04. Changing only that coordinate to zero completes on the
same hardware. Consequently, negative-coordinate behavior is a PTX functional
claim here, not a physical-RTX-5090 agreement claim.

## Closed attention functional path

The repository contains an SM120 attention cubin at
`tests/ci/perf/traces/SM120_RTX5090/llama3-prefill-b2-s128/`
(`_llama3_gqa_attn_fwd_qkv_tiled`). The unit build generates its complete
official SASSIR directly from that cubin. A static nvdisasm census gives:

| Result | Count |
|---|---:|
| Static instructions | 3,640 |
| Official opcodes decoded | 3,640 (100%) |
| Operands covered by the typed grammar | 3,640 (100%) |
| Distinct opcode strings | 92 |
| Opcode strings with a registered SM120 handler | 90 |
| Instructions whose opcode has a registered handler | 3,630 (99.7%) |
| Missing opcode strings | 2 |

The execution gate configures the captured `B=2, S=128, QH=32, KVH=8, D=128`
launch and runs CTA `(0,0,0)` with four physical warp states. All warps start
at PC zero and reach `EXIT` after 13,003 dynamic CTA instructions. The path
executes both QK and PV `HMMA.16816.F32` phases, TMA loads/store, causal mask,
FP32 max/sum reductions through `SHFL.BFLY`, `MUFU.EX2`, per-thread local
spills, U16 shared traffic, and transposed matrix-fragment loads.

Q, K, and V use nonuniform, exactly representable FP16 inputs. The gate checks
all 8,192 FP16 elements of the produced 64x128 query tile against an
independent CPU causal-softmax reference with a 0.02 absolute-error ceiling;
the output is prefilled with a sentinel so an unwritten element also fails.
No PTX functional state, dynamic instruction trace, or approximate-NOP handler
participates.

The only missing opcode strings are five `LEPC` and five `CALL.ABS.NOINC`
instructions on dormant CUDA tensor-copy syscall branches. The selected CTA
does execute the local stack used by the attention data path. Registered
handlers still validate their operand forms, so another launch that selects a
syscall branch fails closed instead of silently skipping it.

## Closed official FlashAttention-2 functional paths

The official FlashAttention-2 build produces four fixed-length FP16
specializations across `D=64/128` and full/causal attention. Their unchanged
SM90 and SM120 smoke GoogleTests execute through strict SASS and retain complete
output/LSE CPU-reference checks:

| Specialization and smoke case | SM120 static SASS | CTAs / warps | Dynamic SASS |
|---|---:|---:|---:|
| D64 full, `H32D64FullB2S128` | 3,096 | 64 / 256 | 467,712 |
| D64 causal, `H32D64CausalB2S128` | 3,696 | 64 / 256 | 584,960 |
| D128 full, `H1D128FullB1S256` | 2,648 | 2 / 8 | 35,456 |
| D128 causal, `H16D128CausalB2S128` | 4,536 | 32 / 128 | 425,088 |

The D128-full image contains 512 `HMMA.16816.F32`, 160 `LDSM`, 56
`LDGSTS.E.BYPASS.LTC128B.128`, and 132 `MUFU.EX2` instructions. SASSIR v3
records its 255 registers, 1,024 bytes of static shared memory, 160-byte stack,
and 472-byte parameter structure. CUDA 13.3 reciprocal integer-division
sequences also exercise the implicit 64-bit accumulator pair of
`IMAD.HI`/`UIMAD.HI`.

The scaling suites additionally launch `B=1, H=1, D=128` full and causal
shapes through `S=4096`; those tuning cases check successful execution, while
the adjacent smoke suites remain the numerical correctness gates. Every image
is regenerated from the exact profile/mode executable, so no specialization
shares a stored sassir merely because its kernel bytes happen to match.

## Closed official FlashAttention-3 smoke paths

The four CUDA 13.3 Hopper FA3 forward specializations across `D=64/128` and
full/causal attention now pass their unchanged numerical smoke tests through
strict execution-driven SASS:

| Specialization and smoke case | Static SASS | CTAs / warps | Dynamic SASS |
|---|---:|---:|---:|
| D64 full, `H2D64FullB2S128` | 2,512 | 132 / 2,112 | 200,464 |
| D64 causal, `H1D64CausalB1S256` | 2,872 | 132 / 2,112 | 178,566 |
| D128 full, `H1D128FullB1S256` | 2,576 | 132 / 1,584 | 142,784 |
| D128 causal, `H1D128CausalB1S256` | 3,208 | 132 / 1,584 | 133,443 |
| D128 full, fixed `B9Sq64Sk128H6` | 2,576 | 132 / 1,584 | 658,512 |

Every image has 100% official instruction and typed-operand coverage. The
runtime launches the original 2,944-byte by-value CUTLASS parameter structure;
SM90's indirect constant-bank ABI maps that structure through the synthetic
functional global-memory window rather than interpreting it as SM120's direct
parameter bank.

The paths exercise static and dynamic persistent tile schedulers, returned-old
global atomic add, register-selected named barriers, mbarrier transaction
arrivals, rank-4 TMA loads, rank-5 output stores, 64x128/176/192
shared/shared HGMMA, wide register/shared PV HGMMA, softmax MUFU sequences, and
the real output/LSE epilogues. Hopper GMMA swizzling is base-aware: descriptor
bases inside a 1-KiB 128-byte-swizzle period participate in the XOR coordinate.
A dedicated unaligned-base unit gate prevents the aligned-base bug exposed by
these kernels from recurring.

`tests/src/fa3/sass_functional_suite_sm90_forward_smoke.json` binds each
filter to an exact GTest and enforces the aggregate 5/5 count. The first four
cases preserve the original output/LSE CPU-reference checks. The fifth reuses
the D128 full image for the repository's fixed rectangular `Sq=64`, `Sk=128`,
`B=9`, `H=6` launch; that original test checks successful execution rather
than a CPU reference. All images are regenerated from the suite's
`standard_tests` executable.

The separately compiled instrumented `baseline` breakdown image also closes
its default `B=1, H=1, S=4096, D=128` full-attention path. Its image contains
3,056 static instructions and executes 132 CTAs, 1,584 warps, and 4,764,222
dynamic SASS instructions. This path adds `LDG.E.64`, lexicographic `.EX`
integer ordering,
and typed `~R` integer operands. The unchanged profile test requires a
successful launch and nonzero clock deltas, but does not compare output to a
CPU reference. `SR_GLOBALTIMERLO` reports nanoseconds during timing execution
and remains a monotonic logical source in flat functional execution. The exact
gate is
`tests/src/fa3/sass_functional_suite_sm90_breakdown_baseline.json`.
The same gate also runs the separately compiled `baseline_noprofile` binary:
its target image is byte-identical to the standard D128-full image and executes
4,281,036 dynamic instructions for the same shape. This provides an exact
uninstrumented counterpart; each executable supplies its own generated cache
entry instead of sharing a stored image.

The four backward specializations now also pass their unchanged dQ, dK, and
dV numerical checks. Each generated image contains the three kernels launched
by the test: preprocess, the warp-specialized TMA/HGMMA mainloop, and
postprocess.

| Specialization and smoke case | Static SASS | CTAs / warps | Dynamic SASS |
|---|---:|---:|---:|
| D64 full, `H2D64FullB2S128` | 2,464 | 12 / 112 | 61,436 |
| D64 causal, `H1D64CausalB1S256` | 3,704 | 6 / 56 | 40,012 |
| D128 full, `H1D128FullB1S256` | 2,544 | 10 / 88 | 69,306 |
| D128 causal, `H1D128CausalB1S256` | 2,928 | 10 / 88 | 56,812 |

This path adds ordinary 128-bit global vectors, rank-4 TMA stores, bulk F32
shared-to-global reduction, Hopper's observed TMA descriptor qualifier,
`HGMMA.64x80x16.F32` and `.tnspA`, and the uniform signed-carry high LEA form.
It also closes a control requirement exposed by causal producer cleanup:
`WARPSYNC.ALL` inside an outer BSSY region collects every serialized arm before
`ELECT`, so a stale nonleader pipeline state cannot execute the producer tail.

`tests/src/fa3/sass_functional_suite_sm90_backward_smoke.json` binds the four
filters to exact GTests and enforces the aggregate 4/4 count.
The static and dynamic instruction counts above remain functional coverage
measurements and do not imply timing accuracy.

The four forward smoke cases also complete through the execution-driven SASS
timing path while retaining their CPU output/LSE checks. This closes native
GMMA group commit/wait, per-participant functional state, predicated CTA
barriers, and register-selected named-barrier ids in the common timing backend.
The gate establishes timing-path coverage only: H100 cycle calibration remains
separate from functional correctness and is not inferred from a passing smoke
test.

The same four backward smoke cases now also complete through the
execution-driven SASS timing path, including preprocess, warp-specialized
mainloop, and postprocess.  Linear `UBLKCP.S.G`, `UBLKCP.G.S`, and
`UBLKRED.G.S.ADD.F32.RN` publish their dynamic source, destination, byte count,
and optional mbarrier address to the existing TMA backend without inventing a
tensor-map descriptor.  The current SM90 configuration reports:

| Backward smoke case | Preprocess cycles | Mainloop cycles | Postprocess cycles |
|---|---:|---:|---:|
| `H2D64FullB2S128` | 9,498 | 14,973 | 8,682 |
| `H1D64CausalB1S256` | 9,324 | 18,756 | 8,504 |
| `H1D128FullB1S256` | 9,697 | 29,686 | 9,459 |
| `H1D128CausalB1S256` | 9,291 | 25,931 | 8,671 |

All four retain their unchanged dQ/dK/dV CPU-reference checks.  These values
establish deterministic timing-path coverage under the repository config;
they are not an H100 calibration claim.

The one-tile PackGQA forward workload also closes both of its separately
compiled ordinary-copy barrier forms. Each official image contains 2,952
static instructions and runs `B=1, QH=4, KVH=1, S=32, D=128` over 132 CTAs and
1,584 warps. The default transaction-count image executes 140,756 dynamic
instructions; the `.noinc` arrival-count image executes 140,692. Both pass the
unchanged output and LSE reference checks in strict SASS mode.

This path adds segmented `SHFL.IDX` clamp/mask behavior, generic Hopper
`LD.E.128`/`ST.E.128` global vectors, and synchronous functional completion of
the LDGSTS barrier arrival forms. The default cubin executes
`SYNCS.ARRIVE.TRANS64.RED.A0T1` plus
`ARRIVES.LDGSTSBAR.64.TRANSCNT`; `.noinc` instead executes
`ARRIVES.LDGSTSBAR.64.ARVCNT`. The declarative PackGQA suite selects
`packgqa_default_tests` and `packgqa_noinc_tests` explicitly and enforces 2/2
tests.

Both PackGQA forms also complete through the SASS timing path. The native
`ARRIVES` instruction reuses the existing cp.async-mbarrier arrival backend:
`TRANSCNT` prepares the ordinary pending arrival, while `ARVCNT` selects the
`.noinc` behavior. Functional execution supplies the per-lane mbarrier
address, and the projection uses the configured cp.async commit pipeline
without introducing a SASS-only latency. This is timing-path coverage, not an
H100 cycle-accuracy claim.

The four forward and four backward `small` cases also reuse these exact
standard cubins at `B=32, S=256`, with `H=32` for D64 and `H=16` for D128.
Both declarative suites pass 4/4 in strict SASS mode. The D64 full forward
case alone traverses 41,569,592 dynamic instructions across 132 persistent
CTAs and 2,112 warps, exercising many scheduler work iterations beyond the
smoke path. Backward closes the same larger runtime shapes across preprocess,
mainloop, and postprocess launches. The repository's unchanged `small` tests
assert successful launch/execution only; the smoke gates above remain the
corresponding numerical correctness evidence.

## Closed repository CUDA launch path

The unchanged SM120 integration test `CudaVectorAddTest` now enters the SASS
frontend through its normal CUDA runtime calls. Its generated image is generated from
the exact linked cubin and contains 32/32 officially decoded instructions plus
the cubin parameter-bank ABI (`c[0x0][0x380]`, four parameters occupying 28
bytes). Registration defers all six fatbins in the combined integration binary
without parsing PTX, while `__cudaRegisterFunction` preserves the selected
kernel's mangled symbol.

Both `BasicVectorAddition` and `SpecialValueTesting` pass. Each launch executes
the entire `(1024,1,1)` grid with `(256,1,1)` threads per CTA: 1,024 CTAs,
8,192 physical warps, and 163,840 dynamic SASS instructions. The test's own
host-side element-by-element checks validate all 262,144 FP32 outputs. The
runtime explicitly logs `no PTX fallback` and skips PTX PDOM analysis.

This closes the launch/memory/ABI mechanism, not repository-wide SASS opcode
coverage. A different launched symbol must exist in the selected generated image and
must have complete functional semantics or the launch fails closed. Cubin
resource metadata is now carried by SASSIR v3; registered globals/constants,
CUDA tensor-map registration, and SM90 decoding remain explicit frontend gaps.

The same strict launch path now closes the eight unchanged SM120
`ldmatrix`/`stmatrix` integration cases. Their single cubin contributes 352/352
officially decoded static instructions across eight kernels. Normal x1/x2/x4
and transposed x1 matrix transfers execute 23 to 55 dynamic instructions per
launch and preserve the tests' full element-by-element expected matrices.

It also closes all eight unchanged `CpAsyncSrcSizeTest` cases. Their 296/296
static instructions lower source-size and ignore-source behavior to
`LDGSTS.E.BYPASS.128[.ZFILL]`, an address offset, and a predicate. Execution
uses those SASS fields directly and passes every per-lane byte check for full,
1/4/8/12/15-byte, zero, cache-policy, and predicate-controlled copies.

Together with address operands, vector operands, vector add, and shared-memory
opt-in, this covers all 23 tests from the six sources in the portable SM120
`integration` group. It does not imply that every architecture-specific or
microbenchmark group in the repository is SASS-enabled.

`tests/src/integration/sass_functional_suite.json` binds those six source
groups to their exact generated images and enforces the aggregate 23-test count.

CuBit's frozen disassembly emits 97 raw words for this cubin: the 54 unknown
instructions above plus 43 instructions that decode to text but do not
round-trip through its encoder. The frontend retains the original 128 bits for
every instruction, so later decoder fixes do not require regenerating a
dynamic trace, but these 43 sites should be audited before their decoded
operands are treated as ground truth.

## Closed MMA integration paths

The F16, BF16, TF32, and signed-INT8 SM120 images close all 29 original MMA
integration cases. CUDA 13.3 lowers them to `HMMA.1684`, `HMMA.1688`,
`HMMA.16816`, and `IMMA.16816` forms; the legacy `m8n8k4` F16 shape follows
the compiler-emitted helper through `SHFL.IDX`, selected-half `HADD2.F32`, and
scalar `FFMA`. Every original numerical reference also passes on the physical
RTX 5090.

`tests/src/mma/sass_functional_suite.json` stores only the four GTest filters
and their aggregate 29-test count. The exact `mma_tests` executable supplies
the fatbin/kernel association, and official SASSIR is generated in the ignored
cache. No PTX functional state or dynamic trace participates.

## Closed executable GEMM paths

Correctness is gated by two controlled CUDA 13.3 `sm_120a` cubins in
`tests/src/unit/fixtures` and the larger Triton cubin used for the census above.
Their SASSIR is generated under `tests/build/generated/sass-unit/`. All three use static cubin decode followed by execution-driven SASS PC
stepping, not traces.

| Kernel | Static decode | Dynamic path | Compute | Checked output |
|---|---:|---:|---|---:|
| `sass_gemm_2x2x4` | 56/56 | 42 | 16 `FFMA` | 4/4 F32 |
| `sass_mma_gemm_m16n8k8` | 40/40 | 28 | 1 warp-wide `HMMA.1688.F32` | 128/128 F32 |
| Triton `kernel_tma_gemm` | 936/936 | 2,787 CTA | 2 K stages, 4 warps, `HMMA.16816.F32` | 4,096/4,096 FP16 |

A separate synthetic execution-driven slice validates the foundational
integer/uniform/branch/shared-memory substrate across all 32 lanes. It is not
used as a substitute for the two real-cubin GEMM gates above.

The tensor-core path closes these exact nvdisasm forms: `LDC`, `LDCU`, `S2R`,
`SHF.L.U32`, `LOP3.LUT`, `IMAD.WIDE.U32`, `LDG.E.U16`, `LEA`,
`HMMA.1688.F32`, statically disabled `UIADD3`, `STG.E`, and `EXIT`. Unknown
forms fail closed. With the same nonuniform, exactly representable FP16 inputs,
the physical RTX 5090 and SASS frontend agree with the CPU GEMM on all 128
outputs.
