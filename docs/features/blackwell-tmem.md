# Blackwell TMEM data movement: shared backend and both frontends

This feature is a functional unit, not a backend-only preparation PR. It brings
allocation, register transfers, shared-memory copies and their completion path
together with PTX/SASS CUDA launch tests. The private preparation branch is
`feature/blackwell-tmem`, combining the reviewed scalar (#64) and gather4 (#65)
prerequisites. Those prerequisite commits must not be presented as new TMEM code.

## Implementation boundary

- `flash/tcgen05/tmem.*` owns physical word storage and register-fragment
  mapping, including warp partition selection and adjacent-column packing.
- `flash/tcgen05/copy.*` maps shared descriptors and broadcast layouts into
  physical source addresses. Both PTX and SASS call this implementation.
- `flash/tcgen05.cc` decodes PTX options; `sass/functional/tmem.*` decodes
  native allocation/transfer/copy operands. Native bitmap allocation follows
  the compiler lowering; it uses the same word storage as the PTX allocator.
- Both frontends enqueue LD/ST/CP and waits/commit into the existing TCGen05
  timing engine. Native wide register dependencies survive pipeline copies;
  the native write barrier is held through the configured completion latency.
- Blackwell TMEM operations and mbarrier polling cannot cross unfinished
  deferred CTA barriers. Compiler allocator helpers (shared bitwise atomics,
  REDUX AND and pool relinquish) are included because actual cubins use them.

No MMA arithmetic, paired-CTA allocation/copy/commit, CLC, SM107, collector,
research placement, ideal-memory switches or new hardware timing constants are
introduced here. Existing ordinary PTX scheduling is not replaced.

## Qualification matrix

The same sm_100a executable contains PTX and cubins. Independent host expected
values are checked after normal CUDA launch/synchronize/copy-back.

| Area | Complete-launch coverage |
|---|---|
| Allocation/lifetime | 32/64/256 columns, multiple CTAs, repeat launches, relinquish and deallocation |
| Register layout | Cross-shape 32x32b, 16x64b, 16x128b, 16x256b and explicit-offset 16x32bx2 loads; plain and half-split stores |
| Pack/unpack | 32x32b and 16x32bx2, four warps using the same base; unpacked words and repacked values both checked |
| Wide dependencies | x128 load then x128 store, every column checked against data initialized with narrow stores |
| Shared copies | 128x256b, 128x128b, 64x128b warpx2::02_13 / 01_23, 32x128b warpx4 |
| Descriptor layouts | Canonical unswizzled K-major and 32/64/128B swizzle on 128x256b |
| Completion | CP → commit → mbarrier wait → LD; explicit LD/ST waits |

Component tests additionally cover invalid descriptor rejection, out-of-range
transfers without partial writes, negative native address offsets, zero stores,
unselected lanes/columns, and preservation of register dependencies beyond the
fixed operand arrays. This is numerical/ordering qualification, not performance
calibration or complete ISA coverage. Other combinations require their own
launch cases; SM103 reduction loads and Rubin compression are not enabled.
The existing PTX-only linear `cp.4x256b` path is retained unchanged, outside
this qualification; its native adapter/layout needs qualification with MMA.

Reference: [NVIDIA PTX ISA, TCGen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions),
especially tensor memory access restrictions, LD/ST and asynchronous copy.

## Reproduction and CI

Build the simulator with CUDA 13.3, then `make -C tests build-sm100-tcgen05`.
Run `tests/scripts/run_tcgen05_dual_frontend.py` with `--binary`, `--config-dir`,
`--cuda`, `--output` and `--disable-ptx-reorder`; see `--help`.

The `sm100-tmem` CI job runs component tests, then
`tests/ci/check_tmem_dual.py` runs the exact same CUDA executable through PTX and
SASS timing, requiring 13 numerical tests / 17 launches for each. Its temporary
configs override frontend selection and disable PTX reordering. The synthetic
inline-PTX kernels do not supply complete SASS line anchors, so the shipped
SASS-guided scheduling mode is not a valid test harness for these cases.
No production config is changed. This CI wiring is not itself a hosted CI pass.

## Subsequent functional units

1. **Blackwell MMA:** descriptor/format/scales, arithmetic, native and PTX
   operand adapters, timing/commit, and end-to-end MMA numerical fixtures.
   Do not publish the old `tcgen05-shared` backend as a separate prerequisite PR.
2. **Paired CTA:** allocation, peer TMEM/shared ownership, paired MMA/copy and
   multicast completion; reuse the merged public cluster/DSM lifecycle.
3. **CLC:** launch cancellation/work queue and PTX/SASS adapters.
4. **Rubin:** separate SM107 instruction/layout and collector/early-read work.

The old `tcgen05-native` stack also contains general conversions, ordinary
memory forms, reductions and already-merged public fixes. Audit those against
#64/#65 and public `flash` before moving them; they do not belong wholesale in
the MMA PR. The historical DeepInfer acceptance of the combined stack is not
acceptance of this newly extracted TMEM branch.

## Local extraction checks

The first extracted version passed 698 selected host component tests (including
PTX timing isolation and the SASS adapters). The new CI job also passed locally:
38 TMEM component checks and 12 numerical tests / 14 launches per frontend.
One-thread and two-thread runs produced identical cycles within each frontend.
These are simulator qualification results; no new GPU measurements were made.
The CUDA 13.3 host sweep omits the existing trace-validation test whose driver
context-creation API belongs to the older-toolkit CI configuration.

## Explicit contiguous copy descriptors

The shared copy decoder also accepts a 128-byte eight-row stride for the
unswizzled 32/64-row broadcast forms. DeepGEMM's retained SM100 projection uses
`0x400800003140` (start `0x31400`, LBO zero, SBO 128); the initial extraction
incorrectly required SBO zero. Both encodings produce the same contiguous image.
The implementation is shared by PTX and SASS. Noncanonical strides, nonzero LBO
and broadcast swizzles remain unqualified and are rejected.

Acceptance includes all three broadcast selectors, plus paired scale copy
feeding a block-scaled MMA across two independent clusters. The retained SASS
`projection-index-q` now completes: its complete 159-byte mismatch evidence is
identical to the already-reviewed combined private stack. This preserves the
prior numerical qualification; it is not bit-exact agreement with hardware.
PTX acceptance of that real operator still needs the packed conversion feature.
