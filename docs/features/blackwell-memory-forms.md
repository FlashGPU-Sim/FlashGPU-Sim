# Blackwell memory forms used by retained DeepInfer operators

This private child of `feature/blackwell-ws-collectors` extracts memory forms
already present in the retained full private implementation. It is a supporting
feature for dual-frontend C4 attention and MegaMoE acceptance, not a public PR or
an assertion that either complete operator has passed on this extracted stack.

## Qualified implementation

- PTX accepts `L1::no_allocate`, `L2::evict_first`, and L2 prefetch-size hints
  64/128/256 bytes. They do not override the existing cache operator. Replacement,
  allocation and prefetch policy effects remain uncalibrated; accepting these
  hints does not imply that their hardware performance effects are modeled.
- PTX `st.bulk{.weak}{.shared::cta}` and native `UMEMSETS.64` share extent
  validation and zeroing in `flash/shared_bulk.h`. Explicit shared and generic
  addresses, 32/64-bit byte counts, and zero-sized operations are covered.
  Nonzero fill, misaligned extents, overflow, remote or non-shared destinations
  are rejected. Native word counts are converted to bytes before the shared
  helper. A uniform native instruction writes once, not once per active lane.
- Both adapters report actual bytes into the existing shared-memory timing path.
  Zero-sized operations create no memory transaction. No new measured UMEMSETS
  latency/throughput or hardware microarchitecture is claimed.
- Native unmasked 256-bit loads use the existing global load implementation.
  The upper and lower 128-bit register groups may be noncontiguous; both groups
  participate in scoreboard and RF write metadata. This extraction registers
  the three exact opcodes produced by the new CUDA 13.3 fixtures. Masked 256-bit
  loads/stores and other native modifier combinations remain separate work.

The required common coalescer width extension is kept in its own commit
(`957a0caa`), with a load/store, contiguous/broadcast and inactive-lane test.
It adds 32-byte per-lane accesses to the existing sector path; other widths
retain their previous behavior.

## Validation

`tests/src/memory_forms/launch_test.cu` retains PTX and sm_100a cubin in one
executable: four shared bulk tests and three cache-hint vector load tests.
Bulk tests initialize nonuniform data, use two CTAs and four lengths including
zero, and check untouched prefix/suffix guards. Loads check all 64-bit payloads.
Host tests cover extent validation, failure propagation, cache predecode,
uniform memory effects, register group ordering, predicates and dependencies.

Run CI job `sm100-memory-forms` with CUDA 13.3. Its blocking post-check launches
the same frozen executable through PTX and SASS timing frontends. Regression
receipts, exact runtime hashes and preliminary failures are preserved outside
git under `runs/memory-forms-20261125`; consult `runs/CURRENT.json` for status.

The next real-module parser blockers after these memory forms are FP8 packed
conversion for `attention-c4-fp8` and `redux.sync` for `mega-moe-l2`.
Those remain extraction/qualification work; module parsing is not numerical
acceptance of a complete operator.

ISA references:
- [PTX st.bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
- [PTX ld.global.nc](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld-global-nc)
- [Cache eviction hints](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints)

## Retained 128-bit no-allocate loads and byte stores

The native `LDG.E.NA.128` and `STG.E.U8` forms use the existing global
load/store functional adapter and LSU timing path. A same-binary PTX/SASS test
checks all four loaded words and predicated one-byte writes with untouched
neighboring bytes. The `NA` cache hint has the same qualification boundary as
the earlier 256-bit forms: decoding it does not introduce a hardware-calibrated
L1 eviction policy.
