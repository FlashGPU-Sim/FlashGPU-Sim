# TCGen05 cleanup review and bounded plan — 2026-11-26

Reviewed integration head: `45e4310c`. This is a focused review of the new
TCGen05 backend, PTX/SASS adapters, timing integration, experimental shared-write
gates, completion/cleanup paths, tests and documentation. It is not an audit of
every historical simulator subsystem. No additional blocking functional defect
was established in this review. STS/TMA contention calibration is deferred by
agreement; the experimental write option stays off.

## Findings

1. **Stale scope documentation.** `src/gpgpu-sim/flash/tcgen05/README.md`
   still describes the early single-CTA extraction, excludes paired support
   and says no new timing calibration is implied. That is no longer an adequate
   description of this assembled branch. Replace duplicated scope claims with
   links to the current functional matrix and the separate timing qualification.
   Owner branches must continue documenting their own narrower scope.
2. **PTX execution and timing wiring share one entry file.**
   `flash/tcgen05.cc` is 707 lines. The first part handles operand access,
   allocation, transfers, commit and cluster ownership; the latter part maps
   configuration and static/dynamic instruction metadata into timing operations.
   These are independently reviewable responsibilities and should be separated.
3. **Native timing projection mixes three responsibilities.**
   `sass/timing/timing_projection.cc` is 1,553 lines. `classify` alone is about
   455 lines, and `populate_registers` about 257. Opcode-to-pipeline mapping,
   implicit register dependencies and runtime effect projection are interleaved.
   The recent mask/scale/TMEM dependency additions make this a useful split
   boundary. Preserve all implicit spans and public instruction behavior.
4. **The MMA host fixture is becoming difficult to navigate.**
   `tests/src/unit/tcgen05_mma_test.cc` is 767 lines. Separate descriptor/layout
   validation, arithmetic and accumulator-mask fixtures by responsibility;
   keep independent expected-value oracles and existing test names/registration.
5. **Large historical subsystems need separate scope.** `flash/tma.cc` has
   3,656 lines and `mbarrier.cc` 2,158. The new shared-landing code is only a
   small part of TMA. A future TMA cleanup can separate AGU/descriptor work,
   transaction/response bookkeeping and completion transport. Do not rewrite
   the common TMA state machine merely to finish this TCGen05 series.

The actual common MMA/TMEM implementation is already reasonably divided:
`descriptor.cc` 444, `mma.cc` 496, `ptx_mma.cc` 490, `timing.cc` 442 and
`tmem.cc` 477 lines. Native adapters range from 15 to 392 lines. There is no
reason to split these solely to meet an arbitrary line-count limit.
`native_recurrence_table.h` is 899 lines of generated measurement data, included
only by the timing implementation. Keep it generated from its provenance-bound
CSV; its generator's `--check` passes. Do not hand-edit or fragment the table.

## Checked boundaries

- PTX and SASS continue calling common descriptor/gather/arithmetic/TMEM code.
  Frontend-specific operand extraction is required adaptation, not a reason to
  create another numerical backend.
- WS masking acts on selected values after collector access; stored collector
  contents remain unmasked. Output-lane masks preserve existing accumulator
  values through common writeback.
- TMEM tagged completion, commit/wait cutoffs and CTA cleanup remain in the
  common timing engine. No new split should duplicate ownership or completion.
- Shared landing is gated by `gpgpu_shared_region_write_model`; unsupported
  DSM/multicast/unaligned paths retain their documented behavior. Logical-offset
  bookkeeping is allocated only for modeled transactions and erased on parent
  response or transaction cleanup. The default-off path is retained.
- Rejected dedicated-INT/fair-collector experiments are not enabled in presets.
  Diagnostic evidence must be preserved. The frontend-independent collector
  initialization fix remains on its separately reproduced public-base branch.

These static checks complement, rather than replace, complete numerical launches.
They do not establish new physical bandwidth or contention calibration.

## Proposed changes, in order

| Commit | Change | Acceptance |
|---|---|---|
| 1 | Correct the source README and consolidate current-scope links | Check links and scope against each owner branch; no model changes |
| 2 | Move PTX TMEM handlers/helpers and TCGen05 timing adaptation into separate translation units; preserve the existing external entry points | Build; TMEM/MMA/pair complete-launch gates; timing and PTX-isolation tests; same numerical outputs and cycles |
| 3 | Split native opcode classification and register-dependency projection from runtime effect projection, using a small internal interface | Existing projection tests, mask/wide-register tests and affected dual-frontend launches; compare source/destination dependencies and timing fields, not only successful execution |
| 4 | Split the large MMA unit fixture without changing its oracles or registration | Verify the same test list and run all resulting components |
| Later, separate cleanup | Decompose generic TMA transaction/AGU/completion implementation | Ordinary PTX TMA, native TMA, cluster/DSM and end-to-end workloads; not a prerequisite to this feature |

Do not combine these changes with new timing parameters, scheduler changes,
formatter sweeps or unsupported ISA expansion. Apply each change to its owning
branch and then integrate it; do not copy numerical algorithms into adapters.
Use existing focused gates after each change and one assembled DSV4 regression
after the final structural change. There is no need for a full DSV4 rerun after
every documentation or file-movement commit.

## Current-head DSV4 regression

Before any source refactoring, head `45e4310c` was built successfully and frozen
with its runtime and decoder hashes. A fresh queue contains all **59 original
SASS entries and 58 available PTX entries**, using four jobs with four threads
per job. The internal cuBLAS hash-routing entry still has no recoverable PTX;
it is not replaced by a substitute workload.

The queue uses the original retained-input acceptance configuration and exact
GPU checker, with experimental shared writes explicitly off. It checks whether
the current assembled implementation runs correctly; it is **not** a calibrated
B200 timing-error sweep. Timing calibration is a separate evidence set.

Compare against the previously reviewed cumulative acceptance: strict GPU
equality, complete unchanged sparse residuals, and reviewed routing permutations
remain distinct categories. A new numerical difference must be inspected;
prior permission to tolerate rounding is not blanket acceptance of new errors.

Evidence: `/home/wzr/flashgpu-dsv4-closeout-20261126/queue/report.json`.
Runtime, decoder, harness/config snapshots and reference acceptance are under
the same directory. `runs/CURRENT.json` records the queue. Heavy output is on
`/home` because `/data` has limited free space; original captures are untouched.
The previous 117-entry cumulative result is not relabeled as this new sweep.
