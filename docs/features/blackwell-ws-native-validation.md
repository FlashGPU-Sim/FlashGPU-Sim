# B200 WS native validation, 2026-11-25

The B200 capture is complete and locally verified. This follow-up qualifies
WS behavior and measures the existing timing model; it does not add timing
constants or change a production configuration.

## Evidence and preservation

Workspace paths (outside this worktree):

- `data/hardware/b200/ws-mask-20261125/manifest.json`: authoritative collection
  index, archive hashes, source revision and storage locations.
- `data/hardware/b200/ws-mask-20261125/host-46.243.147.43/attempt-r2/`:
  raw native clock64 samples, commands, 49 NCU full reports and 49 CSVs.
- `runs/tcgen-ws-b200-20261125/host-46.243.147.43/qualification.json`:
  collection and validation receipt.
- `runs/tcgen-ws-b200-20261125/timing-comparison-r2/report.json`:
  original 32 SASS timing runs, before expanding the matrix.

All 176 native timing cases and 12 numerical tests (13 launches) pass.
1312 files were checked against final remote SHA256 values. The complete
16,703,443-byte compressed capture has SHA256
`d4ce3977afa300f4fefa932bcb4ab45372bfed68e0414b9aad29363cfa7dfbbe`.
Identical verified archives are retained on /data and /home, different disks
on the same host. This is not an off-host backup. Sources, cubins, disassembly,
compiler/CUTLASS provenance, failed earlier attempts and raw reports are kept.
Six unavailable CTC RX/TX NCU metrics are recorded in the original warnings.

The corrected native fixture and shared mask backend use first-span saturation
for out-of-range start counts. Ninety native boundary probes establish this
behavior; both frontends use the same correction. This is a functional fix,
independent of timing accuracy.

## Measured timing scope

BF16 to FP32, M64/N128/K16, K-major, one owner and accumulator; a warm
collector precedes each burst. Values below are unprofiled **issue** slopes,
(median burst128 - median burst32) / 96, in core cycles per MMA. NCU duration
is not substituted for these samples; NCU uses base clock control. Completion
slopes include commit/wait/control effects and are reported separately.

| A storage | Actual shift | Discard/fill | Use | Alternating fill/lastuse |
|---|---:|---:|---:|---:|
| Shared | 0 | 48 | 32 | 45 |
| Shared | 8 | 50 | 32 | 46 |
| Shared | 16 | 52 | 32 | 47 |
| Shared | 32 | 60 | 32 | 49 |
| TMEM | 0 | 44 | 32 | 38 |
| TMEM | 8 | 48 | 32 | 40 |
| TMEM | 16 | 52 | 32 | 42 |
| TMEM | 32 | 60 | 32 | 46 |

Maximum shift 32 with actual shift 0 matches the maximum-0 case. Mixed and
all-zero masks do not improve the measured fill/use slope at shift 32.
Neither observation proves physical fetch granularity or a particular
collector implementation. Functional storage must still retain the complete
N + maximum-shift window so later reuse with a different shift is correct.

## Accuracy policy and model limits

The initial 16 slope comparisons (shared/TMEM A, four modes, shift 0/32)
have at most 13.33% issue-slope error. Existing compute and shared-supply
modeling explains much of this behavior. It does not identify the physical
cause of the remaining shared-A fill/lastuse penalty or the shift-dependent
service cost. Adding a per-transition penalty solely to fit these observations
would not establish a hardware mechanism.

Accept errors within 15% for this qualification; aim for 10% only when a
supported hardware mechanism explains the correction. Do not tune scalar
constants independently for each case. Keep issue, completion and whole-kernel
errors separate. The expanded matrix completed 96/96 SASS runs with all output checks passing,
using the unchanged `adafeef54` runtime. Its 48 slope comparisons have 35 within
10% and 44 within 15%, for both issue and completion slopes. The four exceptions
are maximum-shift 32 / actual-shift 0, discard/fill, shared/TMEM A:

| A storage | Native issue | Simulator issue | Issue error | Completion error |
|---|---:|---:|---:|---:|
| Shared | 48 | 56 | +16.67% | +20.24% |
| TMEM | 44 | 52 | +18.18% | +18.18% |

The simulator charges the full functional lookahead window to shared supply.
Native timing shows no extra cost from declaring the larger window alone.
This identifies a model limitation, but not the physical collector fetch
mechanism. The exceptions remain explicitly unqualified against the 15%
threshold; no transition penalties or per-case constants are introduced.
All other issue-slope errors are at most 13.33%.

Full expanded evidence: `runs/tcgen-ws-b200-20261125/timing-comparison-r2-expanded/`
contains `report.json` (execution/output checks) and `comparison.json` / CSV
(accuracy checks). The full twelve-test native fixture is additionally being
replayed through PTX and SASS; its separate pending receipt is
`runs/tcgen-ws-b200-20261125/native-r2-full-dual/report.json`.
The final local summary is generated at
`runs/tcgen-ws-b200-20261125/local-validation-summary.json` when both finish.
This scope does not qualify other shapes, precisions, multiple owners, or
multi-CTA contention.
