# TCGen05 private extraction status

The preparation consists of independently testable functional units. Each unit
includes its common backend, PTX/SASS operand adapters, and complete numerical
CUDA launches. A backend-only PR followed by a separate native-adapter PR is
not the plan. All branches below are in the private dev repository; no public
PR has been opened for this preparation.

## Current functional branches

| Unit | Private branch | Local head | Qualified code head | Tests / launches per frontend |
|---|---|---|---|---:|
| Single-CTA TMEM | `feature/blackwell-tmem` | `a403abda` | `ace09f0c` | 14 / 18 |
| MMA | `feature/blackwell-mma` | `2343730e` | `bc04128a` | 31 / 68 |
| Paired CTA | `feature/blackwell-paired-cta` | `a95069e7` | `5b3981f6` | 7 / 18 |
| CLC | `feature/blackwell-clc` | `c95d1273` | `93bbda30` | 7 / 13 |
| Numbered Blackwell WS collectors | `feature/blackwell-ws-collectors` | `76c5c3b7` | `efdb755f` | 11 / 11 |

The 2026-11-26 refresh redistributes the native-qualified narrow TMEM stores,
physical copy stride, MMA masks/shared TF32/descriptor checks and WS mask/shift
semantics from integration to their functional owners. It leaves calibrated
B200 timing/config changes in the separate qualification layer. Current heads
are local preparation heads, not a claim that remote refs or hosted CI match.
Changes after each code head are documentation only; source, tests and configs
were compared explicitly.

Current evidence: `/home/wzr/flashgpu-feature-closeout-20261126/final-report.json`,
indexed by `runs/CURRENT.json`. It records local CI, complete PTX/SASS launches,
PTX timing isolation, frozen runtime hashes and historical failing attempts.
Older `runs/tcgen-functional-branches-20261125` reports remain historical evidence.

The dependency structure is:

- TMEM builds on reviewed scalar/gather4 prerequisites at `c6549df8`.
- MMA builds on TMEM; paired CTA and numbered WS collectors both build on MMA.
  WS is a sibling of paired CTA, not a descendant of the entire integration stack.
- CLC builds on paired CTA and includes scalar `.b128` and discarded-result
  global-add prerequisites needed by its complete CUDA fixtures.

All five passed their own local CI and three host PTX timing-isolation checks.
Every numerical launch runs through both frontends, synchronizes, copies back
results, and checks them. No production config differs from the reviewed
scalar/gather4 prerequisite. That prerequisite already includes the reviewed
B200 TMA service-interval change; this is not a claim of no changes versus
public main. SASS remains opt-in. Local gates are not hosted CI or hardware
calibration, and do not qualify every ISA modifier.

## Backend ownership and supported scope

| Unit | Shared implementation | Adapter and qualification boundary |
|---|---|---|
| TMEM | `tcgen05/tmem.*`, `copy.*`, common timing engine | PTX `tcgen05.cc`, native `tmem.*`; allocation, register fragments, shared copies, waits and commit |
| MMA | `descriptor.*`, `mma.*`, `operands.h`, TMEM accumulator storage | PTX `ptx_mma.cc`, native `tcgen_mma_dense.cc` / `tcgen_mma_scaled.cc`; dense/scaled arithmetic and layouts |
| Paired CTA | Same arithmetic/storage/copy backend and common cluster ownership | Peer allocation/operands, paired MMA/copy, multicast completion and native allocator lowering |
| CLC | `flash/clc.cc`, ordinary kernel admission cursor | PTX cancellation and native `UGETNEXTWORKID` share one asynchronous service |
| WS collectors | `tcgen05/collector.h`, CTA-scoped banks in the common TMEM manager | Numbered Blackwell WS B fill/use/lastuse/discard; no duplicate native collector state |

The broadcast-copy stride fix is present in TMEM and its paired scale-copy
consumer: zero stride repeats eight rows; explicit stride 128 advances to the
next group. The encodings are not interchangeable. The negated PTX accumulator predicate fix is in
MMA. WS BF16 fixture inputs are constructed from equivalent raw bits, avoiding
an incidental dependency on the separate scalar-conversion feature.

Detailed matrices and exclusions are in [TMEM](blackwell-tmem.md),
[MMA](blackwell-mma.md), [paired CTA](blackwell-paired-cta.md),
[CLC](blackwell-clc.md), and [WS collectors](blackwell-ws-collectors.md).
Rubin/SM107, research placement, ideal-memory experiments and unqualified ISA
modifiers stay separate. Pair/collector throughput and arbitration timing have
not been newly calibrated by these functional gates.

## Separate timing qualification layer

`feature/blackwell-timing-qualification` retains the assembled implementation
and hardware-calibrated performance layer. It is not a minimal functional PR.
Its model includes typed MMA rates and shape-dependent B200 recurrence, TMEM
LD/ST service and completion widths, CP service, ownership and queueing, and
MMA shared-operand read contention. Both frontends feed common timing metadata.
Ordinary public PTX defaults are not silently replaced by B200 parameters.

The 2026-11-26 retained resource qualification records 540/540 dual-frontend
numerical checks, 144 TMEM timing cases and 30 copy timing cases. Maximum
absolute two-length slope errors are 14.59% and 13.17%, respectively. Mixed
MMA/TMEM numerical checks pass but MMA retains up to about 17% timing residual.
These results belong to frozen integration runtimes, not the owner gate above.
Evidence: `/home/wzr/flashgpu-hardware/b200/resource-model-fix-20261126/qualification-summary.json`.

The existing B200 MMA/LDS regional shared-read model is enabled in integration.
The newer STS/TMA regional write extension remains **experimental and OFF**.
It does not yet reproduce shared-control/operand-delivery contention. It must
not be presented as calibrated simply because its numerical checks pass.
Packed/half-warp TMEM, paired copy/contention timing and SHIFT-window boundaries
remain explicitly unqualified or deferred. These are bounded timing follow-ups,
not a commitment to fit every error before publishing the functional features.
Rubin/SM107 and 3D placement/NoC configurations are outside this series.

## Retained DeepInfer acceptance

The full queue contains 59 captured SASS entries and 58 recoverable PTX entries.
Current cumulative results are selected from frozen repair runtimes; they are
not a fresh sweep of the latest integration head.

The SASS-only entry is `pro-hash-routing-k001`, an internal cuBLAS SGEMM.
Its exact entry is absent from the archived cuBLAS/cuBLASLt embedded PTX dumps.
The original cubin passes SASS numerical checking; no substitute GEMM or guessed
parameter ABI is counted as PTX acceptance. This source-availability limit is
recorded in `runs/feature-series-20261115/deepinfer-dual-frontend-manifest-v13.json`.

| Result | Entries | Interpretation |
|---|---:|---|
| Strict GPU equality | 67 | Full retained checker passes |
| Unchanged prior numerical residual | 46 | Complete sparse GPU difference equals the prior accepted residual |
| Reviewed dispatch permutation | 4 | PTX and SASS L2/L3; proof described below |
| Running / unresolved new failures | 0 | All selected results completed and reviewed |

For the four reviewed MoEs, final output/statistics and all eight remote
combine buffers are byte-identical to the retained reference. All 3072
expert/source-rank token lists preserve their multisets. Aligning active rows
by `(source rank, token, topk)` within each expert makes token data, routing
weights, scales and L2 intermediates byte-identical. Every changed tracked
word is classified as workspace/intermediate data. The separate acceptance
category preserves the raw `NEEDS_REVIEW` result and is bound to hashed
receipts. It does not imply GPU bit-exactness or prove rounding as the cause of
an existing residual.

Authoritative evidence: `runs/native-final-audit-20261125/cumulative-acceptance.json`,
`manual-reviews.json`, and `moe-{l2,l3}-{ptx,sass}-*.json`, indexed by
`runs/CURRENT.json`. `collect_acceptance.py` verifies review receipts rather than
modifying the original checker or queue reports.

The earlier native MoE stall was fixed in its owning
`feature/native-linear-bulk-descriptors` branch (`8ec82c76`): encoded cluster
payload and completion-barrier owners are resolved independently before entering
the existing shared TMA backend. Four host tests and five launches per frontend
passed, including cluster size two. Evidence:
`runs/native-bulk-cluster-20261125/functional-unit-qualification.json`.
This is an ordinary memory/control prerequisite, not new MMA arithmetic.

## Independent common fixes

Fifteen independent fix branches are based directly on public `b3378cb9`, independently
of the TCGen05 stack. Each has public-base reproduction evidence and a passing
local result after the fix. Exact test scope, unchanged-binary checks and limits
remain in the individual reports indexed by
`runs/tcgen-functional-branches-20261125/public-fix-index.json`.
The additional collector ownership fix is indexed by
`/home/wzr/flashgpu-feature-closeout-20261126/public-collector-reproduction.json`.
Its production-helper reproducer fails before the fix, then all eleven
collector tests and three PTX isolation tests pass. No hosted CI or public PR
is claimed.

| Private branch (`fix/…-public`) | Fix head | Scope |
|---|---|---|
| `ptx-signed-byte-load` | `b9430be0` | Scalar signed-byte extension |
| `ptx-shuffle-source` | `39c27501` | Snapshot aliased shuffle sources |
| `ptx-cta-reduction-negation` | `0c5fd996` | Negated reduction predicates |
| `ptx-mbarrier-async-wakeup` | `56051b88` | Wake suspended remote bulk/tensor-TMA barrier owner |
| `ptx-tensor-coordinate-literals` | `45737136` | Ordinary rank-1..5 tensor-coordinate literals; no gather4 expansion |
| `pipeline-dump-empty-threads` | `bc1918de` | Null idle-thread diagnostic guard |
| `ptx-typeless-control` | `0ca9132d` | Trap/brkpt predecode; execution behavior unchanged |
| `ptx-integer-literals` | `7463cec0` | Full-width integer parsing and range checks |
| `ptx-vector-modifier-order` | `1da2dcef` | Valid scalar/vector modifier ordering |
| `ptx-call-argument-ownership` | `b2e3c9a0` | Value-owned device-call argument buffers |
| `tensormap-diagnostic-bounds` | `3902d5eb` | Invalid-rank diagnostic bounds |
| `ptx-funnel-shift-boundaries` | `ce1e3aee` | Existing PTX shift forms and common primitive |
| `shared-store-fence-visibility` | `53225c29` | Fence waits for queued shared-store visibility |
| `ptx-leading-bit-boundaries` | `9ff0a6a0` | Existing signed/unsigned 32/64-bit bfind boundaries |
| `collector-set-ownership` | `ce26c9cc` | Scheduler ownership uses each collector set’s local index, preserving global arbiter IDs |

Two other candidates remain with their owning features:

- `957a0caa`: 32-byte-per-lane coalescing is required by the new native `.256`
  memory forms, not an independently demonstrated public-frontend bug.
- `4a8c7b28`: cached-load completion by dynamic instruction. The ordinary
  predicated-load control passes public PTX and SM120 SASS without the patch.
  The existing failure uses the new `WARPSYNC.EXCLUSIVE` grouping behavior;
  keep the correction with warp reductions until a public-only failing launch
  is demonstrated. Passing this control does not prove all cached-load schedules
  correct. The test-only branch is `audit/cached-load-public-controls`.

The initial SM100 decoder rejection in the latter audit occurred before kernel
execution and is excluded from bug evidence. Ownership records are in
`additional-public-fix-audit.json` and the candidate-specific reports.

## Preparation completion and later submission

Private preparation is complete for the functional units and qualification
matrix above. The final SASS L2 MoE has the same verified dispatch-permutation
classification: 1,408,315 changed tracked words are fully classified, final and
remote-combine outputs are unchanged, and all active intermediates match after
metadata alignment. No selected acceptance entry remains running or unreviewed.
The acceptance still uses frozen repair runtimes, retains the original GPU
residuals, and excludes the unavailable internal SGEMM PTX entry.

Before a later public submission, use each functional branch's exact dependency
and qualification matrix; do not import unrelated research configurations. If
production code changes, rerun the affected complete-launch gates and appropriate
ordinary PTX isolation checks. Current documentation-only commits retain verified
identity with the qualified source/test/config trees. Public submission and
hosted PR CI remain separate future actions; no PR is opened by this preparation.
