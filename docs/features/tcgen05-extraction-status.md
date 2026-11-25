# TCGen05 private extraction status

The preparation consists of independently testable functional units. Each unit
includes its common backend, PTX/SASS operand adapters, and complete numerical
CUDA launches. A backend-only PR followed by a separate native-adapter PR is
not the plan. All branches below are in the private dev repository; no public
PR has been opened for this preparation.

## Current functional branches

| Unit | Private branch | Current head | Qualified code head | Tests / launches per frontend |
|---|---|---|---|---:|
| Single-CTA TMEM | `feature/blackwell-tmem` | `2560b425` | `2560b425` | 13 / 17 |
| MMA | `feature/blackwell-mma` | `d68f7c69` | `d88e6f5a` | 24 / 51 |
| Paired CTA | `feature/blackwell-paired-cta` | `217bf7f5` | `40267f1b` | 5 / 15 |
| CLC | `feature/blackwell-clc` | `4be6ee02` | `c866f925` | 7 / 13 |
| Numbered Blackwell WS collectors | `feature/blackwell-ws-collectors` | `44fa16cd` | `250b577a` | 6 / 6 |

Changes after the qualified code heads are documentation only. Current private
remote refs, unchanged source/test/config content, frozen runtime and executable
hashes, numerical logs and PTX-isolation logs were checked in
`runs/tcgen-functional-branches-20261125/branch-closure-audit.json`.
The canonical gate index is `report.json` in the same directory. The original
`private-publish.json` records an earlier push, not these later documentation heads.

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

The explicit contiguous broadcast-copy stride fix is present in TMEM and its
paired scale-copy consumer. The negated PTX accumulator predicate fix is in
MMA. WS BF16 fixture inputs are constructed from equivalent raw bits, avoiding
an incidental dependency on the separate scalar-conversion feature.

Detailed matrices and exclusions are in [TMEM](blackwell-tmem.md),
[MMA](blackwell-mma.md), [paired CTA](blackwell-paired-cta.md),
[CLC](blackwell-clc.md), and [WS collectors](blackwell-ws-collectors.md).
Rubin/SM107, research placement, ideal-memory experiments and unqualified ISA
modifiers stay separate. Pair/collector throughput and arbitration timing have
not been newly calibrated by these functional gates.

## Retained DeepInfer acceptance

The full queue contains 59 captured SASS entries and 58 recoverable PTX entries.
Current cumulative results are selected from frozen repair runtimes; they are
not a fresh sweep of the latest integration head.

| Result | Entries | Interpretation |
|---|---:|---|
| Strict GPU equality | 67 | Full retained checker passes |
| Unchanged prior numerical residual | 46 | Complete sparse GPU difference equals the prior accepted residual |
| Reviewed dispatch permutation | 3 | PTX L2/L3 and SASS L3; proof described below |
| Still running | 1 | SASS L2 MoE; not accepted yet |

For the three reviewed MoEs, final output/statistics and all eight remote
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

Fourteen private branches are based directly on public `b3378cb9`, independently
of the TCGen05 stack. Each has public-base reproduction evidence and a passing
local result after the fix. Exact test scope, unchanged-binary checks and limits
remain in the individual reports indexed by
`runs/tcgen-functional-branches-20261125/public-fix-index.json`.
No hosted CI or public PR is claimed.

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

## Remaining before submission

1. Finish SASS L2 MoE and review its full numerical result.
2. Use each functional branch's exact dependency and qualification matrix when
   preparing a PR; do not import unrelated private research configurations.
3. If production code changes, rerun the affected complete-launch gates and
   appropriate ordinary PTX isolation checks. Current documentation-only commits
   retain verified identity with the qualified source/test/config trees.
4. Obtain user instruction before publishing these prepared PRs. This work only
   organizes and validates the private branches.
