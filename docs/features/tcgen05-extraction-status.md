# TCGen05 private extraction status

Each functional unit includes its common backend, PTX/SASS operand adapters,
and complete numerical CUDA launch tests. The obsolete split between a
standalone common backend and a later native adapter is not the submission
plan. No new public PR is authorized by this preparation.

## Pinned functional branches

These are historical extraction checkpoints, not aliases for the latest
integration worktree. Later corrections must be included before submission.

| Unit | Private branch / checkpoint | Pinned launch qualification per frontend |
|---|---|---|
| Single-CTA TMEM | `feature/blackwell-tmem`, `b2613ce2` | 12 tests / 14 launches; storage, allocation, LD/ST/CP and completion |
| MMA | `feature/blackwell-mma`, `17a48226` | 23 / 49; dense/scaled arithmetic, TMEM-A, WS default-discard and swizzles |
| Paired CTA | `feature/blackwell-paired-cta`, `efa6866c` | 4 / 14; peer ownership, allocation, MMA and multicast completion |
| CLC | `feature/blackwell-clc`, `0745991b` | 7 / 13; common cancellation service and full launch work accounting |
| Numbered Blackwell WS collectors | `feature/blackwell-ws-collectors`, `c07ebf23` | 6 / 6; fill/use/lastuse/discard, all four slots, shared/TMEM A |

Detailed scope and limitations live in the corresponding feature documents.
The later contiguous-copy fix adds a TMEM launch and a paired-MMA launch.
The negated accumulator predicate fix adds two MMA launches. These newer
qualifications must not be attributed to an older pinned branch.

## Refreshed functional branches

The following private branch heads replace the pinned checkpoints above for
submission preparation. They retain full backend + PTX/SASS + CUDA launch
qualification within each functional unit. No public PR has been opened.

| Unit | Refreshed head | Immediate dependency | New coverage folded in |
|---|---|---|---|
| TMEM | `2560b425` | reviewed scalar/gather4 prerequisites | explicit contiguous broadcast stride; immutable fixture/runtime selection |
| MMA | `d88e6f5a` | TMEM | negated PTX accumulator predicate; 24 tests / 51 launches per frontend |
| Paired CTA | `40267f1b` | MMA | explicit contiguous scale copy feeding paired MMA |
| CLC | `c866f925` | paired CTA, scalar b128, discarded-result global add | inherited corrected TCGen05 backend |
| Numbered WS collectors | `250b577a` | MMA | same six numerical fixtures, without unrelated CLC/TMA/conversion prerequisites |

The WS branch is a sibling of paired CTA, not a descendant of the whole
integration stack. Conflict resolution retains the single-CTA target adapter;
collector state and arithmetic are unchanged. The WS fixture generates the same
exact BF16 inputs from raw bits, with host equivalence checks, so its input
construction does not require the separate scalar conversion feature. All six
collector cases and their output oracles are retained. Ordinary PTX defaults
are retained.

All five refreshed heads passed their own local CI and three ordinary PTX
timing-isolation checks each. Complete CUDA launch qualification per frontend:

| Unit | Numerical tests | Launches |
|---|---:|---:|
| TMEM | 13 | 17 |
| MMA | 24 | 51 |
| Paired CTA | 5 | 15 |
| CLC | 7 | 13 |
| Numbered WS collectors | 6 | 6 |

CLC required rebuilding stale generated BookSim dependencies, without a source
change. The isolated WS input fixture passed both frontends. All five qualified
heads are pushed to the private dev repository; no public PR was opened. The
GitHub Actions matrix planner includes each feature's CUDA 13.3 job, but hosted
CI has not been qualified by these local runs.

Exact heads, frozen runtimes, logs, isolation results and private push evidence
are under `runs/tcgen-functional-branches-20261125/`, indexed by
`runs/CURRENT.json`. These gates do not replace the retained DeepInfer acceptance:
at this checkpoint 67 entries match GPU strictly and 46 retain the prior accepted
numerical residual. Two PTX MoE entries remain running or queued. Two SASS MoE
entries stopped making work progress after about 130,000 cycles; they are not
acceptance passes. Rank-zero snapshots initially showed matching mbarrier state
and a global counter of 96 against an expected 128. Rank-one snapshots then
identified the cause: UBLKCP.S.G credited an encoded cluster address as a local
barrier key, separate from the canonical key used by arrival and wait. The fix
resolves payload and completion owners independently and forwards explicit
targets to the existing shared TMA backend. An eight-combination owner fixture
fails with the frozen pre-fix runtime; the repaired local CI passes four unit
tests and five numerical launches per frontend, including cluster size two.
Full MoE numerical acceptance remains pending. Evidence is under
`runs/native-bulk-cluster-20261125/`. One debugger-invoked pipeline dump crashed
on a null PTX thread; this diagnostic failure is recorded separately from the
original stall. Evidence is indexed by
`tcgen_extracted_sass_moe_stall_20261125` in `runs/CURRENT.json`. The residual
classification does not prove rounding or strict GPU agreement.

## Integration prerequisites and fixes

The retained DeepInfer operators also require ordinary conversions, memory
forms, tensor descriptors, cluster metadata and control instructions. Those
are separate commits with their own tests, not additional MMA arithmetic.
Direct global loads/stores share their native address normalization and retain
the common memory request path. These routing prerequisites are not TCGen05
arithmetic features. For example, CTA reductions share the numerical fold but keep each frontend's
existing arrival/completion interface; native B2R consumes the published result.

Recent independent common fixes include signed-byte load extension, integer
literal tensor coordinates, aliased shuffle source snapshots, and negated CTA
reduction predicates. Their private branches preserve extraction history;
public-main applicability and CI must be verified independently before a public
PR. A private green integration test is not evidence of public-main qualification.
The TCGen05 accumulator predicate correction instead belongs to MMA.

## Public-fix dependency audit

A read-only check against public `flash` at `b3378cb9` confirms that the
production patches for the following candidates apply without the TCGen05
feature stack: signed-byte load extension (`7638a273`), aliased shuffle source
snapshot (`58c96a6f`), negated CTA reduction predicates (`c0122c3d`), and suspended
barrier wakeup (`55178b4e`). Applicability alone is not qualification; the
independent public-base results below supply separate before/after evidence.
Private integration fixtures must not be presented as public-main tests.

Signed-byte loads now have independent public-base evidence: private branch
`fix/ptx-signed-byte-load-public` contains a test-only commit `6cae27c3` and
fix `b9430be0`, both based on public `b3378cb9`. Exhaustive scalar global/shared
byte loads fail before and pass after using the same executable; both per-test
cycle counts are unchanged. The tests are wired into existing SM90/100/120
integration CI, but only local SM100 execution has been qualified so far. This
is not a hosted CI pass or a new public PR. Evidence:
`runs/public-signed-byte-20261125/`.

Aliased shuffle inputs also have public-base evidence on private branch
`fix/ptx-shuffle-source-public`: test-only commit `71a64db9`, fix `39c27501`.
The same executable has three failures before and five passes after, with all
five cycle counts unchanged. Coverage includes up/down/bfly/idx, segmented
widths, and repeated predicated execution. Local SM100 evidence is under
`runs/public-shuffle-20261125/`; hosted CI and public submission remain pending.

Negated CTA reduction predicates now have independent public-base evidence on
`fix/ptx-cta-reduction-negation-public`: test-only commit `3654d264`, fix
`0c5fd996`. Both implicit-count and explicit-count numerical tests fail before
and pass after using the same executable, with unchanged cycles (5482, 5367).
The fixtures exercise popc/and/or and all/none/partial predicates. Evidence is
under `runs/public-cta-reduction-20261125/`; only local SM100 is qualified.

Asynchronous barrier wakeup is qualified on private branch
`fix/ptx-mbarrier-async-wakeup-public`: test-only commits `6885d9c1` and
`ff6830bb`, fix `56051b88`. Remote bulk and tensor TMA each complete all initial
transactions but deadlock with the public baseline's sleeping owner warp.
The same executable/configuration passes all eight transfer rounds after the
fix, in 10491 and 10453 cycles. Two local CTA controls pass before and after.
These tests use ordinary TMA, with no TCGen05 dependency. Evidence:
`runs/public-mbarrier-wakeup-20261125/qualification.json`. Existing SM90 cluster
CI selects the new remote tests; only local SM100 execution is qualified here.

Tensor-coordinate literals (`85f3e067`) need an explicit port: the private
parser context includes the separate narrow mov-pack extension, and its rank
validation references `GATHER4_OPTION`, which is absent from this public base.
A public ordinary-TMA fix must preserve public mov-pack behavior and qualify
ordinary rank-1..5 coordinates; gather4 validation belongs in the dependent
feature. Do not transplant the integration parser hunk wholesale.

The negated accumulator predicate correction (`987680ef`) is different: it is
inside the new TCGen05 MMA adapter and has already been folded into the MMA
functional unit. It is not an independent public-main fix.

## Acceptance evidence boundaries

The original combined private stack and this extracted stack are different
artifacts. Fresh extracted acceptance covers 59 retained SASS entries and 58
recoverable PTX entries. The missing PTX entry is the archived internal cuBLAS
routing SGEMM; its original SASS remains available. Results are indexed by
kernel and frontend, with frozen runtime, executable and captured inputs.

The workspace `runs/CURRENT.json` points to current reports. A cumulative
report across repair rounds preserves each runtime boundary; it does not claim
that all entries have run at the latest commit. Strict GPU equality and complete
sparse-diff equality to a previously accepted residual are separate outcomes.
The latter does not establish GPU bit-exactness or prove rounding as the cause.
Running/queued entries remain unverified.

Static opcode registration is only a diagnostic. The remaining generic
conversion branches in the PyTorch copy kernel are not exercised by its retained
input type, and assert call/trap paths are not successful computation paths.
Their presence in a cubin does not invalidate a completed numerical replay;
neither does that replay qualify those unexecuted forms. They remain unsupported,
not silently emulated as no-ops.

## Before submission

1. Finish the extracted real-operator queue and review every new discrepancy.
2. Fold feature-owned follow-ups into their functional units without importing
   the private research configurations or unrelated instruction extensions.
3. Qualify independent common fixes against public main and document dependencies.
4. Re-run each final branch's own complete-launch gate and ordinary PTX isolation
   checks at its final head. Local CI is not hosted CI or hardware calibration.

Rubin/SM107, research placement, ideal-memory experiments, and unqualified ISA
modifiers remain separate. Production defaults stay opt-in for SASS. The stack
inherits the reviewed gather4 prerequisite's B200 TMA service interval change;
there is no claim that every configuration file in the entire stack is unchanged.
