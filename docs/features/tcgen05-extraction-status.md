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
