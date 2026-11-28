# Dense FP8 mask qualification and public-fix integration

This follow-up belongs to `feature/blackwell-timing-qualification`; it is not
a claim that the individual feature-owner branches have been refreshed.
No public PR is opened by this work.

## Functional change

Ordinary dense `tcgen05.mma.kind::f8f6f4` now accepts the four-word
`disable-output-lane` vector for the existing qualified CTA1, M128, K32,
shared-A E4M3/E5M2 subset. Each set bit preserves its physical TMEM lane,
including when `enable-input-d` is false. Masking does not reduce modeled MMA
work or silently convert the instruction into a sparse operation.

PTX and SASS feed the existing common accumulator writeback. The native
`UTCQMMA` adapter distinguishes a bare mask register group from the wrapped
`tmem[...]` scale-address pair of block-scaled MMA. Dependency projection
includes all four mask registers, while a scale-address pair remains two.
FP8 descriptor qualification is also shared instead of being duplicated in
the two adapters. Truncated register spans are rejected before side effects.

This follows the [PTX output-lane mask semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma).
It does **not** implement the different 64-bit WS B zero-column descriptor.
That descriptor is covered by the later [WS follow-up](blackwell-ws-mask-qualification.md).
Paired FP8, TMEM-A FP8, other unqualified formats and extra scale-input-d
forms remain outside the qualified subset. No Rubin feature or timing
constant is added.

## Reproduction and validation

Evidence is `runs/tcgen-fp8-masks-20261125` at the workspace root. The same
frozen executable fails on the previous runtime in both frontends: PTX
rejects the extra operand and native execution mistakes it for a block-scale
form. New tests check all-enabled, all-disabled and mixed masks; both FP8
formats; 32/64/128-byte shared swizzles; seeded accumulators; and two MMAs
with input-D disabled then enabled. Host expected products use logical input
matrices independently of simulator layout helpers.

Completed validation: 103 host tests pass. For each frontend, 41 legacy
tests / 85 launches pass with exact prior cycles; two added FP8 mask tests /
six launches pass. The separate six-launch shuffle suite passes in both
frontends at one and four threads with identical cycles. Legacy TCGen05 gates
use the FP8-only frozen runtime; final-source host, FP8 and shuffle checks
also include the separately qualified public fix. These are explicitly
separate runtime checkpoints, not a latest-head full DeepInfer sweep.

`collect.py` checks exact binary/runtime hashes, host results, all retained
MMA/pair/WS launch counts, unchanged legacy cycles, and the separate shuffle
integration. See `report.json` for completed counts and runtime boundaries.
The compiled local CI gate now expects 31 MMA tests / 68 launches per
frontend. This is focused simulator qualification, not a new GPU collection,
full DeepInfer sweep, or hosted-CI result.

## Public bug ownership

The feature integration branch still lacked the public candidate's shuffle
snapshot correction (`731cca2b`, regression fixture `ac233e2d`). It now skips
predicated-off source lanes before reading their registers. The test is
adapted to this branch's existing test layout and restores captured stdout
so CI can count every launch. Both frontends and one/four simulation threads
are checked; the old five shuffle cases retain their cycles.

This fix remains part of the independent public execution-semantics bundle;
it is not a new TCGen05 dependency or a fifth public PR. Dense FP8 masks and
their dependency projection belong to the MMA feature. The four prepared
public bundles remain independently based on public `b3378cb9`.
