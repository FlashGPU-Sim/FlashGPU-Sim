# Negated PTX CTA reduction predicates

`bar.red.popc`, `bar.red.and` and `bar.red.or` ignored the operand inversion
flag on `!p`. Convert the stored predicate to a boolean, then apply its operand
negation in both implicit-count and explicit-count paths. Arrival, publication,
and timing behavior are unchanged.

This branch starts directly from public `flash` at `b3378cb9`, with no TCGen05
or private SASS prerequisites. The test-only commit is `3654d264`. Two numerical
launch fixtures exercise all/none/partial predicates, repeated generations,
independent 64-thread groups, and implicit CTA-wide reductions. Both tests fail
on the public baseline and pass after the fix with the same executable. Per-test
cycle counts remain 5482 and 5367 before and after. Qualification results are kept in workspace
`runs/public-cta-reduction-20261125/`.

The fixtures are included in the existing SM90/SM100/SM120 integration jobs;
local qualification uses SM100_B200 and CUDA 13.3. This does not claim hosted CI
or a fresh hardware measurement.

```sh
./tests/run_tests.py run --arch sm100 --group integration --gtest-filter 'CtaReduction.*'
```
