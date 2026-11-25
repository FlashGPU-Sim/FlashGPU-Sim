# Honor negated predicate operands in PTX CTA reductions

`bar.red.{popc,and,or}` previously read the stored predicate without applying
its operand inversion flag. For `bar.red.popc.u32 ..., !p`, 6 true lanes out of
64 therefore returned 6 instead of 58. Both the implicit-count and explicit-
count paths were affected. Apply inversion after converting the simulator's
predicate encoding to a boolean; retain existing arrival, publication and
scheduling behavior.

`tests/src/barrier/cta_reduction_extended_test.cu` checks all three operations,
all/none/partial predicates, two independent 64-thread groups, repeated barrier
generations, and implicit-count reductions. CUDA 13.3 embeds PTX and native code;
PTX source retains the negated operands. The two negative-predicate tests fail
on the prior runtime. Native arithmetic is independent of this fix.

This is a common PTX correctness fix, not a TMEM feature. Qualification against
public main and a separate public PR remain pending. No timing configuration
is changed.
