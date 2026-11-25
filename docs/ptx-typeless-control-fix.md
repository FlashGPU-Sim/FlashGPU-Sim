# Predecode typeless PTX trap and breakpoint instructions

The opcode classifiers requested a scalar type for `trap` and `brkpt`, although
neither instruction has one. Predecode visits them even in an unexecuted branch,
so otherwise valid kernels could assert before simulation began.

Exclude these two control opcodes from the numeric type classifiers, preserving
`UN_OP` and `OTHER_OP`. Their execution handlers are unchanged: this does not
turn a real trap into a no-op or claim that a trapping workload can finish.

This branch starts directly from public `flash` at `b3378cb9`. Test-only commit
`15cf8c3c` loads a small PTX function containing an unconditional branch around
each control instruction, invokes normal dominator/predecode processing, and
checks the resulting opcode/type metadata. Both fixtures assert on the public
baseline. They are included in the existing SM90/SM100/SM120 unit manifests.
Local evidence is retained under workspace `runs/public-typeless-control-20261125/`;
hosted CI and new public submission remain separate.

With the same frozen test executable, both cases pass after the fix on SM100.
