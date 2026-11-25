# PTX shuffle source snapshot (private candidate)

PTX shuffle previously read and wrote each lane sequentially. When the output
aliases the input, an UP or BFLY transfer can read a value already overwritten
by a lower lane. This is a functional bug independent of SASS or TCGen05.

Capture the active warp's source values before the first participating lane
writes. The scratch image belongs to the executing core and is reset for every
warp instruction; it is not global, shared across host workers, or retained in
the timing pipeline. Each lane still writes its own result, preserving the
existing predicate evaluation, PC updates and instruction accounting.

The compiled `WarpShuffle` fixture covers all four directions, source/output
aliasing, multiple segment widths and offsets, result predicates, and a
predicated repeated instruction whose result predicate aliases its guard.
The same binary and temporary configuration must be used for before/after
numerical and cycle comparisons. `check_ptx_shuffle.py` qualifies only PTX;
native direction support is a separate feature. Public-main qualification
remains required before proposing a standalone public fix.
