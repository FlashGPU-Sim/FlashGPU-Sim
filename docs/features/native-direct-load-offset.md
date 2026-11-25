# Native direct global-load uniform offsets

The retained routing SGEMM emits `LDG.E.LTC128B R150[R64.64+UR10]`.
Previously normalization rejected this as an unsupported address before launch.
The adapter now distinguishes the lane's 64-bit base from one 32-bit uniform
byte offset, optionally followed by an immediate displacement. It preserves the
existing implicit register-pair rule for a base without the `.64` suffix.

The common native address evaluator and dependency projection consume that
normalized operand. UR10 remains narrow: UR11 must neither enter the address
nor become a false dependency. Register-pair bases, implicit high words and
qualifiers are not replaced by a second memory implementation.

Component tests cover high addresses, carry from a large unsigned offset,
per-lane addresses and guards, exact dependency width, and rejection of missing
bases, multiple bases/offsets, negated offsets and out-of-range uniform registers.
Full original routing replay is required separately. This is native operand
coverage; ordinary PTX scheduling and memory configuration are unchanged.
