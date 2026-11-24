# Scalar PTX b128 registers (private draft)

The base parser accepts the legacy PTXPlus `.bb128` four-register form. This
extension also accepts PTX `.b128` scalar registers and preserves all 128 bits
on assignment and memory load/store. Vector mov packing accepts four 32-bit
registers or two 64-bit registers. Existing legacy four-register behavior is
retained.

This type-form support is independent of CLC. CLC consumes a scalar opaque
128-bit handle, which exposed the missing parser/assignment path. The complete
CUDA fixture `tests/src/clc/b128_test.cu` checks global/shared roundtrip and both
packing widths with nonzero high halves; it is grouped beside its CLC consumer
but does not execute any CLC instruction. PTX and native runs use the same
executable. No arithmetic on 128-bit scalar values is introduced.
