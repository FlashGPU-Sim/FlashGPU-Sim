# Shared ternary logic

Both frontends use `flash/ternary_logic.h` for 32-bit LUT evaluation and the
optional result-predicate operation. PTX now accepts and executes:

- `lop3.b32 d, a, b, c, immLut`.
- `lop3.{and,or}.b32 d|p, a, b, c, immLut, q`.
- The `_` destination sink with either predicate operation.

All 256 immediate LUTs are supported. Native `LOP3.LUT`, `LOP3.LUT.PAND` and
`ULOP3.LUT` use the same evaluator; the ordinary LUT form now accepts an input
predicate and combines it with the nonzero result using OR. This form is emitted
by CUDA 13.3 for PTX `lop3.or`. Predicate input/output aliasing reads the input
before updating the output. PTX's inverted predicate representation stays in the
adapter.

PTX uses the existing integer-logic latency/initiation settings. Native timing
is unchanged. No new hardware timing calibration is claimed.

The `sm100-ternary-logic` CI job checks all truth tables, native aliases,
inactive lanes, uniform operands, guards and invalid LUT rejection. Five
compiled kernel launches per frontend cover every LUT, both Boolean predicate
operations and both destination sinks, checking against an independent host
reference. The same CUDA executable runs through PTX and SASS.

Reference: [NVIDIA PTX lop3](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-lop3).
