# PTX integer literals

The lexer accepts the optional `U` suffix and all four PTX radices (decimal,
hexadecimal, octal, binary), retaining the full 64-bit bit pattern. Decimal
values above INT64_MAX no longer saturate through `strtoll`. Unary negative
literals use unsigned modulo arithmetic, avoiding host signed overflow.
Magnitudes above UINT64_MAX and malformed octal digits are rejected.

The grammar's existing signed carrier is preserved; instruction/data types
perform the final conversion. This change does not add a constant-expression
evaluator or infer signed expression types from the suffix.

Reference: [NVIDIA PTX integer constants](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-constants).

`sm100-integer-literals` runs parser/predecode tests for all radices, the `U`
suffix, negation, the 2^63/2^64 boundaries and explicit overflow rejection.
This is a common PTX parser fix, with no SASS or timing dependencies.
