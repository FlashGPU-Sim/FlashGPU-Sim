# Leading non-sign bit search

The common `flash/bit_scan.h` primitive implements the bit-index result used by
PTX `bfind` and native `FLO.U32`. It masks 32-bit operands before sign handling,
uses a 64-bit mask for wide values, and returns `0xffffffff` for zero (or signed
all-ones). The previous PTX descending unsigned loop did not terminate for
missing bits, and its `1 << bit` expression was invalid for high/wide bits.

`CI_JOB=sm100-leading-bit-base` checks every bit and signed complement in the
host primitive, then runs the same CUDA binary through PTX and SASS. Two launches
cover unsigned 32/64-bit powers, complements, zero, and word boundaries with
runtime inputs and an independent repeated-division oracle. This is functional
qualification, not new timing calibration.

Semantics: [NVIDIA PTX bfind](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-bfind).
