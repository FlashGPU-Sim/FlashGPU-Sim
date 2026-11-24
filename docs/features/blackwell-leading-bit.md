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

The follow-up feature connects `.shiftamt` in PTX and signed native `FLO` /
`FLO.SH` / `UFLO` / `UFLO.SH` to the same primitive. `CI_JOB=sm100-leading-bit`
adds five numerical launches covering all four PTX types with and without
`.shiftamt`, plus immediate operands, predication and destination/source aliasing.
The native component test directly exercises all eight signed/unsigned,
index/shift, warp/uniform opcode combinations, including source aliasing and a
nonzero first active lane. CUDA lowers 64-bit search to a native sequence;
there is no invented 64-bit FLO opcode.

The basic PTX fix remains a separate commit for independent review. Public-main
cherry-pick qualification and full DeepInfer numerical acceptance are separate
from this focused instruction suite.
