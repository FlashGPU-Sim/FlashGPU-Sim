# Shared BF16 arithmetic

`bfloat_arithmetic.h` owns round-to-nearest BF16 add/sub, multiply and FMA.
The PTX adapter decodes scalar or packed operands and calls those functions;
SASS HADD2/HMUL2/HFMA2 retain their operand decoding and call the same functions.
Existing BF16 conversions remain in `bfloat_conversion.h`; no second conversion
implementation or frontend-specific arithmetic is introduced.

The current qualified scope is the RN forms of `add`, `sub`, `mul` and `fma`
for `.bf16` and `.bf16x2`, following the
[PTX half-precision instruction definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions).
PTX rejects unqualified arithmetic modifiers. FMA `.relu` / `.oob`, BF16
min/max and mixed FP32/BF16 arithmetic remain separate follow-up work.

Arithmetic restores the host thread's floating rounding mode. FMA uses the
retained midpoint-residual correction so a tiny addend cannot disappear before
BF16 rounding. Multiply also corrects FP32-underflow midpoint double rounding.
These are functional semantics, not a new measured BF16 timing calibration.

`CI_JOB=sm100-bfloat-arithmetic` combines component tests with scalar and packed
numerical launches from one binary through PTX and SASS. The independent host
reference rounds a long-double mathematical result directly to the BF16 grid.
Inputs cover finite values, separate packed halves, aliasing, signed zeros,
subnormals, overflow, infinities and NaNs. The component suite separately checks
an FMA midpoint with an addend too small for a long-double oracle to retain.

This migration addresses the retained C4 failure at `mul.bf16x2`. Passing the
instruction fixtures does not by itself qualify complete C4 or DeepInfer;
the corresponding frozen-runtime workload reports are recorded separately.

The literal-multiplier FMA fixture also reproduces an existing native discrepancy:
the old immediate path rounded to FP32 and then BF16 without the register path's
midpoint correction. Sharing the helper fixes the tiny-addend case for both
forms. The original two arithmetic fixtures retain identical native cycles and
results; the frozen old/new literal test records the numerical correction.
