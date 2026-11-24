# Blackwell scalar mixed-precision arithmetic

Scope: `add/sub.f32.{f16,bf16}` and `fma.f32.{f16,bf16}`. The first
source (first two for FMA) contains 16-bit floating-point bits; accumulator
and destination are FP32. Four rounding modes and optional saturation are
supported. Add/sub may omit rounding; FMA requires it. Rubin packed forms
and mixed-precision multiply are outside this scalar extension.

The PTX adapter and native `FHADD[.BF16]` / `FHFMA[.BF16]` decode operands
into the same conversion and FP32 arithmetic helpers. They reuse the existing
`rounded_f32_fma` implementation, including rounding-state restoration and
saturation. Half conversion is extracted unchanged from the native common
helper, so tensor-fragment and conversion clients retain their semantics.
Native low/high halfword selectors, source negation/absolute value, predication
and destination aliasing are handled before entering the shared arithmetic.
The functional handlers now live in `sass/functional/mixed_precision.cc`.

`CI_JOB=sm100-mixed-arithmetic` checks native modifiers/aliases, rounding state
and signed-zero behavior, then six CUDA 13.3 SM100a launches through each
frontend. The kernels exercise 56 distinct PTX modifier forms, finite boundaries,
subnormals, infinities, NaNs, fused underflow/overflow cancellation, and saturation.
The host reference decodes formats independently and rounds bounded exact
long-double sums/products. NaN payload equality is deliberately not required.

Timing reuses the existing FP32 instruction profile. This is functional
qualification, not a new hardware timing calibration. Full DeepInfer acceptance
is tracked separately from these component and launch tests.

Reference: [NVIDIA mixed-precision ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-precision-floating-point-instructions).
