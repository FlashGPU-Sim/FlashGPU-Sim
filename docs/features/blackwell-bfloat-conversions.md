# BF16 conversion launch qualification

This private feature extracts BF16 conversion support needed by retained
DeepInfer projection PTX. It is separate from the TCGen05 compute implementation.
Both frontends use `flash/bfloat_conversion.h` for rounding, NaN canonicalization,
ReLU and saturation; the adapters only unpack operands and write registers.

## Qualified subset

- PTX `cvt.{rn,rz}{.relu}{.satfinite}.{bf16,bf16x2}.f32` and
  `cvt.f32.bf16`. Packed source A occupies the high half, B the low half.
- Native scalar `F2F.BF16.F32[.RZ]` and `F2FP` BF16 `PACK_AB`/`MERGE_C`,
  including ReLU, satfinite and their combination, RN/RZ.
- MERGE_C preserves the low half of C in the high destination half;
  source/destination aliasing and predication are tested separately.

The [PTX conversion specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
is the semantic reference. This is not all BF16 conversion forms: integer/F64,
RP/RM, stochastic rounding and FP8/FP4 packed conversions remain outside this
change. Unsupported BF16 conversion forms fail explicitly. No timing calibration
or change to production configuration is included.

## Validation

`CI_JOB=sm100-bfloat bash tests/ci/run_ci_tests.sh` runs four component tests
and the same CUDA 13.3 executable through PTX and execution-driven SASS:
four numerical tests, eight kernel launches per frontend. The independent
bit-pattern oracle covers signed zero, ties, subnormals, overflow, infinities
and NaN, crossed with RN/RZ and ReLU/satfinite combinations. Component tests
also enumerate all 65,536 exactly representable BF16 encodings.

The real projection was previously blocked at parsing `cvt.rn.bf16x2.f32`.
It now advances past parsing and exposes a separate typeless TRAP predecode
assertion. This document does not claim full DeepInfer acceptance.
