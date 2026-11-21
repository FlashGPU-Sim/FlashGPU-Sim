# Experimental native scalar coverage

Base: public `01bdef86` (merged experimental SASS frontend). Private preparation branch:
`feature/sass-scalar-coverage`. No public PR or default config change.

This change adds integer comparison/shift/bit-reversal forms, FP32 pair
arithmetic, BF16 arithmetic, low-precision pack/unpack, and additional FP32
rounding/comparison/SFU forms observed in the retained Blackwell workloads.
The functional and timing adapters are updated together: pair destinations,
source selectors, uniform register sources and FP/integer pipeline classes
must agree. The latest public barrier and operand fixes remain in the base.

PTX reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
Sections: integer arithmetic, floating-point arithmetic, comparison/selection,
and type conversion. The manual specifies PTX semantics, not native SASS
encoding: native selectors and packing order also require compiler/hardware
qualification. The private source test oracles are retained; tests running
under the SM120 common registration are semantic tests, not a claim that
ptxas emits every added spelling for SM120.

Modifier coverage added in this revision:

- FP32 FADD/FMUL/FFMA and uniform counterparts: implicit RN, RM, RP, RZ,
  each with and without FTZ and SAT. PTX add/mul/fma use the same rounded
  FP32 arithmetic helper as SASS; PTX multiply no longer asserts on RM/RP.
- FSETP/UFSETP and FSET.BF: ordered and unordered comparisons, NUM/NAN,
  AND/OR/XOR, FTZ, and both predicate destinations for predicate forms.
- ISETP/UISETP: signed/U32 comparisons, AND/OR/XOR, EX comparison chaining,
  and both predicate destinations. Existing 64-bit forms are unchanged.

Validation: 665 component tests pass, including modifier-product tests with
NaN/subnormal, saturation, directed rounding, predicate and uniform-register
checks. The real CUDA `ScalarModifiers.*` fixture covers 48 combinations of
PTX FP32 add/multiply/FMA in one binary and passes both PTX and native SASS
replay. Host SFU approximations are intentionally not bit-exact GPU models.
Local CUDA 13.3 component invocation excludes the pre-existing
`trace_validation_test.cc` legacy cuCtxCreate signature; no test assertion or
performance gate was relaxed. Evidence: workspace
`runs/feature-expansion-20261121`, including `scalar-cuda-r2/report.json`.

Remaining scope: native direct-address memory forms, warp collectives, and
architecture-specific tensor/cluster instructions are separate changes.
Nonuniform R2UR execution remains rejected; the private workload-specific
look-ahead exception is deliberately not imported here. SFU functions retain
functional host-math approximations, not bit-exact GPU polynomial models.
