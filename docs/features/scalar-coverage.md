# Experimental native scalar coverage

Base: SASS frontend `59734bc5`. Private preparation branch:
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

Validation: 587 component tests pass (514 base + 73 additional functional and
timing cases). Includes an independent enumeration-based FP8 oracle, NaN/
infinity/subnormal, directed rounding, predicate and register-pair checks.
Local CUDA 13.3 component invocation excludes the pre-existing
`trace_validation_test.cc` legacy cuCtxCreate signature; no test assertion or
performance gate was relaxed. Release build is required separately.

Remaining scope: native direct-address memory forms, warp collectives, and
architecture-specific tensor/cluster instructions are separate changes.
Nonuniform R2UR execution remains rejected; the private workload-specific
look-ahead exception is deliberately not imported here. SFU functions retain
functional host-math approximations, not bit-exact GPU polynomial models.
