# FP16/BF16 unary sign operations

PTX `abs` and `neg` support scalar and packed FP16/BF16. FP16 also accepts
`.ftz`, flushing subnormal inputs/results to signed zero. BF16 `.ftz` is
rejected. A common bit-level helper changes sign without a float conversion;
NaN payload/sign are unspecified by PTX and tests check NaN classification.

The native FP16/BF16 operand modifiers use the same sign helper. The compiler's
FP16 FTZ lowering uses `HADD2.FTZ`; it now honors half-word replication and
flushes both inputs and results. Existing arithmetic paths remain shared.

Two numerical launch tests exercise all twelve scalar/packed/operator/type/FTZ
combinations, aliased destinations, independent packed halves, signed zeros,
subnormals, finite values, infinities and NaNs through the same PTX/SASS binary.
The parent BF16 and mixed-arithmetic suites provide a cycle regression check.
No new timing calibration is included.

Reference: [NVIDIA half-precision abs/neg](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs).
