# Floating-point tensor-map loads

Ordinary tiled TMA loads now apply the same floating conversion helper as
`tile::gather4`. The common helper supports FLOAT32_FTZ, TFLOAT32 and
TFLOAT32_FTZ, including signed subnormal flushing, round-to-nearest-even TF32
conversion and the floating out-of-bounds fill pattern. It preserves a
four-byte shared-memory element. PTX transfers and native SASS UTMALDG both
call this helper; neither frontend has a separate conversion implementation.

The shared SM100 Driver descriptor decoder accepts the retained 2D and 3D
TF32 controls (`0x276390`, `0x2763a0`). These opaque encodings are observed
CUDA Driver payloads, not a portable descriptor ABI. Unknown encodings,
packed payloads and invalid tails remain rejected. Explicit simulator Driver
descriptors support floating loads of ranks 1 through 5. TF32 tensor stores
remain unsupported rather than silently copying unconverted values.

`sm100-tma-float-loads` runs the Driver descriptor host tests and the same
compiled CUDA executable through PTX and SASS. Six numerical tests issue
17 launches per frontend, covering ranks 1–5, swizzling, nonzero coordinates,
negative and upper-bound coordinates, FTZ, NaN/Inf, rounding ties and retained
opaque TF32 descriptors. The numerical oracle uses literal IEEE bit patterns
and does not call simulator conversion helpers.

This extraction reuses the existing gather4 conversion convention. It adds no
new bandwidth, latency or hardware calibration claim. Passing these small
fixtures is not a substitute for full DeepInfer workload acceptance.
