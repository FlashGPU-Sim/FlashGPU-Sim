# Experimental SM100 gather4

Private preparation branch `feature/tma-gather4`, stacked on scalar coverage.
The SM100/SM103 native decoder and launch registration are added for this
feature. This is not a claim of general SM100 native instruction coverage.
SASS remains experimental; no default simulator configuration changes.

PTX `cp.async.bulk.tensor.2d.tile::gather4` and native
`UTMALDG.2D.GATHER4` share the TMA address generator, request queue, memory
service and completion backend. The PTX parser now reaches execution instead
of rejecting this form. Both adapters preserve all five coordinates and
credit all four rows, including zero-filled out-of-bounds rows.

Qualified forms: global to shared, 2D map, boxDim[1]=1, unit strides,
no interleave, no swizzle, raw 1/2/4/8-byte elements and zero OOB fill.
Repeated, permuted and out-of-bounds rows preserve operand order.
The simulator's host tensor-map format is recognized by the native load
path; this does not imply recognition of every NVIDIA driver descriptor ABI.

Not qualified: multicast gather4, scatter4/store, TF32 conversion, interleave,
swizzle, non-unit strides, nonzero OOB fill, arbitrary captured driver maps.
In particular, ordinary PTX and native TMA currently differ in swizzle base
phase handling. Gather4 rejects swizzle rather than inheriting that mismatch.

Tests:

- Address-generator oracle spans six data types, four packet sizes, negative
  and right-edge coordinates, repeated and OOB rows; rejects unsupported maps.
- Native functional test checks byte values, four-row completion and timing
  metadata across all element widths.
- One real CUDA binary, compiled for sm_100a with inline PTX, runs through both
  PTX timing and SASS timing, checking results after two complete launches.
- Host descriptor tests cover raw type sizes, U32/F32 tags and invalid controls.

Build `make -C tests build-sm100-tma`. The standalone `TmaGather4Integration.*`
test can be selected with GoogleTest. Native replay needs the documented SASS
AUTO/TIMING/BINARY/decode tool environment, explicitly set to sm100. For PTX
standalone replay disable SASS-guided PTX reordering in the local test config.

Reference: [NVIDIA PTX ISA 9.5](https://docs.nvidia.com/cuda/parallel-thread-execution/),
`cp.async.bulk.tensor` gather/scatter examples and tensor-map restrictions.
The instruction example's tuple is `{x,y0,y1,y2,y3}`.
