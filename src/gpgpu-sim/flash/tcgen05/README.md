# TCGen05 Support

This directory contains the functional TCGen05/TMEM support used by the
Blackwell FA4 bring-up path.

Current scope:

- `cta_group::1` only.
- Dense `.kind::f16` MMA with f16/bf16 inputs and f32 accumulator output.
- Dense shared/shared `.kind::mxf4.block_scale.block32` MMA with E2M1
  inputs, UE8M0 scale factors, K=64, and f32 accumulator output. Both the CUDA
  12.8 `.scale_vec::2X` spelling and CUDA 12.9+ `.block32` spelling parse.
- TMEM allocation, deallocation, register load/store, and accumulator storage.
- `tcgen05.cp.32x128b.warpx4` bit transpose and four-subpartition broadcast,
  including the compact 8-bit scale-factor layout used by CUTLASS.
- Parser coverage for the common TCGen05 instruction family needed during FA4
  bring-up.
- Optional inline-PTX surface validation through
  `tests/dev/tcgen05/run_phase1_smoke.sh --check-inline`, which requires a
  Blackwell-capable CUDA toolchain.

Not calibrated yet:

- TCGen05 MMA issue throughput and completion latency.
- Separate MXFP4 versus f16 TCGen05 throughput calibration.
- TMEM load/store bandwidth, banking, and queueing.
- `tcgen05.cp`/`shift`/`commit` timing.

Unsupported by design for the current FA4 forward target:

- `cta_group::2`.
- Sparse and weight-stationary MMA variants.
- NVFP4 (`mxf4nvf4.block16`), MXF8/F6/F4, and other narrow-precision kinds
  beyond the strict MXFP4 path above.
- Exact shared-memory descriptor swizzles; operand tiles are still linearized
  in the functional bring-up model.
