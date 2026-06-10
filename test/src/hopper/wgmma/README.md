# WGMMA Integration Tests

End-to-end tests for Hopper warp-group MMA instructions.

These tests follow the same validation pattern as `test/src/integration/mma`:

- launch a real CUDA kernel with inline PTX `wgmma.mma_async`
- compute the expected result on CPU
- copy accumulator-register results back to host
- compare with Google Test tolerances

## Current Coverage

`cuda_wgmma_f16_test.cc` covers the minimal Hopper WGMMA shape:

- `wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16`
- A operand from registers
- B operand from shared memory through a GMMA descriptor
- 128-thread warpgroup execution
- `wgmma.fence`, `wgmma.commit_group`, and `wgmma.wait_group`

Additional dense data-type tests mirror the current NVIDIA PTX ISA WGMMA
coverage using the smallest documented `m64n8` shape for each type family:

- `cuda_wgmma_bf16_test.cc`: `m64n8k16.f32.bf16.bf16`
- `cuda_wgmma_tf32_test.cc`: `m64n8k8.f32.tf32.tf32`
- `cuda_wgmma_fp8_test.cc`: `m64n8k32.f32.e4m3/e5m2`
- `cuda_wgmma_s8_test.cc`: `m64n8k32.s32.s8/u8`
- `cuda_wgmma_b1_test.cc`: `m64n8k256.s32.b1.b1.and.popc`

## Test Organization

This directory follows the data-type split used by `test/src/integration/mma`:

- `cuda_wgmma_f16_test.cc` contains the current FP16-specific inline PTX
  kernel, runner, and gtest cases.
- `tensor_wgmma_test.cuh` contains shared WGMMA test utilities: the current
  m64n8 shape constants, type dispatch, accumulator mapping,
  shared-memory descriptor helpers, random input generation, CPU references,
  CUDA kernel launchers, and common CUDA run result plumbing.

New data types should use their own `cuda_wgmma_<type>_test.cc` file and reuse
the common helpers where the layout matches. Implementation-side variant
dispatch remains centralized in `src/gpgpu-sim/flash/wgmma/tensor_wgmma.cc`.

The first tests use uniform input tiles, so every accumulator register should
contain the same CPU reference value. This validates the instruction path before
adding tests that depend on the full GMMA accumulator-to-matrix layout.

## Running On Hopper

WGMMA requires an architecture-accelerated Hopper build. From `test/`:

```bash
./run_tests.sh build hopper
./run_tests.sh hopper WgmmaF16M64N8K16IntegrationTest
./run_tests.sh hopper WgmmaF16M64N8K16IntegrationTest.AllOnesTest
```

The `hopper` command builds Hopper workloads with `HOPPER_CUDA_ARCH=sm_90a`
and uses `SM90_H200` as the default simulation config unless `-c/--config` is
specified.
