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

The first tests use uniform input tiles, so every accumulator register should
contain the same CPU reference value. This validates the instruction path before
adding tests that depend on the full GMMA accumulator-to-matrix layout.

## Running On Hopper

WGMMA requires an architecture-accelerated Hopper build. From `test/`:

```bash
CUDA_ARCH=sm_90a ./run_tests.sh build
CUDA_ARCH=sm_90a ./run_tests.sh test WgmmaF16M64N8K16
```

On non-Hopper GPUs these tests skip. If the binary was built for an incompatible
architecture, the tests also skip with a message to rebuild using
`CUDA_ARCH=sm_90a`.
