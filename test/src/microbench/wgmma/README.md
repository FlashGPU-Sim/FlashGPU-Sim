# WGMMA Microbenchmarks

The WGMMA microbenchmarks are gtest binaries built by the top-level
`test/Makefile`.

- `wgmma_async_latency_bench.cc`: broad WGMMA shape/group sweeps, FA3-like
  timing probes, and mixed WGMMA/softmax probes.
- `wgmma_n16_chain_bench.cc`: focused `m64n16k16` chain/count sweeps. This is
  intentionally built by the explicit `make wgmma-n16-chain` target because it
  requires SM90a code generation.

The old standalone mixed-softmax file was removed because the same probes now
live in `wgmma_async_latency_bench.cc`.

Build examples from `test/`:

```bash
make bench
make wgmma-n16-chain HOPPER_CUDA_ARCH=sm_90a CUDA_ARCH=sm_90a
```
