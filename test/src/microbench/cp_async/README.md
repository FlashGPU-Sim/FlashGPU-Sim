# cp.async Microbenchmarks

This directory keeps two standalone CUDA calibration binaries:

- `cp_async_ptx_bench.cu`: canonical low-level benchmark. It uses inline PTX
  loops to isolate `cp.async` issue, commit, empty wait, delayed wait, complete,
  and bandwidth-style cases.
- `cp_async_latency_bench.cu`: higher-level chain benchmark for simulator smoke
  tests and FA2-like staged copy patterns.

Build from this directory:

```bash
make all ptx-bench
```

Or from `test/`:

```bash
make cp-async-bench
```

Generated binaries are placed under `test/build/bin/microbench/cp_async/`.
Simulator workdirs created by `gem5-*` targets go under
`test/run/microbench/cp_async/`.

Useful overrides:

```bash
make ARCH=sm_90a PTX_PROFILE=compute_90a all ptx-bench
make ARCH=sm_120a PTX_PROFILE=compute_120a all ptx-bench
```

Keep final reports outside the source tree.
