# CTA Replacement Microbenchmark

This benchmark records the first and last `clock64` timestamp, `%smid`, and
`blockIdx.x` for every physical CTA. Its default 231424-byte dynamic
shared-memory allocation matches the FA4 B200 kernels and limits them to one
resident CTA per SM.

Run two simulator points from the repository root:

```bash
make -C tests/src/microbench/cta CUDA_INSTALL_PATH=/usr/local/cuda-13.2 \
  SIM_LATENCY=0 sim
make -C tests/src/microbench/cta CUDA_INSTALL_PATH=/usr/local/cuda-13.2 \
  SIM_LATENCY=1200 sim
```

On a B200, build an `sm_100a` binary and run the same grid natively:

```bash
make -C tests/src/microbench/cta CUDA_INSTALL_PATH=/path/to/cuda-13 run-hw
```

Use at least two waves (`296` blocks for 148 SMs). Compare the paired
per-SM gap distribution, not host wall time:

```text
gap = next CTA clock64 start - previous CTA clock64 end
```

Sweep `BLOCKS`, `THREADS`, `PAYLOAD_CYCLES`, and `DYNAMIC_SMEM` to distinguish
initial placement, replacement, resource occupancy, and payload-length
effects. The CSV preserves the complete block-to-SM placement and timing
records for topology analysis.
