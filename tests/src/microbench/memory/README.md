# Memory microbenchmarks

This directory contains standalone CUDA microbenchmarks for global-load
bandwidth, HBM streaming bandwidth, L2 latency, L2/HBM interleaving, and L2
partition behavior. They are built separately from the GTest integration
suite and are not run by the normal integration target.

## HBM streaming bandwidth

`hbm_bw_bench.cu` drives ordinary cache-global vector loads through the full
global-memory path. Each load reads 16 bytes with `ld.global.cg.v4.u32` and all
four returned words contribute to a per-thread checksum. The configurable
independent chains provide memory-level parallelism without introducing a
shared-memory producer-consumer pipeline.

Build the B200 target with CUDA 12.8:

```bash
make ARCH=sm_100a PTX_PROFILE=compute_100a hbm-bw
```

Run the preserved 256 MiB calibration point:

```bash
make ARCH=sm_100a PTX_PROFILE=compute_100a run-hbm-bw
```

The equivalent executable command is:

```bash
hbm_bw_bench_cuda128_sm_100a \
  --mib 256 \
  --blocks-per-sm 4 \
  --threads 256 \
  --chains 8 \
  --passes 1 \
  --warmup 0 \
  --iterations 1
```

On hardware, the CUDA-event fields in the program output measure kernel time.
Under FlashGPU-Sim, those fields follow host wall time. Simulator bandwidth
must therefore be derived from simulated cycles or DRAM service statistics.
The retained checksum, logical byte count, and 32-byte sector count remain the
functional oracle in both environments.

### Reference B200 simulation

The preserved command was run against simulator source `c8666a3d` with the
B200 normal clock tuple `1930:1930:1964:3996` MHz. The 256 MiB launch completed
in 84,794 core cycles with a passing checksum and exactly 8,388,608 global-read
32-byte sectors, all of which missed in L2.

The complete launch averaged 6.109871 TB/s. Removing the configured fixed
5,000-core-cycle launch delay gave a 6.492724 TB/s kernel-body average. The
central service window, measured between 10% and 90% of cumulative DRAM issues,
delivered 7.058447 TB/s. That central-window value describes sustained DRAM
service after startup traffic has formed and before the final drain; it is a
different scope from the complete-launch average.
