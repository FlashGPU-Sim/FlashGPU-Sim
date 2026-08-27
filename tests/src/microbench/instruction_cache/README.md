# RTX 5090 Instruction-Cache Calibration

This directory contains a hardware-only microbenchmark for the Blackwell
instruction hierarchy. It varies static SASS text size while keeping the
dynamic instruction count approximately constant, then collects aggregate ICC,
GCC, and GCC-to-L2 counters with Nsight Compute.

The repeated inline PTX `bar.warp.sync` instruction lowers to a stable sequence
of SASS NOPs on SM120. Always inspect the generated cubin after changing CUDA
toolkit versions. The collector records the ELF text size reported by
`cuobjdump`; `FOOTPRINT_STEPS` is not treated as a byte count.

## Build and smoke test

```bash
make all test check
```

`make sweep` builds the full set of footprints around the currently observed
ICC and GCC transitions. Generated binaries are stored under
`tests/build/bin/microbench/instruction_cache/`.

## Profile

```bash
make collect-cold
make collect-warm
make collect-skip
make collect-parallel
```

- `cold` uses kernel replay and flushes caches between replay passes.
- `warm` uses application replay without cache flushing. The program launches
  the same kernel once before the measured launch.
- `skip` sets the loop trip count to zero. Comparing its GCC-to-L2 traffic
  against static text size exposes front-end read-ahead or code preload that is
  not caused by execution of the body.
- `parallel` runs 1--32 warps at distinct device-function PCs on one SM. The
  first nonzero `sm__icc_requests_lookup_miss_tag_unavailable` point bounds ICC
  miss/fill resources independently of sequential capacity.

Reports and raw NCU logs are written below
`tests/run/microbench/instruction_cache/`. They are intentionally ignored by
Git. Override `NCU`, `CUDA_INSTALL_PATH`, `ARCH`, or `PTX_PROFILE` from Make as
needed.

## Interpretation constraints

- `sm__icc_requests` is reported in request cycles, not literal cache lines.
- Capacity transitions are effective capacities for this sequential mapping.
  Conflict-layout probes are required before assigning physical capacity or
  associativity.
- A plateau in the skipped-body experiment proves read-ahead/preload traffic,
  but does not by itself identify a prefetch distance.
- Current RTX 5090 NCU releases expose ICC/GCC as aggregate replay metrics, not
  as PM Sampling metrics. This tool therefore cannot produce a GPUVision time
  series for these units.
