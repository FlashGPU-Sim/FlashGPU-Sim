# RTX 5090 Instruction Fetch Calibration

This document records what the RTX 5090 instruction-cache experiments support,
what the experimental simulator configuration models, and which hierarchy
parameters remain unknown. The raw Nsight Compute reports are intentionally
ignored; the collector and workload generators are checked in under
`tests/src/microbench/instruction_cache/`.

## Collection method

Accel-Sim 2.0 collects hardware data in
`util/hw_stats/run_hw.py` by querying available Nsight Compute metrics, using
application replay with `--cache-control none` where warm application state is
required, exporting raw CSV from the `.ncu-rep`, and retaining the original
report. Its CUPTI PM Sampling tools separately enumerate PM-sampling-compatible
metrics instead of assuming every aggregate NCU metric is sampleable.

The RTX 5090 collector follows the same constraints:

1. Query metric support before constructing the NCU command.
2. Use kernel replay with `--cache-control all` for cold measurements.
3. Use application replay, an explicit warm-up launch, and
   `--cache-control none` for warm measurements.
4. Store raw NCU output alongside parsed CSV and verify
   `requests = hits + misses` for ICC and GCC totals.
5. Measure cubin ELF text size with `cuobjdump`; source-level repetition counts
   are not treated as instruction bytes.

On the installed RTX 5090 toolchain, ICC and GCC counters are available as
aggregate replay metrics but not through PM Sampling. A GPUVision-style time
series therefore cannot currently replace controlled cold/warm experiments.

## Current observations

These values were collected from real `sm_120a` cubins on August 27, 2026.
They are effective transitions for the generated sequential layout, not claims
about undocumented physical organization.

| Observation | Hardware result | Interpretation |
|---|---:|---|
| Front-end/ICC request transition | 5-6 KiB text | A smaller front-end code structure exists before ICC traffic increases |
| ICC miss transition | 64,128 to 65,152 bytes | Effective per-SM ICC capacity is approximately 64 KiB |
| GCC sustained-thrash transition | 131,712 to 148,096 bytes | Effective GCC capacity for this layout is approximately 128-148 KiB |
| First ICC tag-unavailable event | 12 independent warp PC streams | At most an initial bound on concurrent miss/fill resources |
| Skipped-body GCC-to-XBAR plateau | approximately 58-64 KiB text | Hardware performs code read-ahead or preload |

Representative cold counts make the transitions visible:

| Text bytes | ICC misses | GCC requests | GCC misses |
|---:|---:|---:|---:|
| 64,128 | 8 | 509 | 3 |
| 65,152 | 1,048 | 5,713 | 7 |
| 82,560 | 1,706 | 133,541 | 138 |
| 131,712 | 6,100 | 132,609 | 14,044 |
| 148,096 | 56,575 | 132,601 | 128,849 |

Warm replay removes cold GCC misses while the footprint fits, while footprints
above the transition continue to miss. The skipped-body plateau proves that
traffic is fetched ahead of dynamic execution, but it does not reveal the
prefetch distance: aggregate traffic conflates preload granularity, cache-line
size, request merging, and termination at code-section boundaries.

The independent-PC sweep reports zero ICC tag-unavailable events through 11
streams, 6 at 12 streams, and increasing pressure thereafter. This motivates a
12-entry experimental MSHR setting, but does not yet identify the exact number
of physical entries because unavailable events can include retries and other
tag resources.

## Simulator status

`configs/SM120_RTX5090_ICACHE` is deliberately experimental. It changes the
default perfect instruction cache to:

- a 64 KiB per-SM read-only instruction cache with 128-byte simulator lines;
- 12 MSHR entries, based on the independent-PC pressure bound;
- one bounded stream with an eight-line lookahead window and one issue attempt
  per SM cycle.

The stream-buffer depth is a real finite ahead window. Demand progress slides
the window, replacement changes a generation token, and late fills from old or
canceled generations cannot mutate current stream state. Demand reservation
failure also suppresses speculative issue for that cycle.

The eight-line depth and 128-byte line are model parameters, not reverse-
engineered RTX 5090 values. The current implementation also does **not** add a
shared GCC between ICC and L2. Modeling ICC misses as direct lower-memory
requests is incomplete, so this configuration must not replace
`SM120_RTX5090` for cycle validation yet.

## Remaining probes

The next hierarchy change should wait for three targeted measurements:

1. **Set conflicts and associativity:** generate equal-size cubins whose hot
   functions vary only in virtual-address stride and linker placement. Compare
   warm ICC/GCC misses while holding dynamic control flow fixed.
2. **GCC sharing topology:** run cooperating blocks pinned to increasing SM
   sets, warm disjoint code footprints, and detect when one group evicts the
   other. This separates per-SM ICC from GPC-local or chip-global GCC sharing.
3. **Hit/miss latency:** use dependent indirect branches among noinline
   functions and `clock64` around serialized traversal. Sweep warm ICC hit,
   warm ICC miss/GCC hit, and cold GCC miss cases; subtract a same-layout
   control traversal.

Only after those probes identify sharing, capacity/associativity, and latency
should the simulator add a GPC-scoped GCC and route ICC misses through it.

## Commands

```bash
tests/run_tests.py build --arch sm120 --group microbench --profile instruction-cache
make -C tests/src/microbench/instruction_cache collect-cold
make -C tests/src/microbench/instruction_cache collect-warm
make -C tests/src/microbench/instruction_cache collect-skip
make -C tests/src/microbench/instruction_cache collect-parallel
```

The parser can be checked without NCU access:

```bash
python3 -m unittest tests/src/microbench/instruction_cache/test_collect_ncu.py
```
