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

These values were collected from real `sm_120a` cubins on August 27-28, 2026.
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
- a cache-address scale of two, mapping eight-byte PTX PC slots onto Blackwell's
  sixteen-byte SASS address spacing without changing functional PCs;
- one bounded stream with a four-line lookahead window and one issue attempt
  per SM cycle.

The stream-buffer depth is a real finite ahead window. Demand progress slides
the window, replacement changes a generation token, and late fills from old or
canceled generations cannot mutate current stream state. Demand reservation
failure also suppresses speculative issue for that cycle.

The four-line depth, address scale, and 128-byte line are model parameters, not
reverse-engineered RTX 5090 values. Address scale two corrects the obvious
eight-byte PTX versus sixteen-byte SASS slot mismatch, but variable PTX-to-SASS
expansion still requires a real address map. The current implementation also
does **not** add a shared GCC or hardware-like code preload between ICC and L2.
Modeling ICC misses as direct lower-memory requests is incomplete, so this
configuration must not replace `SM120_RTX5090` for general cycle validation.

### Short-kernel correlation

Two frozen cubins keep the validation target in the approximately 10 us range.
They were measured cold with NCU base clock control and replayed from the same
cubin/PTX artifacts. Full counters and cubin hashes are in
`tests/ci/perf/instruction_cache_reference.csv`.

| Workload | RTX 5090 cycles | Simulator cycles | Error | L1I reservation failures |
|---|---:|---:|---:|---:|
| GEMM M512 N16 K512 | 24,600.54 | 24,385 | -0.88% | 0 |
| GPT-2 FA H12 S128 D64 | 25,243.15 | 27,644 | +9.51% | 0 |

These results bound the current short-kernel cycle error but do not validate a
physical ICC/GCC implementation. In particular, the FA run still generates
47,857 simulator L1I misses, while hardware reports 3,889 ICC lookup misses;
the event definitions and hierarchy are not one-to-one.

### Microbenchmark correlation gap

The original footprint probe lowers PTX `bar.warp.sync` to SASS NOP, so its
cycle count is intentionally not compared. A second variant preserves a
one-to-one dependent `mad.lo.u32`/`IMAD` chain. It shows that cycle correlation
is not yet solved even before adding a GCC:

| Steps | RTX 5090 cycles | Perfect-I sim | Experimental-I sim |
|---:|---:|---:|---:|
| 64 | 3,977.42 | 6,103 | 7,683 |
| 1,024 | 7,911.99 | 9,951 | 21,125 |
| 4,096 | 20,425.12 | 22,248 | 64,261 |
| 5,120 | 30,176.85 | 26,347 | 78,607 |

All simulator rows above enable conservative PTX reordering. SASS-guided
reordering remains opt-in because a pure integer kernel currently has no
primary guide anchors. ALU result forwarding makes the dependent IMAD body
track its configured four-cycle latency: the 4,096-step body falls from 32,843
to 16,409 cycles, versus 16,688 cycles measured with `clock64` on hardware.
The remaining perfect-I total-cycle error is dominated by the fixed launch and
front-end overhead for the smaller probes.

The additional experimental-I error comes from fetching each PTX line through
the normal lower-memory path. On hardware, the 66,176-byte probe has 528 ICC
requests but only 10 ICC lookup misses and 16 GCC misses; the simulator has
1,024 L1I misses for the corresponding timing probe. A sensitivity run with
depth 32 and 64 MSHRs removes all late prefetches and lowers the 4,096-step run
to 22,989 cycles, but it also underestimates the short GEMM and FA kernels by
12.14% and 19.13%. Therefore the supported experimental configuration keeps
depth four rather than tuning a rolling stream buffer to hide the missing GCC.

The hardware counters also expose a capacity boundary that a rolling stream
cannot represent: GCC lookup hits reach 509 at 4,096 steps and remain 509 at
5,120 steps, while misses rise from 16 to 144. This is consistent with an
approximately 512-line (64 KiB for 128-byte lines) preload or resident window.
A shared GCC/preload model is required before changing the supported prefetch
depth.

For the final four-line, scale-two configuration, both short kernels complete
without L1I reservation failure. This addresses the observed liveness failure
from the older depth-eight configuration, but it is only a bounded regression,
not proof of the hardware prefetch distance.

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

4. **PTX-to-SASS addresses:** extract per-instruction SASS PCs and correlate
   them with PTX source anchors. Replace the global scale with a kernel-local
   monotonic address map when mappings are complete enough.
5. **Preload timing and extent:** launch skipped-body kernels with controlled
   entry-to-body distance and measure ICC/GCC traffic and launch duration. This
   separates launch-time preload from demand stream lookahead.

Only after those probes identify sharing, capacity/associativity, latency, and
preload semantics should the simulator add a GPC-scoped GCC and route ICC
misses through it.

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
