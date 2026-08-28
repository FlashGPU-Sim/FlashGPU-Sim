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
  per SM cycle;
- a fixed 512-line window, anchored at the first fetch of each kernel context,
  whose accepted prefetch misses return through a 16-cycle local GCC path.

The stream-buffer depth is a real finite ahead window. Demand progress slides
the window, replacement changes a generation token, and late fills from old or
canceled generations cannot mutate current stream state. Demand reservation
failure also suppresses speculative issue for that cycle.

The local GCC return path removes the accepted prefetch from the ordinary L1I
miss queue, then feeds four normal 32-byte sector replies back through the
existing ICC fill path. It therefore preserves ICC tag allocation, MSHR
ownership, sector completion, and fill-port bandwidth. It does not make demand
accesses hit directly, and requests outside the fixed window still use the
normal lower-memory hierarchy.

This is not a shared GCC implementation. The four-line depth, address scale,
128-byte line, and 16-cycle return are model parameters, not fully
reverse-engineered RTX 5090 values. Address scale two corrects the obvious
eight-byte PTX versus sixteen-byte SASS slot mismatch, but variable PTX-to-SASS
expansion still requires a real address map. GCC sharing, capacity,
replacement, and lookup-miss/tag-hit latency remain unmodeled, so this
configuration must not replace `SM120_RTX5090` for general cycle validation.

### Short-kernel correlation

Two frozen cubins keep the validation target in the approximately 10 us range.
They were measured cold with NCU base clock control and replayed from the same
cubin/PTX artifacts. Full counters and cubin hashes are in
`tests/ci/perf/instruction_cache_reference.csv`.

| Workload | RTX 5090 cycles | Simulator cycles | Error | L1I reservation failures |
|---|---:|---:|---:|---:|
| GEMM M512 N16 K512 | 24,600.54 | 21,925 | -10.88% | 0 |
| GPT-2 FA H12 S128 D64 | 25,243.15 | 21,011 | -16.77% | 0 |

The perfect-I baselines are 21,484 cycles for GEMM and 20,103 for FA. The
experimental hierarchy therefore adds 441 and 908 cycles respectively; the
remaining full-kernel error cannot be assigned to instruction fetch alone.

The request counts are substantially closer than the previous direct-memory
model:

| Workload | HW ICC misses | Sim L1I misses | HW GCC req/hit/miss | Sim prefetch/preload-hit/outside-window |
|---|---:|---:|---:|---:|
| GEMM | 346 | 568 | 504 / 487 / 17 | 504 / 440 / 64 |
| GPT-2 FA | 3,889 | 6,596 | 5,856 / 5,738 / 118 | 6,212 / 5,796 / 416 |

The simulator counters are not asserted to be identical hardware events, but
they now have the same order of magnitude and preserve zero L1I reservation
failures on both complex kernels.

### Microbenchmark correlation

The original footprint probe lowers PTX `bar.warp.sync` to SASS NOP, so its
cycle count is intentionally not compared. The timing variant interleaves four
independent dependency chains while preserving one PTX `mad.lo.u32` and one
SASS `IMAD` per static step. This avoids calibrating the instruction hierarchy
against a serialized integer dependency.

The table compares the in-kernel `clock64` interval. Full NCU kernel cycles also
include launch/startup behavior controlled by the separate simulator launch
latency parameter.

| Steps | Text bytes | RTX 5090 body | Perfect-I body | Experimental-I body |
|---:|---:|---:|---:|---:|
| 1,024 | 17,024 | 2,384 | 2,074 | 2,074 |
| 4,096 | 66,176 | 9,095 | 8,253 | 8,724 |
| 5,120 | 82,560 | 18,830 | 10,306 | 22,834 |
| 10,240 | 164,480 | 67,925 | not run | 92,410 |

At 4,096 steps the modeled body is within 4.1% of hardware. The GCC preload-hit
count is 511 and remains 511 at 5,120, 10,240, and 11,520 steps, matching the
approximately 512-line hardware GCC-hit plateau. Beyond that window the model
is 21-36% slow because every outside-window prefetch uses the normal hierarchy,
while hardware reports growing GCC lookup-miss/tag-hit traffic rather than tag
misses. This is the main remaining instruction-hierarchy error.

Four consecutive launches of the 4,096-step probe complete in 15,801, 14,786,
14,786, and 14,786 simulator cycles with zero L1I reservation failures. This
checks stale-fill and reset behavior across launches, but is not proof of the
hardware prefetch distance.

## Remaining probes

The next hierarchy change should wait for these targeted measurements:

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
preload semantics should the local return approximation be replaced by a
GPC-scoped GCC.

## Commands

```bash
tests/run_tests.py build --arch sm120 --group microbench --profile instruction-cache
make -C tests/src/microbench/instruction_cache collect-cold
make -C tests/src/microbench/instruction_cache collect-warm
make -C tests/src/microbench/instruction_cache collect-skip
make -C tests/src/microbench/instruction_cache collect-parallel
make -C tests/src/microbench/instruction_cache collect-timing-cold
```

The parser can be checked without NCU access:

```bash
python3 -m unittest tests/src/microbench/instruction_cache/test_collect_ncu.py
```
