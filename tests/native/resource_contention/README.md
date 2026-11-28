# B200 resource competition qualification

Build locally; no server compilation or simulator library is required:

```
python3 build.py --cutlass /path/to/pinned/cutlass --out /new/package
python3 /new/package/collect.py --dry-run
python3 /new/package/collect.py --out /new/results --ncu /path/to/ncu
python3 /new/package/summarize.py /new/results
```

`run_sim.py` is a local numerical gate for these one-CTA probes. By default it
uses one simulated SM to avoid processing idle cores; `--full-topology` retains
the full B200 preset. Each run freezes its config, runtime and executable hash.
Neither smoke variant establishes native throughput or cross-CTA contention.
Use `run_sim.py --timing-matrix --family solo` (or `transfer`, `mma`, `shared`)
to replay the exact native timing loop lengths. These runs use a frozen runtime
and per-case configurations just like smoke runs; compare their measured clock
intervals to the corresponding native cases, not their whole-kernel duration.

All binaries retain both sm100a cubins and PTX. `source-provenance.json` records
retained private probes and their original hashes; build receipts record the
compiler, CUTLASS revision/patch, commands, SASS, PTX, resource use and file hashes.
Native runs use the bundled real cudart and system driver, never simulator libs.
The local build requires an x86-64 host with glibc 2.38 or newer at runtime
(for example Ubuntu 24.04); the server does not need to compile anything.
No clock-setting or shutdown command is issued. Native clock64 samples and NCU
full/base-clock captures are deliberately separate. Resume requires identical
settings/binaries and matching receipts; incomplete attempts must be preserved.

## Matrix

1. **MMA versus ordinary shared/TMEM traffic.** Three BF16 shapes: WS QK
   M64/N128/K16 (TMEM A), WS PV M64/N256/K16 (shared A, MN-major B), and ordinary
   M128/N128/K16. Matched MMA-only/background-only/mixed runtime modes use the
   same binary and resources. Backgrounds are LDS.128, STS.128, LDTM.x32,
   STTM.x32 and an integer scheduling control. Shared scratch is in the same
   or opposite 128 KiB address half. TMEM background columns 320, 384 and 448
   probe distinct, non-aliasing address ranges. All four TMEM row partitions
   are exercised. Burst32, rounds4/64 and background ratios8/32/128 expose overlap
   and startup/tail sensitivity. Report both worker slowdowns, not just runtime.
2. **Single-path TMEM service.** Retained 32dp load/store/drain/roundtrip controls
   (x4/x32/x64, 1/4 warps) plus new 16dp load/store forms 16x64b.x4,
   16x128b.x2, 16x256b.x1/x2. Depth1/4 and per-instruction/per-burst waits are
   compared at 256/1024 iterations. These measure instruction service/drain,
   not an isolated physical write latency. Full physical storage, guards and
   loaded payload/checksums are checked; skipped stores/copies cannot pass.
3. **Shared-to-TMEM copy.** 128x256b, 128x128b, both 64x128b broadcast selectors
   and 32x128b broadcast. Depth1/4, commit/wait after each copy or after a group,
   same source layout, disjoint destination slots, all copied words and guards.
   This is single-CTA service; it does not calibrate paired-CTA or copy/MMA
   contention.
4. **Shared write competition.** LDS/STS, STS/STS, LDS/TMA, STS/TMA and TMA/TMA,
   matched A-only/B-only/mixed streams. 32 KiB TMA enables disjoint strict-half
   placements; 64 KiB x2 retains the high-pressure historical control but its
   destination straddles halves and is labelled accordingly. TMA/TMA strict
   same-half uses one 32 KiB slot per stream because two independent 128 KiB
   rings cannot fit in one CTA. No unsaturated cell is evidence of no contention.
   TMA sources are deliberately hot in L2 to focus on delivery/shared writes,
   not HBM bandwidth. Per-stream 8/64 MiB sweeps separate fixed overhead.

MMA and shared-pair reporting occurs only after both workers join. No timing
stream writes results while the other worker is still executing. Shared-pair
store timing includes its explicit completion fence; interpret with matched
controls and NCU stalls. The 16dp/CP probe has only outer clocks, with no
per-instruction clock instrumentation. Check native numerical gates and SASS
retention/no-spill before treating any measurement as calibration.
The 32dp single-path probes also join active warps after their end clocks and
before verification/reporting, so a finished warp cannot inject verification
TMEM reads into another warp's timed interval. Two-length slopes are reported
alongside raw cycles; they retain loop/wait/drain costs and are not automatically
an isolated instruction latency.

The collector first runs numerical smoke gates, then three unprofiled repetitions
of every timing case, then full NCU for long/representative points, retaining
both `.ncu-rep` and raw CSV. `--family` supports bounded collection. Native
pressure and overlap are not established by compilation or simulator results.
NCU uses the longer loops to reduce setup/verification contamination. Its
whole-kernel counters still include setup and teardown; the native clock
intervals and two-length slopes isolate the measured loop more closely.
The retained 32dp host driver performs one warmup launch; NCU filters
`tmem_probe` and skips that first matching launch before capturing the measured
one. Numerical checks still execute for both launches.
`summary.json` flags inadequate overlap (<90%) or standalone traffic below 90%
of the best measured same-operation rate; those relative gates are necessary,
not proof of a physical-port peak. Integer background is a scheduling control,
not an instruction-for-instruction substitute. Even high overlap can coexist
with unsaturated ports.
MMA/background overlap is reported in both directions. A long background may
fully cover MMA while its own slowdown is diluted by its tail; the short-ratio
points help check this asymmetry. Both directions must pass the overlap gate
before the summary allows a joint no-contention inference.

Archive the complete package and results under the unified
`data/hardware/b200/resource-contention-20261126/` collection, including failures,
raw clocks, NCU, source/cubins, commands, GPU state and SHA256 receipts. Do not
remove remote data until the downloaded archive and its contents are verified.
