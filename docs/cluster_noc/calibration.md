# H200 cycle-accurate calibration of cluster features

**Scope update (2026-09-08):** the user has deferred `sq_2k`
(M2048 N2048 K8192), all three variants, due to runtime. Do not rerun or tune
it until explicitly requested. Current GEMM acceptance covers six matched
CUDA-event timings: unicast/multicast/no-TMA for M256 N8192 K2048 and
M512 N16384 K2048, each with numerical validation and <10% timing error.
All other requirements remain unchanged. Historical all-nine tables below
are retained as snapshots, not the current acceptance gate. Square partial
results are deferred, not passes; an interrupted raw harness status must
not be interpreted as a newly diagnosed kernel failure. See `todos.md` for
the current execution plan and explicit case selection.
The comparator now supports `--gemm-events-only --gemm-cases cyc_1e5,k2k`;
its default remains the historical nine-point scope. The selected six-point
report at `/tmp/h200-validation-status.BVPJyS/six-comparison/comparison.md`
contains one timing PASS and five FAIL, with no deferred-square missing rows.
This is a scope update, not an improvement in measured timing.

**Status (2026-09-07): JOB 2119329 PARTIALLY ACCEPTED; CORRECTIVE GEMM RERUN
REQUIRED.**

The previous H200 Slurm results remain superseded. Job 2119329 supplies the
current accepted low-level measurements, but it is not a final job pass: the
4096^3 GEMM timed launch deadlocked and the unicast autotune label was wrong.
The failed row is excluded pending a corrective rerun. Full analysis:
`/home/jcliu/H200_results/job_2119329/job_2119329_report.md`.

**Hardware operating-point limit:** the supplied job script requests exclusive
allocation but records only an initial `nvidia-smi`, with no clock-lock
readback or periodic clock/power/throttle telemetry. The GEMM suite's early
busy-loop probe observes 1776.14MHz versus nominal1785MHz; this is not a
frequency trace during the selected GEMMs. The initial snapshot shows all
eight GPUs at100% utilization and598–599W of600W, but lacks per-probe device
identity and subsequent telemetry; it does not establish competing work or
throttling during measurement. Vendor HBM/L2 `cycles` are CUDA-event time
multiplied by nominal `cudaDevAttrClockRate`, not direct active-cycle counts.
Their reported bandwidth remains based on measured elapsed time. Sources:
`H200_profiling/run_h200.sbatch:14–26`, job `output-2119329-H200Profiling.txt:8–46`,
`h200_gemm_compare_2119329.txt:30–34`, and vendor
`tma_bw/benchmark_framework.cuh:253–260`. Keep this uncertainty in the final
report; these files do not justify attributing a specific simulator gap to
clock drift, power limits or external contention.

**Simulator acceptance remains open.** Non-power-of-two DRAM geometry aliasing
in the production 94-channel IPOLY mapping and truncated cache-preload
addresses have been repaired and regression-tested in candidate708 below.
Earlier numerical PASS rows remain historical measurements, not final
validation. Candidate remeasurement has completed for the small GEMM and
several non-GEMM cases; all three small-GEMM timings fail the strict 10% gate.
K2K measurements are complete: only no-TMA passes the timing gate; square
GEMM measurements are user-deferred. Normal-load and TMA L2 pass.
All three HBM paths
have completed; only normal-load bandwidth currently passes.

## Current candidate validation

**Current three-path HBM recheck (complete):** normal load has now also
completed on the integer-lowering candidate, exit 0, with payload `Passed`.
It took 480626 simulator kernel cycles versus 480368 hardware-reported
equivalent cycles. Its 33554432 DRAM reads of 32 bytes account for the full
1073741824-byte payload; there were no DRAM writes. All runs use the 132SM
preset, OMP4, 1 GiB, 132 blocks, 1024 threads, stages16/chunk8192 and one
measured sample without vendor repetition loops.

| HBM path | H200 bytes/cycle | Simulator bytes/cycle | Signed bandwidth error | <10% |
|---|---:|---:|---:|---|
| Normal load | 2235.248443 | 2234.048562 | -0.054% | PASS |
| cp.async | 2292.047679 | 2570.888400 | +12.166% | FAIL |
| TMA | 2298.942583 | 2583.687631 | +12.386% | FAIL |

Hardware source: job 2119329 `h200_tma_hbm_2119329.txt` (operating-point
caveat above applies). Simulator sources:
`/tmp/h200-normal-current-results/metrics.csv` and
`/tmp/h200-async-hbm-current.Uw314B/results/metrics.csv`.
Combined report: `/tmp/h200-three-hbm-current-comparison/comparison.md`;
the comparator exits 1 because two timing rows fail, not because the
executions failed. Normal load used library SHA-256
`abfffd16bb565a58bd87f20baeea3ded55f38201486dcca741b1f19c74c68d53`;
the async runs precede only the reporting-only `n_ref` initialization.
This supersedes the previous normal-load +0.709% measurement. A uniform
HBM slowdown is not justified by this path-dependent discrepancy; the
asynchronous request/service model remains an open calibration issue.

**Current six-point rerun (complete, 2026-09-08):** the reusable harness ran
`gemm_cyc_1e5` and `gemm_k2k` with the rebuilt library and current
132SM preset under `/tmp/h200-six-current.qLOSuv/{small,k2k}`. Each process
uses OMP4, one immediate warmup and one timed sample per variant, numerical
validation, and DSM routing counters. Both completed, exit 0. `sq_2k` is
excluded. **Only one of six event timings passes; calibration is incomplete.**

Current M256 N8192 K2048 results, CUDA-event milliseconds:

| Variant | H200 job 2119329 | Simulator | Signed error | Timing gate |
| --- | ---: | ---: | ---: | --- |
| Unicast | 0.056928 | 0.042795 | -24.826% | FAIL |
| Multicast | 0.055584 | 0.0443042 | -20.293% | FAIL |
| No-TMA | 0.157312 | 0.128900 | -18.061% | FAIL |

All output comparisons and sampled CPU-reference checks pass:
`ok=1 notma_ok=1 ref_ok=1 max_abs=0 ref_abs=0`. The routing checker verifies
21 complete DSM dumps with zero global-TMA DSM payload. These are functional
and routing passes, not timing acceptance. Hardware source:
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`;
simulator source: `/tmp/h200-six-current.qLOSuv/small/metrics.csv` and
`small/logs/gemm_cyc_1e5.log.gz` beneath the same parent directory.
The source-attributed comparison is
`/tmp/h200-six-current.qLOSuv/small-comparison/comparison.md`.

Current M512 N16384 K2048 results, CUDA-event milliseconds:

| Variant | H200 job 2119329 | Simulator | Signed error | Timing gate |
| --- | ---: | ---: | ---: | --- |
| Unicast | 0.111328 | 0.157436 | +41.416% | FAIL |
| Multicast | 0.111168 | 0.141768 | +27.526% | FAIL |
| No-TMA | 0.497216 | 0.469449 | -5.584% | PASS |

K2K completed in 7259.987 host seconds. All three outputs agree bit-for-bit;
16 CPU-reference samples pass with maximum absolute difference 0.0078125
(`ok=1 notma_ok=1 ref_ok=1`, tolerance 2.83). Its route checker verifies
21 complete DSM dumps with zero global-TMA DSM payload. Evidence:
`/tmp/h200-six-current.qLOSuv/k2k/metrics.csv` and
`k2k/logs/gemm_k2k.log.gz` beneath the same parent directory. The complete
six-point source-attributed comparison is
`/tmp/h200-six-current.qLOSuv/comparison/comparison.md`.
This supersedes the partial K2K observations and unvalidated G3-only timing
evidence for the current candidate. It does not close the five failing gates.

**K2K TMA-credit diagnostic (complete, 2026-09-08):** a temporary
unicast-only M512 N16384 K2048 driver reuses the embedded winner, input
generation, `run_once` event window and validators, with one immediate
warmup and one timed launch. It omits preflights, autotuning and other
variants, so its timing is diagnostic—not acceptance. Two fresh processes
compare the frozen candidate708 with per-SM TMA request limits 384 and 768;
the copied configs differ only in that cap and both have a 1M-cycle limit,
OMP4 and 3600-second host guard. Production config/code remain unchanged.
Protocol and hashes: `/tmp/h200-k2k-tma-credit.r0R8yM/provenance.txt`.
The production value 384 and transaction quota 48 are inherited unchanged
from `configs/SM90_H100/gpgpusim.config:59-60`; 768 has no H200 hardware
support and is deliberately diagnostic only.

Both runs exited 0 and passed all-output finite checks and 16 CPU-reference
samples. Raising the cap removed the observed prefix saturation but changed
the timed launch only from 269280 cycles / 0.150857136 ms to 260331 cycles /
0.145843700 ms, a 3.3233% improvement. Warmup improved only 1.0833%.
Timed L2 access counts were identical, while misses and DRAM reads differed
slightly because the larger outstanding window changed request ordering.
This is far smaller than the 27--36% K2K TMA timing gaps and does not justify
changing the production value 384. Full results and caveats:
`/tmp/h200-k2k-tma-credit.r0R8yM/results.md`. Both raw traces were compressed
only after terminal exit and pass `gzip -t`.

The first run's bounded-to-cycle-30000 MF trace proves the 384 limit is
active. It has 540868 MF_ISSUE rows, including 186105 (34.409%) at 383 or
more requests before the source increments the counter, and 2053656
MF_RESPONSE rows, including 816201 (39.744%) at 384 before retirement;
maximum observed pre-event count is 384. The cap-768 trace reached 747
without saturating its new limit. These are event-row fractions,
not stall-cycle fractions. The trace's `global_inflight` name is misleading:
the counter belongs to a per-SM TMA unit, and the CSV has no SM column, so
mixed-SM rows cannot establish dwell time. The config also retains
`gpgpu_tma_tx_quota=48`; actual transactions can exceed that through the
model's borrowing path. This corrects the earlier proposed assumption that
the quota was zero. The completed A/B rejects the request cap as the primary
cause; retain 384 unless future hardware evidence supports a change.

**Paired K2K/small TMA lifecycle diagnostic (complete, 2026-09-08):** exact
unicast reruns with candidate708 and production timing knobs isolate the
scale-dependent reversal. The small M256 N8192 K2048 run is numerically exact
at 0.0428084023 ms versus H200 0.056928 ms (-24.802%); K2K M512 N16384 K2048
passes sampled reference checks at 0.150857136 ms versus H200 0.111328 ms
(+35.506%). Only bounded trace flags differ from production. In the trace
prefix, mean 8192/16384-byte TMA transaction lifetimes rise from
1321/1412 cycles for the all-L2-hit small run to 2777/2890 cycles for K2K,
whose prefix contains 574140 misses with 261.4-cycle mean service.

CTA lifecycle evidence rules out a launch-tail artifact: all 396 initially
resident K2K CTAs dispatch at launch, the remaining 116 dispatch as slots
open, and pending-TMA resource release averages about two cycles (maximum
33). The matched generator and selected c7 PTX also order each iteration as
previous-WGMMA wait, current TMA issue, current mbarrier wait, then current
WGMMA. `STAGES=2` rotates buffers but does not prefetch the next iteration;
the exposed load service is kernel-authored, not proof of simulator-only
serialization.

The selected embedded c7 kernel independently confirms this ordering. The
simulator and supplied H200 suite embed the same 43,296-byte unicast c7 cubin
(SHA-256 `652150efb426e09f5dbf006d0f29aac6ccb7bbff11ca58d04242b85d1cc0f663`).
The nearby `artifacts/triton_gemm_uni_sm90_c7.cubin` is not the file passed to
`cuModuleLoadData` and has a different whole-file hash, but its 11,904-byte
`.text.gemm_tma_kernel` section and `.nv.info*` sections are byte-identical to
the embedded cubin; only 64 `.debug_line` bytes differ. Its SASS therefore
validly shows the selected executable instructions: two TMA loads at offsets
`0x1fd0`/`0x2010`,
performs `MBARRIER.TRYWAIT` at `0x2020`, executes the HGMMA group, then reaches
`WARPGROUP.DEPBAR.LE` at `0x2110` before branching to the next loop iteration.
Unlike the scalar-load probe below, this GEMM does not expose a PTX-versus-SASS
cross-iteration prefetch mismatch. Executable-section SHA-256 is
`73bb8c5d3fcd532c3390b6dff21ceb27cdb196812c481ec6a0e6a1d9ad2445fe`.

The apparent multicast winner in the H200 log is a display-only bug at
`probe_gemm_triton.cu:740`: dispatch still uses `uni_loaded[best_idx]` and the
timed call passes `case_uni_k`. Both retained shapes divide the selected c7
tile (BM256/BN128/BK64, four warps, two stages), so no fallback applies.
Launches match at 64 clusters/128 CTAs for M256 N8192 K2048 and 256 clusters/
512 CTAs for M512 N16384 K2048, with 128 threads/CTA, cluster `(2,1,1)`,
65,552 bytes dynamic shared memory and `SPREAD` preference. The hardware CSV
rows identify c7 and record 0.056928 ms and 0.111328 ms respectively.

This proves current-source, executable-section and recorded-selection
agreement, not a cryptographic chain to the remote executable: job 2119329
did not emit executable/cubin hashes. A second remaining protocol caveat is
history. Hardware takes four timed repetitions (each with its own immediate
warmup inside `run_once`) after the preceding autotune/shape sweep; current
isolated simulator diagnostics take one. Inputs and launch configuration
match, but cache/allocation history is not yet controlled. Test repetition
sensitivity before treating isolated one-sample timings as final acceptance.

The synchronized simulator probe and its reproducible overlay now emit one
`GEMM_SAMPLE` log row per successful timed repetition, including shape,
variant, one-based sample index, CUDA-event milliseconds and globaltimer
nanoseconds. Aggregate CSV output and final validation are unchanged. This
allows the required hardware-matched `--samples 4 --warmup 1` run to retain
raw samples rather than only median/p90/mean. The probe rebuild and
`--gemm-selfcheck` pass; no kernel code or event window changed.

The bounded small-unicast repetition diagnostic is now complete, exit 0:
M256 N8192 K2048, current 132SM preset/OMP4, c7 winner, four `run_once`
samples each with immediate warmup. Event times in sample order are
**0.0428313725, 0.0428235307, 0.0428067222, 0.0428050421 ms**; the full
range is only 0.0615% of the first sample. Final all-output finite checking
and 16 CPU-reference samples pass (`finite_ok=1 ref_ok=1 ref_abs=0`), and
16 complete DSM dumps pass the zero-global-payload routing check.
Evidence: `/tmp/h200-small-repeat.AaJ5OP/run.log`, `diagnostic.cu`, `run.sh`
and `provenance.txt`. `run/diagnostic.csv` contains only the last sample;
use the four `REPEAT_SAMPLE` log rows for repetition analysis.
Repetition does not explain the roughly 25% small-unicast timing gap in
this diagnostic. This does not test K2K repetition sensitivity or reproduce
hardware's preceding autotune/other-shape history. No timing knobs changed.

The response path likewise has no demonstrated duplication: each aligned
128-byte TMA parent is replaced by four distinct 32-byte L2 children, retained
only as bookkeeping, and each child is injected/retired once. The parent
credit is released after all valid bytes return. The 1:4 issued-parent versus
received-child ratio is intentional sectorization, not four copies of one
reply. Altering aggregation or allowing multiple router grants would change
an unvalidated network abstraction rather than repair a proven correctness
bug; the passing standalone TMA-L2 bandwidth point argues against doing so.

A fresh K2K A/B setting only `dram_latency` from the inherited 254 cycles to
zero is also rejected: it remains correct but slows the event from
0.150857136 to 0.153112039 ms (+1.495%), adds 7584 timed L2 misses/DRAM reads,
and increases `gpu_stall_dramfull` observations by 26.33%. Retain 254. Together
with the completed 384-to-768 request-cap result above, this rejects global
clock scaling, fixed residual DRAM latency, CTA dispatch/release and TMA
request capacity as primary fixes. Full tables, provenance and raw-trace
locations: `/tmp/h200-k2k-l2-lifecycle.0Ql0qR/results.md`.

The same complete transaction traces also reject the configured architectural
TMA completion floor as the exposed delay. Every paired `COMPLETE` to
`ARCH_ARRIVE` interval is one simulator cycle: 3840/3840 small transactions
and 5626/5626 K2K transactions. Memory/network completion has already reached
or exceeded the floor before architectural arrival, so lowering the calibrated
base/size completion terms cannot accelerate either measured run.

**Interconnect input-buffer sensitivity (complete, 2026-09-08):** a fresh
K2K unicast run changes the shared input-buffer limit from 512 to 1024
packets; tracing is disabled, while PTX and hint artifacts are byte-identical.
The run passes the same numerical checks. Reply-input-full events disappear,
but the event improves only 2.029%, from 0.150857136 ms (269280 cycles) to
0.147796080 ms (263816 cycles), leaving +32.757% error versus H200. Maximum
reply input occupancy reaches 970; timed `gpu_stall_dramfull` observations
fall from 479653 to 95108 while timed DRAM reads increase by 6472. Thus the
extra storage mainly redistributes queueing rather than explaining the K2K
gap. This knob counts packets per input across all destination VOQs, applies
to both request and reply networks, and does not change the router's one-grant
per-input/output throughput. Neither depth has H200 hardware provenance.
Retain the H100 baseline value 512; do not promote 1024 from this diagnostic.

**Oversized-L2 upper bound (complete, 2026-09-08):** a fresh exact K2K
unicast rerun changes only L2 associativity from 20 to 32 ways, increasing
modeled capacity from the production, datasheet-derived 58.75 MiB to an
unsupported 94 MiB. The run remains numerically correct, and all 16,777,216
timed TMA read sectors (512 MiB) hit L2 with zero timed DRAM reads. The
counter rises from 16,777,216 before the timed launch to 33,554,432 afterward;
the latter is cumulative over warmup and measurement, not timed traffic.
Despite eliminating
capacity misses, event time worsens from 0.150857136 ms / 269280 cycles to
0.152612329 ms / 272413 cycles (+1.163%). Hardware error is still +37.084%.
Reply-input-full events rise from 1.790 to 2.864 per cycle and average reply
input occupancy rises from 189.18 to 199.22 packets, showing that sharper hit
bursts increase backpressure. Retain 58.75 MiB: neither extra cache capacity
nor miss latency explains the K2K timing gap. Evidence:
`/tmp/h200-k2k-l2-upper.xyTurT`.

The bounded request traces also rule out a gross address-distribution hot
spot. Small and K2K requests reach all 188 L2 slices; per-slice request-count
coefficients of variation are 6.02% and 8.54%, respectively. This does not
prove cycle-level fairness, but it rejects a missing-channel explanation.

**Reply matching bound (complete, 2026-09-08):** a temporary opt-in build
sampled each exact unicast run's pre-arbitration VOQ graph and compared the
greedy grants with an exact maximum bipartite matching under the same
one-input/one-output and output-space limits. Small delivered 8633/8693
(99.310%) feasible sampled grants across 118 active snapshots; K2K delivered
7038/7062 (99.660%) across 100. The constructed greedy counterexample is real
and its native control passed, but actual K2K is closer to the bound than
small and loses only 0.340%. This rejects greedy matching inefficiency as the
35.5% scale-dependent cause. The instrumentation was removed after copying
the diagnostic library, and the normal library was rebuilt. Full protocol:
`/tmp/h200-reply-match-results.md`.

**TMA transaction-rotation A/B (complete, 2026-09-08):** the production
scheduler retains a transaction after ordinary request progress and rotates
when it exceeds quota or borrows. A temporary one-line diagnostic instead
rotates after every issued request, retaining the production quota48, cap384,
widths, configuration, PTX and launch history. Small changes only from
0.0428084023 to 0.0428498611 ms (+0.097%); K2K changes from 0.150857136 to
0.151945099 ms (+0.721%) and reply pressure rises. Both remain correct. Thus
bursty transaction selection is not the missing multi-CTA overlap; retain the
production policy. The source was restored and normal library rebuilt. Full
protocol: `/tmp/h200-tma-rr-results.md`.

**Per-SM TMA-stage diagnostic and warp-scheduler A/B (complete,
2026-09-08):** temporary, opt-in local counters preserve the exact event
times (small 0.0428084023 ms; K2K 0.150857136 ms) and satisfy both accounting
identities on all 132 SMs. Small issues on every one of 2162688 nonempty-queue
SM-cycles with zero admission stalls. K2K has 8650752 issue cycles plus
9186742 cycles blocked by the inherited 384-request cap. In both cases every
queued reply is consumed that cycle, reply occupancy never exceeds one, and
response-width-limited cycles are zero. Thus local request admission is
pressured only at scale, while response FIFO depth/width is not the bottleneck.

Changing only `-gpgpu_scheduler lrr` to `gto` reduces correct K2K unicast from
0.150857136 to 0.141735017 ms (6.047%) and cap-blocked cycles by 7.842%, but
still exceeds H200's 0.111328 ms by 27.313%. This supports a multi-CTA phasing
effect but does not meet the strict gate or establish H200 scheduler behavior.
Retain the H100-inherited LRR setting. Raw logs:
`/tmp/h200-tma-stage-small/run3.log`,
`/tmp/h200-tma-stage-k2k/run2.log`, and `/tmp/h200-k2k-gto/run.log`.

An independent interconnect upper-bound A/B enables only
`-icnt_multi_grant_reply 1`, allowing one L2-side input to serve several SM
outputs in a cycle while retaining the one-reply-per-SM limit. It is correct
but slows K2K unicast to 0.152430817 ms (+1.043% versus baseline, +36.920%
versus H200) and raises cap-blocked cycles by 5.045%. Real per-slice bandwidth
support is absent, and the result is counterproductive; retain the inherited
single-grant reply model. Raw log: `/tmp/h200-k2k-reply-multigrant/run.log`.

The opposite transaction-selection bound is insufficient as well. Setting
only `gpgpu_tma_tx_quota=0` lets the selected transaction issue all requests
before rotation. It remains correct and improves K2K unicast to 0.142636970 ms
(5.449%), but is still 28.123% slower than H200. Together with the rejected
per-request rotation result, this brackets request-selection granularity:
neither extreme explains the gap. Retain the H100-inherited quota48. Raw log:
`/tmp/h200-k2k-quota0/run.log`.

The existing idealized-TMA mode provides a compute/barrier lower bound. With
all TMA memory requests completed internally (zero L2, DRAM and interconnect
traffic), correct K2K unicast is 0.0679557398 ms: 2.220x faster than baseline
and 38.959% faster than H200. This mode is deliberately unrealistic and not a
candidate configuration. It proves only that the H200 target lies between the
current memory path and the kernel's non-memory floor; memory-return modeling
can, in principle, account for the remaining K2K error. Raw log:
`/tmp/h200-k2k-ideal-tma/run.log`.

Setting only the unsupported far-L2 charge from 150 to zero is also rejected:
correct K2K unicast slows to 0.151836976 ms (+0.650% versus baseline,
+36.387% versus H200). The charge is applied once per remote sector before
L2 admission, not on both request and response. Its cumulative counter is a
sum of overlapped per-request charges, not elapsed time; zero changes arrival
ordering and sharpens downstream bursts. There is no duplicate charge to fix.
Retain the H100 fallback 150 pending a matched locality probe. Raw log:
`/tmp/h200-k2k-farl2-zero/run.log`.

Two one-knob clock upper bounds separate transport from L2 service. Doubling
only the ICNT clock 1700→3400 MHz keeps K2K correct and improves unicast to
0.138835847 ms (7.969%), still +24.709% versus H200. Doubling only the L2
clock 1700→3400 MHz instead slows it to 0.153602794 ms (+1.820% versus
baseline, +37.973% versus H200). Both clocks are deliberately unsupported
diagnostics and remain unchanged in production. Together with the idealized
memory bound, they show that no single raw transport/L2 service-rate knob
closes the gap; request/return phasing and admission remain the narrower
unresolved area. Raw logs: `/tmp/h200-k2k-icnt2x/run.log` and
`/tmp/h200-k2k-l2clock2x/run.log`.

**TMA parent/transaction tail decomposition (complete, 2026-09-08):** a
read-only join of the existing bounded TMA and L2 traces links each 128-byte
parent request to its four 32-byte L2 sectors. It includes 327,989 complete
small parents and 508,927 complete K2K parents. The dominant scale-dependent
increase occurs after L2 response injection, not in L2 admission:

| Complete-parent component (cycles) | Small mean / median | K2K mean / median |
|---|---:|---:|
| TMA issue → first L2 request | 11.2 / 3 | 46.7 / 2 |
| first L2 request → first cache accept | 489.2 / 466 | 465.2 / 437 |
| first cache accept → last L2 response | 4.2 / 4 | 91.3 / 18 |
| last L2 response → last TMA sector retirement | 321.2 / 316 | 656.1 / 554 |
| TMA issue → last sector retirement | 825.8 / 874 | 1259.3 / 1181 |

At transaction level, issue-done → last return rises from mean 1171.6 cycles
to 2058.2 cycles; the final return and transaction completion occur in the
same cycle for all 3,840/5,627 complete transactions. K2K all-hit parents are
also slow (mean 1198.4 cycles), while parents containing a miss average
1420.1 cycles, so misses amplify but do not create the delivery tail. These
traces cover a bounded warmup prefix and are diagnostic, not the timed event.
They nevertheless localize the next upper bound: the local reply router admits
at most one sector per destination SM per ICNT tick and the TMA consumer width
defaults to one. Test a temporary two-wide destination/consumer pair before
changing any production interconnect knob. Analyzer:
`/tmp/analyze_tma_parent_tails.py`; inputs are the paired lifecycle directories
listed above.

**Two-wide reply-destination upper bound (complete, 2026-09-08):** a temporary
library repeats iSLIP reply arbitration twice while preserving at most one
grant per L2 input per ICNT tick; requests remain one-pass. It is paired only
with `gpgpu_tma_response_width=2`, so the SM and TMA stages can consume the
second sector. A constructed two-input/one-SM router test plus the other six
local-interconnect tests pass. The diagnostic library was copied to
`/tmp/h200-reply-output2-lib`, the source/test changes were removed, and the
normal production library was rebuilt to its prior SHA-256
`9faf414d025b14a37474b116f195bf8e7aa566ae9b54a890723eadb1a48642c0`.

| Unicast shape | Baseline ms | Two-wide ms | H200 ms | Two-wide error | Correct |
|---|---:|---:|---:|---:|---|
| M256 N8192 K2048 | 0.0428084023 | 0.0366112031 | 0.056928 | -35.688% | yes, exact |
| M512 N16384 K2048 | 0.150857136 | 0.102214567 | 0.111328 | -8.186% | yes |

K2K reply throughput rises from 64.239 to 94.446 sectors/ICNT-cycle, reply
input-full events fall from 1.790/cycle to zero, and the event improves 32.24%
into the strict timing gate. Small throughput rises only 53.103→61.027, but
its already-fast event improves another 14.48% and moves farther outside the
gate. Thus one-wide destination delivery is a real K2K bottleneck, while a
fixed two-wide model is not a valid production calibration. The hardware must
either scale reply service with concurrency or expose another low-concurrency
cost absent from this diagnostic. Do not promote either temporary change
without a cross-shape model constrained by the TMA latency/bandwidth probes.
Raw logs: `/tmp/h200-{small,k2k}-reply-output2/run.log`.

The standalone control confirms the rejection. With the same temporary
two-grant router and two-wide consumer, the vendor-derived 1280 MiB TMA-L2
workload finishes in 249799 cycles, or 5373.029 bytes/cycle, versus H200's
4118.245 bytes/cycle (+30.469%, FAIL). Functional validation passes, but the
bandwidth ceiling is not credible. Evidence:
`/tmp/h200-l2-tma-reply-output2/run/run.log`.

**Rate-one reply burst-credit diagnostic (rejected, 2026-09-08):** matched
standalone TMA-L2 sustains 0.950 simulator packets/SM/ICNT-cycle versus 0.975
H200 32-byte sectors/SM/cycle, so sustained width two is physically rejected.
A temporary per-destination token bucket instead refills one sector per tick
and stores at most two; it can spend an unused slot on a later two-sector
burst but cannot raise continuous bandwidth. Requests and the one-grant-per-L2
input invariant are unchanged, and all seven router tests pass, including the
new burst-then-sustained-one check. The correct small unicast result is
0.0428100824 ms versus its 0.0428084023 ms baseline (+0.004%), so the model is
neutral at low load. K2K completed at 0.146668911 ms versus H200's
0.111328 ms (+31.745%, FAIL): only 2.776% faster than the 0.150857136 ms
baseline. This is too small to justify a new stateful router model, so it is
not promoted and no standalone rerun is needed. Raw log:
`/tmp/h200-k2k-reply-burst2/run.log`; temporary library:
`/tmp/h200-reply-burst2-lib`, SHA-256
`81feb46f5a9d50d34e30415d5633eb49b24cf37906b5260e6c3ca5384f02a894`.
This closes reply-network tuning: width one already matches standalone H200
TMA-L2 throughput, while tested burst credit, buffers, multi-grant, matching,
ICNT speed and consumer depth cannot close K2K. Further traffic-dependent
reply policies would be unsupported curve fitting.

**TMA execution-pipeline routing fix (complete, 2026-09-08):** the scheduler's
generic ALU branch did not exclude `TENSOR_MEMORY_ACCELERATOR_OP`, so TMA was
issued to the INT pipeline before the dedicated TMA branch was reachable.
`latency=32, initiation=32` hid the defect at pipeline stage zero; the
hardware-supported issue-latency diagnostic `latency=182, initiation=32`
indexed INT stage 150 and crashed. Excluding TMA from the generic branch then
exposed a missing `ID_OC_TMA` to `OC_EX_TMA` generic operand-collector port.
Both shared routing defects are fixed in `shader.cc`. The repaired 132-SM
small unicast GEMM completed with exact numerical validation at
0.042778153 ms and 170925 total simulator cycles under the 182/32 stress
configuration (`/tmp/h200-small-reply2-tmalat182/run.fixed2.log`). This is
essentially unchanged from the 0.0428084023 ms baseline, so instruction
latency is not a supported GEMM timing lever and production remains 32/32.
The existing `tma_copy_test` also passes functional validation with 182/32 on
the reduced H200 config (`/tmp/tma-latency-regression.1T14s3/run.log`), and
all 27 existing SM90 instruction tests pass after the fix.
The rebuilt production library SHA-256 is
`b6f023346de723b36b768701a7c567c5fcfd8d7f962c03a53a42e950b6c679be`.

**Matched GEMM-loop WGMMA timing (complete, 2026-09-08):** the existing exact
vendor mirror was run for only `f16_ss_n128_g1_o8_same`, the 128-wide,
group-one, eight-operation pattern used by the selected GEMM loop. It retains
the hardware protocol: 132 blocks, three 16-round warmups and one 256-round
timed launch, on the 132-SM preset with OMP4 and candidate708. Simulator total
is 71.9551 cycles/WGMMA versus job 2119329's 72.2275 (-0.377%, PASS). The
components differ—issue 9.0625 versus 12.0342 cycles/WGMMA and wait 495.25
versus 465 cycles/round—but their end-to-end group timing agrees. Therefore
the previously matched single-operation WGMMA result generalizes to the
actual eight-operation group; do not retune WGMMA to fit GEMM event time.
Evidence: `/tmp/h200-wgmma-gemm-pattern.3WFZmM`, simulator CSV
`WgmmaFp16CoreSweep.ss_g1_o8_matched.csv`, and hardware
`WgmmaFp16CoreSweep.ss_g1.csv`.

A multi-CTA WGMMA-overlap hypothesis was audited and rejected without a run.
For the selected m64n128k16 FP16 instruction, 262144 operations divided by
the configured 4096 operations/SM-cycle gives 64 cycles. The global compute
tail enforces that aggregate SM ceiling; the shared RF model independently
adds 32768 bytes/op at 512 bytes/cycle, also 64 cycles. Scoping either service
per CTA would be neutral or could multiply peak throughput unrealistically.
The matched PTX/SASS ordering and CTA/warpgroup wait keys show no dependency
bug, so no code change or expensive A/B is justified.

**Small no-TMA L1-latency sensitivity (complete, 2026-09-08):** two fresh
candidate708 processes run the exact M256 N8192 K2048 no-TMA winner with
identical input and warmup/timed history. Changing only the provisional,
H100-inherited `gpgpu_l1_latency` from 39 to 49 cycles changes the correct
event from 0.125500277 ms (224018 cycles, -20.222% versus H200) to
0.126109242 ms (225105 cycles, -19.835%). The +10-cycle knob produces only a
0.485% slowdown; accesses/misses are identical while timed reservation-fail
retries rise from 9233 to 11703. The added latency is mostly overlapped and
cannot close the target gap. Retain 39. Since 49 fails the target workload,
its staged scalar-load cross-check was not launched. Evidence:
`/tmp/h200-small-notma-l1.kdUaKp/results.md`.

The production-39 scalar cross-check was subsequently run using the exact
vendor `TMALatencyTest.GmemLoadBaseline4KB` source and simulator-shortened
one-warmup/five-sample protocol. Simulator median is 69350 cycles versus job
2119329's 31956 (+117.017%). Thus the serial one-thread global-load baseline
is much too slow while the parallel no-TMA GEMM is too fast. A blanket
ordinary-load latency or bandwidth adjustment cannot fix both; the remaining
error concerns workload-level issue/admission/overlap. The rejected value49
was not run through this independent probe. Evidence:
`/tmp/h200-gmem-l1-check/l1_39/` and job CSV
`TMALatencyTest.GmemLoadBaseline4KB.csv`.

Disassembly resolves the apparent contradiction: the simulator's scheduled
PTX repeatedly executes four `ld.global.nc.u32` operations followed by their
dependent `st.shared.v4.u32`, whereas the hardware executable's SASS hoists
20 `LDG.E.CONSTANT` operations before its first `STS.128`. The scalar probe
therefore does not present matched load-level parallelism. Its 96% global
scoreboard stall share and zero L1 reservation failures agree with this
dependency limitation. Do not use it to tune GEMM memory latency. This is a
probe/compiler-scheduling mismatch, not yet a proven GEMM root cause.

**Non-perfect instruction/constant-cache candidate (rejected,
2026-09-08):** the H100
baseline and H200 candidate both idealize instruction/constant cache hits.
Changing only `gpgpu_perfect_inst_const_cache` from 1 to 0 exercises the
existing configured L1I. The matched warmup remains intact. Correct small
no-TMA increases from 0.125500277 to 0.143414572 ms, reducing error against
H200 0.157312 ms from -20.222% to -8.834% (PASS). The timed launch adds 1,280
L1I misses and 80 L1C misses after the warmup. This is a plausible frontend
model rather than a fixed delay, but the bundled flag does not isolate the two
caches, departs from the H100 baseline, and is not accepted from one shape.
Source audit confirms that the one flag couples two distinct paths: perfect
L1I bypasses tags/MSHR/ICNT and perfect L1C immediately retires all queued
constant and kernel-parameter accesses. With the flag off, cache hits still
have no explicit Hopper hit-latency pipeline, and the normal cache flush does
not invalidate L1I/L1C. Treat this as empirical simulator modeling; check
warmup sensitivity and do not describe it as an H200 datasheet value.
K2K completes at 0.458259940 ms versus H200's 0.497216 ms (-7.835%, PASS),
only 0.210% slower than its already-passing baseline because the timed launch
adds no L1I/L1C misses after warmup. However, the bounded 132-SM regression
gate shows severe timing regressions: TMA 16KiB becomes 2285 versus H200 1887
cycles (+21.092%), and many multicast/mbarrier rows lose their prior agreement.
Production therefore retains the H100 perfect-cache baseline. Evidence:
`/tmp/h200-k2k-notma-icache-real/run.icache.log` and
`/tmp/h200-cache-knob-gate/comparison/comparison.md` (all five cases remain
functionally correct).

Two temporary split builds isolate the effect. Real L1I with perfect L1C gives
0.142687395 ms (-9.297%, apparent PASS); real L1C with perfect L1I gives
0.125611201 ms (-20.152%), effectively baseline. The timed 128-CTA launch
rotates onto four SMs not used by warmup, adding exactly 4x320 L1I and 4x20
L1C cold misses; those four tail CTAs determine the event. This is not a
credible H200 calibration mechanism, so do not add split cache knobs. Logs:
`/tmp/h200-small-notma-real-{i,c}-only/run.real-{i,c}.setup.log`; library
SHA-256 values are `dfc5cbc4...` and `66edc096...` respectively.

The inherited `gpgpu_TB_launch_latency=0` is also not an eligible common
timing fit. The simulator implements it as a pre-launch delay multiplied by
the entire grid size. Any nonnegative value worsens K2K TMA, which is already
27--36% slow. Fitting no-TMA alone would require about 445 cycles/block for
the 128-block small grid but only 139 cycles/block for the 512-block K2K grid,
so one value cannot fit both. Job 2119329 measures only aggregate kernel-launch
latency, not a per-block term. Keep the H100 baseline zero; no A/B is needed.

**Host-copy cache-state audit (2026-09-08):** both H2D and D2H invoke
`perf_memcpy_to_gpu` (`cuda-sim.cc:804,822`), which force-fills L2 tags
without modeling copy-engine timing. Each copy advances an L2-local logical
timestamp offset, also applied to subsequent accesses/fills; that offset
is not an added CUDA-event latency. GEMM copies C back after a variant's
timed samples and before the next variant's immediate warmup. `gpu_memset`
changes functional bytes without a matching cache timing operation. These
are cache-history approximations, not a demonstrated explanation for the
small/K2K timing discrepancy. In particular, a force fill passes `is_write`
to the probe but does not automatically mark every filled line dirty.
A separate native regression confirmed dirty-count drift when a fill cleans
a previously modified line. Both address/index fill overloads now decrement
the count on a modified-to-clean transition, preserving the existing
allocation-stage accounting. `test/check_cache_fill_dirty.cc` uses real cache
objects and verifies both overloads: candidate708 fails four controls;
the current extracted production methods pass all 20 checks, including
mixed-sector and retained-dirty controls. Evidence:
`/tmp/check_cache_fill_dirty_before.log` and `check_cache_fill_dirty_after.log`.
The fix is built in an isolated library at
`/tmp/h200-cache-dirty-build.5rQR0o/lib/libcudart.so` (SHA-256
`9faf414d025b14a37474b116f195bf8e7aa566ae9b54a890723eadb1a48642c0`).
The corrected unit run passes 216/216 tests and the direct-new-library cache
regression passes all 20 checks. Evidence is in
`unit-suite-source-linked.log`, `cache-fill-dirty.log` and `preserved.log`.
An earlier 209/216 invocation lacked the required `src` fixture symlink; all
seven failures explicitly reported missing source text. Its `unit-suite.log`
is retained as an invocation error, not a product regression. Candidate708,
candidate055 and the repository library hashes remained unchanged. This new
library is verified but has not replaced the frozen credit diagnostic or
been promoted as the calibration candidate. L2's dirty-eviction threshold
`m_wr_percent` remains zero in this config, so counter drift alone would not
explain its replacement behavior. No cache knob or copy semantics changed.

**Reproducibility update (2026-09-08):** the simulator overlay now includes
the finite-validator/status fixes and legacy HBM workload controls that had
existed only in the ignored probe tree. A zero-fuzz patch application to
profiling revision `c880781eb1bb5a2864308f8ca33d0dfd8673fab4` reproduces all
nine selected files byte-for-byte. Evidence: before/after trees and old
patch at `/tmp/h200-overlay-roundtrip.KptosE`. The live kernel tree and
binaries were not overwritten. This repair changes reproducibility, not
the measured simulator timing. Harness/comparator self-tests pass.

**Correctness-status propagation repaired:** the probe previously printed
indented `gemm_* FAIL` lines and boolean correctness rows with median zero,
but returned process status zero. The harness did not recognize those lines
or evaluate the boolean rows; the event-only comparator discarded the boolean
evidence. This could incorrectly label a numerically failed run PASS even
with finite timings. The probe now emits an explicit `FAIL: GEMM validation`
line and adds `notma_ok` to timing notes. The harness recognizes legacy failure
lines and rejects all three failed/malformed GEMM correctness metrics. The
comparator rejects explicit failed timing flags and same-shape reference,
multicast or no-TMA correctness rows on either input before event filtering.
Absent historical `notma_ok` is not itself a failure; inspect the boolean row.
Both scripts' `--self-test` pass normally and under `python3 -O`, including
equal-timing/failed-correctness negative controls. Active processes retain
their loaded code and executable: these reporting fixes do not retroactively
validate their outputs or change their timing results. No simulator knob was
changed for this repair.
The updated probe compiled separately at
`/tmp/h200-validation-status.BVPJyS/nvprof` (SHA-256
`833240c204a4fb81ff0741b96de6493c8ce3e24fbc0e62b17f149ad13b1bafad`);
its `--gemm-selfcheck` passes from the `calibration/kernels/h200_probes`
working directory. Build/self-check logs are in the same temporary directory.
The live batch executable remains SHA-256 `01d59253...` unchanged.

**GEMM numerical-validation caveat:** the existing probe binary can silently
accept NaN differences in its vector and sampled-reference checks. The
shared validators in `src/probe_gemm_triton.cu` now reject nonfinite values
and vector-length mismatches. Nineteen explicit controls plus the existing
`--gemm-selfcheck` policies pass in a separate executable; the original
validators fail ten of seventeen safe controls. Evidence:
`/tmp/h200-gemm-validation-fix.mZiTX8/selfcheck-before.log` and
`selfcheck-after.log`. Fixed executable SHA-256:
`8c6f1ae7925eeb80270c3435e4548c2c6c00657bbad3a49e3b6fec3ea7b01be4`.
The live probe remains `01d592535e92a7ecf74e5d7fa33e5cc33b7fd9b5f4411502f2983f6c77998714`;
it was not replaced during the candidate batch. Thus existing `ok=1` flags
are provisional numerical evidence, not proof that all outputs are finite.
No NaN output has been demonstrated in these runs. Keep their timings for
diagnosis, but use the repaired checker for final GEMM acceptance. Timed
launches, tile choices and metric schema are unchanged by this host-only fix.
The separate G3-only entrypoint has no output validation; it is timing-only
and must not substitute for the matched three-variant correctness check.
A post-fix small-shape regression completed separately at
`/tmp/h200-small-finite-regression.sIFFPb` (`run.log`, `metrics.csv`,
`provenance.txt`). It uses the repaired probe with the same candidate708
library, unchanged 132SM timing knobs, OMP4, device-iters1000,
M256 N8192 K2048, matched shipped tile, all preflights, warmup1/sample1,
3M-cycle cap and 7200-second timeout. DSM counter checks are enabled.
It exited zero: full-array multicast/no-TMA comparisons and 16 sampled CPU
references pass with zero absolute error, using finite-value guards. CUDA
event times are exactly unchanged: 0.0427978 / 0.0443042 / 0.125347 ms for
unicast / multicast / no-TMA. The strict routing checker verifies 21 complete
zero-payload DSM dumps. This establishes small-shape numerical regression
success, not hardware timing acceptance. The original square batch was
subsequently stopped by explicit user request; see the current scope note.

The mapping repair, 64-bit cache-preload repair and opt-in DSM statistics
hook are now built together in a separate library, without replacing the
library used by the remaining baseline trace:

- Build: `make -j2 SIM_LIB_DIR=/tmp/h200-memory-fixes-build.CtH6nJ/lib`
  after the standard setup; gcc13.3/CUDA12.8, TRACE=1. Build log:
  `/tmp/h200-memory-fixes-build.CtH6nJ/build.log`.
- Candidate library SHA-256:
  `708c448e5d3234d6c6f2be5dd38e82df5d4ce45a6594a2e3cd736272b523c4b5`.
  The regular library remains `814eedd3...`; no baseline binary was replaced.
- Current 132SM config SHA-256:
  `6caac58faeae5873a5b512c74ebac31a38b73e68857e5e0959fbe5267e5a4c49`.
  Only the obsolete hash comment changed since `1b4e48ac...`, not numeric
  settings. The live baseline already has its copied config and start-time
  provenance; it is not affected by that comment update.

| Candidate verification | Result | Evidence |
| --- | --- | --- |
| Address mapping, including actual 48GiB heap base | 12/12 geometry/mode controls pass | `/tmp/h200-map-regression.L4snoW/high-address.log` |
| High-address H2D preload | First 128 sectors preserve full addresses; benchmark exits 0 | `/tmp/h200-preload-after.dpY0wu/run.log` |
| Full existing unit suite under simulator | 216/216 pass, 26 suites | `/tmp/h200-unit-memory-fixed.UvrStH/run.log` |
| Global multicast masks and legacy tensor fan-out | 5/5 pass; five complete DSM dumps contain zero payload | `/tmp/h200-global-route-fixed.T8GekU/run.log` |
| Mapped shared-to-shared positive control | 8/8 vendor configurations complete with checksum validation; DSM TMA payload detected | `/tmp/h200-mapped-route-fixed.tbjYQs/run.log` and `control.jsonl` |

The before/after preload runs use identical temporary trace-config hashes
`eedbb15bd4bfbe6821b2f148eee20456e2958ec0acdcad3b00f558b8a077fd8b`.
Before emits truncated `0,0x20,...` and fails the checker; after emits
`0xC00000000,0xC00000020,...` and passes. The reported four-step net latency
changes from 429.250 to 338.500 cycles/load, but this tiny diagnostic is not
a hardware calibration point. These checks do not validate full memory
capacity or copy lengths above 4GiB.

The historical nine-case GEMM batch ran sequentially in
`/tmp/h200-gemm-memory-fixed.wPiMCL/results`, OMP4, warmup=1, samples=1,
using the candidate library and unchanged shape/tile/cluster settings.
`FLASHGPU_DSM_STATS=1` also enables a routing check on the explicit tensor
multicast GEMM path, which is not covered by the five short global tests.
No new timing acceptance is claimed while those cases are pending.
The small shape **M256 N8192 K2048** has now completed (1837.739 host
seconds): all three variants have `ok=1`, `ref_ok=1`, `max_abs=0`,
`ref_abs=0`. The strict global routing checker verifies all 21 complete
DSM dumps contain zero payload, now covering explicit tensor multicast in
this GEMM. Larger-shape routing and timings remain pending.

| Small-shape CUDA-event timing | Job 2119329 ms | Candidate ms | Absolute relative error | Gate |
| --- | ---: | ---: | ---: | --- |
| Unicast | 0.056928 | 0.0427978 | 24.821% | FAIL (too fast) |
| Multicast | 0.055584 | 0.0443042 | 20.293% | FAIL (too fast) |
| No-TMA | 0.157312 | 0.125347 | 20.319% | FAIL (too fast) |

The second shape **M512 N16384 K2048** has completed (7703.408 host
seconds). The old probe reports numerical checks passing, subject to the
finite-value caveat above. All 21 complete DSM dumps pass the strict
zero-payload routing check. Its matched CUDA-event results are:

| K2K CUDA-event timing | Job 2119329 ms | Candidate ms | Absolute relative error | Gate |
| --- | ---: | ---: | ---: | --- |
| Unicast | 0.111328 | 0.151900 | 36.444% | FAIL (too slow) |
| Multicast | 0.111168 | 0.141589 | 27.365% | FAIL (too slow) |
| No-TMA | 0.497216 | 0.457300 | 8.028% | Timing PASS (faster) |

That event comparison therefore has one PASS, five FAIL and three
MISSING_SIM rows, not an all-GEMM pass. The square case **M2048 N2048
K8192** was later stopped and is now user-deferred. K2K TMA errors have the opposite sign from small
GEMM, ruling out a common slowdown as a solution for all shapes. These
results are in the same batch's `metrics.csv` / `comparison/`; K2K's closed
trace is `logs/gemm_k2k.log.gz`. Final numerical acceptance must use the
repaired validator.

Hardware source: `/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`.
Candidate source: the run's `metrics.csv`, case `gemm_cyc_1e5`; detailed
comparison is in its `comparison/` directory and now retains three
missing measurements for the unfinished square shape. The matched tile
is 256x128x64, four warps, two stages, cluster2; warmup1/sample1. The memory
repairs invalidate the old small-shape timing passes. Do not undo proven
address/preload fixes to recover accidental timing agreement.
Event-window source audit: the supplied `H200_profiling` and simulator
`src/probe_gemm_triton.cu::run_once` use the same sequence: immediate
warmup/wait, start-stamp kernel, start event, one GEMM, stop event, end-stamp
kernel, then host polling/readout. Stamp kernels, warmup and validation are
outside the CUDA-event window; the 5ms polling interval is not a timing
correction. Hardware reports four samples and runs a wider autotune/shape
sweep in one process; simulation uses one sample of an isolated shape.
Hardware's GEMM implementation hardcodes one immediate warmup per timed
sample despite the global warmup8 header. Cache/allocation and repetition
history therefore remain limitations, even though the timed work matches.
This audit compares supplied sources, not independent remote-binary proof.
No measurement here establishes an additive host-submission overhead to
apply to the event times.
Subtracting the immediately preceding statistics dump from each timed
launch gives the following candidate small-GEMM scheduler samples. These
are per-launch differences, not the log's cumulative percentages:

| Timed launch UID / variant | Kernel cycles | Warp-scheduler samples | WaitTMA | WaitWGMMA | SB_SpInt | SB_MemGlobal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 / unicast | 76394 | 34283110 | 66.484% | 19.436% | 6.700% | 0.013% |
| 17 / multicast | 79083 | 35646848 | 67.976% | 18.908% | 6.364% | 0.013% |
| 21 / no-TMA | 223745 | 109732749 | 0.325% | 6.245% | 69.373% | 9.532% |

Source: `logs/gemm_cyc_1e5.log.gz` in the candidate batch. Samples aggregate
warps/schedulers and are not additive elapsed-time components or proof of
a hardware bottleneck. `SB_SpInt` labels the pending producer's pipeline;
earlier issue traces showed memory backpressure holding destination-zeroing
MOVs there, not intrinsically slow integer arithmetic. The distinct mixes
do not justify one universal slowdown for all three roughly 20% gaps.
The K2K timed unicast launch (UID13) has 271142 kernel cycles; the completed
case's CUDA-event comparison is above. Kernel cycles below are diagnostic
evidence, not a substitute for that event measurement.
Per-launch L2 counter differences (subtract the immediately preceding dump)
show distinct cache regimes:

| L2 read-sector accesses | Small uni | Small multicast | Small no-TMA | K2K uni |
| --- | ---: | ---: | ---: | ---: |
| Total reads | 4194304 | 3145728 | 4194304 | 16777216 |
| Hits | 4194304 | 3145728 | 4194304 | 14385887 |
| Pending hits | 0 | 0 | 0 | 463153 |
| Misses plus sector misses | 0 | 0 | 0 | 1928176 |

Small-shape output-write accesses are 131072, all hits; K2K has 524288,
all misses. Reservation failures are zero. K2K unicast traffic is exactly
four times small unicast, matching four times as many output tiles; it is
not extra traffic per tile. Its read miss fraction is 11.493%, versus zero
for small. `MSHR_HIT` repeats pending-hit accounting here; do not add it
again. Sources are the completed small log and UID12/13 dumps in the live
`logs/gemm_k2k.log`. The 33MiB versus 66MiB A+B footprints straddle the
configured 58.75MiB aggregate L2 capacity, supporting a cache-residency
hypothesis, not proving hardware hit rates. Evaluate the pending L2 and
larger-shape measurements before changing a common memory-service knob.
The completed K2K deltas add two distinctions. Multicast reduces L2 reads
by25% (16777216 to12582912 sectors) but DRAM read service by only0.367%
(58.843 to58.627MiB); it mainly removes repeated cached reads. No-TMA
instead has12757120 L2 read accesses, with43.696% hits,23.420% pending hits
and32.885% misses, and128.026MiB DRAM read service. Read/writeback command
deltas times the configured32B atom give these DRAM bytes; they describe
service during the interval, not exact allocation ownership. Small timed
variants have zero DRAM commands. L2 reservation failures remain zero.
The cumulative TMA response-FIFO maximum stays1, providing no evidence of
queue buildup there, but not excluding upstream reply throttling. Ordinary
`cp_async_debug_*` counters do not measure bulk-TMA quota stalls. K2K
WaitTMA scheduler samples are69.516% unicast and73.644% multicast; no-TMA
MathPipeThrottle rises from1.767% in small to33.997% in K2K. These are
scheduler samples, not elapsed-time shares. Existing aggregates cannot
separate miss-service, quota-held issuance and upstream response delays;
any next trace must distinguish them rather than fit one generic slowdown.

Read-only compute-overlap audit identified one unvalidated dependency:
`shader_core_ctx::issue_wgmma_warpgroup` starts `m_wgmma.add_op` at scheduler
issue (`src/gpgpu-sim/shader.cc`), before operand collection and
`tensor_core::issue`. `wgmma_unit_t::impl_t::cycle`
(`src/gpgpu-sim/flash/wgmma/tensor_wgmma.cc`) advances completion independently
and gates it on remaining time/RF tokens, not tensor-unit admission.
Thus downstream queueing can overlap modeled compute. This is a candidate
modeling concern, not a proven explanation for GEMM error. Before changing
it, correlate one instruction UID through scheduler issue, collector
dispatch, tensor admission and async completion on the matched tile. The
existing width-tail correction must not be fitted again to hide this cost.
A diagnostic-only lifecycle trace has built successfully separately at
`/tmp/h200-wgmma-lifecycle-build.ZNctlp` (build log `build.log`, `make -j2
SIM_LIB_DIR=.../lib`). It does not replace candidate708 or regular814.
`GPGPU_SIM_WGMMA_COLLECTOR_DEBUG=1` selects the first 16 representative UIDs
on SM0; optional `GPGPU_SIM_WGMMA_LIFECYCLE_SM` and
`GPGPU_SIM_WGMMA_LIFECYCLE_MIN_CYCLE` restrict the diagnostic. It records
registration, operand-collector dispatch, FU admission and async completion
with cumulative cycles. `test/check_wgmma_lifecycle.py` validates complete
records and reports early-completion/shortfall spans; normal and Python-O
self-tests pass. The separate library SHA-256 is
`4335a49f98417ade0e4bf91bc36e13ee27b833c23b16b9a6146ca90a0d7700b6`.
It passes 216/216 existing unit tests (26 suites, 14719ms) at
`/tmp/h200-wgmma-lifecycle-unit.1Bagr3/run.log`. The attempted named WGMMA
integration filter at `/tmp/h200-wgmma-lifecycle-smoke.WxE2sd/run.log`
matched zero tests and supplies no correctness evidence.
A matched M256 N8192 K2048 diagnostic ran at
`/tmp/h200-wgmma-lifecycle-matched.YLgqev`: 132SM, OMP4, tile256x128x64w4s2,
cluster2, 160000 cumulative cycle cap, 900-second timeout. The cap is
intentional and truncated the overall probe; require complete selected
instruction lifecycles, not full-suite acceptance, from this artifact.
Completed candidate708 GEMM correctness remains separate evidence.
The first 16 selected SM0 lifecycles are complete and pass the parser:
14 have scheduler-to-FU admission of two cycles, two have three cycles
(mean 2.125); no completion precedes admission. Each REGISTER records
compute64 + tail46 cycles. First operations of two groups complete 108
cycles after admission, versus 110 after registration; later operations
take 161–475 admission-to-completion cycles as the backend serializes.
Thus this early unicast sample does not support collector/FU pre-admission
overlap as the explanation for the 20–25% small-GEMM timing gap. It covers
only 16 early instructions on one SM, not later pressure or all variants.
The cycle-capped diagnostic is terminal: exit 1 with the explicit maximum-
cycles message at 160000 cycles (175 simulator host seconds), as intended.
All 16 lifecycles still pass the checker on the closed log. Do not restart
it; these records are not a full-kernel correctness or calibration pass. Pipeline guards
use base-object warpgroup/type fields, not a downcast to a PTX instruction.

A separate safety audit found the same unsafe downcast in the existing
`tensor_core::issue_queue_enabled_for` predicate: downstream registers hold
base `warp_inst_t` objects, not derived `ptx_instruction` objects. The source
now uses `!inst.is_wgmma_warpgroup()` after the existing depth/type guards.
`test/check_tensor_queue.cc` compiles the actual predicate with real native
objects and tests queue depths0/4 against classic MMA, WGMMA and non-tensor
instructions. All six controls pass under ASan; the original predicate
fails with a four-byte heap-buffer-overflow 112 bytes beyond the base object
in `ptx_instruction::get_opcode`. Evidence:
`/tmp/check_tensor_queue_before_audit.log` and
`/tmp/check_tensor_queue_audit.log`. The linked existing library supplies
constructors only; this is not a rebuilt full-simulator validation.
The current 132SM queue depth defaults to zero and returns before that
cast, so this repair is not a proposed calibration fix. It is not yet in
candidate708 or diagnostic4335. A separate full build now succeeds at
`/tmp/h200-queue-safe-build.jo29Nt` (log `build.log`), library SHA-256
`055d85fa2506b60c970b5580eac419ddb72abeafe17f71a6e20e21134f03bb81`.
The native regression linked directly against that library also passes all
six controls (`check_tensor_queue` in the build directory), without extracting
the predicate. Existing unit suite: 216/216 PASS, 26 suites, 15168ms at
`/tmp/h200-queue-safe-unit.4URJAi/run.log`. This is regression evidence,
not new calibrated GEMM timing. Candidate708 and regular814 hashes remain
unchanged; no running library was replaced or promoted.
The harness now fingerprints external `LD_LIBRARY_PATH` candidates and
records loader search order; external rebuild and changed search-order
regressions pass. Fingerprints enumerate candidate libraries, not an
independent dynamic-loader attestation; these runs explicitly pin the
candidate directory first in `LD_LIBRARY_PATH`.

Short non-GEMM revalidation completed with 6/6 functional PASS in
`/tmp/h200-short-probes-memory-fixed.xF79r9/results` with the same candidate
and config: vendor 256B/1KiB/4KiB/16KiB TMA load latency, representative
TMA multicast, and the mbarrier suite. Compare with the individual job
2119329 `TMALatencyTest.TMALoad*.csv`, `h200_tma_multicast_2119329.csv`
and `h200_mbarrier_2119329.csv`; preserve missing/invalid reference rows,
and do not treat the initial harness PASS as a timing acceptance result.
Post-repair DSM and all three HBM bandwidth remeasurements are complete below.
The old K2K trace has now exited 0 (6837.341 host seconds), freeing capacity
for the candidate HBM run at `/tmp/h200-hbm-memory-fixed.y7xnzH/results`:
normal load, cp.async and TMA, each hardware-matched 1GiB, OMP4, same
candidate library/config, sequential cases. Normal load completed with
payload validation PASS in 1994.277 host seconds: 476985 kernel cycles,
2251.101867 bytes/cycle, **4.01822 TB/s** at 1785MHz. Job 2119329's matched
`h200_tma_hbm_2119329.txt` normal-load row is 480368 cycles,
2235.248443 bytes/cycle, **3.98992 TB/s**. Rate error is **0.709%, PASS**.
These are decimal TB/s; the dataset is binary 1GiB. Source is
`vendor/tma_bw/simple_normal_load_test.cu`, 132 blocks, 1024 threads,
16 producer warps, 16 stages, 8192B chunk. Simulation uses warmup0/iters1;
hardware's reported aggregate uses its profiling repetition protocol.
Source-attributed results are in this run's `comparison/`. cp.async has
also completed with payload PASS (1355.516 host seconds): 418070 cycles,
2568.330241 bytes/cycle, **4.58447 TB/s**, versus hardware 468464 cycles,
2292.047679 bytes/cycle, **4.09131 TB/s**. Its bandwidth is **12.054% high,
FAIL**. It uses the matching vendor `simple_cp_async_test.cu` with the same
1GiB/stages16/chunk8192 setup. TMA HBM also completed with payload PASS
(1227.046 host seconds): 414464 cycles, 2590.675726 bytes/cycle,
**4.62436 TB/s**, versus hardware 467059 cycles, 2298.942583 bytes/cycle,
**4.10361 TB/s**. Its bandwidth is **12.690% high, FAIL**. Source is the
matching vendor `simple_tma_test.cu`, with the same launch/data parameters.
Thus all three cases are functionally complete, but only one of three
bandwidth targets passes. Do not globally throttle HBM from the cp.async/TMA
gaps while the normal-load streaming point already matches.

Current-library recheck (integer lowering factor two, pipeline bounds fix):
cp.async completed with payload PASS in 1601.659 host seconds at **417654
kernel cycles / 2570.888400 bytes/cycle (4.58904 decimal TB/s)**. Against
the same job 2119329 reference above, bandwidth error is **+12.166%, FAIL**.
This supersedes the earlier cp.async timing for the current candidate; it
does not resolve the async HBM gap. Evidence:
`/tmp/h200-async-hbm-current.Uw314B/results/logs/vendor_tma_cp_async.log.gz`,
that directory's parent `metrics.csv`, and
`/tmp/h200-async-hbm-current.Uw314B/cp-async-comparison/comparison.md`.
TMA has also completed, exit 0 with payload PASS: **415585 kernel cycles /
2583.687631 bytes/cycle (4.61188 decimal TB/s)** versus hardware
2298.942583 bytes/cycle, **+12.386%, FAIL**. Its log accounts for all
33554432 32-byte DRAM reads (1 GiB), zero DRAM writes. The complete async
comparison is `/tmp/h200-async-hbm-current.Uw314B/comparison/comparison.md`;
normal-load remains explicitly missing because it was not selected in this
invocation. Both async results supersede their earlier candidate timings.
The current cp.async log still accounts for all 33554432 32-byte DRAM
reads (1 GiB), with zero DRAM writes. Across its 94 partitions, total ACT
commands are 3819579, mean reported bus utilization is 95.32%, and mean
scheduler queue occupancy is 121.067564 / 128. Thus the current bandwidth
failure is not explained by omitted DRAM payload; the highly queued,
efficient streaming behavior persists after integer lowering.

The normal-load pass validates this streaming point, not cold dependent
DRAM latency, bank timings, physical channel identity or full capacity.
The three completed logs each report 33554432 DRAM read commands of 32B
(exactly 1GiB), zero DRAM writes, and identical L2 access/miss totals.
Thus omitted async traffic does not explain the bandwidth discrepancy.
Their DRAM scheduling statistics differ substantially:

| HBM path | Mean scheduler queue occupancy / 128 | Total ACT commands | Mean bus utilization | Mean active-cycle efficiency |
| --- | ---: | ---: | ---: | ---: |
| Normal | 29.397 | 14474787 | 83.46% | 86.57% |
| cp.async | 120.860 | 3799530 | 95.22% | 99.31% |
| TMA | 122.211 | 4153742 | 96.05% | 99.83% |

These are simulator counters averaged over channels, not measured hardware
efficiencies. Definitions are in `src/gpgpu-sim/dram.cc::print`; source logs
are `logs/vendor_tma_{normal,cp_async,tma}.log.gz` in the HBM run above.
The vendor paths legitimately differ: normal loads move through registers;
cp.async commits and waits for group0 per chunk (the alternate pipelined
branch is disabled); TMA issues an 8192B bulk transfer from one lane.
Near-full async queues and fewer activations explain their modeled advantage,
but do not establish whether queue depth, row locality, completion ordering,
or unmodeled hardware costs are responsible for the hardware gap. Refresh
commands/timing are not implemented in `dram_t`. Source inspection also
found its printed `n_ref` counter was never initialized; old printed zero
values are not valid counter evidence. The constructor now initializes it
to zero, a reporting-only fix built successfully after K2K completed
(`/tmp/h200-refresh-counter-build.log`, exit 0). The six-GEMM measurements
predate only this counter initialization, not a timing-model change.
No refresh penalty is fitted without hardware timing evidence.
Review completion dependencies and representative
channel service before fitting inherited H100 queue/timing assumptions;
these data alone do not justify a numerical replacement.
A focused source review found no premature consumer release in these vendor
paths. `src/gpgpu-sim/flash/tma.cc` tracks coalesced cp.async transactions
through commit groups; `wait_group(0)` releases only after all committed
transactions finish. The vendor's subsequent forward-barrier arrival is
therefore ordered after those copies. Candidate cp.async counters report
8388608 transactions started/completed, 1GiB issued/completed, and 131072
blocked waits matched by 131072 releases (none immediate). TMA architectural
arrival likewise requires actual memory completion as well as its configured
floor. This is source and aggregate-counter evidence, not a per-transaction
HBM trace proof. The earlier completion-trace checker covers observed
arrivals only and must run without Python `-O`; it is not an exhaustive
liveness check for every issued transaction. No completion-order repair is
justified by this review.

Candidate vendor DSM bandwidth is complete at
`/tmp/h200-dsm-memory-fixed.eFHCBY/results`: exit 0, 157 samples,
1188.581 host seconds. Source: `vendor/dsm_bw/benchmark.cu`; size-slope
fits compare with job
`/home/jcliu/H200_results/job_2119329/h200_dsm_bw_analysis_2119329/fits.csv`.
Full source-attributed comparison is in the candidate run's `comparison/`.
**10/12 non-exempt slopes pass**, not the historical 12/12:

Provenance caveat: the historical DSM run used build10/config `e009582f...`
and listed GCC13 library `2f1a68f4...`; this run uses build22/config
`6caac58f...` with library `708c448e...` explicitly pinned. All 157 sample
configurations, actual SMID placements, order and rank checksums match,
but intervening source/option changes confound a memory-fix-only attribution.
Old metadata lacks separate source/binary SHA and loader-order attestation.
The timed vendor load/store regions end before global result stores, so
those stores are not a direct timed payload cost. Do not infer a uniform
DSM slowdown or retune the fabric from this historical comparison alone.

| DSM aggregate fitted rate | Hardware bytes/cycle | Candidate bytes/cycle | Absolute error | Gate |
| --- | ---: | ---: | ---: | --- |
| One-way load | 19.879986 | 17.730206 | 10.814% | FAIL |
| Bidirectional load | 30.703601 | 30.453532 | 0.814% | PASS |
| One-way store | 18.945819 | 16.684627 | 11.935% | FAIL |
| Bidirectional store | 36.305473 | 35.792819 | 1.412% | PASS |
| One-way mapped TMA | 21.062079 | 21.331313 | 1.278% | PASS |
| Bidirectional mapped TMA | 41.370888 | 42.502653 | 2.736% | PASS |
| BW7 mixed load/store same | 21.422849 | 20.415043 | 4.704% | PASS, exempt |
| BW8 mixed load/store opposite | 29.510570 | 17.668154 | 40.129% | FAIL, exempt |
| Mixed load/TMA same | 21.383451 | 21.184381 | 0.931% | PASS |
| Mixed load/TMA opposite | 28.795756 | 29.533782 | 2.563% | PASS |
| TMA scale 2 | 41.099177 | 42.553835 | 3.539% | PASS |
| TMA scale 4 | 79.627101 | 85.091596 | 6.863% | PASS |
| TMA scale 8 | 163.851172 | 169.854390 | 3.664% | PASS |
| TMA scale 16 | 335.792440 | 339.500801 | 1.104% | PASS |

The batch's DSM latency case was SKIP because the harness default exclusion
still applies even with `--only`. It was not executed or accepted there.
An explicit `--only dsm_calibration --exclude none` run completed at
`/tmp/h200-dsm-latency-memory-fixed.lxtwwT/results`, same frozen candidate,
OMP4, hardware-matched cluster16: functional PASS, exit 0, 1624.636 host
seconds. Comparison against job `h200_dsm_calibration_2119329.csv` is in
its `comparison/`. All values reproduce the previously diagnosed latency
measurements; the memory repairs did not close local or fastest-pair gaps:

| DSM latency / visibility | Hardware | Candidate | Unit | Absolute error | Gate |
| --- | ---: | ---: | --- | ---: | --- |
| Local matrix diagonal mean | 37.0469 | 50.9219 | cycles/load | 37.453% | FAIL |
| Remote matrix mean | 193.405 | 202.516 | cycles/load | 4.711% | PASS |
| Remote matrix minimum | 178.297 | 202.516 | cycles/load | 13.584% | FAIL |
| Remote matrix maximum | 208.25 | 202.516 | cycles/load | 2.753% | PASS |
| Dependent remote-load RTT | 220.047 | 235.984 | cycles/load | 7.243% | PASS |
| Rank0-to-rank1 dependent RTT | 229.547 | 235.984 | cycles/load | 2.804% | PASS |
| Store-to-peer visibility | 352 | 375 | ns | 6.534% | PASS |
| Producer store plus fence | 660 | 665 | cycles | 0.758% | PASS |
| One active pair | 225.797 | 233.406 | cycles/load | 3.370% | PASS |
| All active pairs | 216.469 | 233.406 | cycles/load | 7.824% | PASS |

The full table has 13 PASS and two FAIL, but includes cluster-size metadata,
derived differences/ratios and duplicate unit conversions. Remote matrix
mean/min/max summarize the same matrix; dependent RTT and rank0-to-rank1
RTT summarize the same chase data. The one-way-hop estimate is half the
matrix remote-minus-local mean, not an independent one-way measurement
or the separate dependent-chase result. Do not count these summary rows
as independent timing passes. Sources are vendor-first
`src/probe_dsm.cu` / `src/probe_dsm_l23.cu` in the synced H200 probes.
Do not repeat the completed bandwidth or latency runs.

Candidate matched L2 normal-load remeasurement completed with exit 0 at
`/tmp/h200-l2-memory-fixed.MIxzLp/run.log`, explicitly pinned to candidate708,
132SM/OMP4, 1.5M-cycle ceiling and 7200-second timeout. It reuses the existing
vendor-wrapper diagnostic: 40MiB working set, repeat32 (1280MiB payload),
one warmup and one timed launch, 132 blocks, 1024 threads, stages16/chunk8192.
Executable `/tmp/h200-l2-normal-diagnostic` SHA-256:
`46cc8d2710ff108759e2fb3241bdcec9918e2f14fa8880e77675c6c7fbdc3d2d`;
source `/tmp/h200-l2-normal-diagnostic.cu` SHA-256:
`7e4994e30ccbda126d04d0dddfc0d0b701274fad9c8280eb896b653250f4bc9e`.
The timed launch (UID2) passes payload validation: **370715 cycles**,
3620.509772 bytes/cycle, **6462.61 GB/s**, versus job
`h200_tma_l2_2119329.txt:564`'s matching 16-producer/8192B row:
358536 cycles, 3743.493764 bytes/cycle, **6682.13 GB/s**.
Bandwidth error is **3.285% low, PASS**; cycle error is 3.397% high.
The warmup (UID1, 370382 cycles) is not the timed result. Hardware uses
warmup1/iters10 versus simulation warmup1/iters1. This validates the matched
normal-load streaming point, not TMA L2 service or GEMM cache residency.
A matching TMA L2 diagnostic completed separately with exit 0 at
`/tmp/h200-l2-tma-memory-fixed.d5MXY0/run.log`, using vendor
`TMAKernelWrapper<16,16>` with the same 40MiB/repeat32, 132-block,
1024-thread, stages16/chunk8192 workload and one warmup/one timed launch.
It pins candidate708, OMP4, unchanged 132SM config, 1.5M-cycle cap and
7200-second timeout. Hardware source is `h200_tma_l2_2119329.txt:193–194`:
325910 cycles, 7351.08 GB/s, warmup1/iters10. Source/executable are
`diagnostic.cu` / `diagnostic` in that temporary directory, SHA-256
`d4142c8eb8e099052eba8a67a20769a4d821e3915bd36d6651c8fca882d3f51b` /
`3e22ac1dc92fee03e75aedda8033856147856ab5e700d2d349bfc36d7224d9c7`.
Timed UID2 is **352027 cycles**, 3812.711184 bytes/cycle,
**6805.69 GB/s**, versus hardware 325910 cycles, 4118.245160 bytes/cycle,
**7351.08 GB/s**. Rate error is **7.419% low, PASS**; cycle error is
8.014% high. Warmup UID1 is 350319 cycles and is excluded. Total simulated
launch cycles are 702346; simulator host time is 1351 seconds. The vendor
checksum passes, checking the accumulated first word per chunk rather than
every payload byte. Timed L2 counter differences show 41943172 accesses,
zero misses and zero DRAM reads/writes; H2D preload already warmed the data.
This checks TMA warm-L2 service relevant to the all-hit small GEMM, not its
complete dependency chain. Both matched L2 bandwidth points are slower than
hardware within 10%; neither supports a global L2 slowdown to hide the
small-GEMM timing error.
A shared-memory service audit identifies a separate coverage limitation:
`smem_service.h` has LSU/TMA/DSM clients, but no WGMMA client. WGMMA shared
operand reads are functional accesses timed through compute/operand models,
not this byte service. The global-TMA completion-floor path used by GEMM
also reaches architectural arrival without consulting this byte service;
fallback delayed non-cluster TMA arrivals use it, while cluster-read entries
bypass it. Consequently a finite `-gpgpu_shmem_bytes_per_cycle` alone would
not model TMA/WGMMA competition. Job 2119329's local DSM latency and remote
bandwidth probes do not isolate that interaction. This is a model-coverage
limitation, not yet an established cause of the GEMM error or a calibrated
contention penalty. The corrected capped same-SM trace is now complete at
`/tmp/h200-wgmma-tma-overlap-retry.jREW8X` (`findings.md` and reproducible
`run_diagnostic.sh`): diagnostic055, OMP4, explicit device-iters1000,
160000-cycle cap, trace-only temporary config changes. It terminates at
the intended cap, not a full-kernel correctness pass. All 16 selected SM0
WGMMA lifecycles validate. Their two groups span cycles63819–64377 and
65845–66403; intervening TMA reads start64395/64462 and reach architectural
arrival65742–65781. Same-run transaction-UID joins to verbose SM IDs show
no overlap in this early unicast sample. Aggregate LSU counters cannot
establish simultaneous demand. Thus this diagnostic supports no contention
penalty; it does not cover later phases, other SMs or variants. The first
attempt at `/tmp/h200-wgmma-tma-overlap.UDbXoT` omitted device-iters1000
and exhausted the cap during setup; it supplies no overlap evidence.

The four vendor TMA latency cases have completed with functional PASS and
zero DSM payload in their completed counter dumps. Their exact comparison
is `tma_latency_comparison/comparison.csv` under the short-probe run:

| TMA load size | Job 2119329 cycles | Candidate cycles | Absolute relative error | Gate |
| --- | ---: | ---: | ---: | --- |
| 256B | 1351 | 1396 | 3.331% | PASS |
| 1KiB | 1346 | 1424 | 5.795% | PASS |
| 4KiB | 1649 | 1538 | 6.731% | PASS |
| 16KiB | 1887 | 1994 | 5.670% | PASS |

Sources are the four individual hardware `TMALatencyTest.TMALoad*.csv`
files named above and the candidate run's `metrics.csv`; the comparator
retains the full paths. These are end-to-end load/wait probe latencies,
not isolated transport-stage constants.

Representative multicast comparison: all 16 unicast/multicast end-to-end
points (256B through 16KiB) pass, with maximum absolute error 7.192%.
This is not an all-metrics pass: `tma_multicast_comparison/comparison.csv`
retains 36 PASS, 30 FAIL, five MISSING_SIM and three INVALID_REFERENCE rows.
These include derived/duplicate metrics, not 74 independent timing tests.
All eight size-specific multicast issue measurements are 35 versus 185
cycles (81.081% error); four multicast-minus-unicast deltas, both slopes,
both derived bytes/cycle rates and nonzero-reference skew measurements
also fail. Six fanout outputs are absent from this representative run:
five have MISSING_SIM status; the fanout4 skew row instead has
INVALID_REFERENCE status because its hardware reference is also zero.
Hardware-zero references cannot support percentage acceptance.

Do not apply the isolated zero-DSM-payload checker to this mixed probe:
`calibration/kernels/h200_probes/src/probe_tma_multicast.cu` gathers each
peer's `done_ns` and `payload_word` through two mapped shared loads after
the timed interval. All 37 complete dumps have zero TMA/store/atomic
commands. Final cumulative counts are 224 loads and 224 read responses,
with 1344 SRAM-load bytes: exactly eight sizes times two launches
(warmup/sample) times seven peers times (8+4) bytes. Thus nonzero DSM
load/read-response counters alone
are not evidence that global TMA used DSM. Keep the isolated routing
checker strict. The completed small GEMM passes its explicit tensor-multicast
routing check (21 complete zero-payload dumps); larger shapes remain pending.

Mbarrier comparison (`mbarrier_comparison/comparison.csv`) has four PASS,
two FAIL and one INVALID_REFERENCE, using the hardware CSV above:

| Mbarrier measurement | Job 2119329 cycles | Candidate cycles | Absolute relative error | Gate |
| --- | ---: | ---: | ---: | --- |
| Pure local arrive | 6 | 6 | 0.000% | PASS |
| Successful local try-wait | 43 | 45 | 4.651% | PASS |
| Arrive-expect-tx(0) plus wait | 71 | 76 | 7.042% | PASS |
| Failed try-wait per operation | 7776.02 | 7769 | 0.090% | PASS |
| Remote arrive issue | 0 | 6 | N/A | INVALID_REFERENCE |
| Owner local wait after remote arrive | 212 | 133 | 37.264% | FAIL |
| Remote end-to-end, converted from globaltimer ns | 284.375 | 155.295 | 45.391% | FAIL |

These remote gaps persist after the memory repairs; they are unresolved,
not evidence for increasing a bypassed legacy remote-hop knob.
A bounded diagnostic completed with exit 0 at `/tmp/h200-remote-mbar-trace.iX3yP7`:
same candidate library and 132SM numeric config, OMP4, one sample, no
warmup, one-million cumulative cycle ceiling and 900-second timeout.
The existing probe has no remote-only switch, so the local prelude is
retained. Its CSV exactly reproduces all seven short-batch mbarrier values.
All-SM warp-0 issue traces and mbarrier release traces give this remote
kernel timeline (cumulative core cycles, kernel UID 10):

| Event | Issuer SM1 | Owner SM2 |
| --- | ---: | ---: |
| Initial globaltimer read | 595691 | — |
| Initial clock64 read | 595692 | 595692 |
| Remote arrive / local try-wait issue | 595693 | 595697 |
| Arrival's following clock64 read | 595699 | — |
| Owner release scheduled | — | 595771 |
| Delayed release and first following instruction | — | 595814 |
| Owner's following clock64 read | — | 595826 |
| Result global store issue | 595716 | 595844 |
| Final globaltimer read | — | 595846 |

Arrival issue to release scheduling is 78 cycles; scheduled release costs
43, then control/return handling to the owner's clock read costs 12.
Owner raw clock span is 134 cycles minus one measured clock-overhead cycle
= 133. Issuer span is 7 minus one = 6. E2E is 155 issue cycles, quantized
to 87ns and converted at 1785MHz to 155.295 cycles. Owner clock-to-final
globaltimer is 20 cycles; the global store issues only two cycles before
that timer. There is no second wait/retry in this sample and no evidence
of a hidden long store stall. The 78-cycle interval is not an independently
observed fabric injection/delivery pair. This trace explains the simulator
measurement, not the hardware's additional cost; it does not justify
doubling the shared transport floor or inflating local try-wait latency.

## 1. Goal and fixed simulator setup

For every accepted kernel, FlashGPU-Sim must be functionally correct and its
matched timing or bandwidth metric must satisfy

```text
|T_sim - T_H200| / T_H200 < 10%
```

Use the same measurement envelope and unit on both sides: CUDA-event
milliseconds for GEMM, clock-counter cycles for instruction probes,
globaltimer nanoseconds (or explicitly clock-converted cycles) for remote
E2E, and matched size slopes or bytes/second for bandwidth. Do not count
reciprocal bandwidth and time as independent tests. Hardware-zero values
are invalid percentage references, not passes.

Use `SM90_H200_CLUSTER132`, the inferred 6x16 + 2x18 full-chip packing, with
`OMP_NUM_THREADS=4`. The reduced H200 configuration is for functional tests,
not published calibration. Fit bandwidth from size slopes rather than a single
throughput point.

The reusable simulator suite writes its current report under
`calibration/results/full_scale_calibration_final/`. It is not calibration
evidence until compared with the accepted job-2119329 rows.

## 2. Full-scale simulator suite

Run job 2119329's kernel families on the 132-SM configuration with:

```bash
python3 scripts/run_cluster_noc_demo.py
```

The representative profile uses `OMP_NUM_THREADS=4` and one timed sample.
GEMMs retain one immediate same-variant warmup, matching the job's source;
other probes omit warmup where supported. The default million-cycle ceiling is cumulative per
process. Case defaults allow 10M for the multi-launch vendor DSM sweep and
3M/5M/8M for the three GEMM shapes; `--max-cycles` overrides these limits.
These are process safety ceilings, not per-kernel calibration targets.
The suite now includes all three GEMM variants at M/N/K = 256/8192/2048,
512/16384/2048, and 2048/2048/8192, using `bm256bn128bk64w4s2` without
autotuning. Vendor HBM drivers use 132 blocks, 1024 threads, 16 stages,
8192-byte chunks, and a hardware-matched 1 GiB working set. The cumulative
HBM ceiling remains 1M cycles; each driver has a 3600-second host timeout.
The DSM latency launch explicitly uses 16 CTAs, matching job 2119329 rather
than the simulator's automatically selected maximum of 18.
The previous 32/37 functional rehearsal is superseded as calibration evidence.
The full MMA instruction sweep, multi-case DSM latency sweep, fixed-loop
cycle gate, and native-only vendor mbarrier test remain excluded by default;
select one with `--exclude none --only CASE`.

Historical calibration diagnostics (2026-09-07; before the memory fixes,
not candidate acceptance):

- `/tmp/h200-calibration-tma-latency-fit`: all four global-to-shared load
  sizes pass, maximum absolute error 6.49% against job 2119329 medians.
- `/tmp/h200-calibration-tma-mc-signed`: all eight 256 B–16 KiB unicast
  and multicast end-to-end measurements pass output checks and are within
  4.44% of job `h200_tma_multicast_2119329.csv`. These are historical fitting
  data from the now-removed size table, not independent validation. The
  multicast delta alone does not meet 10% at every size.
- `/tmp/h200-calibration-tma-linear`: the replacement linear model passes
  all 16 end-to-end comparisons, maximum absolute error **6.26%**; all
  payloads match. `test/check_tma_completion_trace.py` verifies all 32
  transfers in `/tmp/h200-tma-linear-trace.csv`: memory completion precedes
  architectural arrival. Issue latency, multicast delta and skew remain
  separate diagnostics, not implicitly passing because E2E passes.
- `/tmp/h200-calibration-gemm-config-clock`: corrected CUDA-event timing
  gives 0.0517669 / 0.0494179 / 0.133026 ms for unicast / multicast / no-TMA
  at M=256, N=8192, K=2048. Job 2119329 gives 0.056928 / 0.055584 /
  0.157312 ms: errors **-9.07% / -11.09% / -15.44%**. All output checks
  pass, but performance acceptance fails. This diagnostic used build 8.0,
  before the current global-TMA completion model. Build 12.0 reruns all
  three requested shapes in `/tmp/h200-calibration-gemm-linear-current`.
- The 256/8192/2048 GEMM diagnostic validates G1/G2/G3 outputs, but its
  CSV timing is invalid: host-time CUDA events contaminated the inferred SM
  clock. Rerun after the device-cycle event-timing fix; do not quote those
  CSV cycle or TFLOP/s values.
- Full DSM and HBM validation remains open. An unsigned conversion in the
  decreasing 4–8 KiB TMA timing interval caused excessive completion delays;
  that table has since been replaced by a linear floor. The subsequent DSM
  sweep reached 96 KiB but aborted when the spin
  watchdog mistook a finite streaming-load loop for polling. Address progress
  now resets that watchdog; both finite-stream and genuine-spin regressions
  pass on 132 SMs. The HBM normal-load diagnostic also aborted when some
  warps reached their final bar.sync while producers were still active. Its
  mixed-wait watchdog now requires absence of a running sibling warp;
  the active-producer regression and genuine mixed-wait death test both pass
  on 132 SMs. The repaired normal-load run in
  `/tmp/h200-calibration-hbm-progress` passes payload checks and takes
  138542 cycles for 256 MiB: 1937.574569 bytes/cycle, or 3.45857 TB/s at
  1785 MHz. This is 13.32% below the vendor hardware result 3.98992 TB/s
  (`h200_tma_hbm_2119329.txt`, normal-load 8192-byte chunk). Hardware used
  1 GiB, so this remains a size-mismatched diagnostic, not acceptance.
  The cp.async run in `/tmp/h200-calibration-hbm-async` passes checks at
  118887 cycles, 2257.904195 bytes/cycle = **4.03036 TB/s**, versus hardware
  4.09131 TB/s (**-1.49%**). It has the same 256 MiB / 1 GiB size caveat.
  This argues against raising peak bandwidth to repair the normal-load
  discrepancy without first investigating load-path and size effects.
  The TMA run also passes: 116528 cycles, 2303.613346 bytes/cycle =
  **4.11195 TB/s**, versus hardware 4.10361 TB/s (**+0.20%**), with the same
  dataset-size caveat. Neither result justifies changing the HBM roofline.

The incomplete DSM sweep at `/tmp/h200-calibration-dsm-signed` has ten
completed sizes for ordinary load/store: diagnostic slope errors are -7.39%
(load uni), -0.31% (load bi), -7.27% (store uni), and -6.79% (store bi).
These partial fits are not acceptance results. Shared-to-shared TMA was over
10x too fast because it incorrectly entered the global-memory TMA model.
Mapped `shared::cluster <- shared::cta` copies now use the existing DSM
transport/retry queue and notify the destination mbarrier on delivery.
Global-memory TMA multicast remains outside this path. Fresh DSM bandwidth
measurements are required before accepting any of its slopes.
The transport smoke in `/tmp/h200-dsm-transport-smoke` completes all eight
configurations with checksum validation. Its 64 KiB bidirectional TMA result
is 39.82 B/cycle; this is a single-size diagnostic, not the hardware slope
target. The full rerun is `/tmp/h200-calibration-dsm-transport`.

That full DSM rerun now completes all 157 samples with functional checks.
All 12 non-exempt size slopes are within 10% of
`h200_dsm_bw_analysis_2119329/fits.csv`; maximum error is 7.94% (unicast
loads). Shared-to-shared TMA errors are 1.28% unidirectional and 2.93%
bidirectional; 2/4/8/16-SM scaling errors are 3.56/6.93/3.55/1.29%.
BW7 is -9.49%; BW8 is -42.35%, the explicitly permitted PTX-driven exception.
The generic comparator retains BW8 as FAIL rather than hiding the numerical
gap; the supervisor acceptance policy exempts it. Raw data, fits, and a
source-attributed comparison are retained in that run directory. This run
predates the kernel-binding retirement fix described below.

The mirrored DSM matrix host runner now honors `--warmup 0` both for warmup
launches and the internal warmup loop. The measured 64-load chains and full
matrix remain unchanged. Its cumulative process ceiling is 8M cycles.

Global TMA completion now requires both actual memory completion and the
configured latency floor. Verify trace ordering with
`python3 test/check_tma_completion_trace.py TRACE.csv` (generate using
`FLASHGPU_TMA_TRACE_CSV`). The checker rejects arrivals before memory
completion or with outstanding/incomplete payload bytes. The cluster floor
is now `860 + ceil(12*bytes/1024)` cycles from transaction creation, plus
100 cycles for multicast. With the measured roughly 76-cycle probe envelope,
an integer minimax fit predicts all 16 cluster E2E targets within 6.12%.
These are effective fitted parameters, not isolated hardware pipeline
latencies. The fresh simulator sweep measures a maximum E2E error of 6.26%,
including instruction/scheduling effects. Independent GEMM validation remains
open; the earlier table-fit results do not prove acceptance of this model.

CUDA events now record the absolute simulated cycle and convert the elapsed
cycles to milliseconds using the configured core frequency. The new
`CudaEventTiming.DeviceCyclesAndRerecord` regression compares event time with
an in-kernel `clock64` interval and repeats with the same event handles; it
passes on SM120 and `SM90_H200_CLUSTER132` with four OMP threads.
An additional mixed cluster/ordinary-launch regression reproduces a separate
retirement bug: idle SMs pre-bound to a cluster kernel retain its pointer
after that kernel is deleted. Allocator reuse can make a subsequent ordinary
launch bypass the configured launch delay (509 cycles with a configured
8700-cycle delay in `/tmp/h200-launch-binding-four-before.log`). Kernel
retirement now clears matching bindings on every SM. The patched simulator passes both
the new reproducer and the existing event/rerecord test on 132 SMs with OMP=4
(`/tmp/h200-launch-binding-after.log`). GEMM timing validation remains pending. The earlier
ordinary-only and one-cluster tests passed and did not exercise this failure.
The broader build-12.0 regression passed seven basic/multiblock TMA-store
tests, then aborted in `TMAStoreTest.Pipelined2InFlight` with an invalid
shared address. This also reproduces in isolation after the retirement fix
(`/tmp/h200-pipelined-store-isolated.log`); diagnosis remains open. Do not
claim the full regression suite passes.
The register trace (`/tmp/h200-pipelined-store-registers.txt`) identifies the
store failure: a 32-bit add leaves bit 32 set in the 64-bit backing union;
the subsequent address evaluation reads that bit and turns offset zero into
`0x100000000`. Register writes now truncate to the declared logical register
width, preserving true 64-bit addresses and separate arithmetic carry flags.
Build 14.0 passes all four address/pipelined-store regressions, including
the formerly aborting `Pipelined2InFlight` (69926 cycles), seven in-flight
groups, narrow-register address wrap, and offsets above 32 bits
(`/tmp/h200-register-width-after.log`).
The build-12.0 GEMM rerun also reaches a multicast wait stall: 124/128 CTAs
retired, all issued TMA memory traffic complete, two peers waiting while
their partners wait at cluster synchronization. Diagnose this separately;
the new model is not yet GEMM-validated.
The barrier accounting investigation found another architectural mismatch:
`complete_tx` clamped the transaction count to zero, and peer completion
discarded credits before `expect_tx`. PTX explicitly permits a signed
transaction count in [-(2^20-1), 2^20-1] (NVIDIA PTX ISA 8.7, section
9.7.13.15.2, https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf).
`MbarrierTransactionOrder.CompleteBeforeExpectIsRetained` fails on build
the pre-fix simulator (`/tmp/h200-early-completion-before.log`). The model now subtracts
completion bytes without clamping and retains early credits on initialized
peers. Each transaction must be delivered exactly once; duplicate delivery
must be fixed at its source, not hidden by dropping credits. Build and
post-fix tests now pass 25/25, including the new reproducer, remote barriers,
multicast TMA dimensions/data types/OOB/stress, event timing and addresses
(`/tmp/h200-signed-mbarrier-regression.log`). The fresh small GEMM run
(`/tmp/h200-calibration-gemm-signed-mbarrier`) has passed both TMA smoke
launches and entered no-TMA, moving past the earlier multicast stall;
timed acceptance remains pending.

Additional harness/regression findings:

- The completed DSM latency diagnostic used 18 CTAs, while hardware used
  16. It reports local 50.9219 and remote-mean 202.516 cycles/load; these
  are not a matched matrix comparison. The harness now selects 16 explicitly.
- The cycle gate aborted in direct `cudaLaunchKernel`: the runtime assumed
  a compiler-pushed configuration. Direct calls now create a frame when
  absent; `CudaEventTiming.DirectRuntimeLaunchWithoutCompilerPush` reproduces
  the old abort and passes after the fix (`/tmp/h200-direct-launch-after.log`).
- The remote-mbarrier probe reached its cycle limit. The new 32-bit variant
  of `RemoteArriveUnblocksOwner` reproduces the stall after its 64-bit variant
  succeeds (`/tmp/h200-mapa32-before.log`). `mapa.u32` previously truncated
  a generic address and implicitly depended on out-of-width backing bits.
  It now uses the existing compact owner/offset shared-address encoding;
  mbarrier resolution accepts that encoding as well as `mapa.u64`. Both
  variants now pass in `/tmp/h200-mapa32-after.log`; all nine selected
  remote-barrier, transaction-order, address and event/API tests pass.
- Corrected cycle-gate/remote-mbarrier runs completed with functional PASS in
  `/tmp/h200-calibration-barrier-address-fixed`. This is not timing acceptance:
  its `comparison/comparison.csv` compares the job's `h200_cycle_gate_2119329.csv`
  and `h200_mbarrier_2119329.csv` against simulator `metrics.csv`.
  Successful try-wait, remote-load and 32-KiB remote-load gate spans differ by
  5.83%, 6.90% and 2.23%. Repeated-init/arrive and remote-barrier-loop spans
  are 68.28% and 84.01% too short; do not fit these composite loops by
  overwriting the independently measured pure-arrive latency.
  Pure arrive, successful try-wait and arrive.expect_tx(0)+wait are respectively
  6/6, 45/43 and 76/71 simulator/hardware cycles (all within 10%). Remote
  owner-local wait is 151/212 cycles and remote E2E is 173.145/284.375 cycles;
  these remain open model gaps. Hardware's zero-cycle remote-arrive row is
  not a valid relative-error target; false-try-wait's 7776.02 cycles/op
  requires separate scope investigation, not fitting the successful-wait knob.
  **Scope clarification (2026-09-07):** the independent vendor
  `MBarrierLatencyTest.WaitFalse.csv` reports a 7755 cycles/op slope,
  intercept 5 cycles and R-squared 1 across 64/128/192/256 operations; all
  reported percentiles coincide at each length. This is evidence for a
  repeatable false-path timeout, not grounds to discard the measurement.
  [PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
  distinguishes nonblocking `test_wait` from potentially suspending
  `try_wait`: absent a hint, suspension has a system-dependent limit; an
  explicit hint is in nanoseconds. Current `warp_reaches_mbarrier` returns
  false immediately for local incomplete waits without a hint, while
  `arm_trywait_timeout` interpreted explicit hints as cycles (corrected below).
  A bounded default suspension model remains open. Preserve
  the separately measured successful-wait latency. Validate finite expiry,
  early completion and GEMM before accepting a behavioral change.
  **Explicit-hint correction:** the timeout helper now converts ns to core
  cycles using the configured shader clock, rounds up, retains a one-cycle
  minimum and uses 64-bit arithmetic. No configuration value changed.
  The existing `TryWaitTimeoutExpires_PredFalse` regression now measures a
  1000 ns hint with `clock64`; with the simulator clock-property override it
  requires 1785 through 2041 cycles at the 132SM preset's 1785 MHz. Before
  the fix it failed at 1050 cycles (`/tmp/h200-mbar-timeout-before.log`).
  After the fix both this test and `TryWaitTimeoutPhaseDone_PredTrue` pass
  (`/tmp/h200-mbar-timeout-after-isolated.log`, OMP=4, full 132SM config).
  These establish expiry units and an already-completed phase, not arrival
  during suspension or a calibrated default timeout. The broader attempt
  aborted in `function_info::find_ipostdominators` while preparing
  `RemoteArriveUnblocksOwner`, before execution of its first kernel
  (`/tmp/h200-mbar-timeout-after.log`); investigate separately rather than
  claiming broad regression coverage. Test binary SHA256:
  `3cd6a8837dfc7e68417a24f3a6199a93aca4ee5439ac97a4dee4fce41ee0adc0`;
  simulator library SHA256:
  `99748fe2676b96943406431dd1cfecbfc6dfd01b741d40229d73d40f703b01f1`.
  **Regression-kernel follow-up:** compiling the remote-arrive test with
  CUDA 12.8 `-O2 -arch=sm_90a` showed its non-volatile global handshake
  load hoisted before the polling loop. The generated PTX then branched
  unconditionally to itself (`$L__BB1_8`), provoking the postdominator
  assertion. Five handshake-bearing test kernels now use the existing
  neighboring test's volatile-pointer pattern, so polling actually reloads
  the flag. No simulator control-flow assertion was relaxed; volatile is not
  a substitute for synchronization fences. The rebuilt focused selection
  passes 4/4 tests on the full 132SM config with OMP=4:
  `RemoteArriveUnblocksOwner` (both mapa widths),
  `RemoteTryWaitSeesLocalArrive`, `TryWaitTimeoutExpires_PredFalse`, and
  `TryWaitTimeoutPhaseDone_PredTrue`. Log:
  `/tmp/h200-mbar-handshake-focused.log`; rebuilt test binary SHA256:
  `7738b443bb650023c710056f53a74f43c20ce085abe1ceca0279390abe4a0926`.
  `RemoteExpectAndCompleteTx` is not covered by that passing selection;
  its separate run `/tmp/h200-mbar-handshake-fixed.log` exceeded its 180 s
  host timeout (exit 124). Its 10000-load observation loop is expensive;
  timeout is not evidence of functional correctness or a proven deadlock.
  Follow-up: replaced that test-only loop with a 4096-device-cycle observation
  window and added an explicit negative-control launch omitting `expect_tx`.
  The positive launch completes only after `complete_tx`; the negative launch
  detects premature completion. Both pass in **9.416 s**, at 14498 and 10469
  total kernel cycles respectively, on the 132SM config with OMP=4
  (`/tmp/h200-mbar-observation-negative.log`). Test binary SHA256:
  `c3367f1d806523e459a9a35ba0d3f78d932aea842686917dc0c5760c04338d24`.
  This closes that functional regression timeout, not the remote timing gap.
  The observation window is preset-dependent and must be enlarged if testing
  substantially higher-latency configurations. No simulator or profiling
  kernel behavior changed for this test optimization.
  **Default suspension candidate:** added
  `-gpgpu_mbarrier_trywait_default_timeout_ns`, default 0 to preserve H100;
  H200 132SM uses candidate 4320 ns. At 1785 MHz this rounds to 7712 core
  cycles, plus the existing release/issue cost, motivated by the vendor
  7755 cycles/op false-chain slope. Reuses the explicit-hint timer and
  early-wakeup mechanism; does not alter successful-wait latency. It applies
  to uniform local barrier/parity operands only; nonuniform no-hint waits
  remain immediate queries, avoiding an incorrect all-lane predicate on the
  first barrier's completion. Mapped remote-wait protocol is unchanged.
  `/tmp/h200-mbar-default-check.log` passes 5/5 focused tests (132SM, OMP=4),
  including default expiry at **7757 cycles** and early successful wakeup at
  **582 cycles** after another warp's delayed arrival. Explicit-hint expiry,
  already-completed phase, remote arrival (both mapa widths), and remote
  wait also pass. These are functional/model-policy tests, not a matched
  hardware-chain acceptance result. The matched probe subsequently completed
  in `/tmp/h200-mbar-default-calibration` (228.978 s). Its comparison uses
  hardware `/home/jcliu/H200_results/job_2119329/h200_mbarrier_2119329.csv`
  and the run's `metrics.csv`; see `comparison/comparison.csv` there:

  | Metric | Hardware | Simulator | Unit | Absolute relative gap |
  |---|---:|---:|---|---:|
  | Pure arrive | 6 | 6 | cycles | 0.000% |
  | Successful try-wait | 43 | 45 | cycles | 4.651% |
  | arrive.expect_tx(0) + wait | 71 | 76 | cycles | 7.042% |
  | False try-wait chain / operation | 7776.02 | 7769 | cycles/op | 0.090% |
  | Remote-arrival owner's local wait | 212 | 133 | cycles | 37.264% |
  | Remote end-to-end span | 284.375 | 155.295 | cycles, converted from globaltimer ns | 45.391% |

  Hardware remote-arrive is zero after timer correction and is **not a valid
  relative-error target** (simulator 6 cycles). The comparator now preserves
  it as `INVALID_REFERENCE` with no percentage, not an infinite-error FAIL;
  exclude it from percentage acceptance counts but retain it in the report.
  Such rows still prevent an all-accepted process exit. Self-tests cover
  zero, negative and nonfinite references and nonfinite simulator values.
  Likewise the raw false-wait `sim_knob` label is legacy metadata:
  the corrected model uses the default timeout, not successful-wait latency.
  The false-wait match does **not** close the remote gap: owner wait and E2E
  became shorter than their previous 151 and 173.145 cycle measurements.
  Do not increase pure try-wait latency to hide this separate missing cost.
  Follow-up source audit: the CSV's advertised
  `gpgpu_mbarrier_remote_hop_latency` is used only by legacy `cluster_noc.cc`;
  the enabled DSM endpoint path bypasses it. Changing that knob cannot fix
  the production measurement. A return completion packet already exists for
  remote ARRIVE, but retires transaction state rather than waking this
  probe's locally waiting owner; adding another return dependency is not
  justified. Owner release already pays 43 cycles. E2E additionally includes
  timer/clock handling and the owner's `wait_s[i]` global store before its
  final globaltimer timestamp. The later cluster sync and remote done-time
  read are outside the span. Thus the owner and E2E deficits cannot be treated
  as one identical transport penalty. An isolated issue/barrier trace should
  separate issuer arrival, owner wakeup, store and final timestamp before
  any further protocol fit; no new missing round trip has been established.
  Remote-path audit: `inject_remote_mbar` uses `dsm_issue_mbar`; the fabric
  applies its 78-cycle base visibility floor, then the endpoint delivers the
  request to the owner's barrier manager and releases any waiting warp.
  No evidence currently justifies doubling that floor or changing the pure
  43-cycle successful-wait knob. The probe requests `Spread`; the runtime
  stores this via `set_cluster_sched_policy`, but the getter has no caller
  in the issuer. Thus the preference is not modeled. NVIDIA describes
  [Spread](https://nvidia.github.io/cuda-python/cuda-bindings/13.0.2/module/driver.html)
  as spreading blocks within a cluster among SMs, without establishing exact
  physical SM placement. This is a fidelity limitation, **not proof** that
  arbitrary distant-SM placement or an extra hop would close the measured gap.
  All **16/16** tests in the rebuilt existing local `mbarrier_test.cc` pass
  with this candidate (`/tmp/h200-mbar-default-local.log`); test binary SHA256
  `7ced3eb6f88067ef723eb53c18eefde53059ca1a50dd0d3c1ee61add7ba0b5d9`.
  Fresh sequential all-three-shape GEMM evaluation is running in
  `/tmp/h200-gemm-bounded-wait` (OMP=4, default per-shape limits); none of its
  nine timing comparisons is accepted until completed and compared.
  Tested library SHA256:
  `cadf20175b977a9ab56940238e456154a2b083db904e2e2bedf2dca8b17273ed`;
  config SHA256:
  `1b4e48ac5006742c5aae36a82f20295bb02cd728c976481ce90e527484788ccc`;
  test binary SHA256:
  `32b75228a3241795bef24ba9643c8f2764af5ff56da3db5b853ec2c41e06b91f`.
- Broader regression checks completed: 215/215 unit tests in
  `/tmp/h200-full-unit-regression.log`, and 23/23 selected vector-operand,
  matrix load/store and FP16 MMA integration tests on the 132-SM H200 config
  in `/tmp/h200-arithmetic-regression.log`. These cover the tested features,
  not a blanket claim that every simulator feature is bug-free.
- The small GEMM's three smoke launches now complete, including no-TMA
  (235422 simulator cycles). These smoke launches check completion, not
  numerical output; timed measurements and their subsequent numerical
  checks remain pending. Smoke timing is not substituted for CUDA-event
  acceptance timing.
- That run subsequently completed with numerical checks passing for all
  three variants (`ref_ok=1`, unicast/multicast/no-TMA outputs agree).
  For M=256, N=8192, K=2048, event times are 0.0657765/0.0613064/0.133796 ms
  (unicast/multicast/no-TMA), versus job 2119329's
  0.056928/0.055584/0.157312 ms. Signed errors are +15.543%, +10.295%,
  and -14.949%; all three fail the strict timing gate. This is the
  pre-WGMMA-tail-correction baseline, not current-model acceptance.
  Authoritative metrics: `/tmp/h200-calibration-gemm-signed-mbarrier/metrics.csv`;
  hardware: `/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`.
- The matched 16-CTA DSM matrix is active in
  `/tmp/h200-calibration-dsm-latency-matched16`.
- The matched 16-CTA run subsequently completed with functional PASS.
  Its `comparison/comparison.csv` gives local 50.9219 vs 37.0469 cycles/load
  (+37.45%), remote mean 202.516 vs 193.405 (+4.71%), dependent remote RTT
  235.984 vs 220.047 (+7.24%), and one-pair/all-pair latencies both 233.406
  vs 225.797/216.469 (+3.37%/+7.82%). Remote matrix entries are flat in the
  simulator; hardware spans 178.297–208.25 cycles/load, so mean agreement
  does not validate topology variation. Store visibility is 210 vs 352 ns,
  and producer store-plus-fence is 15 vs 660 cycles. Those remain model gaps,
  not grounds for silently accepting the entire suite's timing.
- Hardware-matched 1 GiB HBM runs are active in
  `/tmp/h200-calibration-hbm-matched`; earlier 256 MiB results remain
  explicitly diagnostic. Metric notes now report the parsed dataset size.
  The normal-load case completed with payload checks passing: 2101.338260
  bytes/cycle versus hardware 2235.248443 bytes/cycle, 5.991% low. At the
  configured 1785 MHz this is 3.750889 TB/s versus 3.98992 TB/s. Hardware
  source is the exact 16-producer/16-stage/8192-byte/1-GiB row in
  `h200_tma_hbm_2119329.txt`; the source-attributed comparison is in the run's
  `comparison/` directory. The matched dataset closes the earlier 256-MiB
  normal-load discrepancy without changing HBM knobs. The matched cp.async
  case subsequently completed with payload checks passing in **2415.519 s**:
  **2508.256165 bytes/cycle** (428083 cycles for 1 GiB), versus hardware
  **2292.047679 bytes/cycle** (468464 cycles), **9.433% high**. Vendor-reported
  bandwidth is **4477.24 GB/s** simulated versus **4091.31 GB/s** hardware.
  Both use 132 blocks, 1024 threads, 16 producer warps, 16 stages, chunk
  8192 bytes and repeat=1. Hardware warmup=1/iters=10; the simulator retains
  one measurement and omits warmup for this HBM-streaming case. These rows
  compare sustained payload bandwidth, not DRAM bank timings or cold-load
  latency. Raw simulator log:
  `/tmp/h200-calibration-hbm-matched/logs/vendor_tma_cp_async.log.gz`;
  hardware exact winning row in `h200_tma_hbm_2119329.txt`. The regenerated
  `comparison/` shows both normal and cp.async inside 10%, with TMA still
  `MISSING_SIM` while its process runs. The cp.async margin is only 0.567
  percentage points, so retain the exact shape and recheck the final frozen
  build. This batch spans library rebuilds and predates per-case library
  hashing; do not present its initial library inventory as proving a coherent
  final-build validation. No HBM knob was changed to obtain these results.
- Historical, pre-memory-fix HBM batch (not candidate acceptance): TMA passed payload validation in
  1677.942 host seconds, with 429470 simulated kernel cycles for 1 GiB.
  Its bandwidth is 2500.155596 bytes/cycle (4462.78 GB/s at 1785 MHz),
  versus job 2119329's matched vendor row at 2298.942583 bytes/cycle
  (4103.61 GB/s, 467059 cycles): **8.752% high, PASS**. Regenerated
  `/tmp/h200-calibration-hbm-matched/comparison/comparison.csv` now contains
  all three bandwidth comparisons: normal load -5.991%, cp.async +9.433%,
  TMA +8.752%. This supersedes the preceding running/MISSING_SIM status,
  not its provenance caveat: all three need a coherent frozen-build check.
  These aggregate measurements do not establish bank timings, DRAM residual
  latency, or Far-L2 latency. Preserve the physical bandwidth geometry.

The matched L2 normal-load diagnostic has also finished (exit 0, payload
check PASS), with one warmup and one timed launch. It reuses the vendor
`tma_bw` normal-load wrapper and measurement framework: 132 blocks,
1024 threads, 16 producer warps, 16 stages, 8192-byte chunks, 40-MiB working
set and repeat=32 (1280 MiB logical payload). Job 2119329's exact target is
`h200_tma_l2_2119329.txt:564`, under the 16-producer/8192-byte header.
Hardware retains one warmup and ten measurements; the simulator keeps the
same cache-warming step but only one measurement.

- Timed cycles: simulator **405292**, hardware **358536**, **+13.041%, FAIL**.
- Bandwidth: simulator **3311.630331 bytes/cycle** (5911.26 GB/s), hardware
  **3743.493764 bytes/cycle** (6682.13 GB/s), **-11.536%, FAIL**.
- The preceding **414050-cycle warmup is not the timed result**. Runtime
  reports 4746 host seconds for the diagnostic. These two comparison rows
  are correlated views of one measurement, not two independent tests.
- Raw log: `/tmp/h200-l2-normal-matched.log`; source:
  `/tmp/h200-l2-normal-diagnostic.cu`; derived CSV/Markdown:
  `/tmp/h200-l2-normal-comparison/comparison.{csv,md}`. This diagnostic was
  launched before the default mbarrier-timeout build and needs the same
  frozen-build provenance check as the other older diagnostics.

HBM streaming passes do not establish L2 fidelity. The L2 result does not
isolate the 286-cycle base latency, so no L2 knob was changed solely to fit
this aggregate measurement. Review L2 traffic/service and instruction issue
before attributing the residual to a latency component.

The bounded-default-wait GEMM batch has completed its first shape,
**M=256, N=8192, K=2048**, in 2449.034 host seconds. All three variants pass
payload/reference checks (`ok=1`, `ref_ok=1`, `max_abs=0`, `ref_abs=0`).
The launch uses the hardware-matched 256x128x64 tile, four warps, two stages,
two-CTA clusters and 64 clusters. Primary CUDA-event comparisons against
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv` are:

| Variant | Hardware (ms) | Simulator (ms) | Signed time error | Timing gate |
|---|---:|---:|---:|---|
| Unicast TMA | 0.056928 | 0.0660353 | +15.998% | FAIL |
| Multicast TMA | 0.055584 | 0.0600779 | +8.085% | PASS |
| No-TMA input | 0.157312 | 0.131175 | -16.615% | FAIL |

Evidence: `/tmp/h200-gemm-bounded-wait/event_comparison/comparison.csv`,
raw `csv/gemm_cyc_1e5__h200_gemm_compare_local.csv`, and compressed log
`logs/gemm_cyc_1e5.log.gz` under that run directory. Timed kernel cycles are
117873, 107239 and 234147 respectively. The harness's case-level PASS means
functional completion, **not** that all three timing comparisons passed.
Library SHA256 is `cadf20175b977a9ab56940238e456154a2b083db904e2e2bedf2dca8b17273ed`;
base config SHA256 is `1b4e48ac5006742c5aae36a82f20295bb02cd728c976481ce90e527484788ccc`.
Larger shapes remain queued/running in the same batch. Multicast now passes,
but the opposite unicast/no-TMA errors preclude accepting a global clock
rescaling as a remedy. The raw-PTX reorder-off diagnostic remains separate
from preset acceptance. No additional timing knob was changed for this result.

Build-tag strings are diagnostic labels, not unique revision identifiers:
incremental builds can retain a tag. Use the harness's simulator-library
SHA-256 fingerprints and per-run logs for provenance, not tag numbering.

The WGMMA width investigation uses `/tmp/h200-wgmma-width-diagnostic.cu`,
which includes the mirrored vendor `wgmma_async_latency_bench.cc` unchanged
and calls its existing `run_case` for SS N=64 and N=128. Only the selected
widths and active block count (one) differ from the hardware sweep; the
128-thread warpgroup, three 16-round warmups and 256-round timed loop are
retained. The simulator still uses the full `SM90_H200_CLUSTER132` config
and OMP=4. This is an isolated instruction diagnostic, not full-chip GEMM
acceptance. Baseline log: `/tmp/h200-wgmma-width-before.log`; output CSV:
`/tmp/h200-wgmma-width-before/h200_wgmma_width_diagnostic.csv` when complete.
Compare against job 2119329's `WgmmaAsyncLatencyBench.F16SsShapeSweep.csv`.
Source SHA-256 is `4ddfc0f46e9012c5c0bffbebc8a13f41d5294755e4a1d8686193b3f740b8c235`;
binary SHA-256 is `8807e915501a8996b3b6b20def5cf488e51204627a1a7a7b2bae4caec8323d7a`.

The baseline diagnostic completed: SS N=64/128 issue spans were both
12.75 cycles/op, waits were 104/211 cycles, and total spans were
124.652/231.641 cycles/op. Hardware waits are 75/107 cycles. The simulator
was adding width-dependent compute to a table populated with measured wait
times, then doubling the tail above N=64. The candidate correction uses
SS tail=46 cycles: hardware wait follows 43 + N/2, and the matched simulator
envelope gives compute + tail - 3. Completion-tail lookup now clamps to the
last table width instead of multiplying that tail; compute still scales
with instruction work. This lookup correction also affects wider RS/integer
instructions, whose configured tables are unchanged and require regression
checks. RS tail calibration remains open. Candidate diagnostic log is
`/tmp/h200-wgmma-width-after.log`; `test/check_wgmma_width.py <output.csv>`
fails on the baseline and requires both SS wait targets strictly within 10%.
Independent GEMM acceptance is still required; existing long runs loaded
the earlier model and must not be presented as validating this correction.
The corrected diagnostic completed successfully: N=64/128 waits are exactly
75/107 cycles, and total spans are 95.652/127.641 cycles/op versus hardware
97.8047/129.789 (2.20%/1.66% low). Issue spans remain 12.75 cycles/op versus
4.29688/4.27344; the correction does not establish an issue-timing fit.
The width checker passes on the corrected CSV and fails on the baseline.
Broader SS/RS group and N=256 RF regressions are running under
`/tmp/h200-wgmma-tail-regression`. The corrected diagnostic loaded the
gcc-13.3.0 release simulator library (SHA-256
`25344de577be51655bf6a79f2cd0e27bbef82e4105b53e507a2202965c3b0476`);
do not use the separately installed gcc-15.2.0 library hash for this run.
Those seven SS/RS g1/g2/g4 and RF regression cases subsequently completed
with functional PASS. The fresh small GEMM is running in
`/tmp/h200-gemm-tail-corrected`; it subsequently completed in 2835.208 s
with `ok=1`, `ref_ok=1`, `max_abs=0`, and `ref_abs=0`. This diagnostic
predates the fence and bounded-default-wait fixes; it is not current-preset
acceptance. Shape **M=256, N=8192, K=2048**, tile **256x128x64**, 4 warps,
2 stages, cluster size 2, 64 clusters, OMP=4. CUDA-event comparisons:

| Variant | Job 2119329 (ms) | Simulator (ms) | Signed gap | Gate |
|---|---:|---:|---:|---|
| Unicast TMA | 0.056928 | 0.0652246 | +14.574% | FAIL |
| Multicast TMA | 0.055584 | 0.0625429 | +12.520% | FAIL |
| No-TMA | 0.157312 | 0.131651 | -16.312% | FAIL |

Hardware source: `/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`,
metrics `gemm_unicast_B_cyc_1e5_ms`, `gemm_mcast_B_cyc_1e5_ms`, and
`gemm_notma_cyc_1e5_ms`. Simulator source:
`/tmp/h200-gemm-tail-corrected/csv/gemm_cyc_1e5__h200_gemm_compare_local.csv`.
The same run's `event_comparison/comparison.{csv,md}` contains the three
primary-event comparisons, excluding derived TFLOPS/globaltimer diagnostics
and unmeasured hardware shapes. Config SHA256:
`873b624ac7350492e94dcd9551d57f6b8c4834d9adccb68706bc3a2b5235a573`;
loaded gcc-13.3 simulator library SHA256:
`25344de577be51655bf6a79f2cd0e27bbef82e4105b53e507a2202965c3b0476`.
The WGMMA completion fix alone therefore does not close GEMM timing. The
opposite error directions also rule out a single clock rescaling as a fix.
Per-launch deltas from that run's cumulative L2 statistics confirm distinct
memory paths: timed unicast/multicast (launch IDs 10/13) add TMA reads and no
ordinary global reads; timed no-TMA (launch ID 16) adds **4194304 ordinary
global read lookups**, including **3839080 immediate hits (91.53%)**,
194156 hit-reserved lookups, 40267 misses and 120801 sector misses, with no
additional TMA reads. These are simulator cache-lookup counts, not measured
hardware traffic or unique bytes. Thus the no-TMA variant is not accidentally
executing the TMA input path. The L2-resident diagnostic remains relevant to
its speed discrepancy; HBM streaming bandwidth alone cannot validate it.
Exact compiled-winner audit: extracted the embedded
`triton_gemm_notma_sm90_c7_cubin` (72608 bytes) without modifying it, using
`/tmp/h200-export-notma-c7.cc`, into `/tmp/h200-notma-c7.cubin`.
`cuobjdump --dump-resource-usage` reports **REG=255, STACK=0, LOCAL=0**;
SASS contains **zero LDL/STL** and eight static `HGMMA.64x128x16.F32`
instructions. Therefore native register spills are not an evidenced cause
of this winner's timing gap. Cubin SHA256:
`2c33687df3d479e4fd91808f87953d98af8832213d983ddd71db5e4585ab6008`.
Its embedded `.nv_debug_ptx_txt` section contains 128 static
`ld.global.b16` instructions and eight WGMMA operations, matching the native
128 LDG and eight HGMMA instructions. The 32 packed-half `mov.b32` operations
also correspond in count to 32 native PRMT instructions; counts alone do not
establish their timing or register-bank behavior. **No-TMA means no TMA input
loads here, not a completely TMA-free kernel:** its output still uses
`cp.async.bulk.tensor.2d.global.shared::cta.bulk_group`. Preserve that output
path when comparing with job 2119329; replacing it would change the benchmark.
Do not substitute `artifacts/triton_gemm_notma_sm90.cubin`: its accompanying
metadata describes tile 128x256x64 rather than the required 256x128x64,
and that different artifact reports a 56-byte stack. Both H100 and H200
presets enable ordinary PTX reorder and disable SASS-guided reorder; an
instruction-scheduling explanation still requires a controlled comparison.
Scheduling diagnostic follow-up: SASS-guided mode cannot currently launch
the combined profiling executable. A pinned-current-library attempt in
`/tmp/h200-sass-guided.aCgn1o/pinned-library.log` aborts before kernels with
`expected one selected arch, found sm_90 sm_90a sm_l23`. The architecture
helper uses the first `sm_` substring in a filename, so `probe_dsm_l23...`
also supplies a spurious architecture token. Correcting that token alone
would not resolve the mixed sm_90/sm_90a guide requirement. This optional
path is not enabled in the baseline; no simulator code was changed here.
The earlier unpinned attempt in that directory selected another installed
compiler build and is not evidence about the current simulator.
As a supported scheduling sensitivity check, started the unchanged GEMM
executable for M=256/N=8192/K=2048 in `/tmp/h200-raw-ptx.x8iGhi`, with only
`gpgpu_ptx_reorder=0` in a copied config (plus a 3M cumulative cycle ceiling).
It retains OMP=4, one sample, no extra warmups, all three winner variants,
and the same pinned gcc-13.3 library as `/tmp/h200-gemm-bounded-wait`.
The repository preset remains unchanged. Temporary config SHA256:
`20239acb1570df5f2882b422d929accefe3b7d9dc46d6b7e88a8ae802cc0a688`.
This run is a diagnostic, not an accepted alternative preset or evidence
that disabling reordering improves fidelity.
The raw-PTX run has now finished with exit 0 and exact reference matches
on all three variants (`ok=1`, `ref_ok=1`, `max_abs=0`, `ref_abs=0`).
For M=256, N=8192, K=2048, its primary CUDA-event comparisons are:

| Variant, reordering off | H200 (ms) | Simulator (ms) | Signed time error | Gate |
|---|---:|---:|---:|---|
| Unicast | 0.056928 | 0.0650594 | +14.284% | FAIL |
| Multicast | 0.055584 | 0.0665221 | +19.679% | FAIL |
| No-TMA input | 0.157312 | 0.136501 | -13.229% | FAIL |

Raw CSV: `/tmp/h200-raw-ptx.x8iGhi/h200_gemm_compare_local.csv`;
source-attributed comparison: that directory's
`nine_event_comparison/comparison.{csv,md}` (the six untested larger-shape
rows remain MISSING_SIM). Hardware is job 2119329's original GEMM CSV.
Timed kernel cycles are 116131/118742/243655 respectively. Reordering off
slightly improves unicast and no-TMA errors but loses the multicast pass;
all three still fail. Keep `-gpgpu_ptx_reorder 1` in the preset. This closes
the reorder-off experiment as an insufficient remedy, not GEMM calibration.
An exact-cubin occupancy audit also rules out a register-count mismatch as
the explanation for the TMA variants' three-CTA-per-SM resource limit.
Both embedded c7 winners report `REG=154`, `STACK=0`, `LOCAL=0` with
`cuobjdump -res-usage`; the simulator likewise loads 154 registers/thread.
At 128 threads and 65552 dynamic shared bytes per CTA, three CTAs fit the
configured resources. Native cubins additionally report 1024 static shared
bytes, absent from the embedded PTX's static allocation; including these
still permits three CTAs within the selected 196-KiB shared carveout.
This checks resource accounting, not actual hardware cluster residency or
scheduling. Do not force two resident CTAs merely to fit GEMM timing.
Exact extracted cubin SHA256: unicast
`652150efb426e09f5dbf006d0f29aac6ccb7bbff11ca58d04242b85d1cc0f663`,
multicast `d5a8d8318bd9e2d377ea57067b5f270ef9f3bc2f10c2fe9473e7481219f6d422`.
The extraction/check used `/tmp/h200-export-tma-c7.cc` and the existing
`triton_gemm_{uni,mcast}_sm90_c7_cubin` arrays, not a different artifact tile.
Current bounded-wait all-three-shape evaluation remains separate in
`/tmp/h200-gemm-bounded-wait`.
Host-budget follow-up: the 512x16384x2048 diagnostic completed its no-TMA
smoke after **813124 cycles**, following unicast/multicast smoke at
313457/291954 cycles. This is forward progress, not a proven hang. At the
observed roughly 3-6 ms of host work per simulator cycle, repeating the three
variants for timed measurements can exceed the old two-hour case guard.
The older `/tmp/h200-calibration-gemm-linear-current` batch has now reached
its actual 7200-second host timeout for `gemm_k2k` (exit 124 at 7200.359 s).
The last logged cumulative cycle was 1860000 with TMA requests and responses
still advancing; this is a host-budget failure, not a diagnosed deadlock.
No complete timing CSV was produced, so this case supplies no accepted GEMM
event comparison. That batch proceeded to `gemm_sq_2k`; the current-model
`gemm_k2k` was already running in `/tmp/h200-gemm-bounded-wait`, so no duplicate
retry was launched.

The older batch's square case subsequently reached its real host timeout
as well: **7200.284 s**, with its last logged cumulative cycle at 3910000
during timed no-TMA execution. All three square smoke checks had passed
(unicast/multicast/no-TMA: 514699/437641/1568336 kernel cycles). Timed
unicast and multicast completed at 537680 and 436572 kernel cycles, but
timed no-TMA did not finish; these partial counts are not a complete
CUDA-event comparison and are not promoted into the nine-event gate.
Evidence: `/tmp/h200-calibration-gemm-linear-current/results.json` and
`logs/gemm_sq_2k.log.gz` in that directory. Session 26163 is terminal.
The separate current-model square case in `/tmp/h200-gemm-bounded-wait`
remains live; no duplicate retry was started. This timeout establishes
insufficient host budget, not a diagnosed simulator deadlock.

New harness invocations therefore allow **7200/14400/21600 host seconds**
for cyc_1e5/k2k/sq_2k respectively, with unchanged **3M/5M/8M cumulative
cycle ceilings**, OMP=4 and strict <10% event-time acceptance. The self-test
checks both sets of limits. Existing running Python processes retain their
original two-hour guards; do not assume this edit extends them in place.
Observe those handles to a terminal state before deciding whether a retry is
needed. This changes a host execution budget, not simulated performance.

The current-model square case in `/tmp/h200-gemm-bounded-wait` has now
also terminated: **TIMEOUT, exit 124, 7200.364 host seconds**. Its final
logged cumulative cycle is 4680000, with 8/256 CTAs completed in timed
no-TMA. Preserve its `results.json` and `logs/gemm_sq_2k.log.gz`; no complete
square CSV was produced, so the square comparisons remain missing.
Only after that terminal result, a square-only retry was started in
`/tmp/h200-gemm-square-extended`, using the existing harness's 21600-second
guard, unchanged 8M cycle ceiling, SM90_H200_CLUSTER132 and OMP=4.
The launch explicitly pins the gcc-13.3.0/cuda-12080 release simulator
library; no rebuild, launch-geometry change or timing-knob adjustment was
made. This supersedes the preceding bounded-wait running status.

Current-model k2k completion: `/tmp/h200-gemm-bounded-wait` finished
**M=512, N=16384, K=2048** in 5698.21 host seconds. All three variants
passed correctness (`ok=1`, `ref_ok=1`); this is not a timing pass.
The exact hardware reference is job 2119329's
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`.
The launch uses cluster=2 and tile `uni_sm90_bm256bn128bk64w4s2`.

| Variant | Hardware event time (ms) | Simulator event time (ms) | Signed time error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.111328 | 0.173841 | +56.152% | FAIL |
| Multicast | 0.111168 | 0.161919 | +45.653% | FAIL |
| No-TMA | 0.497216 | 0.455332 | -8.424% | PASS |

Raw simulator CSV: `csv/gemm_k2k__h200_gemm_compare_local.csv` under that
run directory; log: `logs/gemm_k2k.log.gz`. Timed kernel cycles are
310307/289026/812767 respectively. At that point the historical nine-event
comparison recorded 2 PASS, 4 FAIL and 3 MISSING_SIM; square is now deferred.
The timed unicast/multicast kernels report maximum residency of three
CTAs/SM, limited by shared memory/registers; no-TMA reports two, limited by
registers. Differences of cumulative `gpu_stall_dramfull` counters across
the timed kernels are 604361/253352/22761 respectively. These aggregate
stall counts are not additive wall-clock cycles or proof of the bottleneck;
they motivate checking the concurrent TMA memory path before changing bank
timings or occupancy. The opposite error directions and stronger TMA gap
at this shape do not support a global clock rescaling. No knobs changed.

Follow-up counter audit: despite its name, `gpu_stall_dramfull` increments
in `gpgpu_sim::cycle()` when an L2 subpartition cannot reserve
`SECTOR_CHUNCK_SIZE` input-queue slots. It does not directly count a full
DRAM controller, and increments even without checking for a pending ICNT
request. Existing `memory_sub_partition_full_breakdown` counters in the
same k2k log narrow the next investigation:

| Timed variant | Insufficient L2 input slots | Coincident L2 data-port busy | Coincident L2-to-DRAM queue full |
|---|---:|---:|---:|
| Unicast | 604361 | 460974 | 0 |
| Multicast | 253352 | 206631 | 1727 |
| No-TMA | 22761 | 3010 | 3281 |

These are differences of cumulative subpartition observations, not elapsed
kernel cycles; columns overlap and must not be added. The evidence points
to inspecting L2/TMA service and backpressure before attributing the GEMM
gap to HBM bank timings. It does not yet prove a service-model bug or justify
increasing bandwidth/queues. No runtime or configuration changes made.

Service-order caveat from tracing `memory_sub_partition::cache_cycle()`:
the full-state observation precedes `baseline_cache::cycle()`, which
replenishes the data port before the next cache access. A 32-byte sector hit
uses one cycle on the configured 64-byte port, so a recorded busy port can
be free by that same iteration's access check. These counts therefore do
not prove extra port-blocked cycles. Requests are split into 32-byte
sectors, and the existing service loop accepts at most one sector request
per subpartition per L2 tick. Merely widening the 64-byte port would not
increase that request rate. Diagnose request distribution and service
throughput before introducing another bandwidth knob.

The square-only extended retry `/tmp/h200-gemm-square-extended` has now
finished (exit 0, functional PASS, 9913.894 host seconds). This supersedes
its running status and the earlier square timeout, not the warmup caveat:
this process used warmup=0 and is diagnostic, not final acceptance.
M=2048, N=2048, K=8192 uses cluster=2, 128 clusters and the hardware-selected
256x128x64, four-warp, two-stage tile. Against
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`:

| Variant | Hardware CUDA-event ms | Simulator CUDA-event ms | Signed time error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.177376 | 0.301221 | +69.821% | FAIL |
| Multicast | 0.177696 | 0.244578 | +37.638% | FAIL |
| No-TMA | 0.972096 | 0.880210 | -9.452% | PASS |

Raw simulator CSV: `csv/gemm_sq_2k__h200_gemm_compare_local.csv` under that
run directory; comparator output: `nine_event_comparison/comparison.csv`
(one PASS, two FAIL, six MISSING_SIM because this directory contains only
the square shape). Multicast-versus-unicast output maximum absolute error
is 0.03125 within atol=5.66; no-TMA comparison error is zero; all 16 sampled
CPU-reference checks pass with zero error. These are tolerance-based and
sampled checks, not a proof of bitwise full-matrix equivalence. The opposite
timing-error signs again rule out a common clock adjustment as a solution.
No runtime or config knob was changed for this retry.

Square-run response-path audit (same completed log; differences from the
preceding kernel's cumulative counters):

| Timed variant | Core cycles | TMA input sectors | TMA HIT sectors | L2 input-queue shortage observations | Reply injection stalls |
|---|---:|---:|---:|---:|---:|
| Unicast | 537680 | 33554432 | 31118241 | 2069115 | 1046277 |
| Multicast | 436572 | 25165824 | 22289005 | 1292919 | 830208 |
| No-TMA | 1571175 | 0 | 0 | 0 | 0 |

TMA sector counts are 32-byte accesses (not unique data); no-TMA uses a
different access class, so its zero TMA count does not mean no memory reads.
`gpu_stall_icnt2sh` increments per memory subpartition/ICNT tick when an
actual reply cannot enter the network (`gpu-sim.cc`), whereas
`gpu_stall_dramfull` is the previously described L2 input-space observation.
Neither is elapsed kernel stall time. This supports investigating the
L2/reply service path, but does not by itself establish its causal share.

The current TMA response-width default is one token/SM/core cycle.
`simt_core_cluster::icnt_cycle` honors that limit, but the local iSLIP router
also grants at most one packet per output/ICNT tick (`iSLIP_Advance` breaks
after a grant). Its `multi_grant_reply` option relaxes the *input* limit,
not this output limit. Therefore simply setting TMA response width to four
would not create four sustained reply deliveries per SM per ICNT tick.
The completed log reports maximum reply output occupancy of one while
reply inputs reach 512; this is consistent with upstream arbitration/service
pressure rather than a backed-up SM response FIFO. Do not run a costly
response-width-only sweep or interpret a larger FIFO as additional bandwidth.
Next isolate arbitration/request distribution before changing service rates;
the hardware-matched L2 bandwidth failure remains an independent constraint.

The arbitration audit subsequently reproduced a concrete starvation defect:
with VOQs and one grant per input, continuously replenishing a low-numbered
output queue prevents a higher-output packet from ever receiving that input's
grant. `LocalInterconnectTest.VoqDoesNotStarveHigherOutput` fails on both
request and reply routers before the fix and passes afterward. The shared
`iSLIP_Advance` now rotates output scan priority each tick when VOQs compete
for a single input grant. It retains greedy matching, one grant per input
and output, and the existing multi-grant behavior; it is not a claim to
implement full iterative iSLIP or the physical H200 network. No bandwidth,
clock, FIFO, or HBM knob changed. This removes the reproduced fixed-priority
starvation; whether it reduces GEMM error is not yet measured.

All six router tests pass. The direct unit run has 213/216 passing, with
three HostTensorMap tests failing at CUDA driver initialization (error 100),
before kernel execution. The first simulator-backed run passed those tests
but failed seven source-inspection checks because the temporary cwd could
not locate the repository. After linking the repo's `src` directory into
the temporary run directory, the full 132SM simulator-backed suite passed
**216/216 tests, 26 suites, 11.974 seconds**, exit 0, recorded in
`/tmp/h200-router-unit-sim-rooted.log`. This includes the source-level global
TMA multicast/no-DSM-fabric invariant, not a new runtime traffic trace. The warmup=1,
OMP=4, hardware-matched small GEMM comparison is running in
`/tmp/h200-gemm-router-rotation`, with the existing 3M-cycle/7200-second guards.
New library SHA-256:
`814eedd3f0624cea029ecb21d15696a621aebe3391a05e041a37c76408167fb4`;
unchanged 132SM config SHA-256:
`1b4e48ac5006742c5aae36a82f20295bb02cd728c976481ce90e527484788ccc`.
Compare against `/tmp/h200-gemm-small-warm1`, not the zero-warmup baseline;
do not declare the calibration improved until completed results establish it.

The router-rotation small GEMM run has now completed: exit 0, functional
PASS in 1526.959 host seconds. All three variants report `ok=1 ref_ok=1`,
zero cross-variant maximum absolute error, and zero error on the 16 sampled
CPU-reference checks. Shape M=256, N=8192, K=2048, cluster=2, 64 clusters,
tile 256x128x64, four warps, two stages, warmup=1, samples=1, OMP=4.

| Variant | Hardware event ms | Previous warmup=1 simulator ms | Router-rotation simulator ms | New signed error | Strict <10% |
|---|---:|---:|---:|---:|---|
| Unicast | 0.056928 | 0.0674924 | 0.0567266 | -0.354% | PASS |
| Multicast | 0.055584 | 0.0650768 | 0.0501216 | -9.827% | PASS |
| No-TMA | 0.157312 | 0.130073 | 0.130120 | -17.285% | FAIL |

Hardware source remains job 2119329's `h200_gemm_compare_2119329.csv`;
simulator raw CSV is
`/tmp/h200-gemm-router-rotation/csv/gemm_cyc_1e5__h200_gemm_compare_local.csv`.
The existing comparator's `nine_event_comparison/comparison.csv` in that
directory reports two PASS, one FAIL and six MISSING_SIM. Timed kernel
cycles are 101257/89467/232265 for uni/mc/no-TMA. The router defect affected
TMA timing materially, but its fix does not close the no-TMA gap. Multicast
has only 0.173 percentage points of margin; one sample does not establish
robust acceptance. No calibration knob changed.

Both larger shapes were launched on the same `814eedd3...` library and
unchanged config: `/tmp/h200-gemm-router-k2k` (M512 N16384 K2048, 5M cycles,
14400 host seconds) and `/tmp/h200-gemm-router-square` (M2048 N2048 K8192,
8M cycles, 21600 host seconds). Each uses the hardware-selected launch,
one immediate same-variant warmup, one sample, and OMP=4. These are new
post-fix runs, not continuations of the completed zero-warmup diagnostics.
They are now terminal; the square result is historical and user-deferred.
Current six-case GEMM acceptance and frozen-build non-GEMM revalidation remain
open.

The router-fixed M512 N16384 K2048 run has completed. Its raw CSV at
`/tmp/h200-gemm-router-k2k/csv/gemm_k2k__h200_gemm_compare_local.csv`
reports `ok=1 ref_ok=1`, cross-variant maximum absolute error 0, and sampled
reference error 0.0078125 (tolerance 2.83). Launch remains cluster=2,
256 clusters, tile 256x128x64, four warps and two stages; warmup=1 and
samples=1. Comparison against job 2119329's
`h200_gemm_compare_2119329.csv` uses CUDA-event milliseconds:

| Variant | Hardware ms | Simulator ms | Signed error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.111328 | 0.168201 | +51.086% | FAIL |
| Multicast | 0.111168 | 0.155950 | +40.283% | FAIL |
| No-TMA | 0.497216 | 0.455728 | -8.344% | PASS |

The run's `nine_event_comparison/comparison.csv` contains these three rows
and six MISSING_SIM rows for shapes outside this run. The router fix does
not resolve the larger-shape TMA discrepancy. Do not infer acceptance from
functional correctness or tune the HBM roofline to compensate for this gap.
The same-build square run remains active; final all-nine acceptance is open.

The square run subsequently completed with exit 0 and functional PASS in
13440.894 host seconds. All three variants report `ok=1 ref_ok=1`, zero
cross-variant maximum absolute error, and zero sampled CPU-reference error.
Shape M2048 N2048 K8192 uses cluster=2, 128 clusters, tile 256x128x64,
four warps, two stages, warmup=1, samples=1 and OMP=4. Raw simulator CSV:
`/tmp/h200-gemm-router-square/csv/gemm_sq_2k__h200_gemm_compare_local.csv`.
Hardware source is job 2119329's `h200_gemm_compare_2119329.csv`.

| Variant | Hardware event ms | Simulator event ms | Signed error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.177376 | 0.273553 | +54.222% | FAIL |
| Multicast | 0.177696 | 0.248296 | +39.731% | FAIL |
| No-TMA | 0.972096 | 0.878405 | -9.638% | PASS |

The existing comparator generated `nine_event_comparison/` under this run.
All three router-fixed shape runs are now terminal: four of nine event
timings pass and five fail. Functional correctness passes all nine, but
this is not timing acceptance. Square no-TMA has only 0.362 percentage
points of margin and small multicast only 0.173; one sample is not robust
validation. Larger-shape TMA remains substantially slow, whereas small
no-TMA remains fast. A global clock or HBM-bandwidth fit cannot be assumed
to resolve these opposite discrepancies. No knob was changed for these runs.

Post-run service audit: the timed K2K unicast/multicast launches take
300239/278371 kernel cycles. Their cumulative L2 reservation-failure
counters remain 0->0 and 4->4 respectively: neither timed launch adds a
reservation failure. This does not support an MSHR-capacity increase as
the next fit. Reply injection stall observations increase by 102165/55349;
these are summed partition observations, not elapsed blocked kernel cycles.
Source inspection also confirms `LocalInterconnect::Push` accounts for one
packet regardless of its byte size; the local router does not serialize
flits. Thus `icnt_flit_size` is not a throughput-tuning lever in this path.
`iSLIP_Advance` grants at most one packet per output per ICNT tick, and
`simt_core_cluster::icnt_cycle` separately limits TMA response consumption.
Increasing only the response-consumption width cannot increase the router's
sustained per-output grant rate. These observations narrow the next service
diagnostic; they do not establish that either width is physically wrong.

A controlled K2K reply-arbitration diagnostic completed in
`/tmp/h200-reply-multigrant.vtkbrQ/results`. Its temporary config differs
only by `icnt_multi_grant_reply=1`, allowing an input to serve multiple
different outputs while retaining the one-packet/output/tick limit. This
is a sensitivity experiment, not an accepted H200 setting. Config SHA-256:
`9838357de6a06c073985724b04f094c9a26835577f5e397ffe81dedfa35ece6e`.
Library remains `814eedd3f0624cea029ecb21d15696a621aebe3391a05e041a37c76408167fb4`.
The harness retains the matched M512 N16384 K2048 launch, warmup=1,
samples=1, OMP=4, 5M cumulative-cycle and 14400-second host limits.
Compare against the completed `/tmp/h200-gemm-router-k2k` baseline;
do not substitute this temporary config for production acceptance.

The run exited 0 in 6255.781 host seconds; all three variants report
`ok=1 ref_ok=1`, cross-variant maximum absolute error 0 and sampled-reference
error 0.0078125 (tolerance 2.83). Final CUDA-event comparisons against
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`:

| M512 N16384 K2048 variant | Hardware ms | Diagnostic ms | Signed error | Timing gate |
| --- | ---: | ---: | ---: | --- |
| Unicast | 0.111328 | 0.168059 | +50.958% | FAIL |
| Multicast | 0.111168 | 0.164565 | +48.033% | FAIL |
| No-TMA | 0.497216 | 0.456305 | -8.228% | PASS |

Raw results are `csv/gemm_k2k__h200_gemm_compare_local.csv`; the generated
`nine_event_comparison/comparison.csv` records one PASS, two FAIL and six
MISSING_SIM (the other shapes were not part of this experiment). The harness
PASS means functional completion, not timing acceptance. Unicast barely
changes versus the baseline, while multicast becomes about 5.5% slower.
Reject this setting as a remedy for the larger TMA gaps; production config
is unchanged. This experiment does not rule out other reply-path service
limitations, but does not justify increasing reply-input grants.

Transaction-level service evidence (completed old-library diagnostic, not acceptance):
`/tmp/h200-k2k-tx-trace.BL9ylc/results` repeats production K2K with
`FLASHGPU_TMA_TRACE_CSV=/tmp/h200-k2k-tx-trace.BL9ylc/transactions.csv`
and `FLASHGPU_TMA_TRACE_MF=0`, on the same library/config as the baseline.
Exit 0 in 6837.341 host seconds; all three correctness checks pass and
event times exactly reproduce the old baseline: 0.168201/0.155950/0.455728
ms for unicast/multicast/no-TMA. Do not restart this terminal run or use
these pre-memory-fix times as candidate acceptance.
The timed unicast launch completed in exactly 300239 cycles, matching that
baseline. Its cumulative-cycle window is 1805490 through 2105729 (kernel
uid 13). Joining READ events by transaction UID within this window gives:

| Transfer bytes | Completed reads | Mean NEW→first issue cycles | Mean first→last issue cycles | Mean last issue→COMPLETE cycles | Mean COMPLETE→ARCH_ARRIVE cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 32768 | 32.1343 | 185.063 | 2518.34 | 1.00339 |
| 16384 | 16384 | 2.29596 | 409.210 | 3334.39 | 1.09650 |

Maximum COMPLETE→ARCH_ARRIVE delays are 38/192 cycles respectively.
Mean first-response→COMPLETE intervals are 2310.97/3361.73 cycles; these
overlap the issue intervals and must not be added as independent phases.
`src/gpgpu-sim/flash/tma.cc` emits COMPLETE when memory service completes;
ARCH_ARRIVE follows both memory completion and the configured latency floor.
Thus the average post-memory notification delay is negligible here. The
post-last-request service tail, not initial enqueue delay or a large extra
completion-floor wait, is the main transaction interval to investigate.
These are transaction means, not exclusive kernel stall cycles; they do
not yet identify the responsible memory/reply stage or justify a knob change.
Timed multicast has also completed in exactly 278371 cycles, matching its
baseline (UID 17, cumulative window 2404675 through 2683046):

| Transfer bytes | Completed reads | Mean NEW→first issue cycles | Mean first→last issue cycles | Mean last issue→COMPLETE cycles | Mean COMPLETE→ARCH_ARRIVE cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 16384 | 38.8159 | 201.006 | 2298.91 | 100.000 |
| 16384 | 16384 | 1.74268 | 303.093 | 2664.21 | 23.7799 |

Maximum COMPLETE→ARCH_ARRIVE delays are 100/224 cycles; mean
first-response→COMPLETE intervals are 2084.89/2587.85 cycles. The explicit
multicast fan-out completion path retains at least
`gpgpu_tma_multicast_latency=100` cycles after memory completion
(`tma.cc`, `finalize_transaction`); do not generalize the unicast ~1-cycle
observation to multicast. The service tail is still substantially larger.
No-TMA and final correctness checks remain pending; do not yet treat the
whole diagnostic as a completed run.

Reply-path source audit found no proven TMA-only missing sector merge.
L2 splits a 128-byte TMA parent into four independent 32-byte replies;
ordinary SM90 LDG coalescing already uses 32-byte segments. Both traverse
the same reply network, while ordinary loads can additionally hit L1.
TMA parent credit retires after all child payload returns. Request routing,
L2 service, reply routing and SM consumption are pipelined stages, not
independent multiplicative bandwidth penalties.
An existing-hook follow-up can correlate `FLASHGPU_L2_TRACE_CSV` with
`FLASHGPU_L2_TRACE_CACHE_ACCEPT=1`, `FLASHGPU_L2_TRACE_TMA_ONLY=1`, and
TMA per-MF tracing. L2 `orig_uid` joins the TMA parent `mf_uid`; L2 child
UIDs remain distinct. REQ→CACHE_ACCEPT includes fixed/input delays;
CACHE_ACCEPT→RESP includes cache/memory service and output queuing;
RESP→MF_RESPONSE includes reply/consumer delays. RESP is emitted at
successful ICNT injection, not initial memory completion. TMA MF_RESPONSE
uses the parent address, so final-stage correlation is parent-level.
Existing hooks have only inclusive maximum cumulative-core-cycle filters,
not minimum-cycle/SM/kernel filters. TMA cutoff 0 is unlimited, whereas
L2 cutoff 0 is not. Avoid an unbounded all-run per-MF trace; if using a
baseline-derived cutoff, verify the actual new-run launch boundaries and
reject incomplete transaction evidence. No further trace has been launched
for this service-stage split yet.

Existing K2K baseline per-L2-bank counters also show concentrated demand.
Subtract UID12 from UID13 for timed unicast and UID16 from UID17 for timed
multicast (188 banks in each log):

| Timed variant | Total accesses | Minimum / mean / maximum per bank | Maximum / mean |
| --- | ---: | ---: | ---: |
| Unicast | 17301504 | 41264 / 92029.28 / 193968 | 2.108 |
| Multicast | 13107200 | 22512 / 69719.15 / 171376 | 2.458 |

The busiest banks in both launches are 49, 53, 47, 51, 55, 58. Bank 49
has 16544/15500 misses (8.53%/9.04% of accesses) and no new reservation
failures. Its access counts divided by estimated L2 ticks
(`kernel_cycles * 1700/1785`) are 0.6783/0.6464, respectively. These are
whole-launch average access rates, not measured busy fractions or proof of
saturation: service can be bursty and miss/output stalls are not captured
by this ratio. Aggregate bandwidth conceals this uneven demand; investigate
the general address mapping and service stages rather than choosing a
GEMM-specific hash or increasing capacity solely to fit these shapes.

The subsequent decoder audit establishes a real geometry defect, separate
from the bank-demand inference. `/tmp/h200-hash-local.XaPdoS/check.cc` links
the repository's `addrdec.cc`, `hashing.cc`, and `option_parser.cc`, using
the production address mask, `dramid@9` mapping, IPOLY indexing and 94x2
geometry. Both `0x000` and `0x200` decode to chip=0, bank=0, row=0,
column=0, burst=0, subpartition=0 and partition_address=0. These are
distinct byte addresses, not two sectors of the same address.

| Decoder mode, 1 MiB swept in 32-byte steps | Unique complete DRAM tuples | Collisions |
| --- | ---: | ---: |
| Consecutive indexing control | 32768 | 0 |
| IPOLY balanced=0 | 27968 | 4800 |
| IPOLY balanced=1 | 16544 | 16224 |
| Production IPOLY balanced=2 | 6480 | 26288 |

The quotient/remainder channel decomposition removes original channel bits
from intra-channel coordinates; subsequent many-to-one hash reduction loses
channel identity without preserving it in row/column coordinates. Production
linear DRAM bank indexing consumes those decoded coordinates unchanged.
This can distort bank/row locality. L2 tags and functional memory still use
original addresses, so it is not by itself evidence of wrong numerical data.
It is more than an ordinary cache-set collision.

The same diagnostic shows local distribution is not guaranteed by the
1024-to-188 global range balance: with fixed channel quotient q, only 35–36
destinations are used across all 188 decoded seeds (q=0..4095). The top two
virtual-index bits are fixed within q, confining outputs to at most a
47-subpartition quadrant, with actual coverage narrower.

The shared decoder source now replaces all three non-power-of-two reduction
modes with a bijective cyclic permutation:
`destination = (decoded_seed + IPOLY1024(quotient, 0)) % destination_count`.
For a fixed quotient, every decoded channel/slice seed has a unique
destination. Original bank/row/column coordinates retain their identity
instead of being merged by range reduction. Power-of-two and channel-stable
paths are unchanged; legacy `gpgpu_ipoly_non_power2_balanced=0/1/2` values
remain accepted as equivalent compatibility settings. No new tuning knob
or GEMM-shape dependency is introduced. This is a geometry-consistency
repair, not a claim to reproduce NVIDIA's undocumented address hash.

Permanent standalone regression: `test/check_address_mapping.cc` (build/run
command in its header) uses the actual decoder and tests the alias pair,
1MiB/32B full-DRAM and `(subpartition, partition_address)` uniqueness, and
4096 fixed-quotient seed sweeps for 94x2/80x2, all legacy modes and both
channel-stable settings. All 12 combinations pass; main-agent verification
is `/tmp/h200-map-regression.L4snoW/result.log`. The old decoder fails the
non-stable controls. The separate candidate build and completed runtime
regressions are listed at the top of this report; final calibration must
still be remeasured. Baseline diagnostics retain their original library.
The regression was additionally extended to the actual 64-bit heap base
`GLOBAL_HEAP_START=0xC00000000` (48GiB), rather than testing only addresses
near zero. It now sweeps 1MiB/32B at both bases and requires 65536 unique
complete DRAM tuples and partition-address pairs in their union. All 12
combinations pass (`/tmp/h200-map-regression.L4snoW/high-address.log`).

Caller audit also exposed a distinct cache-preload width defect:
`gpgpu_sim::perf_memcpy_to_gpu(size_t,size_t)` used `unsigned` for both its
loop offset and computed destination. Since the heap already starts above
4GiB, truncation is relevant to normal calibration allocations, not only
future huge buffers. Both functional H2D and D2H helpers call this timing
preload routine; downstream decoding and `handle_memcpy_to_gpu(size_t,...)`
can preserve full addresses. Functional memory copies themselves remain
64-bit. The routine now uses `size_t` for both loop offset and destination;
`test/check_memcpy_preload.py` now checks the first isolated high-address
preload's exact 128 sector addresses. Its module documentation gives the
short existing 4KiB L2 benchmark, TRACE=1 build requirement and temporary
132SM trace settings. Synthetic valid/missing/short/truncated/reordered
trace checks pass, including under Python `-O`. These synthetic cases test
the checker; the >4GiB copy-length loop is not exercised.
The real failing-before control has now completed in
`/tmp/h200-preload-before.bl6073/run.log`, exit 0, using the unchanged
`814eedd3...` library, OMP4, and a temporary 132SM config enabling existing
copy-engine tracing (100K-cycle/180-second host limits). The existing L2
benchmark used 4096 bytes, four steps, one sample and no warmup. Its first
preload addresses are `0, 0x20, 0x40, ...`, not `0xC00000000, ...`.
The high-address checker exits 1 as expected. Reported net latency is
429.250 cycles/load; this four-step diagnostic is not a hardware calibration
point. The passing-after control is now recorded in the current candidate
section at the top of this report.
Do not confuse cache warming at a truncated
address with validated hardware cache state, or combine pre/post-fix timing
rows into one final calibrated batch.

Separate inherited capacity limitation: the current H100/H200 address mask
has only 13 row bits; high intra-channel address bits are not extended by
the DRAM model. The H200 coordinate space represents at most 23.5GiB of byte
positions, not the product's 141GB. IPOLY can redistribute high bits across
channels, so 23.5GiB is not a universal full-tuple repetition period.
Functional memory retains 64-bit addresses; this is a timing-coordinate
capacity limitation, not evidence of functional wrap. The bijective repair
does not fix the finite row mask. Device memory reports are also independent
placeholders (2GiB runtime property, 10GB cudaMemGetInfo, 20GB driver total).
Review full-capacity geometry/reporting separately; do not hide this limit
or claim that the present small-buffer calibration validates all 141GB.

No-TMA service follow-up: subtracting cumulative Warp Stall Breakdown for
launch UID 20 from timed UID 21 in each router-fixed baseline log gives:

| Warp-scheduler sample category | M256 N8192 K2048 | M512 N16384 K2048 | M2048 N2048 K8192 |
| --- | ---: | ---: | ---: |
| SB_SpInt | 67.08% | 41.36% | 42.75% |
| MathPipeThrottle | 1.67% | 34.24% | 33.76% |
| SB_MemGlobal | 11.87% | 8.83% | 7.68% |
| WaitWGMMA | 6.06% | 4.97% | 5.47% |
| WaitTMA | 0.33% | 0.29% | 0.07% |

Denominators are 113037167 / 816589333 / 1562981790 scheduler samples.
These shares are not elapsed-cycle fractions. `SB_SpInt` identifies a
pending producer class, not the stalled instruction or exact producer PC;
the scoreboard classification includes RAW and WAW dependencies. Small
no-TMA is more dependency-limited in these samples, while larger launches
show more math-pipeline contention. This supports investigating the repeated
instruction stream, not an arbitrary fixed launch/epilogue adjustment.
The next diagnostic can reuse `FGSIM_ISSUE_TRACE_FILE`,
`FGSIM_ISSUE_TRACE_SM=0`, `FGSIM_ISSUE_TRACE_WARP=0`, and
`FGSIM_ISSUE_TRACE_MAX=0` on the small matched case. Filter cumulative cycles
using that run's launch20/21 boundaries; positive line limits may be consumed
by earlier smoke launches. Inspect `STALL_SCOREBOARD prod=SP_INT` PCs and
surrounding ISSUE rows. Waiting warps are absent from this trace, so gaps
alone cannot classify WGMMA/barrier waits. No new instrumentation is needed.
That diagnostic has completed at `/tmp/h200-small-issue-trace.9OsQho/results`
with `/tmp/h200-small-issue-trace.9OsQho/issue.log`. It reuses the pinned
router-fixed library and production config through the unchanged harness
(`--only gemm_cyc_1e5 --skip-build`), retaining OMP4, warmup=1, samples=1,
the 3M cumulative-cycle cap and 7200-second host cap. Exit 0 in 1918.225 host
seconds; all three variants report `ok=1 ref_ok=1 max_abs=0 ref_abs=0`.
Unicast/multicast/no-TMA event times are 0.0567266/0.0501216/0.130120 ms,
exactly reproducing their prior baseline (101257/89467/232265 kernel cycles).
The comparator records two PASS, one FAIL and six MISSING_SIM (other shapes
not run here). This remains an old-library diagnostic, not acceptance of
the unbuilt mapping/preload repairs.

Completed timed no-TMA interval: cumulative cycles 1179514–1411779,
launch UID21, traced SM0/warp0/dynamic-warp33. It has 148158 SP_INT
scoreboard-stall trace rows across 299 PCs. Top PCs include
`0x4bab0`, `0x4bac0`, `0x4bad0`, `0x4bae0`, `0x4baf0`, each with 1984
rows, all ordinary `ld.global.b16` instructions. Among 4032 adjacent issued
zero-MOV→matching-destination-LD pairs, 1836 have 63-cycle issue gaps,
1044 have 7-cycle gaps and 1002 have 5-cycle gaps; the rest span 6–53 cycles.
These are one warp's trace events/pairs, not chip-wide exclusive stall times.
The completed trace confirms the variable queued zero-MOV/WAW behavior
below; it does not justify a fixed MOV-latency adjustment.
Partial smoke evidence (not the timed launch): SM0/warp0/dynamic-warp9
shows repeated zero-MOV→LD WAW stalls, e.g. MOV at cycle432075 and the
following LD at432138. The same stream also has 5/7-cycle gaps, so this
does not establish a fixed 63-cycle MOV latency. MOV remains latency/init
1/1; scoreboard reservation begins at scheduler issue and ends at writeback.
Generic operand collectors are shared by memory/integer instructions and
retain loads until the memory output slot is available. Memory backpressure
can therefore present as an SP_INT producer stall on a queued zero-MOV.
The exact queue stage still needs observation; do not change MOV latency or
bypass WAW dependencies based on the scheduler label. Small no-TMA is already
too fast, so speeding that path is not a supported calibration remedy.

Routing-verification follow-up: `FLASHGPU_DSM_STATS=1` reuses endpoint
payload counters with explicit enabled-state and complete-cluster framing.
`test/check_dsm_tma_route.py` rejects disabled/missing/truncated evidence,
requires zero payload in global mode, and requires mapped TMA packets in
the separate positive control. Validation remains active under Python `-O`;
both normal and optimized self-tests pass. The separate candidate library
now passes isolated global-zero and mapped-positive runtime controls (see
the current candidate section); explicit GEMM routing remains under test.
Require successful non-skipped functional tests independently of counters.

Historical warmup sensitivity diagnostic (the current harness now uses one
immediate GEMM warmup): the
profiling source `/home/jcliu/H200_profiling/src/probe_gemm_triton.cu`
hardcodes `warm=1` in the comparison suite, despite the job log's generic
`samples: 4 warmup: 8` header. The simulator mirror honors `opt.warmup`,
and the earlier reduced harness used zero. Both place CUDA events around the GEMM
launch only, outside the globaltimer stamp kernels. Earlier smoke launches
exercise all three variants but do not establish identical cache state to
an immediate same-variant warmup. A diagnostic small-shape run was launched
in `/tmp/h200-gemm-small-warm1` by reusing the harness with only that case's
arguments changed to `--warmup 1` (one sample, OMP=4, 3M cycle ceiling,
14400-second host allowance). Kernel/config/library are unchanged; gcc-13.3.0
release library is explicitly pinned. Compare its completed event times
against the zero-warmup run before attributing sensitivity to model knobs.
The source inspection establishes a procedure difference, not its magnitude
or proof that it explains any existing timing failure.

The warmup sensitivity run has now completed, functional PASS in 2340.399
host seconds, with `ok=1 ref_ok=1 max_abs=0 ref_abs=0` for all three
M=256, N=8192, K=2048 variants. Its raw CSV is
`/tmp/h200-gemm-small-warm1/csv/gemm_cyc_1e5__h200_gemm_compare_local.csv`;
the existing comparator generated `nine_event_comparison/comparison.csv`
in that run directory against job 2119329's GEMM CSV.

| Variant | Hardware event ms | Simulator warmup=1 event ms | Signed time error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.056928 | 0.0674924 | +18.557% | FAIL |
| Multicast | 0.055584 | 0.0650768 | +17.078% | FAIL |
| No-TMA | 0.157312 | 0.130073 | -17.315% | FAIL |

Warmup=0 gave 0.0660353/0.0600779/0.131175 ms respectively. Thus one
immediate warmup does not repair the discrepancy, and multicast's former
8.085% pass is not robust to the hardware-like warmup procedure. Preserve
both runs, do not select the passing procedure opportunistically. Final
GEMM validation should retain one immediate warmup unless equivalence is
demonstrated. The square retry still uses warmup=0 and remains diagnostic
for this procedure question. No timing knobs changed.
Future harness GEMM cases now use `--warmup 1`; the self-test checks that
all GEMM comparison cases retain it and other H200 probes remain at zero.
Existing processes retain their original arguments. Cycle ceilings and
host guards are unchanged; final acceptance still requires complete runs.

Per-kernel counter audit of the completed warmup=1 small-shape log further
limits what can be inferred about HBM. Subtracting the preceding kernel's
cumulative counters gives the following for the timed launches (not smoke
or warmup launches). Read accesses are L2 sector-access observations, not
unique input bytes; `HIT_RESERVED` is excluded from the hit percentage.

| Timed variant | Kernel cycles | Input-read L2 accesses | L2 HIT | HIT fraction | L2 input-slot shortage observations |
|---|---:|---:|---:|---:|---:|
| Unicast | 120474 | 4194304 | 3965367 | 94.542% | 155440 |
| Multicast | 116162 | 3145728 | 2983124 | 94.831% | 126180 |
| No-TMA | 232181 | 4194304 | 3993611 | 95.215% | 786 |

Unicast/multicast use `TMA_ACC_R`; no-TMA uses `GLOBAL_ACC_R`. The source is
`/tmp/h200-gemm-small-warm1/logs/gemm_cyc_1e5.log.gz`, with GEMM launch order
smoke uni/mc/no-TMA, then warmup/timed uni, warmup/timed mc, warmup/timed
no-TMA. Their immediate warmup launches took 113143/111442/234416 cycles;
the first two had 230300/312113 input-slot shortage observations. Thus the
timed TMA launches are slower despite fewer such observations. Aggregate
queue pressure alone is not an elapsed-time explanation. These high L2
hit fractions also make this an unsuitable isolated HBM timing fit, though
the remaining misses may still affect critical paths. Investigate TMA/L2
service and launch scheduling before changing inherited HBM timings. No
runtime or configuration change follows from this counter audit.

A controlled small-GEMM diagnostic is running under
`/tmp/h200-gemm-no-floor.mySYj2/results`, reusing the harness with one
immediate warmup, one timed sample, OMP=4 and the 3M cumulative cycle guard.
Its temporary copy of the 132-SM configuration changes only the noncluster
and cluster TMA completion base settings from 1314/860 to zero; zero disables
their added architectural floors, while actual memory-response completion
remains mandatory. The production configuration is unchanged. This is an
ablation to measure the floors' contribution to the timing gap, not an
acceptable calibrated configuration or evidence to discard the latency
probe fits. Compare all three event times and correctness against the
completed `/tmp/h200-gemm-small-warm1` baseline before drawing conclusions.

The no-floor diagnostic completed in 2114.451 host seconds, exit 0 and
functional PASS. Its `nine_event_comparison/comparison.csv` compares the
exported event times against
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`:

| M=256 N=8192 K=2048 variant | Hardware event ms | No-floor simulator event ms | Signed time error | Strict <10% |
|---|---:|---:|---:|---|
| Unicast | 0.056928 | 0.066763 | +17.276% | FAIL |
| Multicast | 0.055584 | 0.0680527 | +22.432% | FAIL |
| No-TMA | 0.157312 | 0.130189 | -17.242% | FAIL |

All rows report `ok=1 ref_ok=1`; the raw notes report `max_abs=0.0469`,
`ref_abs=0`, `atol=2.83`, so this is tolerance-based correctness, not a
bitwise-equality claim. Compared with the enabled-floor warmup=1 baseline,
unicast improves only about 1.1%, multicast worsens about 4.6%, and no-TMA
changes by less than 0.1%. This does not support removing the calibrated
TMA floors to close the GEMM gap. Keep the production settings and inspect
concurrent service/scheduling instead. The diagnostic configuration SHA-256
is `689ce3607fcde5b0bde2f0f44562da3f05736565fbc928c7ab2b820fed373c91`;
the pinned gcc-13.3.0 simulator library SHA-256 is
`cadf20175b977a9ab56940238e456154a2b083db904e2e2bedf2dca8b17273ed`.
The full nine-event gate remains 3 FAIL and 6 MISSING_SIM for this isolated
small-shape diagnostic, not a nine-case validation.

Current-library regression recheck: all 23 tests from 12 suites selected by
`VectorOperandIntegrationTest.*:*LdMatrix*:*StMatrix*:MMAF16*` pass on
`SM90_H200_CLUSTER132`, OMP=4, in 53.678 seconds (exit 0). Log:
`/tmp/h200-arithmetic-current-recheck.log`. This reuses the existing SM120
integration executable under the runner's explicit CC-mismatch allowance;
it checks the selected PTX vector/matrix arithmetic paths, not SM120 timing
fidelity or the full test suite. Executable SHA-256:
`0b9fb76b281c7b934a16b45f70b35135af3f427156a61b166a5c073f5223006d`.
The loaded gcc-13.3.0 library is the same `cadf2017…` build recorded above;
no rebuild or production configuration change was needed.

The harness no longer caches simulator-library hashes across a batch.
New case logs and result records retain their own library hash inventory;
the self-test replaces a library between reads to verify invalidation.
Run-level metadata describes the initial snapshot, not an assertion that a
mixed-revision batch is coherent. Existing processes retain their original
harness behavior; final acceptance requires a stable simulator/config run.

The DSM store-plus-fence gap exposed a separate ordering defect:
`fence.sc.cluster` was not classified as a timing memory barrier, and the
existing barrier release checked only register scoreboard writes. The new
`DsmTest.ScFenceWaitsForRemoteStore` reproduces completion after 20 cycles
despite a configured 245-cycle remote-store delivery floor
(`/tmp/h200-dsm-fence-before.log`). `fence.sc.*` now uses the shared memory
barrier path, which additionally waits for outstanding stores and per-warp
DSM operations; proxy/tensormap fences retain their separate classification.
This enforces ordering without fitting an artificial fixed fence latency.
The regression passes after the fix (`/tmp/h200-dsm-fence-after.log`).
See the [PTX fence specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).
Broader DSM regressions are running in `/tmp/h200-dsm-fence-regression.log`.
The vendor diagnostic in `/tmp/h200-dsm-store-fenced` uses a two-CTA matrix
to save time: only its fixed two-CTA store-visibility rows are matched to
hardware; its other matrix/topology rows are not acceptance evidence.
Both validations subsequently completed: all 13 selected DSM regressions
pass, and the vendor producer store-plus-fence span is 665 vs 660 cycles
(0.758% high), with peer visibility 375 vs 352 ns (6.534% high). Converted
visibility is 669.375 vs 625.177 cycles (7.070% high); ns and cycles are
two representations of the same measurement, not independent passes.
`/tmp/h200-dsm-store-fenced/store_comparison/comparison.{csv,md}` contains
only these matched store rows and their hardware provenance. This closes
the previously large store gaps without changing a latency knob. Local
shared-memory and remote-mbarrier gaps remain open.

The simulator launcher sets `FLASHGPU_SIM_CLOCK_FROM_PROP=1`
so the mirrored device probe uses the fixed configured clock for subsequent
unit conversions. Its shortened busy-loop clock estimate remains diagnostic
only: kernel launch/drain overhead otherwise biases the inferred frequency.
Eight hang-preventer/multicast-fabric unit checks and the existing intentional
barrier-deadlock regression also pass, including the latter on 132 SMs.

To run every listed case, including the long sweeps, cycle gate, and GEMM:

```bash
python3 scripts/run_cluster_noc_demo.py --profile exhaustive --exclude none
```

Use `--resume` after a fix. A pass is reused only when the config, binary,
arguments, launcher, simulator libraries, harness/metric parser, cycle ceiling,
and OMP setting have the same fingerprint.
`report.md` and `suite.log` are supervisor-facing;
`results.csv`, `results.json`, and `metrics.csv` are machine-readable. A
cycle-limit stop is reported as `LIMIT`, never as a passing measurement.

### Triton GEMM functional smoke (2026-09-05)

The 256x256x64 `--gemm-smoke` case now runs the shipped
`bm256bn128bk64w4s2` cubins under `SM90_H200_CLUSTER132` with
`OMP_NUM_THREADS=4`. G1 unicast TMA and G3 no-TMA both complete; their full
outputs match exactly and 16 CPU-reference samples pass. This is functional
evidence only: functional-mode CUDA event/globaltimer values are not
calibration timings.

G2 multicast remains enabled for timing-mode calibration, but is explicitly
skipped by this functional smoke. Functional CTAs execute sequentially and do
not expose a live peer CTA shared-memory allocation, so a multicast result
would not be meaningful. A timing-mode G2 run is still pending.

```bash
PTX_SIM_MODE_FUNC=1 OMP_NUM_THREADS=4 CALIB_TIMEOUT_S=90 \
  CALIB_MAX_CYCLES=1000000 scripts/run_calibration_sim.sh \
  --config SM90_H200_CLUSTER132 --run-dir /tmp/h200-gemm-smoke -- \
  calibration/kernels/h200_probes/nvprof --suite gemm_compare \
  --gemm-smoke --samples 1 --warmup 0 --csv yes \
  --csv-file /tmp/h200-gemm-smoke/results.csv
```

Post-fix regression evidence from the same date: 27/27 cluster/topology and
7/7 hang-preventer unit tests, 31/31 SM120 cluster integration tests (plus
three topology skips), and 26/26 reduced-H200 DSM/remote-mbarrier/TMA tests
pass. The fresh 132-SM suite reports 32 PASS and five intentional SKIP
results. The Triton smoke was rerun on the final library: G1 and G3 completed
with exact output agreement and all 16 CPU-reference samples passing.

## 3. Authoritative hardware workflow

The execution contract is the current sibling suite:

- `../H200_profiling/run_h200.sbatch`: exclusive, isolated one-shot job.
- `../H200_profiling/H200_ONE_SHOT_AUDIT.md`: measurement and validation audit.
- `../H200_profiling/TODO.md`: suite-side checklist.
- `calibration/kernels/h200_probes/`: synchronized local copy; never edit it as
  a second implementation.

The synchronized snapshot is pinned to:

- `H200_profiling` commit `c880781eb1bb5a2864308f8ca33d0dfd8673fab4`.
- `seanzw/random` DSM and TMA commit
  `4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd`.

`SOURCE_MANIFEST.json` records these revisions in each synchronized kernel
tree, and the simulator `results.json` records that manifest plus config,
harness, launcher, binary, and run fingerprints.

The job records `kernel_origin` and `kernel_source` before every suite. A suite
that fails, times out, skips an expected measurement, or reports a validation
failure is not calibration data.

## 4. Vendor-first kernel policy

Use maintained code whenever it already measures the required quantity:

| Area | Canonical implementation | Policy |
|---|---|---|
| DSM bandwidth and topology | `vendor/dsm_bw` (`seanzw/random`) | Use all vendor configurations and its processor |
| Global-memory TMA bandwidth | `vendor/tma_bw` (`seanzw/random`) | Use separate L2-resident and cold-HBM binaries |
| ALU, memory, `cp.async`, TMA, mbarrier, MMA, WGMMA | exact `test/src/microbench` mirror | Run all 26 mirrored binaries |
| Device clock, HBM STREAM | `H200_profiling` | Keep: not covered by the vendor suites |
| DSM dependent latency/store visibility/contention | `H200_profiling` | Keep only these vendor gaps |
| TMA multicast and remote mbarrier | `H200_profiling` | Keep: vendor suites do not cover them |
| End-to-end GEMM | `H200_profiling` Triton/Gluon G1/G2/G3 | Keep as the application validation gate |

The older in-house DSM bandwidth/topology and generic MMA/WGMMA measurements
were not part of job 2119329 because the vendor or exact mirrored
implementations superseded them.

### Hardware-to-simulator mapping

| Hardware output group | Simulator case(s) |
|---|---|
| `device` | `device` |
| `dsm_bw` | `vendor_dsm` |
| `tma_l2`, `tma_hbm` | `vendor_tma_normal`, `vendor_tma_cp_async`, `vendor_tma_tma` at the matching size/cache condition |
| `flashgpu_microbench` | the 26 `flashgpu_microbench` cases |
| `hbm` | `hbm` |
| `dsm_calibration` | `dsm_calibration` |
| `tma_multicast` | `tma_multicast` |
| `cycle_gate` | `cycle_gate` |
| `mbarrier` | `mbarrier` and `mbarrier_trywait` |
| `gemm_compare` | `gemm_compare` |

The comparator joins canonical CSV rows by the exact
`(suite, metric, sim_knob, unit)` tuple. Missing or duplicate mappings are
errors rather than guessed matches:

```bash
python3 scripts/compare_h200_calibration.py \
  --hardware /path/to/h200_*.csv \
  --sim calibration/results/full_scale_calibration_final/metrics.csv
```

For the three vendor HBM drivers, the comparator also accepts the original
`h200_tma_hbm_2119329.txt` directly. It selects only HBM-streaming, 1 GiB,
16 producer warps, 16 stages, 8192-byte chunks and repeat=1; L2-resident and
other sweep points are not interchangeable targets. Hardware bytes/cycle
is logical bytes divided by the vendor-reported cycle count; this is an
aggregate timing result, not a median of raw samples. Simulator bytes/cycle
uses `gpu_sim_cycle`, including its modeled launch overhead, as metric notes
state. Check this measurement-envelope difference before tuning latency.

Use the dedicated event selection for the required three GEMM shapes and
three variants (nine comparisons). It validates M/N/K and millisecond units,
requires all nine hardware references, and reports missing simulator rows
as `MISSING_SIM`. Derived globaltimer cycle estimates, TFLOPS, correctness
flags, other shapes and the failed 4096-cubed case are not timing-gate rows.
Still check the harness's functional results and recorded launch settings;
shape metadata alone does not prove launch or binary equivalence.

```bash
python3 scripts/compare_h200_calibration.py --gemm-events-only \
  --hardware /home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv \
  --sim /tmp/h200-gemm-router-rotation/metrics.csv \
        /tmp/h200-gemm-router-k2k/metrics.csv \
        /tmp/h200-gemm-router-square/metrics.csv \
  --output-dir /tmp/h200-router-all-nine-comparison
```

`--sim` accepts one or more CSVs and retains each metric's source path.
Duplicate compared metric keys across files are rejected, not overwritten;
the self-test exercises multi-file CLI ingestion and duplicate rejection.
Use only runs with compatible config, simulator/kernel binaries and measurement
procedures; the comparator does not establish that compatibility for you.
The three runs above record matching production config and gcc-13 library
hashes (listed earlier). Their combined comparison has nine rows, four PASS,
five FAIL, and no missing rows; exit 1 is expected. Both CSV and Markdown
are in the specified output directory. This is a consolidated baseline,
not the final calibrated result. Acceptance still requires all nine to pass
the strict less-than-10% timing gate and functional validation to pass.

```bash
python3 scripts/compare_h200_calibration.py \
  --hardware /home/jcliu/H200_results/job_2119329/h200_tma_hbm_2119329.txt \
  --sim /tmp/h200-calibration-hbm-matched/metrics.csv \
  --output-dir /tmp/h200-calibration-hbm-matched/comparison
```

Run this command after the matched HBM suite finishes. The adapter's
self-test rejects wrong memory-residency, producer-count, stage and dataset
variants. The real job produces exactly three selected hardware targets.

## 5. H200 job 2119329

`run_h200.sbatch` executes these isolated groups in order:

| Order | Group | Required output |
|---:|---|---|
| 1 | device clock and launch baseline | `h200_device_<job>.{txt,csv}` |
| 2 | vendor DSM bandwidth/topology | `h200_dsm_bw_<job>.{txt,jsonl}` and processed analysis |
| 3 | vendor TMA L2 bandwidth | `h200_tma_l2_<job>.txt` |
| 4 | vendor TMA HBM bandwidth | `h200_tma_hbm_<job>.txt` |
| 5 | 26 exact FlashGPU-Sim microbenchmarks | `h200_flashgpu_microbench_<job>.txt` |
| 6 | correctness-checked HBM STREAM | `h200_hbm_<job>.{txt,csv}` |
| 7 | unique DSM calibration gaps | `h200_dsm_calibration_<job>.{txt,csv}` |
| 8 | TMA multicast size/fanout/skew | `h200_tma_multicast_<job>.{txt,csv}` |
| 9 | approximately 100K-cycle twins | `h200_cycle_gate_<job>.{txt,csv}` |
| 10 | local and remote mbarrier | `h200_mbarrier_<job>.{txt,csv}` |
| 11 | G1 unicast, G2 multicast, G3 no-TMA GEMM | `h200_gemm_compare_<job>.{txt,csv}` |

## 6. Initial accepted knob updates

Only validation-clean rows from job 2119329 are used. Values marked H100 are
same-Hopper fallbacks because the job did not isolate that quantity; they are
not H200 measurements.

| Area | `SM90_H200_CLUSTER132` value | Provenance |
|---|---|---|
| Kernel launch | `8700` cycles | Job median: 8698.67 cycles |
| Local mbarrier | arrive `6`, successful try-wait `43` | Job medians; unchanged |
| `cp.async` issue | latency/initiation `4/4` | 65 cycles / 16 copies |
| `cp.async` commit/wait/release | commit `7/7`, wait `5/5`, release `5` | H100 fallback; not isolated |
| FP16 WGMMA SS issue/completion | issue `4,4,4,4`; completion `47,51,59,75` | Job shape/group sweeps |
| FP16 WGMMA RS issue/completion | issue `9,9,9,10`; completion `37,41,49,65` | Job shape/group sweeps |
| TMA multicast fixed delay | `100` cycles | Robust job median ~99 cycles |
| INT DIV | latency/initiation `21/2` | H100 fallback; not measured |
| FP DIV | latency/initiation `39/2` | H100 fallback; not measured |
| DP tuple | latency `64,64,64,64,330`; initiation `64,64,64,64,130` | H100 fallback; no DP probe |
| TMA pure issue | `32/32` | H100 fallback; job measured completion only |
| ICNT/L2 clocks | `1700/1700` MHz | H100 fallback; unmeasured |
| Far-L2 penalty | `150` cycles | H100 fallback; per-offset distribution absent |
| DRAM residual | `254` cycles | H100 fallback; cold dependent miss absent |

The H200 hard-spec core/DRAM clocks and memory geometry remain 1785/3201 MHz,
94 memory channels, and 188 L2 subpartitions. The measured 1776.69 MHz is an
observed operating clock, not a replacement for the product clock.

## 7. Acceptance and update procedure

1. Confirm the top-level log ends with `failed_suites: none` and
   `timed_out_suites: none`.
2. Confirm every expected per-suite file includes the current job ID.
3. Reject rows with failed correctness, source-manifest, opcode, or timing
   validation.
4. Import the new hardware values with `scripts/compare_h200_calibration.py`
   and add its validation-clean comparison table to this report.
5. Re-run the retained simulator calibration set on `SM90_H200_CLUSTER132`.
6. Fit only knobs constrained by the corresponding kernel; do not retune DSM
   fabric parameters from HBM STREAM or GEMM alone.
7. Update `SM90_H200_CLUSTER132` comments with the new job ID and label each
   fitted value `measured` or `inferred`. Other H200 presets are
   development-only and will be removed.

Until the matching simulator rows and corrective GEMM rerun pass, these are
initial calibration values rather than a frozen published preset.

## 8. K512 occupancy diagnostic (2026-09-08)

This diagnostic is not an acceptance shape. It isolates grid occupancy from
total GEMM work while `sq_2k` is user-deferred. The kernel source is
`calibration/kernels/h200_probes/python/gemm_tma_mc.py`, executed through
`src/probe_gemm_triton.cu` on `SM90_H200_CLUSTER132` (132 SMs,
`OMP_NUM_THREADS=4`). Simulator measurements use one smoke launch plus one
timed launch per variant (`--samples 1 --warmup 0`); repeated machine-style
warmups are deliberately omitted.

Hardware values come from job 2119329,
`/home/jcliu/H200_results/job_2119329/h200_gemm_compare_2119329.csv`, rows
94–108. Simulator evidence is
`/tmp/h200-gemm-k512-complete/run/h200_gemm_compare_local.csv`; the complete
human-readable log is `/tmp/h200-gemm-k512-complete/launcher.log`.

| Variant | What it tests | Sim event (ms) | H200 event (ms) | Event error | Sim TFLOP/s | H200 TFLOP/s | Sim kernel cycles | H200 globaltimer cycles | Correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1 unicast TMA | per-CTA TMA A/B loads + WGMMA | 0.0454241 | 0.040544 | +12.04% | 189.105 | 211.867 | 81,082 | 81,332.8 | PASS |
| G2 multicast TMA | cluster B multicast + WGMMA | 0.0447406 | 0.039968 | +11.94% | 191.994 | 214.920 | 79,862 | 80,139.2 | PASS |
| G3 no-TMA | ordinary global loads + same WGMMA/cluster | 0.119065 | 0.135168 | -11.91% | 72.1449 | 63.5501 | 212,531 | 249,398 | PASS |

Units: event times are milliseconds; throughput is decimal TFLOP/s; cycles are
SM-clock cycles. “Sim kernel cycles” is `gpu_sim_cycle` for the measured GEMM
launch. The H200 cycle column is the probe's globaltimer-derived value, not
event time multiplied by a nominal clock.

The diagnostic is functionally clean: G1/G2 agree bit-for-bit, G3 agrees
bit-for-bit, and 16 sampled outputs agree with the CPU reference
(`max_abs_err=0`, `atol=1.41`). Unicast and multicast kernel-cycle errors
are -0.31% and -0.35% at this one shape. This aggregate match does not prove
the TMA service model is accurate: section 11 finds substantial service
cost already present at short K, and startup/sustained errors may offset.
The event/globaltimer difference warrants separate investigation; it is not
proof that launch overhead alone explains the event errors. This point alone
does not justify changing bandwidth, WGMMA, or reply-network knobs.
G3 is 11.91% fast and remains evidence for an occupancy-sensitive
ordinary-load/operand-collector model gap. The next bounded discriminator is
the job's matched no-TMA `n4k`/`n8k` pair. No `sq_2k` variant was run.

## 9. Matched no-TMA occupancy diagnostic (2026-09-08)

**Validation correction:** these G3-only runs exited successfully but the
original path did not check output values. Numerical correctness remains
unverified. The repaired
path adds finite-output and sampled CPU-reference checks (section 10).

This diagnostic uses the same G3 kernel and fixed tile as the accepted GEMM
cases while holding M=512 and K=4096 constant and doubling N. It isolates the
benefit of placing a second CTA on each active SM without running `sq_2k`.
Both simulator runs use `SM90_H200_CLUSTER132`, `OMP_NUM_THREADS=4`, one smoke
launch, one measured launch, and no measured warmup repetition.

Hardware values come from job 2119329,
`/home/jcliu/H200_results/job_2119329/h200_gemm_notma_sweep_2119329.csv`.
Simulator CSVs and human-readable logs are under `/tmp/h200-notma-n4k` and
`/tmp/h200-notma-n8k`.

| Shape | Active grid | Sim event (ms) | H200 event median (ms) | Event error | Sim TFLOP/s | H200 TFLOP/s | Sim kernel cycles | H200 globaltimer cycles | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `n4k`: M512 N4096 K4096 | 128 CTAs | 0.245036 | 0.294624 | -16.83% | 70.1115 | 58.3112 | 437,390 | 531,988 | UNVERIFIED |
| `n8k`: M512 N8192 K4096 | 256 CTAs | 0.453236 | 0.501632 | -9.65% | 75.8098 | 68.4959 | 809,027 | 900,174 | UNVERIFIED |

Units: event times are milliseconds, throughput is decimal TFLOP/s, and cycle
counts are SM-clock cycles. H200 event values are medians; H200 cycle values
are the probe's globaltimer medians. The simulator's globaltimer estimates are
446,602 cycles for `n4k` (-16.05%) and 818,239 for `n8k` (-9.10%).

Doubling work increases simulator event time by 1.8497x versus 1.7026x on
H200. Equivalently, added occupancy raises simulator throughput by only 8.13%
while H200 gains 17.47%. Stall attribution changes from 69.6% `SB_SpInt` and
1.8% math-pipe throttle at 128 CTAs to 42.0% and 33.6% at 256 CTAs. Therefore
the simulator does hide some latency with a second resident CTA, but only
about half the measured throughput benefit. This rules out kernel-launch cost
and a uniform HBM slowdown as primary fixes. The supported direction is the
ordinary global-load/dependency latency exposed once per K loop; do not fit a
shape/CTA threshold or alter TMA, WGMMA, or DSM knobs from this G3-only pair.

## 10. Ordinary-load lowering diagnostics (2026-09-08)

The supplied c7 no-TMA cubin provides direct ISA evidence that
dependency-critical PTX `add.s64` address operations lower to paired Hopper
instructions (`IADD3`/`IMAD.X` or `LEA`/`LEA.HI`). An opt-in simulator option,
`ptx_int64_add_lowering_factor`, therefore scales only 64-bit integer ADD/SUB;
other presets retain factor one, while `SM90_H200_CLUSTER132` provisionally
uses factor two. Native instruction pairs motivate this candidate but do not
by themselves prove that both latency and initiation should double.

On G3 `cyc_1e5` (M256 N8192 K2048), one smoke and one measured launch with
`OMP_NUM_THREADS=4` complete successfully. Numerical correctness was not
checked by this G3-only path. Measured kernel time is
230117 cycles and CUDA-event time is 0.128917 ms, versus job
2119329's 0.157312 ms (-18.05%). Raw CSV and human-readable log are
`/tmp/h200-int64x2-small/run/h200_gemm_notma_local.csv` and
`/tmp/h200-int64x2-small/launcher.log`. This does not pass the strict event
gate. Correction: the previously cited 212531-cycle baseline was from the
different `k512` shape in section 8. It cannot demonstrate an improvement
for `cyc_1e5`. The now-complete matched factor-one control is
`/tmp/h200-int64x1-small.XdWIWb/h200_gemm_notma_local.csv`: 224189 kernel
cycles, 0.125596 ms, 233401 globaltimer-derived cycles, exit 0. Factor two
increases event time by 2.644%, reducing the hardware gap from -20.161% to
-18.050%, still failing the timing gate.

The factor-two K2K G3-only run also exited 0: 838132 kernel cycles,
0.469542 ms, 847343 globaltimer-derived cycles. Its event error versus
job 2119329's 0.497216 ms is -5.566%, a timing pass only. Raw CSV:
`/tmp/h200-int64x2-k2k-fast/run/h200_gemm_notma_local.csv`.

Source audit found that the G3-only harness never downloaded or validated
its output. Earlier claims that these isolated runs were numerically exact
are withdrawn, including n4k/n8k in section 9. The repaired path downloads
the measured output, checks all values for finiteness and 16 samples against
the existing CPU reference, prints the result, and exits nonzero on failure.
The synchronized source and overlay both contain the fix; a clean fuzz-zero
overlay application reproduces the source exactly. The probe rebuild and
selfcheck pass. A new M256 N256 K64 G3 smoke on the full 132SM preset with
OMP4 passes: all output values are finite and all 16 CPU-reference samples
match exactly (`finite_ok=1 ref_ok=1 ref_abs=0`), exit 0. Evidence:
`/tmp/h200-g3-validation-smoke2.log` and
`/tmp/h200-g3-validation-smoke2/h200_gemm_notma_local.csv`. The CSV now
records the validation fields. This small smoke does not retrospectively
validate the earlier full-shape results. None is final acceptance.
The comparator rejects explicit `finite_ok=0` as it does the existing
reference/match failure flags. Its regression previously reproduced a false
timing PASS for that annotation and now passes under normal Python and `-O`.

The native regression `test/check_int64_lowering.cc` checks the registered
default and 75 actual instruction-decoder timing combinations, including
factor-zero clamping, 32-bit controls, and unchanged multiply timing. It
passes against the current gcc-13.3.0/cuda-12080 production library. This
verifies implementation scope, not hardware calibration accuracy.

An additional native control exposed a bounds bug: integer execution-pipeline
sizing did not include the scaled ADD/SUB latency. With ADD=12 cycles and
factor two it allocated depth 14 instead of at least 24. The source now
includes this bound. The test fails against the prior method and passes with
the actual repaired method extracted and linked against the same library
(`/tmp/check_int64_lowering_bound_{before,after}`). The current H200 preset
still uses depth 14 because its scaled ADD latency is only eight cycles;
this repair does not improve or otherwise retime that preset. A normal
library rebuild succeeded. The native timing/bounds regression and all 217
unit tests pass against it (`/tmp/h200-pipeline-final-unit/unit.log`). Its
SHA-256 is `2ac8eaacdc401a79193cc75d616621be4dcac278716253dc25e28fbea29f8f8b`.

Two bounded alternatives were rejected and removed. Mapping the 32 packed-half
moves to four-cycle PRMT behavior changed the smoke launch by only 113 cycles
(0.05%). Sending all shared stores through the one-wide shared-load latency
pipeline exceeded 510000 cycles without completing the first G3 smoke; it
models store visibility by serializing throughput and is not credible. The
interrupted diagnostics are `/tmp/h200-int64x2-prmt4-small` and
`/tmp/h200-shstore-small`; their nonzero exits are deliberate early stops, not
functional failures. No `sq_2k` work was launched.

A shared-store conflict review does not justify penalizing subword writes
merely because they share one 32-bit word. The shared path in
`warp_inst_t::generate_mem_accesses` counts distinct words per bank; NVIDIA's
[CUDA 12.4.1 Programming Guide](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_C_Programming_Guide.pdf)
documents same-word accesses as conflict-free, including writes. This rules
out that proposed extra-conflict rule, not every possible shared-memory
timing defect. No source or timing change was made for this hypothesis.

## 11. Short-K TMA transaction cross-check (2026-09-08)

The trace `/tmp/h200-k512-lifecycle/tma_lifecycle.csv` contains several
variants and a partial later launch. To avoid mixing them, this analysis uses
only READ events before cycle 145625, the first multicast NEW event. This
selects the complete initial unicast smoke for M512 N16384 K512: 4096
16-KiB and 8192 8-KiB transactions, with no incomplete transactions. Join
events by `tx_uid`; lifetime is `COMPLETE.cycle - NEW.cycle`.

| Mean transaction lifetime (SM cycles) | Small M256 N8192 K2048 | M512 N16384 K512 | M512 N16384 K2048 |
|---|---:|---:|---:|
| 8 KiB | 1321.464 | 2522.383 | 2776.683 |
| 16 KiB | 1412.266 | 2729.846 | 2890.322 |

Small and K2K values are the bounded-prefix diagnostics documented in
`/tmp/h200-k2k-l2-lifecycle.0Ql0qR/results.md`; they are not full-launch
averages. K512's first-issue-to-issue-done means are 482.101/993.139 cycles
for 8/16 KiB; issue-done-to-complete means are 1956.962/1728.262 cycles.
The short-K case already experiences much of the transaction service cost
seen at long K. Thus its near-matching measured kernel total (section 8)
cannot establish that startup and sustained service are individually
accurate; their errors may offset. This is simulator diagnostic evidence,
not a new hardware measurement or justification for a CTA/shape threshold.

Regression status: the current library passes 217 existing unit tests with
`SM90_H200_CLUSTER132` and four OpenMP threads, exit 0. Evidence is
`/tmp/h200-current-unit.XQeXnO/unit.log`; library SHA-256 is
`c3829a9b68ee8d002e140a1e313eae5860b66eb9e039fec305f51f254e15e0fb`.
These checks establish regression coverage, not completion of the timing
calibration gates.
