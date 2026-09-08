# Living checklist (agent entry point)

**Audience:** a future agent with **no** chat history. Read this file, then only the docs listed under the item you pick.

**Working spec:** [`README.md`](README.md). Do not treat `docs/cluster.md` or the old `docs/cluster_noc.md` stub as design. Supervisor plan v2 is a local reference (not in this tree); ignore its chapter 18.

## How to take a task

NEXT REQUIRED INPUT: hardware profiling evidence for the retained cyc_1e5
and k2k GEMMs (unicast, multicast, no-TMA), particularly cache traffic,
warp stalls and achieved residency, with device/clock and launch provenance.
No Nsight captures or named stall/occupancy profile CSVs were found in
`/home/jcliu/H200_results` or `/home/jcliu/H200_profiling`; the job's
microbenchmark archive also has no matching profiler entries. Local
`nvidia-smi` cannot communicate with the driver, so this environment cannot
collect the missing H200 evidence. The user has been asked for captures or
an H200 profiling run. The completed local diagnostics do not establish a
supported correction for the opposite-signed small/K2K TMA errors. Do not
repeat unchanged long runs or introduce shape-dependent timing fits while
this input is outstanding. Acceptance is NOT achieved.

LATEST STATUS: normal-load HBM is terminal, exit 0; do not restart session
86079. Payload Passed, 480626 cycles, full 1 GiB DRAM-read accounting and
zero writes. Current normal-load bandwidth error is -0.054% (PASS), whereas
cp.async is +12.166% and TMA +12.386% (FAIL). Combined authoritative report:
`/tmp/h200-three-hbm-current-comparison/comparison.md`. This supersedes
all pending normal-load notes below. No calibration run remains live from
this batch. Do not apply a uniform HBM slowdown merely to fit the two async
paths. GEMM acceptance remains one of six; hardware counter evidence is
still needed to distinguish the remaining hypotheses. Keep sq_2k deferred.

CURRENT RUN STATUS: both six-point GEMM harness processes and both async HBM
rechecks are terminal, exit 0. Do not restart them. K2K validation passes
(`ok=1 notma_ok=1 ref_ok=1 max_abs=0 ref_abs=0.00781`, tolerance 2.83),
and its routing checker verifies 21 dumps with zero global-TMA DSM payload.
K2K event times: unicast 0.157436 ms (+41.416%, FAIL), multicast 0.141768 ms
(+27.526%, FAIL), no-TMA 0.469449 ms (-5.584%, PASS). Small remains three
timing failures. Final six-point comparison:
`/tmp/h200-six-current.qLOSuv/comparison/comparison.md`.
These results supersede the earlier pending-run notes below. Only one of six
GEMM timings passes. The reporting-only DRAM counter initialization has now
rebuilt successfully (`/tmp/h200-refresh-counter-build.log`, exit 0); it is
not a timing calibration fix. Current library SHA-256:
`abfffd16bb565a58bd87f20baeea3ded55f38201486dcca741b1f19c74c68d53`.
The native integer-lowering regression passes all controls. All 217 existing
unit tests pass, exit 0, in `/tmp/h200-counter-final-unit.tyV2w6/unit.log`
with the 132SM preset and OMP4.
Current normal-load HBM revalidation is now running under
`/tmp/h200-normal-current-results`, launcher log
`/tmp/h200-normal-current-launcher.log`: `--only vendor_tma_normal`, same
1 GiB / 132 blocks / 1024 threads / stages16 / chunk8192, OMP4, one sample.
The earlier +0.709% pass predates integer lowering; do not claim it is a
measurement of the current candidate until this run completes. Vendor binary
build log: `/tmp/h200-normal-current-build.log`. Keep this run intact.
Job 2119329 contains no named Nsight counter reports; hardware GEMM stall,
cache/service and actual residency evidence has been requested from the user
to help distinguish remaining model errors. Do not invent counter values or
apply an unsupported shape-dependent correction in their absence.
Local repetition-history diagnostic now runs in
`/tmp/h200-small-repeat.AaJ5OP/run.log`: existing isolated unicast driver,
M256 N8192 K2048, four samples each with `run_once`'s immediate warmup,
same embedded c7 winner and validators, current 132SM/OMP4, 1M cumulative
cycle ceiling and 3600-second host guard. Driver build passed. Parse all four
`REPEAT_SAMPLE` rows and require terminal correctness/routing checks before
interpreting them. Final `diagnostic.csv` contains only the last sample;
the raw log retains all four. This omits hardware's earlier autotune/shape
history and is diagnostic, not a replacement for the six-point acceptance.
This runs alongside the current normal-load HBM recheck; do not rebuild.
The repetition diagnostic is now terminal, exit 0: four event samples are
0.0428313725, 0.0428235307, 0.0428067222, 0.0428050421 ms (range 0.0615%).
Final finite/reference checks pass with ref_abs=0; route checker passes 16
complete DSM dumps. Repetition does not explain the small-unicast gap.
Do not restart this diagnostic. Normal-load HBM remains live.

Current no-TMA candidate: Hopper's shipped c7 SASS lowers dependency-critical
PTX 64-bit address adds to low/high instruction pairs, so the 132SM preset now
uses the opt-in `ptx_int64_add_lowering_factor=2`. The bounded `cyc_1e5`
G3-only check completes at 230117 measured kernel cycles, but its event
time is still 18.05% fast. The previously cited 212531-cycle baseline belongs
to `k512`, so it cannot establish a same-shape improvement. The matched
factor-one control now gives 224189 cycles / 0.125596 ms; factor two adds
2.644% event time. A packed-move latency A/B changed the smoke
by only 113 cycles (0.05%) and was removed. Pipelining every shared store
through the one-wide shared-load writeback path exceeded 510k cycles before
the `cyc_1e5` G3 smoke completed, versus 288501 H200 cycles for that shape;
that over-serializing patch was stopped and removed. The
factor-two `k2k` G3-only check completed, exit 0, at 838132 cycles /
0.469542 ms (-5.566% versus H200) under
`/tmp/h200-int64x2-k2k-fast`; keep `sq_2k` excluded.
The matched small factor-one control completed, exit 0, from
`/tmp/h200-int64x1-small.XdWIWb/run.sh`. Its config differs from the actual
factor-two small run by exactly the lowering factor, and its copied probe
is byte-identical to the current binary. Output is `launcher.log` in that
directory. Both used the same simulator library and
one smoke and one timed launch. Resource inspection showed 32 available CPUs,
load below 6, and 48 GiB available memory, so it runs alongside K2K, each at
OMP4. Both are terminal; do not restart them.

CRITICAL validation correction: the G3-only harness lacked output checking.
These runs and the earlier G3-only n4k/n8k pair establish timings and launch
completion, not numerical correctness. Earlier claims of exact validation
for those paths are withdrawn. The source now copies the measured output,
checks every element for finiteness and 16 CPU-reference samples, prints the
result, and exits nonzero on failure. The overlay preserves this repair and
applies with fuzz zero, reproducing the source byte-for-byte. Probe and
simulator rebuilds succeeded, and the existing selfcheck passes. A bounded
G3 functional validation passed, exit 0, at `/tmp/h200-g3-validation-smoke2.log`:
M256 N256 K64, full-output finite check and 16 CPU-reference samples,
`finite_ok=1 ref_ok=1 ref_abs=0`, 132SM config and OMP4. The CSV also records
these validation fields. The first attempt falsely failed because
the new call passed zero tolerance to a strict-less-than helper; that call
now uses a positive tolerance for the finite-only self-comparison.
The comparator now also rejects explicit `finite_ok=0` annotations. A
regression reproduced an equal-time false PASS before this change; both
normal and optimized-Python self-tests pass with the fix.
The six-point current-candidate run is now active under
`/tmp/h200-six-current.qLOSuv/{small,k2k}`. Two reusable harness invocations
select `gemm_cyc_1e5` and `gemm_k2k` respectively, using the rebuilt library,
132SM preset, OMP4 each, one immediate warmup and one measured sample per
variant, and `FLASHGPU_DSM_STATS=1`. Logs are `small.log` and `k2k.log` in
the parent directory. Poll these processes; do not restart or rebuild while
they run. On completion use both `metrics.csv` files with the comparator's
`--gemm-events-only --gemm-cases cyc_1e5,k2k`, and run the DSM route checker
on each completed per-case log. The gate remains open; the small smoke does
not establish full-shape correctness. `sq_2k` remains deferred.
The `small` invocation is now terminal, exit 0: all three output comparisons
and sampled CPU-reference checks pass (`ok=1 notma_ok=1 ref_ok=1`,
`max_abs=0 ref_abs=0`). The route checker verifies 21 complete DSM dumps
with zero global-TMA DSM payload. Current event times are 0.042795,
0.0443042 and 0.128900 ms for unicast/multicast/no-TMA respectively;
errors are -24.826%, -20.293% and -18.061% (all FAIL). Evidence is
`/tmp/h200-six-current.qLOSuv/small-comparison/comparison.md` and
`small/logs/gemm_cyc_1e5.log.gz`. Do not restart small; K2K remains running.
K2K has emitted a provisional unicast sample: 0.157436416 ms versus
0.111328 ms H200, +41.417% (timing FAIL). Its live `gemm_k2k.log` is the
source; final correctness/routing checks and the other samples are pending.
The multicast sample is now 0.141768068 ms versus 0.111168 ms H200,
+27.526% (timing FAIL). No-TMA and final correctness/routing checks remain
pending; the same K2K process is continuing. Do not restart it.
Current-config async HBM revalidation also runs sequentially under
`/tmp/h200-async-hbm-current.Uw314B/results`: `vendor_tma_cp_async` then
`vendor_tma_tma`, 1 GiB / 132 blocks / 1024 threads / stages16 / chunk8192,
warmup0/iterations1, OMP4. Both vendor binaries were rebuilt with the harness's
Hopper representative flags. This third process runs alongside the two GEMM
processes; host inspection showed load 5 and 35 GiB available memory. Compare
with job 2119329 `h200_tma_hbm_2119329.txt` after payload checks complete.
Earlier bandwidth errors were +12.054% and +12.690%; these are not current
measurements until this run completes. Do not rebuild the simulator meanwhile.

Current cp.async recheck has now completed with a payload PASS: 417654
cycles, 2570.888400 bytes/cycle versus H200 2292.047679 bytes/cycle,
**+12.166% bandwidth error (FAIL)**. Source is the completed compressed log
and `metrics.csv` in that results directory; comparison is
`/tmp/h200-async-hbm-current.Uw314B/cp-async-comparison/comparison.md`.
The missing TMA/normal-load rows in this partial comparison are not passes;
that partial snapshot predates TMA completion, and normal-load was not selected.
The async HBM invocation is now terminal, exit 0. TMA payload validation
passes at 415585 cycles / 2583.687631 bytes/cycle, versus hardware
2298.942583 bytes/cycle: **+12.386% bandwidth error (FAIL)**. All 1 GiB
is accounted for by 33554432 DRAM reads, zero writes. Final async comparison:
`/tmp/h200-async-hbm-current.Uw314B/comparison/comparison.md`.
Do not restart either completed HBM case; only the K2K harness remains live.

Reporting-only source fix pending the next safe simulator rebuild:
`dram_t` printed an uninitialized `n_ref` member. Its sole constructor now
sets it to zero; no refresh command/timing model exists, so historical zeros
must not be used as measurement evidence. No modeled service rate changes.
Keep the live K2K library intact; rebuild after it finishes.

Native regression `test/check_int64_lowering.cc` passes the option-default
check and 75 actual decoder timing controls against the current simulator
library (factors 0/1/2, five ADD/SUB/MUL opcodes, five integer types).
It also exposed a pipeline-size bug: `set_pipeline_latency()` ignored the
lowered ADD latency. A factor-two/12-cycle ADD control needs depth 24 but
received 14. The source now includes the scaled ADD bound; the native test
fails before and passes with the extracted current method linked against
the unchanged library. Production H200 remains depth 14 (4*2 < 14), so this
repair does not retime its runs. The rebuilt library passes the native test
and all 217 existing unit tests (`/tmp/h200-pipeline-final-unit/unit.log`);
the 217-test library evidence below predates this small bounds repair.
Harness, comparator, and routing-checker self-tests pass; these are checker
regressions, not replacement evidence for the six-GEMM timing gate.
The current library also passes all 217 existing unit tests on the 132SM
preset, exit 0, with source fixtures linked into the run directory:
`/tmp/h200-current-unit.XQeXnO/unit.log`. Library SHA-256:
`c3829a9b68ee8d002e140a1e313eae5860b66eb9e039fec305f51f254e15e0fb`.

The existing `k512` lifecycle trace has now been separated by launch: its
first complete unicast smoke has all 12288 read transactions completed before
the first multicast NEW at cycle 145625. Mean 8/16-KiB lifetimes are
2522.383/2729.846 cycles, already close to K2K's bounded-prefix
2776.683/2890.322. Do not pool the trace's later multicast and partial timed
launch. Matching one `k512` total time does not exclude offsetting startup
and sustained-service errors; see calibration report section 11.

Harness reproducibility recheck: the H200 overlay applies to the pinned
profiling tree with `patch --fuzz=0`, and patched `probe_gemm_triton.cu` and
`Makefile` are byte-identical to the local synchronized copies. The sync
script now handles `--help`/`-h` before any filesystem or network work; its
timestamp-preservation check passes. Temporary round-trip tree:
`/tmp/h200-overlay-check.H6OQ08`.

New correctness fix: both `tag_array::fill` overloads now decrement `m_dirty`
when a fill cleans the last modified sector/line. Native regression
`test/check_cache_fill_dirty.cc` fails four controls with candidate708 and
passes all 20 with the current extracted methods; logs are
`/tmp/check_cache_fill_dirty_{before,after}.log`. The isolated build is
`/tmp/h200-cache-dirty-build.5rQR0o/lib/libcudart.so` (SHA `9faf414d...`);
it passes 216/216 tests and all 20 direct cache controls. Use
`unit-suite-source-linked.log` as the valid suite evidence. The earlier
`unit-suite.log` lacked the source-fixture symlink; its seven missing-source
failures are an invocation error, not a regression.
The candidate708 K2K TMA-credit A/B is complete; see
`/tmp/h200-k2k-tma-credit.r0R8yM/results.md`. Raising the per-SM cap from
384 to 768 improved its isolated timed launch by only 3.32%, so do not
change the production cap from this diagnostic.
The fix does not establish a GEMM timing cause (L2 dirty threshold is zero).

The paired small/K2K lifecycle trace and the K2K `dram_latency=0` A/B are
complete; see `/tmp/h200-k2k-l2-lifecycle.0Ql0qR/results.md`. K2K TMA
transaction service roughly doubles under miss/congestion pressure, while
CTA dispatch/release is prompt. Zero residual DRAM latency is 1.495% slower,
so retain 254. The selected kernel deliberately waits for the previous WGMMA
before issuing the current TMA loads; do not treat its two rotating buffers as
proof of cross-iteration prefetch. The 512-to-1024 interconnect input-buffer
sensitivity run is now terminal and correct, but improves K2K only 2.029% and
still misses hardware by 32.757%. It removes reply-input-full events while
maximum occupancy grows to 970 and timed DRAM reads increase, so it mainly
redistributes queueing. Retain 512: this packet-count limit is shared across
all VOQs, applies to both request and reply networks, does not change router
throughput, and has no H200 hardware provenance.

The oversized-L2 upper-bound run is also terminal and correct. Increasing
only L2 associativity from 20 to 32 ways (58.75 to an unsupported 94 MiB)
makes all 16,777,216 timed K2K TMA read sectors (512 MiB) hit with zero DRAM reads, but
slows the event by 1.163% and increases reply backpressure. Retain the
datasheet-derived 58.75 MiB. Capacity misses and residual DRAM latency are
closed as primary K2K causes. Bounded requests reach all 188 slices with
6.02%/8.54% small/K2K count CV, so there is no gross slice hot spot. See
`/tmp/h200-k2k-l2-upper.xyTurT` and the lifecycle report.
Traffic-accounting correction: the earlier 33,554,432-sector value included
warmup plus timed launch. Subtract the preceding stamp kernel's cumulative
16,777,216 counter. Timing and the all-hit conclusion are unchanged.

The sampled exact reply-matching bound is terminal. Small greedy arbitration
delivers 8633/8693 feasible grants (99.310%); K2K delivers 7038/7062
(99.660%). K2K is closer to the maximum, so the real greedy corner case does
not explain the 35.5% scale-dependent error. The temporary instrumentation
was removed and the normal library rebuilt. Do not rewrite arbitration from
this result; see `/tmp/h200-reply-match-results.md`.

Per-request TMA transaction rotation is also rejected. With quota48/cap384
unchanged, it changes small by +0.097% and makes K2K 0.721% slower while
raising reply pressure; both runs remain correct. The temporary one-line
change was removed and the normal library rebuilt. Keep the production
selection policy; see `/tmp/h200-tma-rr-results.md`.

Per-SM stage accounting is also terminal: small has no local TMA admission
stall, while K2K has 9.19M aggregate SM-cycles blocked at cap384. Neither run
queues more than one reply or hits response width. GTO reduces K2K unicast by
only 6.047% to 0.141735017 ms, still +27.313% versus H200. Retain inherited
LRR; the result indicates multi-CTA phasing but is not a calibrated fix.

Reply-side multi-grant is rejected too: it slows correct K2K unicast by
1.043% to 0.152430817 ms and increases cap blocking. Retain the inherited
single-grant model; `/tmp/h200-k2k-reply-multigrant/run.log` is diagnostic
only.

The opposite quota bound (`gpgpu_tma_tx_quota=0`, finish selected transaction)
improves correct K2K unicast by only 5.449% to 0.142636970 ms, still +28.123%
versus H200. With per-request rotation also rejected, retain inherited
quota48; transaction-selection granularity is closed as the primary cause.

Idealized TMA gives a correct K2K non-memory lower bound of 0.0679557398 ms
(38.959% faster than H200). It is not a candidate setting; it proves the
remaining K2K gap is closable within the memory-return path rather than by
retuning the already-passing WGMMA group.

Far-L2 zero is counterproductive too: correct K2K unicast is 0.151836976 ms,
0.650% slower than baseline. Source audit finds one pre-L2 sector charge and
no response-side duplicate; summed extra-latency counters are not elapsed
stall cycles. Retain the H100 fallback150 pending matched locality evidence.

Clock upper bounds are closed: ICNT×2 improves correct K2K unicast by 7.969%
but remains +24.709% versus H200; L2×2 is 1.820% slower. Do not change either
clock. No single raw transport/L2 rate explains the remaining gap; audit the
hardware c7-selection provenance before changing request/return admission.

The exact 132-block SS N128/group1/eight-operation WGMMA diagnostic is also
terminal: 71.9551 cycles/WGMMA versus H200 72.2275 (-0.377%). See
`/tmp/h200-wgmma-gemm-pattern.3WFZmM`. Do not tune WGMMA from the GEMM gap;
its actual end-to-end operation-group pattern passes despite offsetting issue
and wait component errors. Continue the no-TMA investigation in the ordinary
load/L1 pressure path, and the TMA investigation in miss-tail/backpressure.

The first no-TMA load-path A/B is terminal: changing only inherited
`gpgpu_l1_latency` 39 to 49 slows the exact small no-TMA event by just 0.485%
and leaves 19.835% hardware error while increasing retry observations. See
`/tmp/h200-small-notma-l1.kdUaKp/results.md`. Retain 39 and do not spend time
on the staged scalar-load cross-check for the already-rejected value. The
remaining no-TMA discrepancy is not explained by WGMMA group timing or this
fixed L1 lookup latency; inspect issue/collector pressure before another A/B.

The exact scalar 4 KiB global-load probe at production latency39 is also
terminal: simulator median69350 cycles versus H20031956 (+117.017%). Since
the serial scalar path is too slow while parallel no-TMA GEMM is too fast, do
not apply a blanket ordinary-load speedup/slowdown. The next no-TMA diagnostic
must isolate issue/admission overlap under many resident warps; fixed L1
latency is closed.

The scalar probe is not a matched load-ILP control: scheduled PTX groups four
loads before a dependent shared store, while hardware SASS hoists 20 loads
before its first store. Its 96% global-scoreboard stall share and zero L1
reservation failures support dependency-limited issue. Do not use this
compiler-ordering mismatch to tune GEMM latency.

The selected c7 unicast GEMM does not share that mismatch: the job cubin SASS
orders TMA issue and mbarrier wait after the previous iteration's WGMMA wait,
matching the simulator PTX at the loop level. Do not pursue speculative
cross-iteration TMA prefetch as the K2K fix.

Hardware-selection provenance is now source-grounded: the misleading
multicast winner text is a tag-only bug, while dispatch and timing use the
unicast c7 kernel. Simulator and supplied H200 sources embed the same cubin;
the adjacent artifact differs only in 64 `.debug_line` bytes and has identical
executable and kernel-info sections. Exact remote-binary provenance is still
unavailable because job 2119329 recorded no binary hashes. Before another
production knob A/B, test the remaining measurement-history difference:
hardware records four timed repetitions after its earlier sweep, whereas the
isolated simulator diagnostics record one. Keep `sq_2k` excluded.
Raw `GEMM_SAMPLE` rows are now emitted by the synchronized probe and preserved
by the overlay; rebuild and `--gemm-selfcheck` pass. Use `--samples 4`
and `--warmup 1` for final matched runs, after the shorter sensitivity diagnostic
shows that the repetition cost is warranted.

The completed small/K2K transaction traces show a one-cycle
`COMPLETE`→`ARCH_ARRIVE` gap for every paired TMA transaction (3840 and 5626
pairs). The architectural completion floor is not exposed; do not lower its
calibrated base or size terms to fit these GEMMs.

A complete parent-sector join further localizes the K2K scale penalty. Mean
last-L2-response→last-TMA-retirement grows from 321.2 to 656.1 cycles and mean
transaction issue-done→last-return from 1171.6 to 2058.2 cycles, while first
L2 request→accept is slightly faster at scale (489.2→465.2). All-hit K2K
parents remain slow, so DRAM misses are not the root cause. The next bounded
diagnostic is a temporary two-grant-per-SM reply output paired with
`gpgpu_tma_response_width=2`; production remains one-wide unless both GEMM
and standalone TMA evidence plus hardware provenance support a change.

That upper bound is now terminal. It moves correct K2K unicast from
0.150857136 to 0.102214567 ms (-8.186% versus H200, PASS) but moves correct
small unicast from 0.0428084023 to 0.0366112031 ms (-35.688%, FAIL). K2K reply
throughput rises 64.239→94.446 sectors/ICNT-cycle and input-full events vanish;
small rises only 53.103→61.027. This proves a scale-sensitive reply bottleneck
but rejects a fixed two-wide production setting. Next test only a
hardware-supported low-concurrency cost or concurrency-scaled service model;
do not sweep arbitrary thresholds. Logs:
`/tmp/h200-{small,k2k}-reply-output2/run.log`.
The standalone 1280 MiB TMA-L2 control also fails: 249799 cycles and
5373.029 bytes/cycle versus H200's 4118.245 (+30.469%). Keep the production
reply path one-wide. Evidence: `/tmp/h200-l2-tma-reply-output2/run/run.log`.

Reject the rate-one, capacity-two reply burst-credit model. Seven router tests
pass and small unicast is neutral at 0.0428100824 ms (+0.004% from baseline),
but K2K is 0.146668911 ms versus H200's 0.111328 ms (+31.745%, FAIL), only
2.776% faster than baseline. Do not add this stateful model or spend time on a
standalone rerun. Evidence: `/tmp/h200-k2k-reply-burst2/run.log`.
Close reply-network tuning here. Width one is constrained by standalone H200
TMA-L2 throughput, and the bounded buffer/arbitration alternatives are either
insufficient or worse; do not add traffic-shape thresholds or CTA-dependent
service without new hardware evidence.

The dedicated TMA execution path is now functional. The generic ALU issue
branch had captured TMA as INT before its own branch, and the generic operand
collector lacked the TMA port. The minimal shared fixes survive the
`latency=182, initiation=32` stress case on the full 132-SM small GEMM:
exact validation, 0.042778153 ms, clean exit. Since that result is effectively
the 0.0428084023 ms baseline, keep production TMA instruction timing at 32/32;
it does not explain the GEMM gap. Evidence:
`/tmp/h200-small-reply2-tmalat182/run.fixed2.log`. The existing reduced-H200
`tma_copy_test` also passes at 182/32; log:
`/tmp/tma-latency-regression.1T14s3/run.log`. All 27 existing SM90
instruction tests pass after the routing fix.
Current production library SHA-256:
`b6f023346de723b36b768701a7c567c5fcfd8d7f962c03a53a42e950b6c679be`.

Do not split WGMMA compute completion per CTA/warpgroup. The selected op costs
64 cycles at the configured per-SM tensor throughput, and its shared RF traffic
also costs 64 cycles; both are aggregate SM ceilings. The matched PTX/SASS
dependency order is correct, so per-CTA overlap would be unsupported and can
overstate peak throughput.

Do not use `gpgpu_TB_launch_latency` as a fit: its grid-size delay can only
worsen K2K TMA, the two no-TMA shapes imply incompatible per-block values, and
job 2119329 does not measure this term. Retain the H100 baseline zero.

Reject the bundled non-perfect instruction/constant-cache A/B.
Small no-TMA moves from 0.125500277 to 0.143414572 ms, reducing H200 error
from -20.222% to -8.834% with exact output; the timed launch adds 1,280 L1I
misses and 80 L1C misses after the matched warmup. The bundled flag does not
isolate the two caches, includes kernel-parameter loads, and leaves L1I/L1C
warm across the simulator's normal cache flush. K2K completes at
0.458259940 ms versus H200's 0.497216 ms (-7.835%, PASS), only 0.210% slower
than its prior baseline. Both no-TMA shapes pass, but the five-case regression
gate severely degrades accepted TMA/mbarrier timing despite 5/5 functional
PASS. Keep the H100 perfect-cache baseline. Evidence:
`/tmp/h200-k2k-notma-icache-real/run.icache.log` and
`/tmp/h200-cache-knob-gate/comparison/comparison.md`.
Split diagnostics confirm the apparent gain is entirely L1I: real-I/perfect-C
is 0.142687395 ms (-9.297%), while perfect-I/real-C is 0.125611201 ms
(-20.152%). Four cold-SM tail CTAs create the pass. Do not add separate cache
knobs; investigate the low-occupancy ordinary-load/collector path instead.

### Current scope override — 2026-09-08

User explicitly deferred `sq_2k` (M2048 N2048 K8192), including unicast,
multicast and no-TMA, because of simulation runtime. Do not launch or tune
this shape until the user requests it again. Preserve its partial artifacts;
an interrupted run is user-deferred, not a numerical failure or an accepted
timing measurement. This overrides historical all-nine/three-shape gates below.

The current GEMM acceptance gate is **six CUDA-event timings**: all three
variants for `cyc_1e5` (M256 N8192 K2048) and `k2k` (M512 N16384 K2048),
with matched hardware launch configuration, valid numerical checks, and
strictly less than 10% error each. All non-GEMM requirements, reasonable-knob
constraints, zero global-TMA DSM routing, source-attributed reporting and
the no-commit instruction remain unchanged. Retain the square case in the
reusable harness for later explicit selection; do not silently remove it
from historical reports. For current GEMM-only runs use
`--only gemm_cyc_1e5,gemm_k2k`; `gemm_sq_2k` is also excluded by the harness
default and must not be appended until the user explicitly restores it.

For the current six-point comparison use
`scripts/compare_h200_calibration.py --gemm-events-only --gemm-cases cyc_1e5,k2k`
with the usual `--hardware`, `--sim` and `--output-dir` arguments. Default
comparison still selects all nine historical timings. Selected hardware
must be complete; missing selected simulator rows remain explicit. The
updated comparator's normal/`-O` self-tests pass. Current six-point report:
`/tmp/h200-validation-status.BVPJyS/six-comparison/comparison.md` (one PASS,
five FAIL; exit 1 is the expected unsuccessful calibration gate).

The product goal is active but retains the old objective text; this explicit
user scope change is the working plan. No token budget applies.

### K512 occupancy diagnostic — 2026-09-08

The bounded 132-SM `k512` diagnostic (M512 N16384 K512, 512 CTAs) completed
with `OMP_NUM_THREADS=4`, one smoke launch and one measured launch per variant,
and no `sq_2k` work. All output/reference checks pass exactly. Raw log and CSV:
`/tmp/h200-gemm-k512-complete/{launcher.log,run/h200_gemm_compare_local.csv}`.

| Variant | Sim event (ms) | H200 event (ms) | Error | Sim kernel cycles | H200 globaltimer cycles |
|---|---:|---:|---:|---:|---:|
| unicast TMA | 0.0454241 | 0.040544 | +12.04% | 81,082 | 81,332.8 |
| multicast TMA | 0.0447406 | 0.039968 | +11.94% | 79,862 | 80,139.2 |
| no-TMA | 0.119065 | 0.135168 | -11.91% | 212,531 | 249,398 |

For both TMA variants, kernel execution cycles match H200 within 0.35%; do not
retune TMA bandwidth, WGMMA throughput, or the reply network from the roughly
12% short-event discrepancy. The simulator event includes about 9,211 cycles
of adjacent one-CTA timing/setup work, which is material at this shape. Report
event and kernel/globaltimer evidence separately. The no-TMA result supports
the existing low-occupancy ordinary-load/collector investigation: higher CTA
occupancy closes most of the `cyc_1e5` no-TMA error without a global memory
slowdown, but remains 11.91% fast. Next use the matched `n4k`/`n8k` no-TMA pair
to isolate occupancy scaling; keep `sq_2k` deferred.

### Matched no-TMA occupancy result — 2026-09-08

The `n4k`/`n8k` discriminator completed on the 132-SM configuration, but its
G3-only path did not validate numerical output. It ran one smoke plus one timed launch per shape, warmup zero,
with no `sq_2k` work.

| Shape | CTAs | Sim event (ms) | H200 median (ms) | Error | Sim/H200 throughput uplift |
|---|---:|---:|---:|---:|---:|
| M512 N4096 K4096 (`n4k`) | 128 | 0.245036 | 0.294624 | -16.83% | baseline |
| M512 N8192 K4096 (`n8k`) | 256 | 0.453236 | 0.501632 | -9.65% | 1.0813x / 1.1747x |

Raw evidence: `/tmp/h200-notma-n4k` and `/tmp/h200-notma-n8k`. The simulator
captures only about half of H200's occupancy uplift. Investigate the ordinary
global-load/dependency latency exposed per K loop; reject launch, uniform HBM,
shape-threshold, and TMA/WGMMA retunes for this result. Before any production
timing edit, use the existing short TMA lifecycle trace on unicast `k512` to
separate TMA startup latency from sustained return service.

Reproducibility update: `scripts/calibration_overlays/h200_probes_sim.patch`
now preserves the latest finite validators, failure-status reporting, and
previously omitted legacy HBM workload controls. Applying it with
`patch --batch --fuzz=0 -p1` to the pinned profiling revision reproduces all
nine selected files byte-for-byte. Before/after trees and the old patch are
at `/tmp/h200-overlay-roundtrip.KptosE`. The local probe tree was not
resynchronized or overwritten. Harness/comparator self-tests pass.

GEMM status propagation is repaired in the probe, suite harness and comparator:
explicit numerical failures must not become timing/functional PASS. Both
Python self-tests pass normally and under `-O`, including failed reference,
multicast and no-TMA checks with equal timings. Use these updated scripts for
final reporting; running processes retain their old loaded code. The probe
adds an explicit failure line and `notma_ok` timing note; no timing knobs change.

Final GEMM acceptance must use the repaired finite-value validators in
`src/probe_gemm_triton.cu`: old NaN differences could silently pass.
Separate `/tmp/h200-gemm-validation-fix.mZiTX8/nvprof-after` passes 19 direct
controls and existing harness policies; original validators fail ten controls.
The live candidate batch retains the old executable, so its correctness flags
are provisional (no NaN output has been demonstrated). Do not replace that
binary mid-run. Rebuild/use the repaired probe for final acceptance; the
G3-only timing entrypoint has no numerical check and is not a substitute.

Priority correctness finding: the production non-power-of-two IPOLY mapping
aliases distinct addresses to identical full DRAM coordinates. Reproducer:
`/tmp/h200-hash-local.XaPdoS/check` (`0x000` and `0x200` both decode to all-zero
coordinates); 1MiB/32B sweep gives 26288 collisions versus zero for consecutive
indexing. Legacy balanced=0/1 also alias. The shared decoder source now uses
a bijective IPOLY-derived rotation for all three legacy non-power2 modes;
power2 and channel-stable paths remain unchanged. Permanent standalone
regression `test/check_address_mapping.cc` passes all 12 H200/H100-sized
geometry/mode/control combinations; main verification log is
`/tmp/h200-map-regression.L4snoW/result.log`. A separate candidate library is
now built at `/tmp/h200-memory-fixes-build.CtH6nJ/lib/libcudart.so` (SHA 708c448e...).
The regular baseline library is unchanged. The H200 config's obsolete hash
comment is corrected; numeric settings are unchanged and the active baseline
already uses a copied config with captured start-time provenance.
Read the decoder audit in `calibration.md`. Repair and regress this before
final timing acceptance; existing passes are historical, not validation of
the flawed mapping. Keep the active trace libraries unchanged until terminal.
Separate deferred geometry review: the inherited 13-row-bit mask has only
23.5GiB H200 DRAM-coordinate capacity, and CUDA memory-size reports are
inconsistent placeholders. This is not functional address wrap; the hash
repair does not extend capacity. Document and review before any full-141GB
claim. The current calibration buffers do not validate that capacity.
The mapping regression now also sweeps the actual 48GiB heap base, with
65536 unique tuples/pairs across two 1MiB regions; all 12 cases pass in
`/tmp/h200-map-regression.L4snoW/high-address.log`. A separate caller defect
has a source fix: `perf_memcpy_to_gpu` now keeps its loop offset and preload
address as size_t, instead of truncating the 64-bit heap address to unsigned.
Run `test/check_memcpy_preload.py` on the documented short 132SM copy-engine
trace after rebuilding; synthetic checker cases pass under Python `-O`.
H2D/D2H functional storage was not truncated. The short test does not cover
a >4GiB copy length.
Failing-before runtime control is now terminal, exit 0:
`/tmp/h200-preload-before.bl6073/run.log`. The old library emits low preload
addresses `0, 0x20, ...`; the checker correctly exits 1. Reuse the same
short benchmark/trace settings for the completed passing-after proof at
`/tmp/h200-preload-after.dpY0wu/run.log`: all 128 high-address sectors match.

Candidate runtime checks now pass: 216/216 unit tests in
`/tmp/h200-unit-memory-fixed.UvrStH/run.log`; five global multicast tests
with zero DSM payload in `/tmp/h200-global-route-fixed.T8GekU/run.log`;
8/8 checksum-validated vendor smoke controls with mapped TMA payload in
`/tmp/h200-mapped-route-fixed.tbjYQs/run.log`.
The historical nine-case batch ran sequentially in
`/tmp/h200-gemm-memory-fixed.wPiMCL/results` with candidate 708c448e..., OMP4,
warmup=1/sample=1 and DSM counters enabled. Config 6caac58f... differs from the
baseline only in comments. Check correctness, all nine event errors, and
global-zero DSM counters before acceptance. The harness now fingerprints
external libraries and loader search order; self-tests pass. See the current
candidate section at the top of `calibration.md` for full hashes and limits.
Candidate small GEMM (M256 N8192 K2048) is complete: all outputs/reference
checks pass, and all 21 complete DSM dumps have zero payload, including
explicit tensor multicast. All three event timings fail: unicast
0.0427978ms (24.821% fast), multicast 0.0443042ms (20.293% fast), no-TMA
0.125347ms (20.319% fast). K2K completed: unicast0.151900ms (36.444% slow),
multicast0.141589ms (27.365% slow), no-TMA0.457300ms (8.028% fast, timing
PASS). All21 K2K DSM dumps have zero payload. Its square attempt was later
stopped and is now user-deferred; the retained historical comparison has
1PASS/5FAIL/3MISSING. Numerical flags remain provisional under
the old validator; final acceptance requires the repaired finite checks.
The batch's `comparison/` preserves all nine targets, six still missing.
Do not restore the erroneous memory mapping/preload to regain old passes.
Follow-up if candidate GEMM gaps persist: trace WGMMA scheduler issue,
collector dispatch, tensor admission and async completion by instruction
UID. Async counting currently begins at scheduler issue and has no explicit
tensor-admission dependency. Measure any overlap with queueing before
changing that abstraction; no new timing fix is established by source
inspection alone.
Initial bounded matched-GEMM lifecycle evidence is now available at
`/tmp/h200-wgmma-lifecycle-matched.YLgqev/run.log`: 16 complete SM0 UIDs,
2–3 cycles scheduler-to-admission (mean 2.125), zero completion-before-
admission. This early sample does not explain the 20–25% small-GEMM gap;
do not change WGMMA timing on this evidence. Diagnostic library4335a49f...
passes216/216 unit tests and is separate from the candidate708 runs.
Short non-GEMM revalidation completed with 6/6 functional PASS in
`/tmp/h200-short-probes-memory-fixed.xF79r9/results`: vendor TMA latency
256B/1KiB/4KiB/16KiB, representative TMA multicast, and mbarrier. Same
candidate library/config and OMP4; preserve these matched measurements for
the combined comparison. The old K2K trace is now terminal (exit 0), and
the three-path 1GiB HBM remeasurement has completed sequentially at
`/tmp/h200-hbm-memory-fixed.y7xnzH/results` on the same frozen candidate.
Normal-load HBM now passes payload checks and bandwidth: 476985 cycles,
4.01822 TB/s versus hardware 480368 cycles, 3.98992 TB/s (0.709% rate gap).
cp.async has completed with payload PASS but bandwidth FAIL: 418070 cycles,
4.58447 TB/s versus hardware 4.09131 TB/s (12.054% high). TMA completed with
payload PASS but bandwidth FAIL: 414464 cycles, 4.62436 TB/s versus hardware
467059 cycles, 4.10361 TB/s (12.690% high). All three are functionally complete;
only one bandwidth target passes. See the run's `comparison/`. Do not globally
throttle the passing normal-load path to hide the cp.async/TMA gaps.
Candidate DSM bandwidth completed at `/tmp/h200-dsm-memory-fixed.eFHCBY/results`
(exit 0, 157 samples, 1188.581 host seconds): 10/12 non-exempt slopes pass.
One-way load/store fail by 10.814%/11.935%; BW7 passes (4.704%), exempt BW8
fails (40.129%). See `comparison/` for all 14 rows; old 12/12 is not current.
The 16-CTA latency case was skipped by the default exclusion list, not run.
It is now explicitly enabled with `--exclude none --only dsm_calibration`
at `/tmp/h200-dsm-latency-memory-fixed.lxtwwT/results`, same frozen library,
config and OMP4. That latency run is now complete (exit 0,1624.636s): remote
mean/RTT/store visibility pass; local latency and remote minimum fail by
37.453%/13.584%. The full table's13passes include derived/metadata rows,
not13independent timing tests. Matched L2 normal-load completed with exit0 at
`/tmp/h200-l2-memory-fixed.MIxzLp/run.log` with candidate708 and one warmup,
one timed launch: payload PASS, 370715 timed cycles, 6462.61 GB/s versus
hardware 358536 cycles, 6682.13 GB/s (3.285% bandwidth error, PASS).
Warmup370382 cycles is not the measured result. Matched vendor TMA L2 also
completed at `/tmp/h200-l2-tma-memory-fixed.d5MXY0/run.log`: vendor checksum
PASS, 352027 timed cycles and 6805.69 GB/s versus hardware325910 cycles and
7351.08 GB/s (7.419% rate error, PASS). Warmup350319 cycles is excluded.
Timed L2 misses and DRAM accesses are zero. Both L2 rates are slightly below
hardware; do not globally slow L2 to fit small GEMM. Provenance and hardware
references are in `calibration.md`; no new knob fit has been applied.
The short-probe run's four vendor TMA latency cases pass correctness, zero-DSM routing,
and strict <10% timing: 1396/1424/1538/1994 cycles versus hardware
1351/1346/1649/1887 (maximum gap 6.731%). Comparison artifacts are in
`tma_latency_comparison/` under that run. All 16 representative multicast
and unicast E2E points pass (maximum 7.192%), but issue, skew and several
derived metrics still fail; five fanout metrics are missing. Keep those
visible in `tma_multicast_comparison/`. This probe gathers results using
explicit post-timing DSM loads, so it is not an isolated zero-payload test.
Mbarrier has four local timing passes, remote owner/E2E failures of
37.264%/45.391%, and an invalid zero hardware remote-arrive reference;
see `mbarrier_comparison/`. Resolve remote timing before acceptance.
Remote mbarrier audit: the CSV-advertised remote-hop knob is bypassed by
the enabled DSM path; do not tune it. Owner wait and E2E have different
measurement envelopes (E2E includes the owner's global result store), and
an ARRIVE return packet already exists. Separate these phases before adding
any transport penalty; see `calibration.md` for source-level conclusions.
The completed `/tmp/h200-remote-mbar-trace.iX3yP7` diagnostic (exit 0)
reproduces the short-batch values. Arrival issue→owner release scheduling
is 78 cycles, release delay 43, return/control→owner clock 12; owner
clock→final globaltimer is 20, including a store issued two cycles before
the timer. No second wait or long store stall appears. Hardware's missing
cost is still unexplained; see the exact timeline in `calibration.md`.

Current experiment: fixed-output-priority VOQ arbitration can starve a higher
output under sustained low-output traffic. A failing-before/passing-after
regression covers both network directions; the router now rotates its output
scan without increasing bandwidth. Small warmup=1 GEMM completed in
`/tmp/h200-gemm-router-rotation`: unicast -0.354% PASS, multicast -9.827%
PASS, no-TMA -17.285% FAIL, all correctness checks pass. Multicast has very
little timing margin. The larger M512 N16384 K2048 warmup=1 run completed
in `/tmp/h200-gemm-router-k2k`: unicast +51.086% FAIL, multicast +40.283%
FAIL, no-TMA -8.344% PASS, with correctness passing. The square warmup=1
run completed in `/tmp/h200-gemm-router-square` on the same build:
unicast +54.222% FAIL, multicast +39.731% FAIL, no-TMA -9.638% PASS;
all correctness checks pass. All three runs are terminal. Four of nine
GEMM event timings pass; five fail. Diagnose larger-shape TMA service and
small-shape no-TMA timing before further acceptance runs.
Full simulator-backed unit regression passed 216/216 in
`/tmp/h200-router-unit-sim-rooted.log`. All-nine acceptance remains open.
See `calibration.md` for library/config hashes and exact results.
The three same-config/same-library baseline CSVs are now combined in
`/tmp/h200-router-all-nine-comparison/{comparison.csv,comparison.md}`:
all nine rows present, four PASS/five FAIL. The comparator now accepts
multiple `--sim` paths with source preservation and duplicate rejection;
its CLI regression passes. This remains a baseline, not final acceptance.

Completed diagnostic: `/tmp/h200-reply-multigrant.vtkbrQ/results` ran K2K
with only `icnt_multi_grant_reply=1` added to a temporary 132SM config.
It tests reply-input arbitration sensitivity, not an accepted hardware
bandwidth change. Compare against `/tmp/h200-gemm-router-k2k`; both use
warmup=1, samples=1, OMP=4 and the same router-fixed library. Production
config remains unchanged. Exit 0, all correctness checks pass, 6255.781 host
seconds. Event errors are +50.958% unicast, +48.033% multicast, -8.228%
no-TMA. Unicast barely changes and multicast worsens versus baseline;
reject this setting as a remedy. Do not restart this terminal run. Continue
investigating larger TMA service and small no-TMA timing; see calibration.md.

Completed evidence run: `/tmp/h200-k2k-tx-trace.BL9ylc/results` repeated the
production-config K2K case with existing transaction tracing enabled at
`/tmp/h200-k2k-tx-trace.BL9ylc/transactions.csv` (per-memory-fetch tracing
disabled). No build or knob change. Separate NEW/FIRST_MF_ISSUE,
ISSUE_DONE and completion events before choosing another service change.
Keep warmup=1, samples=1, OMP=4; this is a diagnostic, not final acceptance.
Exit 0 in 6837.341 host seconds, all correctness checks pass, and all three
event times exactly match the old baseline; do not restart it.
Timed unicast has completed in 300239 cycles, exactly matching baseline.
Its 8192/16384-byte reads average 2518/3334 cycles from last issue to memory
completion, versus only 1.00/1.10 cycles from completion to barrier arrival.
Investigate memory/reply service before changing the completion floor; these
transaction intervals are not exclusive kernel stalls. Full table and cycle
window are in `calibration.md`. Timed multicast now matches baseline at
278371 cycles; 8K/16K post-last-issue means are 2299/2664 cycles, followed
by 100/23.78 mean cycles to barrier arrival (the explicit multicast path
retains its configured 100-cycle delay). No-TMA and final correctness remain
pending. The opt-in DSM counter hook/checker now rejects disabled DSM and
retains checks under Python optimization; normal and `-O` self-tests pass.
The candidate build passes isolated global-zero and mapped-positive runtime
controls; explicit tensor-multicast GEMM routing remains under test.
No-TMA timed scheduler deltas are documented in `calibration.md`: small
SB_SpInt share is 67.08%, versus 41.36%/42.75% for larger shapes. Reuse a
single-SM/single-warp issue trace next to identify dependency-stalled PCs;
do not add instrumentation or treat sample shares as elapsed-cycle shares.
This existing-hook diagnostic is now terminal in
`/tmp/h200-small-issue-trace.9OsQho/results`, with its issue trace at
`/tmp/h200-small-issue-trace.9OsQho/issue.log`. It uses the production 132SM
config, pinned router-fixed library, OMP4, matched small case, warmup=1 and
samples=1; SM=0/warp=0 and unlimited trace rows. No rebuild or knob change.
Exit 0 in 1918.225 host seconds; all correctness checks pass and all three
event times exactly reproduce baseline. Timed no-TMA interval 1179514–1411779
has 148158 traced SP_INT stalls; top PCs are half-loads following queued
zero-MOVs, not proof of long MOV execution. Full extraction is documented in
`calibration.md`. Do not restart this terminal run; the K2K transaction trace
has also completed on the old library. The separate candidate build does not replace
that library and is already under validation; do not repeat its build/tests.

GEMM measurement update: `/tmp/h200-gemm-bounded-wait` is terminal (square
host timeout); the square-only retry `/tmp/h200-gemm-square-extended` is
now terminal, functional PASS in 9913.894 host seconds. Square-shape event
errors are +69.821% unicast, +37.638% multicast, -9.452% no-TMA. It used
warmup=0 and remains diagnostic; the first two fail the timing gate.
The completed `/tmp/h200-gemm-small-warm1` run matches the profiling source's
one immediate warmup and reports small-shape time errors +18.557% unicast,
+17.078% multicast, -17.315% no-TMA: all fail despite correctness passing.
The former zero-warmup multicast pass is not robust. Future harness GEMMs
retain `--warmup 1`; do not mix warmup procedures into final acceptance.
Read `calibration.md` for full results/provenance and the L2 service audit.

Calibration follow-up (2026-09-07): audit bounded `mbarrier.try_wait`
suspension before further remote-wait tuning. Job 2119329 vendor WaitFalse
has a repeatable 7755 cycles/op slope. Explicit ns-to-cycle conversion is
now fixed with a failing-before/passing-after expiry regression. The 4320 ns
default-timeout candidate passes bounded expiry, early wakeup, and 16 local
regressions; matched false-chain gap is 0.090%. Remote owner/E2E gaps remain
37.264%/45.391%, so do not declare remote calibration complete. All nine
GEMMs are being checked in `/tmp/h200-gemm-bounded-wait`: the 256x8192x2048
shape is functionally complete, with CUDA-event errors +15.998% unicast
(FAIL), +8.085% multicast (PASS), -16.615% no-TMA (FAIL). The two larger
shapes have since finished (see the update above and `calibration.md`);
no all-nine timing acceptance yet. No-hint
nonuniform local waits retain immediate polling. The standalone postdominator assertion
was traced to compiler-hoisted non-volatile test handshake loads; corrected
polling now passes four focused remote/timeout tests. Still validate the
remote timing independently. `RemoteExpectAndCompleteTx` now passes with a
bounded observation window and a deliberate missing-expect negative control
(`/tmp/h200-mbar-observation-negative.log`); this resolves its former timeout.
See the scope clarification in
[`calibration.md`](calibration.md). Keep successful-wait latency separate;
require finite-expiry, early-wakeup and all nine GEMM timing comparisons
after any fix. This is unfinished calibration work, not an accepted fit.

The historical pre-memory-fix 1-GiB HBM batch finished with payload checks passing on all
three paths: normal load -5.991%, cp.async +9.433%, TMA +8.752% bandwidth
error against job 2119329. Evidence is in
`/tmp/h200-calibration-hbm-matched/comparison/comparison.csv`. The batch spans
library revisions: repeat on one frozen build before final acceptance, and
do not interpret streaming bandwidth as validation of inherited bank timings
or the DRAM residual. No HBM knob change was needed for these three results.
The separate matched L2 normal-load diagnostic completed with correct
payload but fails timing: 405292 vs 358536 cycles (+13.041%), or -11.536%
bandwidth. See `/tmp/h200-l2-normal-comparison/comparison.csv`; the 414050
warmup cycles must not be used as its measured result. L2 remains open.

1. Pick the **lowest undone ID** whose **Prereqs** are all `[x]`.
2. Read the **Read first** links. Do not skim unrelated chapters.
3. Implement **only** that ID. Do not mix rename + fabric + calibration in one patch.
4. Run the **Verify** commands.
5. Flip `[ ]` → `[x]`, add date + one-line evidence (test names, config).
6. Stop. Leave follow-ons to their own IDs. Do **not** `git commit`. Reply with the list of modified files and a draft commit message. The human reviews and commits.

### ID rules

- Never reuse an ID.
- Do not re-open **A-*** items marked done unless you have new failing evidence.
- New work gets `B-*` or `A-F4-*`, not a revived `L2-1`.
- English only in checkboxes and close-out notes.
- Checklist IDs (`B1`, `Track B`, `A-F1`, …) live **only** in this file. Do not put them in source comments, config comments, or commit messages. Name the change in product terms (GPC rename, per-SM global NoC, delay-line hop, …).

### Suggested order

```text
C1 (docs — this rewrite)
  → B0 baseline
  → B1 naming / topology
  → B2 per-SM global NoC
  → B3a transport primitives
  → B3b dsm_fabric_t
  → B3c endpoint protocol
  → B4 functional DSM on fabric
  → B5 scoreboard + SRAM service   // remote load = local SMEM load
  → B6 full-scale H200 preset (not fabric-only)
       B6a–c  (done) fabric latency/slopes
       B6f    mixed 132-SM GPC map + occupancy 2048/32 + CLUSTER132
       B6g    one exclusive H200 job = standalone dump of every kernel
       B6d    TMA GEMM sim gate (config CLUSTER132)
       B6h    fit non-fabric knobs from B6g CSV
       B6e    freeze comments on SM90_H200, CLUSTER132, reduced 16×2
  → B-DEPR deprecate delay-line knobs
  → B7 per-bank SRAM (optional)
  → B9 research extras (not first delivery)
```

`A-F4` (`red`/`red.async`) is independent and may wait until a dump emits those ops.

---

# Track A — Functional baseline (delay-line era)

Shipped on `cluster_cta2_support`. Do not re-implement. Tests: [`tests.md`](tests.md).

- [x] **A-F1** Land delay-line NoC + cluster launch + TMA cluster + DSM + remote mbar. Evidence: ClusterNoc*, DsmTest*, MbarrierClusterTest*, TMAClusterOneProducer* on SM120 reduced and SM90 H200 reduced.
- [x] **A-F2** Used mbarrier ops + `try_wait` timeout dest-pred.
- [x] **A-F3** `mapa` / CG DSM map and `barrier.cluster.arrive` + `barrier.cluster.wait` are supported. Cluster barriers wait for every active warp in every CTA of the launched TB cluster; this was promoted from the former non-goal because the upstream bandwidth kernels use `cluster.sync()`.
- [ ] **A-F4** PTX `red` / `red.async`. Still `inst_not_implemented`. nvcc 12.8 `atomicAdd`+`mapa` emits generic `atom.add` (already works). Implement only if a real dump emits `red`.
- [x] **A-F5** `mapa` of inactive rank **aborts**.
- [x] **A-F6** TMA corners: used paths work; 96B swizzle / unused tensormap abort.
- [x] **A-F7** Hang preventers (bare peer spin; mixed `bar.sync`+`try_wait`).
- [x] **A-F8** Integration/unit surface listed in [`tests.md`](tests.md).
- [x] **A-F9** `-gpgpu_dsm_store_immediate` default 0 (write on deliver).

---

# Track B — Target GPC / DSM fabric

## B0 — Baseline and evidence labels

- [x] **B0** Freeze what “no change” means before refactors. Closed 2026-08-25. No C++ / knob change.

**Read first:** [`evidence.md`](evidence.md), [`tests.md`](tests.md).

**Why:** Later phases must prove they did not silently drop L2 bandwidth or break DsmTest.

**Work:**

1. Record current `icnt` shader node count vs `n_clusters` vs total SMs for `SM90_H200_REDUCED_CLUSTER4x4` (later replaced by `CLUSTER16x2`) and one SM120 reduced cluster config.
2. Run the functional filters in [`tests.md`](tests.md) §1; paste PASS lines in the close-out.
3. Pin the `dsm_bw` commit hash from [`evidence.md`](evidence.md) in the close-out (already in that file).
4. In any new comment/preset, tag numbers `measured` / `patent` / `inferred` / `unresolved`.

**Do not:** Change knobs or C++.

**Verify:** The same filters PASS on a clean tree.

**Exit:** A short note under this item with node counts + test commands + date.

**Prereqs:** C1. **Next:** B1.

**Close-out (2026-08-25):** Observed-in-tree topology (config knobs + existing `icnt_create(m_shader_config->n_simt_clusters, …)` in `src/gpgpu-sim/gpu-sim.cc`; shader nodes = `-gpgpu_n_clusters` / `n_simt_clusters`, **not** total SMs). Not H200 hardware fact. No `measured` / `patent` / `inferred` / `unresolved` numbers copied from [`evidence.md`](evidence.md).

| Config | n_clusters | n_cores_per_cluster | total SMs | icnt shader nodes |
|--------|-----------:|--------------------:|----------:|------------------:|
| `SM90_H200_REDUCED_CLUSTER4x4` (superseded by `CLUSTER16x2`) | 4 | 4 | 16 | 4 |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | 1 | 2 | 2 | 1 |
| `SM120_RTX5090_REDUCED_CLUSTER4x4` | 4 | 4 | 16 | 4 |

total SMs = `n_clusters * n_cores_per_cluster`. `g_icnt_n_shader` is set from that first `icnt_create` argument.

Pinned `dsm_bw` hash already recorded in [`evidence.md`](evidence.md): `4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd` (`Add standalone H200 DSM bandwidth benchmark`).

[`tests.md`](tests.md) §1 commands (plus a second `ClusterNoc*` unit run):

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 build test --target sm120 --group integration
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 run test --target sm120 --group integration \
  "*ClusterLaunch*:*TMACluster*:*MultiCluster*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures; skip allowed):

```text
# unit ClusterNoc* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 8 tests.
✓ test/sm120/unit passed!

# SM120 integration (SM120_RTX5090_REDUCED_CLUSTER4x4)
[  PASSED  ] 31 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] ClusterLaunchApiTest.ExLaunch_ClusterLargerThanPhysical_Fails
✓ Tests passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

The SM120 skip is `GTEST_SKIP` because that case needs `n_cores_per_cluster == 1`; 4x4 has m=4. Not a product failure.

---

## B1 — Naming and `gpu_topology_t`

- [x] **B1** Zero-behavior rename: physical cluster → GPC; introduce topology table with PG'd slots. Closed 2026-08-25.

**Read first:** [`architecture.md`](architecture.md) §§2–5.

**Why:** Stop using `tpc`/`sid` as GPC. Enable non-uniform / PG'd SM maps without `sid % n` math.

**Files (expected):**

- New: `src/gpgpu-sim/gpu_topology.{h,cc}` (names may match style)
- `src/gpgpu-sim/gpu-sim.{h,cc}`, `shader.{h,cc}`, `abstract_hardware_model.h`
- `mem_fetch` requester fields
- `shader_core_config::reg_options` aliases
- Unit: `test/src/unit/` topology round-trip + PG map
- Docs: keep using `gpc_t` in comments as you rename

**Work:**

1. Add `gpu_topology_t` / `sm_location_t`. **All** SM↔GPC↔slot maps go through it.
2. CPC = 6 slots; `cpcs_per_gpc` default 3; extra slots **PG'd** when `num_sms_per_gpc` is smaller.
3. Typedef/alias `simt_core_cluster` → `gpc_t` **or** rename with a temporary typedef so callers compile. Prefer real rename if grep-complete.
4. Add `-gpgpu_num_gpcs` / `-gpgpu_num_sms_per_gpc`; keep `-gpgpu_n_clusters` / `-gpgpu_n_cores_per_cluster` as aliases. Conflict → abort.
5. `mem_fetch`: `m_requester_sm_id`; delete redundant `m_tpc` if it duplicates GPC/SM.
6. TB-cluster fields: `tb_cluster_*` prefix where you touch them.
7. Forward old getters with deprecation comments; do not add **new** `sid`/`tpc` APIs.

**Do not:** Change icnt node count (that is B2). Do not change DSM hop behavior. Do not model TPCARB.

**Verify:**

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

Plus new topology unit tests: round trip; PG'd slot has no `shader_core_ctx`; `sm_id %` is not used outside topology + documented shaper.

**Exit:** Baseline tests bit-identical in **functional** results; no interconnect node count change. Close-out lists the alias knobs.

**Prereqs:** B0. **Next:** B2.

**Close-out (2026-08-25):** Physical cluster is `gpc_t` (`typedef simt_core_cluster gpc_t`). SM↔GPC↔CPC-slot maps go through `gpu_topology_t` (CPC = 6 slots; `-gpgpu_dsm_cpcs_per_gpc` default 3; extra slots PG'd). `create_shader_core_ctx` still allocates only enabled local SMs. `icnt_create` first argument remains GPC count (`n_simt_clusters`). DSM delay-line hops unchanged. No TPCARB.

Alias knobs (conflict at start-up aborts):

| Old (kept) | New |
|------------|-----|
| `-gpgpu_n_clusters` | `-gpgpu_num_gpcs` |
| `-gpgpu_n_cores_per_cluster` | `-gpgpu_num_sms_per_gpc` |

`mem_fetch` requester is `m_requester_sm_id` (`get_sid()` / `get_tpc()` deprecated). TB-cluster storage touched here: `m_cta_tb_cluster_group`, `m_cta_tb_cluster_rank`, `m_next_tb_cluster_group_id`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*:GpuTopology*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures):

```text
# unit ClusterNoc* + GpuTopology* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 17 tests.
✓ test/sm120/unit passed!

# GpuTopology* rerun after exclusive-end SM range
[  PASSED  ] 9 tests.
✓ test/sm120/unit passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

---

## B2 — Global NoC endpoint per SM

- [x] **B2** One shader icnt node per **enabled SM**, not per GPC. Closed 2026-08-26.

**Read first:** [`architecture.md`](architecture.md) §4.

**Why:** Grouping SMs into a GPC must not cut L2/global bandwidth by `m`.

**Files:** `gpu-sim.cc` `icnt_create`; cluster/GPC icnt push/pop; `m_response_fifo` → per-SM; shader memory cycle ejection; stats; SST/gem5 adapter **or** a comment that they are unsupported until an adapter exists.

**Work:**

1. `icnt_create(num_sms, num_l2_subpartitions)` (enabled SM count).
2. SM injects from `global_sm_node_id(sm_id)`; L2 replies to that node.
3. GPC walks **each member SM** ejection port.
4. Per-SM response FIFO and ingress/dispatch budgets.
5. Ordinary CTA issue width restored per member SM.
6. Stats: per-SM and per-GPC, not “per cluster node”.

**Do not:** Put DSM packets on this icnt. Do not use `gpc_id` as a node.

**Verify:** Two configs, **same total SMs**, different GPC sizes (e.g. 16 SMs as 4×4 vs 16×1): shader endpoint count equal; a local L2-hit bandwidth microbench (or existing memory test) does not drop. Functional cluster filters still PASS.

**Exit:** Written table: config A/B, `num_sms`, `icnt` shader nodes, note on L2 roofline.

**Prereqs:** B1. **Next:** B3a.

**Close-out (2026-08-26):** `icnt_create` is sized from enabled SMs (`num_shader()`). `global_sm_node_id(sm)` is the SM; `global_l2_node_id` starts at `num_sms`. Each GPC ejects per member SM into a per-SM response FIFO. Ordinary CTA issue may place one CTA on each member SM per cycle. Incoming interconnect stats are counted per SM and summed per GPC. SST and gem5 still index one shader port per GPC and are unsupported until a per-SM adapter exists. DSM stays off this interconnect.

| Config / map | n_gpcs | sms/gpc | num_sms | icnt shader nodes |
|--------------|-------:|--------:|--------:|------------------:|
| `SM90_H200_REDUCED_CLUSTER16x2` | 2 | 16 | 32 | 32 |
| `gpu_topology_t::build(16, 1, 3)` | 16 | 1 | 16 | 16 |
| `SM120_RTX5090_REDUCED_CLUSTER2x1` | 1 | 2 | 2 | 2 |

L2 roofline: no dual-config kernel run. Independent SM ports (`GpuTopology.ShaderIcntNodeCountIndependentOfGpcGrouping`, `LocalInterconnectTest.PerSmShaderPortsDoNotShareBuffer`) mean packing SMs into fewer GPCs cannot cut L2 ports by `m`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "GpuTopology*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "ClusterNoc*:GpuTopology*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration \
  "DsmTest.*:MbarrierClusterTest.*:TMAClusterOneProducer*"
```

PASS evidence (0 failures):

```text
# unit GpuTopology* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 12 tests.
✓ test/sm120/unit passed!

# unit LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!

# unit ClusterNoc* + GpuTopology* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 20 tests.
✓ test/sm120/unit passed!

# H200 DSM / remote mbar / OneProducer (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 20 tests.
✓ Tests passed!
```

---

## B3a — Shared transport primitives

- [x] **B3a** Bounded VOQ, RR arbiter, flit-credit counters, stats — **no** DSM policy. Closed 2026-08-27.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §§6 and 9.

**Why:** DSM fabric needs queues/credits. `xbar_router` assumes request/reply subnets and whole-packet moves.

**Work:**

1. New small headers (plan-v2 names OK): `transport_packet_metadata_t`, `bounded_voq_t`, `round_robin_arbiter_t`, `interconnect_sink_t`, occupancy/stall/latency stats. Occupancy in **payload flits**.
2. Unit tests: occupancy, HOL per VOQ, credit borrow must not cross a second queue.
3. Optionally wrap `LocalInterconnect` **without** changing its cycle behavior.

**Do not:** Instantiate two xbars as VCs. Do not put shader/memory endpoint assumptions in the primitive.

**Verify:** Existing local_interconnect / memory tests unchanged if you wrap them; new unit tests PASS.

**Exit:** Primitives have **no** `REQ_NET`/`REPLY_NET` in their API.

**Prereqs:** B1 (B2 preferred). **Next:** B3b.

**Close-out (2026-08-27):** Header-only primitives in `src/gpgpu-sim/transport.h`: `transport_packet_metadata_t`, `bounded_voq_t`, `flit_credit_counters_t`, `round_robin_arbiter_t`, `interconnect_sink_t`, `interconnect_stats_t`. Occupancy unit = **payload flits**. API has no `REQ_NET`/`REPLY_NET` and no shader/memory endpoint roles. `LocalInterconnect` was not wrapped.

Unit tests (`Transport*`): `OccupancyCountsPayloadFlits`, `FullDestDoesNotHolSibling`, `CreditTakeDoesNotCrossQueue`, `RoundRobinRotates`, `SinkOccupancyIsPayloadFlits`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "Transport*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
```

PASS evidence (0 failures):

```text
# unit Transport* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!

# unit LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 5 tests.
✓ test/sm120/unit passed!
```

---

## B3b — `dsm_fabric_t`

- [x] **B3b** Network-only DSM transport: VCs, 32 B **payload** flits, shaper, GPCMMU hash, GX count, CPC 6→4, PG slots.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) **all**, [`knobs.md`](knobs.md) §3, [`evidence.md`](evidence.md).

**Why:** This is the physical model. Delay-line hop cannot reproduce directional VC sharing.

**Hard requirements (do not “simplify away”):**

1. **Payload per grant = 32 B** (`-gpgpu_dsm_flit_payload_bytes`, default 32). Header/metadata may exist on a real package; **do not** charge extra lane occupancy.
2. Two VCs: **independent queues and credits**; **shared** physical lane scheduler.
3. `-gpgpu_dsm_gx_planes` default **2**. Never name planes request/response.
4. GPCMMU = **hash** `(addr, src, dst, uid) → (gx_plane, lane)`. Deterministic, replaceable.
5. Shaper configurable. Implement **all three**: `fixed_tdm` (plan-v2 CPC slots `{0,1,2,3}/{2,3,4,5}/{0,1,4,5}`), `skip_mod` (skip 1 of 3, index `sm_id` or `cpc_slot`), `hard_rate_cap`. Idle eligible slots are **wasted**. Do not claim one policy is Hopper silicon.
6. CPC: 6 slots, 4 lanes. PG'd slots never inject, never take eligibility.
7. Destination VOQ so one blocked dest does not HOL another dest on the same VC.
8. `can_inject` false → **LSU stall**, not “write immediately”.
9. Control packets (`read_command`, `write_ack`) occupy **one** payload-flit slot each.
10. 128 B data = **four** grants.

**Files:** new `dsm_fabric.{h,cc}`, config knobs, per-`gpc_t` instance, network-only gtests, stats dump.

**Work (suggested sequence inside this ID):**

1. Packet struct: plan-v2 §8.1 fields (`packet_id`, `transaction_id`, network src/dst, transaction requester/target, `vc`, `packet_class`, `payload_bytes`, `total_flits`, `remaining_flits`, `route_lane`, created/injected/tail cycles).
2. Ingress `[src][vc][dst]`, egress `[dst][vc]`, credits per dest VC.
3. Shaper + CPC lane arbiter + GX select via hash.
4. API: `can_inject` / `inject` / `top` / `pop` / `cycle` / `busy` / `display_state` (plan-v2 §9.1).
5. Isolation: GPC A fabric must not see GPC B queues.
6. Stats in [`dsm_fabric.md`](dsm_fabric.md) §7.1.

**Do not:** Complete RF/scoreboard here (B5). Do not coalesce ACKs here (B3c). Do not hook TMA multicast yet (B8) except injecting unicast packets in unit tests. Do not model TPCARB. Do not hard-code 21.0; let 2/3 × 32 fall out of the shaper.

**Verify (network-only tests — add them):**

- One SM, others idle: ~21.33 B payload/cycle cap (2 grants per 3 cycles × 32 B).
- Idle neighbor does not raise that rate.
- Same-dir request data + response data share the cap.
- Opposite dirs can exceed one-dir cap.
- `read_command` is one reverse flit.
- Request buffer full does not consume response credits.
- `gx_planes=1` reduces routes/bandwidth vs default 2.
- PG'd slot never sends.
- `skip_mod` and `fixed_tdm` both enforce ~2/3 rate; their co-eligible sets **differ**.

**Exit:** Those tests PASS. Stats print eligibility used vs wasted.

**Prereqs:** B3a, B1. **Next:** B3c.

**Close-out (2026-08-27):** `dsm_fabric_t` in `src/gpgpu-sim/dsm_fabric.{h,cc}`. Payload grant 32 B (`-gpgpu_dsm_flit_payload_bytes`, alias `-gpgpu_dsm_flit_bytes`). Request/response VCs have independent VOQs and credits and share one physical-lane scheduler. Destination VOQ, GPCMMU hash `(addr,src,dst,uid)→(gx_plane,lane)`, GX default 2, CPC 6→4, PG slots never eligible. Shapers `skip_mod` / `fixed_tdm` / `hard_rate_cap`. Per-GPC instance on `simt_core_cluster` (delay-line remains the functional SM↔SM path). `can_inject` false refuses inject.

Unit tests (`DsmFabric*`): `OneSmIdleNeighborsCap`, `IdleNeighborDoesNotRaiseRate`, `SameDirRequestResponseShareCap`, `OppositeDirsExceedOneDirCap`, `ReadCommandIsOneReverseFlit`, `RequestFullDoesNotConsumeResponseCredits`, `DestVoqBlockedDoesNotHolSibling`, `Data128BNeedsFourGrants`, `CanInjectFalseRefusesInject`, `GxPlanesOneReducesRoutesVsTwo`, `PgdSlotNeverSends`, `SkipModAndFixedTdmCoEligibleDiffer`, `HardRateCapAveragesTwoThirds`, `TwoFabricsIsolated`, `DisplayStateShowsUsedAndWasted`.

Verify:

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "DsmFabric*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "Transport*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 run test --target sm120 --group unit "LocalInterconnect*"
```

PASS evidence (0 failures):

```text
# unit DsmFabric* (run 1 and rerun, SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 15 tests.
✓ test/sm120/unit passed!

# unit Transport* + LocalInterconnect* (SM120_RTX5090_REDUCED_CLUSTER2x1)
[  PASSED  ] 10 tests.
✓ test/sm120/unit passed!

# display_state (DsmFabric.DisplayStateShowsUsedAndWasted)
eligibility used=32 wasted=160 slots=192
```

---

## B3c — Endpoint protocol

- [x] **B3c** Outstanding transactions + coalesced **write_ack** + read_command/data correlation.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §§3 and 7.

**Work:**

1. `dsm_endpoint_protocol_t` per enabled SM.
2. Track tx id → packets remaining.
3. ACK debt per original requester; flush on threshold, timeout, or idle response path; one-flit `write_ack` with count.
4. Remote load always pairs `read_command` + `read_data` (no data coalesce).
5. Outstanding cap backpressure via `can_inject`.
6. Dump: plan-v2 §17 transaction fields + ACK debt, coalescing ratio, timeout flush, outstanding count.

**Do not:** Put coalescing inside the lane arbiter. Do not treat outstanding cap as VC credit. Do not skip SRAM service (still a stub until B5; network-only tests may loop back without SRAM).

**Verify:** Finite buffers + heavy stores: no deadlock; ACK count < store packet count (coalesce). Symmetric-store unit scenario shows **far fewer** reverse flits than payload flits.

**Exit:** Tests named in close-out. **Next:** B4.

**Prereqs:** B3b.

**Close-out (2026-08-27):** `dsm_endpoint_protocol_t` in `src/gpgpu-sim/dsm_endpoint.{h,cc}`. Per-SM outstanding window (`-gpgpu_dsm_max_outstanding_per_sm`, default 16) is not a VC credit. ACK debt is per original requester; one-flit `write_ack` on the response VC flushes on threshold (default 4), timeout (default 64 cycles), or idle response path. Remote load always emits `read_command` + `read_data` (no data coalescing). SRAM service is a stub. Delay-line remains the functional SM↔SM path.

Unit tests (`DsmEndpoint*`, topology 2 GPCs × 16 SMs): `ReducedCluster16x2Packing`, `HeavyStoresDrainNoDeadlock`, `AckPacketsFewerThanStores`, `SymmetricStoresReverseFlitsFarFewer`, `RemoteLoadPairsCommandAndData`, `OutstandingFullDoesNotTouchVcCredits`, `AckFromOneTargetDoesNotRetireSiblingStore`, `ThresholdFlushProducesWriteAck`, `TimeoutFlushProducesWriteAck`, `IdleResponseFlushProducesWriteAck`, `DumpShowsOutstandingDebtRatioTimeout`.

Verify:

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 --group unit "DsmEndpoint*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 --group unit "DsmFabric*"
```

PASS evidence (0 failures):

```text
# unit DsmEndpoint* (run 1 and rerun, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 11 tests.
✓ test/sm120/unit passed!

# unit DsmFabric* regress (SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 15 tests.
✓ test/sm120/unit passed!
```

---

## B4 — Functional DSM on the fabric

- [x] **B4** Ordinary `.shared::cluster` ld/st/atom use resolver + fabric packets. Existing DsmTest* PASS. Closed 2026-08-27.

**Read first:** [`programming_model.md`](programming_model.md) §2, [`architecture.md`](architecture.md) §7.

**Work:**

1. Unified `resolve_tb_cluster_rank` used by DSM, and called from TMA/mbar paths even if those still inject delay-line messages until B8.
2. Keep logical generic address; strip owner bits for target smem offset.
3. Mixed-target warp: group by `(target_sm, cta_slot)`, join.
4. Remote st → `write_data` packets; remote ld → `read_command` (data path completion in B5 may still be temporary).
5. Illegal cross-GPC / dead rank: abort (same as today).
6. Dual-run: delay-line still available behind a knob until B-DEPR.

**Do not:** Fill RF in `ld_impl` for the fabric path (if you must keep delay-line RF fill, gate it on the old knob only). Do not implement `red`.

**Verify:** `DsmTest.*` `MbarrierClusterTest.*` on H200 reduced. New tests: mixed-lane target; two TB-cluster groups isolated; cross-GPC reject.

**Exit:** Fabric path is default on H200 reduced **or** documented dual-path with fabric-on tests green.

**Prereqs:** B3c. **Next:** B5.

**Close-out (2026-08-27):** One TB-cluster resolver (`resolve_tb_cluster_rank` / `resolve_tb_cluster_owner_sm` in `src/gpgpu-sim/flash/tb_cluster.{h,cc}`) used by DSM, TMA (`for_each_tb_cluster_peer`), and remote mbarrier. Logical generic address kept; owner bits stripped for the target smem offset. Fabric-on remote st injects `write_data`, remote ld injects `read_command`, remote `atom.add` RMW on owner smem. Mixed-target warps group by `(target_sm, cta_slot)` and join. Fabric-path `ld_impl` does not fill dest RF. Cross-GPC / dead rank abort. Fabric default on `SM90_H200_REDUCED_CLUSTER16x2` via `-gpgpu_dsm_enable 1`; delay-line DSM via `0`. TMA multicast and remote mbarrier still use the delay-line.

Verify (run twice, 0 failures):

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration "DsmTest.*:MbarrierClusterTest.*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group unit "DsmEndpoint*:DsmFabric*"
```

PASS evidence:

```text
# integration DsmTest.*:MbarrierClusterTest.* (run 1 and 2, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 23 tests.
# includes DsmTest.MixedLaneTargetJoin, TwoTbClusterGroupsIsolated,
# VectorRemoteLdSt, CrossGpcReject; 9 MbarrierClusterTest.*

# unit DsmEndpoint*:DsmFabric* (run 1 and 2, SM90_H200_REDUCED_CLUSTER16x2)
[  PASSED  ] 28 tests.
✓ test/sm120/unit passed!
```

---

## B5 — Remote load = local SMEM load + SRAM service

- [x] **B5** Scoreboard and LDST writeback for remote DSM **match local `ld.shared`**. Target SRAM is a real service. Closed 2026-08-27.

**Read first:** [`pipeline.md`](pipeline.md) **all**.

**Why:** Supervisor: remote load must behave almost the same as local SMEM load. Variable fabric delay makes execute-time RF fill **wrong**.

**Work:**

1. Issue remote load: `reserveRegisters` as **shared** (`PROD_MEM_SHARED`), same as local smem.
2. **Do not** write destination regs in `ld_impl` on the fabric path.
3. Instruction stays in the **LD/ST shared** pipe until data is ready (pending shared load / same writeback as local smem).
4. After `read_data` tail **and** target `shared_memory_service_t` grant: RF write + `releaseRegisters` on that writeback path.
5. A following ALU must `checkCollision` until then.
6. Stores: owner write only on SRAM grant; ACK via B3c; CTA cannot exit with outstanding DSM.
7. `shared_memory_service_t`: local LSU, TMA local, DSM ingress share `gpgpu_shmem_bytes_per_cycle` (or named successor). Two-phase grant.
8. Zero-cycle bypass forbidden: arrival, SRAM, response inject take ≥1 cycle each as in [`architecture.md`](architecture.md) §6.

**Do not:** Add a DSM-specific scoreboard class unless `Scoreboard` literally cannot hold dest regs (it can). Do not classify remote DSM as `PROD_MEM_GLOBAL` without new evidence. Do not keep `dispatch_delay = 2×hop` on the fabric path. Do not sample peer smem at execute “for convenience.”

**Verify:**

- Unit/integration: warp issues `ld` remote then `add` using that dest; `add` issues only after response writeback.
- Compare issue/writeback sequencing to a **local** `ld.shared` of the same footprint (same unit, same scoreboard API).
- `DsmTest.RemoteLoad*` still PASS (data correct).
- Local-only smem bandwidth test does not starve forever when DSM is idle.

**Exit:** Written note: file:line for reserve, RF write, release. **Next:** B6 (calibration) and B8 (TMA/mbar packets).

**Prereqs:** B4.

**Close-out (2026-08-27):** Remote cluster loads reserve dest as shared (`Scoreboard::reserveRegisters` + `reclassifyShared` → `PROD_MEM_SHARED`), stay in the LD/ST shared pipe until `read_data` + SRAM grant, then RF write + `releaseRegister` on `ldst_unit::writeback` (same path as local `ld.shared`). Fabric-path `ld_impl` does not fill dest RF. `dispatch_delay = 2×hop` is delay-line only. Per-SM `shared_memory_service_t` (`-gpgpu_shmem_bytes_per_cycle`, 0=unlimited) two-phase grant shared by local LSU, TMA landing, and DSM ingress; arrival / grant / inject each ≥1 cycle. Stores write owner smem only on grant. CTA exit waits for outstanding DSM.

Verify (run twice, 0 failures, `SM90_H200_REDUCED_CLUSTER16x2`):

```bash
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group integration "DsmTest.*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER16x2 \
  run test --target sm120 --group unit "SmemService*:DsmEndpoint*"
```

PASS evidence:

```text
# integration DsmTest.* (run 1 and 2)
[  PASSED  ] 16 tests.
# includes RemoteLoadFromPeer_TwoCtas, RemoteLoadThenAdd, LocalSharedLoadThenAdd

# unit SmemService*:DsmEndpoint* (run 1 and 2)
[  PASSED  ] 20 tests.
✓ test/sm120/unit passed!
```

Path note: reserve `shader.cc:2191` / `1670`; RF write `shader.cc:4371` + `6876`; release `shader.cc:4388`.

---

## B6 — Full-scale H200 config (SM/memory/ALU + cluster fabric + GEMM)

- [ ] **B6** Close when **B6a–B6h** that apply are `[x]`. Not a single GB/s point. Not fabric-only.

Future published cycle numbers use **`SM90_H200_CLUSTER132`** (inferred
6×16+2×18 = 132 SMs). Functional filters stay on
`SM90_H200_REDUCED_CLUSTER16x2`.

**Read first:** [`calibration.md`](calibration.md) (kernels, H200 baselines, simulator results), [`evidence.md`](evidence.md), [`tests.md`](tests.md) §3.

**Why:** First delivery needs cluster features **functionally correct and cycle-accurate**. Supervisor gate: for each calibration kernel,

```text
|T_sim − T_H200| / T_H200  <  10%
```

`T` uses each matched measurement's envelope and unit: CUDA-event ms for
GEMM, clock-counter cycles for instruction probes, globaltimer ns (or
explicitly converted cycles) for remote E2E, and size slopes or bytes/s for
bandwidth. Reciprocal time/bandwidth metrics are not independent tests.

**Hard rules (every B6\* run):**

1. Config: **`SM90_H200_CLUSTER132`** (inferred 6 GPCs × 16 SMs + 2 GPCs × 18 SMs = 132, fabric on). **Not** `SM90_H200_REDUCED_CLUSTER16x2` (functional only). **Not** shipped `SM90_H200` (132 × 1).
2. `export OMP_NUM_THREADS=4`.
3. Cycle-gate kernels: inner loop count so the timed region is **~1e5 cycles** (8e4–1.2e5). Grow **repeats**, not unique smem.
4. Slope kernels (`dsm_bw` 16–96 KiB) stay unique-address; they may be shorter than 1e5 cycles.
5. Kernel sources live in git-ignored `calibration/kernels/` (refresh: `bash scripts/sync_calibration_kernels.sh`). Do not `git add` that tree.
6. Fill Sim / Error columns in [`calibration.md`](calibration.md) as you go. Record config, commit, `OMP_NUM_THREADS`, iteration count.

**Do not (applies to all B6\*):** Reintroduce `-gpgpu_dsm_bytes_per_cycle` as the model. Overfit GPCMMU hash to one camping trace. Publish numbers from the reduced 32-SM config. Stamp H200 numbers as Blackwell fact. Mix a large knob retune with an unrelated rename.

**Prereqs:** B5 and remote-mbarrier fabric support. TMA multicast is intentionally outside the fabric. **Next:** B-DEPR after B6e.

**Hardware status (2026-09-07):** job 2119329 is partially accepted. Its
low-level measurements are usable, but the 4096^3 GEMM timed launch deadlocked;
a corrective rerun is required before the preset is frozen. Previous Slurm
measurements remain superseded. Do not close a B6 item
from them.

**Simulator status (2026-09-07):** calibration is still in progress. The suite
now includes all nine requested GEMM comparisons and the vendor DSM size
sweep. Do not use the earlier functional pass count as calibrated evidence.
TMA's size table has been replaced by a linear completion floor; full multicast
payload checks pass, with end-to-end errors below 6.26%. GEMM outputs pass
for 256/8192/2048, but its first diagnostic timing is invalid because CUDA
events used host time. Device-cycle event timing now passes its regression
on SM120 and the 132-SM H200 config; the launcher uses the configured clock
for unit conversion. Resume fingerprints include simulator libraries, and
the multi-launch cases have explicit cumulative cycle ceilings. DSM/HBM
validation remains open. Shared-to-shared TMA previously bypassed DSM
contention; mapped copies now use the existing DSM transport and pass all
eight vendor smoke cases. Both false-positive watchdog patterns (finite
address streams and active producers alongside bar.sync) have 132-SM
regressions, while genuine polling/mixed-wait death tests remain passing.
The full DSM sweep now passes 12/12 non-exempt slopes (<7.95% error); BW7
is -9.49% and exempt BW8 is -42.35%. HBM cp.async/TMA diagnostics are
-1.49%/+0.20%, while normal loads remain -13.32% (all use 256 MiB versus
hardware's 1 GiB). Do not retune the roofline to fix only normal loads.
New regression work found a retired-kernel binding lifetime bug: sparse
cluster launches can leave idle SMs pointing at deleted kernels and bypass
the next launch's delay. The 8700-cycle regression fails at 509 cycles before
the cleanup fix; both post-fix event tests pass on build 13.0. The broad TMA regression
also aborted in Pipelined2InFlight with an invalid shared address. The trace
identified a 32-bit arithmetic carry leaking into a 64-bit address read;
register-width truncation fixes the isolated test on build 14.0. Complete
the remaining address/pipelined-store checks and broader regressions before
declaring regression safety. Build-12.0 GEMM also stalls at 124/128 multicast
CTAs despite completed TMA memory traffic. Signed transaction accounting
now retains completion credits before expect_tx, as PTX requires; the
reproducer plus 24 related regressions pass. The new GEMM run has progressed
past both TMA smoke launches; final timing remains open.
Direct cudaLaunchKernel now creates its missing launch frame (reproduced
cycle-gate host abort; regression passes). Register-width enforcement also
exposed mapa.u32's reliance on out-of-width generic-pointer bits. Compact
owner/offset encoding and mbarrier decoding now pass both 32-/64-bit remote
arrival tests. Full remote probes now complete functionally, but owner wait
and E2E timing are still 28.77% and 39.11% too short. Repeated-init and
remote-barrier cycle gates also remain outside tolerance. Pure local
arrive/try-wait/composite measurements pass; preserve their isolated fits.
Broader checks pass 215/215 unit tests and 23/23 selected arithmetic/matrix
integration tests; see calibration.md for exact logs and scope.
Historical pre-memory-fix checkpoint, not candidate acceptance: the harness
was updated to hardware's 1 GiB HBM stream and 16-CTA DSM matrix.
Earlier 256 MiB/18-CTA numbers remain diagnostics.
The matched 1-GiB normal-load HBM case now passes payload checks and is
5.991% below hardware bandwidth without further HBM tuning. The matched
16-CTA DSM matrix also completes: remote mean is within 4.711%, but local
latency and store visibility remain outside tolerance. A new fenced-store
regression exposed fence.sc being treated as an ordinary instruction;
classification and outstanding-store/DSM draining now fix that reproducer.
Full vendor store-row and broader DSM validation are running; see
calibration.md for exact artifacts and the two-CTA diagnostic's limits.
These validations now pass: 13/13 selected DSM tests; producer fenced-store
span 665/660 simulator/hardware cycles, peer visibility 375/352 ns. No
latency knob was changed. Preserve the store-only comparison and do not
count visibility ns/cycles as independent measurements. Local SMEM and
remote-mbarrier timing still need work.
Global TMA architectural arrival now requires actual memory completion;
the fresh linear-model trace passes all 32 transfer-order checks. Independent
GEMM validation remains open: corrected-clock small-shape baseline errors are
-9.07% (unicast), -11.09% (multicast), and -15.44% (no-TMA), before the new
completion model. All three shapes are being rerun on build 12.0. Fitted
probe agreement alone does not establish predictive accuracy. See
[`calibration.md`](calibration.md) for diagnostic paths and limitations.

**Historical functional rehearsal (2026-09-05):** the reusable 132-SM suite is
`scripts/run_cluster_noc_demo.py`. It mirrors all hardware kernel families,
caps each case at one million cycles, and produces both supervisor-facing and
machine-readable results. The completed representative run reports 33 PASS,
four explicit SKIP, and no FAIL/TIMEOUT/LIMIT results. The full MMA instruction
sweep, DSM calibration sweep, fixed-loop cycle gate, and GEMM are skipped by
default to avoid redundant or unbounded simulator work. Post-fix regressions
pass 27/27 cluster/topology and 7/7 hang-preventer unit tests, 31/31 SM120
cluster integration tests (three expected skips), and 26/26 reduced-H200
DSM/remote-mbarrier/TMA tests.

---

### B6a — Full-chip config and sim harness

- [x] **B6a** Full-chip packed config + build/run path for the git-ignored kernels. Closed 2026-08-28. **No H200 job.** Later B6 work uses `SM90_H200_CLUSTER132`.

**Work:**

1. New full-chip packed config dir: copy SM/memory/clock/latency knobs from `SM90_H200`; fabric on (`-gpgpu_dsm_enable 1`, `-gpgpu_mbarrier_cluster_enable 1`); fabric knobs from [`knobs.md`](knobs.md) §3; no delay-line BPC. README states: full-chip calibration only; reduced 16x2 remains the functional filter. (This ID first landed a uniform 128-SM packing; published default is now `SM90_H200_CLUSTER132`.)
2. Register the config so `./test/run_tests.sh list-configs` lists it.
3. Sim-side Makefile/wrapper under `calibration/` (git-ignored is fine) that (a) sources `setup_environment`, (b) links sim `libcudart`, (c) copies this config into the run cwd, (d) forces `OMP_NUM_THREADS=4`, (e) accepts a loop-count override.
4. Confirm `calibration/kernels/{dsm_bw,tma_bw,h200_probes,hopper_paper}` exist; if not, run `bash scripts/sync_calibration_kernels.sh`. `dsm_bw` and `tma_bw` are verbatim copies of [seanzw/random](https://github.com/seanzw/random); do not rewrite the kernels.

**Verify:** Config starts; a hello kernel on this config prints `gpu_sim_cycle` and uses 4 OpenMP threads. `DsmTest.RemoteLoadFromPeer_TwoCtas` still PASS on **reduced** 16x2 (do not replace functional CI with full-chip packing).

**Exit:** Config path named in [`calibration.md`](calibration.md). **Next:** B6b.

**Prereqs:** B5, B8.

**Close-out (2026-08-28):** Setup only. Landed a full-chip packed fabric-on
config and `scripts/run_calibration_sim.sh`. Kernels are under git-ignored
`calibration/kernels/{dsm_bw,tma_bw,h200_probes,hopper_paper}`. The default
full-chip packing is now `SM90_H200_CLUSTER132`.

---

### B6b — Latency (mbarrier, DSM RTT, TMA e2e)

- [ ] **B6b** Revalidate simulator latency against accepted job-2119329 rows.
  Earlier simulator work remains provisional until matched rows pass the gate.

**Work:** For each accepted latency kernel, compare equivalent hardware and
simulator timing. Tune only the knobs constrained by that kernel. Convert
`globaltimer` measurements using the device frequency reported by the same job.

**Verify:** Every validation-clean latency row has error < 10%. Do not infer a
topology or multicast-fabric model before the new measurements exist.

**Exit:** The latency result table is filled. **Next:** B6c.

**Prereqs:** B6a.

---

### B6c — DSM / TMA bandwidth slopes

- [ ] **B6c** Refit and revalidate against accepted job-2119329 vendor DSM/TMA
  results. Existing simulator knobs remain provisional compatibility defaults.

**Work:** **Copy-paste** `calibration/kernels/dsm_bw/` from seanzw/random (`kernels.cuh`: `load_kernel`, `store_kernel`, `tma_kernel`, `mixed_kernel`). Do **not** rewrite them from `H200_profiling`. Size sweep 16–96 KiB unique addresses, one TB-cluster, checksum after the timer. Record BW1–BW11 (one-way load/store/TMA, duplex, same vs opposite mix, 2/4/8/16 TMA, idle neighbor). Then a **cycle-gate** repeat of one saturated one-way TMA put and one symmetric load at ~1e5 cycles (iteration override only). Tune shaper period, VC depths, ACK threshold/timeout, optional `base_latency`. Idle neighbor must not raise the active SM’s rate. `tma_bw/` from the same repo is GMEM TMA vs L2/HBM (not DSM); run the simple TMA test as a TMA-to-memory check, do not treat it as a DSM slope.

**Verify:** For each accepted vendor size sweep, fit \(1/\beta\) and require the
matching simulator slope to be within 10%. Check directionality, scaling, and
idle-neighbor behavior without assuming their previous values.

**Exit:** The bandwidth result table is filled. **Next:** B6d (GEMM may start
in parallel; do not freeze knobs until B6e).

**Prereqs:** B6a. B6b preferred so latency residuals are not eaten by the shaper.

---

### B6d — Triton autotuned GEMM (unicast vs multicast B)

- [ ] **B6d** Real kernel: TMA + WGMMA, cluster of 2, same autotune winner for uni and mcast.

**Work:**

1. Use the 19 validation-clean job-2119329 shapes; exclude the failed 4096^3
   row until its corrective hardware rerun succeeds.
2. Use the same autotuned tile for unicast, multicast, and no-TMA variants.
3. Simulate the exact accepted cubin/PTX under
   `calibration/kernels/h200_probes/artifacts/` on `SM90_H200_CLUSTER132`.
4. Compare like-for-like timing and record the selected configuration.

**Verify:** All three variants pass hardware and simulator correctness checks;
accepted simulator timing errors are <10%. Do not assume multicast must win.

**Exit:** Fill the GEMM section of [`calibration.md`](calibration.md) from the
new job and matched simulator runs. **Next:** B6e.

**Prereqs:** B6a. B6b/B6c preferred.

**Progress (2026-09-05):** The 256x256x64 G1 unicast-TMA and G3 no-TMA
functional smoke passes on `SM90_H200_CLUSTER132` and their outputs match.
G2 and all timing acceptance remain open pending timing-mode simulation; do
not close B6d from functional evidence alone or from the failed 4096^3 row.

---

### B6e — Freeze preset and close the report

- [ ] **B6e** Write fitted knobs into `SM90_H200_CLUSTER132` comments (`measured` / `inferred`). No pending rows on the accepted set in [`calibration.md`](calibration.md).

**Work:** Fit and accept only `SM90_H200_CLUSTER132` (fabric on). Other H200
presets are development-only and slated for removal; maintain compatibility
and geometry comments as needed, not separate calibration fits.
Do not leave `-gpgpu_dsm_bytes_per_cycle` as a non-zero bandwidth model.

**Verify:** Re-run the accepted L/BW/G set once on the frozen preset; errors still < 10%. Reduced functional filters still PASS.

**Exit:** Close-out under **B6** with date, config, commit, and a pointer to the filled [`calibration.md`](calibration.md). Then B-DEPR.

**Prereqs:** B6b, B6c, B6d, B6f, B6g, and B6h.

---

### B6f — Mixed 132-SM GPC map

- [x] **B6f** `-gpgpu_gpc_sms 16,16,16,16,16,16,18,18` + `SM90_H200_CLUSTER132` + occupancy `2048:32` / `cta 32`. Label packing `inferred`. Closed 2026-09-04.

**Close-out (2026-09-04):** Updated the 132-SM preset and related topology
examples/docs from 4×17+4×16 to the inferred 6×16+2×18 packing. The topology
unit test validates all eight GPC counts and powered-gated slots; the reduced
16×2 DSM/TMA integration regression remains passing.

**Verify:** unit `GpuTopology.HeteroH200SixBy16TwoBy18`; reduced 16×2 DSM/TMA still PASS. Do not replace CI with 132 SMs.

**Prereqs:** B1. **Next:** B6g.

---

### B6g — Standalone exclusive H200 job

- [ ] **B6g — JOB 2119329 PARTIALLY ACCEPTED.** The vendor-first result is
  available under `/home/jcliu/H200_results/job_2119329`. Low-level rows pass,
  but the 4096^3 GEMM timed launch deadlocked and the unicast winner label was
  wrong. Exclude that row and rerun it before closing B6g.

Before-result preparation is complete: the reusable simulator harness records
run fingerprints, resumes only exact matches, rejects skip/timeout markers,
and emits strict canonical CSV. The hardware/simulator comparator and explicit
case mapping are documented in [`calibration.md`](calibration.md); H200 and
vendor kernel source revisions are pinned. A fresh 132-SM representative run
is the only local result retained.

`SM90_H200_CLUSTER132` is the sole H200 configuration intended to survive the
development phase. `SM90_H200`, `SM90_H200_CLUSTER16x8`, and both reduced H200
presets are development-only and will be removed. Their 94-controller / 188-L2
slice geometry is kept consistent only to avoid misleading interim behavior;
do not spend calibration effort on them.

**Prereqs:** B6f preferred for sim; hardware job does not need sim topology.

---

### B6h — Fit non-fabric knobs from B6g

- [ ] **B6h** ALU, L1, L2, DRAM, STREAM, occupancy comments `measured`/`inferred`. Do not retune fabric from STREAM or G3.

The H200 NVL hard-spec baseline now uses `-gpgpu_n_mem 94`, derived from the
[NVIDIA product brief](https://dam-cdn.nvd.orangelogic.com/AssetLink/7n7vya4684sdccfyy6kv37ek5lw702h7.pdf)'s
6016-bit bus (`94 × 64 bits`) and 3201 MHz clock. With
`-dram_data_command_freq_ratio 2`, this models 4.814 TB/s versus the published
4.813 TB/s. Its 188 L2 subpartitions retain the existing per-slice geometry and
therefore model 58.75 MiB. Review the following provisional knobs when B6g
is reviewed; none may be accepted from the superseded Slurm jobs:

| Area to review | Knobs / current assumption | Required check |
|---|---|---|
| Memory geometry and L2 capacity | `-gpgpu_n_mem 94`, 188 L2 slices / 58.75 MiB | Confirm device bus width, reported L2, address distribution, and HBM STREAM peak. |
| Shared-memory latency | `-gpgpu_smem_latency 30` | Direct SMEM dependency test; do not subtract DSM pointer-chase overhead. |
| Clock domains | `-gpgpu_clock_domains 1785:1700:1700:3201` | Core/DRAM are H200 hard specs; ICNT/L2 are H100 fallbacks because job 2119329 did not measure them. |
| Kernel launch | `-gpgpu_kernel_launch_latency 8700` | Job 2119329 median converts to 8698.67 cycles. |
| DRAM latency/timing | `-dram_latency 254`, copied H100 `-gpgpu_dram_timing_opt` | H100 fallback: job 2119329 did not isolate a cold dependent HBM miss. |
| HBM bank/controller assumptions | H100 bank/group counts, timing tuple, burst length, FR-FCFS and queues retained | Unchanged DRAM-cycle timings at 3201 vs 2617 MHz imply approximately 18% shorter nanoseconds; validate rather than infer latency from bandwidth. The 254-cycle residual uses the core-cycle counter, not DRAM cycles. |
| L2 capacity discrepancy | Model 58.75 MiB; job's vendor driver reports 60.00 MiB | Separate effective bus-width decomposition from physical L2 geometry; do not claim 188 slices or 94 controllers are measured physical counts. |
| L2 latency/locality | `-gpgpu_l2_rop_latency 286`, `-gpgpu_l2_partition_extra_latency 150` | Base remains provisional; Far-L2 is the H100 fallback because per-offset distributions were not exported. |
| Local mbarrier | arrive `6`, successful try-wait `43` | Accepted job 2119329 medians; false-path 7755 is not the release knob. |
| `cp.async` | issue `4/4`; commit `7/7`; wait `5/5`; release `5` | Issue is job-derived; remaining fields are H100 fallbacks because they were not isolated. |
| DP arithmetic | latency `64,64,64,64,330`, initiation `64,64,64,64,130` | H100 same-GH100 fallback; job 2119329 executed no DP probe. |
| Integer division | INT DIV latency/initiation `21/2` | H100 fallback; the job's scalar suite did not execute integer division. |
| WGMMA | SS issue `4,4,4,4`, candidate completion tail `46,46,46,46`; RS issue `9,9,9,10`, completion `37,41,49,65` | SS matched width diagnostic found compute double-counting; validate corrected tail and all GEMMs. RS still needs compute-subtracted tail validation. |
| Remaining instruction fits | FP DIV, SFU, MMA, TMA and integer WGMMA tuples | Revalidate with all 26 exact mirrored microbenchmarks; do not extrapolate one shape/opcode across tuples. |
| DSM/TMA fabric | DSM latency/floor/visibility, shaper, VC/ACK and TMA completion curve | Refit latency and size slopes from vendor-first measurements. |
| Unlimited shared service | `-gpgpu_shmem_bytes_per_cycle 0` | Keep as compatibility default only until a kernel constrains aggregate SMEM service bandwidth. |

- [ ] Validate WGMMA width extrapolation before final GEMM acceptance.
  `cuda-sim.cc::wgmma_latency_for_shape` multiplies the N=64 entry by
  ceil(N/64) for issue, initiation and completion tail alike. The loaded
  no-TMA winning GEMM uses `m64n128k16` SS instructions. Job 2119329's
  `WgmmaAsyncLatencyBench.F16SsShapeSweep.csv` reports issue 4.29688/4.27344
  and wait 75/107 cycles at N=64/128. Those measurements do not support
  blindly doubling every timing component. First reproduce the matched
  issue/wait sweep and distinguish measured wait from modeled compute plus
  tail; do not substitute measured wait directly for an internal tail.
  The matched one-SM diagnostic now confirms the error and its correction:
  waits change from 104/211 to 75/107 cycles, matching hardware N=64/128.
  `test/check_wgmma_width.py` fails before and passes after. Total spans are
  within 2.3%, but issue spans remain too high. Wider RS/integer regression
  and all-shape GEMM acceptance remain required before closing this item.
| Non-hardware simulator choices | inferred `6×16+2×18` GPC map; PTX allocator disabled | Confirm topology if exposed; retain allocator workaround only while its calibration probe requires it. |

**Prereqs:** B6g CSV.

---

## B7 — Per-bank shared memory (optional, after first delivery)

- [ ] **B7** `shared_access_plan_t`: bank + broadcast constraints on top of aggregate bytes.

**Prereqs:** B5. **Do not** start before B6 unless a kernel is bank-bound and mispredicted.

---

## B8 — Remote mbarrier on the fabric; TMA multicast decoupled

- [x] **B8** Remote mbarrier uses the DSM fabric; TMA multicast is functional
  fan-out with an optional fixed completion latency and no network model.
  Revised and closed 2026-09-04.

**Read first:** [`dsm_fabric.md`](dsm_fabric.md) §8, [`programming_model.md`](programming_model.md) §§3–4.

**Original work plan (superseded):** Route TMA multicast and remote mbarrier
through fabric packets, with configurable source or fabric replication. The
multicast portion was rejected because there is no evidence that TMA shares
the DSM fabric; only remote mbarrier uses the fabric now.

**Verify:** topology/fabric-isolation unit tests plus
`TMAClusterMulticastTest.*`, `TmaMulticastMaskTest.*`, and a combined reduced
H200 DSM/TMA regression.

**Prereqs:** B5.

**Revised (2026-09-04):** TMA `.shared::cluster` multicast no longer injects `tma_data` into either `cluster_noc_t` or `dsm_fabric_t`; there is no multicast bandwidth, route, queue, SRAM-service, or contention model. Functional fan-out remains, and `-gpgpu_tma_multicast_latency` adds a fixed delay before `complete_tx` (default 0). Remote mbarrier `arrive` / `expect_tx` / `complete_tx` / `try_wait` still use `mbarrier_request` / `mbarrier_completion` on the DSM fabric.

Verification (2026-09-04, 0 failures):

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 \
  run test --target sm120 --group integration \
  "TMAClusterMulticastTest.*:TmaMulticastMaskTest.*"
FLASHGPU_ALLOW_CC_MISMATCH=1 ./test/run_tests.sh \
  -c SM90_H200_REDUCED_CLUSTER16x2 run test --target sm120 \
  --group integration \
  "DsmTest.RemoteLoadFromPeer_TwoCtas:TMAClusterOneProducerTest.OneProducerPeerConsumers"
```

PASS evidence:

```text
[  PASSED  ] 4 focused topology/fabric unit tests.
[  PASSED  ] 6 TMA multicast/mask integration tests.
[  PASSED  ] 2 reduced-H200 DSM/TMA integration tests.
```

---

## B-DEPR — Remove the delay line

- [ ] **B-DEPR** Once `dsm_fabric_t` is the only SM↔SM path, **deprecate and delete delay-line knobs** and update **all** config files.

**Current state (2026-09-04):** TMA multicast message types, injection, and
delivery have been deleted from `cluster_noc_t`; multicast now has no NoC
callsite. The module itself remains because fabric-disabled remote mbarrier is
still routed through `inject_mbar_remote`. DSM store/load delay-line entry
points are otherwise uncalled. B-DEPR remains open until that fallback and the
module, helpers, knobs, matrices, build entries, and tests are removed together.

**After B-DEPR, delete these (do not leave unused files or knobs in tree):**

- all `dsm_latency_matrix_*.csv`
- `-gpgpu_dsm_latency_matrix_file`
- `-gpgpu_dsm_remote_latency` as a hop / bandwidth knob

The hop-matrix CSV and the two knobs above are delay-line-only. They are **not** the fabric timing model. After this ID, grep for them must be empty.

**Read first:** [`knobs.md`](knobs.md) §§2–3.

**Why:** Two timing models in tree will drift. Agents must not keep `ready_cycle = hop + BPC`.

**Work:**

1. Delete the three items in the After-B-DEPR list above. No leftover CSV, no unused knob registration, no config that still points at a matrix file.
2. Delete the remaining `cluster_noc_t` remote-mbarrier fallback, then remove the module and its helper/test build entries. Also delete unused `-gpgpu_dsm_bytes_per_cycle` and `-gpgpu_dsm_store_immediate` if fabric stores are deliver-only. The old TMA multicast message types and network knobs are already gone; keep only the fixed `-gpgpu_tma_multicast_latency` knob.
3. Keep: hang watchdog, mbarrier cluster enable, TMA data-before-mbar, topology knobs, fabric knobs, possibly `-gpgpu_dsm_local_latency` / `base_latency` if still used as SMEM/floor. Do **not** keep a pairwise hop table or a scalar remote hop as the DSM timing model.
4. Grep configs: `configs/SM90_H200*`, `configs/SM120_*CLUSTER*`, test overlays, `FLASH.md`, comments in `gpu-sim.cc` / `shader.h` / `cluster_noc.*`.
5. Update every `gpgpusim.config` that set the deleted knobs.
6. Rewrite or drop unit tests that only parsed the hop CSV (`ClusterNocMatrix.H200ReducedMatrixFile` and similar). Do not keep the CSV as an unused file.
7. Delete `cluster_noc.{h,cc}`, `cluster_noc_helpers.cc`, their build entries,
   and delay-line-only tests after the remote-mbarrier fallback is gone. Do not
   replace them with wrappers around `dsm_fabric_t`.

**Do not:** Leave “0 = unlimited BPC” in any shipped config. Do not keep a second inject path that writes smem immediately when `can_inject` is false.

**Verify:** Full cluster integration filters. Grep is empty for `dsm_latency_matrix`, `dsm_latency_matrix_file`, `gpgpu_dsm_remote_latency`, and `dsm_bytes_per_cycle`. `cluster_noc_enable` has a defined migration (renamed to `dsm_enable` or documented alias).

**Exit:** Grep output empty (or aliases listed). Config list in the close-out.

**Prereqs:** B6 and B8 green on fabric; delay line unused.

---

## B9 — Not first delivery

- [ ] **B9a** Multi-hop / escape VC / wormhole proof
- [ ] **B9b** Independent DSM clock domain
- [ ] **B9c** Blackwell preset from **Blackwell** measurements
- [ ] **B9d** Power model
- [ ] **B9e** SST/gem5 adapter complete

---

# Track C — Documentation

- [x] **C1** Rewrite living spec under `docs/cluster_noc/` (this directory).
- [x] **C2** Stub old `docs/cluster*.md`. Memory-xbar argument lives in `dsm_fabric.md` §9 (no `use_xbar_roter.md`).
- [x] **C3** Point `FLASH.md`, `CLAUDE.md`, config READMEs, C++ comments at this directory.
- [x] **C4** Midterm is historical; not a second spec.

Agents doing **code** should not edit C1–C4 except to fix a broken link after a rename.

---

# Retired IDs (do not revive)

`L0`–`L4`, `L2-1`…`L2-6`, `L3-1`…`L3-5`, `L4-1`… as living IDs. Content now lives in **B5** (scoreboard), **B3b** (BW), **B6** (calibration).

Do not model TPC/TPCARB. Do not put DSM on `xbar_router` request/reply subnets. Do not “fit BPC from a multi-cluster GB/s point.”

---

# Review bans

Reject patches that:

- Use `tpc` to mean GPC, SM, or icnt node
- Map GPC with division/modulo **outside** `gpu_topology_t` (shaper `sm_id % period` is allowed and documented)
- Give DSM two physical endpoints (requester vs service)
- Treat request/reply **subnets** as VCs or give them separate physical BW
- Name GX planes as request/response VCs
- Infer response `network_src` from the transaction requester
- Teleport a whole packet after numeric budget
- Treat outstanding-tx limits as link/VC credit
- Reply from target SRAM **without** `shared_memory_service_t` (after B5)
- Fire-and-forget store and allow CTA exit
- Implicit local/TMA/DSM priority via SM walk order
- Nested OpenMP on one GPC fabric
- Label H200 inferred numbers as Blackwell hardware fact
- Mix large rename and timing behavior in one commit
- Charge header bits as extra **payload** occupancy (payload grant is 32 B)
- Fill RF in `ld_impl` on the fabric path
- Map `can_inject == false` to immediate peer write

---

# First delivery (stop and calibrate)

Declare first delivery when **all** are true:

1. Names: `gpc_t`, SM, TB-cluster, global node, DSM endpoint, VC are unambiguous.
2. Each enabled SM has its own global NoC endpoint.
3. No DSM traffic ⇒ L2 steady-state BW does not drop when GPC grouping changes.
4. One host thread advances each GPC fabric.
5. Request/response: independent queues/credits, shared physical 32 B-payload lanes.
6. Same-dir mix shares a ceiling; opposite dirs run together.
7. Shaper is configurable (`skip_mod` default, `sm_id % period`); idle slots wasted; stats exist.
8. Remote load uses the **local SMEM** scoreboard/LDST completion path.
9. Ordinary DSM ld/st/atom: mixed target, response/ACK, pending join.
10. DSM vs local LSU/TMA compete in `shared_memory_service_t`.
11. Finite buffers + ACK coalesce: no constructed deadlock; pending DSM blocks warp/CTA/sim end.
12. OMP on/off functionally equal; fixed config completion cycles repeat.
13. H200 preset explains mbarrier / TMA / DSM latency and `dsm_bw` slopes; sim cycle counts on the calibration suite (including Triton multicast GEMM) are within 10% of H200 ([`calibration.md`](calibration.md), B6).
14. TMA multicast uses the common TB-cluster resolver but deliberately does not use the DSM fabric/SRAM contention model (B8 revision).
15. Delay-line knobs gone or hard-deprecated (**B-DEPR**).

Then, and only then, B7/B9.
