# Native resource contention preparation (2026-11-26)

The prepared suite is `tests/native/resource_contention/`. It reuses the
archived B200 shared/TMEM probes, records their source provenance, and builds
eight standalone executables locally with both sm100a cubins and PTX. The server
needs no compiler. The package includes the real CUDA runtime, SASS/PTX dumps,
compiler commands, CUTLASS revision/patch, resource reports and file hashes.

## Measurement matrix

* MMA versus LDS, STS, explicit TMEM load/store, and an integer scheduling
  control. Three BF16 shapes, shared lower/upper halves, three disjoint TMEM
  column windows, and matched MMA-only/background-only/concurrent modes.
* 32dp TMEM service/drain/roundtrip controls; 16dp load/store shapes
  16x64b.x4, 16x128b.x2, and 16x256b.x1/x2; one/four warps and depths one/four.
* Shared-to-TMEM copy: 128x256b, 128x128b, both 64x128b broadcast selectors,
  and 32x128b broadcast; per-instruction versus per-group commit/wait.
* LDS/STS, STS/STS, LDS/TMA, STS/TMA and TMA/TMA shared write competition.
  Strict-half 32 KiB placements and separately labelled 64 KiB x2 controls
  that cross the half boundary. Sources are hot in L2.

There are **792 timing configurations**, **270 distinct numerical smoke
configurations**, and **297 selected full NCU captures**. Native timing uses
three unprofiled repetitions. NCU keeps the existing base-clock policy; native
clock settings are unchanged. Each configuration has two loop lengths where
applicable. Reports retain raw cycles and slopes, per-worker slowdown, logical
payload bytes, and overlap in both directions. Insufficient standalone pressure
or overlap blocks a no-contention inference. These gates are necessary evidence,
not proof of a physical-port topology or peak.

Workers join after their end clocks and before result writes or verification
loads. Shared reservations are checked completely in parallel, so a serial
one-thread verifier cannot dominate the profiler trace. Longer NCU loops reduce
setup/teardown contamination; NCU still observes the complete kernel. The 32dp
driver's warmup launch is excluded from the NCU capture by a matching-kernel
launch filter.

## Functional changes found during preparation

The initial PTX transfer sweep passed all 63 configurations. SASS rejected 12
configurations because `STTM.16dp64bit.x4` and `STTM.16dp128bit.x2` were missing
from native decoding/registration. They now use the existing shared TMEM
fragment mapping, including the corresponding eight-register 128-bit form.
No separate SASS storage backend or timing constant was added.

The regression first failed with the missing native opcode, then passed after
the adapter fix. Tests check every affected physical word, untouched rows and
columns, all four issuing-warp partitions, load-back results, register
dependencies and the common timing class. Incremental host builds now track
the TMEM decode header for the timing projection support object.

The broader host run exposed two stale expectations from earlier feature
integration: BFLY had retained an extra lower-bound check in its test, and
CREDUX still expected an active NaN to be rejected. Their expectations now
match the already-shared shuffle/reduction semantics. No execution behavior
was changed for these two corrections. **All 661 host tests pass.**

## Evidence and boundaries

Workspace receipts live at `runs/tcgen-closeout-20261126/`; consult
`runs/CURRENT.json` for final completion status and package/archive paths.
The prepared package passed **270 SASS and 270 PTX numerical smoke runs**.
Its compressed archive is
`/home/wzr/flashgpu-native-packages/resource-contention-20261126-final.tar.zst`
(3,563,469 bytes; SHA256
`f9d1ade5bcea5220d512775ba49eb76ebad88da130a2c76defefbb80149b1df1`).
All 92 manifest entries were checked against the archived contents. The
archive contains prepared workloads and local validation, not new GPU results.
`qualify.py` joins local replay evidence by exact binary hash and arguments,
verifies configuration/log receipts, and requires both frontends for every
packaged numerical smoke configuration. One-CTA numerical smoke may use one
simulated SM; it is not native timing calibration.

The four remaining DeepInfer MoE acceptance entries are reviewed separately
against their immutable captured inputs and frozen replay runtime. Dispatch
permutations must preserve all source-token multisets, final outputs, remote
combine contributions, and active intermediates after alignment by
`(source rank, token, topk)`. An accepted permutation does not mean bit-exact
agreement with the original GPU run or erase previously accepted residuals.
The interrupted PTX L2 replay completed at 969,950 cycles. All four outstanding
MoE entries passed the complete permutation checks. The **117-entry capture
acceptance is closed**: 67 strict GPU passes, 46 unchanged accepted numerical
residuals, and four byte-proven dispatch permutations. This covers 59 captured
SASS entries and 58 available PTX entries under the recorded frozen runtimes;
it is not a new same-head timing-accuracy sweep.

No new hardware contention results are claimed here. Paired-CTA port behavior,
copy/MMA contention, and a complete physical TMEM/shared write model remain
outside this preparation. The unused WS shift-window timing corner remains
deferred; no parameter was introduced to fit it. Future collection must retain
raw clocks, full `.ncu-rep`/CSV, executable/source/configuration provenance,
GPU state and hashes in `data/hardware/b200/resource-contention-20261126/`.
There is no automatic shutdown command.

## B200 collection: broadcast descriptor correction

The 2026-11-26 native smoke gate found a functional error before collecting
transfer timings: the broadcast-copy fixture omitted the shared descriptor's
eight-row stride, while the common backend treated zero as implicit contiguous
storage. B200 instead repeats the first eight rows. Dumps of all 128 destination
rows confirm this for both `64x128b` selectors and `32x128b.warpx4`.
This follows the K-major eight-row stride described in the
[PTX matrix descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-descriptor).

The backend now honors the qualified zero and 128B strides. The performance
fixture uses explicit 128B stride; zero-stride semantic tests retain zero and
check the repeated rows. The paired block-scaled MMA oracle also accounts for
the resulting repeated scale rows. No timing constants change.

Validation: the host regression fails before the correction and passes after
it; all 661 host tests pass; 13 copy/paired-MMA tests pass on B200 and in each
frontend; the corrected transfer package passes all 63 smoke configurations in
each frontend (126 runs). The ad-hoc first cluster replay omitted DSM and is
retained as a configuration failure; the qualifying rerun uses the existing
2-GPC x 4-SM cluster-validation setup with DSM enabled.

The original failed hardware smoke and its package are preserved. Independent
MMA, solo-TMEM and shared-competition families retain their original binaries.
Only the transfer family uses the versioned
`resource-contention-20261126-cp-stride-v2` package. Native timing and NCU
collection status are tracked separately in the workspace hardware manifest;
these functional passes do not assert that resource-contention calibration is
complete.

## Collected B200 evidence and transfer timing

The complete collection passed 270 native numerical smokes, 792 timing cases
with three repetitions, and 297 full NCU captures. The raw evidence archive is
`resource-contention-b200-20261126-145240.tar.zst`, SHA256
`e7627d6225f7c66018f19d1f608eee8e0b609bb149d4169d9c5b4550c2798e93`.
It contains both versioned packages, the original failed broadcast smoke,
corrected native results, NCU reports/CSV and immutable simulator comparisons.

`-gpgpu_tmem_b200_transfer_timing` is opt-in (default off, enabled in B200).
It qualifies ordinary 32dp stores and the 16dp 64/128/256-bit transfer family.
Both adapters normalize geometry into the common backend; stores and reads
use partition-local admission and wait for actual execution retirement.
Independent read/write admission does not assert independent physical SRAM
ports under arbitrary MMA traffic. Packed and half-warp-split forms retain
legacy timing and are not qualified by these measurements.

The effective FU timing includes a width-dependent service cost after operand
collection. For ordinary stores the service is `max(1,W/2)` below W=16 and
`W/2-4` above, with completion at least six cycles. The negative intercept is
an effective pipeline residual, not negative physical memory latency.
For the qualified 16dp family, the service extent is W for 64/128-bit forms and
2W for 256-bit forms; load completion adds six cycles, store completion adds
ten. The narrow store initiation equals completion: the per-instruction and
per-group drain measurements do not support the formerly unrestricted overlap.
These are first-order effective service models, not inferred physical bank
wiring. Native measured widths are ordinary x4/x32/x64 and narrow forms
16x64b.x4, 16x128b.x2, 16x256b.x1/x2; other legal widths are extrapolations.

On 144 exact SASS timing configurations, every numerical check passes and all
72 two-length loop slopes are within 14.59% of native. Previously the worst
error exceeded 65%. Ordinary 32dp loads remain unchanged; the three ordinary
store per-instruction-drain slopes now match exactly. Burst stores and dependent
roundtrips retain up to 14.59% residual error. Slopes include loop/drain costs;
they are not single-instruction latencies. Workspace receipts are under
`runs/CURRENT.json` -> `tcgen_resource_model_fix_20261126`.
This closes neither CP service timing nor shared write contention, which are
tracked separately. It is not a full DSV4 timing requalification.


## B200 single-CTA CP effective service

The B200 preset now uses the existing common CP pipeline controls: completion
192 cycles and initiation 64 cycles for 128x256b, 128x128b, 64x128b and
32x128b. The unmeasured 4x256b entry retains legacy timing. No global mbarrier
latency or PTX/SASS-specific completion bypass is added.

This is a conservative effective asynchronous service model: finite acceptance
and completion, with overlapping copies. The measurements include election,
commit, polling and notification; they do not uniquely separate those costs
from the copy engine. In particular, 192 must not be presented as a measured
physical SRAM latency, nor 64 as a measured byte-bandwidth limit. The five
measured forms have nearly identical loops despite different source sizes.
Paired-CTA CP and CP/MMA port contention remain unqualified.

All 30 exact timing configurations and 30 dual-frontend numerical smokes pass.
Across the 15 two-length slopes, the largest absolute error is 13.17%.
For 128x256b, single-copy/group-of-four/wait-each loops are respectively
335/585/1010 cycles on hardware and 291/523/1143 in simulation.
The measured tuples, before/after cycles and exact case identifiers are in
`data/calibration/b200-tcgen05-copy-20261126.csv`. The same immutable native
archive and corrected stride package identified above supply the reference.


## Experimental shared write landing model (not a calibrated default)

`-gpgpu_shared_region_write_model 1` connects ordinary STS and aligned local
TMA landing to the common per-128KiB write budgets. LSU loads and stores share
one issue-side byte budget; TMA has its own path. Read and write SRAM budgets
are distinct. No bandwidth constant is fitted: the B200 experiment retains
128 B/cycle per path and per region/direction. Future TMA landing reservations
receive service first; STS consumes the remainder. Completion notifications
wait until all reserved write service has finished. Ideal global memory still
pays shared landing service.

This option is default **off**, including the B200 preset. DSM, multicast,
mapped peer copies, completion-mask/fixed architectural-latency paths and
unaligned destinations retain their existing timing. Region placement is based
on CTA-local addresses, not physical placement across CTAs. Out-of-order memory
responses carry their logical tile offsets; packed expansion charges the shared
footprint, and OOB fill also charges writes. There is no new functional copy.

The 168 exact SASS shared-pair runs and 168 PTX/SASS numerical smokes pass
numerical checks. They do **not** pass
timing qualification. LDS/TMA and split STS/TMA improve, but same-half STS/TMA
still has opposing per-worker errors: STS is about 39% fast and TMA 55-56% slow
in the 32KiB control. A wall-time error near 5% hides this failure. The 64KiB
cross-boundary control has a TMA error up to 75.5%. STS/STS retains a 15.3%
fast outlier; TMA/TMA one-slot tests remain unsaturated.

The next diagnostic separates TMA control instruction delivery from byte
landing. The actual loop reloads its descriptor pointer through LDCU; currently
constant and shared operations enter the same LSU dispatch path. Per-warp
issue/operand/retirement traces must establish where that delay occurs before
changing frontend queues or enabling this model in a preset. A larger shared
bandwidth or a workload-specific delay would not establish the missing cause.

The first issue traces (`tma-control-trace-analysis.json` in the workspace
receipt) narrow this further: the same-half TMA warp adds about 76k ready-but-
blocked samples, mainly on integer/control operations. Once admitted, its two
loop LDCU sites average only 4.69 and 3.78 cycles versus 2 cycles alone.
Thus constant-load execution alone does not account for the gap. A dedicated
integer-collector control is needed to distinguish common collector blockage
from the shared-memory write service itself.

The follow-up controls did not close this gap: reserving only INT collectors
leaves same-half TMA 61.2% slow, and a temporary oldest-input arbitration patch
leaves it 53.5% slow. The latter patch is retained only in experiment evidence
and removed from the production source. Neither diagnostic configuration is
enabled. Collector-set scheduler initialization did expose an independent
correctness bug; see `../bugs/operand-collector-set-partition.md`.

After the accepted TMEM/CP changes and collector initialization fix, the current
B200 preset passes the complete 270-case resource numerical smoke in each
frontend (540/540). The additional 108 exact MMA/TMEM timing runs put background
TMEM service within 3.58% of native, including concurrent MMA. MMA itself retains
its previous roughly 6-17% fast residual; this change does not qualify a new MMA
recurrence or shared-write model. Per-phase runtime/config hashes and rejected
controls remain in `qualification-summary.json` under the workspace receipt.

## STS/TMA causal diagnosis: transaction service versus replenishment

The follow-up uses the same immutable 32KiB, two-slot cubin and exact 8MiB
arguments, runtime-v4, and the experimental shared-write option. Four traced
runs (TMA alone/mixed, same/opposite half) all pass numerically. The 16 warmup
transactions are excluded from the following averages; 256 timed transactions
remain. These are **simulator stage times**, not hardware transaction latency
measurements. Events and receipts are under `tma-transaction-diagnostic` in the
resource-model evidence root.

| Simulated stage, core cycles | TMA alone | STS + TMA, same half | STS + TMA, opposite halves |
| --- | ---: | ---: | ---: |
| Transaction creation to final memory response | 524.41 | 525.22 | 524.26 |
| Final response to mbarrier notification | 31.00 | 31.00 | 31.00 |
| Slot notification to next creation using that slot | 54.00 | 534.25 | 330.39 |
| Average consecutive transaction creation interval | 304.30 | 543.90 | 441.89 |
| No live transaction, fraction of timed transaction window | 0.00% | 28.07% | 9.68% |

Transaction service is virtually unchanged. Replenishment is delayed, leaving
write bandwidth available to STS. This explains why the simulated mixed STS
worker is too fast while the TMA worker is too slow. It is not evidence for a
slower L2 response path or a need to increase shared write bandwidth. Whole
kernel wall time masks the opposing worker errors. These occupancy fractions
cover the full timed transaction window, including any tail after STS finishes;
they are not overlap-only occupancy or physical GPU utilization.

A trace-only build adds explicit `INT_PIPE` classification and captures eleven
blocked pipeline snapshots between cycles 20000 and 20200. It produces exactly
the same stream/wall cycles as runtime-v4. Nine snapshots have all 16 generic collectors occupied by STS; the other
two have 15 occupied. The four `OC_EX_MEM` slots are also occupied; the LSU is
servicing STS, including a `DATA_PORT_STALL` snapshot. Register-bank request
queues and all four INT execution input slots are empty. SMSP 0's integer
input slot is occupied by an earlier STS-worker control instruction. The TMA
warp cannot enter that input: the execution unit is idle, but memory pressure
has filled the shared operand-collection resources upstream. The measured TMA
loop has 76204 `INT_PIPE` ready-but-blocked samples.

The dedicated-INT diagnostic does remove this particular input blockage, but
TMA still uses generic collectors. It leaves 63977 dependency-stall samples at
`R2UR` PC 0x27d0 waiting for read barrier bit 0x4. This is the **SASS operand read
barrier of the previous UTMALDG**, not the shared-memory completion mbarrier.
For example, the consumer waits from cycle 13151, UTMALDG finally finishes
operand collection at 13250, and the consumer proceeds at 13252. Constant loads
also retain the shared memory-input path. Thus the earlier negative dedicated-
INT result does not rule out collector pressure: protecting only INT moves the
bottleneck to TMA operand delivery.

The concrete remaining modeling gap is memory backpressure spilling into the
control/TMA operand-delivery path. Hardware mixed-worker measurements do not
show this degree of TMA slowdown. They do not uniquely identify the hardware's
collector partitioning or queue depths. A fix should test resource isolation or
admission/backpressure behavior across the entire shared matrix, not add a
case-specific TMA latency or simply increase byte bandwidth. Neither dedicated
INT nor the rejected arbitration patch is enabled in a preset. Snapshot hooks
exist only in the frozen diagnostic runtime/patch; production retains only the
more precise `INT_PIPE` trace label, with no timing change.

The independent collector-set initialization fix `f9658cc0` remains queued for
extraction onto public main with its reproducer. It is not the above performance
modeling change and has no SASS-frontend dependency.
