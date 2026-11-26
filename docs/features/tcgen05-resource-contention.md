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
