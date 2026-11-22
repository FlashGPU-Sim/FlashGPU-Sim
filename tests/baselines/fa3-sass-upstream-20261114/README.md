# FA3 medium regression reference after the public-upstream rebase

This is a **simulator-version regression reference**, not a new hardware
calibration. The [2026-09-14 hardware/frozen record](../fa3-sass-ccd2-20260914/README.md)
is unchanged. Large cases have not been requalified in this update.

The production implementation is `bf79e42033e4c12f8de4eedf91de98be7b006869`.
The exact CI Docker recipe (CUDA 13.3, GCC 11.4), `SM90_H100_SASS_FRONTEND`,
four threads, and separate cold processes produced the values below. All four
forward cases pass both O and LSE reference checks. Independent GCC 13 host runs
agree exactly for full attention and within 0.14% for causal attention.

| Case | Old frozen cycles | Current CI cycles | Change | Current time vs historical NCU |
|---|---:|---:|---:|---:|
| D64 full | 178429 | 177209 | -0.684% | -1.049% |
| D64 causal | 122817 | 122647 | -0.138% | -7.254% |
| D128 full | 116150 | 118460 | +1.989% | +10.126% |
| D128 causal | 97882 | 97073 | -0.827% | -0.914% |

The last column uses kernel cycles / 1500 MHz and the original NCU durations.
It is not the regression gate and is not evidence of new hardware measurements.
In particular, the D128 full hardware discrepancy is slightly larger now.

## Why a separate reference is needed

The old reference predates public timing-model changes. First-divergence
traces and isolated legacy-behavior controls identify four relevant changes:

- Public `dc355451e`: operand collector allocation can select an eligible
  scheduler partition when the oldest partition is occupied. Covered by
  `OperandCollectorAllocationTest.BusyPartitionDoesNotBlockFreePartition`.
- SASS `82f179f6`: the final CTA barrier arriver must retain its timing handoff.
  Covered by `SassFunctionalTest.LastDeferredArriverRetainsTimingHandoff`.
- Public crossbar updates (`cd70d7278`, `eb7c9ef36`): output rotation and
  round-based multi-grant arbitration change request/response order at the same
  configured width. Fairness and grant budgets have dedicated interconnect tests.
- Public `1970fb86b`: shared-memory expose/grant service adds a service tick even
  with unlimited byte budget. This is intentional upstream behavior, explicitly
  covered by `SmemService.UnlimitedStillDelaysOneCycle`.

These changes are retained in production. Diagnostic libraries alone restore
`allocate_cu`, `execute_cta_barrier_deferred`, crossbar `Advance`/`RR_Advance`/
`iSLIP_Advance`, and `ldst_unit::shared_cycle` from historical `b343810a`.

The original build with the current host executable reproduces D128 full at
exactly 116150 cycles. With the first three old behaviors restored, all 6336 TMA
transaction events across 132 SMs match through cycle 14000; the next divergence
is the shared-memory stage at epilogue STSM around cycle 24310. Restoring all four
behaviors brings the complete controls inside the original 0.5% gate (see
`provenance.json`). Small residual changes remain; this is not a claim of exact
cycle equality or complete attribution of every individual cycle.

## Gate and provenance

`check_sass_medium.py` uses this version's four medium expectations while keeping
**the same inclusive +/-0.5% threshold**, one-known-case/one-kernel requirement,
actual execution-driven SASS timing marker, and successful O/LSE checks. Nine
positive/negative checker tests include rejection one cycle outside either
boundary; they run in CI planning.

Binary, library, configuration and log hashes are recorded in `provenance.json`.
Raw generated binaries and logs remain in the experiment evidence store rather
than the public source tree. Never update these numbers simply because a later
run fails: reproduce the failure and explain the model or code change first.
