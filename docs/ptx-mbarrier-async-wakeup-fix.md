# Wake suspended PTX waits on asynchronous barrier completion

Remote TMA completed its transfers but could leave the owner warp asleep in
`mbarrier.try_wait`. Completion callbacks released a warp barrier directly;
PTX also needs its pending wait rechecked and its predicate written. Route
transaction completions and remote arrivals through the existing phase-change
notification helper. That helper retains the fallback release for native SASS
waiters. No polling hint, timing configuration or TMA transaction limit changes.

This branch starts directly from public `flash` at `b3378cb9`. The test-only
commits `6885d9c1` and `ff6830bb` extend the existing unicast/address fixtures,
reusing their data and sentinel checks. Long-suspended local transfers pass on
the baseline and serve as controls. Remote bulk and tensor transfers are tested
with the same complete launch executable before and after the production fix.
The remote tests use two clusters of two CTAs and eight barrier generations.

The tests are included in the existing integration manifests and selected by
the SM90 cluster CI job's `TMACluster*` filter. A single-SM-per-cluster preset
skips remote launches explicitly; it cannot qualify this fix. Local SM100
qualification uses an isolated two-GPC/four-SM-per-GPC configuration with DSM.
Evidence is in workspace `runs/public-mbarrier-wakeup-20261125/`. Hosted CI and
new hardware calibration are not claimed.

Both remote fixtures deadlock before the fix after all four initial transfers
have completed. With the same executable and configuration, the fix completes
all eight rounds with exact payloads/sentinels in 10491 and 10453 cycles. This
is a functional wakeup correction, not a shape-dependent timing calibration.

Both local CTA controls also retain numerical correctness after the fix.
