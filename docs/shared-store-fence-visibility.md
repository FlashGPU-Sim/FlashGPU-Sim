# Shared-store visibility at ordinary memory fences

The public backend already tracks queued shared stores and their visibility
cycle for named barriers. Ordinary memory fences checked register producers,
global-store acknowledgements and DSM traffic, but omitted that shared-store
state. A fence could therefore release while a preceding shared store was
still in service or before its configured visibility cycle.

Both paths now use the same readiness predicate. The existing PTX/SASS fence
backend is retained; no new instruction modifier, timing parameter or production
configuration change is introduced.

Two numerical/timing launches cover `st.shared -> fence.cta` and `fence.sc.cta`,
checking stored values and the clock span. An artificial 512-cycle visibility
delay exposes the omission: both public-base cases returned in 9 cycles. The
same frozen executable/configuration passes both bounds after the fix.
Three ordinary PTX vector/pack controls pass before and after with identical
cycles (5317, 5323, 5331). This delay is a regression-test setting, not a
hardware calibration.

The `sm100-shared-fence-visibility` CI job includes a blocking post-check that
creates its own configuration with delay 512 and requires both tests to run
without skips. Its post-check passed locally with CUDA 13.3 / SM100; the full
hosted CI job has not been run. The public qualification uses PTX; dual-frontend
feature acceptance remains separate. No TCGen05 dependency is needed.

Workspace evidence: `runs/public-shared-fence-20261125/qualification.json`.
