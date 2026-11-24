# Ordinary memory fences and shared-store completion

PTX ordinary `fence` and SASS `MEMBAR` use the same `MEMORY_BARRIER_OP`
backend. The backend drains register producers, global store acknowledgements
and DSM traffic. It must also wait for ordinary shared stores to leave the
shared service path and reach the configured visibility timestamp. That last
condition was previously used by named barriers but omitted from memory fences.
The shared condition is now reused by both paths; no independent PTX/SASS
completion mechanism or new timing constant is introduced.

`CI_JOB=sm100-fence-visibility` runs two numerical/timing launches per frontend:
`st.shared -> fence.cta -> clock64` and the equivalent `fence.sc.cta` sequence.
The test reads its visibility bound from the actual simulator config; CI uses
512 cycles to expose premature release. This is an artificial regression delay,
not a hardware calibration. The unmodified production configuration remains
unchanged. Tests retain the stored-value check as well as the clock bound.

With the same binary/config, the preceding runtime released the PTX sequence
in 9 cycles and native sequences in 13/14 cycles despite the 512-cycle setting.
Evidence lives under `runs/memory-fences-20261125` in the experiment workspace.

This common-backend correction is a separate commit from the PTX `.acq_rel`
parser extension. Its standalone public-main applicability still requires a
separate dependency audit: it relies on the extracted shared-store visibility
state. It is not a newly calibrated physical fence latency model.
