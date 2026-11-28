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

## Acquire/release fence extraction

The PTX adapter accepts explicit `.acq_rel` for ordinary fences at CTA, cluster,
GPU and system scope. They enter the existing memory-barrier path, with no
second completion engine. `.acq_rel` on other instruction families or proxy
fences fails closed; those forms have not been qualified by this slice.

`CI_JOB=sm100-acq-rel-fence` checks predecode classification of twelve ordinary
forms (default, `.sc`, `.acq_rel` at four scopes), then eleven dual-frontend
launches. CTA publication uses two warps; wider scopes publish across CTA pairs,
with explicit two-CTA cluster launches for cluster scope. Both `.acq_rel` and `.sc` publish at all four scopes. Another launch checks
predication, and two more check volatile 32/64-bit transfers with guard words.
The two shared-store visibility regressions also run in this job.

The publication fixture exposed missing native `LDG/STG.E[.64].STRONG.GPU/SYS`
forms, along with `MEMBAR.ALL.SYS` and `MEMBAR.SC.GPU/SYS`. These now use the existing load/store handlers and timing projection;
wide variants retain both register words. A component test checks all eight
load/store, 32/64-bit and GPU/SYS combinations, including predication, memory
extents and projected load cache policy. This does not introduce host coherence.

This qualifies device-side publication and the common completion path, not a
complete weak-memory litmus suite or CPU/GPU system-coherence implementation.
The backend conservatively drains preceding traffic and does not distinguish
scope-specific physical fence latency. Ordinary fences do not replace TMA or
TCGen05 async completion waits or proxy fences.

Reference: [PTX memory fences](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).
